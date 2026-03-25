/*
 * full_forward.m — Complete transformer forward pass in C/Metal.
 *
 * Replaces the entire MLX decode path for Qwen3.5-35B-A3B:
 *   embed -> 40 layers (attention + MoE) -> final norm -> LM head
 *
 * Two attention types:
 *   - GatedDeltaNet linear attention (30 layers: layer_idx % 4 != 3)
 *   - Full GQA self-attention with KV cache (10 layers: 3,7,11,...,39)
 *
 * All layers have MoE (256 experts, 8 active, shared expert).
 * Expert weights loaded on-demand via pread from packed binary files.
 * Non-expert weights set from Python via bakan_full_set_weight().
 *
 * Model dimensions (from config.json):
 *   hidden_size=2048, vocab_size=248320, num_layers=40
 *   Full attention: 16 Q heads, 2 KV heads, head_dim=256
 *     q_proj: 2048 -> 8192 (16*256*2, includes gate), k/v_proj: 2048 -> 512
 *     partial_rotary_factor=0.25, rope_dim=64, rope_theta=10000000
 *   Linear attention: 16 K heads, 32 V heads, key_dim=128/head, val_dim=128/head
 *     key_dim_total=2048, value_dim_total=4096, conv_dim=8192
 *     State: float32[32, 128, 128] per layer
 *   MoE: 256 experts, 8 active, expert_dim=512, shared_expert_dim=512
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#import <sys/stat.h>
#import <fcntl.h>
#include <dlfcn.h>
#import <math.h>
#import "full_forward.h"
#import "bakan_forward.h"  // for MoE block weight types

// ---------------------------------------------------------------------------
// Model constants — Qwen3.5-122B-A10B
// ---------------------------------------------------------------------------
#define HIDDEN_SIZE         3072
#define VOCAB_SIZE          248320
#define NUM_LAYERS          48
#define GROUP_SIZE          64
#define RMS_NORM_EPS        1e-6f

// Full attention (every 4th layer: 3,7,11,...,47)
#define NUM_Q_HEADS         32
#define NUM_KV_HEADS        2
#define HEAD_DIM            256
#define ROPE_DIM            64        // partial_rotary_factor=0.25 * 256
#define ROPE_THETA          10000000.0f
#define Q_PROJ_OUT          16384     // 32 * 256 * 2 (includes gate)
#define K_PROJ_OUT          512       // 2 * 256
#define V_PROJ_OUT          512       // 2 * 256
#define ATTN_OUT_DIM        8192      // 32 * 256
#define MAX_SEQ_LEN         4096      // max KV cache length

// Linear attention (GatedDeltaNet) — 36 of 48 layers
#define LIN_NUM_K_HEADS     16
#define LIN_NUM_V_HEADS     64
#define LIN_KEY_DIM         128       // per head
#define LIN_VAL_DIM         128       // per head
#define LIN_KEY_TOTAL       2048      // 16 * 128
#define LIN_VAL_TOTAL       8192      // 64 * 128
#define LIN_CONV_DIM        12288     // key_dim*2 + value_dim = 2048*2 + 8192
#define CONV_KERNEL_SIZE    4

// Combined projection dimensions
// Linear: QKV(12288) + Z(8192) + B(64) + A(64) = 20608
#define LIN_COMBINED_PROJ_OUT 20608
// Full: Q+gate(16384) + K(512) + V(512) = 17408
#define ATTN_COMBINED_PROJ_OUT 17408

// Offsets into combined output for linear attention
#define LIN_COMBINED_QKV_OFF    0
#define LIN_COMBINED_Z_OFF      12288
#define LIN_COMBINED_B_OFF      20480
#define LIN_COMBINED_A_OFF      20544

// Offsets into combined output for full attention
#define ATTN_COMBINED_Q_OFF     0
#define ATTN_COMBINED_K_OFF     16384
#define ATTN_COMBINED_V_OFF     16896

// MoE — 256 experts, 8 active, dynamic expert size
#define NUM_EXPERTS_TOTAL   256
#define NUM_EXPERTS_PER_TOK 8
#define EXPERT_DIM          1024      // moe_intermediate_size for 122B
#define EXPERT_SIZE         5308416UL // from packed binary header
#define HEADER_SIZE     4096UL
#define MAX_EXPERTS         16

// Expert tensor byte offsets (from packed binary header — 122B specific)
// These are computed from the tensor shapes:
// gate_proj.weight: (1024, 384) uint32 = 1572864 bytes
// gate_proj.scales: (1024, 48)  bf16   = 98304 bytes
// gate_proj.biases: (1024, 48)  bf16   = 98304 bytes
// up_proj: same as gate_proj
// down_proj.weight: (3072, 128) uint32 = 1572864 bytes
// down_proj.scales: (3072, 16)  bf16   = 98304 bytes
// down_proj.biases: (3072, 16)  bf16   = 98304 bytes
#define EXPERT_GATE_WEIGHT_OFFSET   0
#define EXPERT_GATE_SCALES_OFFSET   1572864
#define EXPERT_GATE_BIASES_OFFSET   1671168
#define EXPERT_UP_WEIGHT_OFFSET     1769472
#define EXPERT_UP_SCALES_OFFSET     3342336
#define EXPERT_UP_BIASES_OFFSET     3440640
#define EXPERT_DOWN_WEIGHT_OFFSET   3538944
#define EXPERT_DOWN_SCALES_OFFSET   5111808
#define EXPERT_DOWN_BIASES_OFFSET   5210112

// ---------------------------------------------------------------------------
// Weight name hash table: maps parameter name -> (layer, type) pair
// We store weights as Metal buffers organized by name.
// ---------------------------------------------------------------------------

// Maximum number of unique weight names
#define MAX_WEIGHTS         2048

typedef struct {
    char name[256];
    id<MTLBuffer> buffer;
    int layer;           // -1 for non-layer weights
} WeightEntry;

// ---------------------------------------------------------------------------
// ExpertMLPParams — must match expert_mlp.metal
// ---------------------------------------------------------------------------
typedef struct __attribute__((packed)) {
    int32_t  expert_indices[MAX_EXPERTS];
    int32_t  num_experts;
    int32_t  hidden_size;
    int32_t  expert_dim;
    int32_t  group_size;
    uint64_t header_size;
    uint64_t expert_size;
    uint64_t gate_weight_offset;
    uint64_t gate_scales_offset;
    uint64_t gate_biases_offset;
    uint64_t up_weight_offset;
    uint64_t up_scales_offset;
    uint64_t up_biases_offset;
    uint64_t down_weight_offset;
    uint64_t down_scales_offset;
    uint64_t down_biases_offset;
} ExpertMLPParams;

// ---------------------------------------------------------------------------
// MoeBlockParams — matches moe_block.metal
// ---------------------------------------------------------------------------
typedef struct {
    int32_t  hidden_size;
    int32_t  expert_dim;
    int32_t  num_experts;
    int32_t  num_experts_per_tok;
    int32_t  group_size;
    float    rms_norm_eps;
} MoeBlockParams;

// ---------------------------------------------------------------------------
// Engine state
// ---------------------------------------------------------------------------
typedef struct {
    id<MTLDevice>                device;
    id<MTLCommandQueue>          queue;

    // --- Compute pipelines ---
    // From attention.metal:
    id<MTLComputePipelineState>  embedLookupPipeline;
    id<MTLComputePipelineState>  rmsnormPipeline;
    id<MTLComputePipelineState>  matvec4bitPipeline;
    id<MTLComputePipelineState>  matvec4bitF32Pipeline;
    id<MTLComputePipelineState>  ropePipeline;
    id<MTLComputePipelineState>  gqaAttnPipeline;
    id<MTLComputePipelineState>  kvCacheAppendPipeline;
    id<MTLComputePipelineState>  rmsnormPerHeadPipeline;
    id<MTLComputePipelineState>  rmsnormPerHeadNoWtPipeline;
    id<MTLComputePipelineState>  sigmoidGateMulPipeline;
    id<MTLComputePipelineState>  residualAddPipeline;
    id<MTLComputePipelineState>  conv1dSiluPipeline;
    id<MTLComputePipelineState>  gatedDeltaStatePipeline;
    id<MTLComputePipelineState>  rmsnormGatedPipeline;
    id<MTLComputePipelineState>  scaleVectorPipeline;
    id<MTLComputePipelineState>  convBufUpdatePipeline;
    id<MTLComputePipelineState>  splitQkvPipeline;
    id<MTLComputePipelineState>  splitQGatePipeline;
    id<MTLComputePipelineState>  rmsnormMatvec4bitPipeline;

    // From expert_mlp.metal:
    id<MTLComputePipelineState>  expertUpGatePipeline;
    id<MTLComputePipelineState>  expertDownPipeline;
    id<MTLComputePipelineState>  expertMLPFusedPipeline;

    // From moe_block.metal:
    id<MTLComputePipelineState>  routerMatmulPipeline;
    id<MTLComputePipelineState>  sharedUpGatePipeline;
    id<MTLComputePipelineState>  sharedDownPipeline;
    id<MTLComputePipelineState>  sharedGatePipeline;
    id<MTLComputePipelineState>  blendPipeline;
    id<MTLComputePipelineState>  blendResidualPipeline;
    id<MTLComputePipelineState>  routerSoftmaxTopkPipeline;

    // --- Weight storage (by name) ---
    WeightEntry*    weights;
    int             weight_count;

    // --- Expert file descriptors ---
    int*            layer_fds;
    int             num_layers;

    // --- Scratch buffers ---
    id<MTLBuffer>   hidden_buf;          // half[HIDDEN_SIZE] — main hidden state
    id<MTLBuffer>   hidden_buf2;         // half[HIDDEN_SIZE] — secondary (for residual)
    id<MTLBuffer>   normed_buf;          // half[HIDDEN_SIZE] — after RMSNorm
    id<MTLBuffer>   attn_out_buf;        // half[HIDDEN_SIZE] — attention output
    id<MTLBuffer>   proj_buf;            // half[max(Q_PROJ_OUT, LIN_CONV_DIM + LIN_VAL_TOTAL)]
    id<MTLBuffer>   proj_buf2;           // half[max(K_PROJ_OUT, LIN_NUM_V_HEADS*2)]
    id<MTLBuffer>   proj_buf3;           // half[V_PROJ_OUT]
    id<MTLBuffer>   gate_buf;            // half[ATTN_OUT_DIM] — for attention gate
    id<MTLBuffer>   gated_attn_buf;      // half[ATTN_OUT_DIM] — sigmoid(gate)*attn
    id<MTLBuffer>   q_buf;               // half[ATTN_OUT_DIM] — Q after RoPE
    id<MTLBuffer>   k_buf;               // half[K_PROJ_OUT] — K after RoPE
    id<MTLBuffer>   v_buf;               // half[V_PROJ_OUT] — V
    id<MTLBuffer>   attn_raw_buf;        // half[ATTN_OUT_DIM] — raw attention output

    // Combined projection output (fused attention input projections)
    id<MTLBuffer>   combined_proj_buf;   // half[max(LIN_COMBINED_PROJ_OUT, ATTN_COMBINED_PROJ_OUT)]

    // Linear attention scratch
    id<MTLBuffer>   lin_qkv_buf;         // half[LIN_CONV_DIM] — conv output (q,k,v interleaved)
    id<MTLBuffer>   lin_z_buf;           // half[LIN_VAL_TOTAL] — z gate
    id<MTLBuffer>   lin_b_buf;           // half[LIN_NUM_V_HEADS] — beta input
    id<MTLBuffer>   lin_a_buf;           // half[LIN_NUM_V_HEADS] — alpha input
    id<MTLBuffer>   lin_q_buf;           // half[LIN_KEY_TOTAL]
    id<MTLBuffer>   lin_k_buf;           // half[LIN_KEY_TOTAL]
    id<MTLBuffer>   lin_v_buf;           // half[LIN_VAL_TOTAL]
    id<MTLBuffer>   lin_out_buf;         // half[LIN_VAL_TOTAL] — state update output
    id<MTLBuffer>   lin_normed_buf;      // half[LIN_VAL_TOTAL] — after gated norm

    // Conv1d buffers: [num_layers][conv_dim * kernel_size]
    // Stored as CFTypeRef to avoid ARC issues with calloc'd arrays
    CFTypeRef*      conv_bufs;           // per-layer conv state

    // KV cache: per full-attention layer
    // k_cache[attn_layer_idx]: half[NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM]
    // v_cache[attn_layer_idx]: same
    CFTypeRef*      k_caches;
    CFTypeRef*      v_caches;
    int             num_attn_layers;     // 10

    // Linear attention state: per linear-attention layer
    // state[lin_layer_idx]: float[LIN_NUM_V_HEADS * LIN_VAL_DIM * LIN_KEY_DIM]
    CFTypeRef*      lin_states;
    int             num_lin_layers;      // 30

    // MoE scratch buffers
    id<MTLBuffer>   staging_buffer;      // for pread experts
    id<MTLBuffer>   expert_activated;    // float[K*EXPERT_DIM]
    id<MTLBuffer>   expert_output;       // float[K*HIDDEN_SIZE]
    id<MTLBuffer>   shared_activated;    // float[EXPERT_DIM]
    id<MTLBuffer>   shared_out_buf;      // half[HIDDEN_SIZE]
    id<MTLBuffer>   shared_gate_buf;     // float[1]
    id<MTLBuffer>   moe_out_buf;         // half[HIDDEN_SIZE]
    id<MTLBuffer>   scores_buffer;       // float[K]
    id<MTLBuffer>   expert_params_buf;   // ExpertMLPParams
    id<MTLBuffer>   moe_params_buf;      // MoeBlockParams
    id<MTLBuffer>   router_logits_buf;   // float[256]
    id<MTLBuffer>   expert_indices_buf;  // int32[K] — GPU softmax+topk output

    // LM head output
    id<MTLBuffer>   logits_buf;          // float[VOCAB_SIZE]

    // --- Pre-allocated constant parameter buffers (reused every forward pass) ---
    // Dimension buffers (int32)
    id<MTLBuffer>   cbuf_hidden_size;     // HIDDEN_SIZE (2048)
    id<MTLBuffer>   cbuf_vocab_size;      // VOCAB_SIZE (248320)
    id<MTLBuffer>   cbuf_group_size;      // GROUP_SIZE (64)
    id<MTLBuffer>   cbuf_q_proj_out;      // Q_PROJ_OUT (8192)
    id<MTLBuffer>   cbuf_k_proj_out;      // K_PROJ_OUT (512)
    id<MTLBuffer>   cbuf_v_proj_out;      // V_PROJ_OUT (512)
    id<MTLBuffer>   cbuf_attn_out_dim;    // ATTN_OUT_DIM (4096)
    id<MTLBuffer>   cbuf_num_q_heads;     // NUM_Q_HEADS (16)
    id<MTLBuffer>   cbuf_num_kv_heads;    // NUM_KV_HEADS (2)
    id<MTLBuffer>   cbuf_head_dim;        // HEAD_DIM (256)
    id<MTLBuffer>   cbuf_rope_dim;        // ROPE_DIM (64)
    id<MTLBuffer>   cbuf_max_seq_len;     // MAX_SEQ_LEN (4096)
    id<MTLBuffer>   cbuf_lin_conv_dim;    // LIN_CONV_DIM (8192)
    id<MTLBuffer>   cbuf_conv_kernel;     // CONV_KERNEL_SIZE (4)
    id<MTLBuffer>   cbuf_lin_num_k_heads; // LIN_NUM_K_HEADS (16)
    id<MTLBuffer>   cbuf_lin_key_dim;     // LIN_KEY_DIM (128)
    id<MTLBuffer>   cbuf_lin_key_total;   // LIN_KEY_TOTAL (2048)
    id<MTLBuffer>   cbuf_lin_num_v_heads; // LIN_NUM_V_HEADS (32)
    id<MTLBuffer>   cbuf_lin_val_dim;     // LIN_VAL_DIM (128)
    id<MTLBuffer>   cbuf_lin_val_total;   // LIN_VAL_TOTAL (4096)

    // Combined projection dimension constants
    id<MTLBuffer>   cbuf_lin_combined_proj_out;  // LIN_COMBINED_PROJ_OUT (12352)
    id<MTLBuffer>   cbuf_attn_combined_proj_out; // ATTN_COMBINED_PROJ_OUT (9216)

    // Float constant buffers
    id<MTLBuffer>   cbuf_rms_norm_eps;    // RMS_NORM_EPS (1e-6)
    id<MTLBuffer>   cbuf_rope_theta;      // ROPE_THETA (10000000.0)
    id<MTLBuffer>   cbuf_q_scale;         // 1/(key_dim) = 1/128
    id<MTLBuffer>   cbuf_k_scale;         // 1/sqrt(key_dim) = 1/sqrt(128)

    // Reusable mutable buffers for per-call changing values (int32)
    id<MTLBuffer>   cbuf_position;        // updated each forward call
    id<MTLBuffer>   cbuf_kv_len;          // = position + 1, updated each forward call

    // Is the engine ready for forward passes?
    int             built;
} BakanFullEngine;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline uint16_t float_to_half(float f) {
    _Float16 h = (_Float16)f;
    uint16_t result;
    memcpy(&result, &h, sizeof(result));
    return result;
}

static inline float half_to_float(uint16_t h) {
    _Float16 f16;
    memcpy(&f16, &h, sizeof(f16));
    return (float)f16;
}

// CPU bfloat16 -> float
static inline float bf16_to_float(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// CPU dequant dot product (4-bit) for router on CPU
static float cpu_dequant_dot_f16_4bit(
    const uint32_t* weight_row,
    const uint16_t* x_f16,
    const uint16_t* scales_bf16,
    const uint16_t* biases_bf16,
    int in_dim,
    int group_size
) {
    float acc = 0.0f;
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_float(scales_bf16[g]);
        float bias  = bf16_to_float(biases_bf16[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            int x_base = g * group_size + p * 8;

            for (int n = 0; n < 8; n++) {
                float nibble = (float)((packed >> (n * 4)) & 0xF);
                float w = nibble * scale + bias;
                acc += w * half_to_float(x_f16[x_base + n]);
            }
        }
    }
    return acc;
}

// Helper: get MTLBuffer from CFTypeRef
static inline id<MTLBuffer> buf_from_ref(CFTypeRef ref) {
    if (!ref) return nil;
    return (__bridge id<MTLBuffer>)ref;
}

// Find weight by name
static id<MTLBuffer> find_weight(BakanFullEngine* e, const char* name) {
    for (int i = 0; i < e->weight_count; i++) {
        if (strcmp(e->weights[i].name, name) == 0) {
            return e->weights[i].buffer;
        }
    }
    return nil;
}

// Make a constant int32 buffer
static id<MTLBuffer> make_int_buf(BakanFullEngine* e, int32_t val) {
    id<MTLBuffer> buf = [e->device newBufferWithLength:sizeof(int32_t)
                                               options:MTLResourceStorageModeShared];
    *(int32_t*)buf.contents = val;
    return buf;
}

// Make a constant float buffer
static id<MTLBuffer> make_float_buf(BakanFullEngine* e, float val) {
    id<MTLBuffer> buf = [e->device newBufferWithLength:sizeof(float)
                                               options:MTLResourceStorageModeShared];
    *(float*)buf.contents = val;
    return buf;
}

// ---------------------------------------------------------------------------
// Compile shader file
// ---------------------------------------------------------------------------
static id<MTLLibrary> compile_shader_file(id<MTLDevice> device,
                                           const char* packed_dir,
                                           NSString* filename) {
    // Find metal shader files relative to this dylib's location
    Dl_info dl_info;
    NSString* dylibDir = @".";
    if (dladdr((void*)compile_shader_file, &dl_info)) {
        dylibDir = [[NSString stringWithUTF8String:dl_info.dli_fname]
                     stringByDeletingLastPathComponent];
    }
    NSArray* searchPaths = @[
        [NSString stringWithFormat:@"%@/%@", dylibDir, filename],
        [NSString stringWithFormat:@"kandiga/metal/%@", filename],
        [NSString stringWithFormat:@"%s/../../../kandiga/metal/%@", packed_dir, filename],
    ];
    NSString* source = nil;
    for (NSString* p in searchPaths) {
        source = [NSString stringWithContentsOfFile:p
                                           encoding:NSUTF8StringEncoding
                                              error:nil];
        if (source) {
            fprintf(stderr, "[bakan_full] Loaded shader: %s\n", p.UTF8String);
            break;
        }
    }
    if (!source) {
        fprintf(stderr, "[bakan_full] ERROR: Could not find %s\n", filename.UTF8String);
        return nil;
    }

    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_1;
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
    if (!lib) {
        fprintf(stderr, "[bakan_full] ERROR: Compilation of %s failed: %s\n",
                filename.UTF8String, error.localizedDescription.UTF8String);
    }
    return lib;
}

// ---------------------------------------------------------------------------
// bakan_full_init
// ---------------------------------------------------------------------------
void* bakan_full_init(const char* packed_expert_dir, int num_layers) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[bakan_full] ERROR: No Metal device found\n");
            return NULL;
        }

        // Compile all shader files
        id<MTLLibrary> attnLib   = compile_shader_file(device, packed_expert_dir, @"attention.metal");
        id<MTLLibrary> expertLib = compile_shader_file(device, packed_expert_dir, @"expert_mlp.metal");
        id<MTLLibrary> moeLib    = compile_shader_file(device, packed_expert_dir, @"moe_block.metal");

        if (!attnLib || !expertLib || !moeLib) return NULL;

        // Create compute pipelines
        NSError* error = nil;

        #define PIPELINE(lib, name, var) \
            id<MTLFunction> name##Func = [lib newFunctionWithName:@#name]; \
            if (!name##Func) { \
                fprintf(stderr, "[bakan_full] ERROR: Function '%s' not found\n", #name); \
                return NULL; \
            } \
            id<MTLComputePipelineState> var = \
                [device newComputePipelineStateWithFunction:name##Func error:&error]; \
            if (!var) { \
                fprintf(stderr, "[bakan_full] ERROR: Pipeline '%s': %s\n", \
                        #name, error.localizedDescription.UTF8String); \
                return NULL; \
            }

        // attention.metal
        PIPELINE(attnLib, embed_lookup,              embedPL)
        PIPELINE(attnLib, rmsnorm_forward,           normPL)
        PIPELINE(attnLib, matvec_4bit,               mv4PL)
        PIPELINE(attnLib, matvec_4bit_f32out,        mv4f32PL)
        PIPELINE(attnLib, rope_kernel,               ropePL)
        PIPELINE(attnLib, gqa_attention_decode,      gqaPL)
        PIPELINE(attnLib, kv_cache_append,           kvAppPL)
        PIPELINE(attnLib, rmsnorm_per_head,          normPerHeadPL)
        PIPELINE(attnLib, rmsnorm_per_head_no_weight, normPerHeadNWPL)
        PIPELINE(attnLib, sigmoid_gate_mul,          sigGatePL)
        PIPELINE(attnLib, residual_add,              resAddPL)
        PIPELINE(attnLib, depthwise_conv1d_silu,     conv1dPL)
        PIPELINE(attnLib, gated_delta_state_update,  gdStatePL)
        PIPELINE(attnLib, rmsnorm_gated_kernel,      normGatedPL)
        PIPELINE(attnLib, scale_vector,              scalePL)
        PIPELINE(attnLib, conv1d_buffer_update,      convUpdPL)
        PIPELINE(attnLib, split_qkv,                 splitQkvPL)
        PIPELINE(attnLib, split_q_gate,              splitQGatePL)
        PIPELINE(attnLib, rmsnorm_matvec_4bit,      rnMv4PL)

        // expert_mlp.metal
        PIPELINE(expertLib, expert_up_gate_swiglu,   expUpPL)
        PIPELINE(expertLib, expert_down_proj,        expDownPL)
        PIPELINE(expertLib, expert_mlp_fused,        expFusedPL)

        // moe_block.metal
        PIPELINE(moeLib, router_matmul_kernel,       routerPL)
        PIPELINE(moeLib, shared_expert_up_gate_kernel, shUpGPL)
        PIPELINE(moeLib, shared_expert_down_kernel,  shDownPL)
        PIPELINE(moeLib, shared_gate_kernel,         shGatePL)
        PIPELINE(moeLib, blend_kernel,               blendPL)
        PIPELINE(moeLib, blend_residual_kernel,     blendResPL)
        PIPELINE(moeLib, router_softmax_topk,       routerStkPL)

        #undef PIPELINE

        // Allocate engine
        BakanFullEngine* e = (BakanFullEngine*)calloc(1, sizeof(BakanFullEngine));
        if (!e) return NULL;

        e->device = device;
        e->queue  = [device newCommandQueue];
        e->num_layers = num_layers;
        e->built = 0;

        // Assign pipelines
        e->embedLookupPipeline      = embedPL;
        e->rmsnormPipeline          = normPL;
        e->matvec4bitPipeline       = mv4PL;
        e->matvec4bitF32Pipeline    = mv4f32PL;
        e->ropePipeline             = ropePL;
        e->gqaAttnPipeline          = gqaPL;
        e->kvCacheAppendPipeline    = kvAppPL;
        e->rmsnormPerHeadPipeline   = normPerHeadPL;
        e->rmsnormPerHeadNoWtPipeline = normPerHeadNWPL;
        e->sigmoidGateMulPipeline   = sigGatePL;
        e->residualAddPipeline      = resAddPL;
        e->conv1dSiluPipeline       = conv1dPL;
        e->gatedDeltaStatePipeline  = gdStatePL;
        e->rmsnormGatedPipeline     = normGatedPL;
        e->scaleVectorPipeline      = scalePL;
        e->convBufUpdatePipeline    = convUpdPL;
        e->splitQkvPipeline         = splitQkvPL;
        e->splitQGatePipeline       = splitQGatePL;
        e->rmsnormMatvec4bitPipeline = rnMv4PL;
        e->expertUpGatePipeline     = expUpPL;
        e->expertDownPipeline       = expDownPL;
        e->expertMLPFusedPipeline   = expFusedPL;
        e->routerMatmulPipeline     = routerPL;
        e->sharedUpGatePipeline     = shUpGPL;
        e->sharedDownPipeline       = shDownPL;
        e->sharedGatePipeline       = shGatePL;
        e->blendPipeline            = blendPL;
        e->blendResidualPipeline    = blendResPL;
        e->routerSoftmaxTopkPipeline = routerStkPL;

        // Weight storage
        e->weights = (WeightEntry*)calloc(MAX_WEIGHTS, sizeof(WeightEntry));
        e->weight_count = 0;

        // Open expert file descriptors
        e->layer_fds = (int*)calloc((size_t)num_layers, sizeof(int));
        for (int i = 0; i < num_layers; i++) e->layer_fds[i] = -1;

        NSString* dirStr = [NSString stringWithUTF8String:packed_expert_dir];
        for (int i = 0; i < num_layers; i++) {
            NSString* filename = [NSString stringWithFormat:@"layer_%02d.bin", i];
            NSString* path = [dirStr stringByAppendingPathComponent:filename];
            int fd = open(path.UTF8String, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "[bakan_full] WARNING: Cannot open %s (expert file): %s\n",
                        path.UTF8String, strerror(errno));
                // Not fatal — might not have expert files for all layers
            }
            e->layer_fds[i] = fd;
        }

        // Allocate scratch buffers
        #define ALLOC_BUF(var, sz) \
            e->var = [device newBufferWithLength:(sz) options:MTLResourceStorageModeShared]; \
            if (!e->var) { \
                fprintf(stderr, "[bakan_full] ERROR: Buffer '%s' alloc failed (%zu)\n", #var, (size_t)(sz)); \
                free(e->weights); free(e->layer_fds); free(e); return NULL; \
            }

        ALLOC_BUF(hidden_buf,       HIDDEN_SIZE * sizeof(uint16_t))
        ALLOC_BUF(hidden_buf2,      HIDDEN_SIZE * sizeof(uint16_t))
        ALLOC_BUF(normed_buf,       HIDDEN_SIZE * sizeof(uint16_t))
        ALLOC_BUF(attn_out_buf,     HIDDEN_SIZE * sizeof(uint16_t))

        // Proj buffers: largest projection is q_proj = 8192, or lin qkv = 8192+4096 = 12288
        size_t proj_size = (LIN_CONV_DIM + LIN_VAL_TOTAL) * sizeof(uint16_t);
        ALLOC_BUF(proj_buf,         proj_size)

        // Combined projection output: max(12352, 9216) = 12352
        size_t combined_size = LIN_COMBINED_PROJ_OUT * sizeof(uint16_t);
        ALLOC_BUF(combined_proj_buf, combined_size)
        ALLOC_BUF(proj_buf2,        K_PROJ_OUT * sizeof(uint16_t))
        ALLOC_BUF(proj_buf3,        V_PROJ_OUT * sizeof(uint16_t))
        ALLOC_BUF(gate_buf,         ATTN_OUT_DIM * sizeof(uint16_t))
        ALLOC_BUF(gated_attn_buf,   ATTN_OUT_DIM * sizeof(uint16_t))
        ALLOC_BUF(q_buf,            ATTN_OUT_DIM * sizeof(uint16_t))
        ALLOC_BUF(k_buf,            K_PROJ_OUT * sizeof(uint16_t))
        ALLOC_BUF(v_buf,            V_PROJ_OUT * sizeof(uint16_t))
        ALLOC_BUF(attn_raw_buf,     ATTN_OUT_DIM * sizeof(uint16_t))

        // Linear attention
        ALLOC_BUF(lin_qkv_buf,      LIN_CONV_DIM * sizeof(uint16_t))
        ALLOC_BUF(lin_z_buf,        LIN_VAL_TOTAL * sizeof(uint16_t))
        ALLOC_BUF(lin_b_buf,        LIN_NUM_V_HEADS * sizeof(uint16_t))
        ALLOC_BUF(lin_a_buf,        LIN_NUM_V_HEADS * sizeof(uint16_t))
        ALLOC_BUF(lin_q_buf,        LIN_KEY_TOTAL * sizeof(uint16_t))
        ALLOC_BUF(lin_k_buf,        LIN_KEY_TOTAL * sizeof(uint16_t))
        ALLOC_BUF(lin_v_buf,        LIN_VAL_TOTAL * sizeof(uint16_t))
        ALLOC_BUF(lin_out_buf,      LIN_VAL_TOTAL * sizeof(uint16_t))
        ALLOC_BUF(lin_normed_buf,   LIN_VAL_TOTAL * sizeof(uint16_t))

        // Conv1d buffers per linear layer
        e->num_lin_layers = 0;
        e->num_attn_layers = 0;
        for (int i = 0; i < num_layers; i++) {
            int is_attn = ((i + 1) % 4 == 0);
            if (is_attn) e->num_attn_layers++;
            else e->num_lin_layers++;
        }

        e->conv_bufs = (CFTypeRef*)calloc((size_t)e->num_lin_layers, sizeof(CFTypeRef));
        size_t conv_buf_size = (size_t)LIN_CONV_DIM * CONV_KERNEL_SIZE * sizeof(uint16_t);
        for (int i = 0; i < e->num_lin_layers; i++) {
            id<MTLBuffer> cb = [device newBufferWithLength:conv_buf_size
                                                   options:MTLResourceStorageModeShared];
            memset(cb.contents, 0, conv_buf_size);
            e->conv_bufs[i] = CFBridgingRetain(cb);
        }

        // KV caches for full attention layers
        e->k_caches = (CFTypeRef*)calloc((size_t)e->num_attn_layers, sizeof(CFTypeRef));
        e->v_caches = (CFTypeRef*)calloc((size_t)e->num_attn_layers, sizeof(CFTypeRef));
        size_t kv_size = (size_t)NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(uint16_t);
        for (int i = 0; i < e->num_attn_layers; i++) {
            id<MTLBuffer> kb = [device newBufferWithLength:kv_size
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> vb = [device newBufferWithLength:kv_size
                                                   options:MTLResourceStorageModeShared];
            memset(kb.contents, 0, kv_size);
            memset(vb.contents, 0, kv_size);
            e->k_caches[i] = CFBridgingRetain(kb);
            e->v_caches[i] = CFBridgingRetain(vb);
        }

        // Linear attention states
        e->lin_states = (CFTypeRef*)calloc((size_t)e->num_lin_layers, sizeof(CFTypeRef));
        size_t state_size = (size_t)LIN_NUM_V_HEADS * LIN_VAL_DIM * LIN_KEY_DIM * sizeof(float);
        for (int i = 0; i < e->num_lin_layers; i++) {
            id<MTLBuffer> sb = [device newBufferWithLength:state_size
                                                   options:MTLResourceStorageModeShared];
            memset(sb.contents, 0, state_size);
            e->lin_states[i] = CFBridgingRetain(sb);
        }

        // MoE scratch
        size_t staging_size = (size_t)MAX_EXPERTS * EXPERT_SIZE;
        ALLOC_BUF(staging_buffer,    staging_size)
        ALLOC_BUF(expert_activated,  MAX_EXPERTS * EXPERT_DIM * sizeof(float))
        ALLOC_BUF(expert_output,     MAX_EXPERTS * HIDDEN_SIZE * sizeof(float))
        ALLOC_BUF(shared_activated,  EXPERT_DIM * sizeof(float))
        ALLOC_BUF(shared_out_buf,    HIDDEN_SIZE * sizeof(uint16_t))
        ALLOC_BUF(shared_gate_buf,   sizeof(float))
        ALLOC_BUF(moe_out_buf,      HIDDEN_SIZE * sizeof(uint16_t))
        ALLOC_BUF(scores_buffer,     MAX_EXPERTS * sizeof(float))
        ALLOC_BUF(expert_params_buf, sizeof(ExpertMLPParams))
        ALLOC_BUF(moe_params_buf,    sizeof(MoeBlockParams))
        ALLOC_BUF(router_logits_buf, NUM_EXPERTS_TOTAL * sizeof(float))
        ALLOC_BUF(expert_indices_buf, MAX_EXPERTS * sizeof(int32_t))

        // LM head
        ALLOC_BUF(logits_buf,        VOCAB_SIZE * sizeof(float))

        #undef ALLOC_BUF

        // Pre-allocate constant parameter buffers (reused every forward pass)
        // These replace ~34 make_int_buf/make_float_buf calls per layer (1360/token)
        #define CONST_INT_BUF(field, val) \
            e->field = [device newBufferWithLength:sizeof(int32_t) \
                                           options:MTLResourceStorageModeShared]; \
            *(int32_t*)e->field.contents = (int32_t)(val);

        #define CONST_FLOAT_BUF(field, val) \
            e->field = [device newBufferWithLength:sizeof(float) \
                                           options:MTLResourceStorageModeShared]; \
            *(float*)e->field.contents = (float)(val);

        // Integer dimension constants
        CONST_INT_BUF(cbuf_hidden_size,     HIDDEN_SIZE)
        CONST_INT_BUF(cbuf_vocab_size,      VOCAB_SIZE)
        CONST_INT_BUF(cbuf_group_size,      GROUP_SIZE)
        CONST_INT_BUF(cbuf_q_proj_out,      Q_PROJ_OUT)
        CONST_INT_BUF(cbuf_k_proj_out,      K_PROJ_OUT)
        CONST_INT_BUF(cbuf_v_proj_out,      V_PROJ_OUT)
        CONST_INT_BUF(cbuf_attn_out_dim,    ATTN_OUT_DIM)
        CONST_INT_BUF(cbuf_num_q_heads,     NUM_Q_HEADS)
        CONST_INT_BUF(cbuf_num_kv_heads,    NUM_KV_HEADS)
        CONST_INT_BUF(cbuf_head_dim,        HEAD_DIM)
        CONST_INT_BUF(cbuf_rope_dim,        ROPE_DIM)
        CONST_INT_BUF(cbuf_max_seq_len,     MAX_SEQ_LEN)
        CONST_INT_BUF(cbuf_lin_conv_dim,    LIN_CONV_DIM)
        CONST_INT_BUF(cbuf_conv_kernel,     CONV_KERNEL_SIZE)
        CONST_INT_BUF(cbuf_lin_num_k_heads, LIN_NUM_K_HEADS)
        CONST_INT_BUF(cbuf_lin_key_dim,     LIN_KEY_DIM)
        CONST_INT_BUF(cbuf_lin_key_total,   LIN_KEY_TOTAL)
        CONST_INT_BUF(cbuf_lin_num_v_heads, LIN_NUM_V_HEADS)
        CONST_INT_BUF(cbuf_lin_val_dim,     LIN_VAL_DIM)
        CONST_INT_BUF(cbuf_lin_val_total,   LIN_VAL_TOTAL)

        // Combined projection dimensions
        CONST_INT_BUF(cbuf_lin_combined_proj_out,  LIN_COMBINED_PROJ_OUT)
        CONST_INT_BUF(cbuf_attn_combined_proj_out, ATTN_COMBINED_PROJ_OUT)

        // Float constants
        CONST_FLOAT_BUF(cbuf_rms_norm_eps,  RMS_NORM_EPS)
        CONST_FLOAT_BUF(cbuf_rope_theta,    ROPE_THETA)
        {
            float inv_scale = 1.0f / sqrtf((float)LIN_KEY_DIM);
            CONST_FLOAT_BUF(cbuf_q_scale, inv_scale * inv_scale)  // 1/key_dim
            CONST_FLOAT_BUF(cbuf_k_scale, inv_scale)              // 1/sqrt(key_dim)
        }

        // Reusable mutable buffers for per-call values
        CONST_INT_BUF(cbuf_position, 0)
        CONST_INT_BUF(cbuf_kv_len,   1)

        #undef CONST_INT_BUF
        #undef CONST_FLOAT_BUF

        fprintf(stderr, "[bakan_full] Pre-allocated %d constant parameter buffers\n", 26);

        // Initialize MoE constant params
        ExpertMLPParams* ep = (ExpertMLPParams*)e->expert_params_buf.contents;
        memset(ep, 0, sizeof(ExpertMLPParams));
        ep->hidden_size        = HIDDEN_SIZE;
        ep->expert_dim         = EXPERT_DIM;
        ep->group_size         = GROUP_SIZE;
        ep->header_size        = 0;
        ep->expert_size        = EXPERT_SIZE;
        ep->gate_weight_offset = EXPERT_GATE_WEIGHT_OFFSET;
        ep->gate_scales_offset = EXPERT_GATE_SCALES_OFFSET;
        ep->gate_biases_offset = EXPERT_GATE_BIASES_OFFSET;
        ep->up_weight_offset   = EXPERT_UP_WEIGHT_OFFSET;
        ep->up_scales_offset   = EXPERT_UP_SCALES_OFFSET;
        ep->up_biases_offset   = EXPERT_UP_BIASES_OFFSET;
        ep->down_weight_offset = EXPERT_DOWN_WEIGHT_OFFSET;
        ep->down_scales_offset = EXPERT_DOWN_SCALES_OFFSET;
        ep->down_biases_offset = EXPERT_DOWN_BIASES_OFFSET;

        MoeBlockParams* mp = (MoeBlockParams*)e->moe_params_buf.contents;
        mp->hidden_size        = HIDDEN_SIZE;
        mp->expert_dim         = EXPERT_DIM;
        mp->num_experts        = NUM_EXPERTS_TOTAL;
        mp->num_experts_per_tok = NUM_EXPERTS_PER_TOK;
        mp->group_size         = GROUP_SIZE;
        mp->rms_norm_eps       = RMS_NORM_EPS;

        fprintf(stderr, "[bakan_full] Initialized: %d layers (%d attn, %d linear), "
                "device=%s\n", num_layers, e->num_attn_layers, e->num_lin_layers,
                device.name.UTF8String);

        return e;
    }
}

// ---------------------------------------------------------------------------
// bakan_full_set_weight
// ---------------------------------------------------------------------------
int bakan_full_set_weight(void* engine, const char* name, const void* data,
                          size_t size, int ndim, const int64_t* shape, int dtype) {
    @autoreleasepool {
        BakanFullEngine* e = (BakanFullEngine*)engine;
        if (!e) return -1;
        if (e->weight_count >= MAX_WEIGHTS) {
            fprintf(stderr, "[bakan_full] ERROR: Too many weights (max %d)\n", MAX_WEIGHTS);
            return -1;
        }

        // Skip weights we don't need (vision, etc.)
        if (strstr(name, "vision") != NULL) return 1;
        if (strstr(name, "mtp.") != NULL) return 1;

        // Create Metal buffer with data
        id<MTLBuffer> buf = [e->device newBufferWithBytes:data
                                                   length:size
                                                  options:MTLResourceStorageModeShared];
        if (!buf) {
            fprintf(stderr, "[bakan_full] ERROR: Buffer alloc for '%s' failed (%zu bytes)\n",
                    name, size);
            return -1;
        }

        // Parse layer index from name
        int layer = -1;
        const char* layers_str = strstr(name, "layers.");
        if (layers_str) {
            layer = atoi(layers_str + 7);
        }

        WeightEntry* we = &e->weights[e->weight_count];
        strncpy(we->name, name, sizeof(we->name) - 1);
        we->name[sizeof(we->name) - 1] = '\0';
        we->buffer = buf;
        we->layer = layer;
        e->weight_count++;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// bakan_full_build
// ---------------------------------------------------------------------------
int bakan_full_build(void* engine) {
    @autoreleasepool {
        BakanFullEngine* e = (BakanFullEngine*)engine;
        if (!e) return -1;

        // Verify critical weights exist
        if (!find_weight(e, "embed_tokens.weight")) {
            fprintf(stderr, "[bakan_full] ERROR: embed_tokens.weight not found\n");
            return -1;
        }
        if (!find_weight(e, "norm.weight")) {
            fprintf(stderr, "[bakan_full] ERROR: norm.weight not found\n");
            return -1;
        }

        // Check for lm_head (not tied in this model)
        if (!find_weight(e, "lm_head.weight")) {
            fprintf(stderr, "[bakan_full] ERROR: lm_head.weight not found\n");
            return -1;
        }

        fprintf(stderr, "[bakan_full] Build complete: %d weights loaded\n", e->weight_count);
        e->built = 1;
        return 0;
    }
}

// ---------------------------------------------------------------------------
// Helper: build weight name for a layer
// ---------------------------------------------------------------------------
static void layer_weight_name(char* buf, int bufsz, int layer, const char* suffix) {
    snprintf(buf, bufsz, "layers.%d.%s", layer, suffix);
}

// ---------------------------------------------------------------------------
// Helper: dispatch matvec_4bit on given encoder
// ---------------------------------------------------------------------------
// Helper: get pre-allocated int buffer for a given constant dimension value
static id<MTLBuffer> get_dim_buf(BakanFullEngine* e, int val) {
    if (val == HIDDEN_SIZE)       return e->cbuf_hidden_size;
    if (val == VOCAB_SIZE)        return e->cbuf_vocab_size;
    if (val == GROUP_SIZE)        return e->cbuf_group_size;
    if (val == Q_PROJ_OUT)        return e->cbuf_q_proj_out;
    if (val == K_PROJ_OUT)        return e->cbuf_k_proj_out;
    if (val == V_PROJ_OUT)        return e->cbuf_v_proj_out;
    if (val == ATTN_OUT_DIM)      return e->cbuf_attn_out_dim;
    if (val == LIN_CONV_DIM)      return e->cbuf_lin_conv_dim;
    if (val == LIN_VAL_TOTAL)     return e->cbuf_lin_val_total;
    if (val == LIN_KEY_TOTAL)     return e->cbuf_lin_key_total;
    if (val == LIN_NUM_V_HEADS)   return e->cbuf_lin_num_v_heads;
    if (val == LIN_NUM_K_HEADS)   return e->cbuf_lin_num_k_heads;
    if (val == LIN_KEY_DIM)       return e->cbuf_lin_key_dim;
    if (val == LIN_VAL_DIM)       return e->cbuf_lin_val_dim;
    if (val == NUM_Q_HEADS)       return e->cbuf_num_q_heads;
    if (val == NUM_KV_HEADS)      return e->cbuf_num_kv_heads;
    if (val == HEAD_DIM)          return e->cbuf_head_dim;
    if (val == ROPE_DIM)          return e->cbuf_rope_dim;
    if (val == MAX_SEQ_LEN)       return e->cbuf_max_seq_len;
    if (val == CONV_KERNEL_SIZE)  return e->cbuf_conv_kernel;
    if (val == LIN_COMBINED_PROJ_OUT)  return e->cbuf_lin_combined_proj_out;
    if (val == ATTN_COMBINED_PROJ_OUT) return e->cbuf_attn_combined_proj_out;
    // Fallback for unexpected values (should not happen with correct usage)
    fprintf(stderr, "[bakan_full] WARNING: No pre-allocated buf for int %d, creating\n", val);
    return make_int_buf(e, val);
}

static void dispatch_matvec_4bit(
    BakanFullEngine* e,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> input,
    const char* w_name, const char* s_name, const char* b_name,
    id<MTLBuffer> output,
    int out_dim, int in_dim, int group_size
) {
    id<MTLBuffer> w = find_weight(e, w_name);
    id<MTLBuffer> s = find_weight(e, s_name);
    id<MTLBuffer> b = find_weight(e, b_name);

    if (!w || !s || !b) {
        fprintf(stderr, "[bakan_full] ERROR: Weight not found: %s\n", w_name);
        return;
    }

    // Use pre-allocated constant param buffers
    id<MTLBuffer> odBuf = get_dim_buf(e, out_dim);
    id<MTLBuffer> idBuf = get_dim_buf(e, in_dim);
    id<MTLBuffer> gsBuf = e->cbuf_group_size;

    [enc setComputePipelineState:e->matvec4bitPipeline];
    [enc setBuffer:input   offset:0 atIndex:0];
    [enc setBuffer:w       offset:0 atIndex:1];
    [enc setBuffer:s       offset:0 atIndex:2];
    [enc setBuffer:b       offset:0 atIndex:3];
    [enc setBuffer:output  offset:0 atIndex:4];
    [enc setBuffer:odBuf   offset:0 atIndex:5];
    [enc setBuffer:idBuf   offset:0 atIndex:6];
    [enc setBuffer:gsBuf   offset:0 atIndex:7];

    // SIMD dispatch: 8 rows per threadgroup (256 threads = 8 SIMD groups)
    NSUInteger num_tg = ((NSUInteger)out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ---------------------------------------------------------------------------
// Helper: dispatch fused RMSNorm + matvec_4bit
//
// Fuses input_layernorm + first projection into one kernel.
// Eliminates one dispatch + one barrier per layer.
// The fused kernel also writes the normed input to normed_out for
// subsequent projections to use.
// ---------------------------------------------------------------------------
static void dispatch_rmsnorm_matvec_4bit(
    BakanFullEngine* e,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> input,         // pre-norm hidden state
    const char* norm_wt_name,    // RMSNorm weight name
    const char* w_name, const char* s_name, const char* b_name,
    id<MTLBuffer> output,        // matvec output
    id<MTLBuffer> normed_out,    // normed input (for other projections)
    int out_dim, int in_dim, int group_size
) {
    id<MTLBuffer> nw = find_weight(e, norm_wt_name);
    id<MTLBuffer> w = find_weight(e, w_name);
    id<MTLBuffer> s = find_weight(e, s_name);
    id<MTLBuffer> b = find_weight(e, b_name);

    if (!nw || !w || !s || !b) {
        fprintf(stderr, "[bakan_full] ERROR: Weight not found for fused norm+mv: %s\n", w_name);
        return;
    }

    id<MTLBuffer> odBuf = get_dim_buf(e, out_dim);
    id<MTLBuffer> idBuf = get_dim_buf(e, in_dim);
    id<MTLBuffer> gsBuf = e->cbuf_group_size;
    id<MTLBuffer> epsBuf = e->cbuf_rms_norm_eps;

    [enc setComputePipelineState:e->rmsnormMatvec4bitPipeline];
    [enc setBuffer:input      offset:0 atIndex:0];
    [enc setBuffer:nw         offset:0 atIndex:1];
    [enc setBuffer:w          offset:0 atIndex:2];
    [enc setBuffer:s          offset:0 atIndex:3];
    [enc setBuffer:b          offset:0 atIndex:4];
    [enc setBuffer:output     offset:0 atIndex:5];
    [enc setBuffer:normed_out offset:0 atIndex:6];
    [enc setBuffer:odBuf      offset:0 atIndex:7];
    [enc setBuffer:idBuf      offset:0 atIndex:8];
    [enc setBuffer:gsBuf      offset:0 atIndex:9];
    [enc setBuffer:epsBuf     offset:0 atIndex:10];

    // SIMD dispatch: 8 rows per threadgroup (256 threads = 8 SIMD groups)
    NSUInteger num_tg = ((NSUInteger)out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(num_tg, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ---------------------------------------------------------------------------
// Helper: dispatch RMSNorm
// ---------------------------------------------------------------------------
static void dispatch_rmsnorm(
    BakanFullEngine* e,
    id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> input,
    const char* weight_name,
    id<MTLBuffer> output,
    int dim
) {
    id<MTLBuffer> w = find_weight(e, weight_name);
    if (!w) {
        fprintf(stderr, "[bakan_full] ERROR: Norm weight not found: %s\n", weight_name);
        return;
    }

    // Use pre-allocated constant buffers
    id<MTLBuffer> dimBuf = get_dim_buf(e, dim);
    id<MTLBuffer> epsBuf = e->cbuf_rms_norm_eps;

    [enc setComputePipelineState:e->rmsnormPipeline];
    [enc setBuffer:input   offset:0 atIndex:0];
    [enc setBuffer:w       offset:0 atIndex:1];
    [enc setBuffer:output  offset:0 atIndex:2];
    [enc setBuffer:dimBuf  offset:0 atIndex:3];
    [enc setBuffer:epsBuf  offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ---------------------------------------------------------------------------
// Forward declarations for debug functions (used before definition)
static int g_debug_forward;

static void debug_dump_half(const char* label, id<MTLBuffer> buf, int n) {
    if (!g_debug_forward) return;
    uint16_t* h = (uint16_t*)buf.contents;
    fprintf(stderr, "[debug] %s:", label);
    for (int i = 0; i < n && i < 8; i++) {
        fprintf(stderr, " %.4f", half_to_float(h[i]));
    }
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = half_to_float(h[i]);
        sum_sq += v * v;
    }
    fprintf(stderr, "  (L2=%.4f)\n", sqrtf(sum_sq));
}

static void debug_dump_float(const char* label, id<MTLBuffer> buf, int n) {
    if (!g_debug_forward) return;
    float* f = (float*)buf.contents;
    fprintf(stderr, "[debug] %s:", label);
    for (int i = 0; i < n && i < 8; i++) {
        fprintf(stderr, " %.6f", f[i]);
    }
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += f[i] * f[i];
    }
    fprintf(stderr, "  (L2=%.4f)\n", sqrtf(sum_sq));
}

void bakan_full_set_debug(void* engine, int enable) {
    (void)engine;
    g_debug_forward = enable;
    fprintf(stderr, "[bakan_full] Debug mode %s\n", enable ? "ON" : "OFF");
}

// ---------------------------------------------------------------------------
// bakan_full_forward — Optimized command buffer version
//
// Per-layer structure (2 command buffer submissions per layer):
//
// FULL ATTENTION LAYERS:
//   CB1: fused input_layernorm + combined Q+K+V projection (1 dispatch)
//        + split_q_gate (GPU) + Q/K norm + RoPE + KV cache + GQA
//        + gate mul + o_proj + attn residual add
//        + post_attention_layernorm + router matmul
//        + GPU softmax + top-K (router_softmax_topk kernel)
//   -> WAIT (CPU: read 8 expert indices (32 bytes), parallel pread via GCD)
//   CB2: fused expert MLP + shared expert + blend_residual
//   -> COMMIT (NO WAIT — Metal in-order queue guarantees completion)
//
// LINEAR ATTENTION LAYERS:
//   CB1: fused input_layernorm + combined QKV+Z+B+A projection (1 dispatch)
//        + conv1d_buf_update + conv1d + split_qkv (GPU) + Q/K norm + scale
//        + gated_delta_state + rmsnorm_gated + out_proj
//        + attn residual add + post_attention_layernorm + router matmul
//        + GPU softmax + top-K (router_softmax_topk kernel)
//   -> WAIT (CPU: read 8 expert indices (32 bytes), parallel pread via GCD)
//   CB2: fused expert MLP + shared expert + blend_residual
//   -> COMMIT (NO WAIT — Metal in-order queue guarantees completion)
//
// Four optimizations over previous version:
//   1. Combined projection weights: 3-4 matvec dispatches -> 1 per layer
//   2. GCD dispatch groups for parallel pread (8 experts simultaneously)
//   3. Fused expert MLP kernel (shared memory intermediate, 1 dispatch not 2)
//   4. Deferred MoE CB (no waitUntilCompleted — next CB implicitly waits)
//
// Total: 1 wait/layer * 40 layers + 1 (final norm+LM head) = 41 waits
// Down from 81 waits (2/layer). Each eliminated wait saves ~0.2ms.
// ---------------------------------------------------------------------------
int bakan_full_forward(void* engine, int token_id, int position,
                       float* logits_out, int* vocab_size_out) {
    @autoreleasepool {
        BakanFullEngine* e = (BakanFullEngine*)engine;
        if (!e || !e->built) {
            fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
            return -1;
        }

        // Update per-call mutable parameter buffers
        *(int32_t*)e->cbuf_position.contents = (int32_t)position;
        *(int32_t*)e->cbuf_kv_len.contents   = (int32_t)(position + 1);

        if (g_debug_forward) {
            fprintf(stderr, "\n[debug] === forward token_id=%d position=%d ===\n",
                    token_id, position);
        }

        // === Step 1: Embedding lookup (CPU — single row, trivially fast) ===
        {
            id<MTLBuffer> ew = find_weight(e, "embed_tokens.weight");
            id<MTLBuffer> es = find_weight(e, "embed_tokens.scales");
            id<MTLBuffer> eb = find_weight(e, "embed_tokens.biases");

            if (!ew || !es || !eb) {
                fprintf(stderr, "[bakan_full] ERROR: Embedding weights not found\n");
                return -1;
            }

            uint32_t* w_ptr = (uint32_t*)ew.contents;
            uint16_t* s_ptr = (uint16_t*)es.contents;
            uint16_t* b_ptr = (uint16_t*)eb.contents;
            uint16_t* out = (uint16_t*)e->hidden_buf.contents;

            int packed_per_row = HIDDEN_SIZE / 8;
            int groups_per_row = HIDDEN_SIZE / GROUP_SIZE;

            for (int i = 0; i < HIDDEN_SIZE; i++) {
                int group_idx = i / GROUP_SIZE;
                int pos_in_group = i % GROUP_SIZE;
                int packed_idx = pos_in_group / 8;
                int nibble_idx = pos_in_group % 8;

                float scale = bf16_to_float(s_ptr[token_id * groups_per_row + group_idx]);
                float bias  = bf16_to_float(b_ptr[token_id * groups_per_row + group_idx]);
                uint32_t packed = w_ptr[token_id * packed_per_row + group_idx * (GROUP_SIZE / 8) + packed_idx];
                float nibble = (float)((packed >> (nibble_idx * 4)) & 0xF);
                float val = nibble * scale + bias;
                out[i] = float_to_half(val);
            }
        }

        debug_dump_half("embed", e->hidden_buf, HIDDEN_SIZE);

        // === Step 2: Process each layer ===
        int attn_idx = 0;
        int lin_idx = 0;

        for (int layer = 0; layer < e->num_layers; layer++) {
            int is_attn = ((layer + 1) % 4 == 0);  // layers 3,7,11,...,39
            char wn[256], sn[256], bn[256];

            if (is_attn) {
                // =============================================================
                // FULL ATTENTION LAYER
                // =============================================================

                // --- CB1: input_layernorm + Q/K/V projections + GPU split_q_gate
                //          + Q/K norm + RoPE + KV cache + GQA + gate mul + o_proj
                //          + attn residual + post_norm + router matmul ---
                {
                    id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                    // Combined fused input_layernorm + Q+K+V projection:
                    // Single dispatch: RMSNorm(hidden) -> normed_buf
                    //   AND combined matmul -> combined_proj_buf(9216)
                    // Replaces 3 separate dispatches (rmsnorm+Q, K, V) with 1.
                    // Output layout: Q+gate[0..8191], K[8192..8703], V[8704..9215]
                    {
                        char nwn[256];
                        layer_weight_name(nwn, sizeof(nwn), layer, "input_layernorm.weight");
                        layer_weight_name(wn, sizeof(wn), layer, "self_attn.combined_proj.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "self_attn.combined_proj.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "self_attn.combined_proj.biases");
                        dispatch_rmsnorm_matvec_4bit(e, enc, e->hidden_buf, nwn,
                                                     wn, sn, bn,
                                                     e->combined_proj_buf, e->normed_buf,
                                                     ATTN_COMBINED_PROJ_OUT, HIDDEN_SIZE, GROUP_SIZE);
                    }

                    // Barrier: combined projection -> GPU split_q_gate + K/V copy
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GPU split: combined_proj_buf[0..8191] -> q_buf[4096] + gate_buf[4096]
                    // Layout: [head0_q(256), head0_gate(256), head1_q(256), ...]
                    [enc setComputePipelineState:e->splitQGatePipeline];
                    [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_Q_OFF * sizeof(uint16_t)) atIndex:0];
                    [enc setBuffer:e->q_buf              offset:0 atIndex:1];
                    [enc setBuffer:e->gate_buf           offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(ATTN_OUT_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: split -> Q/K norm
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Q norm (per-head RMSNorm with weight)
                    layer_weight_name(wn, sizeof(wn), layer, "self_attn.q_norm.weight");
                    id<MTLBuffer> qnw = find_weight(e, wn);

                    [enc setComputePipelineState:e->rmsnormPerHeadPipeline];
                    [enc setBuffer:e->q_buf              offset:0 atIndex:0];
                    [enc setBuffer:qnw                   offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(NUM_Q_HEADS, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    // K norm: reads/writes K at combined_proj_buf offset
                    layer_weight_name(wn, sizeof(wn), layer, "self_attn.k_norm.weight");
                    id<MTLBuffer> knw = find_weight(e, wn);

                    [enc setComputePipelineState:e->rmsnormPerHeadPipeline];
                    [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                    [enc setBuffer:knw                   offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_num_kv_heads  offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(NUM_KV_HEADS, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    // Barrier: Q/K norm -> RoPE
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // RoPE on Q (position is updated at top of forward pass)
                    [enc setComputePipelineState:e->ropePipeline];
                    [enc setBuffer:e->q_buf              offset:0 atIndex:0];
                    [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_rope_dim      offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_position      offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_rope_theta    offset:0 atIndex:5];
                    NSUInteger rope_q_threads = NUM_Q_HEADS * (ROPE_DIM / 2);
                    [enc dispatchThreads:MTLSizeMake(rope_q_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    // RoPE on K: reads/writes K at combined_proj_buf offset
                    [enc setComputePipelineState:e->ropePipeline];
                    [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                    [enc setBuffer:e->cbuf_num_kv_heads  offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_rope_dim      offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_position      offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_rope_theta    offset:0 atIndex:5];
                    NSUInteger rope_k_threads = NUM_KV_HEADS * (ROPE_DIM / 2);
                    [enc dispatchThreads:MTLSizeMake(rope_k_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    // Barrier: RoPE -> KV cache append
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // KV cache append: K and V from combined_proj_buf at offsets
                    [enc setComputePipelineState:e->kvCacheAppendPipeline];
                    [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                    [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_V_OFF * sizeof(uint16_t)) atIndex:1];
                    [enc setBuffer:buf_from_ref(e->k_caches[attn_idx])           offset:0 atIndex:2];
                    [enc setBuffer:buf_from_ref(e->v_caches[attn_idx])           offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_num_kv_heads                          offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_head_dim                              offset:0 atIndex:5];
                    [enc setBuffer:e->cbuf_position                              offset:0 atIndex:6];
                    [enc setBuffer:e->cbuf_max_seq_len                           offset:0 atIndex:7];
                    NSUInteger kv_threads = NUM_KV_HEADS * HEAD_DIM;
                    [enc dispatchThreads:MTLSizeMake(kv_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: KV cache -> GQA
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GQA attention decode (kv_len updated at top of forward pass)
                    [enc setComputePipelineState:e->gqaAttnPipeline];
                    [enc setBuffer:e->q_buf                                       offset:0 atIndex:0];
                    [enc setBuffer:buf_from_ref(e->k_caches[attn_idx])            offset:0 atIndex:1];
                    [enc setBuffer:buf_from_ref(e->v_caches[attn_idx])            offset:0 atIndex:2];
                    [enc setBuffer:e->attn_raw_buf                                offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_num_q_heads                            offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_num_kv_heads                           offset:0 atIndex:5];
                    [enc setBuffer:e->cbuf_head_dim                               offset:0 atIndex:6];
                    [enc setBuffer:e->cbuf_kv_len                                 offset:0 atIndex:7];
                    [enc setBuffer:e->cbuf_max_seq_len                            offset:0 atIndex:8];
                    [enc dispatchThreadgroups:MTLSizeMake(NUM_Q_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: GQA -> gate multiply
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Sigmoid gate multiply
                    [enc setComputePipelineState:e->sigmoidGateMulPipeline];
                    [enc setBuffer:e->attn_raw_buf       offset:0 atIndex:0];
                    [enc setBuffer:e->gate_buf           offset:0 atIndex:1];
                    [enc setBuffer:e->gated_attn_buf     offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_attn_out_dim  offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(ATTN_OUT_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: gate mul -> o_proj
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Output projection: gated_attn(4096) -> attn_out(2048)
                    layer_weight_name(wn, sizeof(wn), layer, "self_attn.o_proj.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "self_attn.o_proj.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "self_attn.o_proj.biases");
                    dispatch_matvec_4bit(e, enc, e->gated_attn_buf, wn, sn, bn,
                                         e->attn_out_buf, HIDDEN_SIZE, ATTN_OUT_DIM, GROUP_SIZE);

                    // Barrier: o_proj -> residual add
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Attention residual: hidden_buf2 = hidden_buf + attn_out_buf
                    [enc setComputePipelineState:e->residualAddPipeline];
                    [enc setBuffer:e->hidden_buf         offset:0 atIndex:0];
                    [enc setBuffer:e->attn_out_buf       offset:0 atIndex:1];
                    [enc setBuffer:e->hidden_buf2        offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_hidden_size   offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: residual -> post_attention_layernorm
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // post_attention_layernorm: hidden_buf2 -> normed_buf
                    layer_weight_name(wn, sizeof(wn), layer, "post_attention_layernorm.weight");
                    dispatch_rmsnorm(e, enc, e->hidden_buf2, wn, e->normed_buf, HIDDEN_SIZE);

                    // Barrier: post_norm -> router matmul
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Router matmul (GPU): normed -> router_logits(256 floats)
                    layer_weight_name(wn, sizeof(wn), layer, "mlp.gate.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "mlp.gate.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "mlp.gate.biases");
                    id<MTLBuffer> rw = find_weight(e, wn);
                    id<MTLBuffer> rs = find_weight(e, sn);
                    id<MTLBuffer> rb = find_weight(e, bn);

                    [enc setComputePipelineState:e->routerMatmulPipeline];
                    [enc setBuffer:e->normed_buf         offset:0 atIndex:0];
                    [enc setBuffer:rw                     offset:0 atIndex:1];
                    [enc setBuffer:rs                     offset:0 atIndex:2];
                    [enc setBuffer:rb                     offset:0 atIndex:3];
                    [enc setBuffer:e->router_logits_buf  offset:0 atIndex:4];
                    [enc setBuffer:e->moe_params_buf     offset:0 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: router matmul -> GPU softmax + top-K
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GPU softmax + top-K: logits(256) -> indices(8) + scores(8)
                    [enc setComputePipelineState:e->routerSoftmaxTopkPipeline];
                    [enc setBuffer:e->router_logits_buf  offset:0 atIndex:0];
                    [enc setBuffer:e->expert_indices_buf  offset:0 atIndex:1];
                    [enc setBuffer:e->scores_buffer       offset:0 atIndex:2];
                    [enc setBuffer:e->moe_params_buf      offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // hidden_buf2 now has the post-attention hidden state.
                // blend_residual in CB2 will read it from hidden_buf2 and write
                // the final result (h + moe_out) to hidden_buf — no memcpy needed.

                attn_idx++;

            } else {
                // =============================================================
                // LINEAR ATTENTION LAYER (GatedDeltaNet)
                // =============================================================

                // --- CB1: input_layernorm + projections + conv1d + GPU split_qkv
                //          + Q/K norm + scale + state update + gated_norm + out_proj
                //          + attn residual + post_norm + router matmul ---
                {
                    id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                    // Combined fused input_layernorm + QKV+Z+B+A projection:
                    // Single dispatch: RMSNorm(hidden) -> normed_buf
                    //   AND combined matmul -> combined_proj_buf(12352)
                    // Replaces 4 separate dispatches (rmsnorm+QKV, Z, B, A) with 1.
                    // Output layout: QKV[0..8191], Z[8192..12287], B[12288..12319], A[12320..12351]
                    {
                        char nwn[256];
                        layer_weight_name(nwn, sizeof(nwn), layer, "input_layernorm.weight");
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.combined_proj.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "linear_attn.combined_proj.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "linear_attn.combined_proj.biases");
                        dispatch_rmsnorm_matvec_4bit(e, enc, e->hidden_buf, nwn,
                                                     wn, sn, bn,
                                                     e->combined_proj_buf, e->normed_buf,
                                                     LIN_COMBINED_PROJ_OUT, HIDDEN_SIZE, GROUP_SIZE);
                    }

                    // Barrier: combined projection -> conv1d + downstream
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Conv1d buffer update: reads QKV from combined_proj_buf[0..8191]
                    [enc setComputePipelineState:e->convBufUpdatePipeline];
                    [enc setBuffer:buf_from_ref(e->conv_bufs[lin_idx]) offset:0 atIndex:0];
                    [enc setBuffer:e->combined_proj_buf                 offset:(LIN_COMBINED_QKV_OFF * sizeof(uint16_t)) atIndex:1];
                    [enc setBuffer:e->cbuf_lin_conv_dim                 offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_conv_kernel                  offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: conv buffer update -> conv1d compute
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Conv1d + SiLU
                    layer_weight_name(wn, sizeof(wn), layer, "linear_attn.conv1d.weight");
                    id<MTLBuffer> cw = find_weight(e, wn);

                    [enc setComputePipelineState:e->conv1dSiluPipeline];
                    [enc setBuffer:buf_from_ref(e->conv_bufs[lin_idx]) offset:0 atIndex:0];
                    [enc setBuffer:cw                                   offset:0 atIndex:1];
                    [enc setBuffer:e->lin_qkv_buf                      offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_lin_conv_dim                 offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_conv_kernel                  offset:0 atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: conv1d -> GPU split_qkv
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GPU split: lin_qkv_buf[8192] -> lin_q_buf[2048] + lin_k_buf[2048] + lin_v_buf[4096]
                    [enc setComputePipelineState:e->splitQkvPipeline];
                    [enc setBuffer:e->lin_qkv_buf          offset:0 atIndex:0];
                    [enc setBuffer:e->lin_q_buf            offset:0 atIndex:1];
                    [enc setBuffer:e->lin_k_buf            offset:0 atIndex:2];
                    [enc setBuffer:e->lin_v_buf            offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_lin_key_total   offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_lin_key_total   offset:0 atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: split -> Q/K norm
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // RMSNorm on Q (16 heads, 128 dim)
                    [enc setComputePipelineState:e->rmsnormPerHeadNoWtPipeline];
                    [enc setBuffer:e->lin_q_buf              offset:0 atIndex:0];
                    [enc setBuffer:e->cbuf_lin_num_k_heads   offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_lin_key_dim       offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_rms_norm_eps      offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(LIN_NUM_K_HEADS, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    // RMSNorm on K (16 heads, 128 dim)
                    [enc setComputePipelineState:e->rmsnormPerHeadNoWtPipeline];
                    [enc setBuffer:e->lin_k_buf              offset:0 atIndex:0];
                    [enc setBuffer:e->cbuf_lin_num_k_heads   offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_lin_key_dim       offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_rms_norm_eps      offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(LIN_NUM_K_HEADS, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    // Barrier: Q/K norm -> scale
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Scale Q by 1/128
                    [enc setComputePipelineState:e->scaleVectorPipeline];
                    [enc setBuffer:e->lin_q_buf          offset:0 atIndex:0];
                    [enc setBuffer:e->cbuf_q_scale       offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_lin_key_total offset:0 atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(LIN_KEY_TOTAL, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Scale K by 1/sqrt(128)
                    [enc setComputePipelineState:e->scaleVectorPipeline];
                    [enc setBuffer:e->lin_k_buf          offset:0 atIndex:0];
                    [enc setBuffer:e->cbuf_k_scale       offset:0 atIndex:1];
                    [enc setBuffer:e->cbuf_lin_key_total offset:0 atIndex:2];
                    [enc dispatchThreads:MTLSizeMake(LIN_KEY_TOTAL, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: scale -> gated delta state update
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GatedDeltaNet state update
                    layer_weight_name(wn, sizeof(wn), layer, "linear_attn.A_log");
                    id<MTLBuffer> alog = find_weight(e, wn);
                    layer_weight_name(wn, sizeof(wn), layer, "linear_attn.dt_bias");
                    id<MTLBuffer> dtb = find_weight(e, wn);

                    // A and B from combined_proj_buf at offsets
                    [enc setComputePipelineState:e->gatedDeltaStatePipeline];
                    [enc setBuffer:e->lin_q_buf                                    offset:0 atIndex:0];
                    [enc setBuffer:e->lin_k_buf                                    offset:0 atIndex:1];
                    [enc setBuffer:e->lin_v_buf                                    offset:0 atIndex:2];
                    [enc setBuffer:e->combined_proj_buf                            offset:(LIN_COMBINED_A_OFF * sizeof(uint16_t)) atIndex:3];
                    [enc setBuffer:e->combined_proj_buf                            offset:(LIN_COMBINED_B_OFF * sizeof(uint16_t)) atIndex:4];
                    [enc setBuffer:alog                                             offset:0 atIndex:5];
                    [enc setBuffer:dtb                                              offset:0 atIndex:6];
                    [enc setBuffer:buf_from_ref(e->lin_states[lin_idx])             offset:0 atIndex:7];
                    [enc setBuffer:e->lin_out_buf                                  offset:0 atIndex:8];
                    [enc setBuffer:e->cbuf_lin_num_k_heads                          offset:0 atIndex:9];
                    [enc setBuffer:e->cbuf_lin_num_v_heads                          offset:0 atIndex:10];
                    [enc setBuffer:e->cbuf_lin_key_dim                              offset:0 atIndex:11];
                    [enc setBuffer:e->cbuf_lin_val_dim                              offset:0 atIndex:12];
                    [enc dispatchThreads:MTLSizeMake(LIN_NUM_V_HEADS, LIN_VAL_DIM, 1)
                       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                    // Barrier: state update -> gated RMSNorm
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // RMSNormGated: silu(z) * rms_norm(out, weight)
                    layer_weight_name(wn, sizeof(wn), layer, "linear_attn.norm.weight");
                    id<MTLBuffer> nw = find_weight(e, wn);

                    // Z from combined_proj_buf at offset
                    [enc setComputePipelineState:e->rmsnormGatedPipeline];
                    [enc setBuffer:e->lin_out_buf        offset:0 atIndex:0];
                    [enc setBuffer:e->combined_proj_buf  offset:(LIN_COMBINED_Z_OFF * sizeof(uint16_t)) atIndex:1];
                    [enc setBuffer:nw                     offset:0 atIndex:2];
                    [enc setBuffer:e->lin_normed_buf     offset:0 atIndex:3];
                    [enc setBuffer:e->cbuf_lin_num_v_heads offset:0 atIndex:4];
                    [enc setBuffer:e->cbuf_lin_val_dim   offset:0 atIndex:5];
                    [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:6];
                    [enc dispatchThreadgroups:MTLSizeMake(LIN_NUM_V_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    // Barrier: gated norm -> output projection
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Output projection: normed(4096) -> attn_out(2048)
                    layer_weight_name(wn, sizeof(wn), layer, "linear_attn.out_proj.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "linear_attn.out_proj.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "linear_attn.out_proj.biases");
                    dispatch_matvec_4bit(e, enc, e->lin_normed_buf, wn, sn, bn,
                                         e->attn_out_buf, HIDDEN_SIZE, LIN_VAL_TOTAL, GROUP_SIZE);

                    // Barrier: o_proj -> residual add
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Attention residual: hidden_buf2 = hidden_buf + attn_out_buf
                    [enc setComputePipelineState:e->residualAddPipeline];
                    [enc setBuffer:e->hidden_buf         offset:0 atIndex:0];
                    [enc setBuffer:e->attn_out_buf       offset:0 atIndex:1];
                    [enc setBuffer:e->hidden_buf2        offset:0 atIndex:2];
                    [enc setBuffer:e->cbuf_hidden_size   offset:0 atIndex:3];
                    [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: residual -> post_attention_layernorm
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // post_attention_layernorm: hidden_buf2 -> normed_buf
                    layer_weight_name(wn, sizeof(wn), layer, "post_attention_layernorm.weight");
                    dispatch_rmsnorm(e, enc, e->hidden_buf2, wn, e->normed_buf, HIDDEN_SIZE);

                    // Barrier: post_norm -> router matmul
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Router matmul (GPU): normed -> router_logits(256 floats)
                    layer_weight_name(wn, sizeof(wn), layer, "mlp.gate.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "mlp.gate.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "mlp.gate.biases");
                    id<MTLBuffer> rw = find_weight(e, wn);
                    id<MTLBuffer> rs = find_weight(e, sn);
                    id<MTLBuffer> rb = find_weight(e, bn);

                    [enc setComputePipelineState:e->routerMatmulPipeline];
                    [enc setBuffer:e->normed_buf         offset:0 atIndex:0];
                    [enc setBuffer:rw                     offset:0 atIndex:1];
                    [enc setBuffer:rs                     offset:0 atIndex:2];
                    [enc setBuffer:rb                     offset:0 atIndex:3];
                    [enc setBuffer:e->router_logits_buf  offset:0 atIndex:4];
                    [enc setBuffer:e->moe_params_buf     offset:0 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    // Barrier: router matmul -> GPU softmax + top-K
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // GPU softmax + top-K: logits(256) -> indices(8) + scores(8)
                    [enc setComputePipelineState:e->routerSoftmaxTopkPipeline];
                    [enc setBuffer:e->router_logits_buf  offset:0 atIndex:0];
                    [enc setBuffer:e->expert_indices_buf  offset:0 atIndex:1];
                    [enc setBuffer:e->scores_buffer       offset:0 atIndex:2];
                    [enc setBuffer:e->moe_params_buf      offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // hidden_buf2 now has the post-attention hidden state.
                // blend_residual in CB2 will read it from hidden_buf2 and write
                // the final result (h + moe_out) to hidden_buf — no memcpy needed.

                lin_idx++;
            }

            // =============================================================
            // COMMON: Read GPU top-k results + pread experts, then MoE CB2
            // =============================================================

            // --- CPU: Read expert indices from GPU buffer (32 bytes) ---
            // Softmax + top-K already computed on GPU in router_softmax_topk kernel.
            // Scores are already in scores_buffer on GPU (used directly by blend kernel).
            int K = NUM_EXPERTS_PER_TOK;
            int32_t* top_indices = (int32_t*)e->expert_indices_buf.contents;

            // --- CPU: Parallel pread experts using GCD dispatch group ---
            // Reads 8 experts simultaneously. OS page cache serves cached experts
            // instantly; SSD reads overlap for cold experts.
            int fd = e->layer_fds[layer];
            if (fd < 0) {
                fprintf(stderr, "[bakan_full] ERROR: No expert file for layer %d\n", layer);
                return -1;
            }
            char* staging = (char*)e->staging_buffer.contents;
            {
                dispatch_group_t group = dispatch_group_create();
                dispatch_queue_t io_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
                __block int pread_error = 0;
                for (int k = 0; k < K; k++) {
                    int expert_idx = top_indices[k];
                    size_t off = (size_t)HEADER_SIZE + (size_t)expert_idx * EXPERT_SIZE;
                    char* dst = staging + (size_t)k * EXPERT_SIZE;
                    dispatch_group_async(group, io_queue, ^{
                        ssize_t n = pread(fd, dst, EXPERT_SIZE, (off_t)off);
                        if (n != (ssize_t)EXPERT_SIZE) {
                            pread_error = 1;
                        }
                    });
                }
                dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                if (pread_error) {
                    fprintf(stderr, "[bakan_full] ERROR: pread layer %d failed\n", layer);
                    return -1;
                }
            }

            // --- CB2: Fused Expert MLP + Shared expert + Blend with residual ---
            // Optimization: single fused kernel replaces 2 dispatches + 1 barrier.
            // Intermediate activated values stay in threadgroup shared memory.
            // Deferred commit: no waitUntilCompleted — Metal in-order queue
            // guarantees this CB finishes before next layer's attention CB starts.
            {
                ExpertMLPParams* ep = (ExpertMLPParams*)e->expert_params_buf.contents;
                for (int k = 0; k < K; k++) {
                    ep->expert_indices[k] = k;
                }
                ep->num_experts = K;
                ep->header_size = 0;

                id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                // Fused expert MLP: up_proj + gate_proj + SwiGLU + down_proj
                // in one kernel with shared memory intermediate (no device write/read)
                [enc setComputePipelineState:e->expertMLPFusedPipeline];
                [enc setBuffer:e->staging_buffer     offset:0 atIndex:0];
                [enc setBuffer:e->normed_buf          offset:0 atIndex:1];
                [enc setBuffer:e->expert_output      offset:0 atIndex:2];
                [enc setBuffer:e->expert_params_buf  offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)K, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(EXPERT_DIM, 1, 1)];

                // Shared expert up + gate + SwiGLU
                layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert.gate_proj.weight");
                layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert.gate_proj.scales");
                layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert.gate_proj.biases");
                char wn2[256], sn2[256], bn2[256];
                layer_weight_name(wn2, sizeof(wn2), layer, "mlp.shared_expert.up_proj.weight");
                layer_weight_name(sn2, sizeof(sn2), layer, "mlp.shared_expert.up_proj.scales");
                layer_weight_name(bn2, sizeof(bn2), layer, "mlp.shared_expert.up_proj.biases");

                [enc setComputePipelineState:e->sharedUpGatePipeline];
                [enc setBuffer:e->normed_buf                         offset:0 atIndex:0];
                [enc setBuffer:find_weight(e, wn)                    offset:0 atIndex:1];
                [enc setBuffer:find_weight(e, sn)                    offset:0 atIndex:2];
                [enc setBuffer:find_weight(e, bn)                    offset:0 atIndex:3];
                [enc setBuffer:find_weight(e, wn2)                   offset:0 atIndex:4];
                [enc setBuffer:find_weight(e, sn2)                   offset:0 atIndex:5];
                [enc setBuffer:find_weight(e, bn2)                   offset:0 atIndex:6];
                [enc setBuffer:e->shared_activated                   offset:0 atIndex:7];
                [enc setBuffer:e->moe_params_buf                     offset:0 atIndex:8];
                [enc dispatchThreads:MTLSizeMake(EXPERT_DIM, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                // Barrier: fused expert + shared up+gate -> shared down_proj + shared gate
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Shared expert down_proj
                layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert.down_proj.weight");
                layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert.down_proj.scales");
                layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert.down_proj.biases");

                [enc setComputePipelineState:e->sharedDownPipeline];
                [enc setBuffer:e->shared_activated               offset:0 atIndex:0];
                [enc setBuffer:find_weight(e, wn)                offset:0 atIndex:1];
                [enc setBuffer:find_weight(e, sn)                offset:0 atIndex:2];
                [enc setBuffer:find_weight(e, bn)                offset:0 atIndex:3];
                [enc setBuffer:e->shared_out_buf                 offset:0 atIndex:4];
                [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                // Shared expert gate
                layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert_gate.weight");
                layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert_gate.scales");
                layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert_gate.biases");

                [enc setComputePipelineState:e->sharedGatePipeline];
                [enc setBuffer:e->normed_buf                     offset:0 atIndex:0];
                [enc setBuffer:find_weight(e, wn)                offset:0 atIndex:1];
                [enc setBuffer:find_weight(e, sn)                offset:0 atIndex:2];
                [enc setBuffer:find_weight(e, bn)                offset:0 atIndex:3];
                [enc setBuffer:e->shared_gate_buf                offset:0 atIndex:4];
                [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                // Barrier: expert down + shared down + shared gate -> blend_residual
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                // Blend with residual: output = hidden_buf2 + experts + shared
                // Reads post-attention state from hidden_buf2, writes to hidden_buf
                [enc setComputePipelineState:e->blendResidualPipeline];
                [enc setBuffer:e->hidden_buf2                    offset:0 atIndex:0];
                [enc setBuffer:e->expert_output                  offset:0 atIndex:1];
                [enc setBuffer:e->scores_buffer                  offset:0 atIndex:2];
                [enc setBuffer:e->shared_out_buf                 offset:0 atIndex:3];
                [enc setBuffer:e->shared_gate_buf                offset:0 atIndex:4];
                [enc setBuffer:e->hidden_buf                     offset:0 atIndex:5];
                [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc endEncoding];
                [cmd commit];
                // Deferred MoE CB: NO waitUntilCompleted here.
                // Metal command buffers on the same queue execute in order,
                // so the next layer's attention CB will implicitly wait for
                // this MoE CB to finish before reading hidden_buf.
                // This eliminates 40 GPU stalls (one per layer), saving ~8ms/token.

                // Debug mode: must wait to read hidden_buf on CPU
                if (g_debug_forward && layer < 5) {
                    [cmd waitUntilCompleted];
                    if (cmd.status == MTLCommandBufferStatusError) {
                        fprintf(stderr, "[bakan_full] ERROR: MoE GPU failed layer %d: %s\n",
                                layer, cmd.error.localizedDescription.UTF8String);
                        return -1;
                    }
                    char lbl[64];
                    snprintf(lbl, sizeof(lbl), "layer %d out", layer);
                    debug_dump_half(lbl, e->hidden_buf, HIDDEN_SIZE);
                }
            }

            // hidden_buf now contains h + moe_out (from blend_residual_kernel)
            // (written by GPU — available when next CB on same queue starts)
        }

        // Debug: wait for last deferred MoE CB before reading hidden_buf on CPU
        if (g_debug_forward) {
            // Submit an empty CB and wait — flushes all prior deferred work
            id<MTLCommandBuffer> syncCmd = [e->queue commandBuffer];
            [syncCmd commit];
            [syncCmd waitUntilCompleted];
        }
        debug_dump_half("final hidden", e->hidden_buf, HIDDEN_SIZE);

        // === Step 3: Final norm + LM head (single command buffer) ===
        {
            id<MTLBuffer> lhw = find_weight(e, "lm_head.weight");
            id<MTLBuffer> lhs = find_weight(e, "lm_head.scales");
            id<MTLBuffer> lhb = find_weight(e, "lm_head.biases");

            if (!lhw || !lhs || !lhb) {
                fprintf(stderr, "[bakan_full] ERROR: lm_head weights not found\n");
                return -1;
            }

            id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // Final RMSNorm
            dispatch_rmsnorm(e, enc, e->hidden_buf, "norm.weight",
                             e->normed_buf, HIDDEN_SIZE);

            // Barrier: norm -> LM head
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // LM head matmul: normed(2048) -> logits(248320)
            // Multi-SIMD dispatch: 8 rows per threadgroup (256 threads)
            [enc setComputePipelineState:e->matvec4bitF32Pipeline];
            [enc setBuffer:e->normed_buf       offset:0 atIndex:0];
            [enc setBuffer:lhw                  offset:0 atIndex:1];
            [enc setBuffer:lhs                  offset:0 atIndex:2];
            [enc setBuffer:lhb                  offset:0 atIndex:3];
            [enc setBuffer:e->logits_buf       offset:0 atIndex:4];
            [enc setBuffer:e->cbuf_vocab_size  offset:0 atIndex:5];
            [enc setBuffer:e->cbuf_hidden_size offset:0 atIndex:6];
            [enc setBuffer:e->cbuf_group_size  offset:0 atIndex:7];

            [enc dispatchThreadgroups:MTLSizeMake((VOCAB_SIZE + 7) / 8, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[bakan_full] ERROR: LM head GPU failed: %s\n",
                        cmd.error.localizedDescription.UTF8String);
                return -1;
            }
        }

        // Copy logits out
        memcpy(logits_out, e->logits_buf.contents, VOCAB_SIZE * sizeof(float));
        if (vocab_size_out) *vocab_size_out = VOCAB_SIZE;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// bakan_full_prefill — Batch prefill: all prompt tokens in one call
//
// Processes all N prompt tokens through the model, building up KV cache
// and linear attention state. Returns logits for the LAST token only.
//
// Optimizations over calling bakan_full_forward() N times from Python:
//   1. Skip LM head (248320-dim matmul) for N-1 non-last tokens (~2ms/token)
//   2. Eliminate Python ctypes overhead, logits alloc, numpy per token
//   3. Bulk CPU embedding for all tokens
//   4. Single C function call (no Python loop overhead)
//   5. Layer-major ordering: process all tokens through layer L before L+1
//      (better for future batched matmul kernels)
//
// Per-token GPU compute is the dominant cost (~9ms/token/layer * 40 layers).
// Expert pread from disk requires a GPU wait between attention and MoE for
// each token, preventing further CB batching.
//
// Structure: layer-major loop. For each layer, process tokens sequentially
// (attention state is recurrent), with 2 CBs per token (attention+router,
// then MoE after CPU pread). Final norm + LM head only for last token.
// ---------------------------------------------------------------------------
#define MAX_PREFILL 1024

int bakan_full_prefill(void* engine, const int32_t* token_ids, int num_tokens,
                       float* logits_out, int* vocab_size_out) {
    @autoreleasepool {
        BakanFullEngine* e = (BakanFullEngine*)engine;
        if (!e || !e->built) {
            fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
            return -1;
        }

        if (num_tokens <= 0) {
            fprintf(stderr, "[bakan_full] ERROR: num_tokens must be > 0\n");
            return -1;
        }
        if (num_tokens > MAX_PREFILL) {
            fprintf(stderr, "[bakan_full] ERROR: num_tokens %d exceeds MAX_PREFILL %d\n",
                    num_tokens, MAX_PREFILL);
            return -1;
        }

        // For a single token, just delegate to the existing optimized path
        if (num_tokens == 1) {
            return bakan_full_forward(engine, token_ids[0], 0, logits_out, vocab_size_out);
        }

        // === Allocate per-token hidden state buffer ===
        // Holds all N hidden states simultaneously. Each layer reads from
        // here, processes attention+MoE, and writes back.
        size_t per_token_bytes = HIDDEN_SIZE * sizeof(uint16_t);
        size_t total_hidden_bytes = (size_t)num_tokens * per_token_bytes;
        id<MTLBuffer> prefill_hidden = [e->device newBufferWithLength:total_hidden_bytes
                                                              options:MTLResourceStorageModeShared];
        if (!prefill_hidden) {
            fprintf(stderr, "[bakan_full] ERROR: Failed to allocate prefill hidden buffer (%zu bytes)\n",
                    total_hidden_bytes);
            return -1;
        }

        // === Step 1: Embed all tokens (CPU — trivially fast) ===
        {
            id<MTLBuffer> ew = find_weight(e, "embed_tokens.weight");
            id<MTLBuffer> es = find_weight(e, "embed_tokens.scales");
            id<MTLBuffer> eb = find_weight(e, "embed_tokens.biases");

            if (!ew || !es || !eb) {
                fprintf(stderr, "[bakan_full] ERROR: Embedding weights not found\n");
                return -1;
            }

            uint32_t* w_ptr = (uint32_t*)ew.contents;
            uint16_t* s_ptr = (uint16_t*)es.contents;
            uint16_t* b_ptr = (uint16_t*)eb.contents;
            uint16_t* all_hidden = (uint16_t*)prefill_hidden.contents;

            int packed_per_row = HIDDEN_SIZE / 8;
            int groups_per_row = HIDDEN_SIZE / GROUP_SIZE;

            for (int t = 0; t < num_tokens; t++) {
                int tid = token_ids[t];
                uint16_t* out = all_hidden + (size_t)t * HIDDEN_SIZE;
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    int group_idx = i / GROUP_SIZE;
                    int pos_in_group = i % GROUP_SIZE;
                    int packed_idx = pos_in_group / 8;
                    int nibble_idx = pos_in_group % 8;

                    float scale = bf16_to_float(s_ptr[tid * groups_per_row + group_idx]);
                    float bias  = bf16_to_float(b_ptr[tid * groups_per_row + group_idx]);
                    uint32_t packed = w_ptr[tid * packed_per_row + group_idx * (GROUP_SIZE / 8) + packed_idx];
                    float nibble = (float)((packed >> (nibble_idx * 4)) & 0xF);
                    float val = nibble * scale + bias;
                    out[i] = float_to_half(val);
                }
            }
        }

        // === Step 2: Process each layer (layer-major loop) ===
        int attn_idx = 0;
        int lin_idx = 0;

        for (int layer = 0; layer < e->num_layers; layer++) {
            int is_attn = ((layer + 1) % 4 == 0);  // layers 3,7,11,...,39
            char wn[256], sn[256], bn[256];

            // Process each token through this layer sequentially
            for (int t = 0; t < num_tokens; t++) {
                int position = t;
                size_t h_off = (size_t)t * HIDDEN_SIZE * sizeof(uint16_t);

                // Update per-call mutable parameter buffers
                *(int32_t*)e->cbuf_position.contents = (int32_t)position;
                *(int32_t*)e->cbuf_kv_len.contents   = (int32_t)(position + 1);

                // Load this token's hidden state into working buffer.
                // The previous MoE CB was waited on (not deferred) so hidden_buf
                // is available for CPU access.
                memcpy(e->hidden_buf.contents, (char*)prefill_hidden.contents + h_off,
                       per_token_bytes);

                if (is_attn) {
                    // =========================================================
                    // FULL ATTENTION LAYER
                    // =========================================================

                    // --- CB1: attention + router + topk ---
                    {
                        id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                        // Fused input_layernorm + combined Q+K+V projection
                        {
                            char nwn[256];
                            layer_weight_name(nwn, sizeof(nwn), layer, "input_layernorm.weight");
                            layer_weight_name(wn, sizeof(wn), layer, "self_attn.combined_proj.weight");
                            layer_weight_name(sn, sizeof(sn), layer, "self_attn.combined_proj.scales");
                            layer_weight_name(bn, sizeof(bn), layer, "self_attn.combined_proj.biases");
                            dispatch_rmsnorm_matvec_4bit(e, enc, e->hidden_buf, nwn,
                                                         wn, sn, bn,
                                                         e->combined_proj_buf, e->normed_buf,
                                                         ATTN_COMBINED_PROJ_OUT, HIDDEN_SIZE, GROUP_SIZE);
                        }

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GPU split: combined_proj_buf -> q_buf + gate_buf
                        [enc setComputePipelineState:e->splitQGatePipeline];
                        [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_Q_OFF * sizeof(uint16_t)) atIndex:0];
                        [enc setBuffer:e->q_buf              offset:0 atIndex:1];
                        [enc setBuffer:e->gate_buf           offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:4];
                        [enc dispatchThreads:MTLSizeMake(ATTN_OUT_DIM, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Q norm
                        layer_weight_name(wn, sizeof(wn), layer, "self_attn.q_norm.weight");
                        id<MTLBuffer> qnw = find_weight(e, wn);
                        [enc setComputePipelineState:e->rmsnormPerHeadPipeline];
                        [enc setBuffer:e->q_buf              offset:0 atIndex:0];
                        [enc setBuffer:qnw                   offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:4];
                        [enc dispatchThreads:MTLSizeMake(NUM_Q_HEADS, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        // K norm
                        layer_weight_name(wn, sizeof(wn), layer, "self_attn.k_norm.weight");
                        id<MTLBuffer> knw = find_weight(e, wn);
                        [enc setComputePipelineState:e->rmsnormPerHeadPipeline];
                        [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                        [enc setBuffer:knw                   offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_num_kv_heads  offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:4];
                        [enc dispatchThreads:MTLSizeMake(NUM_KV_HEADS, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // RoPE on Q
                        [enc setComputePipelineState:e->ropePipeline];
                        [enc setBuffer:e->q_buf              offset:0 atIndex:0];
                        [enc setBuffer:e->cbuf_num_q_heads   offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_rope_dim      offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_position      offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_rope_theta    offset:0 atIndex:5];
                        NSUInteger rope_q_threads = NUM_Q_HEADS * (ROPE_DIM / 2);
                        [enc dispatchThreads:MTLSizeMake(rope_q_threads, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                        // RoPE on K
                        [enc setComputePipelineState:e->ropePipeline];
                        [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                        [enc setBuffer:e->cbuf_num_kv_heads  offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_rope_dim      offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_position      offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_rope_theta    offset:0 atIndex:5];
                        NSUInteger rope_k_threads = NUM_KV_HEADS * (ROPE_DIM / 2);
                        [enc dispatchThreads:MTLSizeMake(rope_k_threads, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // KV cache append
                        [enc setComputePipelineState:e->kvCacheAppendPipeline];
                        [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_K_OFF * sizeof(uint16_t)) atIndex:0];
                        [enc setBuffer:e->combined_proj_buf  offset:(ATTN_COMBINED_V_OFF * sizeof(uint16_t)) atIndex:1];
                        [enc setBuffer:buf_from_ref(e->k_caches[attn_idx]) offset:0 atIndex:2];
                        [enc setBuffer:buf_from_ref(e->v_caches[attn_idx]) offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_num_kv_heads  offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_head_dim      offset:0 atIndex:5];
                        [enc setBuffer:e->cbuf_position      offset:0 atIndex:6];
                        [enc setBuffer:e->cbuf_max_seq_len   offset:0 atIndex:7];
                        NSUInteger kv_threads = NUM_KV_HEADS * HEAD_DIM;
                        [enc dispatchThreads:MTLSizeMake(kv_threads, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GQA attention decode
                        [enc setComputePipelineState:e->gqaAttnPipeline];
                        [enc setBuffer:e->q_buf                            offset:0 atIndex:0];
                        [enc setBuffer:buf_from_ref(e->k_caches[attn_idx]) offset:0 atIndex:1];
                        [enc setBuffer:buf_from_ref(e->v_caches[attn_idx]) offset:0 atIndex:2];
                        [enc setBuffer:e->attn_raw_buf                     offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_num_q_heads                 offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_num_kv_heads                offset:0 atIndex:5];
                        [enc setBuffer:e->cbuf_head_dim                    offset:0 atIndex:6];
                        [enc setBuffer:e->cbuf_kv_len                      offset:0 atIndex:7];
                        [enc setBuffer:e->cbuf_max_seq_len                 offset:0 atIndex:8];
                        [enc dispatchThreadgroups:MTLSizeMake(NUM_Q_HEADS, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Sigmoid gate multiply
                        [enc setComputePipelineState:e->sigmoidGateMulPipeline];
                        [enc setBuffer:e->attn_raw_buf       offset:0 atIndex:0];
                        [enc setBuffer:e->gate_buf           offset:0 atIndex:1];
                        [enc setBuffer:e->gated_attn_buf     offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_attn_out_dim  offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(ATTN_OUT_DIM, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Output projection
                        layer_weight_name(wn, sizeof(wn), layer, "self_attn.o_proj.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "self_attn.o_proj.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "self_attn.o_proj.biases");
                        dispatch_matvec_4bit(e, enc, e->gated_attn_buf, wn, sn, bn,
                                             e->attn_out_buf, HIDDEN_SIZE, ATTN_OUT_DIM, GROUP_SIZE);

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Attention residual
                        [enc setComputePipelineState:e->residualAddPipeline];
                        [enc setBuffer:e->hidden_buf         offset:0 atIndex:0];
                        [enc setBuffer:e->attn_out_buf       offset:0 atIndex:1];
                        [enc setBuffer:e->hidden_buf2        offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_hidden_size   offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // post_attention_layernorm
                        layer_weight_name(wn, sizeof(wn), layer, "post_attention_layernorm.weight");
                        dispatch_rmsnorm(e, enc, e->hidden_buf2, wn, e->normed_buf, HIDDEN_SIZE);

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Router matmul
                        layer_weight_name(wn, sizeof(wn), layer, "mlp.gate.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "mlp.gate.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "mlp.gate.biases");
                        id<MTLBuffer> rw = find_weight(e, wn);
                        id<MTLBuffer> rs = find_weight(e, sn);
                        id<MTLBuffer> rb = find_weight(e, bn);
                        [enc setComputePipelineState:e->routerMatmulPipeline];
                        [enc setBuffer:e->normed_buf         offset:0 atIndex:0];
                        [enc setBuffer:rw                     offset:0 atIndex:1];
                        [enc setBuffer:rs                     offset:0 atIndex:2];
                        [enc setBuffer:rb                     offset:0 atIndex:3];
                        [enc setBuffer:e->router_logits_buf  offset:0 atIndex:4];
                        [enc setBuffer:e->moe_params_buf     offset:0 atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GPU softmax + top-K
                        [enc setComputePipelineState:e->routerSoftmaxTopkPipeline];
                        [enc setBuffer:e->router_logits_buf   offset:0 atIndex:0];
                        [enc setBuffer:e->expert_indices_buf  offset:0 atIndex:1];
                        [enc setBuffer:e->scores_buffer       offset:0 atIndex:2];
                        [enc setBuffer:e->moe_params_buf      offset:0 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                    }

                } else {
                    // =========================================================
                    // LINEAR ATTENTION LAYER (GatedDeltaNet)
                    // =========================================================

                    // --- CB1: attention + router + topk ---
                    {
                        id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                        // Fused input_layernorm + combined QKV+Z+B+A projection
                        {
                            char nwn[256];
                            layer_weight_name(nwn, sizeof(nwn), layer, "input_layernorm.weight");
                            layer_weight_name(wn, sizeof(wn), layer, "linear_attn.combined_proj.weight");
                            layer_weight_name(sn, sizeof(sn), layer, "linear_attn.combined_proj.scales");
                            layer_weight_name(bn, sizeof(bn), layer, "linear_attn.combined_proj.biases");
                            dispatch_rmsnorm_matvec_4bit(e, enc, e->hidden_buf, nwn,
                                                         wn, sn, bn,
                                                         e->combined_proj_buf, e->normed_buf,
                                                         LIN_COMBINED_PROJ_OUT, HIDDEN_SIZE, GROUP_SIZE);
                        }

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Conv1d buffer update
                        [enc setComputePipelineState:e->convBufUpdatePipeline];
                        [enc setBuffer:buf_from_ref(e->conv_bufs[lin_idx]) offset:0 atIndex:0];
                        [enc setBuffer:e->combined_proj_buf  offset:(LIN_COMBINED_QKV_OFF * sizeof(uint16_t)) atIndex:1];
                        [enc setBuffer:e->cbuf_lin_conv_dim  offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_conv_kernel   offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Conv1d + SiLU
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.conv1d.weight");
                        id<MTLBuffer> cw = find_weight(e, wn);
                        [enc setComputePipelineState:e->conv1dSiluPipeline];
                        [enc setBuffer:buf_from_ref(e->conv_bufs[lin_idx]) offset:0 atIndex:0];
                        [enc setBuffer:cw                    offset:0 atIndex:1];
                        [enc setBuffer:e->lin_qkv_buf        offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_lin_conv_dim  offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_conv_kernel   offset:0 atIndex:4];
                        [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GPU split: lin_qkv_buf -> q, k, v
                        [enc setComputePipelineState:e->splitQkvPipeline];
                        [enc setBuffer:e->lin_qkv_buf          offset:0 atIndex:0];
                        [enc setBuffer:e->lin_q_buf            offset:0 atIndex:1];
                        [enc setBuffer:e->lin_k_buf            offset:0 atIndex:2];
                        [enc setBuffer:e->lin_v_buf            offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_lin_key_total   offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_lin_key_total   offset:0 atIndex:5];
                        [enc dispatchThreads:MTLSizeMake(LIN_CONV_DIM, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // RMSNorm on Q
                        [enc setComputePipelineState:e->rmsnormPerHeadNoWtPipeline];
                        [enc setBuffer:e->lin_q_buf              offset:0 atIndex:0];
                        [enc setBuffer:e->cbuf_lin_num_k_heads   offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_lin_key_dim       offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_rms_norm_eps      offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(LIN_NUM_K_HEADS, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        // RMSNorm on K
                        [enc setComputePipelineState:e->rmsnormPerHeadNoWtPipeline];
                        [enc setBuffer:e->lin_k_buf              offset:0 atIndex:0];
                        [enc setBuffer:e->cbuf_lin_num_k_heads   offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_lin_key_dim       offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_rms_norm_eps      offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(LIN_NUM_K_HEADS, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Scale Q
                        [enc setComputePipelineState:e->scaleVectorPipeline];
                        [enc setBuffer:e->lin_q_buf          offset:0 atIndex:0];
                        [enc setBuffer:e->cbuf_q_scale       offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_lin_key_total offset:0 atIndex:2];
                        [enc dispatchThreads:MTLSizeMake(LIN_KEY_TOTAL, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        // Scale K
                        [enc setComputePipelineState:e->scaleVectorPipeline];
                        [enc setBuffer:e->lin_k_buf          offset:0 atIndex:0];
                        [enc setBuffer:e->cbuf_k_scale       offset:0 atIndex:1];
                        [enc setBuffer:e->cbuf_lin_key_total offset:0 atIndex:2];
                        [enc dispatchThreads:MTLSizeMake(LIN_KEY_TOTAL, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GatedDeltaNet state update
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.A_log");
                        id<MTLBuffer> alog = find_weight(e, wn);
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.dt_bias");
                        id<MTLBuffer> dtb = find_weight(e, wn);

                        [enc setComputePipelineState:e->gatedDeltaStatePipeline];
                        [enc setBuffer:e->lin_q_buf              offset:0 atIndex:0];
                        [enc setBuffer:e->lin_k_buf              offset:0 atIndex:1];
                        [enc setBuffer:e->lin_v_buf              offset:0 atIndex:2];
                        [enc setBuffer:e->combined_proj_buf      offset:(LIN_COMBINED_A_OFF * sizeof(uint16_t)) atIndex:3];
                        [enc setBuffer:e->combined_proj_buf      offset:(LIN_COMBINED_B_OFF * sizeof(uint16_t)) atIndex:4];
                        [enc setBuffer:alog                       offset:0 atIndex:5];
                        [enc setBuffer:dtb                        offset:0 atIndex:6];
                        [enc setBuffer:buf_from_ref(e->lin_states[lin_idx]) offset:0 atIndex:7];
                        [enc setBuffer:e->lin_out_buf            offset:0 atIndex:8];
                        [enc setBuffer:e->cbuf_lin_num_k_heads    offset:0 atIndex:9];
                        [enc setBuffer:e->cbuf_lin_num_v_heads    offset:0 atIndex:10];
                        [enc setBuffer:e->cbuf_lin_key_dim        offset:0 atIndex:11];
                        [enc setBuffer:e->cbuf_lin_val_dim        offset:0 atIndex:12];
                        [enc dispatchThreads:MTLSizeMake(LIN_NUM_V_HEADS, LIN_VAL_DIM, 1)
                           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // RMSNormGated
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.norm.weight");
                        id<MTLBuffer> nw = find_weight(e, wn);
                        [enc setComputePipelineState:e->rmsnormGatedPipeline];
                        [enc setBuffer:e->lin_out_buf        offset:0 atIndex:0];
                        [enc setBuffer:e->combined_proj_buf  offset:(LIN_COMBINED_Z_OFF * sizeof(uint16_t)) atIndex:1];
                        [enc setBuffer:nw                     offset:0 atIndex:2];
                        [enc setBuffer:e->lin_normed_buf     offset:0 atIndex:3];
                        [enc setBuffer:e->cbuf_lin_num_v_heads offset:0 atIndex:4];
                        [enc setBuffer:e->cbuf_lin_val_dim   offset:0 atIndex:5];
                        [enc setBuffer:e->cbuf_rms_norm_eps  offset:0 atIndex:6];
                        [enc dispatchThreadgroups:MTLSizeMake(LIN_NUM_V_HEADS, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Output projection
                        layer_weight_name(wn, sizeof(wn), layer, "linear_attn.out_proj.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "linear_attn.out_proj.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "linear_attn.out_proj.biases");
                        dispatch_matvec_4bit(e, enc, e->lin_normed_buf, wn, sn, bn,
                                             e->attn_out_buf, HIDDEN_SIZE, LIN_VAL_TOTAL, GROUP_SIZE);

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Attention residual
                        [enc setComputePipelineState:e->residualAddPipeline];
                        [enc setBuffer:e->hidden_buf         offset:0 atIndex:0];
                        [enc setBuffer:e->attn_out_buf       offset:0 atIndex:1];
                        [enc setBuffer:e->hidden_buf2        offset:0 atIndex:2];
                        [enc setBuffer:e->cbuf_hidden_size   offset:0 atIndex:3];
                        [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // post_attention_layernorm
                        layer_weight_name(wn, sizeof(wn), layer, "post_attention_layernorm.weight");
                        dispatch_rmsnorm(e, enc, e->hidden_buf2, wn, e->normed_buf, HIDDEN_SIZE);

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // Router matmul
                        layer_weight_name(wn, sizeof(wn), layer, "mlp.gate.weight");
                        layer_weight_name(sn, sizeof(sn), layer, "mlp.gate.scales");
                        layer_weight_name(bn, sizeof(bn), layer, "mlp.gate.biases");
                        id<MTLBuffer> rw = find_weight(e, wn);
                        id<MTLBuffer> rs = find_weight(e, sn);
                        id<MTLBuffer> rb = find_weight(e, bn);
                        [enc setComputePipelineState:e->routerMatmulPipeline];
                        [enc setBuffer:e->normed_buf         offset:0 atIndex:0];
                        [enc setBuffer:rw                     offset:0 atIndex:1];
                        [enc setBuffer:rs                     offset:0 atIndex:2];
                        [enc setBuffer:rb                     offset:0 atIndex:3];
                        [enc setBuffer:e->router_logits_buf  offset:0 atIndex:4];
                        [enc setBuffer:e->moe_params_buf     offset:0 atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                        // GPU softmax + top-K
                        [enc setComputePipelineState:e->routerSoftmaxTopkPipeline];
                        [enc setBuffer:e->router_logits_buf   offset:0 atIndex:0];
                        [enc setBuffer:e->expert_indices_buf  offset:0 atIndex:1];
                        [enc setBuffer:e->scores_buffer       offset:0 atIndex:2];
                        [enc setBuffer:e->moe_params_buf      offset:0 atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                        [enc endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                    }
                }

                // =========================================================
                // COMMON: CPU pread experts + MoE command buffer
                // =========================================================

                int K = NUM_EXPERTS_PER_TOK;
                int32_t* top_indices = (int32_t*)e->expert_indices_buf.contents;

                // Parallel pread experts via GCD
                int fd = e->layer_fds[layer];
                if (fd < 0) {
                    fprintf(stderr, "[bakan_full] ERROR: No expert file for layer %d\n", layer);
                    return -1;
                }
                char* staging = (char*)e->staging_buffer.contents;
                {
                    dispatch_group_t group = dispatch_group_create();
                    dispatch_queue_t io_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
                    __block int pread_error = 0;
                    for (int k = 0; k < K; k++) {
                        int expert_idx = top_indices[k];
                        size_t off = (size_t)HEADER_SIZE + (size_t)expert_idx * EXPERT_SIZE;
                        char* dst = staging + (size_t)k * EXPERT_SIZE;
                        dispatch_group_async(group, io_queue, ^{
                            ssize_t n = pread(fd, dst, EXPERT_SIZE, (off_t)off);
                            if (n != (ssize_t)EXPERT_SIZE) {
                                pread_error = 1;
                            }
                        });
                    }
                    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
                    if (pread_error) {
                        fprintf(stderr, "[bakan_full] ERROR: pread layer %d failed\n", layer);
                        return -1;
                    }
                }

                // --- CB2: Fused Expert MLP + Shared expert + Blend with residual ---
                {
                    ExpertMLPParams* ep = (ExpertMLPParams*)e->expert_params_buf.contents;
                    for (int k = 0; k < K; k++) {
                        ep->expert_indices[k] = k;
                    }
                    ep->num_experts = K;
                    ep->header_size = 0;

                    id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

                    // Fused expert MLP
                    [enc setComputePipelineState:e->expertMLPFusedPipeline];
                    [enc setBuffer:e->staging_buffer     offset:0 atIndex:0];
                    [enc setBuffer:e->normed_buf          offset:0 atIndex:1];
                    [enc setBuffer:e->expert_output      offset:0 atIndex:2];
                    [enc setBuffer:e->expert_params_buf  offset:0 atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)K, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(EXPERT_DIM, 1, 1)];

                    // Shared expert up + gate + SwiGLU
                    layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert.gate_proj.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert.gate_proj.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert.gate_proj.biases");
                    char wn2[256], sn2[256], bn2[256];
                    layer_weight_name(wn2, sizeof(wn2), layer, "mlp.shared_expert.up_proj.weight");
                    layer_weight_name(sn2, sizeof(sn2), layer, "mlp.shared_expert.up_proj.scales");
                    layer_weight_name(bn2, sizeof(bn2), layer, "mlp.shared_expert.up_proj.biases");

                    [enc setComputePipelineState:e->sharedUpGatePipeline];
                    [enc setBuffer:e->normed_buf                         offset:0 atIndex:0];
                    [enc setBuffer:find_weight(e, wn)                    offset:0 atIndex:1];
                    [enc setBuffer:find_weight(e, sn)                    offset:0 atIndex:2];
                    [enc setBuffer:find_weight(e, bn)                    offset:0 atIndex:3];
                    [enc setBuffer:find_weight(e, wn2)                   offset:0 atIndex:4];
                    [enc setBuffer:find_weight(e, sn2)                   offset:0 atIndex:5];
                    [enc setBuffer:find_weight(e, bn2)                   offset:0 atIndex:6];
                    [enc setBuffer:e->shared_activated                   offset:0 atIndex:7];
                    [enc setBuffer:e->moe_params_buf                     offset:0 atIndex:8];
                    [enc dispatchThreads:MTLSizeMake(EXPERT_DIM, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Shared expert down_proj
                    layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert.down_proj.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert.down_proj.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert.down_proj.biases");

                    [enc setComputePipelineState:e->sharedDownPipeline];
                    [enc setBuffer:e->shared_activated               offset:0 atIndex:0];
                    [enc setBuffer:find_weight(e, wn)                offset:0 atIndex:1];
                    [enc setBuffer:find_weight(e, sn)                offset:0 atIndex:2];
                    [enc setBuffer:find_weight(e, bn)                offset:0 atIndex:3];
                    [enc setBuffer:e->shared_out_buf                 offset:0 atIndex:4];
                    [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:5];
                    [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

                    // Shared expert gate
                    layer_weight_name(wn, sizeof(wn), layer, "mlp.shared_expert_gate.weight");
                    layer_weight_name(sn, sizeof(sn), layer, "mlp.shared_expert_gate.scales");
                    layer_weight_name(bn, sizeof(bn), layer, "mlp.shared_expert_gate.biases");

                    [enc setComputePipelineState:e->sharedGatePipeline];
                    [enc setBuffer:e->normed_buf                     offset:0 atIndex:0];
                    [enc setBuffer:find_weight(e, wn)                offset:0 atIndex:1];
                    [enc setBuffer:find_weight(e, sn)                offset:0 atIndex:2];
                    [enc setBuffer:find_weight(e, bn)                offset:0 atIndex:3];
                    [enc setBuffer:e->shared_gate_buf                offset:0 atIndex:4];
                    [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    // Blend with residual: output = hidden_buf2 + experts + shared
                    [enc setComputePipelineState:e->blendResidualPipeline];
                    [enc setBuffer:e->hidden_buf2                    offset:0 atIndex:0];
                    [enc setBuffer:e->expert_output                  offset:0 atIndex:1];
                    [enc setBuffer:e->scores_buffer                  offset:0 atIndex:2];
                    [enc setBuffer:e->shared_out_buf                 offset:0 atIndex:3];
                    [enc setBuffer:e->shared_gate_buf                offset:0 atIndex:4];
                    [enc setBuffer:e->hidden_buf                     offset:0 atIndex:5];
                    [enc setBuffer:e->moe_params_buf                 offset:0 atIndex:6];
                    [enc dispatchThreads:MTLSizeMake(HIDDEN_SIZE, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                    [enc endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                // Save this token's output hidden state back to prefill_hidden
                memcpy((char*)prefill_hidden.contents + h_off,
                       e->hidden_buf.contents, per_token_bytes);

            } // end token loop

            // Update layer type indices
            if (is_attn) attn_idx++;
            else lin_idx++;

        } // end layer loop

        // hidden_buf now has the last token's output from the final layer.
        // Compute logits for it (final norm + LM head).

        // === Step 4: Final norm + LM head for last token only ===
        {
            id<MTLBuffer> lhw = find_weight(e, "lm_head.weight");
            id<MTLBuffer> lhs = find_weight(e, "lm_head.scales");
            id<MTLBuffer> lhb = find_weight(e, "lm_head.biases");

            if (!lhw || !lhs || !lhb) {
                fprintf(stderr, "[bakan_full] ERROR: lm_head weights not found\n");
                return -1;
            }

            id<MTLCommandBuffer> cmd = [e->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            // Final RMSNorm
            dispatch_rmsnorm(e, enc, e->hidden_buf, "norm.weight",
                             e->normed_buf, HIDDEN_SIZE);

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            // LM head matmul: normed(2048) -> logits(248320)
            [enc setComputePipelineState:e->matvec4bitF32Pipeline];
            [enc setBuffer:e->normed_buf       offset:0 atIndex:0];
            [enc setBuffer:lhw                  offset:0 atIndex:1];
            [enc setBuffer:lhs                  offset:0 atIndex:2];
            [enc setBuffer:lhb                  offset:0 atIndex:3];
            [enc setBuffer:e->logits_buf       offset:0 atIndex:4];
            [enc setBuffer:e->cbuf_vocab_size  offset:0 atIndex:5];
            [enc setBuffer:e->cbuf_hidden_size offset:0 atIndex:6];
            [enc setBuffer:e->cbuf_group_size  offset:0 atIndex:7];

            [enc dispatchThreadgroups:MTLSizeMake((VOCAB_SIZE + 7) / 8, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if (cmd.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "[bakan_full] ERROR: LM head GPU failed: %s\n",
                        cmd.error.localizedDescription.UTF8String);
                return -1;
            }
        }

        // Copy logits out
        memcpy(logits_out, e->logits_buf.contents, VOCAB_SIZE * sizeof(float));
        if (vocab_size_out) *vocab_size_out = VOCAB_SIZE;

        return 0;
    }
}

// ---------------------------------------------------------------------------
// bakan_full_reset_cache
// ---------------------------------------------------------------------------
void bakan_full_reset_cache(void* engine) {
    BakanFullEngine* e = (BakanFullEngine*)engine;
    if (!e) return;

    // Clear KV caches
    size_t kv_size = (size_t)NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * sizeof(uint16_t);
    for (int i = 0; i < e->num_attn_layers; i++) {
        memset(buf_from_ref(e->k_caches[i]).contents, 0, kv_size);
        memset(buf_from_ref(e->v_caches[i]).contents, 0, kv_size);
    }

    // Clear linear attention states
    size_t state_size = (size_t)LIN_NUM_V_HEADS * LIN_VAL_DIM * LIN_KEY_DIM * sizeof(float);
    for (int i = 0; i < e->num_lin_layers; i++) {
        memset(buf_from_ref(e->lin_states[i]).contents, 0, state_size);
    }

    // Clear conv1d buffers
    size_t conv_size = (size_t)LIN_CONV_DIM * CONV_KERNEL_SIZE * sizeof(uint16_t);
    for (int i = 0; i < e->num_lin_layers; i++) {
        memset(buf_from_ref(e->conv_bufs[i]).contents, 0, conv_size);
    }

    fprintf(stderr, "[bakan_full] Cache reset\n");
}

// ---------------------------------------------------------------------------
// bakan_full_set_kv_cache — Set KV cache for hybrid engine
// ---------------------------------------------------------------------------
int bakan_full_set_kv_cache(void* engine, int attn_layer_idx,
                             const void* keys_data, const void* values_data,
                             int seq_len, int num_kv_heads, int head_dim) {
    BakanFullEngine* e = (BakanFullEngine*)engine;
    if (!e || !e->built) {
        fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
        return -1;
    }
    if (attn_layer_idx < 0 || attn_layer_idx >= e->num_attn_layers) {
        fprintf(stderr, "[bakan_full] ERROR: attn_layer_idx %d out of range [0, %d)\n",
                attn_layer_idx, e->num_attn_layers);
        return -1;
    }
    if (seq_len <= 0 || seq_len > MAX_SEQ_LEN) {
        fprintf(stderr, "[bakan_full] ERROR: seq_len %d out of range (0, %d]\n",
                seq_len, MAX_SEQ_LEN);
        return -1;
    }
    if (num_kv_heads != NUM_KV_HEADS || head_dim != HEAD_DIM) {
        fprintf(stderr, "[bakan_full] ERROR: KV cache dimensions mismatch: "
                "got heads=%d dim=%d, expected heads=%d dim=%d\n",
                num_kv_heads, head_dim, NUM_KV_HEADS, HEAD_DIM);
        return -1;
    }

    // Copy keys and values into the Metal KV cache buffers.
    // Source layout: half[num_kv_heads * seq_len * head_dim] — [head][seq][dim]
    // Metal layout:  half[NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM] — [head][seq][dim]
    // We need to copy each head's data with stride MAX_SEQ_LEN, since source
    // has stride seq_len.
    id<MTLBuffer> k_cache = buf_from_ref(e->k_caches[attn_layer_idx]);
    id<MTLBuffer> v_cache = buf_from_ref(e->v_caches[attn_layer_idx]);

    uint16_t* k_dst = (uint16_t*)k_cache.contents;
    uint16_t* v_dst = (uint16_t*)v_cache.contents;
    const uint16_t* k_src = (const uint16_t*)keys_data;
    const uint16_t* v_src = (const uint16_t*)values_data;

    // Clear first (in case previous data exists at seq positions > seq_len)
    size_t total_kv = (size_t)NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM;
    memset(k_dst, 0, total_kv * sizeof(uint16_t));
    memset(v_dst, 0, total_kv * sizeof(uint16_t));

    for (int h = 0; h < NUM_KV_HEADS; h++) {
        // Source: head h starts at h * seq_len * head_dim
        // Dest:   head h starts at h * MAX_SEQ_LEN * head_dim
        size_t src_off = (size_t)h * seq_len * HEAD_DIM;
        size_t dst_off = (size_t)h * MAX_SEQ_LEN * HEAD_DIM;
        size_t copy_bytes = (size_t)seq_len * HEAD_DIM * sizeof(uint16_t);
        memcpy(k_dst + dst_off, k_src + src_off, copy_bytes);
        memcpy(v_dst + dst_off, v_src + src_off, copy_bytes);
    }

    fprintf(stderr, "[bakan_full] KV cache set: attn_layer=%d, seq_len=%d\n",
            attn_layer_idx, seq_len);
    return 0;
}

// ---------------------------------------------------------------------------
// bakan_full_set_linear_state — Set delta-net state for hybrid engine
// ---------------------------------------------------------------------------
int bakan_full_set_linear_state(void* engine, int lin_layer_idx,
                                 const void* state_data, int state_bytes) {
    BakanFullEngine* e = (BakanFullEngine*)engine;
    if (!e || !e->built) {
        fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
        return -1;
    }
    if (lin_layer_idx < 0 || lin_layer_idx >= e->num_lin_layers) {
        fprintf(stderr, "[bakan_full] ERROR: lin_layer_idx %d out of range [0, %d)\n",
                lin_layer_idx, e->num_lin_layers);
        return -1;
    }

    size_t expected = (size_t)LIN_NUM_V_HEADS * LIN_VAL_DIM * LIN_KEY_DIM * sizeof(float);
    if ((size_t)state_bytes != expected) {
        fprintf(stderr, "[bakan_full] ERROR: state_bytes %d != expected %zu\n",
                state_bytes, expected);
        return -1;
    }

    id<MTLBuffer> state_buf = buf_from_ref(e->lin_states[lin_layer_idx]);
    memcpy(state_buf.contents, state_data, expected);
    return 0;
}

// ---------------------------------------------------------------------------
// bakan_full_set_conv_state — Set conv1d buffer for hybrid engine
// ---------------------------------------------------------------------------
int bakan_full_set_conv_state(void* engine, int lin_layer_idx,
                               const void* conv_data, int conv_bytes) {
    BakanFullEngine* e = (BakanFullEngine*)engine;
    if (!e || !e->built) {
        fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
        return -1;
    }
    if (lin_layer_idx < 0 || lin_layer_idx >= e->num_lin_layers) {
        fprintf(stderr, "[bakan_full] ERROR: lin_layer_idx %d out of range [0, %d)\n",
                lin_layer_idx, e->num_lin_layers);
        return -1;
    }

    size_t expected = (size_t)LIN_CONV_DIM * CONV_KERNEL_SIZE * sizeof(uint16_t);
    if ((size_t)conv_bytes != expected) {
        fprintf(stderr, "[bakan_full] ERROR: conv_bytes %d != expected %zu\n",
                conv_bytes, expected);
        return -1;
    }

    id<MTLBuffer> conv_buf = buf_from_ref(e->conv_bufs[lin_layer_idx]);
    memcpy(conv_buf.contents, conv_data, expected);
    return 0;
}

// ---------------------------------------------------------------------------
// bakan_full_set_position — Set sequence position for hybrid engine
// ---------------------------------------------------------------------------
int bakan_full_set_position(void* engine, int position) {
    BakanFullEngine* e = (BakanFullEngine*)engine;
    if (!e || !e->built) {
        fprintf(stderr, "[bakan_full] ERROR: Engine not built\n");
        return -1;
    }
    if (position < 0 || position > MAX_SEQ_LEN) {
        fprintf(stderr, "[bakan_full] ERROR: position %d out of range [0, %d]\n",
                position, MAX_SEQ_LEN);
        return -1;
    }

    // These buffers are read by the forward pass to determine where in the
    // KV cache to write and for RoPE computation.
    *(int32_t*)e->cbuf_position.contents = (int32_t)position;
    *(int32_t*)e->cbuf_kv_len.contents   = (int32_t)(position + 1);

    fprintf(stderr, "[bakan_full] Position set to %d\n", position);
    return 0;
}

// ---------------------------------------------------------------------------
// bakan_full_destroy
// ---------------------------------------------------------------------------
void bakan_full_destroy(void* engine) {
    if (!engine) return;

    @autoreleasepool {
        BakanFullEngine* e = (BakanFullEngine*)engine;

        // Close expert file descriptors
        if (e->layer_fds) {
            for (int i = 0; i < e->num_layers; i++) {
                if (e->layer_fds[i] >= 0) close(e->layer_fds[i]);
            }
            free(e->layer_fds);
        }

        // Free weight entries
        if (e->weights) {
            for (int i = 0; i < e->weight_count; i++) {
                e->weights[i].buffer = nil;
            }
            free(e->weights);
        }

        // Free per-layer buffers (CFBridgingRelease to balance CFBridgingRetain)
        if (e->conv_bufs) {
            for (int i = 0; i < e->num_lin_layers; i++) {
                if (e->conv_bufs[i]) CFRelease(e->conv_bufs[i]);
            }
            free(e->conv_bufs);
        }
        if (e->k_caches) {
            for (int i = 0; i < e->num_attn_layers; i++) {
                if (e->k_caches[i]) CFRelease(e->k_caches[i]);
                if (e->v_caches[i]) CFRelease(e->v_caches[i]);
            }
            free(e->k_caches);
            free(e->v_caches);
        }
        if (e->lin_states) {
            for (int i = 0; i < e->num_lin_layers; i++) {
                if (e->lin_states[i]) CFRelease(e->lin_states[i]);
            }
            free(e->lin_states);
        }

        // Nil out all scratch buffers (ARC handles release)
        e->hidden_buf = nil;
        e->hidden_buf2 = nil;
        e->normed_buf = nil;
        e->attn_out_buf = nil;
        e->proj_buf = nil;
        e->proj_buf2 = nil;
        e->proj_buf3 = nil;
        e->combined_proj_buf = nil;
        e->gate_buf = nil;
        e->gated_attn_buf = nil;
        e->q_buf = nil;
        e->k_buf = nil;
        e->v_buf = nil;
        e->attn_raw_buf = nil;
        e->lin_qkv_buf = nil;
        e->lin_z_buf = nil;
        e->lin_b_buf = nil;
        e->lin_a_buf = nil;
        e->lin_q_buf = nil;
        e->lin_k_buf = nil;
        e->lin_v_buf = nil;
        e->lin_out_buf = nil;
        e->lin_normed_buf = nil;
        e->staging_buffer = nil;
        e->expert_activated = nil;
        e->expert_output = nil;
        e->shared_activated = nil;
        e->shared_out_buf = nil;
        e->shared_gate_buf = nil;
        e->moe_out_buf = nil;
        e->scores_buffer = nil;
        e->expert_params_buf = nil;
        e->moe_params_buf = nil;
        e->router_logits_buf = nil;
        e->expert_indices_buf = nil;
        e->logits_buf = nil;

        // Nil out pre-allocated constant parameter buffers
        e->cbuf_hidden_size = nil;
        e->cbuf_vocab_size = nil;
        e->cbuf_group_size = nil;
        e->cbuf_q_proj_out = nil;
        e->cbuf_k_proj_out = nil;
        e->cbuf_v_proj_out = nil;
        e->cbuf_attn_out_dim = nil;
        e->cbuf_num_q_heads = nil;
        e->cbuf_num_kv_heads = nil;
        e->cbuf_head_dim = nil;
        e->cbuf_rope_dim = nil;
        e->cbuf_max_seq_len = nil;
        e->cbuf_lin_conv_dim = nil;
        e->cbuf_conv_kernel = nil;
        e->cbuf_lin_num_k_heads = nil;
        e->cbuf_lin_key_dim = nil;
        e->cbuf_lin_key_total = nil;
        e->cbuf_lin_num_v_heads = nil;
        e->cbuf_lin_val_dim = nil;
        e->cbuf_lin_val_total = nil;
        e->cbuf_rms_norm_eps = nil;
        e->cbuf_rope_theta = nil;
        e->cbuf_q_scale = nil;
        e->cbuf_k_scale = nil;
        e->cbuf_position = nil;
        e->cbuf_kv_len = nil;

        // Release pipelines
        e->device = nil;
        e->queue = nil;

        free(e);
        fprintf(stderr, "[bakan_full] Engine destroyed\n");
    }
}

// ---------------------------------------------------------------------------
// bakan_full_num_layers / bakan_full_vocab_size
// ---------------------------------------------------------------------------
int bakan_full_num_layers(void* engine) {
    if (!engine) return 0;
    return ((BakanFullEngine*)engine)->num_layers;
}

int bakan_full_vocab_size(void* engine) {
    (void)engine;
    return VOCAB_SIZE;
}
