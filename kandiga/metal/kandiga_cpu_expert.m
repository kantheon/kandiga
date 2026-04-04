/*
 * kandiga_cpu_expert.m -- CPU-side expert MLP computation for MoE models.
 *
 * Computes the expert MLP entirely on CPU using:
 *   - pread for weight loading (OS page cache, zero-copy)
 *   - Vectorized 4-bit dequantization (NEON intrinsics on Apple Silicon)
 *   - Parallel expert computation via GCD dispatch groups
 *
 * Runs in parallel with GPU attention computed by MLX. On Apple Silicon's
 * unified memory, CPU and GPU share the same physical DRAM, so there's no
 * data transfer overhead.
 *
 * Build:
 *   clang -shared -o libkandiga_cpu_expert.dylib kandiga_cpu_expert.m \
 *         -framework Foundation -fobjc-arc -O2 -march=native
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dispatch/dispatch.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#include "kandiga_cpu_expert.h"

/* ----------------------------------------------------------------------- */
/* Model constants (must match packed binary format)                         */
/* ----------------------------------------------------------------------- */
#define MAX_EXPERTS     16
#define HIDDEN_SIZE     2048
#define EXPERT_DIM      512
#define GROUP_SIZE      64
#define HEADER_SIZE     4096UL
#define EXPERT_SIZE     1769472UL

/* Expert tensor byte offsets within each expert block — 4-bit */
#define GATE_WEIGHT_OFF   0
#define GATE_SCALES_OFF   524288
#define GATE_BIASES_OFF   557056
#define UP_WEIGHT_OFF     589824
#define UP_SCALES_OFF     1114112
#define UP_BIASES_OFF     1146880
#define DOWN_WEIGHT_OFF   1179648
#define DOWN_SCALES_OFF   1703936
#define DOWN_BIASES_OFF   1736704

/* Expert tensor byte offsets within each expert block — 3-bit */
#define GATE_WEIGHT_OFF_3  0
#define GATE_SCALES_OFF_3  393216
#define GATE_BIASES_OFF_3  425984
#define UP_WEIGHT_OFF_3    458752
#define UP_SCALES_OFF_3    851968
#define UP_BIASES_OFF_3    884736
#define DOWN_WEIGHT_OFF_3  917504
#define DOWN_SCALES_OFF_3  1310720
#define DOWN_BIASES_OFF_3  1343488
#define EXPERT_SIZE_3BIT   1376256UL

/* ----------------------------------------------------------------------- */
/* Engine state                                                             */
/* ----------------------------------------------------------------------- */
typedef struct {
    int*    layer_fds;
    int     num_layers;
    int     bits;          /* 3 or 4 — auto-detected from header */
    size_t  expert_size;   /* bytes per expert block */
    char*   expert_bufs[MAX_EXPERTS];
} KandigaCPUExpertEngine;

/* ----------------------------------------------------------------------- */
/* Helpers                                                                  */
/* ----------------------------------------------------------------------- */
static inline float bf16_to_float(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

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

/* ----------------------------------------------------------------------- */
/* NEON-optimized 3-bit dequant matrix-vector multiply                      */
/* MLX 3-bit packing: 3 bytes → 8 values (same as quantized.h)             */
/* ----------------------------------------------------------------------- */
static void cpu_dequant_matvec_3bit(
    const uint8_t*  __restrict weight,  /* packed 3-bit: out_dim × (in_dim*3/8) bytes */
    const uint16_t* __restrict scales,
    const uint16_t* __restrict biases,
    const float*    __restrict x,
    float*          __restrict output,
    int out_dim,
    int in_dim,
    int group_size
) {
    int bytes_per_row = in_dim * 3 / 8;      /* 3 bytes per 8 values */
    int num_groups = in_dim / group_size;
    int packs_per_group = group_size / 8;     /* 8 values per 3-byte pack */

    for (int row = 0; row < out_dim; row++) {
        const uint8_t*  w_row = weight + row * bytes_per_row;
        const uint16_t* s_row = scales + row * num_groups;
        const uint16_t* b_row = biases + row * num_groups;

#ifdef __ARM_NEON__
        float32x4_t acc_vec = vdupq_n_f32(0.0f);

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_float(s_row[g]);
            float bias  = bf16_to_float(b_row[g]);

            float32x4_t v_scale = vdupq_n_f32(scale);
            float32x4_t v_bias  = vdupq_n_f32(bias);

            const uint8_t* wp = w_row + g * packs_per_group * 3;
            int xb = g * group_size;

            /* Process 2 packs per iteration = 6 bytes → 16 values → 4 NEON FMAs */
            for (int p = 0; p < packs_per_group; p += 2) {
                const uint8_t* b0 = wp + p * 3;       /* first pack: 3 bytes → 8 vals */
                const uint8_t* b1 = wp + p * 3 + 3;   /* second pack: 3 bytes → 8 vals */
                const float* xp = x + xb + p * 8;

                /* Unpack first 3 bytes → 8 values */
                float32x4_t n0 = {
                    (float)( b0[0]       & 0x07),
                    (float)((b0[0] >> 3) & 0x07),
                    (float)(((b0[0] >> 6) & 0x03) | ((b0[1] & 0x01) << 2)),
                    (float)((b0[1] >> 1) & 0x07)
                };
                float32x4_t n1 = {
                    (float)((b0[1] >> 4) & 0x07),
                    (float)(((b0[1] >> 7) & 0x01) | ((b0[2] & 0x03) << 1)),
                    (float)((b0[2] >> 2) & 0x07),
                    (float)((b0[2] >> 5) & 0x07)
                };

                /* Unpack second 3 bytes → 8 values */
                float32x4_t n2 = {
                    (float)( b1[0]       & 0x07),
                    (float)((b1[0] >> 3) & 0x07),
                    (float)(((b1[0] >> 6) & 0x03) | ((b1[1] & 0x01) << 2)),
                    (float)((b1[1] >> 1) & 0x07)
                };
                float32x4_t n3 = {
                    (float)((b1[1] >> 4) & 0x07),
                    (float)(((b1[1] >> 7) & 0x01) | ((b1[2] & 0x03) << 1)),
                    (float)((b1[2] >> 2) & 0x07),
                    (float)((b1[2] >> 5) & 0x07)
                };

                /* Dequant + FMA: w = nibble * scale + bias; acc += w * x */
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n0, v_scale), vld1q_f32(xp));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n1, v_scale), vld1q_f32(xp+4));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n2, v_scale), vld1q_f32(xp+8));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n3, v_scale), vld1q_f32(xp+12));
            }
        }

        float32x2_t sum_pair = vadd_f32(vget_low_f32(acc_vec), vget_high_f32(acc_vec));
        float32x2_t sum_final = vpadd_f32(sum_pair, sum_pair);
        output[row] = vget_lane_f32(sum_final, 0);
#else
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_float(s_row[g]);
            float bias  = bf16_to_float(b_row[g]);

            const uint8_t* wp = w_row + g * packs_per_group * 3;

            for (int p = 0; p < packs_per_group; p++) {
                const uint8_t* b0 = wp + p * 3;
                int xi = g * group_size + p * 8;

                float v0 = (float)( b0[0]       & 0x07);
                float v1 = (float)((b0[0] >> 3) & 0x07);
                float v2 = (float)(((b0[0] >> 6) & 0x03) | ((b0[1] & 0x01) << 2));
                float v3 = (float)((b0[1] >> 1) & 0x07);
                float v4 = (float)((b0[1] >> 4) & 0x07);
                float v5 = (float)(((b0[1] >> 7) & 0x01) | ((b0[2] & 0x03) << 1));
                float v6 = (float)((b0[2] >> 2) & 0x07);
                float v7 = (float)((b0[2] >> 5) & 0x07);

                acc += (v0 * scale + bias) * x[xi + 0];
                acc += (v1 * scale + bias) * x[xi + 1];
                acc += (v2 * scale + bias) * x[xi + 2];
                acc += (v3 * scale + bias) * x[xi + 3];
                acc += (v4 * scale + bias) * x[xi + 4];
                acc += (v5 * scale + bias) * x[xi + 5];
                acc += (v6 * scale + bias) * x[xi + 6];
                acc += (v7 * scale + bias) * x[xi + 7];
            }
        }
        output[row] = acc;
#endif
    }
}

/* ----------------------------------------------------------------------- */
/* NEON-optimized 4-bit dequant matrix-vector multiply                      */
/* ----------------------------------------------------------------------- */
static void cpu_dequant_matvec_4bit(
    const uint32_t* __restrict weight,
    const uint16_t* __restrict scales,
    const uint16_t* __restrict biases,
    const float*    __restrict x,
    float*          __restrict output,
    int out_dim,
    int in_dim,
    int group_size
) {
    int packed_per_row = in_dim / 8;
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;

    for (int row = 0; row < out_dim; row++) {
        const uint32_t* w_row = weight + row * packed_per_row;
        const uint16_t* s_row = scales + row * num_groups;
        const uint16_t* b_row = biases + row * num_groups;

#ifdef __ARM_NEON__
        float32x4_t acc_vec = vdupq_n_f32(0.0f);

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_float(s_row[g]);
            float bias  = bf16_to_float(b_row[g]);

            float32x4_t v_scale = vdupq_n_f32(scale);
            float32x4_t v_bias  = vdupq_n_f32(bias);

            for (int p = 0; p < packed_per_group; p += 4) {
                const uint32_t* wp = w_row + g * packed_per_group + p;
                uint32_t pk0 = wp[0], pk1 = wp[1], pk2 = wp[2], pk3 = wp[3];
                int xb = g * group_size + p * 8;
                const float* xp = x + xb;

                float32x4_t n0 = {(float)(pk0&0xF),(float)((pk0>>4)&0xF),(float)((pk0>>8)&0xF),(float)((pk0>>12)&0xF)};
                float32x4_t n1 = {(float)((pk0>>16)&0xF),(float)((pk0>>20)&0xF),(float)((pk0>>24)&0xF),(float)((pk0>>28)&0xF)};
                float32x4_t n2 = {(float)(pk1&0xF),(float)((pk1>>4)&0xF),(float)((pk1>>8)&0xF),(float)((pk1>>12)&0xF)};
                float32x4_t n3 = {(float)((pk1>>16)&0xF),(float)((pk1>>20)&0xF),(float)((pk1>>24)&0xF),(float)((pk1>>28)&0xF)};
                float32x4_t n4 = {(float)(pk2&0xF),(float)((pk2>>4)&0xF),(float)((pk2>>8)&0xF),(float)((pk2>>12)&0xF)};
                float32x4_t n5 = {(float)((pk2>>16)&0xF),(float)((pk2>>20)&0xF),(float)((pk2>>24)&0xF),(float)((pk2>>28)&0xF)};
                float32x4_t n6 = {(float)(pk3&0xF),(float)((pk3>>4)&0xF),(float)((pk3>>8)&0xF),(float)((pk3>>12)&0xF)};
                float32x4_t n7 = {(float)((pk3>>16)&0xF),(float)((pk3>>20)&0xF),(float)((pk3>>24)&0xF),(float)((pk3>>28)&0xF)};

                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n0, v_scale), vld1q_f32(xp));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n1, v_scale), vld1q_f32(xp+4));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n2, v_scale), vld1q_f32(xp+8));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n3, v_scale), vld1q_f32(xp+12));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n4, v_scale), vld1q_f32(xp+16));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n5, v_scale), vld1q_f32(xp+20));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n6, v_scale), vld1q_f32(xp+24));
                acc_vec = vfmaq_f32(acc_vec, vfmaq_f32(v_bias, n7, v_scale), vld1q_f32(xp+28));
            }
        }

        float32x2_t sum_pair = vadd_f32(vget_low_f32(acc_vec), vget_high_f32(acc_vec));
        float32x2_t sum_final = vpadd_f32(sum_pair, sum_pair);
        output[row] = vget_lane_f32(sum_final, 0);
#else
        float acc = 0.0f;
        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_float(s_row[g]);
            float bias  = bf16_to_float(b_row[g]);

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[g * packed_per_group + p];
                int x_base = g * group_size + p * 8;

                for (int n = 0; n < 8; n++) {
                    float nibble = (float)((packed >> (n * 4)) & 0xF);
                    float w = nibble * scale + bias;
                    acc += w * x[x_base + n];
                }
            }
        }
        output[row] = acc;
#endif
    }
}

/* ----------------------------------------------------------------------- */
/* Single expert MLP: gate + up + SwiGLU + down                             */
/* ----------------------------------------------------------------------- */
static int compute_single_expert(
    const char* expert_data,
    const float* x,
    float* output
) {
    const uint32_t* gate_w = (const uint32_t*)(expert_data + GATE_WEIGHT_OFF);
    const uint16_t* gate_s = (const uint16_t*)(expert_data + GATE_SCALES_OFF);
    const uint16_t* gate_b = (const uint16_t*)(expert_data + GATE_BIASES_OFF);

    const uint32_t* up_w = (const uint32_t*)(expert_data + UP_WEIGHT_OFF);
    const uint16_t* up_s = (const uint16_t*)(expert_data + UP_SCALES_OFF);
    const uint16_t* up_b = (const uint16_t*)(expert_data + UP_BIASES_OFF);

    const uint32_t* down_w = (const uint32_t*)(expert_data + DOWN_WEIGHT_OFF);
    const uint16_t* down_s = (const uint16_t*)(expert_data + DOWN_SCALES_OFF);
    const uint16_t* down_b = (const uint16_t*)(expert_data + DOWN_BIASES_OFF);

    /* gate_proj */
    float gate_out[EXPERT_DIM];
    cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, x, gate_out,
                            EXPERT_DIM, HIDDEN_SIZE, GROUP_SIZE);

    /* up_proj */
    float up_out[EXPERT_DIM];
    cpu_dequant_matvec_4bit(up_w, up_s, up_b, x, up_out,
                            EXPERT_DIM, HIDDEN_SIZE, GROUP_SIZE);

    /* SwiGLU activation: silu(gate) * up */
    float activated[EXPERT_DIM];
    for (int i = 0; i < EXPERT_DIM; i += 4) {
        float silu0 = gate_out[i]   / (1.0f + expf(-gate_out[i]));
        float silu1 = gate_out[i+1] / (1.0f + expf(-gate_out[i+1]));
        float silu2 = gate_out[i+2] / (1.0f + expf(-gate_out[i+2]));
        float silu3 = gate_out[i+3] / (1.0f + expf(-gate_out[i+3]));

        activated[i]   = silu0 * up_out[i];
        activated[i+1] = silu1 * up_out[i+1];
        activated[i+2] = silu2 * up_out[i+2];
        activated[i+3] = silu3 * up_out[i+3];
    }

    /* down_proj */
    cpu_dequant_matvec_4bit(down_w, down_s, down_b, activated, output,
                            HIDDEN_SIZE, EXPERT_DIM, GROUP_SIZE);

    return 0;
}

/* ----------------------------------------------------------------------- */
/* Single expert MLP: gate + up + SwiGLU + down — 3-bit variant            */
/* ----------------------------------------------------------------------- */
static int compute_single_expert_3bit(
    const char* expert_data,
    const float* x,
    float* output
) {
    const uint8_t*  gate_w = (const uint8_t*) (expert_data + GATE_WEIGHT_OFF_3);
    const uint16_t* gate_s = (const uint16_t*)(expert_data + GATE_SCALES_OFF_3);
    const uint16_t* gate_b = (const uint16_t*)(expert_data + GATE_BIASES_OFF_3);

    const uint8_t*  up_w = (const uint8_t*) (expert_data + UP_WEIGHT_OFF_3);
    const uint16_t* up_s = (const uint16_t*)(expert_data + UP_SCALES_OFF_3);
    const uint16_t* up_b = (const uint16_t*)(expert_data + UP_BIASES_OFF_3);

    const uint8_t*  down_w = (const uint8_t*) (expert_data + DOWN_WEIGHT_OFF_3);
    const uint16_t* down_s = (const uint16_t*)(expert_data + DOWN_SCALES_OFF_3);
    const uint16_t* down_b = (const uint16_t*)(expert_data + DOWN_BIASES_OFF_3);

    /* gate_proj */
    float gate_out[EXPERT_DIM];
    cpu_dequant_matvec_3bit(gate_w, gate_s, gate_b, x, gate_out,
                            EXPERT_DIM, HIDDEN_SIZE, GROUP_SIZE);

    /* up_proj */
    float up_out[EXPERT_DIM];
    cpu_dequant_matvec_3bit(up_w, up_s, up_b, x, up_out,
                            EXPERT_DIM, HIDDEN_SIZE, GROUP_SIZE);

    /* SwiGLU activation: silu(gate) * up */
    float activated[EXPERT_DIM];
    for (int i = 0; i < EXPERT_DIM; i += 4) {
        float silu0 = gate_out[i]   / (1.0f + expf(-gate_out[i]));
        float silu1 = gate_out[i+1] / (1.0f + expf(-gate_out[i+1]));
        float silu2 = gate_out[i+2] / (1.0f + expf(-gate_out[i+2]));
        float silu3 = gate_out[i+3] / (1.0f + expf(-gate_out[i+3]));

        activated[i]   = silu0 * up_out[i];
        activated[i+1] = silu1 * up_out[i+1];
        activated[i+2] = silu2 * up_out[i+2];
        activated[i+3] = silu3 * up_out[i+3];
    }

    /* down_proj */
    cpu_dequant_matvec_3bit(down_w, down_s, down_b, activated, output,
                            HIDDEN_SIZE, EXPERT_DIM, GROUP_SIZE);

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_init                                                    */
/* ----------------------------------------------------------------------- */
void* bakan_cpu_expert_init(const char* packed_dir, int num_layers) {
    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)calloc(1, sizeof(KandigaCPUExpertEngine));
    if (!engine) {
        fprintf(stderr, "[kandiga] ERROR: Failed to allocate engine\n");
        return NULL;
    }

    engine->num_layers = num_layers;
    engine->bits = 4;               /* default */
    engine->expert_size = EXPERT_SIZE;

    engine->layer_fds = (int*)calloc((size_t)num_layers, sizeof(int));
    if (!engine->layer_fds) {
        fprintf(stderr, "[kandiga] ERROR: Failed to allocate fd array\n");
        free(engine);
        return NULL;
    }
    for (int i = 0; i < num_layers; i++) engine->layer_fds[i] = -1;

    for (int i = 0; i < num_layers; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", packed_dir, i);
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "[kandiga] ERROR: Cannot open %s: %s\n",
                    path, strerror(errno));
            bakan_cpu_expert_destroy(engine);
            return NULL;
        }
        engine->layer_fds[i] = fd;

        /* Auto-detect bits from first file's expert_size in header */
        if (i == 0) {
            uint8_t hdr[24];
            ssize_t n = pread(fd, hdr, 24, 0);
            if (n >= 20 && hdr[0] == 'B' && hdr[1] == 'K' && hdr[2] == 'E' && hdr[3] == 'X') {
                uint64_t esz;
                memcpy(&esz, hdr + 12, 8);
                engine->expert_size = (size_t)esz;
                if (esz == EXPERT_SIZE_3BIT) {
                    engine->bits = 3;
                }
            }
        }
    }

    for (int i = 0; i < MAX_EXPERTS; i++) {
        engine->expert_bufs[i] = (char*)malloc(engine->expert_size);
        if (!engine->expert_bufs[i]) {
            fprintf(stderr, "[kandiga] ERROR: Failed to allocate expert buffer %d\n", i);
            bakan_cpu_expert_destroy(engine);
            return NULL;
        }
    }

    fprintf(stderr, "[kandiga] Initialized: %d layers, %d expert buffers "
            "(%zu KB each), %d-bit\n", num_layers, MAX_EXPERTS,
            engine->expert_size / 1024, engine->bits);

    return engine;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_mlp -- Parallel CPU expert computation via GCD           */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_mlp(
    void* engine_ptr,
    int layer_idx,
    const float* x_f32,
    const int32_t* expert_indices,
    int num_experts,
    float* output_f32
) {
    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)engine_ptr;
    if (!engine) return -1;

    if (layer_idx < 0 || layer_idx >= engine->num_layers) {
        fprintf(stderr, "[kandiga] ERROR: layer_idx %d out of range [0, %d)\n",
                layer_idx, engine->num_layers);
        return -1;
    }
    if (num_experts <= 0 || num_experts > MAX_EXPERTS) {
        fprintf(stderr, "[kandiga] ERROR: num_experts %d out of range [1, %d]\n",
                num_experts, MAX_EXPERTS);
        return -1;
    }

    int fd = engine->layer_fds[layer_idx];

    /* Phase 1: Parallel pread */
    __block int pread_error = 0;
    {
        dispatch_group_t group = dispatch_group_create();
        dispatch_queue_t io_queue = dispatch_get_global_queue(
            QOS_CLASS_USER_INTERACTIVE, 0);

        size_t esz = engine->expert_size;
        for (int k = 0; k < num_experts; k++) {
            int expert_idx = expert_indices[k];
            char* buf = engine->expert_bufs[k];
            dispatch_group_async(group, io_queue, ^{
                off_t offset = (off_t)HEADER_SIZE + (off_t)expert_idx * (off_t)esz;
                ssize_t n = pread(fd, buf, esz, offset);
                if (n != (ssize_t)esz) {
                    pread_error = 1;
                }
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }

    if (pread_error) {
        fprintf(stderr, "[kandiga] ERROR: pread failed for layer %d\n", layer_idx);
        return -1;
    }

    /* Phase 2: Parallel expert MLP computation */
    __block int compute_error = 0;
    {
        dispatch_group_t group = dispatch_group_create();
        dispatch_queue_t compute_queue = dispatch_get_global_queue(
            QOS_CLASS_USER_INTERACTIVE, 0);

        int use_3bit = (engine->bits == 3);
        for (int k = 0; k < num_experts; k++) {
            const char* expert_data = engine->expert_bufs[k];
            float* expert_output = output_f32 + k * HIDDEN_SIZE;
            dispatch_group_async(group, compute_queue, ^{
                int ret = use_3bit
                    ? compute_single_expert_3bit(expert_data, x_f32, expert_output)
                    : compute_single_expert(expert_data, x_f32, expert_output);
                if (ret != 0) {
                    compute_error = 1;
                }
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }

    if (compute_error) {
        fprintf(stderr, "[kandiga] ERROR: Expert computation failed for layer %d\n",
                layer_idx);
        return -1;
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_mlp_batch_f16 — Batched prefill: N tokens, deduplicated */
/*                                                                          */
/* Instead of N separate calls (each doing K preads), this:                 */
/* 1. Collects ALL unique expert indices across N tokens                    */
/* 2. Preads each unique expert ONCE                                        */
/* 3. Computes all tokens using cached expert data                          */
/*                                                                          */
/* For 500 tokens with K=4: ~200 unique experts vs 2000 preads.            */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_mlp_batch_f16(
    void* engine_ptr,
    int layer_idx,
    const void* x_f16_batch,       /* float16[num_tokens][hidden_size] */
    const int32_t* indices_batch,  /* int32[num_tokens][K] */
    int num_tokens,
    int K,
    void* output_f16_batch         /* float16[num_tokens][K][hidden_size] */
) {
    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)engine_ptr;
    if (!engine || layer_idx < 0 || layer_idx >= engine->num_layers) return -1;
    if (num_tokens <= 0 || K <= 0 || K > MAX_EXPERTS) return -1;

    int fd = engine->layer_fds[layer_idx];

    /* Step 1: Collect all unique expert indices */
    int unique_experts[256];
    int num_unique = 0;
    char seen[256] = {0};

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < K; k++) {
            int eidx = indices_batch[t * K + k];
            if (eidx >= 0 && eidx < 256 && !seen[eidx]) {
                seen[eidx] = 1;
                unique_experts[num_unique++] = eidx;
            }
        }
    }

    /* Step 2: Pread unique experts in parallel */
    /* Use a staging area: expert_bufs[0..num_unique-1] */
    /* Map: expert_index -> buffer_slot */
    int expert_to_slot[256];
    memset(expert_to_slot, -1, sizeof(expert_to_slot));

    __block int pread_error = 0;
    {
        dispatch_group_t group = dispatch_group_create();
        dispatch_queue_t io_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

        for (int u = 0; u < num_unique && u < MAX_EXPERTS; u++) {
            int eidx = unique_experts[u];
            expert_to_slot[eidx] = u;
            char* buf = engine->expert_bufs[u];
            dispatch_group_async(group, io_queue, ^{
                off_t offset = (off_t)HEADER_SIZE + (off_t)eidx * (off_t)engine->expert_size;
                ssize_t n = pread(fd, buf, engine->expert_size, offset);
                if (n != (ssize_t)engine->expert_size) pread_error = 1;
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    }
    /* Handle overflow: if more than MAX_EXPERTS unique, pread remaining */
    for (int u = MAX_EXPERTS; u < num_unique; u++) {
        int eidx = unique_experts[u];
        expert_to_slot[eidx] = u % MAX_EXPERTS;
        off_t offset = (off_t)HEADER_SIZE + (off_t)eidx * (off_t)engine->expert_size;
        pread(fd, engine->expert_bufs[u % MAX_EXPERTS], engine->expert_size, offset);
    }
    if (pread_error) return -1;

    /* Step 3: Compute all tokens using cached expert data */
    const uint16_t* x_batch = (const uint16_t*)x_f16_batch;
    uint16_t* out_batch = (uint16_t*)output_f16_batch;

    for (int t = 0; t < num_tokens; t++) {
        /* Convert f16 input to f32 */
        float x_f32[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            x_f32[i] = half_to_float(x_batch[t * HIDDEN_SIZE + i]);
        }

        /* Compute each expert for this token */
        float output_f32[MAX_EXPERTS * HIDDEN_SIZE];
        for (int k = 0; k < K; k++) {
            int eidx = indices_batch[t * K + k];
            int slot = expert_to_slot[eidx];
            if (slot < 0) continue;
            const char* expert_data = engine->expert_bufs[slot];
            if (engine->bits == 3)
                compute_single_expert_3bit(expert_data, x_f32, output_f32 + k * HIDDEN_SIZE);
            else
                compute_single_expert(expert_data, x_f32, output_f32 + k * HIDDEN_SIZE);
        }

        /* Convert f32 output to f16 */
        for (int i = 0; i < K * HIDDEN_SIZE; i++) {
            out_batch[t * K * HIDDEN_SIZE + i] = float_to_half(output_f32[i]);
        }
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_mlp_f16                                                 */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_mlp_f16(
    void* engine_ptr,
    int layer_idx,
    const void* x_f16,
    const int32_t* expert_indices,
    int num_experts,
    void* output_f16
) {
    const uint16_t* x_half = (const uint16_t*)x_f16;
    float x_f32[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        x_f32[i] = half_to_float(x_half[i]);
    }

    float output_f32[MAX_EXPERTS * HIDDEN_SIZE];
    int ret = bakan_cpu_expert_mlp(engine_ptr, layer_idx, x_f32,
                                    expert_indices, num_experts, output_f32);
    if (ret != 0) return ret;

    uint16_t* out_half = (uint16_t*)output_f16;
    int total = num_experts * HIDDEN_SIZE;
    for (int i = 0; i < total; i++) {
        out_half[i] = float_to_half(output_f32[i]);
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_mlp_bf16                                                */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_mlp_bf16(
    void* engine_ptr,
    int layer_idx,
    const void* x_bf16,
    const int32_t* expert_indices,
    int num_experts,
    void* output_f16
) {
    const uint16_t* x_raw = (const uint16_t*)x_bf16;
    float x_f32[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        x_f32[i] = bf16_to_float(x_raw[i]);
    }

    float output_f32[MAX_EXPERTS * HIDDEN_SIZE];
    int ret = bakan_cpu_expert_mlp(engine_ptr, layer_idx, x_f32,
                                    expert_indices, num_experts, output_f32);
    if (ret != 0) return ret;

    uint16_t* out_half = (uint16_t*)output_f16;
    int total = num_experts * HIDDEN_SIZE;
    for (int i = 0; i < total; i++) {
        out_half[i] = float_to_half(output_f32[i]);
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_destroy                                                 */
/* ----------------------------------------------------------------------- */
void bakan_cpu_expert_destroy(void* engine_ptr) {
    if (!engine_ptr) return;

    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)engine_ptr;

    if (engine->layer_fds) {
        for (int i = 0; i < engine->num_layers; i++) {
            if (engine->layer_fds[i] >= 0) {
                close(engine->layer_fds[i]);
            }
        }
        free(engine->layer_fds);
    }

    for (int i = 0; i < MAX_EXPERTS; i++) {
        if (engine->expert_bufs[i]) {
            free(engine->expert_bufs[i]);
        }
    }

    free(engine);
    fprintf(stderr, "[kandiga] Engine destroyed\n");
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_num_layers                                              */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_num_layers(void* engine_ptr) {
    if (!engine_ptr) return 0;
    return ((KandigaCPUExpertEngine*)engine_ptr)->num_layers;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_mlp_prefill_f16 — Process grouped tokens per expert     */
/*                                                                          */
/* Instead of calling mlp_f16 once per token (500 calls per layer):        */
/* 1. Group tokens by expert ID (Python does this)                          */
/* 2. For each expert: pread once, compute ALL grouped tokens               */
/*    Expert weights stay in L1/L2 cache across tokens = ~3x faster         */
/* 3. One C call per layer instead of 500                                   */
/*                                                                          */
/* Input layout:                                                            */
/*   x_f16[num_tokens][hidden_size] — all token vectors                     */
/*   sorted_token_indices[total_assignments] — token index for each assign  */
/*   sorted_expert_indices[total_assignments] — expert ID for each assign   */
/*   sorted_k_positions[total_assignments] — which K slot (0..K-1)          */
/*   group_starts[num_unique_experts+1] — CSR-style group boundaries        */
/*   unique_experts[num_unique_experts] — which experts to load             */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_mlp_prefill_f16(
    void* engine_ptr,
    int layer_idx,
    const void* x_f16,                 /* half[num_tokens * hidden_size] */
    int num_tokens,
    int K,
    int hidden_size_param,
    const int32_t* sorted_token_indices,  /* which token */
    const int32_t* sorted_expert_indices, /* which expert */
    const int32_t* sorted_k_positions,    /* which K slot */
    const int32_t* group_starts,          /* CSR boundaries */
    const int32_t* unique_experts,        /* expert IDs to load */
    int num_unique_experts,
    void* output_f16                   /* half[num_tokens * K * hidden_size] */
) {
    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)engine_ptr;
    if (!engine || layer_idx < 0 || layer_idx >= engine->num_layers) return -1;

    int fd = engine->layer_fds[layer_idx];
    uint16_t* out = (uint16_t*)output_f16;
    const uint16_t* x_in = (const uint16_t*)x_f16;

    /* Process each unique expert: pread once, compute all assigned tokens */
    for (int g = 0; g < num_unique_experts; g++) {
        int eidx = unique_experts[g];
        int start = group_starts[g];
        int end = group_starts[g + 1];
        int group_size_tokens = end - start;
        if (group_size_tokens <= 0) continue;

        /* pread this expert's weights ONCE */
        char* buf = engine->expert_bufs[0];  /* reuse buffer 0 */
        off_t offset = (off_t)HEADER_SIZE + (off_t)eidx * (off_t)engine->expert_size;
        ssize_t n = pread(fd, buf, engine->expert_size, offset);
        if (n != (ssize_t)engine->expert_size) return -1;

        /* Compute for each token assigned to this expert */
        /* Expert weights are HOT in L1/L2 cache for subsequent tokens */
        for (int i = start; i < end; i++) {
            int token_idx = sorted_token_indices[i];
            int k_pos = sorted_k_positions[i];

            /* Convert input f16 -> f32 */
            float x_f32[HIDDEN_SIZE];
            const uint16_t* x_tok = x_in + token_idx * HIDDEN_SIZE;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                x_f32[j] = half_to_float(x_tok[j]);
            }

            /* Compute expert MLP */
            float out_f32[HIDDEN_SIZE];
            if (engine->bits == 3)
                compute_single_expert_3bit(buf, x_f32, out_f32);
            else
                compute_single_expert(buf, x_f32, out_f32);

            /* Convert output f32 -> f16 and write to correct position */
            uint16_t* out_pos = out + (token_idx * K + k_pos) * HIDDEN_SIZE;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                out_pos[j] = float_to_half(out_f32[j]);
            }
        }
    }

    return 0;
}

/* ----------------------------------------------------------------------- */
/* bakan_cpu_expert_pread_batch — Batch pread unique experts into buffer    */
/*                                                                          */
/* Reads N unique experts in parallel (GCD), returns raw bytes in order.   */
/* Python then creates mx.arrays directly from the buffer — no per-expert  */
/* pread syscall from Python.                                               */
/* ----------------------------------------------------------------------- */
int bakan_cpu_expert_pread_batch(
    void* engine_ptr,
    int layer_idx,
    const int32_t* expert_ids,
    int num_experts_to_read,
    void* output_buffer  /* caller-allocated: num_experts * expert_size bytes */
) {
    KandigaCPUExpertEngine* engine = (KandigaCPUExpertEngine*)engine_ptr;
    if (!engine || layer_idx < 0 || layer_idx >= engine->num_layers) return -1;

    int fd = engine->layer_fds[layer_idx];
    char* out = (char*)output_buffer;
    size_t esz = engine->expert_size;

    /* Parallel pread via GCD */
    __block int err = 0;
    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    for (int i = 0; i < num_experts_to_read; i++) {
        int eidx = expert_ids[i];
        char* dst = out + (size_t)i * esz;
        dispatch_group_async(group, q, ^{
            off_t offset = (off_t)HEADER_SIZE + (off_t)eidx * (off_t)esz;
            ssize_t n = pread(fd, dst, esz, offset);
            if (n != (ssize_t)esz) err = 1;
        });
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    return err ? -1 : 0;
}
