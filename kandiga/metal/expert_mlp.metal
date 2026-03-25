// expert_mlp.metal — Optimized Metal compute shaders for MoE expert MLP
//
// Two kernels for 4-bit quantized expert computation:
//   1. expert_up_gate_swiglu: up_proj + gate_proj + SwiGLU activation
//   2. expert_down_proj: down_proj from activated intermediates
//
// Optimizations:
//   - Threadgroup shared memory for input vector (eliminates redundant device reads)
//   - FMA-optimized dequantization (fused multiply-add chains)
//   - Unrolled 8-nibble extraction per uint32
//   - Model-agnostic: reads dimensions from params buffer

#include <metal_stdlib>
using namespace metal;







// ---------------------------------------------------------------------------
// Parameter buffer — matches ExpertMLPParams in bakan_metal.m
// ---------------------------------------------------------------------------
struct ExpertMLPParams {
    int32_t expert_indices[16]; // max 16 active experts
    int32_t num_experts;
    int32_t hidden_size;        // e.g., 2048 (35B) or 4096 (397B)
    int32_t expert_dim;         // e.g., 512  (35B) or varies (397B)
    int32_t group_size;         // 64
    uint64_t header_size;       // 0 when using staging buffer
    uint64_t expert_size;       // bytes per expert block
    // Tensor byte offsets within each expert block
    uint64_t gate_weight_offset;
    uint64_t gate_scales_offset;
    uint64_t gate_biases_offset;
    uint64_t up_weight_offset;
    uint64_t up_scales_offset;
    uint64_t up_biases_offset;
    uint64_t down_weight_offset;
    uint64_t down_scales_offset;
    uint64_t down_biases_offset;
};

// ---------------------------------------------------------------------------
// Dequant dot product — 4-bit packed weights × half input
// Uses FMA chains: acc = fma(nibble, scale*x[i], bias*x[i] + acc)
// ---------------------------------------------------------------------------
inline float dequant_dot_half(
    device const uint32_t* weight_row,
    threadgroup const half* x_shared,
    device const bfloat*   scales_row,
    device const bfloat*   biases_row,
    uint in_dim,
    uint group_size
) {
    float acc = 0.0f;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    for (uint g = 0; g < num_groups; g++) {
        float scale = float(scales_row[g]);
        float bias  = float(biases_row[g]);

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            uint x_base = g * group_size + p * 8;

            // Pre-load 8 x values and pre-multiply with scale/bias
            float x0 = float(x_shared[x_base + 0]);
            float x1 = float(x_shared[x_base + 1]);
            float x2 = float(x_shared[x_base + 2]);
            float x3 = float(x_shared[x_base + 3]);
            float x4 = float(x_shared[x_base + 4]);
            float x5 = float(x_shared[x_base + 5]);
            float x6 = float(x_shared[x_base + 6]);
            float x7 = float(x_shared[x_base + 7]);

            // FMA chain: acc += (nibble * scale + bias) * x
            acc = fma(fma(float((packed      ) & 0xF), scale, bias), x0, acc);
            acc = fma(fma(float((packed >>  4) & 0xF), scale, bias), x1, acc);
            acc = fma(fma(float((packed >>  8) & 0xF), scale, bias), x2, acc);
            acc = fma(fma(float((packed >> 12) & 0xF), scale, bias), x3, acc);
            acc = fma(fma(float((packed >> 16) & 0xF), scale, bias), x4, acc);
            acc = fma(fma(float((packed >> 20) & 0xF), scale, bias), x5, acc);
            acc = fma(fma(float((packed >> 24) & 0xF), scale, bias), x6, acc);
            acc = fma(fma(float((packed >> 28) & 0xF), scale, bias), x7, acc);
        }
    }
    return acc;
}

// Variant with float input (for down_proj using activated intermediates)
inline float dequant_dot_float(
    device const uint32_t* weight_row,
    device const float*    x_row,
    device const bfloat*   scales_row,
    device const bfloat*   biases_row,
    uint in_dim,
    uint group_size
) {
    float acc = 0.0f;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    for (uint g = 0; g < num_groups; g++) {
        float scale = float(scales_row[g]);
        float bias  = float(biases_row[g]);

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = weight_row[g * packed_per_group + p];
            uint x_base = g * group_size + p * 8;

            float x0 = x_row[x_base + 0];
            float x1 = x_row[x_base + 1];
            float x2 = x_row[x_base + 2];
            float x3 = x_row[x_base + 3];
            float x4 = x_row[x_base + 4];
            float x5 = x_row[x_base + 5];
            float x6 = x_row[x_base + 6];
            float x7 = x_row[x_base + 7];

            acc = fma(fma(float((packed      ) & 0xF), scale, bias), x0, acc);
            acc = fma(fma(float((packed >>  4) & 0xF), scale, bias), x1, acc);
            acc = fma(fma(float((packed >>  8) & 0xF), scale, bias), x2, acc);
            acc = fma(fma(float((packed >> 12) & 0xF), scale, bias), x3, acc);
            acc = fma(fma(float((packed >> 16) & 0xF), scale, bias), x4, acc);
            acc = fma(fma(float((packed >> 20) & 0xF), scale, bias), x5, acc);
            acc = fma(fma(float((packed >> 24) & 0xF), scale, bias), x6, acc);
            acc = fma(fma(float((packed >> 28) & 0xF), scale, bias), x7, acc);
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Kernel 1: expert_up_gate_swiglu
//
// Uses threadgroup shared memory for input x to avoid redundant device reads.
// Grid: (expert_dim, num_experts, 1)
// Threadgroup: (min(expert_dim, 256), 1, 1)
// ---------------------------------------------------------------------------
kernel void expert_up_gate_swiglu(
    device const uint8_t*  layer_data   [[buffer(0)]],
    device const half*     x            [[buffer(1)]],
    device       float*    activated    [[buffer(2)]],
    device const ExpertMLPParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 tpg [[thread_position_in_threadgroup]],
    uint2 tgs [[threads_per_threadgroup]]
) {
    uint row       = tid.x;
    uint expert_k  = tid.y;

    if (expert_k >= uint(params.num_experts) || row >= uint(params.expert_dim))
        return;

    // Load input x into threadgroup shared memory (cooperative load)
    threadgroup half x_shared[4096]; // max hidden_size
    uint in_dim = uint(params.hidden_size);
    uint lid = tpg.x;
    uint tg_size = tgs.x;
    for (uint i = lid; i < in_dim; i += tg_size) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int expert_idx = params.expert_indices[expert_k];
    uint64_t expert_base = params.header_size + uint64_t(expert_idx) * params.expert_size;

    uint group_size = uint(params.group_size);
    uint weight_row_stride = in_dim / 8;
    uint sb_row_stride = in_dim / group_size;

    // gate_proj
    device const uint32_t* gate_w = (device const uint32_t*)(layer_data + expert_base + params.gate_weight_offset) + row * weight_row_stride;
    device const bfloat* gate_s = (device const bfloat*)(layer_data + expert_base + params.gate_scales_offset) + row * sb_row_stride;
    device const bfloat* gate_b = (device const bfloat*)(layer_data + expert_base + params.gate_biases_offset) + row * sb_row_stride;
    float gate_val = dequant_dot_half(gate_w, x_shared, gate_s, gate_b, in_dim, group_size);

    // up_proj
    device const uint32_t* up_w = (device const uint32_t*)(layer_data + expert_base + params.up_weight_offset) + row * weight_row_stride;
    device const bfloat* up_s = (device const bfloat*)(layer_data + expert_base + params.up_scales_offset) + row * sb_row_stride;
    device const bfloat* up_b = (device const bfloat*)(layer_data + expert_base + params.up_biases_offset) + row * sb_row_stride;
    float up_val = dequant_dot_half(up_w, x_shared, up_s, up_b, in_dim, group_size);

    // SwiGLU: silu(gate) * up
    float silu_gate = gate_val / (1.0f + exp(-gate_val));
    activated[expert_k * params.expert_dim + row] = silu_gate * up_val;
}

// ---------------------------------------------------------------------------
// Kernel 2: expert_down_proj
//
// Grid: (hidden_size, num_experts, 1)
// Threadgroup: (min(hidden_size, 256), 1, 1)
// ---------------------------------------------------------------------------
kernel void expert_down_proj(
    device const uint8_t*  layer_data   [[buffer(0)]],
    device const float*    activated    [[buffer(1)]],
    device       float*    output       [[buffer(2)]],
    device const ExpertMLPParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint row       = tid.x;
    uint expert_k  = tid.y;

    if (expert_k >= uint(params.num_experts) || row >= uint(params.hidden_size))
        return;

    int expert_idx = params.expert_indices[expert_k];
    uint64_t expert_base = params.header_size + uint64_t(expert_idx) * params.expert_size;

    uint in_dim = uint(params.expert_dim);
    uint group_size = uint(params.group_size);
    uint weight_row_stride = in_dim / 8;
    uint sb_row_stride = in_dim / group_size;

    device const uint32_t* down_w = (device const uint32_t*)(layer_data + expert_base + params.down_weight_offset) + row * weight_row_stride;
    device const bfloat* down_s = (device const bfloat*)(layer_data + expert_base + params.down_scales_offset) + row * sb_row_stride;
    device const bfloat* down_b = (device const bfloat*)(layer_data + expert_base + params.down_biases_offset) + row * sb_row_stride;

    device const float* act_row = activated + expert_k * in_dim;

    float acc = dequant_dot_float(down_w, act_row, down_s, down_b, in_dim, group_size);
    output[expert_k * params.hidden_size + row] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 3: expert_mlp_fused
//
// Fuses up_proj + gate_proj + SwiGLU + down_proj into ONE kernel per expert.
// The intermediate activated values stay in threadgroup shared memory,
// eliminating one dispatch + one barrier + one device memory write/read.
//
// Each threadgroup handles one expert (K threadgroups total).
// 512 threads per threadgroup = one thread per expert_dim element.
//
// Phase 1: Each thread computes one element of gate/up projection + SwiGLU
//           and stores to shared memory.
// Phase 2: After barrier, each thread computes 4 rows of down_proj
//           (2048 output / 512 threads = 4 rows/thread), reading from shared.
//
// Shared memory usage:
//   - x_shared: half[2048] = 4KB (input vector, cooperatively loaded)
//   - shared_activated: float[512] = 2KB (intermediate after SwiGLU)
//   Total: 6KB per threadgroup (well within 32KB limit)
//
// Dispatch: threadgroups=(K, 1, 1), threadsPerThreadgroup=(512, 1, 1)
// ---------------------------------------------------------------------------
kernel void expert_mlp_fused(
    device const uint8_t*  layer_data   [[buffer(0)]],
    device const half*     x            [[buffer(1)]],
    device       float*    output       [[buffer(2)]],   // float[K][hidden_size]
    device const ExpertMLPParams& params [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],           // expert_k: 0..K-1
    uint tid  [[thread_index_in_threadgroup]]              // 0..511
) {
    uint expert_k = tgid;
    if (expert_k >= uint(params.num_experts)) return;

    // --- Cooperative load of input x into shared memory ---
    threadgroup half x_shared[4096]; // max hidden_size (3072 for 122B, 4096 headroom)
    uint in_dim = uint(params.hidden_size);
    uint tg_size = uint(params.expert_dim); // threads per threadgroup = expert_dim
    for (uint i = tid; i < in_dim; i += tg_size) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int expert_idx = params.expert_indices[expert_k];
    uint64_t expert_base = params.header_size + uint64_t(expert_idx) * params.expert_size;

    uint group_size = uint(params.group_size);
    uint weight_row_stride = in_dim / 8;
    uint sb_row_stride = in_dim / group_size;

    // === Phase 1: up_proj + gate_proj + SwiGLU ===
    // Each thread handles one row (one element of expert_dim)
    uint row = tid;
    threadgroup float shared_activated[1024]; // expert_dim for 122B

    if (row < uint(params.expert_dim)) {
        // gate_proj dot product
        device const uint32_t* gate_w = (device const uint32_t*)(layer_data + expert_base + params.gate_weight_offset) + row * weight_row_stride;
        device const bfloat* gate_s = (device const bfloat*)(layer_data + expert_base + params.gate_scales_offset) + row * sb_row_stride;
        device const bfloat* gate_b = (device const bfloat*)(layer_data + expert_base + params.gate_biases_offset) + row * sb_row_stride;
        float gate_val = dequant_dot_half(gate_w, x_shared, gate_s, gate_b, in_dim, group_size);

        // up_proj dot product
        device const uint32_t* up_w = (device const uint32_t*)(layer_data + expert_base + params.up_weight_offset) + row * weight_row_stride;
        device const bfloat* up_s = (device const bfloat*)(layer_data + expert_base + params.up_scales_offset) + row * sb_row_stride;
        device const bfloat* up_b = (device const bfloat*)(layer_data + expert_base + params.up_biases_offset) + row * sb_row_stride;
        float up_val = dequant_dot_half(up_w, x_shared, up_s, up_b, in_dim, group_size);

        // SwiGLU activation
        float silu_gate = gate_val / (1.0f + exp(-gate_val));
        shared_activated[row] = silu_gate * up_val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Phase 2: down_proj ===
    // expert_dim threads handle hidden_size output rows
    uint hidden_size = uint(params.hidden_size);
    uint expert_dim = uint(params.expert_dim);
    uint rows_per_thread = (hidden_size + tg_size - 1) / tg_size;

    uint dw_row_stride = expert_dim / 8;          // 512/8 = 64
    uint dsb_row_stride = expert_dim / group_size; // 512/64 = 8

    device const uint32_t* dw_base = (device const uint32_t*)(layer_data + expert_base + params.down_weight_offset);
    device const bfloat* ds_base = (device const bfloat*)(layer_data + expert_base + params.down_scales_offset);
    device const bfloat* db_base = (device const bfloat*)(layer_data + expert_base + params.down_biases_offset);

    uint num_groups = expert_dim / group_size;     // 512/64 = 8
    uint packed_per_group = group_size / 8;        // 64/8 = 8

    for (uint r = 0; r < rows_per_thread; r++) {
        uint out_row = tid * rows_per_thread + r;
        if (out_row >= hidden_size) break;

        device const uint32_t* w_row = dw_base + out_row * dw_row_stride;
        device const bfloat* s_row = ds_base + out_row * dsb_row_stride;
        device const bfloat* b_row = db_base + out_row * dsb_row_stride;

        float acc = 0.0f;
        for (uint g = 0; g < num_groups; g++) {
            float scale = float(s_row[g]);
            float bias = float(b_row[g]);
            for (uint p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[g * packed_per_group + p];
                uint x_base = g * group_size + p * 8;

                // Unrolled 8-nibble FMA chain reading from shared memory
                acc = fma(fma(float((packed      ) & 0xF), scale, bias), shared_activated[x_base + 0], acc);
                acc = fma(fma(float((packed >>  4) & 0xF), scale, bias), shared_activated[x_base + 1], acc);
                acc = fma(fma(float((packed >>  8) & 0xF), scale, bias), shared_activated[x_base + 2], acc);
                acc = fma(fma(float((packed >> 12) & 0xF), scale, bias), shared_activated[x_base + 3], acc);
                acc = fma(fma(float((packed >> 16) & 0xF), scale, bias), shared_activated[x_base + 4], acc);
                acc = fma(fma(float((packed >> 20) & 0xF), scale, bias), shared_activated[x_base + 5], acc);
                acc = fma(fma(float((packed >> 24) & 0xF), scale, bias), shared_activated[x_base + 6], acc);
                acc = fma(fma(float((packed >> 28) & 0xF), scale, bias), shared_activated[x_base + 7], acc);
            }
        }

        output[expert_k * hidden_size + out_row] = acc;
    }
}
