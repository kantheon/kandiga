// TQ3_1S Metal compute shader — fused WHT + centroid dot product
//
// Computes matrix-vector product between TQ3-quantized weights and
// float16 activations directly in the WHT domain.
//
// Block format: 4 bytes (2x fp16 scales) + 12 bytes (packed 3-bit indices) = 16 bytes per 32 weights

#include <metal_stdlib>
using namespace metal;

// Lloyd-Max centroids for 3-bit quantization
constant float TQ3_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f,
};

// Deterministic sign pattern (golden ratio hash)
constant float TQ3_SIGNS[32] = {
    +1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, +1, -1, +1,
    -1, -1, +1, -1, +1, -1, -1, +1,
    -1, +1, +1, -1, +1, -1, -1, +1,
};

// TQ3_1S block: 16 bytes per 32 weights
struct TQ3Block {
    half d0;         // scale for elements 0-15
    half d1;         // scale for elements 16-31
    uchar qs[12];    // packed 3-bit indices
};

// Unpack 3-bit index from packed byte triplet
inline uint8_t tq3_unpack(uint32_t packed, int r) {
    return (packed >> (3 * r)) & 7;
}

// Apply WHT butterfly to 32 float values in-place (threadgroup shared memory)
inline void wht_butterfly_32(threadgroup float* buf, uint tid) {
    // 5 stages for n=32
    for (int step = 1; step < 32; step <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int pair = tid ^ step;
        if (pair > int(tid)) {
            float a = buf[tid];
            float b = buf[pair];
            buf[tid] = a + b;
            buf[pair] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Fused TQ3 matvec: compute one output element (dot product of one row with activation vector)
// Each threadgroup handles one row of the weight matrix
kernel void tq3_matvec(
    device const TQ3Block* weights [[buffer(0)]],  // quantized weight matrix
    device const half*     act     [[buffer(1)]],   // activation vector (pre-rotated)
    device       float*    out     [[buffer(2)]],   // output vector
    constant     uint&     ncols   [[buffer(3)]],   // number of columns (input dimension)
    uint gid     [[threadgroup_position_in_grid]],   // row index
    uint tid     [[thread_index_in_threadgroup]],    // thread within group
    uint tgsize  [[threads_per_threadgroup]]
) {
    const uint row = gid;
    const uint nblocks = ncols / 32;
    const uint blocks_per_row = nblocks;

    float sum = 0.0f;

    // Each thread processes multiple blocks
    for (uint b = tid; b < blocks_per_row; b += tgsize) {
        device const TQ3Block& blk = weights[row * blocks_per_row + b];
        const float scale0 = float(blk.d0);
        const float scale1 = float(blk.d1);

        // Unpack indices and compute dot product with activation
        for (int g = 0; g < 4; g++) {
            uint32_t packed = uint32_t(blk.qs[g*3])
                            | (uint32_t(blk.qs[g*3+1]) << 8)
                            | (uint32_t(blk.qs[g*3+2]) << 16);

            for (int r = 0; r < 8; r++) {
                int idx = g * 8 + r;
                uint8_t qi = tq3_unpack(packed, r);
                float w = TQ3_CENTROIDS[qi] * (idx < 16 ? scale0 : scale1);
                float a = float(act[b * 32 + idx]);
                sum += w * a;
            }
        }
    }

    // Warp reduction
    // Use threadgroup shared memory for reduction
    threadgroup float partial_sums[256];
    partial_sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[row] = partial_sums[0];
    }
}

// Pre-rotate activation vector into WHT domain
// This is done once per token, then reused across all weight rows
kernel void tq3_rotate_activations(
    device       float* act    [[buffer(0)]],  // activation vector (modified in-place)
    constant     uint&  n      [[buffer(1)]],  // vector length (must be multiple of 32)
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint block_start = gid * 32;
    if (block_start + 31 >= n) return;

    threadgroup float buf[32];

    // Load and apply sign flips
    buf[tid] = act[block_start + tid] * TQ3_SIGNS[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // WHT butterfly (5 stages)
    for (int step = 1; step < 32; step <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int pair = tid ^ step;
        if (pair > int(tid) && pair < 32) {
            float a = buf[tid];
            float b = buf[pair];
            buf[tid] = a + b;
            buf[pair] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize and write back
    act[block_start + tid] = buf[tid] / sqrt(32.0f);
}
