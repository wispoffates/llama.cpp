#include "moe-weighted-reduction.cuh"

// p100-series: products and sums are rounded explicitly (no FMA contraction), in the order of the unfused graph -- mul (experts *
// weights), then views and a left-to-right chain of adds -- so the fused kernel gives the same bits as the fallback path. Which path
// runs depends on buffer placement (graph_optimize, which adds the alloc deps, never runs under -sm tensor), so this keeps the result
// independent of it.
static __device__ __forceinline__ float moe_wr_term(const float e, const float * __restrict__ expert_scale, const float w, const uint64_t row) {
    return __fmul_rn(expert_scale != nullptr ? __fmul_rn(e, expert_scale[row]) : e, w);
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row = (uint64_t) token * n_expert_used;
    float          sum       = moe_wr_term(experts[first_row * n_embd + col], expert_scale, weights[first_row], first_row);

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        sum = __fadd_rn(sum, moe_wr_term(experts[row * n_embd + col], expert_scale, weights[row], row));
    }
    dst[token * n_embd + col] = sum;
}

// p100-series: one token, one block. The router weights (and expert scales) are staged in shared memory before any thread writes,
// so they may alias dst -- the allocator reuses their buffer for dst once the unfused graph's first mul has read them, and without
// graph_optimize's alloc deps (-sm tensor) that is the common case (gemma4 decode: the fusion was rejected 93 % of the time).
static __global__ void moe_weighted_reduction_f32_1tok(const float * __restrict__ experts,
                                                       const float * expert_scale,
                                                       const float * weights,
                                                       float * dst,
                                                       const int64_t n_embd,
                                                       const int     n_expert_used) {
    __shared__ float w_s[32];
    __shared__ float s_s[32];
    if (threadIdx.x < n_expert_used) {
        w_s[threadIdx.x] = weights[threadIdx.x];
        s_s[threadIdx.x] = expert_scale != nullptr ? expert_scale[threadIdx.x] : 1.0f;
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < n_embd; col += blockDim.x) {
        float sum = __fmul_rn(expert_scale != nullptr ? __fmul_rn(experts[col], s_s[0]) : experts[col], w_s[0]);
        for (int expert = 1; expert < n_expert_used; ++expert) {
            const float e = experts[(int64_t) expert * n_embd + col];
            sum = __fadd_rn(sum, __fmul_rn(expert_scale != nullptr ? __fmul_rn(e, s_s[expert]) : e, w_s[expert]));
        }
        dst[col] = sum;
    }
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    if (n_tokens == 1) {
        GGML_ASSERT(n_expert_used <= 32);
        moe_weighted_reduction_f32_1tok<<<1, 1024, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        return;
    }
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
