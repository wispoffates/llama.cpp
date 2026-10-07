#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

// Radix select: 4 histogram passes over the row, then one gather. Also used on CUDA when CUB has
// no DeviceTopK (CCCL < 3.2, e.g. CUDA 12.0): the fallback there is a full segmented radix sort
// of the row, which for a 248k-entry vocabulary is the bulk of a speculative draft step's sampling.
#if !defined(CUB_TOP_K_AVAILABLE)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

#endif // !defined(CUB_TOP_K_AVAILABLE)

// p100-series 10-07: short rows (<= 2048 values, e.g. qwen4exp's attention indexer picking pools at decode) in ONE block per row:
// 4-pass 8-bit radix select of the k-th largest key in shared memory, then a gather. The bitonic argsort it replaces sorts the
// whole padded row (1024 threads, ~54 us on a P100 for one decode row). Output is the top-k set (GGML_OP_TOP_K is unordered):
// keys above the threshold in index order, then threshold ties in index order (deterministic).
#define TOPK_BLOCK_THREADS 1024
#define TOPK_BLOCK_E       2      // values per thread

static __device__ __forceinline__ uint32_t topk_block_key(const float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// exclusive block scan of one flag per thread; returns this thread's offset, *total = block sum
static __device__ __forceinline__ int topk_block_scan(const bool flag, int * wsum, int * total) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const uint32_t m = __ballot_sync(0xffffffffu, flag);
    const int in_warp = __popc(m & ((1u << lane) - 1u));
    if (lane == 0) {
        wsum[wid] = __popc(m);
    }
    __syncthreads();
    if (wid == 0) {
        const int v = wsum[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, s, o);
            s += lane >= o ? t : 0;
        }
        wsum[lane] = s - v;               // exclusive
        if (lane == 31) {
            *total = s;
        }
    }
    __syncthreads();
    const int r = wsum[wid] + in_warp;
    __syncthreads();                      // wsum is reused by the next call
    return r;
}

static __global__ void __launch_bounds__(TOPK_BLOCK_THREADS) top_k_block(const float * __restrict__ src, int * __restrict__ dst,
                                                                         const int ncols, const int k) {
    __shared__ int hist[256];
    __shared__ int wsum[32];
    __shared__ int s_total;
    __shared__ uint32_t s_prefix;
    __shared__ int s_krem;
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31;
    src += (int64_t) row*ncols;
    dst += (int64_t) row*k;

    uint32_t key[TOPK_BLOCK_E];
#pragma unroll
    for (int e = 0; e < TOPK_BLOCK_E; ++e) {
        const int c = t + e*TOPK_BLOCK_THREADS;
        key[e] = c < ncols ? topk_block_key(src[c]) : 0u;
    }
    if (t == 0) {
        s_prefix = 0;
        s_krem = k;
    }
    uint32_t pmask = 0;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) {
            hist[t] = 0;
        }
        __syncthreads();
        const uint32_t prefix = s_prefix;
#pragma unroll
        for (int e = 0; e < TOPK_BLOCK_E; ++e) {
            const int c = t + e*TOPK_BLOCK_THREADS;
            if (c < ncols && (key[e] & pmask) == prefix) {
                atomicAdd(&hist[(key[e] >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (t < 32) {
            // lane l owns bins 255-8l .. 248-8l (descending); find the bin where the running count from the top reaches krem
            const int krem = s_krem;
            int c8[8], sl = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c8[i] = hist[255 - 8*lane - i];
                sl += c8[i];
            }
            int incl = sl;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                incl += lane >= o ? v : 0;
            }
            const int excl = incl - sl;
            const uint32_t hit = __ballot_sync(0xffffffffu, excl < krem && incl >= krem);
            if (lane == __ffs(hit) - 1) {
                int run = excl;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (run + c8[i] >= krem) {
                        s_prefix = prefix | ((uint32_t) (255 - 8*lane - i) << shift);
                        s_krem = krem - run;
                        break;
                    }
                    run += c8[i];
                }
            }
        }
        pmask |= 255u << shift;
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    const int ntie = s_krem;   // ties at the threshold to take
    int base_gt = 0, base_tie = 0;
#pragma unroll
    for (int e = 0; e < TOPK_BLOCK_E; ++e) {
        const int c = t + e*TOPK_BLOCK_THREADS;
        const bool gt = c < ncols && key[e] > thr;
        const int off = topk_block_scan(gt, wsum, &s_total);
        if (gt) {
            dst[base_gt + off] = c;
        }
        base_gt += s_total;
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < TOPK_BLOCK_E; ++e) {
        const int c = t + e*TOPK_BLOCK_THREADS;
        const bool tie = c < ncols && key[e] == thr;
        const int off = topk_block_scan(tie, wsum, &s_total);
        if (tie && base_tie + off < ntie) {
            dst[base_gt + base_tie + off] = c;
        }
        base_tie += s_total;
        __syncthreads();
    }
    GGML_UNUSED(lane);
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    // GGML_CUDA_TOPK_BLOCK=0: short rows back to the sort-based paths below
    static const bool block_on = [] { const char * e = getenv("GGML_CUDA_TOPK_BLOCK"); return !e || atoi(e) != 0; }();
    if (block_on && ncols <= TOPK_BLOCK_THREADS*TOPK_BLOCK_E && k >= 1 && k <= ncols) {
        top_k_block<<<(unsigned) nrows, TOPK_BLOCK_THREADS, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    static const bool radix = [] {
        const char * e = getenv("GGML_CUDA_TOPK_RADIX");
        return !e || atoi(e) != 0;
    }();
    if (radix && ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
