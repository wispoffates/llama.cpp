#include "ar-rank.cuh"

// p100-series: two-GPU AllReduce launched PER DEVICE, for the meta backend's threaded dispatch (one host thread per device).
//
// The threaded path's own AllReduce (peer copy + ADD) needs two host barriers per reduction: the copy may only be enqueued once
// the peer's partial is enqueued, the ADD only once the peer's pull is enqueued. On gemma4 decode the GPU then idles ~37 us per
// reduction waiting for the host threads (60 reductions per token). Here each device's thread enqueues one kernel and moves on;
// the GPUs synchronize among themselves:
//   phase 1: write my partial into the PEER's inbox (P2P stores into its device memory)
//   phase 2: thread 0 of each block writes the call token into the peer's arrival slot, then spins on its OWN slot (local read)
//   phase 3: sum = mine + inbox (my device memory), in place
// Both devices compute every element (x0 + x1 on device 0, x1 + x0 on device 1: the same IEEE result), exactly what the copy +
// ADD path computes. Two inbox / arrival slots alternate by call parity: a device can only start call N after it saw the peer's
// token for N - 1, which the peer writes after its call N - 2 kernel (the last reader of slot N % 2) has completed.
// Based on ggml-cuda/allreduce.cu's chunked kernel (pinned-host staging, sm70+ for __nanosleep); this one is P2P-only and spins
// with a plain volatile read (Pascal).

static constexpr int    AR_RANK_SLOTS        = 2;
static constexpr int    AR_RANK_MAX_BLOCKS   = 8;
static constexpr int    AR_RANK_THREADS      = 256;
static constexpr size_t AR_RANK_STRIDE       = 64; // bytes between arrival ints (one cache line each)

struct ggml_cuda_ar_rank_ctx {
    int      devices[2];
    size_t   max_bytes;
    char   * inbox[2];    // device i: AR_RANK_SLOTS * max_bytes, written by the peer
    char   * arrival[2];  // device i: AR_RANK_SLOTS * AR_RANK_MAX_BLOCKS * AR_RANK_STRIDE, written by the peer
    uint32_t calls[2];    // per rank; both ranks issue the same sequence of reductions
};

static __global__ void ggml_cuda_ar_rank_kernel(float * data, const int n, float * __restrict__ wire_out, const float * __restrict__ wire_in,
                                                int * signal_out, const int * signal_in, const int token) {
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const int n4   = n / 4;
    const bool vec = ((uintptr_t) data & 15) == 0;

    // phase 1: my partial -> peer inbox (groups of 4 elements striped over the grid; tail by block 0)
    for (int i = gtid; i < n4; i += gnt) {
        float4 v;
        if (vec) {
            v = ((const float4 *) data)[i];
        } else {
            v = make_float4(data[4*i + 0], data[4*i + 1], data[4*i + 2], data[4*i + 3]);
        }
        ((float4 *) wire_out)[i] = v;
    }
    if (bid == 0 && tid < n - 4*n4) {
        wire_out[4*n4 + tid] = data[4*n4 + tid];
    }
    __threadfence_system();
    __syncthreads();

    // phase 2: per-block handshake (the peer's block bid reads exactly the stripe our block bid wrote)
    if (tid == 0) {
        *(volatile int *) (signal_out + bid * (AR_RANK_STRIDE / sizeof(int))) = token;
        __threadfence_system();
        const volatile int * in = (const volatile int *) (signal_in + bid * (AR_RANK_STRIDE / sizeof(int)));
        while (*in != token) {
        }
    }
    __syncthreads();
    __threadfence();

    // phase 3: in-place sum with the peer's partial
    for (int i = gtid; i < n4; i += gnt) {
        const float4 o = __ldcg((const float4 *) wire_in + i);
        if (vec) {
            float4 v = ((float4 *) data)[i];
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
            ((float4 *) data)[i] = v;
        } else {
            data[4*i + 0] += o.x; data[4*i + 1] += o.y; data[4*i + 2] += o.z; data[4*i + 3] += o.w;
        }
    }
    if (bid == 0 && tid < n - 4*n4) {
        data[4*n4 + tid] += __ldcg(wire_in + 4*n4 + tid);
    }
}

ggml_cuda_ar_rank_ctx * ggml_cuda_ar_rank_init(int dev0, int dev1, size_t max_bytes) {
    if (getenv("GGML_CUDA_P2P") == nullptr) {
        return nullptr; // peer access is only enabled (ggml_cuda_init) under GGML_CUDA_P2P
    }
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    if (!can01 || !can10) {
        return nullptr;
    }
    auto * c = new ggml_cuda_ar_rank_ctx{};
    c->devices[0] = dev0;
    c->devices[1] = dev1;
    c->max_bytes  = max_bytes;
    const size_t arrival_bytes = (size_t) AR_RANK_SLOTS * AR_RANK_MAX_BLOCKS * AR_RANK_STRIDE;
    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(c->devices[i]);
        ok = cudaMalloc((void **) &c->inbox[i], AR_RANK_SLOTS * max_bytes) == cudaSuccess &&
             cudaMalloc((void **) &c->arrival[i], arrival_bytes) == cudaSuccess &&
             cudaMemset(c->arrival[i], 0, arrival_bytes) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        ggml_cuda_ar_rank_free(c);
        return nullptr;
    }
    return c;
}

void ggml_cuda_ar_rank_free(ggml_cuda_ar_rank_ctx * c) {
    if (c == nullptr) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(c->devices[i]);
        if (c->inbox[i])   { CUDA_CHECK(cudaFree(c->inbox[i])); }
        if (c->arrival[i]) { CUDA_CHECK(cudaFree(c->arrival[i])); }
    }
    delete c;
}

bool ggml_cuda_ar_rank_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t) {
    return c != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t) <= c->max_bytes && ggml_nelements(t) > 0;
}

void ggml_cuda_ar_rank_reduce(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t) {
    const int      peer  = rank ^ 1;
    const uint32_t token = ++c->calls[rank];
    const int      slot  = token % AR_RANK_SLOTS;
    const int      n     = (int) ggml_nelements(t);
    const int      n4    = n / 4;
    const int      nb    = std::max(1, std::min(AR_RANK_MAX_BLOCKS, (n4 + AR_RANK_THREADS - 1) / AR_RANK_THREADS));

    float * wire_out   = (float *) (c->inbox[peer] + (size_t) slot * c->max_bytes);
    float * wire_in    = (float *) (c->inbox[rank] + (size_t) slot * c->max_bytes);
    int   * signal_out = (int *) (c->arrival[peer] + (size_t) slot * AR_RANK_MAX_BLOCKS * AR_RANK_STRIDE);
    int   * signal_in  = (int *) (c->arrival[rank] + (size_t) slot * AR_RANK_MAX_BLOCKS * AR_RANK_STRIDE);

    ggml_cuda_ar_rank_kernel<<<nb, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, wire_out, wire_in, signal_out, signal_in, (int) token);
    CUDA_CHECK(cudaGetLastError());
}
