#pragma once

#include "common.cuh"

// p100-series: per-device two-GPU AllReduce for the meta backend's threaded dispatch (see ar-rank.cu)
struct ggml_cuda_ar_rank_ctx;

ggml_cuda_ar_rank_ctx * ggml_cuda_ar_rank_init(int dev0, int dev1, size_t max_bytes);
void                    ggml_cuda_ar_rank_free(ggml_cuda_ar_rank_ctx * c);
bool                    ggml_cuda_ar_rank_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t);
void                    ggml_cuda_ar_rank_reduce(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t);
