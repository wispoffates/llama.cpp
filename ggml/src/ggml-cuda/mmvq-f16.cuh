#pragma once

#include "common.cuh"

// fp16 multi-column q6_K matvec for Pascal (MTP verify widths). Returns false if it does not handle
// the shape, in which case the caller runs the integer path. See mmvq-f16.cu.
bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, int64_t ncols);

// Forget the cached fp16 activation (called at the start of every graph compute, like the q8_1 cache).
void ggml_cuda_mmvq_f16_invalidate(ggml_backend_cuda_context & ctx);
