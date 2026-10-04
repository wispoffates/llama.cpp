#pragma once

#include "common.cuh"

// fp16 one-column q4_0 matvec for Pascal (decode: dense and one-token MUL_MAT_ID). Returns false if it does not handle the
// shape, in which case the caller runs the integer path. See mmvq-h1.cu.
bool ggml_cuda_mmvq_h1_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst);

// Forget the cached fp16 activation (called at the start of every graph compute, like the q8_1 cache).
void ggml_cuda_mmvq_h1_invalidate(ggml_backend_cuda_context & ctx);
