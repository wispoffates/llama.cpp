#pragma once

#include "common.cuh"

// fp32 one-column matvec for low-bit and lookup-grid types on Pascal (Q2_0, IQ1_M, IQ2_XXS, IQ2_S, IQ3_S, IQ4_NL, IQ4_XS; decode:
// dense and one-token MUL_MAT_ID). Returns false if it does not handle the shape, in which case the caller runs the integer path.
// See mmvq-lb.cu.
bool ggml_cuda_mmvq_lb_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst);
