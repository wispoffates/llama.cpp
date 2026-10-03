#pragma once

#include "common.cuh"

// Fused q6_K x f32 tile GEMM for Pascal at 9..80 columns (n-gram / draft verify batches), in place of
// dequantize -> fp16 -> cuBLAS hgemm. Returns false if it does not handle the shape. See tile-q6k.cu.
bool ggml_cuda_tile_q6_K_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
