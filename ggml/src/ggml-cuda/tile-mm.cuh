#pragma once

#include "common.cuh"

// Fused quantized x f32 tile GEMM for Pascal at 9..80 columns (n-gram / draft verify batches, short prompts), in place of
// dequantize -> fp16 -> cuBLAS hgemm. Handles q6_K and q8_0. Returns false if it does not handle the shape. See tile-mm.cu.
bool ggml_cuda_tile_mm_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
