#pragma once

#include "common.cuh"

// fp16 one-column q4_0 / q8_0 matvec for Pascal (decode: dense and one-token MUL_MAT_ID). Returns false if it does not handle the
// shape, in which case the caller runs the integer path. See mmvq-h1.cu.
bool ggml_cuda_mmvq_h1_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst);

// Forget the cached fp16 activation (called at the start of every graph compute, like the q8_1 cache).
void ggml_cuda_mmvq_h1_invalidate(ggml_backend_cuda_context & ctx);

// rms_norm(x) * w for one row (decode) that also writes the fp16 activation cache entry for the h1 matvecs reading the result.
// nodes[i] = RMS_NORM, nodes[i + 1] = MUL (already checked fusable). Returns false if it does not apply.
bool ggml_cuda_mmvq_h1_add_rms_norm_mul_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
bool ggml_cuda_mmvq_h1_rms_norm_mul_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// out = res * scale[ids] per expert slot (MUL_MAT_ID per-expert weight scale), one launch.
void ggml_cuda_mmid_slot_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * res, const ggml_tensor * scale,
                               const ggml_tensor * ids, ggml_tensor * out);

// RMS_NORM -> SCALE -> MUL(w) for one row in one launch (Pascal). Returns false if it does not apply.
bool ggml_cuda_rms_norm_scale_mul_try(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * scl,
                                      ggml_tensor * mul);

// gate + up + GLU (GEGLU / SWIGLU) for one column in one launch, bit-identical to the three separate launches (Pascal, q4_0 / q8_0).
// _ok: whether the up / gate / glu nodes qualify; _try: run it (dst = the GLU node).
bool ggml_cuda_mmvq_h1_glu_ok(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu);
bool ggml_cuda_mmvq_h1_glu_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * gate_w, ggml_glu_op op, ggml_tensor * dst);

// gemma4 layer end + next layer start (RMS_NORM MUL ADD RMS_NORM MUL ADD MUL RMS_NORM MUL at nodes[i..i+8]) in one launch,
// bit-identical. Returns the number of nodes to skip (8) or 0. GGML_CUDA_FUSE_TAIL=0 disables.
int ggml_cuda_layer_tail_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// one-token rms_norm (no weight) -> view -> set_rows into the KV cache in one launch (Pascal). Returns false if it does not apply.
bool ggml_cuda_rms_norm_set_rows_try(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * view,
                                     const ggml_tensor * set_rows);
