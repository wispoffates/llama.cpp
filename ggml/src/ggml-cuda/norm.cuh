#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx,
                       ggml_tensor *               dst,
                       ggml_tensor *               mul = nullptr,
                       ggml_tensor *               add = nullptr);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// dst = silu(z) * (rms_norm(x) * w), x = rms_norm->src[0]; sigmoid(z) with sigmoid_gate; z may be strided
void ggml_cuda_op_rms_norm_mul_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * w,
                                     const ggml_tensor * z, ggml_tensor * dst, bool sigmoid_gate = false);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// p100-series: residual ADD -> RMS_NORM -> MUL(w) in one launch (bit-identical); false = not applicable, nothing launched
bool ggml_cuda_add_rms_norm_mul_ok(const ggml_tensor * add, const ggml_tensor * rms, const ggml_tensor * mul);
bool ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul);

// p100-series: the qwen35moe / qwen4exp shared-expert tail, x = (a + s*sigmoid(g[row])) + r = the unfused SIGMOID, MUL (one gate
// value per row, broadcast), ADD (moe_out), ADD (residual), with exactly their roundings. sum_out = x (the next residual).
// Pointers and row strides in floats.
struct ggml_cuda_shexp_gate {
    const float * a;
    const float * s;
    const float * g;
    const float * r;
    float       * sum_out;
    int64_t sa, ss, sg, sr, so;
    int64_t ncols, nrows;
};
// x only (no norm after it): one launch instead of three
void ggml_cuda_op_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_gate & p);
// x, then the next pre-norm rms_norm(x) * w (add_rms_norm_mul_f32's register path); false = not applicable, nothing launched
bool ggml_cuda_op_shexp_gate_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_cuda_shexp_gate & p, const ggml_tensor * rms,
                                          ggml_tensor * mul);
