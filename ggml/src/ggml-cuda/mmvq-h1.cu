#include "mmvq-h1.cuh"
#include "norm.cuh"
#include "unary.cuh"
#include "convert.cuh"
#include "ggml-impl.h"

#include <unordered_map>

// q4_0 / q8_0 x f32 matvec for ONE column on Pascal, in fp16 (decode: dense matmuls and the one-token MUL_MAT_ID of MoE experts).
//
// The integer path (mmvq.cu) multiplies against q8_1 activations with DP4A, which sm_60 lacks; its emulation costs several
// instructions per multiply-add and, together with the shared-memory staging, holds a big matvec at 300-350 GB/s of the ~570 a
// linear read gets. Here every weight pair becomes a half2 with one PRMT and one HSUB2 (the 0x64nn = 1024 + n trick, exact) and
// meets the activation in one HFMA2.
//
// Layout: a warp owns RPW rows. For each 32-block chunk of K (576 bytes of a row) lane l owns block l: it keeps that block's 32
// activation values in registers (16 half2) for all RPW rows. The rows' chunks are staged U at a time through a per-warp shared
// buffer with coalesced 16-byte loads from the 16-byte-aligned address at or below the chunk (rows need only 2-byte alignment;
// the staging reads up to 15 bytes past a chunk, which the quantized-tensor row padding covers -- see the guard), and each lane
// decodes its block from there. Per-row partials stay in registers; a butterfly whose last log2(RPW) steps are transposed leaves
// row r's sum in lane r. One launch, no K split, deterministic.
//
// Numerics: per 32-value block the activation is divided by its max |x| (fp32 scale s), so |x'| <= 1 and the block's fp16 chain
// of 16 HFMA2 stays <= 256; x' is rounded to 11 bits (q8_1: 8 bits), the weights are exact, and the block sum is folded into fp32
// as d*s*sum. Harness (bench-tools/tilegemm/gemv4.cu, random data): relative error ~6e-4 against an fp32 reference, the fp16
// rounding of x'.
//
// Harness timings (one P100, activation prep included), rows x K: 131072 x 2816 (gemma4 lm_head per GPU) 690 -> 402 us,
// 4096 x 14336 95 -> 70 us, 2816 x 2048 12.3 -> 11.1 us, 2048 x 2816 12.5 -> 13.2 us (small shapes are launch-bound either way).

#define MMVQ_H1_WPB 4   // warps per block (independent row groups)
#define MMVQ_H1_U   2   // rows staged at a time
#define MMVQ_H1_RPW 4   // rows per warp

static constexpr int H1_QK = 32;   // values per block (q4_0 and q8_0)

// per 32-value block: s = max|x| (1 if 0), x' = x/s as half. One thread per value, grid (ceil(K/256), rows of x).
static __global__ void mmvq_h1_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XH, float * __restrict__ XS,
                                    const int K) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= K) {
        return;   // K % 32 == 0: whole warps leave together
    }
    const int64_t c = blockIdx.y;
    const float v = X[c*sx + i];
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    }
    const float s = m > 0.0f ? m : 1.0f;
    XH[c*K + i] = __float2half(v/s);
    if ((i & 31) == 0) {
        XS[c*(K/32) + i/32] = s;
    }
}

// 4 bytes of nibbles (already masked into the low half of each byte) -> two half2 of n - 8, exact
static __device__ __forceinline__ void mmvq_h1_nib(const uint32_t b, half2 & h01, half2 & h23) {
    const uint32_t m = 0x64646464u;
    uint32_t r01 = __byte_perm(b, m, 0x4140), r23 = __byte_perm(b, m, 0x4342);   // 0x64nn = 1024 + n
    const half2 off = __halves2half2(__ushort_as_half(0x6408), __ushort_as_half(0x6408));   // 1032
    h01 = __hsub2(*(half2 *) &r01, off);
    h23 = __hsub2(*(half2 *) &r23, off);
}

// int8 bytes -> two half2, exact: q ^ 0x80 = q + 128 in 0..255, 0x64uu = 1024 + u, minus 1152
static __device__ __forceinline__ void mmvq_h1_i8(const uint32_t q, half2 & h01, half2 & h23) {
    const uint32_t u = q ^ 0x80808080u, m = 0x64646464u;
    uint32_t r01 = __byte_perm(u, m, 0x4140), r23 = __byte_perm(u, m, 0x4342);
    const half2 off = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152
    h01 = __hsub2(*(half2 *) &r01, off);
    h23 = __hsub2(*(half2 *) &r23, off);
}

template <ggml_type type> struct mmvq_h1_traits;
template <> struct mmvq_h1_traits<GGML_TYPE_Q4_0> { static constexpr int BB = 18, NWD = 5; };   // half d + 16 nibble bytes
template <> struct mmvq_h1_traits<GGML_TYPE_Q8_0> { static constexpr int BB = 34, NWD = 9; };   // half d + 32 int8

// one block from its NWD staged words (hi: d in the high half of wd[0], quants word-aligned from wd[1]; else d low, quants at
// +2 bytes); returns d * sum(q * x') in fp32 from a 16-HFMA2 fp16 chain (|x'| <= 1: q4_0 |chain| <= 256, q8_0 <= 4096)
template <ggml_type type>
static __device__ __forceinline__ float mmvq_h1_block(const uint32_t * wd, const bool hi, const half2 * xv) {
    const float d = __half2float(__ushort_as_half((unsigned short) (hi ? wd[0] >> 16 : wd[0] & 0xffff)));
    const int sh = hi ? 32 : 16;
    half2 h = __float2half2_rn(0.0f);
    if constexpr (type == GGML_TYPE_Q4_0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = __funnelshift_rc(wd[i], wd[i + 1], sh);
            half2 a, b;
            mmvq_h1_nib(q & 0x0f0f0f0fu, a, b);
            h = __hfma2(a, xv[2*i], h);
            h = __hfma2(b, xv[2*i + 1], h);
            mmvq_h1_nib((q >> 4) & 0x0f0f0f0fu, a, b);
            h = __hfma2(a, xv[8 + 2*i], h);
            h = __hfma2(b, xv[8 + 2*i + 1], h);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = __funnelshift_rc(wd[i], wd[i + 1], sh);
            half2 a, b;
            mmvq_h1_i8(q, a, b);
            h = __hfma2(a, xv[2*i], h);
            h = __hfma2(b, xv[2*i + 1], h);
        }
    }
    const float2 f = __half22float2(h);
    return d*(f.x + f.y);
}

// row r of the warp's group: S-step of the transposed butterfly (lanes with bit S set keep the upper half of the rows)
template <int S>
static __device__ __forceinline__ void mmvq_h1_bfly(float * acc, const int lane) {
    const bool up = lane & S;
#pragma unroll
    for (int i = 0; i < S; ++i) {
        const float send = up ? acc[i] : acc[i + S];
        const float keep = up ? acc[i + S] : acc[i];
        acc[i] = keep + __shfl_xor_sync(0xffffffffu, send, S);
    }
}

// grid (ceil(M/(WPB*RPW)), slots); ID: slot y -> expert ids[y], activation row (nxr == 1 ? 0 : y), dst row y
template <ggml_type type, int WPB, int U, int RPW, bool ID>
static __global__ void __launch_bounds__(WPB*32) mmvq_h1_q(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const int64_t expert_bytes, const int32_t * __restrict__ ids,
        const __half * __restrict__ XH, const float * __restrict__ XS, const int nxr, float * __restrict__ Y, const int64_t sy,
        const int M, const int K) {
    constexpr int BB  = mmvq_h1_traits<type>::BB, NWD = mmvq_h1_traits<type>::NWD;
    constexpr int STG = (32*BB + 15 + 15)/16;      // a 32-block chunk + slack for the aligned-down start
    constexpr int NLD = (STG + 31)/32;             // 16-byte loads per lane per row
    __shared__ uint4 stage[WPB][U][STG];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int r0 = (blockIdx.x*WPB + wid)*RPW;
    if (r0 >= M) {
        return;   // no block-wide barriers below
    }
    const int slot = blockIdx.y;
    if (ID) {
        W += (int64_t) ids[slot]*expert_bytes;
    }
    const int xr = nxr == 1 ? 0 : slot;
    XH += (int64_t) xr*K;
    XS += (int64_t) xr*(K/H1_QK);
    const int nb = K/H1_QK;

    float acc[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        acc[i] = 0.0f;
    }
    for (int cb = 0; cb < nb; cb += 32) {
        const int nblk = min(32, nb - cb);
        half2 xv[16];
        float s = 0.0f;
        if (lane < nblk) {
            const uint4 * xp = (const uint4 *) (XH + (int64_t) (cb + lane)*H1_QK);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint4 v = xp[i];
                xv[4*i + 0] = *(const half2 *) &v.x; xv[4*i + 1] = *(const half2 *) &v.y;
                xv[4*i + 2] = *(const half2 *) &v.z; xv[4*i + 3] = *(const half2 *) &v.w;
            }
            s = XS[cb + lane];
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                xv[i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int rr = 0; rr < RPW; rr += U) {
            uint4 v[U][NLD];
            int   mis[U], nu4[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                // rows past M re-read the last row; their results are never written
                const int row = min(r0 + rr + u, M - 1);
                const uintptr_t a = (uintptr_t) (W + (int64_t) row*row_bytes + (int64_t) cb*BB);
                mis[u] = (int) (a & 15);
                nu4[u] = (mis[u] + nblk*BB + 15)/16;
                const uint4 * src = (const uint4 *) (a - mis[u]);
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        v[u][j] = __ldg(src + lane + 32*j);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        stage[wid][u][lane + 32*j] = v[u][j];
                    }
                }
            }
            __syncwarp();
            if (lane < nblk) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int o = mis[u] + BB*lane;   // byte offset of this lane's block in the stage (even)
                    const uint32_t * p = (const uint32_t *) ((const char *) stage[wid][u] + (o & ~3));
                    uint32_t wd[NWD];
#pragma unroll
                    for (int i = 0; i < NWD; ++i) {
                        wd[i] = p[i];
                    }
                    acc[rr + u] += s*mmvq_h1_block<type>(wd, o & 2, xv);
                }
            }
            __syncwarp();
        }
    }
    // high lane bits: plain reductions; the low log2(RPW) bits: transposed, so lane r ends with row r
#pragma unroll
    for (int sft = 16; sft >= RPW; sft >>= 1) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], sft);
        }
    }
    if constexpr (RPW >= 32) { mmvq_h1_bfly<16>(acc, lane); }
    if constexpr (RPW >= 16) { mmvq_h1_bfly<8>(acc, lane); }
    if constexpr (RPW >= 8)  { mmvq_h1_bfly<4>(acc, lane); }
    if constexpr (RPW >= 4)  { mmvq_h1_bfly<2>(acc, lane); }
    if constexpr (RPW >= 2)  { mmvq_h1_bfly<1>(acc, lane); }
    if (lane < RPW && r0 + lane < M) {
        Y[(int64_t) slot*sy + r0 + lane] = acc[0];
    }
}

// p100-series 10-06: gate + up + GLU for one column in one launch (gemma4's dense q4_0 FFN: Pascal fuses gate/up only through the
// Q4_K/Q5_K hybrid kernels, so it ran two h1 matvecs and a GLU kernel per layer). Each matrix's rows are accumulated exactly as
// mmvq_h1_q accumulates them (same chunk order, block math, fold and butterfly); the epilogue is the fused path's
// up * act(gate) = the gated unary kernel's act(gate) * up, so the output is bit-identical to the three launches.
// Warp-split layout: the block's first WPB/2 warps take up rows, the others the same rows of gate, each warp exactly mmvq_h1_q's
// work for its RPW rows; the row sums meet in shared memory for the epilogue. (A first version had every warp stream both
// matrices: twice the latency per block with gemma4's ~540 rows = 34 blocks on 56 SMs, 4.5 % slower than three launches.)
template <ggml_type type, int WPB, int U, int RPW, ggml_glu_op GLU>
static __global__ void __launch_bounds__(WPB*32) mmvq_h1_q_glu(
        const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg, const int64_t row_bytes,
        const __half * __restrict__ XH, const float * __restrict__ XS, float * __restrict__ Y, const int M, const int K) {
    static_assert(WPB % 2 == 0, "warp pairs");
    constexpr int HW  = WPB/2;
    constexpr int BB  = mmvq_h1_traits<type>::BB, NWD = mmvq_h1_traits<type>::NWD;
    constexpr int STG = (32*BB + 15 + 15)/16;
    constexpr int NLD = (STG + 31)/32;
    __shared__ uint4 stage[WPB][U][STG];
    __shared__ float res[2][HW*RPW];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int mat  = wid / HW;                                   // 0: up, 1: gate
    const int r0   = (blockIdx.x*HW + wid % HW)*RPW;
    const uint8_t * W = mat == 0 ? Wu : Wg;
    const int nb = K/H1_QK;

    if (r0 < M) {
        float acc[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[i] = 0.0f;
        }
        for (int cb = 0; cb < nb; cb += 32) {
            const int nblk = min(32, nb - cb);
            half2 xv[16];
            float s = 0.0f;
            if (lane < nblk) {
                const uint4 * xp = (const uint4 *) (XH + (int64_t) (cb + lane)*H1_QK);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 v = xp[i];
                    xv[4*i + 0] = *(const half2 *) &v.x; xv[4*i + 1] = *(const half2 *) &v.y;
                    xv[4*i + 2] = *(const half2 *) &v.z; xv[4*i + 3] = *(const half2 *) &v.w;
                }
                s = XS[cb + lane];
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    xv[i] = __float2half2_rn(0.0f);
                }
            }
#pragma unroll
            for (int rr = 0; rr < RPW; rr += U) {
                uint4 v[U][NLD];
                int   mis[U], nu4[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int row = min(r0 + rr + u, M - 1);
                    const uintptr_t a = (uintptr_t) (W + (int64_t) row*row_bytes + (int64_t) cb*BB);
                    mis[u] = (int) (a & 15);
                    nu4[u] = (mis[u] + nblk*BB + 15)/16;
                    const uint4 * src = (const uint4 *) (a - mis[u]);
#pragma unroll
                    for (int j = 0; j < NLD; ++j) {
                        if (lane + 32*j < nu4[u]) {
                            v[u][j] = __ldg(src + lane + 32*j);
                        }
                    }
                }
#pragma unroll
                for (int u = 0; u < U; ++u) {
#pragma unroll
                    for (int j = 0; j < NLD; ++j) {
                        if (lane + 32*j < nu4[u]) {
                            stage[wid][u][lane + 32*j] = v[u][j];
                        }
                    }
                }
                __syncwarp();
                if (lane < nblk) {
#pragma unroll
                    for (int u = 0; u < U; ++u) {
                        const int o = mis[u] + BB*lane;
                        const uint32_t * p = (const uint32_t *) ((const char *) stage[wid][u] + (o & ~3));
                        uint32_t wd[NWD];
#pragma unroll
                        for (int i = 0; i < NWD; ++i) {
                            wd[i] = p[i];
                        }
                        acc[rr + u] += s*mmvq_h1_block<type>(wd, o & 2, xv);
                    }
                }
                __syncwarp();
            }
        }
#pragma unroll
        for (int sft = 16; sft >= RPW; sft >>= 1) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], sft);
            }
        }
        if constexpr (RPW >= 32) { mmvq_h1_bfly<16>(acc, lane); }
        if constexpr (RPW >= 16) { mmvq_h1_bfly<8>(acc, lane); }
        if constexpr (RPW >= 8)  { mmvq_h1_bfly<4>(acc, lane); }
        if constexpr (RPW >= 4)  { mmvq_h1_bfly<2>(acc, lane); }
        if constexpr (RPW >= 2)  { mmvq_h1_bfly<1>(acc, lane); }
        if (lane < RPW) {
            res[mat][(wid % HW)*RPW + lane] = acc[0];
        }
    }
    __syncthreads();
    const int t = threadIdx.x;
    if (t < HW*RPW) {
        const int row = blockIdx.x*HW*RPW + t;
        if (row < M) {
            const float up = res[0][t], g = res[1][t];
            float r;
            if constexpr (GLU == GGML_GLU_OP_GEGLU) {
                r = up * ggml_cuda_op_gelu_single(g);
            } else {
                r = up * ggml_cuda_op_silu_single(g);
            }
            Y[row] = r;
        }
    }
}

// NC columns (dense, 2..8: speculative verify, parallel decode): mmvq_h1_q's layout and staging, but each block's weights are
// decoded to half once and then meet every column's activation (NC 16-HFMA2 chains, per-column per-block scale s), so the weights
// are still read once. Per column the arithmetic is mmvq_h1_q's (same chain order, same fp32 fold). grid ceil(M/(WPB*RPW)).
template <ggml_type type>
static __device__ __forceinline__ void mmvq_h1_decode(const uint32_t * wd, const bool hi, float & d, half2 (&w)[16]) {
    d = __half2float(__ushort_as_half((unsigned short) (hi ? wd[0] >> 16 : wd[0] & 0xffff)));
    const int sh = hi ? 32 : 16;
    if constexpr (type == GGML_TYPE_Q4_0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q = __funnelshift_rc(wd[i], wd[i + 1], sh);
            mmvq_h1_nib(q & 0x0f0f0f0fu, w[2*i], w[2*i + 1]);
            mmvq_h1_nib((q >> 4) & 0x0f0f0f0fu, w[8 + 2*i], w[8 + 2*i + 1]);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t q = __funnelshift_rc(wd[i], wd[i + 1], sh);
            mmvq_h1_i8(q, w[2*i], w[2*i + 1]);
        }
    }
}

template <ggml_type type, int NC, int WPB, int U, int RPW>
static __global__ void __launch_bounds__(WPB*32) mmvq_h1_qn(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XH, const float * __restrict__ XS,
        float * __restrict__ Y, const int64_t sy, const int M, const int K) {
    constexpr int BB  = mmvq_h1_traits<type>::BB, NWD = mmvq_h1_traits<type>::NWD;
    constexpr int STG = (32*BB + 15 + 15)/16;
    constexpr int NLD = (STG + 31)/32;
    __shared__ uint4 stage[WPB][U][STG];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int r0 = (blockIdx.x*WPB + wid)*RPW;
    if (r0 >= M) {
        return;
    }
    const int nb = K/H1_QK;

    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int cb = 0; cb < nb; cb += 32) {
        const int nblk = min(32, nb - cb);
        half2 xv[NC][16];
        float s[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[c] = 0.0f;
            if (lane < nblk) {
                const uint4 * xp = (const uint4 *) (XH + (int64_t) c*K + (int64_t) (cb + lane)*H1_QK);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint4 v = xp[i];
                    xv[c][4*i + 0] = *(const half2 *) &v.x; xv[c][4*i + 1] = *(const half2 *) &v.y;
                    xv[c][4*i + 2] = *(const half2 *) &v.z; xv[c][4*i + 3] = *(const half2 *) &v.w;
                }
                s[c] = XS[(int64_t) c*nb + cb + lane];
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    xv[c][i] = __float2half2_rn(0.0f);
                }
            }
        }
#pragma unroll
        for (int rr = 0; rr < RPW; rr += U) {
            uint4 v[U][NLD];
            int   mis[U], nu4[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int row = min(r0 + rr + u, M - 1);
                const uintptr_t a = (uintptr_t) (W + (int64_t) row*row_bytes + (int64_t) cb*BB);
                mis[u] = (int) (a & 15);
                nu4[u] = (mis[u] + nblk*BB + 15)/16;
                const uint4 * src = (const uint4 *) (a - mis[u]);
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        v[u][j] = __ldg(src + lane + 32*j);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        stage[wid][u][lane + 32*j] = v[u][j];
                    }
                }
            }
            __syncwarp();
            if (lane < nblk) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int o = mis[u] + BB*lane;
                    const uint32_t * p = (const uint32_t *) ((const char *) stage[wid][u] + (o & ~3));
                    uint32_t wd[NWD];
#pragma unroll
                    for (int i = 0; i < NWD; ++i) {
                        wd[i] = p[i];
                    }
                    float d;
                    half2 w[16];
                    mmvq_h1_decode<type>(wd, o & 2, d, w);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        // mmvq_h1_block's chain: (w[0], w[1]) .. in its order
                        half2 h = __float2half2_rn(0.0f);
                        if constexpr (type == GGML_TYPE_Q4_0) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                h = __hfma2(w[2*i],         xv[c][2*i],         h);
                                h = __hfma2(w[2*i + 1],     xv[c][2*i + 1],     h);
                                h = __hfma2(w[8 + 2*i],     xv[c][8 + 2*i],     h);
                                h = __hfma2(w[8 + 2*i + 1], xv[c][8 + 2*i + 1], h);
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 16; ++i) {
                                h = __hfma2(w[i], xv[c][i], h);
                            }
                        }
                        const float2 f = __half22float2(h);
                        acc[c][rr + u] += s[c]*(d*(f.x + f.y));
                    }
                }
            }
            __syncwarp();
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int sft = 16; sft >= RPW; sft >>= 1) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                acc[c][i] += __shfl_xor_sync(0xffffffffu, acc[c][i], sft);
            }
        }
        if constexpr (RPW >= 16) { mmvq_h1_bfly<8>(acc[c], lane); }
        if constexpr (RPW >= 8)  { mmvq_h1_bfly<4>(acc[c], lane); }
        if constexpr (RPW >= 4)  { mmvq_h1_bfly<2>(acc[c], lane); }
        if constexpr (RPW >= 2)  { mmvq_h1_bfly<1>(acc[c], lane); }
        if (lane < RPW && r0 + lane < M) {
            Y[(int64_t) c*sy + r0 + lane] = acc[c][0];
        }
    }
}

// The fp16 activation (+ scales) is cached across consecutive matvecs that read the same src1 (q/k/v, an expert's gate/up); the
// contents are fixed within one graph evaluation, cleared at the start of each. A few entries per CUDA context (per device), so
// that an activation written ahead by the fused rms_norm (below) survives a second norm issued before its matvecs (gemma4: the
// dense-MLP and MoE pre-norms). Keys use the view base at offset 0, so a reshape of the norm output finds the norm's entry.
struct mmvq_h1_entry {
    const ggml_tensor * src1 = nullptr;
    const void *        data = nullptr;
    int64_t             nxr = 0, K = 0, sx = 0;
    void *              buf  = nullptr;
    size_t              cap  = 0;
};
static constexpr int MMVQ_H1_NCACHE = 4;
struct mmvq_h1_cache {
    mmvq_h1_entry e[MMVQ_H1_NCACHE];
    int           next   = 0;
    int           device = -1;
};
// lives in ctx.mmvq_h1_cache (one per backend context, so one per dispatch thread); never freed, like the context's other pools
static mmvq_h1_cache & mmvq_h1_cache_of(ggml_backend_cuda_context & ctx) {
    if (ctx.mmvq_h1_cache == nullptr) {
        ctx.mmvq_h1_cache = new mmvq_h1_cache();
    }
    return *(mmvq_h1_cache *) ctx.mmvq_h1_cache;
}

void ggml_cuda_mmvq_h1_invalidate(ggml_backend_cuda_context & ctx) {
    if (ctx.mmvq_h1_cache != nullptr) {
        for (mmvq_h1_entry & e : ((mmvq_h1_cache *) ctx.mmvq_h1_cache)->e) {
            e.src1 = nullptr;
            e.data = nullptr;
        }
    }
}

static const ggml_tensor * mmvq_h1_key(const ggml_tensor * t) {
    return t->view_src != nullptr && t->view_offs == 0 ? t->view_src : t;
}

// entry for (key, data, nxr, K, sx): hit = already holds that activation; otherwise the next entry round robin, sized for it
static mmvq_h1_entry & mmvq_h1_slot(ggml_backend_cuda_context & ctx, const ggml_tensor * key, const void * data, const int64_t nxr,
                                    const int64_t K, const int64_t sx, bool & hit) {
    mmvq_h1_cache & cache = mmvq_h1_cache_of(ctx);
    if (cache.device != ctx.device) {
        for (mmvq_h1_entry & e : cache.e) {
            if (e.buf) {
                ggml_cuda_set_device(cache.device);
                CUDA_CHECK(cudaFree(e.buf));
                ggml_cuda_set_device(ctx.device);
            }
        }
        cache = mmvq_h1_cache();
        cache.device = ctx.device;
    }
    for (mmvq_h1_entry & e : cache.e) {
        if (e.src1 == key && e.data == data && e.nxr == nxr && e.K == K && e.sx == sx) {
            hit = true;
            return e;
        }
    }
    hit = false;
    mmvq_h1_entry & e = cache.e[cache.next];
    cache.next = (cache.next + 1) % MMVQ_H1_NCACHE;
    const size_t need = (size_t) nxr*K*sizeof(__half) + (size_t) nxr*(K/H1_QK)*sizeof(float);
    if (e.cap < need) {
        if (e.buf) {
            CUDA_CHECK(cudaFree(e.buf)); // synchronizes; only when the buffer grows
        }
        CUDA_CHECK(cudaMalloc(&e.buf, need));
        e.cap = need;
    }
    e.src1 = key; e.data = data; e.nxr = nxr; e.K = K; e.sx = sx;
    return e;
}

bool ggml_cuda_mmvq_h1_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {
    // GGML_CUDA_MMVQ_H1=0: leave one-column q4_0 to the integer path (A/B)
    static const bool enabled = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1"); return !s || atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // GGML_CUDA_MMVQ_H1_Q8=0: leave q8_0 to the integer path
    static const bool q8_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_Q8"); return !s || atoi(s) != 0; }();
    const ggml_type type = src0->type;
    if (!enabled || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) ||
            !(type == GGML_TYPE_Q4_0 || (type == GGML_TYPE_Q8_0 && q8_on)) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t BB = type == GGML_TYPE_Q4_0 ? mmvq_h1_traits<GGML_TYPE_Q4_0>::BB : mmvq_h1_traits<GGML_TYPE_Q8_0>::BB;
    const int64_t row_bytes = (K/H1_QK)*BB;
    // the staging reads whole 16-byte units from the 16-byte-aligned address at or below each chunk, so it can run up to 15
    // bytes past a row's end: fine when rows are 16-byte multiples on a 16-byte-aligned base (no overrun at all), or when the
    // tensor owns its allocation and K is not a multiple of MATRIX_ROW_PADDING (the last row is padded to the next multiple)
    const bool rows_ok = K % H1_QK == 0 && src0->nb[0] == (size_t) BB && src0->nb[1] == (size_t) row_bytes && src0->ne[3] == 1 &&
                         ((uintptr_t) src0->data % 2) == 0 &&
                         ((row_bytes % 16 == 0 && (uintptr_t) src0->data % 16 == 0 && src0->nb[2] % 16 == 0) ||
                          (src0->view_src == nullptr && K % MATRIX_ROW_PADDING != 0));
    // fewer than 32 blocks per row leaves lanes idle in the per-chunk layout: the integer path's short-K geometry wins there
    // (gemma4 ffn_down_exps split by K, 352: 21.1 us integer vs 30.0 here)
    if (K < 32*H1_QK) {
        return false;
    }
    if (!rows_ok || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
            dst->nb[1] % sizeof(float) != 0 || src1->ne[0] != K || dst->ne[0] != M) {
        return false;
    }
    int64_t nxr, slots;
    if (ids) {
        // one token: ids [n_used, 1], src1 [K, 1 (shared) or n_used, 1], dst [M, n_used, 1]
        const int64_t n_used = ids->ne[0];
        if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[1] != n_used ||
                dst->ne[2] != 1 || (src1->ne[1] != 1 && src1->ne[1] != n_used) || ids->nb[0] != sizeof(int32_t)) {
            return false;
        }
        nxr = src1->ne[1];
        slots = n_used;
    } else {
        // GGML_CUDA_MMVQ_H1_NC=0: 2..8 columns back to the integer path (A/B)
        static const bool nc_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_NC"); return !s || atoi(s) != 0; }();
        const int64_t ncols = src1->ne[1];
        if (src0->ne[2] != 1 || ncols < 1 || ncols > 8 || (ncols > 1 && !nc_on) || dst->ne[1] != ncols || src1->ne[2] != 1 ||
                src1->ne[3] != 1 || (ncols > 1 && (src1->nb[1] % 16 != 0 || (uintptr_t) src1->data % 16 != 0))) {
            return false;
        }
        nxr = ncols;
        slots = 1;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t sx = nxr > 1 ? (int64_t) (src1->nb[1]/sizeof(float)) : 0;

    bool hit;
    mmvq_h1_entry & entry = mmvq_h1_slot(ctx, mmvq_h1_key(src1), src1->data, nxr, K, sx, hit);
    __half * xh = (__half *) entry.buf;
    float  * xs = (float *) ((char *) entry.buf + (size_t) nxr*K*sizeof(__half));
    if (!hit) {
        const dim3 pg((unsigned) ((K + 255)/256), (unsigned) nxr);
        mmvq_h1_prep<<<pg, 256, 0, stream>>>((const float *) src1->data, sx, xh, xs, (int) K);
    }
    constexpr int RB = MMVQ_H1_WPB*MMVQ_H1_RPW;
    if (!ids && nxr > 1) {
        // 8 rows per warp: each warp re-reads the NC columns' activation per chunk, so more rows per warp cut that L2 traffic (4 rows,
        // 4 columns: 89 MB of activation reads vs 47 MB of weights at 8704 x 5120 q8_0). q8_0 at 4 columns, rows x K, 4 -> 8 rows:
        // 8704x5120 145 -> 131 us, 5120x8704 145 -> 126, 5120x5120 87 -> 77, 5120x3072 55 -> 49 (integer path 175/173/106/66);
        // 16 rows leaves too few blocks. GGML_CUDA_MMVQ_H1_NC_RPW=4/16 for A/B.
        static const int rpw_n = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_NC_RPW"); return s ? atoi(s) : 8; }();
        const int64_t syn = (int64_t) (dst->nb[1]/sizeof(float));
        auto launchn = [&](auto type_tag, auto nc_tag) {
            constexpr ggml_type T = decltype(type_tag)::value;
            constexpr int NC = decltype(nc_tag)::value;
            auto go = [&](auto rpw_tag) {
                constexpr int R = decltype(rpw_tag)::value;
                const dim3 gridn((unsigned) ((M + MMVQ_H1_WPB*R - 1)/(MMVQ_H1_WPB*R)));
                mmvq_h1_qn<T, NC, MMVQ_H1_WPB, MMVQ_H1_U, R><<<gridn, MMVQ_H1_WPB*32, 0, stream>>>(
                    (const uint8_t *) src0->data, row_bytes, xh, xs, (float *) dst->data, syn, (int) M, (int) K);
            };
            switch (rpw_n) {
                case 16: go(std::integral_constant<int, 16>{}); break;
                case 8:  go(std::integral_constant<int, 8>{});  break;
                default: go(std::integral_constant<int, 4>{});  break;
            }
        };
        auto bytype = [&](auto type_tag) {
            switch (nxr) {
                case 2:  launchn(type_tag, std::integral_constant<int, 2>{}); break;
                case 3:  launchn(type_tag, std::integral_constant<int, 3>{}); break;
                case 4:  launchn(type_tag, std::integral_constant<int, 4>{}); break;
                case 5:  launchn(type_tag, std::integral_constant<int, 5>{}); break;
                case 6:  launchn(type_tag, std::integral_constant<int, 6>{}); break;
                case 7:  launchn(type_tag, std::integral_constant<int, 7>{}); break;
                default: launchn(type_tag, std::integral_constant<int, 8>{}); break;
            }
        };
        if (type == GGML_TYPE_Q4_0) {
            bytype(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});
        } else {
            bytype(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const dim3 grid((unsigned) ((M + RB - 1)/RB), (unsigned) slots);
    const int64_t sy = ids ? (int64_t) (dst->nb[1]/sizeof(float)) : 0;
    auto launch = [&](auto type_tag) {
        constexpr ggml_type T = decltype(type_tag)::value;
        if (ids) {
            mmvq_h1_q<T, MMVQ_H1_WPB, MMVQ_H1_U, MMVQ_H1_RPW, true><<<grid, MMVQ_H1_WPB*32, 0, stream>>>(
                (const uint8_t *) src0->data, row_bytes, (int64_t) src0->nb[2], (const int32_t *) ids->data, xh, xs, (int) nxr,
                (float *) dst->data, sy, (int) M, (int) K);
        } else {
            mmvq_h1_q<T, MMVQ_H1_WPB, MMVQ_H1_U, MMVQ_H1_RPW, false><<<grid, MMVQ_H1_WPB*32, 0, stream>>>(
                (const uint8_t *) src0->data, row_bytes, 0, nullptr, xh, xs, 1, (float *) dst->data, 0, (int) M, (int) K);
        }
    };
    if (type == GGML_TYPE_Q4_0) {
        launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});
    } else {
        launch(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// GGML_CUDA_MMVQ_H1_GLU=0: keep gate / up / GLU as three launches
static bool mmvq_h1_glu_enabled() {
    static const bool on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_GLU"); return !s || atoi(s) != 0; }();
    return on;
}

bool ggml_cuda_mmvq_h1_glu_ok(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
    if (!mmvq_h1_glu_enabled() || up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT || up->src[1] != gate->src[1]) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    if (op != GGML_GLU_OP_GEGLU && op != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    const ggml_tensor * w = up->src[0], * wg = gate->src[0], * x = up->src[1];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || (w->type != GGML_TYPE_Q4_0 && w->type != GGML_TYPE_Q8_0) ||
            wg->type != w->type || !ggml_are_same_shape(w, wg) || !ggml_are_same_stride(w, wg) || x->type != GGML_TYPE_F32 ||
            x->ne[1] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(glu) || glu->ne[1] != 1) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t BB = w->type == GGML_TYPE_Q4_0 ? mmvq_h1_traits<GGML_TYPE_Q4_0>::BB : mmvq_h1_traits<GGML_TYPE_Q8_0>::BB;
    const int64_t row_bytes = (K/H1_QK)*BB;
    auto rows_ok = [&](const ggml_tensor * t) {
        return K % H1_QK == 0 && t->nb[0] == (size_t) BB && t->nb[1] == (size_t) row_bytes && t->ne[2] == 1 && t->ne[3] == 1 &&
               ((uintptr_t) t->data % 2) == 0 &&
               ((row_bytes % 16 == 0 && (uintptr_t) t->data % 16 == 0) || (t->view_src == nullptr && K % MATRIX_ROW_PADDING != 0));
    };
    return K >= 32*H1_QK && rows_ok(w) && rows_ok(wg) && x->nb[0] == sizeof(float) && x->ne[0] == K;
}

bool ggml_cuda_mmvq_h1_glu_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * gate_w, const ggml_glu_op op, ggml_tensor * dst) {
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t BB = src0->type == GGML_TYPE_Q4_0 ? mmvq_h1_traits<GGML_TYPE_Q4_0>::BB : mmvq_h1_traits<GGML_TYPE_Q8_0>::BB;
    const int64_t row_bytes = (K/H1_QK)*BB;
    cudaStream_t stream = ctx.stream();
    bool hit;
    mmvq_h1_entry & entry = mmvq_h1_slot(ctx, mmvq_h1_key(src1), src1->data, 1, K, 0, hit);
    __half * xh = (__half *) entry.buf;
    float  * xs = (float *) ((char *) entry.buf + (size_t) K*sizeof(__half));
    if (!hit) {
        const dim3 pg((unsigned) ((K + 255)/256), 1);
        mmvq_h1_prep<<<pg, 256, 0, stream>>>((const float *) src1->data, 0, xh, xs, (int) K);
    }
    constexpr int RB = (MMVQ_H1_WPB/2)*MMVQ_H1_RPW; // rows per block: half the warps take up, half gate
    const dim3 grid((unsigned) ((M + RB - 1)/RB));
    auto launch = [&](auto type_tag, auto glu_tag) {
        constexpr ggml_type   T = decltype(type_tag)::value;
        constexpr ggml_glu_op G = decltype(glu_tag)::value;
        mmvq_h1_q_glu<T, MMVQ_H1_WPB, MMVQ_H1_U, MMVQ_H1_RPW, G><<<grid, MMVQ_H1_WPB*32, 0, stream>>>(
            (const uint8_t *) src0->data, (const uint8_t *) gate_w->data, row_bytes, xh, xs, (float *) dst->data, (int) M, (int) K);
    };
    auto bytype = [&](auto glu_tag) {
        if (src0->type == GGML_TYPE_Q4_0) {
            launch(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{}, glu_tag);
        } else {
            launch(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{}, glu_tag);
        }
    };
    if (op == GGML_GLU_OP_GEGLU) {
        bytype(std::integral_constant<ggml_glu_op, GGML_GLU_OP_GEGLU>{});
    } else {
        bytype(std::integral_constant<ggml_glu_op, GGML_GLU_OP_SWIGLU>{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// rms_norm * weight with the h1 activation prep fused in: one row of K (decode), the register path of rms_norm_f32 (same
// arithmetic, so dst is bit-identical), and each warp's 32 consecutive outputs -- one h1 block -- also written as x/s in fp16 with
// s = max|x| (mmvq_h1_prep's arithmetic). Saves the prep launch of every matvec fed by a norm (gemma4: 91 per GPU per token).
// ADD: x = x + b first (the residual ADD, exactly the unfused a + b), stored to sum_out (the next residual), then the norm.
template <int block_size, bool ADD = false>
static __global__ void __launch_bounds__(block_size) mmvq_h1_rms_norm_mul(
        const float * x, const float * w, float * dst, __half * __restrict__ XH,
        float * __restrict__ XS, const int ncols, const float eps, const float * b = nullptr, float * sum_out = nullptr) {
    // no __restrict__ on x / b / sum_out / dst: with ADD the sum may be written in place of an input (row-for-row alias)
    constexpr int max_regs = 8;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;

    float xv[max_regs];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if constexpr (ADD) {
            xv[u] = col < ncols ? x[col] + b[col] : 0.0f;
        } else {
            xv[u] = col < ncols ? x[col] : 0.0f;
        }
        tmp += xv[u] * xv[u];
    }
    if constexpr (ADD) {
#pragma unroll
        for (int u = 0; u < max_regs; ++u) {
            const int col = tid + u*block_size;
            if (col < ncols) {
                sum_out[col] = xv[u];
            }
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {   // ncols % 32 == 0: whole warps
            const float v = scale * xv[u] * w[col];
            dst[col] = v;
            float m = fabsf(v);
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            }
            const float s = m > 0.0f ? m : 1.0f;
            XH[col] = __float2half(v/s);
            if ((col & 31) == 0) {
                XS[col/32] = s;
            }
        }
    }
}

// add != nullptr: cgraph->nodes[i] is the RMS_NORM of `add` (the residual ADD at i - 1), fused in front
static bool mmvq_h1_rms_norm_mul_try_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, ggml_tensor * add);

bool ggml_cuda_mmvq_h1_rms_norm_mul_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    return mmvq_h1_rms_norm_mul_try_impl(ctx, cgraph, i, nullptr);
}

bool ggml_cuda_mmvq_h1_add_rms_norm_mul_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // nodes i, i+1, i+2 = ADD, RMS_NORM, MUL (checked by the caller)
    ggml_tensor * add = cgraph->nodes[i];
    if (!ggml_cuda_add_rms_norm_mul_ok(add, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) || ggml_nrows(add) != 1) {
        return false;
    }
    return mmvq_h1_rms_norm_mul_try_impl(ctx, cgraph, i + 1, add);
}

static bool mmvq_h1_rms_norm_mul_try_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, ggml_tensor * add) {
    // GGML_CUDA_MMVQ_H1_NORM=0: keep the separate prep (A/B)
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_H1");
        const char * n = getenv("GGML_CUDA_MMVQ_H1_NORM");
        return (!s || atoi(s) != 0) && (!n || atoi(n) != 0);
    }();
    static const bool q8_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_Q8"); return !s || atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    const ggml_tensor * rms = cgraph->nodes[i];
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    const ggml_tensor * x   = rms->src[0];
    const ggml_tensor * w   = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t K = x->ne[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || ggml_nrows(x) != 1 ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(mul) || w->ne[0] != K || ggml_nrows(w) != 1 ||
            !ggml_are_same_shape(mul, x) || K % H1_QK != 0 || K < 32*H1_QK || K > 1024*8) {
        return false;
    }
    // only when an h1-eligible matvec reads the output soon (else the cache entry would be wasted work)
    bool consumed = false;
    for (int j = i + 2; j < cgraph->n_nodes && j < i + 16 && !consumed; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && mmvq_h1_key(t->src[1]) == mul && t->src[0]->ne[0] == K &&
                (t->src[0]->type == GGML_TYPE_Q4_0 || (t->src[0]->type == GGML_TYPE_Q8_0 && q8_on))) {
            consumed = true;
        }
    }
    if (!consumed) {
        return false;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    bool hit;
    mmvq_h1_entry & entry = mmvq_h1_slot(ctx, mul, mul->data, 1, K, 0, hit);
    __half * xh = (__half *) entry.buf;
    float  * xs = (float *) ((char *) entry.buf + (size_t) K*sizeof(__half));
    if (add != nullptr) {
        mmvq_h1_rms_norm_mul<1024, true><<<1, 1024, 0, ctx.stream()>>>(
            (const float *) add->src[0]->data, (const float *) w->data, (float *) mul->data, xh, xs, (int) K, eps,
            (const float *) add->src[1]->data, (float *) add->data);
    } else {
        mmvq_h1_rms_norm_mul<1024><<<1, 1024, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) w->data, (float *) mul->data, xh, xs, (int) K, eps);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- other Pascal decode fusions (fewer launches: decode under tensor split is host-bound, perf 10-05) ----

// out[:, e, t] = res[:, e, t] * scale[ids[e, t]]: the per-expert weight scale of a MUL_MAT_ID (llama-graph's reshape -> repeat ->
// get_rows -> mul, 3 launches) in one, read straight from the scale tensor. Same single multiply, so bit-identical.
static __global__ void mmid_slot_scale(const float * __restrict__ res, const float * __restrict__ scale, const int32_t * __restrict__ ids,
                                       float * __restrict__ out, const int M, const int64_t res_s1, const int64_t res_s2,
                                       const int64_t out_s1, const int64_t out_s2, const int64_t ids_s1) {
    const int e = blockIdx.x, t = blockIdx.y;
    const float s = scale[ids[t*ids_s1 + e]];
    const float * r = res + t*res_s2 + e*res_s1;
    float       * o = out + t*out_s2 + e*out_s1;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        o[i] = r[i] * s;
    }
}

void ggml_cuda_mmid_slot_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * res, const ggml_tensor * scale,
                               const ggml_tensor * ids, ggml_tensor * out) {
    const dim3 grid((unsigned) res->ne[1], (unsigned) res->ne[2]);
    mmid_slot_scale<<<grid, 256, 0, ctx.stream()>>>((const float *) res->data, (const float *) scale->data,
        (const int32_t *) ids->data, (float *) out->data, (int) res->ne[0], res->nb[1]/sizeof(float), res->nb[2]/sizeof(float),
        out->nb[1]/sizeof(float), out->nb[2]/sizeof(float), ids->nb[1]/sizeof(int32_t));
    CUDA_CHECK(cudaGetLastError());
}

// (scale_out * (rms_norm(x))) * w for one row: the gemma4 MoE router input (rms_norm -> scale 1/sqrt(n_embd) -> mul
// ffn_gate_inp_s), the register path and expression order of rms_norm_f32<..., do_scale> followed by the flat mul, so the
// result is bit-identical; 1 launch instead of 2.
template <int block_size>
static __global__ void __launch_bounds__(block_size) rms_norm_scale_mul(
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ dst, const int ncols, const float eps,
        const float scale_out) {
    constexpr int max_regs = 8;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;

    float xv[max_regs];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        xv[u] = col < ncols ? x[col] : 0.0f;
        tmp += xv[u] * xv[u];
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {
            const float v = scale_out * (scale * xv[u]);
            dst[col] = v * w[col];
        }
    }
}

bool ggml_cuda_rms_norm_scale_mul_try(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * scl,
                                      ggml_tensor * mul) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == scl ? mul->src[1] : mul->src[0];
    const int64_t K = x->ne[0];
    float scale_out, bias;
    memcpy(&scale_out, (const float *) scl->op_params + 0, sizeof(float));
    memcpy(&bias,      (const float *) scl->op_params + 1, sizeof(float));
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || ggml_nrows(x) != 1 || bias != 0.0f ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(w, x) ||
            !ggml_are_same_shape(mul, x) || K < 1024 || K > 1024*8) {
        return false;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    rms_norm_scale_mul<1024><<<1, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul->data,
        (int) K, eps, scale_out);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- gemma4 decode chain fusions (p100-series 2026-10-11, phase GC census: 993 kernels per GPU per token on v5.17) ----

// h1 cache entry for `mul` if an h1-eligible matvec reads it within the next 16 nodes (else nullptr: no prep needed)
static mmvq_h1_entry * mmvq_h1_entry_for(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i_mul, const ggml_tensor * mul) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_H1");
        const char * n = getenv("GGML_CUDA_MMVQ_H1_NORM");
        return (!s || atoi(s) != 0) && (!n || atoi(n) != 0);
    }();
    static const bool q8_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_Q8"); return !s || atoi(s) != 0; }();
    const int64_t K = mul->ne[0];
    if (!enabled || K % H1_QK != 0 || K < 32*H1_QK) {
        return nullptr;
    }
    for (int j = i_mul + 1; j < cgraph->n_nodes && j < i_mul + 16; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && mmvq_h1_key(t->src[1]) == mul && t->src[0]->ne[0] == K &&
                (t->src[0]->type == GGML_TYPE_Q4_0 || (t->src[0]->type == GGML_TYPE_Q8_0 && q8_on))) {
            bool hit;
            return &mmvq_h1_slot(ctx, mul, mul->data, 1, K, 0, hit);
        }
    }
    return nullptr;
}

// The end of a gemma4 layer and the start of the next, one row (decode), in ONE launch instead of four:
//   o2 = rms_norm(x0)*w2 + mlp            (post_ffw_norm_2 + combine;   was rms_norm_f32<1024, mul, add>)
//   o5 = rms_norm(o2)*w3 + ao             (post_ffw_norm + residual;    was rms_norm_f32<1024, mul, add>)
//   o6 = o5 * s[0]                        (layer_output_scale;          was k_bin_bcast)
//   o8 = rms_norm(o6)*w4  (+ h1 prep)     (next layer's attn_norm;      was mmvq_h1_rms_norm_mul)
// Every value comes from the same expressions in the same order as those kernels (register path, per-thread accumulation over
// cols tid + u*1024, block_reduce<SUM, 1024>), so all four outputs are bit-identical. All inputs are read into registers before the
// first block reduction (a barrier), so outputs may alias inputs freely.
template <bool H1>
static __global__ void __launch_bounds__(1024) layer_tail_f32(
        const float * x0, const float * __restrict__ w2, const float * mlp, const float * __restrict__ w3, const float * ao,
        const float * __restrict__ s, const float * __restrict__ w4, float * o2, float * o5, float * o6, float * o8,
        __half * __restrict__ XH, float * __restrict__ XS, const int ncols, const float eps0, const float eps3, const float eps7) {
    constexpr int bs = 1024, R = 8;
    __shared__ float s_a[32], s_b[32], s_c[32];
    const int tid = threadIdx.x;
    float xv[R], mv[R], av[R];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*bs;
        xv[u] = col < ncols ? x0[col] : 0.0f;
        tmp += xv[u] * xv[u];
        mv[u] = col < ncols ? mlp[col] : 0.0f;
        av[u] = col < ncols ? ao[col] : 0.0f;
    }
    const float sc = s[0];
    tmp = block_reduce<block_reduce_method::SUM, bs>(tmp, s_a);
    const float scale0 = rsqrtf(tmp / ncols + eps0);

    float t2 = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*bs;
        if (col < ncols) {
            const float v = scale0 * xv[u] * w2[col] + mv[u];
            o2[col] = v;
            xv[u] = v;
        } else {
            xv[u] = 0.0f;
        }
        t2 += xv[u] * xv[u];
    }
    t2 = block_reduce<block_reduce_method::SUM, bs>(t2, s_b);
    const float scale3 = rsqrtf(t2 / ncols + eps3);

    float t3 = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*bs;
        if (col < ncols) {
            const float v = scale3 * xv[u] * w3[col] + av[u];
            o5[col] = v;
            const float v6 = v * sc;
            o6[col] = v6;
            xv[u] = v6;
        } else {
            xv[u] = 0.0f;
        }
        t3 += xv[u] * xv[u];
    }
    t3 = block_reduce<block_reduce_method::SUM, bs>(t3, s_c);
    const float scale7 = rsqrtf(t3 / ncols + eps7);

#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*bs;
        if (col < ncols) {
            const float v = scale7 * xv[u] * w4[col];
            o8[col] = v;
            if constexpr (H1) {   // ncols % 32 == 0: whole warps
                float m = fabsf(v);
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                }
                const float sm = m > 0.0f ? m : 1.0f;
                XH[col] = __float2half(v/sm);
                if ((col & 31) == 0) {
                    XS[col/32] = sm;
                }
            }
        }
    }
}

int ggml_cuda_layer_tail_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // GGML_CUDA_FUSE_TAIL=0: back to four launches (A/B)
    static const bool on = [] { const char * s = getenv("GGML_CUDA_FUSE_TAIL"); return !s || atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!on || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || i + 8 >= cgraph->n_nodes) {
        return 0;
    }
    static const ggml_op ops[9] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD,
                                    GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL };
    ggml_tensor * n[9];
    for (int k = 0; k < 9; ++k) {
        n[k] = cgraph->nodes[i + k];
        if (n[k]->op != ops[k] || n[k]->type != GGML_TYPE_F32 || !(n[k]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return 0;
        }
    }
    const int64_t K = n[0]->ne[0];
    auto row = [K](const ggml_tensor * t) {
        return t && t->type == GGML_TYPE_F32 && t->ne[0] == K && ggml_nrows(t) == 1 && ggml_is_contiguous(t);
    };
    // the other operand of a binary node whose one operand is `a`
    auto other = [](const ggml_tensor * b, const ggml_tensor * a) -> const ggml_tensor * {
        return b->src[0] == a ? b->src[1] : (b->src[1] == a ? b->src[0] : nullptr);
    };
    const ggml_tensor * x0 = n[0]->src[0];
    const ggml_tensor * w2 = other(n[1], n[0]);
    const ggml_tensor * ml = other(n[2], n[1]);
    const ggml_tensor * w3 = other(n[4], n[3]);
    const ggml_tensor * ao = other(n[5], n[4]);
    const ggml_tensor * s  = other(n[6], n[5]);
    const ggml_tensor * w4 = other(n[8], n[7]);
    if (K < 1024 || K > 1024*8 || !row(x0) || !row(w2) || !row(ml) || !row(w3) || !row(ao) || !row(w4) || !s ||
            s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1 || n[3]->src[0] != n[2] || n[7]->src[0] != n[6]) {
        return 0;
    }
    for (int k = 0; k < 9; ++k) {
        if (!row(n[k])) {
            return 0;
        }
    }
    // the elided intermediates (norm outputs and the products feeding the adds) are never written: nothing else may read them
    for (int k : { 0, 1, 3, 4, 7 }) {
        if (ggml_node_get_use_count(cgraph, i + k) != 1 || (n[k]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }
    float e0, e3, e7;
    memcpy(&e0, n[0]->op_params, sizeof(float));
    memcpy(&e3, n[3]->op_params, sizeof(float));
    memcpy(&e7, n[7]->op_params, sizeof(float));
    mmvq_h1_entry * e = mmvq_h1_entry_for(ctx, cgraph, i + 8, n[8]);
    cudaStream_t stream = ctx.stream();
    if (e) {
        __half * xh = (__half *) e->buf;
        float  * xs = (float *) ((char *) e->buf + (size_t) K*sizeof(__half));
        layer_tail_f32<true><<<1, 1024, 0, stream>>>((const float *) x0->data, (const float *) w2->data, (const float *) ml->data,
            (const float *) w3->data, (const float *) ao->data, (const float *) s->data, (const float *) w4->data,
            (float *) n[2]->data, (float *) n[5]->data, (float *) n[6]->data, (float *) n[8]->data, xh, xs, (int) K, e0, e3, e7);
    } else {
        layer_tail_f32<false><<<1, 1024, 0, stream>>>((const float *) x0->data, (const float *) w2->data, (const float *) ml->data,
            (const float *) w3->data, (const float *) ao->data, (const float *) s->data, (const float *) w4->data,
            (float *) n[2]->data, (float *) n[5]->data, (float *) n[6]->data, (float *) n[8]->data, nullptr, nullptr, (int) K, e0, e3, e7);
    }
    CUDA_CHECK(cudaGetLastError());
    return 8;
}

// rms_norm (no weight) of one token's heads written straight into a KV cache row (gemma4 V: rms_norm -> view -> set_rows).
// Per head the register path of rms_norm_f32<BS> (BS = 256 below 1024 columns, as its launcher picks), then the cast set_rows
// does: bit-identical cache contents, one launch instead of two.
template <int BS, typename T>
static __global__ void __launch_bounds__(BS) rms_norm_set_rows_1tok(const float * __restrict__ x, const int64_t sx1,
        T * __restrict__ dst, const int64_t * __restrict__ idx, const int64_t dst_s1, const int ncols, const float eps) {
    constexpr int R = 8;
    __shared__ float s_sum[32];
    const int h = blockIdx.x, tid = threadIdx.x;
    x += h*sx1;
    T * d = dst + idx[0]*dst_s1 + (int64_t) h*ncols;
    float xv[R];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*BS;
        xv[u] = col < ncols ? x[col] : 0.0f;
        tmp += xv[u] * xv[u];
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);
#pragma unroll
    for (int u = 0; u < R; ++u) {
        const int col = tid + u*BS;
        if (col < ncols) {
            d[col] = ggml_cuda_cast<T>(scale * xv[u]);
        }
    }
}

bool ggml_cuda_rms_norm_set_rows_try(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * view,
                                     const ggml_tensor * set_rows) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * idx = set_rows->src[1];
    const int64_t ncols = rms->ne[0];
    if (cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || rms->op != GGML_OP_RMS_NORM || view->op != GGML_OP_VIEW ||
            set_rows->op != GGML_OP_SET_ROWS || set_rows->src[0] != view || view->src[0] != rms || x->type != GGML_TYPE_F32 ||
            rms->type != GGML_TYPE_F32 || rms->ne[2] != 1 || rms->ne[3] != 1 || x->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(rms) || !ggml_is_contiguous(view) || view->ne[0] != rms->ne[0]*rms->ne[1] || view->ne[1] != 1 ||
            idx->type != GGML_TYPE_I64 || ggml_nelements(idx) != 1 || ncols > 8*1024 || set_rows->ne[0] != view->ne[0] ||
            (set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_F32) || set_rows->nb[0] != ggml_type_size(set_rows->type)) {
        return false;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int64_t sx1 = x->nb[1]/sizeof(float);
    const int64_t ds1 = set_rows->nb[1]/ggml_type_size(set_rows->type);
    const dim3 grid((unsigned) rms->ne[1]);
    cudaStream_t stream = ctx.stream();
    const int64_t * id = (const int64_t *) idx->data;
    if (set_rows->type == GGML_TYPE_F16) {
        if (ncols < 1024) {
            rms_norm_set_rows_1tok<256, half><<<grid, 256, 0, stream>>>((const float *) x->data, sx1, (half *) set_rows->data, id, ds1, (int) ncols, eps);
        } else {
            rms_norm_set_rows_1tok<1024, half><<<grid, 1024, 0, stream>>>((const float *) x->data, sx1, (half *) set_rows->data, id, ds1, (int) ncols, eps);
        }
    } else {
        if (ncols < 1024) {
            rms_norm_set_rows_1tok<256, float><<<grid, 256, 0, stream>>>((const float *) x->data, sx1, (float *) set_rows->data, id, ds1, (int) ncols, eps);
        } else {
            rms_norm_set_rows_1tok<1024, float><<<grid, 1024, 0, stream>>>((const float *) x->data, sx1, (float *) set_rows->data, id, ds1, (int) ncols, eps);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---- more gemma4 decode merges (2026-10-11, second batch) ----

// (d) GEGLU of one token's expert slots that feed an integer-path MUL_MAT_ID (gemma4 ffn_down_exps, K 320 < the h1 minimum):
// write the GLU result AND its q8_1 quantization into the mul_mat_vec_q activation cache, so the matvec skips its quantize
// launch. Values: unary_gated_op_kernel's expression, then quantize_q8_1's exactly (padding columns quantize 0) -> same bits.
static __global__ void geglu_q8_1(const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
                                  block_q8_1 * __restrict__ y, const int ne00, const int64_t ox, const int64_t og, const int ne0p) {
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (i0 >= ne0p) {
        return;   // ne0p % 32 == 0: whole warps
    }
    float xi = 0.0f;
    if (i0 < ne00) {
        xi = ggml_cuda_cast<float>(ggml_cuda_op_gelu_single(ggml_cuda_cast<float>(x[row*ox + i0])) * ggml_cuda_cast<float>(g[row*og + i0]));
        dst[(int64_t) row*ne00 + i0] = xi;
    }
    const int64_t i_cont = (int64_t) row*ne0p + i0;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;
    if (iqs > 0) {
        return;
    }
    y[ib].ds = make_half2(d, sum);
}

bool ggml_cuda_geglu_q8_1_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // GGML_CUDA_FUSE_GLU_Q8=0 disables
    static const bool on = [] { const char * s = getenv("GGML_CUDA_FUSE_GLU_Q8"); return !s || atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_tensor * glu = cgraph->nodes[i];
    if (!on || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_GEGLU) {
        return false;
    }
    const ggml_tensor * a = glu->src[0], * b = glu->src[1];
    if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) ||
            a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || !ggml_is_contiguous_1(a) || !ggml_is_contiguous_1(b) ||
            glu->ne[2] != 1 || glu->ne[3] != 1 || a->ne[0] != glu->ne[0] || b->ne[0] != glu->ne[0] || glu->ne[1] > 8 ||
            a->ne[1] != glu->ne[1] || b->ne[1] != glu->ne[1] || glu->ne[0] % QK8_1 != 0) {
        return false;
    }
    // the consumer: the next computed node, a MUL_MAT_ID on this GLU whose K is below the h1 minimum (-> integer mmvq + q8_1)
    const ggml_tensor * mm = nullptr;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 4; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] == glu) {
            mm = t;
            break;
        }
        if (!ggml_is_empty(t) && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) {
            return false;
        }
    }
    const int64_t K = glu->ne[0];
    if (!mm || !ggml_is_quantized(mm->src[0]->type) || K >= 32*H1_QK || mm->src[0]->ne[0] != K) {
        return false;
    }
    const int64_t ne0p = GGML_PAD(K, MATRIX_ROW_PADDING);
    const size_t need = (size_t) glu->ne[1]*ne0p*sizeof(block_q8_1)/QK8_1;
    const int dev = ggml_cuda_get_device();
    if (ctx.mmvq_q8_1_cap[dev] < need) {
        if (ctx.mmvq_q8_1_ptr[dev] != nullptr) {
            CUDA_CHECK(cudaFree(ctx.mmvq_q8_1_ptr[dev]));
            ctx.mmvq_q8_1_ptr[dev] = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&ctx.mmvq_q8_1_ptr[dev], need + 16));   // + 16: same slack as mul_mat_vec_q's allocation
        ctx.mmvq_q8_1_cap[dev] = need;
    }
    const dim3 grid((unsigned) ((ne0p + 255)/256), (unsigned) glu->ne[1]);
    geglu_q8_1<<<grid, 256, 0, ctx.stream()>>>((const float *) a->data, (const float *) b->data, (float *) glu->data,
        (block_q8_1 *) ctx.mmvq_q8_1_ptr[dev], (int) K, (int64_t) (a->nb[1]/sizeof(float)), (int64_t) (b->nb[1]/sizeof(float)), (int) ne0p);
    CUDA_CHECK(cudaGetLastError());
    ctx.mmvq_q8_1_src1[dev] = glu;
    ctx.mmvq_q8_1_data[dev] = glu->data;
    ctx.mmvq_q8_1_type[dev] = mm->src[0]->type;
    ctx.mmvq_q8_1_need[dev] = need;
    return true;
}

// (e) the attention output -> output projection: flash-attention's combine kernel writes the h1 prep of its result (each warp
// writes 32 consecutive values = one h1 block). _wants: called at the FLASH_ATTN_EXT node, records the request when an h1 matvec
// reads the result; _reserve: called by launch_fattn when it actually runs the combine kernel for one token.
bool ggml_cuda_mmvq_h1_fa_wants(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool on = [] { const char * s = getenv("GGML_CUDA_FUSE_FA_H1"); return !s || atoi(s) != 0; }();
    static const bool q8_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_Q8"); return !s || atoi(s) != 0; }();
    ctx.fa_h1_dst = nullptr;
    const ggml_tensor * fa = cgraph->nodes[i];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!on || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || fa->type != GGML_TYPE_F32 || fa->ne[2] != 1 || fa->ne[3] != 1 ||
            !ggml_is_contiguous(fa)) {
        return false;
    }
    const int64_t K = fa->ne[0]*fa->ne[1];
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 5; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_MUL_MAT && mmvq_h1_key(t->src[1]) == fa && t->src[1]->data == fa->data && t->src[1]->ne[0] == K &&
                ggml_nrows(t->src[1]) == 1 && t->src[0]->ne[0] == K && K % H1_QK == 0 && K >= 32*H1_QK &&
                (t->src[0]->type == GGML_TYPE_Q4_0 || (t->src[0]->type == GGML_TYPE_Q8_0 && q8_on))) {
            ctx.fa_h1_dst = fa;
            return true;
        }
        if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE && t->op != GGML_OP_CONT) {
            return false;
        }
    }
    return false;
}

bool ggml_cuda_mmvq_h1_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * key, const int64_t K, __half ** xh, float ** xs) {
    bool hit;
    mmvq_h1_entry & e = mmvq_h1_slot(ctx, key, key->data, 1, K, 0, hit);
    *xh = (__half *) e.buf;
    *xs = (float *) ((char *) e.buf + (size_t) K*sizeof(__half));
    return true;
}

// ---- h1s: fp16 one-token MUL_MAT_ID for SHORT K (2026-10-12, optimization item #2) ----
// gemma4's ffn_down_exps is split by K under -sm tensor (320 / 384 per GPU = 10 / 12 q4_0 blocks per row), below the 32 blocks
// the h1 layout needs, so it ran on the integer path (quantize_q8_1 + mul_mat_vec_q short-K, ~215 GB/s). Here ONE THREAD OWNS ONE
// ROW: a block stages RB consecutive rows of the slot's expert (contiguous bytes, 16-byte coalesced loads) in shared memory, builds
// the fp16 activation in shared memory (per 32-value block x / max|x|, h1's numerics), and each thread runs h1's HFMA2 chain over
// its row. Optional GLU mode computes op(gate) * up on the fly from the two GLU operands, so the GLU node needs no launch.
// GGML_CUDA_MMVQ_H1S=0 disables, GGML_CUDA_MMVQ_H1S_GLU=0 keeps the GLU launch, GGML_CUDA_MMVQ_H1S_RB=32|64|128 rows per chunk,
// GGML_CUDA_MMVQ_H1S_CPB=1..8 chunks per block.
template <int RB, int GLU>   // GLU: 0 = plain fp32 activation, 1 = GEGLU(gate, up), 2 = SWIGLU(gate, up)
static __global__ void __launch_bounds__(RB) mmvq_h1s_id(
        const uint8_t * __restrict__ W, const int row_bytes, const int64_t nb2, const int32_t * __restrict__ ids,
        const float * __restrict__ X, const int64_t sx, const float * __restrict__ U, const int64_t su,
        float * __restrict__ dst, const int64_t sy, const int M, const int K, const int cpb) {
    extern __shared__ __align__(16) uint8_t h1s_smem[];
    const int nb = K / H1_QK;
    float  * xs = (float *) h1s_smem;                         // nb scales (<= 16)
    __half * xh = (__half *) (h1s_smem + 64);                  // K halfs (<= 512)
    uint8_t * ws = h1s_smem + 64 + 1024;                       // RB rows of weights (one chunk)
    const int tid = threadIdx.x, lane = tid % 32;
    const int slot = blockIdx.y;
    // this block: chunks blockIdx.x*cpb .. +cpb-1 (RB rows each) of the slot's expert; the activation is built once for all of them,
    // and chunk c+1's weights are loaded into registers while chunk c is computed from shared memory
    const int c0 = blockIdx.x*cpb;
    const int nchunks = (M + RB - 1)/RB;
    const int c_end = min(c0 + cpb, nchunks);
    const uint8_t * wexp = W + (int64_t) ids[slot]*nb2;

    constexpr int MAXU = (14*128)/RB < 28 ? (14*128)/RB : 28;   // uint4 per thread per chunk (K 384 at RB 128 needs 13.5)
    uint4 v[MAXU];
    auto load = [&](const int c, int & n16, bool & vec) {
        const int nrows = min(RB, M - c*RB);
        const uint8_t * wb = wexp + (int64_t) c*RB*row_bytes;
        const int nbytes = nrows*row_bytes;
        vec = ((uintptr_t) wb & 15) == 0 && (nbytes & 15) == 0 && nbytes/16 <= MAXU*RB;
        n16 = nbytes/16;
        if (vec) {
            const uint4 * src = (const uint4 *) wb;
#pragma unroll
            for (int u = 0; u < MAXU; ++u) {
                const int j = tid + u*RB;
                if (j < n16) {
                    v[u] = __ldg(src + j);
                }
            }
        }
    };
    auto store = [&](const int c, const int n16, const bool vec) {
        if (vec) {
            uint4 * d = (uint4 *) ws;
#pragma unroll
            for (int u = 0; u < MAXU; ++u) {
                const int j = tid + u*RB;
                if (j < n16) {
                    d[j] = v[u];
                }
            }
        } else {   // unaligned / oversized chunk: plain 2-byte copy (rows are 2-byte aligned)
            const uint16_t * src = (const uint16_t *) (wexp + (int64_t) c*RB*row_bytes);
            uint16_t * d = (uint16_t *) ws;
            const int n2 = min(RB, M - c*RB)*row_bytes/2;
            for (int j = tid; j < n2; j += RB) {
                d[j] = src[j];
            }
        }
    };

    int n16; bool vec;
    load(c0, n16, vec);

    // activation -> fp16 blocks. Thread t owns elements t + RB*k: all its loads are issued first, and since RB is a multiple of 32 each
    // warp's 32 elements of one k are one h1 block (warp max = the block scale).
    {
        constexpr int MAXK = 512/RB;
        float a[MAXK], g[MAXK];
#pragma unroll
        for (int k = 0; k < MAXK; ++k) {
            const int i = tid + RB*k;
            if (i < K) {
                a[k] = X[slot*sx + i];
                if constexpr (GLU != 0) {
                    g[k] = U[slot*su + i];
                }
            }
        }
#pragma unroll
        for (int k = 0; k < MAXK; ++k) {
            const int i = tid + RB*k;
            if (i < K) {   // K % 32 == 0: whole warps
                float x;
                if constexpr (GLU == 1) {
                    x = ggml_cuda_op_gelu_single(a[k]) * g[k];
                } else if constexpr (GLU == 2) {
                    x = ggml_cuda_op_silu_single(a[k]) * g[k];
                } else {
                    x = a[k];
                }
                float m = fabsf(x);
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) {
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                }
                const float sc = m > 0.0f ? m : 1.0f;
                xh[i] = __float2half(x/sc);
                if (lane == 0) {
                    xs[i/H1_QK] = sc;
                }
            }
        }
    }

    const half2 * xh2 = (const half2 *) xh;
    for (int c = c0; c < c_end; ++c) {
        store(c, n16, vec);
        __syncthreads();
        if (c + 1 < c_end) {
            load(c + 1, n16, vec);   // in flight while this chunk is computed
        }
        const int nrows = min(RB, M - c*RB);
        if (tid < nrows) {
            const uint8_t * wr = ws + tid*row_bytes;
            float acc = 0.0f;
            for (int b = 0; b < nb; ++b) {
                const uint16_t * blk = (const uint16_t *) (wr + b*18);
                const float d = __half2float(__ushort_as_half(blk[0]));
                half2 h = __float2half2_rn(0.0f);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t q = (uint32_t) blk[1 + 2*i] | ((uint32_t) blk[2 + 2*i] << 16);
                    half2 lo, hi;
                    mmvq_h1_nib(q & 0x0f0f0f0fu, lo, hi);
                    h = __hfma2(lo, xh2[b*16 + 2*i], h);
                    h = __hfma2(hi, xh2[b*16 + 2*i + 1], h);
                    mmvq_h1_nib((q >> 4) & 0x0f0f0f0fu, lo, hi);
                    h = __hfma2(lo, xh2[b*16 + 8 + 2*i], h);
                    h = __hfma2(hi, xh2[b*16 + 8 + 2*i + 1], h);
                }
                const float2 f = __half22float2(h);
                acc += xs[b]*(d*(f.x + f.y));
            }
            dst[slot*sy + c*RB + tid] = acc;
        }
        __syncthreads();   // before the next chunk overwrites ws
    }
}

// h1s v7 (2026-10-12, microbench tilegemm/moebw.cu): per-warp streaming. The v1-v4 one-thread-per-row kernels sat at ~19-21 us for gemma4's
// down (the pure load pattern takes 10 us): per row one dependent 16-HFMA2 chain, nothing to overlap it with. Here a CTA of WPB warps builds
// the slot's fp16 activation once; lane l owns 32-value block (l % NB) of row (l / NB) of a step of RPS = 32/NB rows, keeps that block's 16
// activation half2 in registers, and computes it for G steps at once (G independent chains, h1's rows-per-warp trick). A warp streams NG
// groups of G*RPS consecutive rows through a double-buffered staging buffer (next group loading while this one computes); row sums go through
// a small scratch in block order (= v4's order). moebw: K 320 18.9 -> 13.2 us, K 384 20.8 -> 16.6 us (G 3, NG 4, WPB 4 best of 14).
template <int NB, int GLU>
static __global__ void __launch_bounds__(128) mmvq_h1s7_id(
        const uint8_t * __restrict__ W, const int64_t nb2, const int32_t * __restrict__ ids,
        const float * __restrict__ X, const int64_t sx, const float * __restrict__ U, const int64_t su,
        float * __restrict__ dst, const int64_t sy, const int M) {
    constexpr int G = 3, NG = 4, WPB = 4, K = NB*H1_QK, ROWB = NB*18;
    constexpr int RPS = 32/NB, GROWS = G*RPS, GB = GROWS*ROWB, NU4 = (GB + 15 + 15)/16, LPL = (NU4 + 31)/32;
    __shared__ __align__(16) uint8_t stage[WPB][2][NU4*16];
    __shared__ __align__(16) __half xh[K];
    __shared__ float xs[NB];
    __shared__ float red[WPB][G][32];
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32, slot = blockIdx.y;
    constexpr int NT = WPB*32, MAXK = (K + NT - 1)/NT;

    float a[MAXK], g[MAXK];
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int i = tid + NT*k;
        if (i < K) {
            a[k] = X[slot*sx + i];
            if constexpr (GLU != 0) {
                g[k] = U[slot*su + i];
            }
        }
    }
    const int row0 = (blockIdx.x*WPB + wid)*NG*GROWS;
    const uint8_t * wexp = W + (int64_t) ids[slot]*nb2;
    // group gr: rows [row0 + gr*GROWS, +GROWS) clipped to M; loads only the bytes of the valid rows (+ the <= 15-byte alignment over-read,
    // covered by the tensor padding -- see mmvq_h1s_ok); a group wholly past M loads nothing (warp-uniform)
    auto ld = [&](const int gr, uint4 * v, int & mis) {
        const int r = row0 + gr*GROWS;
        mis = 0;
        if (r >= M) {
            return;
        }
        const int nbytes = min(GROWS, M - r)*ROWB;
        const uintptr_t ad = (uintptr_t) (wexp + (int64_t) r*ROWB);
        mis = (int) (ad & 15);
        const int nu4 = (mis + nbytes + 15)/16;
        const uint4 * p = (const uint4 *) (ad - mis);
#pragma unroll
        for (int j = 0; j < LPL; ++j) {
            if (lane + 32*j < nu4) {
                v[j] = __ldg(p + lane + 32*j);
            }
        }
    };
    auto st = [&](const int buf, const uint4 * v) {
#pragma unroll
        for (int j = 0; j < LPL; ++j) {
            if (lane + 32*j < NU4) {
                ((uint4 *) stage[wid][buf])[lane + 32*j] = v[j];
            }
        }
    };
    uint4 v[LPL];
    int mis;
    ld(0, v, mis);

#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int i = tid + NT*k;
        if (i < K) {   // K % 32 == 0: whole warps
            float x;
            if constexpr (GLU == 1) {
                x = ggml_cuda_op_gelu_single(a[k]) * g[k];
            } else if constexpr (GLU == 2) {
                x = ggml_cuda_op_silu_single(a[k]) * g[k];
            } else {
                x = a[k];
            }
            float m = fabsf(x);
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            }
            const float sc = m > 0.0f ? m : 1.0f;
            xh[i] = __float2half(x/sc);
            if (lane == 0) {
                xs[i/H1_QK] = sc;
            }
        }
    }
    __syncthreads();

    const int rr = lane / NB, b = lane % NB;
    const bool act = rr < RPS;
    half2 xv[16];
    float sb = 0.0f;
    if (act) {
        const uint4 * xp = (const uint4 *) (xh + b*H1_QK);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 q = xp[i];
            xv[4*i + 0] = *(const half2 *) &q.x; xv[4*i + 1] = *(const half2 *) &q.y;
            xv[4*i + 2] = *(const half2 *) &q.z; xv[4*i + 3] = *(const half2 *) &q.w;
        }
        sb = xs[b];
    }
    st(0, v);
    int mis_cur = mis;
    __syncwarp();

    for (int gr = 0; gr < NG; ++gr) {
        if (row0 + gr*GROWS >= M) {
            break;   // warp-uniform
        }
        int mis_nx = 0;
        if (gr + 1 < NG) {
            ld(gr + 1, v, mis_nx);
        }
        float p[G];
        if (act) {
            uint32_t w[G][5];
            bool hi[G];
#pragma unroll
            for (int q = 0; q < G; ++q) {
                const int off = mis_cur + (q*RPS + rr)*ROWB + b*18;   // 2-byte aligned; rows past M read stale staging, never written
                const uint32_t * wd = (const uint32_t *) (stage[wid][gr & 1] + (off & ~3));
                hi[q] = off & 2;
#pragma unroll
                for (int i = 0; i < 5; ++i) {
                    w[q][i] = wd[i];
                }
            }
            half2 h[G];
#pragma unroll
            for (int q = 0; q < G; ++q) {
                h[q] = __float2half2_rn(0.0f);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int q = 0; q < G; ++q) {
                    const uint32_t qq = __funnelshift_rc(w[q][i], w[q][i + 1], hi[q] ? 32 : 16);
                    half2 lo, hh;
                    mmvq_h1_nib(qq & 0x0f0f0f0fu, lo, hh);
                    h[q] = __hfma2(lo, xv[2*i], h[q]);
                    h[q] = __hfma2(hh, xv[2*i + 1], h[q]);
                    mmvq_h1_nib((qq >> 4) & 0x0f0f0f0fu, lo, hh);
                    h[q] = __hfma2(lo, xv[8 + 2*i], h[q]);
                    h[q] = __hfma2(hh, xv[8 + 2*i + 1], h[q]);
                }
            }
#pragma unroll
            for (int q = 0; q < G; ++q) {
                const float d = __half2float(__ushort_as_half((unsigned short) (hi[q] ? w[q][0] >> 16 : w[q][0] & 0xffff)));
                const float2 f = __half22float2(h[q]);
                p[q] = sb*(d*(f.x + f.y));
            }
        } else {
#pragma unroll
            for (int q = 0; q < G; ++q) {
                p[q] = 0.0f;
            }
        }
#pragma unroll
        for (int q = 0; q < G; ++q) {
            red[wid][q][lane] = p[q];
        }
        __syncwarp();
        if (lane < GROWS) {
            const int q = lane / RPS, r2 = lane % RPS, r = row0 + gr*GROWS + lane;
            if (r < M) {
                float acc = 0.0f;
                for (int k = 0; k < NB; ++k) {
                    acc += red[wid][q][r2*NB + k];
                }
                dst[slot*sy + r] = acc;
            }
        }
        __syncwarp();
        if (gr + 1 < NG) {
            st((gr + 1) & 1, v);
        }
        mis_cur = mis_nx;
        __syncwarp();
    }
}

static bool mmvq_h1s_on() {
    static const bool on = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_H1S");
        const char * h = getenv("GGML_CUDA_MMVQ_H1");
        return (!s || atoi(s) != 0) && (!h || atoi(h) != 0);
    }();
    return on;
}

// shape check shared by the matvec and the GLU deferral (they must agree: a deferred GLU is only computed inside the matvec)
static bool mmvq_h1s_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!mmvq_h1s_on() || !ids) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t row_bytes = (K/H1_QK)*18;
    const int64_t n_used = ids->ne[0];
    return cc < GGML_CUDA_CC_VOLTA && !GGML_CUDA_CC_IS_AMD(cc) && src0->type == GGML_TYPE_Q4_0 && K % H1_QK == 0 && K >= 64 &&
           K <= 512 && src0->nb[0] == 18 && src0->nb[1] == (size_t) row_bytes && src0->ne[3] == 1 && ((uintptr_t) src0->data % 2) == 0 &&
           src0->nb[2] == (size_t) (M*row_bytes) &&
           // v7 stages from the 16-byte-aligned address at or below each row group: up to 15 bytes past the last row of the last expert,
           // covered when the tensor owns its allocation and K is not a multiple of MATRIX_ROW_PADDING (last row padded), or when nothing
           // is ever misaligned
           ((row_bytes % 16 == 0 && (uintptr_t) src0->data % 16 == 0) || (src0->view_src == nullptr && K % MATRIX_ROW_PADDING != 0)) &&
           ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ids->nb[0] == sizeof(int32_t) && n_used <= 65535 &&
           src1->type == GGML_TYPE_F32 && src1->ne[0] == K && src1->nb[0] == sizeof(float) && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           (src1->ne[1] == 1 || src1->ne[1] == n_used) && src1->nb[1] % sizeof(float) == 0 &&
           dst->type == GGML_TYPE_F32 && dst->ne[0] == M && dst->ne[1] == n_used && dst->ne[2] == 1 && dst->nb[0] == sizeof(float) &&
           dst->nb[1] % sizeof(float) == 0;
}

// GLU node whose only reader is an h1s matvec: skip its launch, the matvec computes it (ctx.h1s_glu). nodes[i] = the GLU.
bool ggml_cuda_mmvq_h1s_glu_defer(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1S_GLU"); return !s || atoi(s) != 0; }();
    const ggml_tensor * glu = cgraph->nodes[i];
    if (!on || glu->op != GGML_OP_GLU) {
        return false;
    }
    const ggml_glu_op op = ggml_get_glu_op(glu);
    const ggml_tensor * a = glu->src[0], * b = glu->src[1];
    if ((op != GGML_GLU_OP_GEGLU && op != GGML_GLU_OP_SWIGLU) || !a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            glu->type != GGML_TYPE_F32 || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || a->ne[0] != glu->ne[0] ||
            b->ne[0] != glu->ne[0] || a->ne[1] != glu->ne[1] || b->ne[1] != glu->ne[1] || a->ne[2] != 1 || b->ne[2] != 1 ||
            a->nb[1] % sizeof(float) != 0 || b->nb[1] % sizeof(float) != 0 || ggml_node_get_use_count(cgraph, i) != 1 ||
            (glu->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 4; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] == glu) {
            if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE) || !mmvq_h1s_ok(t->src[0], glu, t->src[2], t)) {
                return false;
            }
            ctx.h1s_glu = glu;
            return true;
        }
        if (!ggml_is_empty(t) && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE) {
            return false;
        }
    }
    return false;
}

bool ggml_cuda_mmvq_h1s_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            const ggml_tensor * ids, ggml_tensor * dst) {
    const bool deferred = ctx.h1s_glu != nullptr && ctx.h1s_glu == src1;
    if (!mmvq_h1s_ok(src0, src1, ids, dst)) {
        GGML_ASSERT(!deferred && "h1s: a GLU was deferred to a matvec that does not take it");
        return false;
    }
    // rows per chunk (= threads per block): 128 default (H1C in-model: 128 20.0 us, 64 22.4, 32 27.4 -- per-block overhead dominates)
    static const int rb = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1S_RB"); const int v = s ? atoi(s) : 128; return v == 32 || v == 64 ? v : 128; }();
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int row_bytes = (int) ((K/H1_QK)*18);
    const int64_t n_used = ids->ne[0];
    const int64_t sy = dst->nb[1]/sizeof(float);
    const float * X; const float * Uv = nullptr; int64_t sx, su = 0; int glu = 0;
    if (deferred) {
        const ggml_tensor * a = src1->src[0], * b = src1->src[1];
        glu = ggml_get_glu_op(src1) == GGML_GLU_OP_GEGLU ? 1 : 2;
        X = (const float *) a->data; sx = a->nb[1]/sizeof(float);
        Uv = (const float *) b->data; su = b->nb[1]/sizeof(float);
        if (src1->ne[1] == 1) { sx = 0; su = 0; }
        ctx.h1s_glu = nullptr;
    } else {
        X = (const float *) src1->data;
        sx = src1->ne[1] == 1 ? 0 : (int64_t) (src1->nb[1]/sizeof(float));
    }
    // chunks per block: 1 by default. H1D in-model (gemma4 ffn_down_exps, 128-row chunks): CPB 1 19.4 us, 2 21.9, 3 25.2, 4 25.0 -- block
    // count (parallelism) beats reusing the activation and prefetching the next chunk; all variants sit near ~200 GB/s (open question)
    static const int cpb = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1S_CPB"); const int v = s ? atoi(s) : 1; return v >= 1 && v <= 8 ? v : 1; }();
    const int nchunks = (int) ((M + rb - 1)/rb);
    const dim3 grid((unsigned) ((nchunks + cpb - 1)/cpb), (unsigned) n_used);
    const size_t smem = 64 + 1024 + (size_t) rb*row_bytes;
    cudaStream_t stream = ctx.stream();
    static const bool v7 = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1S_V7"); return !s || atoi(s) != 0; }();
    const int nb = (int) (K/H1_QK);
    if (v7 && nb >= 8 && nb <= 16) {
        // rows per CTA = WPB 4 x NG 4 x G 3 x RPS (32/NB)
        const int rows_cta = 4*4*3*(32/nb);
        const dim3 grid7((unsigned) ((M + rows_cta - 1)/rows_cta), (unsigned) n_used);
        auto go7 = [&](auto nb_tag, auto glu_tag) {
            constexpr int NBv = decltype(nb_tag)::value, Gv = decltype(glu_tag)::value;
            mmvq_h1s7_id<NBv, Gv><<<grid7, 128, 0, stream>>>((const uint8_t *) src0->data, (int64_t) src0->nb[2],
                (const int32_t *) ids->data, X, sx, Uv, su, (float *) dst->data, sy, (int) M);
        };
        auto by_glu7 = [&](auto nb_tag) {
            switch (glu) {
                case 1:  go7(nb_tag, std::integral_constant<int, 1>{}); break;
                case 2:  go7(nb_tag, std::integral_constant<int, 2>{}); break;
                default: go7(nb_tag, std::integral_constant<int, 0>{}); break;
            }
        };
        switch (nb) {
            case 8:  by_glu7(std::integral_constant<int, 8>{});  break;
            case 9:  by_glu7(std::integral_constant<int, 9>{});  break;
            case 10: by_glu7(std::integral_constant<int, 10>{}); break;
            case 11: by_glu7(std::integral_constant<int, 11>{}); break;
            case 12: by_glu7(std::integral_constant<int, 12>{}); break;
            case 13: by_glu7(std::integral_constant<int, 13>{}); break;
            case 14: by_glu7(std::integral_constant<int, 14>{}); break;
            case 15: by_glu7(std::integral_constant<int, 15>{}); break;
            default: by_glu7(std::integral_constant<int, 16>{}); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    auto go = [&](auto rb_tag, auto glu_tag) {
        constexpr int R = decltype(rb_tag)::value, G = decltype(glu_tag)::value;
        mmvq_h1s_id<R, G><<<grid, R, smem, stream>>>((const uint8_t *) src0->data, row_bytes, (int64_t) src0->nb[2],
            (const int32_t *) ids->data, X, sx, Uv, su, (float *) dst->data, sy, (int) M, (int) K, cpb);
    };
    auto by_glu = [&](auto rb_tag) {
        switch (glu) {
            case 1:  go(rb_tag, std::integral_constant<int, 1>{}); break;
            case 2:  go(rb_tag, std::integral_constant<int, 2>{}); break;
            default: go(rb_tag, std::integral_constant<int, 0>{}); break;
        }
    };
    if (rb == 32) {
        by_glu(std::integral_constant<int, 32>{});
    } else if (rb == 128) {
        by_glu(std::integral_constant<int, 128>{});
    } else {
        by_glu(std::integral_constant<int, 64>{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
