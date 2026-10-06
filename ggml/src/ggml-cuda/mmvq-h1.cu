#include "mmvq-h1.cuh"

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

// split-K variant for SMALL dense matrices (one column): the plain layout gives a 1024 x 2816 matvec 64 blocks of 4 warps
// (~4 warps per SM) and reaches ~175 GB/s. Here a block is KS warps on the SAME RPW rows; warp w takes a contiguous, balanced
// range of blocks [w*nb/KS, (w+1)*nb/KS) (in chunks of <= 32, lane = block as above), so a row's K is read by KS warps in
// parallel without re-reading the activation. Partial row sums meet in shared memory and are added in warp order:
// deterministic, but not the plain kernel's summation order. grid ceil(M/RPW).
template <ggml_type type, int KS, int U, int RPW>
static __global__ void __launch_bounds__(KS*32) mmvq_h1_qs(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XH, const float * __restrict__ XS,
        float * __restrict__ Y, const int M, const int K) {
    constexpr int BB  = mmvq_h1_traits<type>::BB, NWD = mmvq_h1_traits<type>::NWD;
    constexpr int STG = (32*BB + 15 + 15)/16;
    constexpr int NLD = (STG + 31)/32;
    __shared__ uint4 stage[KS][U][STG];
    __shared__ float part[KS][RPW];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int r0 = blockIdx.x*RPW;
    const int nb = K/H1_QK;
    const int b0 = (int) ((int64_t) wid*nb/KS), b1 = (int) ((int64_t) (wid + 1)*nb/KS);

    float acc[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        acc[i] = 0.0f;
    }
    for (int cb = b0; cb < b1; cb += 32) {
        const int nblk = min(32, b1 - cb);
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
    if constexpr (RPW >= 8)  { mmvq_h1_bfly<4>(acc, lane); }
    if constexpr (RPW >= 4)  { mmvq_h1_bfly<2>(acc, lane); }
    if constexpr (RPW >= 2)  { mmvq_h1_bfly<1>(acc, lane); }
    if (lane < RPW) {
        part[wid][lane] = acc[0];
    }
    __syncthreads();
    if (wid == 0 && lane < RPW && r0 + lane < M) {
        float sum = part[0][lane];
#pragma unroll
        for (int w = 1; w < KS; ++w) {
            sum += part[w][lane];
        }
        Y[r0 + lane] = sum;
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
    // small dense one-column matrices: split K across 2..4 warps per row group when the plain grid would give fewer than 4 blocks
    // per SM (GGML_CUDA_MMVQ_H1_SPLITK=0 for A/B)
    static const bool splitk_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_H1_SPLITK"); return !s || atoi(s) != 0; }();
    const int64_t nchunks = (K/H1_QK + 31)/32;
    if (splitk_on && !ids && nxr == 1 && nchunks >= 2 && (M + RB - 1)/RB < 4*ggml_cuda_info().devices[ctx.device].nsm) {
        const int ks = (int) std::min<int64_t>(nchunks, 4);
        const dim3 grids((unsigned) ((M + MMVQ_H1_RPW - 1)/MMVQ_H1_RPW));
        auto launchs = [&](auto type_tag) {
            constexpr ggml_type T = decltype(type_tag)::value;
            switch (ks) {
                case 2:  mmvq_h1_qs<T, 2, MMVQ_H1_U, MMVQ_H1_RPW><<<grids, 2*32, 0, stream>>>((const uint8_t *) src0->data, row_bytes, xh, xs, (float *) dst->data, (int) M, (int) K); break;
                case 3:  mmvq_h1_qs<T, 3, MMVQ_H1_U, MMVQ_H1_RPW><<<grids, 3*32, 0, stream>>>((const uint8_t *) src0->data, row_bytes, xh, xs, (float *) dst->data, (int) M, (int) K); break;
                default: mmvq_h1_qs<T, 4, MMVQ_H1_U, MMVQ_H1_RPW><<<grids, 4*32, 0, stream>>>((const uint8_t *) src0->data, row_bytes, xh, xs, (float *) dst->data, (int) M, (int) K); break;
            }
        };
        if (type == GGML_TYPE_Q4_0) {
            launchs(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});
        } else {
            launchs(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
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

// rms_norm * weight with the h1 activation prep fused in: one row of K (decode), the register path of rms_norm_f32 (same
// arithmetic, so dst is bit-identical), and each warp's 32 consecutive outputs -- one h1 block -- also written as x/s in fp16 with
// s = max|x| (mmvq_h1_prep's arithmetic). Saves the prep launch of every matvec fed by a norm (gemma4: 91 per GPU per token).
template <int block_size>
static __global__ void __launch_bounds__(block_size) mmvq_h1_rms_norm_mul(
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ dst, __half * __restrict__ XH,
        float * __restrict__ XS, const int ncols, const float eps) {
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

bool ggml_cuda_mmvq_h1_rms_norm_mul_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
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
    mmvq_h1_rms_norm_mul<1024><<<1, 1024, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) w->data, (float *) mul->data, xh, xs, (int) K, eps);
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
