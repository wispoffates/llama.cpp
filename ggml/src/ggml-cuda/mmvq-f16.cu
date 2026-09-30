#include "mmvq-f16.cuh"

#include <unordered_map>

// q6_K x f32 matvec for 2..5 columns on Pascal, in fp16 with short chains folded into fp32.
//
// The integer path (mmvq.cu) is ALU-bound here: sm_60 has no DP4A, and its emulation costs ~2.2
// instructions per multiply-add. This path converts each weight pair to fp16 once (PRMT, HSUB2,
// HMUL2) and shares it across the columns, then does one HFMA2 per two multiply-adds.
//
// Numerics, per column c and window w of 1024 values (4 q6_K blocks):
//   x    -> half, prescaled by a power of 2 so |x| < 1 (the scaling is exact; one rounding of x
//           to 11 bits, against the integer path's rounding of x to 8 bits in q8_1)
//   w    = (d*1024) * sc * (q - 32) in half: one rounding of the weight
//   each lane runs a fused fp16 chain of 16 HFMA2 over its window, and the two lanes are added and
//   folded into an fp32 accumulator once per window.
// Against a double-precision reference on this model's real weights: NMSE ~4.6e-7, where the
// integer path's q8_1 activations give ~1.3e-4 (OPTLOG 203, 206).
//
// Layout constraints: each row must hold an even number of q6_K blocks, so that every row and every
// 4-block window starts 4-byte aligned and block b's 2-byte phase (b & 1) is known at compile time.

#define MMVQ_F16_NW  2   // warps per block
#define MMVQ_F16_RPW 4   // rows per warp
#define MMVQ_F16_NBF 4   // q6_K blocks per window (fold)
static constexpr int MMVQ_F16_WIN = MMVQ_F16_NBF*256;

// one warp per (window, column)
static __global__ void mmvq_f16_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                     float * __restrict__ S, const int K) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    if (w >= nw) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*MMVQ_F16_WIN;
    constexpr int PER = MMVQ_F16_WIN/32;
    const int len = min(MMVQ_F16_WIN, K - w*MMVQ_F16_WIN);
    float v[PER];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (j*128 < len) {
            f = __ldg((const float4 *) x + j*32 + threadIdx.x);
        }
        v[4*j] = f.x; v[4*j + 1] = f.y; v[4*j + 2] = f.z; v[4*j + 3] = f.w;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*MMVQ_F16_WIN;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        if (j*128 >= len) {
            break;
        }
        const __half2 a = __floats2half2_rn(v[4*j]*inv, v[4*j + 1]*inv), b = __floats2half2_rn(v[4*j + 2]*inv, v[4*j + 3]*inv);
        uint2 u;
        u.x = *(const uint32_t *) &a;
        u.y = *(const uint32_t *) &b;
        ((uint2 *) xs)[j*32 + threadIdx.x] = u;
    }
    if (threadIdx.x == 0) {
        S[c*nw + w] = ldexpf(1.0f, e - 10); // also undoes the 1024 on the weights
    }
}

// 32-bit field at byte offset OFF of block B, relative to a 4-aligned window base
template <int B, int OFF>
static __device__ __forceinline__ uint32_t mmvq_f16_fld(const uint32_t * wb, const int lane_words) {
    constexpr int A = 210*B + OFF;
    if constexpr ((A & 3) == 0) {
        return wb[A/4 + lane_words];
    } else {
        return __byte_perm(wb[A/4 + lane_words], wb[A/4 + 1 + lane_words], 0x5432);
    }
}

template <int NC, int RPW, int NBLK, int B = 0>
static __device__ __forceinline__ void mmvq_f16_blocks(
        const uint32_t * const * wb, const uint2 * const * scs, const __half * const * xw, __half2 (&t)[NC][RPW],
        const int P, const int ql_w, const int qh_w, const int vh_shift, const int sp) {
    if constexpr (B < NBLK) {
        const __half2 k1056 = __float2half2_rn(1056.0f);
        __half2 xa[NC][2], xb[NC][2];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const __half * xs = xw[c] + B*256;
            const uint2 a = __ldg((const uint2 *) (xs + P)), bb = __ldg((const uint2 *) (xs + P + 64));
            xa[c][0] = *(const __half2 *) &a.x;  xa[c][1] = *(const __half2 *) &a.y;
            xb[c][0] = *(const __half2 *) &bb.x; xb[c][1] = *(const __half2 *) &bb.y;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint32_t vl = mmvq_f16_fld<B, 0>(wb[i], ql_w);
            const uint32_t vh = mmvq_f16_fld<B, 128>(wb[i], qh_w) >> vh_shift;
            // scales so and so+4, precomputed per window by mmvq_f16_scales (the same half ops)
            const uint2 sAB = scs[i][B*8 + sp];
            const __half2 sA = *(const __half2 *) &sAB.x, sB = *(const __half2 *) &sAB.y;
            const uint32_t qa = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
            const uint32_t qb = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
            // half(1024 + q) per byte, minus 1056, times the scale
            const uint32_t h0 = __byte_perm(qa, 0x64646464u, 0x5140), h1 = __byte_perm(qa, 0x64646464u, 0x5342);
            const uint32_t h2 = __byte_perm(qb, 0x64646464u, 0x5140), h3 = __byte_perm(qb, 0x64646464u, 0x5342);
            const __half2 w0 = __hmul2(__hsub2(*(const __half2 *) &h0, k1056), sA);
            const __half2 w1 = __hmul2(__hsub2(*(const __half2 *) &h1, k1056), sA);
            const __half2 w2 = __hmul2(__hsub2(*(const __half2 *) &h2, k1056), sB);
            const __half2 w3 = __hmul2(__hsub2(*(const __half2 *) &h3, k1056), sB);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                __half2 u = B == 0 ? __hmul2(w0, xa[c][0]) : __hfma2(w0, xa[c][0], t[c][i]);
                u = __hfma2(w1, xa[c][1], u);
                u = __hfma2(w2, xb[c][0], u);
                t[c][i] = __hfma2(w3, xb[c][1], u);
            }
        }
        mmvq_f16_blocks<NC, RPW, NBLK, B + 1>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp);
    }
}

// A block's 16 scales, as the (sA, sB) half2 pairs its lanes use: lane l of the warp computes pair
// p = l % 8 (scales j and j + 4, j = p < 4 ? p : p + 4) of block l / 8 of the window, once, instead
// of every lane recomputing its pair for every block (~12 instructions per lane, block and row).
// The half arithmetic is exactly the per-lane version's: sc2 = (half(1152 + sc) - 1152) * (d*1024).
template <int RPW>
static __device__ __forceinline__ void mmvq_f16_scales(const uint32_t * const * wb, uint2 (*scst)[MMVQ_F16_NBF*8],
                                                       const int nblk, const int lane) {
    const int B = lane >> 3, p = lane & 7, j = p < 4 ? p : p + 4;
    if (B >= nblk) {
        return;
    }
    const __half2 k1152 = __float2half2_rn(1152.0f), k1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * bb = (const uint8_t *) wb[i] + 210*B;
        const uint32_t sab = (uint32_t) bb[192 + j] | ((uint32_t) bb[196 + j] << 8);
        const uint32_t sh  = __byte_perm(sab ^ 0x8080u, 0x64646464u, 0x5140); // half(1024 + 128 + sc)
        const uint32_t d16 = *(const uint16_t *) (bb + 208);
        const uint32_t dd  = d16 | (d16 << 16);
        const __half2 d2  = __hmul2(*(const __half2 *) &dd, k1024);
        const __half2 sc2 = __hmul2(__hsub2(*(const __half2 *) &sh, k1152), d2);
        const __half2 sA  = __low2half2(sc2), sB = __high2half2(sc2);
        scst[i][B*8 + p] = make_uint2(*(const uint32_t *) &sA, *(const uint32_t *) &sB);
    }
}

template <int NC, int RPW, int NWT, bool KS, int MINB = 12/NWT>
__launch_bounds__(NWT*WARP_SIZE, MINB) // 12/NWT: 168 registers, no spills: 6 blocks per SM (OPTLOG 210)
static __global__ void mmvq_f16_q6_K(const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XS,
                                     const float * __restrict__ S, float * __restrict__ Y, const int64_t sy,
                                     const int rows, const int K) {
    constexpr int RPB = NWT*RPW, WB = MMVQ_F16_NBF*210, NU = (WB + 15 + 15)/16;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + MMVQ_F16_NBF - 1)/MMVQ_F16_NBF;
    // KS (split K): the block's warps share its RPW rows and take every NWT-th window each
    const int row0 = KS ? blockIdx.x*RPW : blockIdx.x*RPB + wid*RPW;

    __shared__ uint4 wst[NWT][RPW][NU];

    // this lane's 8 values of a block: P..P+3 (scale so) and P+64..P+67 (scale so+4)
    const int iqs = lane, P = 128*(iqs/16) + 4*(iqs % 16);
    const int ql_w = iqs, qh_w = 8*(iqs/16) + iqs % 8, vh_shift = 2*((iqs % 16)/8), so = 8*(iqs/16) + (iqs % 16)/4;
    const int sp = so < 8 ? so : so - 4; // this lane's scale pair (so, so + 4), see mmvq_f16_scales
    __shared__ uint2 scst[NWT][RPW][MMVQ_F16_NBF*8];
    const uint2 * scs[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        scs[i] = scst[wid][i];
    }

    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (const char *) W + (int64_t) min(row0 + i, rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0);
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0);
    }

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const int nblk = min(MMVQ_F16_NBF, nb - win*MMVQ_F16_NBF);
        __syncwarp();
        const uint32_t * wb[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const char * g = rp[i];
            rp[i] += KS ? (int64_t) NWT*WB : WB;
            const int m = (int) ((uintptr_t) g & 15); // a multiple of 4
            wb[i] = (const uint32_t *) wst[wid][i] + (m >> 2);
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + nblk*210 + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        mmvq_f16_scales<RPW>(wb, scst[wid], nblk, lane);
        __syncwarp();
        __half2 t[NC][RPW];
        switch (nblk) {
            case 4: mmvq_f16_blocks<NC, RPW, 4>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break;
            default: mmvq_f16_blocks<NC, RPW, 2>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break; // nb % 4 == 2
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        // per-warp partials, then warp 0 adds them in a fixed order
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            float v = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v += __shfl_xor_sync(0xFFFFFFFF, v, o);
            }
            if (lane == i && row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
    }
}

// The prescaled fp16 activation, reused across consecutive matmuls that read the same src1 (gate
// and up, the q/k/v projections), like the integer path's q8_1 cache. Keyed on the src1 node, whose
// contents are fixed within one graph evaluation; cleared at the start of each one. One persistent
// buffer per CUDA context (contexts are per device).
struct mmvq_f16_cache {
    const ggml_tensor * src1 = nullptr;
    const void *        data = nullptr;
    int64_t             ncols = 0, K = 0;
    void *              buf  = nullptr;
    size_t              cap  = 0;
};
static std::unordered_map<const ggml_backend_cuda_context *, mmvq_f16_cache> mmvq_f16_caches;

void ggml_cuda_mmvq_f16_invalidate(ggml_backend_cuda_context & ctx) {
    auto it = mmvq_f16_caches.find(&ctx);
    if (it != mmvq_f16_caches.end()) {
        it->second.src1 = nullptr;
        it->second.data = nullptr;
    }
}

bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_F16");
        return !s || atoi(s) != 0;
    }();
    // GGML_CUDA_MMVQ_F16_MINROWS: the smallest row count that takes this path
    static const int64_t min_rows = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MINROWS"); return s ? (int64_t) atoll(s) : (int64_t) 16; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t K = src0->ne[0], rows = src0->ne[1];
    // Where it measured faster than the integer path (OPTLOG 206), test-backend-ops at 5 columns, rows x K:
    // 8704x5120 154 -> 133 us, 6144x5120 115 -> 97, 5120x5120 98 -> 89, 3072x5120 67 -> 61,
    // 5120x8704 161 -> 146, 5120x3072 61 -> 57. K must hold an even number of q6_K blocks.
    if (!enabled || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || src0->type != GGML_TYPE_Q6_K ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ncols < 2 || ncols > 5 ||
            K % 512 != 0 || rows < min_rows || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[1] != (size_t) (K/256)*210 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            (src1->nb[1] % 16) != 0 || ((uintptr_t) src1->data % 16) != 0 || ((uintptr_t) src0->data % 4) != 0) {
        return false;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    cudaStream_t stream = ctx.stream();

    mmvq_f16_cache & cache = mmvq_f16_caches[&ctx];
    const size_t xs_bytes = (size_t) ncols*K*sizeof(__half);
    const size_t need     = xs_bytes + (size_t) ncols*nw*sizeof(float);
    __half * xs_ptr;
    float  * sc_ptr;
    const bool hit = cache.src1 == src1 && cache.data == src1->data && cache.ncols == ncols && cache.K == K;
    if (!hit) {
        if (cache.cap < need) {
            if (cache.buf) {
                CUDA_CHECK(cudaFree(cache.buf)); // synchronizes; only when the buffer grows
            }
            CUDA_CHECK(cudaMalloc(&cache.buf, need));
            cache.cap = need;
        }
        cache.src1 = src1; cache.data = src1->data; cache.ncols = ncols; cache.K = K;
    }
    xs_ptr = (__half *) cache.buf;
    sc_ptr = (float *) ((char *) cache.buf + xs_bytes);
    if (!hit) {
        const dim3 pb(WARP_SIZE, 4), pg((nw + 3)/4, ncols);
        mmvq_f16_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float), xs_ptr, sc_ptr, (int) K);
    }
    const int64_t sy = dst->nb[1]/sizeof(float);
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    // big matrices: 2 warps x 4 rows per block; mid-size: 1 row per warp; small (< 256 rows): the 4
    // warps of a block split K over one row, so there are enough blocks and warps to cover the GPU
    auto launch = [&](auto rpw, auto nwt, auto ks, auto minb) {
        constexpr int  RPW  = decltype(rpw)::value;
        constexpr int  NWT  = decltype(nwt)::value;
        constexpr bool KS   = decltype(ks)::value;
        constexpr int  MINB = decltype(minb)::value;
        const dim3 bdk(WARP_SIZE, NWT);
        const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
        switch (ncols) {
            case 2:  mmvq_f16_q6_K<2, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr, sc_ptr, Y, sy, (int) rows, (int) K); break;
            case 3:  mmvq_f16_q6_K<3, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr, sc_ptr, Y, sy, (int) rows, (int) K); break;
            case 4:  mmvq_f16_q6_K<4, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr, sc_ptr, Y, sy, (int) rows, (int) K); break;
            default: mmvq_f16_q6_K<5, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr, sc_ptr, Y, sy, (int) rows, (int) K); break;
        }
    };
    using I = std::integral_constant<int, 1>;
    if (rows >= 3072) {
        // 6 or 7 blocks per SM (168 or 128 registers): take 7 when it needs fewer waves over the
        // SMs (3072 rows: 2 -> 1 wave, 49 vs 57 us at 5 columns; 6144: 3 -> 2; 8704: 4 -> 3).
        const int64_t nblk = (rows + MMVQ_F16_NW*MMVQ_F16_RPW - 1)/(MMVQ_F16_NW*MMVQ_F16_RPW);
        const int64_t nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int64_t w6 = (nblk + 6*nsm - 1)/(6*nsm), w7 = (nblk + 7*nsm - 1)/(7*nsm);
        if (w7 < w6) {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 7>{});
        } else {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 6>{});
        }
    } else if (rows >= 256) {
        // one warp of 2 rows per block: at 512 rows x 5120 and 5 columns (wk/wv under -sm tensor)
        // 13.3 us against 17.7 for 2 warps of 1 row (test-backend-ops, prep included). The rows'
        // arithmetic doesn't depend on the grouping, so the results are identical.
        // GGML_CUDA_MMVQ_F16_MID=0 restores 2 warps x 1 row.
        static const bool mid2 = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MID"); return !s || atoi(s) != 0; }();
        if (mid2) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 16>{});
        } else {
            launch(I{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 12/MMVQ_F16_NW>{});
        }
    } else {
        launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
