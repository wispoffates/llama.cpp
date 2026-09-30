#include "mmvq-f16.cuh"

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

template <int NC, int NBLK, int B = 0>
static __device__ __forceinline__ void mmvq_f16_blocks(
        const uint32_t * const * wb, const __half * const * xw, __half2 (&t)[NC][MMVQ_F16_RPW],
        const int P, const int ql_w, const int qh_w, const int vh_shift,
        const int sc_we, const int sc_wo, const uint32_t sc_sel_e, const uint32_t sc_sel_o) {
    if constexpr (B < NBLK) {
        const __half2 k1056 = __float2half2_rn(1056.0f), k1152 = __float2half2_rn(1152.0f), k1024 = __float2half2_rn(1024.0f);
        __half2 xa[NC][2], xb[NC][2];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const __half * xs = xw[c] + B*256;
            const uint2 a = __ldg((const uint2 *) (xs + P)), bb = __ldg((const uint2 *) (xs + P + 64));
            xa[c][0] = *(const __half2 *) &a.x;  xa[c][1] = *(const __half2 *) &a.y;
            xb[c][0] = *(const __half2 *) &bb.x; xb[c][1] = *(const __half2 *) &bb.y;
        }
#pragma unroll
        for (int i = 0; i < MMVQ_F16_RPW; ++i) {
            const uint32_t vl = mmvq_f16_fld<B, 0>(wb[i], ql_w);
            const uint32_t vh = mmvq_f16_fld<B, 128>(wb[i], qh_w) >> vh_shift;
            // scales so and so+4: window-relative bytes 210*B + 192 + so and +4, one byte lane apart by a word
            uint32_t sab;
            if constexpr ((B & 1) == 0) {
                constexpr int C = (210*B + 192)/4;
                sab = __byte_perm(wb[i][C + sc_we], wb[i][C + sc_we + 1], sc_sel_e);
            } else {
                constexpr int C = (210*B + 190)/4;
                sab = __byte_perm(wb[i][C + sc_wo], wb[i][C + sc_wo + 1], sc_sel_o);
            }
            const uint32_t sh = __byte_perm(sab ^ 0x8080u, 0x64646464u, 0x5140); // half(1024 + 128 + sc)
            constexpr int DA = 210*B + 208;
            const uint32_t dw = wb[i][DA/4];
            const uint32_t dd = ((DA & 3) == 0) ? __byte_perm(dw, 0, 0x1010) : __byte_perm(dw, 0, 0x3232);
            const __half2 d2  = __hmul2(*(const __half2 *) &dd, k1024);
            const __half2 sc2 = __hmul2(__hsub2(*(const __half2 *) &sh, k1152), d2);
            const __half2 sA  = __low2half2(sc2), sB = __high2half2(sc2);
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
        mmvq_f16_blocks<NC, NBLK, B + 1>(wb, xw, t, P, ql_w, qh_w, vh_shift, sc_we, sc_wo, sc_sel_e, sc_sel_o);
    }
}

template <int NC>
__launch_bounds__(MMVQ_F16_NW*WARP_SIZE, 1)
static __global__ void mmvq_f16_q6_K(const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XS,
                                     const float * __restrict__ S, float * __restrict__ Y, const int64_t sy,
                                     const int rows, const int K) {
    constexpr int RPB = MMVQ_F16_NW*MMVQ_F16_RPW, WB = MMVQ_F16_NBF*210, NU = (WB + 15 + 15)/16;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + MMVQ_F16_NBF - 1)/MMVQ_F16_NBF;
    const int row0 = blockIdx.x*RPB + wid*MMVQ_F16_RPW;

    __shared__ uint4 wst[MMVQ_F16_NW][MMVQ_F16_RPW][NU];

    // this lane's 8 values of a block: P..P+3 (scale so) and P+64..P+67 (scale so+4)
    const int iqs = lane, P = 128*(iqs/16) + 4*(iqs % 16);
    const int ql_w = iqs, qh_w = 8*(iqs/16) + iqs % 8, vh_shift = 2*((iqs % 16)/8), so = 8*(iqs/16) + (iqs % 16)/4;
    const int sc_we = so >> 2, sc_wo = (so + 2) >> 2;
    const uint32_t sc_sel_e = (so & 3) | ((4 + (so & 3)) << 4), sc_sel_o = ((so + 2) & 3) | ((4 + ((so + 2) & 3)) << 4);

    float acc[NC][MMVQ_F16_RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < MMVQ_F16_RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const char * rp[MMVQ_F16_RPW];
#pragma unroll
    for (int i = 0; i < MMVQ_F16_RPW; ++i) {
        rp[i] = (const char *) W + (int64_t) min(row0 + i, rows - 1)*row_bytes;
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K;
    }

    for (int win = 0; win < nw; ++win) {
        const int nblk = min(MMVQ_F16_NBF, nb - win*MMVQ_F16_NBF);
        __syncwarp();
        const uint32_t * wb[MMVQ_F16_RPW];
#pragma unroll
        for (int i = 0; i < MMVQ_F16_RPW; ++i) {
            const char * g = rp[i];
            rp[i] += WB;
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
        __half2 t[NC][MMVQ_F16_RPW];
        switch (nblk) {
            case 4: mmvq_f16_blocks<NC, 4>(wb, xw, t, P, ql_w, qh_w, vh_shift, sc_we, sc_wo, sc_sel_e, sc_sel_o); break;
            default: mmvq_f16_blocks<NC, 2>(wb, xw, t, P, ql_w, qh_w, vh_shift, sc_we, sc_wo, sc_sel_e, sc_sel_o); break; // nb % 4 == 2
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += MMVQ_F16_WIN;
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < MMVQ_F16_RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < MMVQ_F16_RPW; ++i) {
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

bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_F16");
        return !s || atoi(s) != 0;
    }();
    // GGML_CUDA_MMVQ_F16_MINROWS: the smallest row count that takes this path
    static const int64_t min_rows = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MINROWS"); return s ? (int64_t) atoll(s) : (int64_t) 3072; }();
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

    ggml_cuda_pool_alloc<__half> xs(ctx.pool(), ncols*K);
    ggml_cuda_pool_alloc<float>  sc(ctx.pool(), ncols*nw);
    {
        const dim3 pb(WARP_SIZE, 4), pg((nw + 3)/4, ncols);
        mmvq_f16_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float), xs.get(), sc.get(), (int) K);
    }
    const dim3 bd(WARP_SIZE, MMVQ_F16_NW);
    const int  g  = (int) ((rows + MMVQ_F16_NW*MMVQ_F16_RPW - 1)/(MMVQ_F16_NW*MMVQ_F16_RPW));
    const int64_t sy = dst->nb[1]/sizeof(float);
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    switch (ncols) {
        case 2: mmvq_f16_q6_K<2><<<g, bd, 0, stream>>>(W, src0->nb[1], xs.get(), sc.get(), Y, sy, (int) rows, (int) K); break;
        case 3: mmvq_f16_q6_K<3><<<g, bd, 0, stream>>>(W, src0->nb[1], xs.get(), sc.get(), Y, sy, (int) rows, (int) K); break;
        case 4: mmvq_f16_q6_K<4><<<g, bd, 0, stream>>>(W, src0->nb[1], xs.get(), sc.get(), Y, sy, (int) rows, (int) K); break;
        default: mmvq_f16_q6_K<5><<<g, bd, 0, stream>>>(W, src0->nb[1], xs.get(), sc.get(), Y, sy, (int) rows, (int) K); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
