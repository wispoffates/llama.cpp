#include "mmvq-lb.cuh"

// Low-bit / lookup-grid types x f32 matvec for ONE column on Pascal, in fp32 (decode: dense matmuls and the one-token MUL_MAT_ID
// of MoE experts). Types: Q2_0, IQ1_M, IQ2_XXS, IQ2_S (expert banks of 2-bit qwen4exp builds), IQ3_S, IQ4_NL, IQ4_XS (their
// attention / GDN projections).
//
// The integer path (mmvq.cu) quantizes the activation to q8_1 and multiplies with DP4A, which sm_60 emulates in several
// instructions; for these types it also expands every grid entry to int8 lanes first. On Sakura (Swift-1.5 Flash-Next K352 IQ2_XS)
// that ran the Q2_0 expert down-projection (K = 640) at ~72 GB/s and the IQ2_S gate/up and IQ4_XS projections at ~150-250 GB/s.
// Here the weights decode straight to fp32 and meet the fp32 activation.
//
// Layout (v2, 10-07): eight lanes share a 32-value unit (one IQ sub-block, half a Q2_0 block, one IQ4_NL block), four values each,
// so a warp covers 4 units = 128 values per step and each lane reads its 4 activation values as one float4 -- coalesced, straight
// from L1/L2, and kept in registers for all RPW rows of the warp. A block is WPB warps x RPW rows = 8 rows, so a 1280-row expert
// matrix is 160 blocks per expert slot. Each lane decodes only its 4 weights (one grid entry or one 32-bit word of nibbles).
// (v1 staged the whole activation per block in shared memory and packed 32-64 rows per block: ~200 blocks for the expert
// matmuls, 5 serial units per warp, and a block-wide stage + barrier -- slower than the integer path on most shapes.)
// Accumulation is fp32 throughout; the order differs from the integer path, so results match it to rounding, not bit for bit.
// GGML_CUDA_MMVQ_LB=0 disables the path; GGML_CUDA_MMVQ_LB_DENSE=0 keeps dense (non-MUL_MAT_ID) matmuls on the integer path.

#define MMVQ_LB_WPB 4   // warps per block
#define MMVQ_LB_RPW 2   // rows per warp

static __device__ __forceinline__ float lb_sgn(const float v, const uint32_t bit) {
    return __int_as_float(__float_as_int(v) ^ (int) (bit << 31));
}

// lane's 4 values of unit u (values 4*s .. 4*s+3, s = lane%8 = sub-lane) dotted with y4; exactly dequantize_row_* weights
template <ggml_type type>
static __device__ __forceinline__ float lb_quad(const uint8_t * __restrict__ row, const int u, const int s, const float4 y,
                                                const float * __restrict__ kv) {
    const float yv[4] = {y.x, y.y, y.z, y.w};
    if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0 * b = (const block_q2_0 *) row + (u >> 1);
        const uint32_t q = b->qs[8*(u & 1) + s];                    // 4 codes, value j at bits 2j
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += (float) ((int) ((q >> 2*j) & 3) - 1)*yv[j];
        }
        return __half2float(b->d)*acc;
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) row + (u >> 3);
        const uint8_t * q2 = (const uint8_t *) (b->qs + 4*(u & 7));
        const int il = s >> 1, jj = 4*(s & 1);
        const uint32_t aux1 = q2[4] | (q2[5] << 8) | (q2[6] << 16) | ((uint32_t) q2[7] << 24);
        const uint32_t g = (uint32_t) (iq2xxs_grid[q2[il]] >> 8*jj);   // entries jj..jj+3
        const uint32_t sg = ksigns_iq2xs[(aux1 >> 7*il) & 127] >> jj;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += (float) ((g >> 8*j) & 0xff)*lb_sgn(yv[j], (sg >> j) & 1);
        }
        return __half2float(b->d)*(0.5f + (aux1 >> 28))*0.25f*acc;
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        const block_iq2_s * b = (const block_iq2_s *) row + (u >> 3);
        const int ib = u & 7, il = s >> 1, jj = 4*(s & 1);
        const uint32_t qh = b->qh[ib];
        const uint32_t g = (uint32_t) (iq2s_grid[b->qs[4*ib + il] | ((qh << (8 - 2*il)) & 0x300)] >> 8*jj);
        const uint32_t sg = (uint32_t) b->qs[QK_K/8 + 4*ib + il] >> jj;
        const float dl = __half2float(b->d)*(0.5f + ((b->scales[ib] >> 4*(il/2)) & 0xf))*0.25f;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += (float) ((g >> 8*j) & 0xff)*lb_sgn(yv[j], (sg >> j) & 1);
        }
        return dl*acc;
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        const block_iq1_m * b = (const block_iq1_m *) row + (u >> 3);
        const int ib = u & 7, il = s >> 1, h = s & 1;
        const uint16_t * sc = (const uint16_t *) b->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const int ib16 = 2*ib + il/2;
        const float dl = __half2float(scale.f16)*(2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
        const uint32_t qhb = b->qh[2*ib + il/2];
        const float delta = qhb & (0x08 << 4*(il%2)) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
        // entries 0..3 = low nibbles of the grid bytes, 4..7 = high nibbles
        const uint32_t g = iq1s_grid_gpu[b->qs[4*ib + il] | (((qhb >> 4*(il%2)) & 7) << 8)] >> 4*h;
        float sq = 0.0f, sy = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sq += (float) ((g >> 8*j) & 0x0f)*yv[j];
            sy += yv[j];
        }
        return dl*(sq + delta*sy);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const block_iq3_s * b = (const block_iq3_s *) row + (u >> 3);
        const int ib = u & 7, il = s >> 1, h = s & 1;
        const uint32_t qh = b->qh[ib];
        const uint32_t g = iq3s_grid[b->qs[8*ib + 2*il + h] | ((qh << (8 - 2*il - h)) & 256)];   // entries 4h..4h+3
        const uint32_t sg = (uint32_t) b->signs[4*ib + il] >> 4*h;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += (float) ((g >> 8*j) & 0xff)*lb_sgn(yv[j], (sg >> j) & 1);
        }
        return __half2float(b->d)*(1 + 2*((b->scales[ib/2] >> 4*(ib%2)) & 0xf))*acc;
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // values 0..15 = low nibbles of qs[0..15], 16..31 = high nibbles: sub-lane s takes bytes 4*(s%4).. of the low (s < 4) or
        // high (s >= 4) half; qs sits at +2 in 18-byte blocks (2-aligned)
        const block_iq4_nl * b = (const block_iq4_nl *) row + u;
        const uint16_t * q16 = (const uint16_t *) (b->qs + 4*(s & 3));
        const uint32_t w = (q16[0] | ((uint32_t) q16[1] << 16)) >> (s < 4 ? 0 : 4);
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += kv[(w >> 8*j) & 0xf]*yv[j];
        }
        return __half2float(b->d)*acc;
    } else {
        static_assert(type == GGML_TYPE_IQ4_XS, "unsupported type");
        const block_iq4_xs * b = (const block_iq4_xs *) row + (u >> 3);
        const int ib = u & 7;
        const int ls = ((b->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((b->scales_h >> 2*ib) & 3) << 4);
        const uint32_t w = *(const uint32_t *) (b->qs + 16*ib + 4*(s & 3)) >> (s < 4 ? 0 : 4);   // qs at +8 in 136-byte blocks
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc += kv[(w >> 8*j) & 0xf]*yv[j];
        }
        return __half2float(b->d)*(ls - 32)*acc;
    }
}

// grid (ceil(M/(WPB*RPW)), slots); ID: slot y -> expert ids[y], activation row (nxr == 1 ? 0 : y), dst column y
template <ggml_type type, bool ID>
static __global__ void __launch_bounds__(MMVQ_LB_WPB*32) mmvq_lb(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const int64_t expert_bytes, const int32_t * __restrict__ ids,
        const float * __restrict__ X, const int64_t sx, const int nxr, float * __restrict__ Y, const int64_t sy, const int M, const int K) {
    __shared__ float kv[16];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x < 16) {
        kv[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    }
    __syncthreads();
    const int slot = blockIdx.y;
    const int r0 = (blockIdx.x*MMVQ_LB_WPB + wid)*MMVQ_LB_RPW;
    if (r0 >= M) {
        return;
    }
    ggml_cuda_pdl_sync();
    if (ID) {
        W += (int64_t) ids[slot]*expert_bytes;
    }
    X += (int64_t) (nxr == 1 ? 0 : slot)*sx;
    const uint8_t * rows[MMVQ_LB_RPW];
    float acc[MMVQ_LB_RPW];
#pragma unroll
    for (int r = 0; r < MMVQ_LB_RPW; ++r) {
        rows[r] = W + (int64_t) min(r0 + r, M - 1)*row_bytes;   // rows past M re-read the last row; never written
        acc[r] = 0.0f;
    }
    const int s = lane & 7;
    const int nsteps = K/128;
    for (int st = 0; st < nsteps; ++st) {
        const int u = 4*st + (lane >> 3);
        const float4 y = __ldg((const float4 *) (X + 32*u + 4*s));
#pragma unroll
        for (int r = 0; r < MMVQ_LB_RPW; ++r) {
            acc[r] += lb_quad<type>(rows[r], u, s, y, kv);
        }
    }
    ggml_cuda_pdl_lc();
#pragma unroll
    for (int r = 0; r < MMVQ_LB_RPW; ++r) {
        acc[r] = warp_reduce_sum<32>(acc[r]);
    }
    if (lane < MMVQ_LB_RPW && r0 + lane < M) {
        Y[(int64_t) slot*sy + r0 + lane] = lane == 0 ? acc[0] : acc[MMVQ_LB_RPW - 1];
    }
    static_assert(MMVQ_LB_RPW == 2, "the store picks acc[0] / acc[1]");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// v3 (10-08): dense IQ4_XS / IQ4_NL only, staged like mmvq-h1. Phase Q3: v2's fp32 activation made Sakura's decode KLD 3.4x smaller
// (0.000434 vs 0.001481), all of it from the dense IQ4_XS projections (experts-only: no gain), but v2 issued 1-4-byte loads per lane
// and ran the 124160-row IQ4_XS output projection at 1173 us vs the integer path's 603. Here a warp stages a whole chunk of each row
// (32 sub-blocks: 4 IQ4_XS superblocks = 544 B, or 32 IQ4_NL blocks = 576 B) into shared memory with 16-byte loads from the
// 16-byte-aligned address at or below it (rows are 8-aligned), and lane l decodes sub-block l from there. Lane l keeps its 32 fp32
// activation values in registers for the warp's RPW rows. fp32 multiply-adds, one warp reduction per row, deterministic.
#define MMVQ_LB3_WPB 4
#define MMVQ_LB3_RPW 4
#define MMVQ_LB3_U   2   // rows staged at a time

template <bool XS> struct lb3_traits;
template <> struct lb3_traits<true>  { static constexpr int CH = 4*sizeof(block_iq4_xs); };   // 544 B per 32 sub-blocks
template <> struct lb3_traits<false> { static constexpr int CH = 32*sizeof(block_iq4_nl); };  // 576 B

template <bool XS>
static __global__ void __launch_bounds__(MMVQ_LB3_WPB*32) mmvq_lb3(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const float * __restrict__ X, float * __restrict__ Y,
        const int M, const int K) {
    constexpr int CH  = lb3_traits<XS>::CH;
    constexpr int STG = (CH + 15 + 15)/16;      // chunk + slack for the aligned-down start
    constexpr int NLD = (STG + 31)/32;
    __shared__ uint4 stage[MMVQ_LB3_WPB][MMVQ_LB3_U][STG];
    __shared__ float kv[16];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x < 16) {
        kv[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    }
    __syncthreads();
    const int r0 = (blockIdx.x*MMVQ_LB3_WPB + wid)*MMVQ_LB3_RPW;
    if (r0 >= M) {
        return;   // no block-wide barriers below
    }
    ggml_cuda_pdl_sync();
    const int nsub = K/32;
    float acc[MMVQ_LB3_RPW];
#pragma unroll
    for (int i = 0; i < MMVQ_LB3_RPW; ++i) {
        acc[i] = 0.0f;
    }
    for (int cb = 0; cb < nsub; cb += 32) {
        const int nblk = min(32, nsub - cb);
        float yv[32];
        if (lane < nblk) {
            const float4 * xp = (const float4 *) (X + (int64_t) (cb + lane)*32);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float4 v = __ldg(xp + i);
                yv[4*i + 0] = v.x; yv[4*i + 1] = v.y; yv[4*i + 2] = v.z; yv[4*i + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < 32; ++i) {
                yv[i] = 0.0f;
            }
        }
        const int64_t coff  = XS ? (int64_t) (cb/8)*sizeof(block_iq4_xs) : (int64_t) cb*sizeof(block_iq4_nl);
        const int     nbyte = XS ? (nblk/8)*(int) sizeof(block_iq4_xs) : nblk*(int) sizeof(block_iq4_nl);
#pragma unroll
        for (int rr = 0; rr < MMVQ_LB3_RPW; rr += MMVQ_LB3_U) {
            uint4 v[MMVQ_LB3_U][NLD];
            int   mis[MMVQ_LB3_U], nu4[MMVQ_LB3_U];
#pragma unroll
            for (int u = 0; u < MMVQ_LB3_U; ++u) {
                const int row = min(r0 + rr + u, M - 1);   // rows past M re-read the last row; never written
                const uintptr_t a = (uintptr_t) (W + (int64_t) row*row_bytes + coff);
                mis[u] = (int) (a & 15);
                nu4[u] = (mis[u] + nbyte + 15)/16;
                const uint4 * src = (const uint4 *) (a - mis[u]);
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        v[u][j] = __ldg(src + lane + 32*j);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < MMVQ_LB3_U; ++u) {
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
                for (int u = 0; u < MMVQ_LB3_U; ++u) {
                    const uint8_t * sb = (const uint8_t *) stage[wid][u] + mis[u];
                    float dl, s = 0.0f;
                    if constexpr (XS) {
                        const uint8_t * b = sb + (lane >> 3)*sizeof(block_iq4_xs);   // 8-aligned
                        const int ib = lane & 7;
                        const uint32_t sh = *(const uint16_t *) (b + 2);
                        const int ls = ((b[4 + ib/2] >> 4*(ib & 1)) & 0xf) | (((sh >> 2*ib) & 3) << 4);
                        dl = __half2float(*(const __half *) b)*(ls - 32);
                        const uint32_t * q = (const uint32_t *) (b + 8 + 16*ib);
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t w = q[k];
#pragma unroll
                            for (int m = 0; m < 4; ++m) {
                                s += kv[(w >> 8*m) & 0xf]*yv[4*k + m] + kv[(w >> (8*m + 4)) & 0xf]*yv[4*k + m + 16];
                            }
                        }
                    } else {
                        const uint8_t * b = sb + lane*sizeof(block_iq4_nl);           // 2-aligned
                        dl = __half2float(*(const __half *) b);
                        const uint16_t * q = (const uint16_t *) (b + 2);
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            const uint32_t w = q[k];
                            s += kv[w & 0xf]*yv[2*k] + kv[(w >> 4) & 0xf]*yv[2*k + 16] +
                                 kv[(w >> 8) & 0xf]*yv[2*k + 1] + kv[(w >> 12) & 0xf]*yv[2*k + 17];
                        }
                    }
                    acc[rr + u] += dl*s;
                }
            }
            __syncwarp();
        }
    }
    ggml_cuda_pdl_lc();
#pragma unroll
    for (int i = 0; i < MMVQ_LB3_RPW; ++i) {
        acc[i] = warp_reduce_sum<32>(acc[i]);
    }
    if (lane < MMVQ_LB3_RPW && r0 + lane < M) {
        float v = acc[0];
#pragma unroll
        for (int i = 1; i < MMVQ_LB3_RPW; ++i) {
            v = lane == i ? acc[i] : v;
        }
        Y[r0 + lane] = v;
    }
}

// dense one-column IQ4_XS / IQ4_NL through v3; false = not handled here
static bool mmvq_lb3_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_IQ4_NL) {
        return false;
    }
    const bool xs = type == GGML_TYPE_IQ4_XS;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t qk = xs ? QK_K : QK4_NL;
    const int64_t row_bytes = (K/qk)*(int64_t) ggml_type_size(type);
    // rows 8-aligned (IQ4_XS: 8-aligned header / qs inside the stage). The aligned-down 16-byte staging can read up to 15 bytes past
    // a row's end: fine when rows are 16-byte multiples on a 16-byte-aligned base (no overrun at all), or when the tensor owns its
    // allocation and K is not a multiple of MATRIX_ROW_PADDING (the last row is padded) -- the same rule as mmvq-h1.
    const bool rows_ok = (row_bytes % 16 == 0 && (uintptr_t) src0->data % 16 == 0) ||
                         (src0->view_src == nullptr && K % MATRIX_ROW_PADDING != 0);
    if (!rows_ok || K % qk != 0 || K % 32 != 0 || src0->nb[0] != ggml_type_size(type) || (int64_t) src0->nb[1] != row_bytes ||
            row_bytes % 8 != 0 || ((uintptr_t) src0->data % 8) != 0 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[0] != K || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) ||
            ((uintptr_t) src1->data % 16) != 0 || dst->ne[0] != M || dst->ne[1] != 1 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    constexpr int RB = MMVQ_LB3_WPB*MMVQ_LB3_RPW;
    const dim3 grid((unsigned) ((M + RB - 1)/RB));
    if (xs) {
        mmvq_lb3<true><<<grid, MMVQ_LB3_WPB*32, 0, ctx.stream()>>>((const uint8_t *) src0->data, row_bytes, (const float *) src1->data,
            (float *) dst->data, (int) M, (int) K);
    } else {
        mmvq_lb3<false><<<grid, MMVQ_LB3_WPB*32, 0, ctx.stream()>>>((const uint8_t *) src0->data, row_bytes, (const float *) src1->data,
            (float *) dst->data, (int) M, (int) K);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mmvq_lb_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {
    // GGML_CUDA_MMVQ_LB=0: everything back to the integer path; GGML_CUDA_MMVQ_LB_DENSE=0: no dense matmuls (v3 is dense-only);
    // GGML_CUDA_MMVQ_LB_V2=1: also the v2 kernel for the remaining types and the one-token MUL_MAT_ID (off: slower than the
    // integer path, and the 2-bit expert banks gain no accuracy from an fp32 activation)
    static const bool enabled  = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB");       return !s || atoi(s) != 0; }();
    static const bool dense_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB_DENSE"); return !s || atoi(s) != 0; }();
    static const bool v2_on    = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB_V2");    return  s && atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_type type = src0->type;
    if (!enabled || (!ids && !dense_on) || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ids && mmvq_lb3_try(ctx, src0, src1, dst)) {
        return true;
    }
    if (!v2_on) {
        return false;
    }
    int qk;
    switch (type) {
        case GGML_TYPE_Q2_0:    qk = QK2_0;  break;
        case GGML_TYPE_IQ4_NL:  qk = QK4_NL; break;
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:  qk = QK_K;   break;
        default: return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t row_bytes = (K/qk)*(int64_t) ggml_type_size(type);
    // K % 128: whole 4-unit warp steps; float4 activation loads need a 16-byte aligned activation row
    if (K % qk != 0 || K % 128 != 0 || src0->nb[0] != ggml_type_size(type) || (int64_t) src0->nb[1] != row_bytes ||
            src0->ne[3] != 1 || ((uintptr_t) src0->data % 8) != 0 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->nb[1] % 16 != 0 || ((uintptr_t) src1->data % 16) != 0 || dst->nb[1] % sizeof(float) != 0 ||
            src1->ne[0] != K || dst->ne[0] != M) {
        return false;
    }
    int64_t nxr, slots;
    if (ids) {
        // one token: ids [n_used, 1], src1 [K, 1 (shared) or n_used, 1], dst [M, n_used, 1]
        const int64_t n_used = ids->ne[0];
        if (ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[1] != n_used ||
                dst->ne[2] != 1 || (src1->ne[1] != 1 && src1->ne[1] != n_used) || ids->nb[0] != sizeof(int32_t) ||
                src0->nb[2] % 8 != 0) {
            return false;
        }
        nxr = src1->ne[1];
        slots = n_used;
    } else {
        if (src0->ne[2] != 1 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[1] != 1) {
            return false;
        }
        nxr = 1;
        slots = 1;
    }
    const int64_t sx = nxr > 1 ? (int64_t) (src1->nb[1]/sizeof(float)) : 0;
    const int64_t sy = ids ? (int64_t) (dst->nb[1]/sizeof(float)) : 0;
    constexpr int RB = MMVQ_LB_WPB*MMVQ_LB_RPW;
    const dim3 grid((unsigned) ((M + RB - 1)/RB), (unsigned) slots);
    cudaStream_t stream = ctx.stream();
    const uint8_t * W = (const uint8_t *) src0->data;
    const int64_t eb = ids ? (int64_t) src0->nb[2] : 0;
    const int32_t * idp = ids ? (const int32_t *) ids->data : nullptr;

    auto launch = [&](auto type_tag) {
        constexpr ggml_type T = decltype(type_tag)::value;
        if (ids) {
            mmvq_lb<T, true><<<grid, MMVQ_LB_WPB*32, 0, stream>>>(W, row_bytes, eb, idp, (const float *) src1->data, sx, (int) nxr,
                (float *) dst->data, sy, (int) M, (int) K);
        } else {
            mmvq_lb<T, false><<<grid, MMVQ_LB_WPB*32, 0, stream>>>(W, row_bytes, 0, nullptr, (const float *) src1->data, 0, 1,
                (float *) dst->data, 0, (int) M, (int) K);
        }
    };
    switch (type) {
        case GGML_TYPE_Q2_0:    launch(std::integral_constant<ggml_type, GGML_TYPE_Q2_0>{});    break;
        case GGML_TYPE_IQ2_XXS: launch(std::integral_constant<ggml_type, GGML_TYPE_IQ2_XXS>{}); break;
        case GGML_TYPE_IQ2_S:   launch(std::integral_constant<ggml_type, GGML_TYPE_IQ2_S>{});   break;
        case GGML_TYPE_IQ1_M:   launch(std::integral_constant<ggml_type, GGML_TYPE_IQ1_M>{});   break;
        case GGML_TYPE_IQ3_S:   launch(std::integral_constant<ggml_type, GGML_TYPE_IQ3_S>{});   break;
        case GGML_TYPE_IQ4_NL:  launch(std::integral_constant<ggml_type, GGML_TYPE_IQ4_NL>{});  break;
        default:                launch(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{});  break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
