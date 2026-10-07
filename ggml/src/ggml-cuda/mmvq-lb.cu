#include "mmvq-lb.cuh"

// Low-bit / lookup-grid types x f32 matvec for ONE column on Pascal, in fp32 (decode: dense matmuls and the one-token MUL_MAT_ID
// of MoE experts). Types: Q2_0, IQ1_M, IQ2_XXS, IQ2_S (expert banks of 2-bit qwen4exp builds), IQ3_S, IQ4_NL, IQ4_XS (their
// attention / GDN projections).
//
// The integer path (mmvq.cu) quantizes the activation to q8_1 and multiplies with DP4A, which sm_60 emulates in several
// instructions; for these types it also expands every grid entry to int8 lanes first. On Sakura (Swift-1.5 Flash-Next K352 IQ2_XS)
// that ran the Q2_0 expert down-projection (K = 640) at ~72 GB/s and the IQ2_S gate/up and IQ4_XS projections at ~150-250 GB/s.
// Here the activation stays fp32: the block stages it once in shared memory (33-float stride per 32 values, so lanes on
// consecutive 32-value units hit distinct banks), and each lane decodes its unit's weights straight to fp32 and multiplies.
//
// Work unit = 32 consecutive K values (one IQ sub-block, half a Q2_0 block, one IQ4_NL block). upr = K/32 units per row.
//   long rows (upr >= 32): a warp owns RPW rows; lane l walks units l, l+32, ... of all RPW rows at once (RPW loads in flight),
//                          then one warp reduction per row.
//   short rows (upr < 32): the warp's RPW*upr (row, unit) pairs are spread over the 32 lanes, partials meet in shared memory and
//                          lane r adds row r's upr partials in K order. (Q2_0 down at K = 640 is 20 units: one row per warp would
//                          leave 12 lanes idle.)
// Accumulation is fp32 throughout; the order differs from the integer path, so results match it to rounding, not bit for bit.
// GGML_CUDA_MMVQ_LB=0 disables the path; GGML_CUDA_MMVQ_LB_DENSE=0 keeps dense (non-MUL_MAT_ID) matmuls on the integer path.

#define MMVQ_LB_WPB 8   // warps per block
#define MMVQ_LB_SK  33  // shared-memory floats per 32-value unit (32 + 1 skew)

static __device__ __forceinline__ float lb_sgn(const float v, const uint32_t bit) {
    return __int_as_float(__float_as_int(v) ^ (int) (bit << 31));
}

template <ggml_type type> struct lb_traits;
template <> struct lb_traits<GGML_TYPE_Q2_0>    { static constexpr int QK = QK2_0;  static constexpr int BS = sizeof(block_q2_0);    };
template <> struct lb_traits<GGML_TYPE_IQ2_XXS> { static constexpr int QK = QK_K;   static constexpr int BS = sizeof(block_iq2_xxs); };
template <> struct lb_traits<GGML_TYPE_IQ2_S>   { static constexpr int QK = QK_K;   static constexpr int BS = sizeof(block_iq2_s);   };
template <> struct lb_traits<GGML_TYPE_IQ1_M>   { static constexpr int QK = QK_K;   static constexpr int BS = sizeof(block_iq1_m);   };
template <> struct lb_traits<GGML_TYPE_IQ3_S>   { static constexpr int QK = QK_K;   static constexpr int BS = sizeof(block_iq3_s);   };
template <> struct lb_traits<GGML_TYPE_IQ4_NL>  { static constexpr int QK = QK4_NL; static constexpr int BS = sizeof(block_iq4_nl);  };
template <> struct lb_traits<GGML_TYPE_IQ4_XS>  { static constexpr int QK = QK_K;   static constexpr int BS = sizeof(block_iq4_xs);  };

// dot of unit u of a row (its 32 weights, exactly as dequantize_row_* produces them) with the staged activation y[0..31]
template <ggml_type type>
static __device__ __forceinline__ float lb_unit(const uint8_t * __restrict__ row, const int u, const float * __restrict__ y,
                                                const float * __restrict__ kv) {
    if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0 * b = (const block_q2_0 *) row + (u >> 1);
        const uint16_t * q16 = (const uint16_t *) (b->qs + 8*(u & 1));
        float s1 = 0.0f, s0 = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = q16[k];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const float v = y[8*k + m];
                s1 += (float) ((w >> 2*m) & 3) * v;   // code c, weight (c - 1)*d
                s0 += v;
            }
        }
        return __half2float(b->d)*(s1 - s0);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) row + (u >> 3);
        const uint16_t * q2 = b->qs + 4*(u & 7);
        const uint32_t aux0 = q2[0] | ((uint32_t) q2[1] << 16), aux1 = q2[2] | ((uint32_t) q2[3] << 16);
        float acc = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t g = iq2xxs_grid[(aux0 >> 8*il) & 0xff];
            const uint32_t s = ksigns_iq2xs[(aux1 >> 7*il) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += (float) ((g >> 8*j) & 0xff) * lb_sgn(y[8*il + j], (s >> j) & 1);
            }
        }
        return __half2float(b->d)*(0.5f + (aux1 >> 28))*0.25f*acc;
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        const block_iq2_s * b = (const block_iq2_s *) row + (u >> 3);
        const int ib = u & 7;
        const uint32_t qh = b->qh[ib], sc = b->scales[ib];
        float acc[2] = {0.0f, 0.0f};
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint64_t g = iq2s_grid[b->qs[4*ib + il] | ((qh << (8 - 2*il)) & 0x300)];
            const uint32_t s = b->qs[QK_K/8 + 4*ib + il];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[il/2] += (float) ((g >> 8*j) & 0xff) * lb_sgn(y[8*il + j], (s >> j) & 1);
            }
        }
        return __half2float(b->d)*0.25f*((0.5f + (sc & 0xf))*acc[0] + (0.5f + (sc >> 4))*acc[1]);
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        const block_iq1_m * b = (const block_iq1_m *) row + (u >> 3);
        const int ib = u & 7;
        const uint16_t * sc = (const uint16_t *) b->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = __half2float(scale.f16);
        float tot = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const int ib16 = 2*ib + il/2;
            const float dl = d*(2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
            const uint32_t qhb = b->qh[2*ib + il/2];
            const float delta = qhb & (0x08 << 4*(il%2)) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
            const uint32_t g = iq1s_grid_gpu[b->qs[4*ib + il] | (((qhb >> 4*(il%2)) & 7) << 8)];
            float sq = 0.0f, sy = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v0 = y[8*il + j], v1 = y[8*il + 4 + j];
                sq += (float) ((g >> 8*j) & 0x0f)*v0 + (float) ((g >> (8*j + 4)) & 0x0f)*v1;
                sy += v0 + v1;
            }
            tot += dl*(sq + delta*sy);
        }
        return tot;
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const block_iq3_s * b = (const block_iq3_s *) row + (u >> 3);
        const int ib = u & 7;
        const uint8_t * qs = b->qs + 8*ib;
        const uint32_t qh = b->qh[ib];
        float acc = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = iq3s_grid[qs[2*il + 0] | ((qh << (8 - 2*il)) & 256)];
            const uint32_t g2 = iq3s_grid[qs[2*il + 1] | ((qh << (7 - 2*il)) & 256)];
            const uint32_t s = b->signs[4*ib + il];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += (float) ((g1 >> 8*j) & 0xff) * lb_sgn(y[8*il + j],     (s >> j) & 1);
                acc += (float) ((g2 >> 8*j) & 0xff) * lb_sgn(y[8*il + 4 + j], (s >> (j + 4)) & 1);
            }
        }
        return __half2float(b->d)*(1 + 2*((b->scales[ib/2] >> 4*(ib%2)) & 0xf))*acc;
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * b = (const block_iq4_nl *) row + u;
        const uint16_t * q16 = (const uint16_t *) b->qs;
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t w = q16[k];
            acc += kv[w & 0xf]*y[2*k] + kv[(w >> 4) & 0xf]*y[2*k + 16] + kv[(w >> 8) & 0xf]*y[2*k + 1] + kv[(w >> 12) & 0xf]*y[2*k + 17];
        }
        return __half2float(b->d)*acc;
    } else {
        static_assert(type == GGML_TYPE_IQ4_XS, "unsupported type");
        const block_iq4_xs * b = (const block_iq4_xs *) row + (u >> 3);
        const int ib = u & 7;
        const int ls = ((b->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((b->scales_h >> 2*ib) & 3) << 4);
        const uint2 * q8 = (const uint2 *) (b->qs + 16*ib);   // 8-byte aligned: 136-byte blocks, qs at +8
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const uint2 w2 = q8[k];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t w = h ? w2.y : w2.x;
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int j = 8*k + 4*h + m;
                    acc += kv[(w >> 8*m) & 0xf]*y[j] + kv[(w >> (8*m + 4)) & 0xf]*y[j + 16];
                }
            }
        }
        return __half2float(b->d)*(ls - 32)*acc;
    }
}

// grid (ceil(M/(WPB*RPW)), slots); ID: slot y -> expert ids[y], activation row (nxr == 1 ? 0 : y), dst column y
template <ggml_type type, int RPW, bool SHORT, bool ID>
static __global__ void __launch_bounds__(MMVQ_LB_WPB*32) mmvq_lb(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const int64_t expert_bytes, const int32_t * __restrict__ ids,
        const float * __restrict__ X, const int64_t sx, const int nxr, float * __restrict__ Y, const int64_t sy, const int M, const int K) {
    extern __shared__ float lb_smem[];
    float * ys   = lb_smem;                                   // K/32 units x MMVQ_LB_SK
    float * kv   = ys + (K/32)*MMVQ_LB_SK;                    // 16 IQ4 values
    float * part = kv + 16;                                   // SHORT: per warp RPW*32 partials
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int slot = blockIdx.y;
    const int xr = nxr == 1 ? 0 : slot;
    const int upr = K/32;

    ggml_cuda_pdl_sync();
    X += (int64_t) xr*sx;
    for (int i = threadIdx.x; i < K; i += MMVQ_LB_WPB*32) {
        ys[(i >> 5)*MMVQ_LB_SK + (i & 31)] = X[i];
    }
    if (threadIdx.x < 16) {
        kv[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    }
    __syncthreads();

    const int r0 = (blockIdx.x*MMVQ_LB_WPB + wid)*RPW;
    if (r0 >= M) {
        return;   // no block-wide barriers below
    }
    if (ID) {
        W += (int64_t) ids[slot]*expert_bytes;
    }
    Y += (int64_t) slot*sy;

    if constexpr (!SHORT) {
        const uint8_t * rows[RPW];
        float acc[RPW];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            rows[r] = W + (int64_t) min(r0 + r, M - 1)*row_bytes;   // rows past M re-read the last row; never written
            acc[r] = 0.0f;
        }
        for (int u = lane; u < upr; u += 32) {
            const float * yu = ys + u*MMVQ_LB_SK;
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
                acc[r] += lb_unit<type>(rows[r], u, yu, kv);
            }
        }
        ggml_cuda_pdl_lc();
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
            acc[r] = warp_reduce_sum<32>(acc[r]);
        }
        if (lane < RPW && r0 + lane < M) {
            float v = acc[0];
#pragma unroll
            for (int r = 1; r < RPW; ++r) {
                v = lane == r ? acc[r] : v;
            }
            Y[r0 + lane] = v;
        }
    } else {
        float * pw = part + wid*(RPW*32);
        const int n = RPW*upr;
        for (int i = lane; i < n; i += 32) {
            const int r = i / upr, u = i - r*upr;
            const uint8_t * row = W + (int64_t) min(r0 + r, M - 1)*row_bytes;
            pw[i] = lb_unit<type>(row, u, ys + u*MMVQ_LB_SK, kv);
        }
        ggml_cuda_pdl_lc();
        __syncwarp();
        if (lane < RPW && r0 + lane < M) {
            float v = 0.0f;
            for (int u = 0; u < upr; ++u) {
                v += pw[lane*upr + u];
            }
            Y[r0 + lane] = v;
        }
    }
}

static constexpr int MMVQ_LB_MAX_K = 10240;   // staged activation: 10240*33/32 floats = 42 KB of shared memory

bool ggml_cuda_mmvq_lb_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           const ggml_tensor * ids, ggml_tensor * dst) {
    static const bool enabled  = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB");       return !s || atoi(s) != 0; }();
    static const bool dense_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB_DENSE"); return !s || atoi(s) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_type type = src0->type;
    if (!enabled || (!ids && !dense_on) || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
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
    if (K % qk != 0 || K > MMVQ_LB_MAX_K || src0->nb[0] != ggml_type_size(type) || (int64_t) src0->nb[1] != row_bytes ||
            src0->ne[3] != 1 || ((uintptr_t) src0->data % 8) != 0 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0 || src1->ne[0] != K || dst->ne[0] != M) {
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
    const int upr = (int) (K/32);
    const bool shrt = upr < 32;
    const int rpw = shrt ? 8 : 4;
    const int64_t sx = nxr > 1 ? (int64_t) (src1->nb[1]/sizeof(float)) : 0;
    const int64_t sy = ids ? (int64_t) (dst->nb[1]/sizeof(float)) : 0;
    const size_t smem = ((size_t) upr*MMVQ_LB_SK + 16 + (shrt ? MMVQ_LB_WPB*rpw*32 : 0))*sizeof(float);
    const dim3 grid((unsigned) ((M + MMVQ_LB_WPB*rpw - 1)/(MMVQ_LB_WPB*rpw)), (unsigned) slots);
    cudaStream_t stream = ctx.stream();
    const uint8_t * W = (const uint8_t *) src0->data;
    const int64_t eb = ids ? (int64_t) src0->nb[2] : 0;
    const int32_t * idp = ids ? (const int32_t *) ids->data : nullptr;

    auto launch = [&](auto type_tag) {
        constexpr ggml_type T = decltype(type_tag)::value;
        if (shrt) {
            if (ids) {
                mmvq_lb<T, 8, true, true><<<grid, MMVQ_LB_WPB*32, smem, stream>>>(W, row_bytes, eb, idp, (const float *) src1->data, sx,
                    (int) nxr, (float *) dst->data, sy, (int) M, (int) K);
            } else {
                mmvq_lb<T, 8, true, false><<<grid, MMVQ_LB_WPB*32, smem, stream>>>(W, row_bytes, 0, nullptr, (const float *) src1->data, 0,
                    1, (float *) dst->data, 0, (int) M, (int) K);
            }
        } else {
            if (ids) {
                mmvq_lb<T, 4, false, true><<<grid, MMVQ_LB_WPB*32, smem, stream>>>(W, row_bytes, eb, idp, (const float *) src1->data, sx,
                    (int) nxr, (float *) dst->data, sy, (int) M, (int) K);
            } else {
                mmvq_lb<T, 4, false, false><<<grid, MMVQ_LB_WPB*32, smem, stream>>>(W, row_bytes, 0, nullptr, (const float *) src1->data, 0,
                    1, (float *) dst->data, 0, (int) M, (int) K);
            }
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
