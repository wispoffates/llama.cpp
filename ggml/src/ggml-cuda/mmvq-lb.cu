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

// v3b (10-08): Q4 found small matrices short of blocks at 4 rows per warp (640 rows = 40 blocks on 56 SMs: 14.1 vs 8.7 us integer)
// -> rows per warp RPW in {1, 2, 4} by M; and IQ3_S added (Sakura's attention gates / GDN out / shared-expert gate+up), which v3
// had left on the integer path (decode KLD 0.001290 vs v2's 0.000434).
enum lb3_fmt { LB3_IQ4_NL = 0, LB3_IQ4_XS = 1, LB3_IQ3_S = 2 };
template <int F> struct lb3_traits;
template <> struct lb3_traits<LB3_IQ4_NL> { static constexpr int CH = 32*sizeof(block_iq4_nl); static constexpr int SB = sizeof(block_iq4_nl); static constexpr int PER = 1; };
template <> struct lb3_traits<LB3_IQ4_XS> { static constexpr int CH = 4*sizeof(block_iq4_xs);  static constexpr int SB = sizeof(block_iq4_xs); static constexpr int PER = 8; };
template <> struct lb3_traits<LB3_IQ3_S>  { static constexpr int CH = 4*sizeof(block_iq3_s);   static constexpr int SB = sizeof(block_iq3_s);  static constexpr int PER = 8; };

// sub-block (32 values) of a staged block b, sub-block index ib, dotted with yv (fp32), exactly dequantize_row_* weights
template <int F>
static __device__ __forceinline__ float lb3_sub(const uint8_t * b, const int ib, const float * yv, const float * kv) {
    if constexpr (F == LB3_IQ4_XS) {
        const uint32_t sh = b[2] | (b[3] << 8);
        const int ls = ((b[4 + ib/2] >> 4*(ib & 1)) & 0xf) | (((sh >> 2*ib) & 3) << 4);
        const float dl = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))))*(ls - 32);
        const uint32_t * q = (const uint32_t *) (b + 8 + 16*ib);   // 4-aligned in the stage (8-aligned rows)
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = q[k];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                s += kv[(w >> 8*m) & 0xf]*yv[4*k + m] + kv[(w >> (8*m + 4)) & 0xf]*yv[4*k + m + 16];
            }
        }
        return dl*s;
    } else if constexpr (F == LB3_IQ4_NL) {
        const float dl = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t w = b[2 + k];
            s += kv[w & 0xf]*yv[k] + kv[w >> 4]*yv[k + 16];
        }
        return dl*s;
    } else {
        // iq3_s: d(2) qs[64] qh[8] signs[32] scales[4]
        const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
        const uint8_t * qs = b + 2 + 8*ib;
        const uint32_t qh = b[2 + 64 + ib];
        const uint8_t * sg = b + 2 + 64 + 8 + 4*ib;
        const float db = d*(1 + 2*((b[2 + 64 + 8 + 32 + ib/2] >> 4*(ib & 1)) & 0xf));
        float s = 0.0f;
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t g1 = iq3s_grid[qs[2*il + 0] | ((qh << (8 - 2*il)) & 256)];
            const uint32_t g2 = iq3s_grid[qs[2*il + 1] | ((qh << (7 - 2*il)) & 256)];
            const uint32_t sgn = sg[il];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s += (float) ((g1 >> 8*j) & 0xff)*lb_sgn(yv[8*il + j],     (sgn >> j) & 1);
                s += (float) ((g2 >> 8*j) & 0xff)*lb_sgn(yv[8*il + 4 + j], (sgn >> (j + 4)) & 1);
            }
        }
        return db*s;
    }
}

// v3c (10-08): Q5 found IQ3_S never taken (its 1100-byte rows failed the over-read rule) and mid-size matrices (2560-6144 rows)
// 20-30 % slower than the integer path, lm_head and small M at parity. Three changes:
//  - tail-safe staging: only the last row's final chunk can end inside a 16-byte segment past the tensor; it loads whole uint4s up to
//    the last full segment and the rest byte by byte, so no row-size / padding rule is needed (IQ3_S becomes eligible);
//  - the chunk activation is staged once per block in shared memory with coalesced float4 loads (v3b: each lane read its own 128 B
//    sub-block = 32 cache lines per load instruction, per warp), padded to 36 floats per sub-block so lane reads are conflict-free;
//  - software pipeline: the next row-pair's weights (and the next chunk's activation) are loaded into registers before the current
//    rows are decoded, so DRAM latency overlaps the decode instead of serialising with it.
// Rows per warp come from an occupancy-aware wave estimate (GGML_CUDA_MMVQ_LB3_RPW=1/2/4 forces). 5 blocks per SM caps registers at
// 96 without spills (6 spills; unbounded the 4-row kernels take 115 = 4 blocks).
template <int F, int RPW>
static __global__ void __launch_bounds__(MMVQ_LB3_WPB*32, 5) mmvq_lb3(
        const uint8_t * __restrict__ W, const int64_t row_bytes, const float * __restrict__ X, float * __restrict__ Y,
        const int M, const int K) {
    using tr = lb3_traits<F>;
    constexpr int CH  = tr::CH;
    constexpr int U   = RPW < 2 ? RPW : 2;      // rows staged at a time
    constexpr int NS  = RPW/U;                  // row steps per chunk
    constexpr int STG = (CH + 15 + 15)/16;      // chunk + slack for the aligned-down start
    constexpr int NLD = (STG + 31)/32;
    constexpr int NT  = MMVQ_LB3_WPB*32;
    constexpr int NX  = 32*8/NT;                // float4 of chunk activation per thread
    static_assert(NX*NT == 32*8, "activation staging");
    __shared__ uint4  stage[MMVQ_LB3_WPB][U][STG];
    __shared__ float4 xs[32][9];                // 32 sub-blocks x 32 floats, padded to 36
    __shared__ float  kv[16];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (threadIdx.x < 16) {
        kv[threadIdx.x] = kvalues_iq4nl[threadIdx.x];
    }
    const int r0 = (blockIdx.x*MMVQ_LB3_WPB + wid)*RPW;   // warps past M run along for the block barriers, never write
    const int nsub = K/32;
    ggml_cuda_pdl_sync();

    uint4 v[U][NLD];
    int   mis[U], nu4[U], ntl[U];
    uint32_t tb[U];
    // registers <- weights of rows r0+rr .. r0+rr+U-1, chunk cb
    auto load_w = [&](const int cb, const int rr) {
        const int nblk  = min(32, nsub - cb);
        const int64_t coff = (int64_t) (cb/tr::PER)*tr::SB;
        const int nbyte = (nblk/tr::PER)*tr::SB;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int row = min(r0 + rr + u, M - 1);       // rows past M re-read the last row
            const uintptr_t a = (uintptr_t) (W + (int64_t) row*row_bytes + coff);
            mis[u] = (int) (a & 15);
            const bool tail = row == M - 1 && cb + 32 >= nsub;   // the only chunk whose last segment may leave the tensor
            nu4[u] = tail ? (mis[u] + nbyte) >> 4 : (mis[u] + nbyte + 15) >> 4;
            ntl[u] = tail ? (mis[u] + nbyte) & 15 : 0;
            const uint4 * src = (const uint4 *) (a - mis[u]);
#pragma unroll
            for (int j = 0; j < NLD; ++j) {
                if (lane + 32*j < nu4[u]) {
                    v[u][j] = __ldg(src + lane + 32*j);
                }
            }
            if (lane < ntl[u]) {
                tb[u] = __ldg((const uint8_t *) (src + nu4[u]) + lane);
            }
        }
    };
    float4 xr[NX];
    auto load_x = [&](const int cb) {
        const float4 * xp = (const float4 *) (X + (int64_t) cb*32);
        const int nf4 = min(32, nsub - cb)*8;
#pragma unroll
        for (int k = 0; k < NX; ++k) {
            const int q = threadIdx.x + NT*k;
            xr[k] = q < nf4 ? __ldg(xp + q) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };

    float acc[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        acc[i] = 0.0f;
    }
    load_x(0);
    load_w(0, 0);
    for (int cb = 0; cb < nsub; cb += 32) {
        const int nblk = min(32, nsub - cb);
        __syncthreads();                            // previous chunk's activation reads done (and kv written)
#pragma unroll
        for (int k = 0; k < NX; ++k) {
            const int q = threadIdx.x + NT*k;
            xs[q >> 3][q & 7] = xr[k];
        }
        __syncthreads();
        if (cb + 32 < nsub) {
            load_x(cb + 32);                        // lands while this chunk decodes
        }
        float yv[32];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 t = xs[lane][i];
            yv[4*i + 0] = t.x; yv[4*i + 1] = t.y; yv[4*i + 2] = t.z; yv[4*i + 3] = t.w;
        }
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            int sm[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                sm[u] = mis[u];
#pragma unroll
                for (int j = 0; j < NLD; ++j) {
                    if (lane + 32*j < nu4[u]) {
                        stage[wid][u][lane + 32*j] = v[u][j];
                    }
                }
                if (lane < ntl[u]) {
                    ((uint8_t *) stage[wid][u])[16*nu4[u] + lane] = (uint8_t) tb[u];
                }
            }
            __syncwarp();
            // prefetch the next step's rows (same chunk, or the first rows of the next chunk)
            if (s + 1 < NS) {
                load_w(cb, (s + 1)*U);
            } else if (cb + 32 < nsub) {
                load_w(cb + 32, 0);
            }
            if (lane < nblk) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const uint8_t * b = (const uint8_t *) stage[wid][u] + sm[u] + (lane/tr::PER)*tr::SB;
                    acc[s*U + u] += lb3_sub<F>(b, lane % tr::PER, yv, kv);
                }
            }
            __syncwarp();
        }
    }
    ggml_cuda_pdl_lc();
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        acc[i] = warp_reduce_sum<32>(acc[i]);
    }
    if (lane < RPW && r0 + lane < M) {
        float o = acc[0];
#pragma unroll
        for (int i = 1; i < RPW; ++i) {
            o = lane == i ? acc[i] : o;
        }
        Y[r0 + lane] = o;
    }
}

// blocks of mmvq_lb3<F, R> resident per SM (cached per kernel and device)
template <int F, int R>
static int mmvq_lb3_cap(const int device) {
    static int cap[GGML_CUDA_MAX_DEVICES] = {0};
    if (cap[device] == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mmvq_lb3<F, R>, MMVQ_LB3_WPB*32, 0));
        cap[device] = std::max(1, nb)*ggml_cuda_info().devices[device].nsm;
    }
    return cap[device];
}

// dense one-column IQ4_XS / IQ4_NL / IQ3_S through v3; false = not handled here
static bool mmvq_lb3_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool iq3s_on = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB3_IQ3S"); return !s || atoi(s) != 0; }();
    const ggml_type type = src0->type;
    int fmt;
    switch (type) {
        case GGML_TYPE_IQ4_NL: fmt = LB3_IQ4_NL; break;
        case GGML_TYPE_IQ4_XS: fmt = LB3_IQ4_XS; break;
        case GGML_TYPE_IQ3_S:  if (!iq3s_on) { return false; } fmt = LB3_IQ3_S; break;
        default: return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1];
    const int64_t qk = type == GGML_TYPE_IQ4_NL ? QK4_NL : QK_K;
    const int64_t row_bytes = (K/qk)*(int64_t) ggml_type_size(type);
    // IQ4_XS reads its qs words 4-aligned inside the stage -> 8-aligned rows; the others use byte reads. Over-reads: none (v3c).
    const bool align_ok = fmt != LB3_IQ4_XS || (row_bytes % 8 == 0 && (uintptr_t) src0->data % 8 == 0);
    if (!align_ok || K % qk != 0 || K % 32 != 0 || src0->nb[0] != ggml_type_size(type) || (int64_t) src0->nb[1] != row_bytes ||
            src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[0] != K || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) ||
            ((uintptr_t) src1->data % 16) != 0 || dst->ne[0] != M || dst->ne[1] != 1 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    static const int rpw_env = [] { const char * s = getenv("GGML_CUDA_MMVQ_LB3_RPW"); return s ? atoi(s) : 0; }();
    const uint8_t * Wp = (const uint8_t *) src0->data;
    const float   * Xp = (const float *) src1->data;
    float         * Yp = (float *) dst->data;
    cudaStream_t stream = ctx.stream();
    const int device = ctx.device;
    auto byfmt = [&](auto fmt_tag) {
        constexpr int F = decltype(fmt_tag)::value;
        // rows per warp: fewest (waves x per-warp work), per-warp work ~ R + 1 (rows + the chunk activation staging); ties -> more rows
        int rpw = rpw_env == 1 || rpw_env == 2 || rpw_env == 4 ? rpw_env : 0;
        if (rpw == 0) {
            const int64_t caps[3] = { mmvq_lb3_cap<F, 4>(device), mmvq_lb3_cap<F, 2>(device), mmvq_lb3_cap<F, 1>(device) };
            int64_t best = INT64_MAX;
            for (int i = 0; i < 3; ++i) {
                const int64_t R = 4 >> i, nb = (M + MMVQ_LB3_WPB*R - 1)/(MMVQ_LB3_WPB*R);
                const int64_t cost = (nb + caps[i] - 1)/caps[i]*(R + 1);
                if (cost < best) {
                    best = cost; rpw = (int) R;
                }
            }
        }
        auto go = [&](auto rpw_tag) {
            constexpr int R = decltype(rpw_tag)::value;
            const dim3 grid((unsigned) ((M + MMVQ_LB3_WPB*R - 1)/(MMVQ_LB3_WPB*R)));
            mmvq_lb3<F, R><<<grid, MMVQ_LB3_WPB*32, 0, stream>>>(Wp, row_bytes, Xp, Yp, (int) M, (int) K);
        };
        switch (rpw) {
            case 1:  go(std::integral_constant<int, 1>{}); break;
            case 2:  go(std::integral_constant<int, 2>{}); break;
            default: go(std::integral_constant<int, 4>{}); break;
        }
    };
    switch (fmt) {
        case LB3_IQ4_NL: byfmt(std::integral_constant<int, LB3_IQ4_NL>{}); break;
        case LB3_IQ4_XS: byfmt(std::integral_constant<int, LB3_IQ4_XS>{}); break;
        default:         byfmt(std::integral_constant<int, LB3_IQ3_S>{});  break;
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
