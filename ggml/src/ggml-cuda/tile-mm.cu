#include "tile-mm.cuh"

#include <atomic>
#include <cstdio>

// Quantized x f32 GEMM for 9..80 columns on Pascal (sm_60), fused: the weights are read once and decoded tile by tile in shared
// memory instead of being dequantized to a full fp16 copy for cuBLAS. Types: q6_K, q8_0.
//
// Why: sm_60 has no DP4A, so there is no MMQ and every batch wider than the matvec kernels (> 8 columns, e.g. Swift's n-gram
// verify batches of 49..65 tokens) goes dequantize -> fp16 temp -> maxwell_hgemm_256x128 with fp16 accumulation over all of K.
// At 65 columns that tile is half empty (~6 TFLOPS) and the dequant pass writes 2 bytes per weight every call.
//
// Layout (tilegemm harness v3, ~/claude/bench-tools/tilegemm):
//   block = 64 rows x 16*TN columns, 256 threads; thread (rg, cg) owns rows rg + 16*i (i < 4), columns cg*TN + j.
//   The tile's raw weight bytes are staged one unit at a time (q6_K: a 256-value superblock, 210 bytes per row, copied from the
//   aligned-down address; q8_0: four 32-value blocks, 136 bytes per row, always 4-aligned) global -> registers -> shared, with
//   coalesced 32-bit loads. Re-reading the bytes every 64-value step capped the fetch at ~65-110 GB/s on the L1/tex path;
//   staging each byte once reaches ~400 GB/s. The next unit is prefetched into registers during the math.
//   Each 64-value step decodes the raw bytes to fp16 (x1024, away from subnormals) and multiplies with HFMA2; activations are
//   prescaled per 1024-value window by a power of 2 (|x| < 1, as in mmvq-f16.cu) and the fp16 chains are folded into fp32
//   every 32 k-pairs. The activation k-pairs are rotated by one inside each group of 4 so that the two HFMA2 sources never sit
//   in the same register bank.
//   Split-K over units fills the GPU; partial sums go to a pool buffer and a small kernel reduces them in a fixed order
//   (deterministic).
// q6_K: NMSE ~9e-7 against a double-precision reference on random weights (cuBLAS path ~1.1e-5). Measured (P100, 150 W, harness
// incl. activation prep and reduce), rows x K at 64 columns, against cuBLAS hgemm alone (production adds the dequant pass, ~320 us
// at 8704x5120): 8704x5120 1151 -> 844 us, 5120x8704 1152 -> 874, 6144x5120 898 -> 629, 5120x5120 775 -> 542, 3072x5120 404 -> 339.
// Swift serving (v5.1 + --spec-ngram-mod-n-max 63): edit +13.6 %, edit2 +19 %, short-prompt prefill +20 %.

static constexpr int TMM_WIN = 1024;   // activation prescale window (values)
static constexpr int TMM_KP  = 32;     // k-pairs per step (64 values)
static constexpr int TMM_LD  = 36;     // shared row stride in half2: 128-bit loads are conflict-free
static constexpr int TMM_BM  = 64;

template <ggml_type type> struct tile_mm_traits;
template <> struct tile_mm_traits<GGML_TYPE_Q6_K> {
    static constexpr int  UV      = QK_K;   // values per staged unit (one superblock)
    static constexpr int  RW      = 53;     // 32-bit words per staged row: the 210-byte block plus up to 2 bytes of alignment slack
    static constexpr bool aligned = false;
};
template <> struct tile_mm_traits<GGML_TYPE_Q8_0> {
    static constexpr int  UV      = 4*QK8_0; // four blocks: 136 bytes per row, 4-aligned when the row holds an even number of blocks
    static constexpr int  RW      = 34;
    static constexpr bool aligned = true;
};

// byte address of staged unit u of row m
template <ggml_type type>
static __device__ __forceinline__ const unsigned char * tile_mm_unit(const void * W, const int64_t m, const int64_t row_bytes, const int u) {
    constexpr int64_t unit_bytes = type == GGML_TYPE_Q6_K ? (int64_t) sizeof(block_q6_K) : (int64_t) (4*sizeof(block_q8_0));
    return (const unsigned char *) W + m*row_bytes + u*unit_bytes;
}

// decode step st (64 values) of a staged unit: lane j returns k-pair j of the step as fp16 x1024
template <ggml_type type>
static __device__ __forceinline__ half2 tile_mm_decode(const unsigned char * rb, const int st, const int lane) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        // g = 2sh + j/16, l = 2*(j%16), +1 (h = st/2, sh = st%2)
        const int h = st >> 1, sh = st & 1, g = 2*sh + (lane >> 4), lp = lane & 15;
        const uint32_t qlw = *(const unsigned short *) (rb + 64*h + 2*lane);
        const uint32_t qhw = *(const unsigned short *) (rb + 128 + 32*h + 2*lp);
        const int      sc  = (int) (signed char) rb[192 + 8*h + (lp >> 3) + 2*g];
        const float    d   = __half2float(*(const __half *) (rb + 208))*1024.0f*sc;
        const int nsh = 4*sh, hsh = 2*g;
        const int q0 = ((qlw >> nsh) & 0xF) | (((qhw >> hsh) & 3) << 4);
        const int q1 = (((qlw >> 8) >> nsh) & 0xF) | ((((qhw >> 8) >> hsh) & 3) << 4);
        return __floats2half2_rn(d*(q0 - 32), d*(q1 - 32));
    } else {
        // block 2st + j/16 of the unit, quants 2*(j%16), +1
        const unsigned char * b = rb + (2*st + (lane >> 4))*sizeof(block_q8_0);
        const float    d  = __half2float(*(const __half *) b)*1024.0f;
        const uint32_t qw = *(const unsigned short *) (b + 2 + 2*(lane & 15));
        return __floats2half2_rn(d*(int) (signed char) (qw & 0xFF), d*(int) (signed char) (qw >> 8));
    }
}

// X f32 [N][K] (row stride sx floats) -> XS half [N][K] prescaled by 2^-e per (column, window) so |x| < 1; S[n][w] = 2^e.
static __global__ void tile_mm_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                      float * __restrict__ S, const int K, const int nwin) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y;
    if (w >= nwin) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*TMM_WIN;
    const int len = min(TMM_WIN, K - w*TMM_WIN);
    float m = 0.0f;
    for (int i = threadIdx.x; i < len; i += WARP_SIZE) {
        m = fmaxf(m, fabsf(x[i]));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*TMM_WIN;
    for (int i = threadIdx.x; i < len; i += WARP_SIZE) {
        xs[i] = __float2half(x[i]*inv);
    }
    if (threadIdx.x == 0) {
        S[(int64_t) c*nwin + w] = ldexpf(1.0f, e);
    }
}

// TN = 5 (80 columns) needs > 32 KB of shared memory, so only one block fits per SM anyway: let it use the registers
// (with the 2-block bound it spilled; 65-column verify batches -- ngram-mod's 64-token drafts + 1 -- land here)
template <ggml_type type, int TN>
static __global__ void __launch_bounds__(256, TN >= 5 ? 1 : 2) tile_mm(const void * __restrict__ W, const int64_t row_bytes,
        const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ P,
        const int M, const int K, const int N, const int nwin, const int u_per) {
    using tr = tile_mm_traits<type>;
    constexpr int TM = 4, BM = TMM_BM, BN = 16*TN, RPW = BM/8, RW = tr::RW, UV = tr::UV, SPU = UV/64;
    constexpr int XI = (BN*8 + 255)/256;
    extern __shared__ __align__(16) unsigned char tmm_smem[];
    uint32_t * Rs = (uint32_t *) tmm_smem;              // [BM][RW] raw bytes of the current unit
    half2    * Ws = (half2 *) (Rs + BM*RW);             // [BM][TMM_LD] decoded weights of one step
    half2    * Xs = Ws + BM*TMM_LD;                     // [BN][TMM_LD] activations of one step (rotated k-pairs)
    __shared__ int Roff[BM];                            // byte offset of the unit inside its staged row (0 or 2; q6_K only)

    const int t = threadIdx.x, rg = t % 16, cg = t / 16, lane = t % WARP_SIZE, wid = t / WARP_SIZE;
    const int m0 = blockIdx.x*BM, n0 = blockIdx.y*BN;
    const int nu = K/UV;
    const int u0 = blockIdx.z*u_per, u1 = min(nu, u0 + u_per);

    // warp w stages rows w + 8i: lane j holds words j and 32 + j of the row's (aligned-down) unit copy
    uint32_t raw[2*RPW];
#define TMM_FETCH(uu) do {                                                                                            \
        _Pragma("unroll") for (int i = 0; i < RPW; ++i) {                                                             \
            const int m = min(m0 + wid + 8*i, M - 1);                                                                 \
            const uint32_t * w = (const uint32_t *) ((uintptr_t) tile_mm_unit<type>(W, m, row_bytes, (uu)) & ~(uintptr_t) 3); \
            raw[2*i]     = __ldg(w + lane);                                                                           \
            raw[2*i + 1] = lane < RW - 32 ? __ldg(w + 32 + lane) : 0u;                                                \
        } } while (0)

    float accf[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            accf[i][j] = 0.0f;
        }
    }

    if (u0 < u1) {
        TMM_FETCH(u0);
    }
    for (int u = u0; u < u1; ++u) {
        __syncthreads();                               // the previous unit is done with Rs
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            uint32_t * rr = Rs + (wid + 8*i)*RW;
            rr[lane] = raw[2*i];
            if (lane < RW - 32) {
                rr[32 + lane] = raw[2*i + 1];
            }
        }
        if (!tr::aligned && t < BM) {
            Roff[t] = (int) (((uintptr_t) tile_mm_unit<type>(W, min(m0 + t, M - 1), row_bytes, u)) & 3);
        }
        if (u + 1 < u1) {
            TMM_FETCH(u + 1);                          // in flight during this unit's steps
        }
#pragma unroll 1
        for (int st = 0; st < SPU; ++st) {
            __syncthreads();                           // Rs staged / the previous step is done with Ws, Xs
#pragma unroll 2
            for (int r = wid; r < BM; r += 8) {
                const unsigned char * rb = (const unsigned char *) (Rs + r*RW) + (tr::aligned ? 0 : Roff[r]);
                Ws[r*TMM_LD + lane] = m0 + r < M ? tile_mm_decode<type>(rb, st, lane) : __float2half2_rn(0.0f);
            }
            const int k0 = u*UV + st*64;
            {   // activations: small and L2-resident, rotated by one k-pair inside each group of 4
#pragma unroll
                for (int v = 0; v < XI; ++v) {
                    const int it = t + 256*v, c = it % BN, q = it / BN, n = n0 + c;
                    if (it < BN*8) {
                        const uint4 x = n < N ? __ldg((const uint4 *) (XS + (int64_t) n*K + k0 + 8*q)) : make_uint4(0, 0, 0, 0);
                        *(uint4 *) (Xs + c*TMM_LD + 4*q) = make_uint4(x.y, x.z, x.w, x.x);
                    }
                }
            }
            __syncthreads();
            float sw[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int n = n0 + cg*TN + j;
                sw[j] = n < N ? S[(int64_t) n*nwin + k0/TMM_WIN]*(1.0f/1024.0f) : 0.0f;
            }
            half2 acc[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __float2half2_rn(0.0f);
                }
            }
#pragma unroll
            for (int kq = 0; kq < TMM_KP; kq += 4) {
                uint4 x4[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    x4[j] = *(const uint4 *) (Xs + (cg*TN + j)*TMM_LD + kq);
                }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint4 w4 = *(const uint4 *) (Ws + (rg + 16*i)*TMM_LD + kq);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
#pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] = __hfma2(((const half2 *) &w4)[e], ((const half2 *) &x4[j])[(e + 3) & 3], acc[i][j]);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const float2 f = __half22float2(acc[i][j]);
                    accf[i][j] = fmaf(f.x + f.y, sw[j], accf[i][j]);
                }
            }
        }
    }
#undef TMM_FETCH
    float * p = P + (int64_t) blockIdx.z*N*M;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + cg*TN + j;
        if (n >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = m0 + rg + 16*i;
            if (m < M) {
                p[(int64_t) n*M + m] = accf[i][j];
            }
        }
    }
}

// dst[n][m] (row stride sd floats) = sum over the splits, in a fixed order
static __global__ void tile_mm_reduce(const float * __restrict__ P, float * __restrict__ D, const int M, const int N,
                                        const int64_t sd, const int splits) {
    const int64_t NM = (int64_t) N*M;
    for (int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x; i < NM; i += (int64_t) gridDim.x*blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) {
            s += P[z*NM + i];
        }
        D[(i / M)*sd + i % M] = s;
    }
}

template <ggml_type type, int TN>
static void tile_mm_launch(const void * W, const int64_t row_bytes, const __half * XS, const float * S, float * P, const int M,
                           const int K, const int N, const int nwin, const int splits, const int u_per, cudaStream_t stream) {
    constexpr int BN = 16*TN;
    const size_t smem = (size_t) TMM_BM*tile_mm_traits<type>::RW*sizeof(uint32_t) + (size_t) (TMM_BM + BN)*TMM_LD*sizeof(half2);
    const dim3 grid((M + TMM_BM - 1)/TMM_BM, (N + BN - 1)/BN, splits);
    tile_mm<type, TN><<<grid, 256, smem, stream>>>(W, row_bytes, XS, S, P, M, K, N, nwin, u_per);
}

template <ggml_type type>
static void tile_mm_launch_tn(const int tn, const void * W, const int64_t row_bytes, const __half * XS, const float * S, float * P,
                              const int M, const int K, const int N, const int nwin, const int splits, const int u_per, cudaStream_t stream) {
    switch (tn) {
        case 1:  tile_mm_launch<type, 1>(W, row_bytes, XS, S, P, M, K, N, nwin, splits, u_per, stream); break;
        case 2:  tile_mm_launch<type, 2>(W, row_bytes, XS, S, P, M, K, N, nwin, splits, u_per, stream); break;
        case 3:  tile_mm_launch<type, 3>(W, row_bytes, XS, S, P, M, K, N, nwin, splits, u_per, stream); break;
        case 4:  tile_mm_launch<type, 4>(W, row_bytes, XS, S, P, M, K, N, nwin, splits, u_per, stream); break;
        default: tile_mm_launch<type, 5>(W, row_bytes, XS, S, P, M, K, N, nwin, splits, u_per, stream); break;
    }
}

// GGML_CUDA_TILE_STATS=1 (or the older GGML_CUDA_TILE_Q6K_STATS=1): histogram of the column counts of every q6_K / q8_0 matmul
// offered to this hook (all widths > 8 on Pascal reach it), printed to stderr at exit. Diagnostics only.
static std::atomic<uint64_t> tmm_hist[2][2][9];   // [type: q6_K, q8_0][taken][bucket]
static const char * tmm_bucket_name[9] = {"9-16", "17-32", "33-48", "49-64", "65-80", "81-128", "129-256", "257-512", ">512"};
static void tmm_stats_print() {
    const char * tname[2] = {"q6_K", "q8_0"};
    for (int ty = 0; ty < 2; ++ty) {
        fprintf(stderr, "tile_mm stats %s (matmul calls by column count): taken / cuBLAS\n", tname[ty]);
        for (int b = 0; b < 9; ++b) {
            fprintf(stderr, "  %-8s %10llu / %10llu\n", tmm_bucket_name[b], (unsigned long long) tmm_hist[ty][1][b].load(),
                    (unsigned long long) tmm_hist[ty][0][b].load());
        }
    }
}
static void tmm_stats_note(const ggml_type type, const int64_t n, const bool taken) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_TILE_STATS");
        if (!e) {
            e = getenv("GGML_CUDA_TILE_Q6K_STATS");
        }
        if (e && e[0] == '1') {
            atexit(tmm_stats_print);
            return true;
        }
        return false;
    }();
    if (!on) {
        return;
    }
    const int b = n <= 16 ? 0 : n <= 32 ? 1 : n <= 48 ? 2 : n <= 64 ? 3 : n <= 80 ? 4 : n <= 128 ? 5 : n <= 256 ? 6 : n <= 512 ? 7 : 8;
    tmm_hist[type == GGML_TYPE_Q8_0][taken][b]++;
}

static bool tmm_env_on(const char * name) {
    const char * e = getenv(name);
    return !e || e[0] != '0';
}

bool ggml_cuda_tile_mm_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // GGML_CUDA_TILE_Q6K=0 / GGML_CUDA_TILE_Q8_0=0 turn a type off; GGML_CUDA_TILE_MIN / _MAX (or the older _Q6K_MIN / _MAX) set the
    // column range that takes this path (default 9..80)
    static const bool q6k_on = tmm_env_on("GGML_CUDA_TILE_Q6K"), q80_on = tmm_env_on("GGML_CUDA_TILE_Q8_0");
    static const int nmin = [] { const char * e = getenv("GGML_CUDA_TILE_MIN"); if (!e) { e = getenv("GGML_CUDA_TILE_Q6K_MIN"); } return e ? atoi(e) : 9; }();
    static const int nmax = [] { const char * e = getenv("GGML_CUDA_TILE_MAX"); if (!e) { e = getenv("GGML_CUDA_TILE_Q6K_MAX"); } return e ? std::min(atoi(e), 80) : 80; }();
    const ggml_type type = src0->type;
    if ((type != GGML_TYPE_Q6_K && type != GGML_TYPE_Q8_0) || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    bool ok = src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && N >= nmin && N <= nmax &&
              src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
              src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
              (src1->nb[1] % sizeof(float)) == 0 && (dst->nb[1] % sizeof(float)) == 0;
    if (type == GGML_TYPE_Q6_K) {
        // the aligned-down 53-word staging reads up to 2 bytes past a 4-aligned block but never past the tensor when M*nbk is even
        // (the last block then ends 4-aligned)
        const int64_t nbk = K/QK_K;
        ok = ok && q6k_on && K % QK_K == 0 && (M*nbk) % 2 == 0 && src0->nb[1] == (size_t) nbk*sizeof(block_q6_K) &&
             ((uintptr_t) src0->data % 2) == 0;
    } else {
        // units of four blocks: K % 128 keeps every row 4-aligned (even block count) and the units whole
        ok = ok && q80_on && K % (4*QK8_0) == 0 && src0->nb[1] == (size_t) (K/QK8_0)*sizeof(block_q8_0) &&
             ((uintptr_t) src0->data % 4) == 0;
    }
    tmm_stats_note(type, N, ok);
    if (!ok) {
        return false;
    }
    const int uv   = type == GGML_TYPE_Q6_K ? tile_mm_traits<GGML_TYPE_Q6_K>::UV : tile_mm_traits<GGML_TYPE_Q8_0>::UV;
    const int tn   = N <= 16 ? 1 : N <= 32 ? 2 : N <= 48 ? 3 : N <= 64 ? 4 : 5;
    const int nwin = (int) ((K + TMM_WIN - 1)/TMM_WIN);
    const int nu   = (int) (K/uv);
    const int tiles = (int) (((M + TMM_BM - 1)/TMM_BM)*((N + 16*tn - 1)/(16*tn)));
    const int nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    // split-K: fill ~8 blocks per SM, at least 512 values per split
    int splits = std::max(1, std::min((int) (K/512), (8*nsm + tiles - 1)/tiles));
    const int u_per = (nu + splits - 1)/splits;
    splits = (nu + u_per - 1)/u_per;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<__half> xs(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float>  sc(ctx.pool(), N*nwin);
    ggml_cuda_pool_alloc<float>  part(ctx.pool(), (int64_t) splits*N*M);
    {
        const dim3 pb(WARP_SIZE, 4), pg((nwin + 3)/4, (unsigned) N);
        tile_mm_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, (int64_t) (src1->nb[1]/sizeof(float)), xs.get(), sc.get(),
                                            (int) K, nwin);
    }
    if (type == GGML_TYPE_Q6_K) {
        tile_mm_launch_tn<GGML_TYPE_Q6_K>(tn, src0->data, (int64_t) src0->nb[1], xs.get(), sc.get(), part.get(), (int) M, (int) K,
                                          (int) N, nwin, splits, u_per, stream);
    } else {
        tile_mm_launch_tn<GGML_TYPE_Q8_0>(tn, src0->data, (int64_t) src0->nb[1], xs.get(), sc.get(), part.get(), (int) M, (int) K,
                                          (int) N, nwin, splits, u_per, stream);
    }
    tile_mm_reduce<<<std::min<int64_t>(1024, (N*M + 255)/256), 256, 0, stream>>>(part.get(), (float *) dst->data, (int) M, (int) N,
                                                                                (int64_t) (dst->nb[1]/sizeof(float)), splits);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
