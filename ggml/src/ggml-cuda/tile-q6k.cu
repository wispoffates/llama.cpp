#include "tile-q6k.cuh"

#include <atomic>
#include <cstdio>

// q6_K x f32 GEMM for 9..80 columns on Pascal (sm_60), fused: the weights are read once and decoded tile by tile in shared
// memory instead of being dequantized to a full fp16 copy for cuBLAS.
//
// Why: sm_60 has no DP4A, so there is no MMQ and every batch wider than the matvec kernels (> 8 columns, e.g. Swift's n-gram
// verify batches of 49..65 tokens) goes dequantize_q6_K -> fp16 temp -> maxwell_hgemm_256x128 with fp16 accumulation over all
// of K. At 65 columns that tile is half empty (~6 TFLOPS) and the dequant pass writes 2 bytes per weight every call.
//
// Layout (tilegemm harness v3, ~/claude/bench-tools/tilegemm):
//   block = 64 rows x 16*TN columns, 256 threads; thread (rg, cg) owns rows rg + 16*i (i < 4), columns cg*TN + j.
//   Per q6_K superblock (256 values) the tile's raw bytes are copied global -> registers -> shared once, with coalesced
//   32-bit loads of the aligned-down block (re-reading the bytes every 64-value step capped the fetch at ~65-110 GB/s on the
//   L1/tex path; once per superblock reaches ~400 GB/s). The next superblock is prefetched into registers during the math.
//   Each 64-value step decodes the raw bytes to fp16 (x1024, away from subnormals) and multiplies with HFMA2; activations are
//   prescaled per 1024-value window by a power of 2 (|x| < 1, as in mmvq-f16.cu) and the fp16 chains are folded into fp32
//   every 32 k-pairs. The activation k-pairs are rotated by one inside each group of 4 so that the two HFMA2 sources never sit
//   in the same register bank.
//   Split-K over superblocks fills the GPU; partial sums go to a pool buffer and a small kernel reduces them in a fixed order
//   (deterministic).
// Accuracy against a double-precision reference on random q6_K weights: NMSE ~9e-7, cuBLAS path ~1.1e-5.
// Measured (P100, 150 W, harness incl. activation prep and reduce), rows x K at 64 columns, against cuBLAS hgemm alone (the
// production path adds the dequant pass, ~320 us at 8704x5120):
//   8704x5120 1151 -> 844 us, 5120x8704 1152 -> 874, 6144x5120 898 -> 629, 5120x5120 775 -> 542, 3072x5120 404 -> 339.

static constexpr int TQ6_WIN = 1024;   // activation prescale window (values)
static constexpr int TQ6_KP  = 32;     // k-pairs per step (64 values)
static constexpr int TQ6_LD  = 36;     // shared row stride in half2: 128-bit loads are conflict-free
static constexpr int TQ6_RW  = 53;     // 32-bit words per staged row: the 210-byte block plus the up-to-2-byte alignment slack
static constexpr int TQ6_BM  = 64;

// X f32 [N][K] (row stride sx floats) -> XS half [N][K] prescaled by 2^-e per (column, window) so |x| < 1; S[n][w] = 2^e.
static __global__ void tile_q6_K_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                      float * __restrict__ S, const int K, const int nwin) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y;
    if (w >= nwin) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*TQ6_WIN;
    const int len = min(TQ6_WIN, K - w*TQ6_WIN);
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
    __half * xs = XS + (int64_t) c*K + (int64_t) w*TQ6_WIN;
    for (int i = threadIdx.x; i < len; i += WARP_SIZE) {
        xs[i] = __float2half(x[i]*inv);
    }
    if (threadIdx.x == 0) {
        S[(int64_t) c*nwin + w] = ldexpf(1.0f, e);
    }
}

// TN = 5 (80 columns) needs 34 KB of shared memory, so only one block fits per SM anyway: let it use the registers
// (with the 2-block bound it spilled; 65-column verify batches -- ngram-mod's 64-token drafts + 1 -- land here)
template <int TN>
static __global__ void __launch_bounds__(256, TN >= 5 ? 1 : 2) tile_q6_K(const block_q6_K * __restrict__ W, const __half * __restrict__ XS,
        const float * __restrict__ S, float * __restrict__ P, const int M, const int K, const int N, const int nwin, const int sb_per) {
    constexpr int TM = 4, BM = TQ6_BM, BN = 16*TN, RPW = BM/8;
    constexpr int XI = (BN*8 + 255)/256;
    extern __shared__ __align__(16) unsigned char tq6_smem[];
    uint32_t * Rs = (uint32_t *) tq6_smem;              // [BM][TQ6_RW] raw bytes
    half2    * Ws = (half2 *) (Rs + BM*TQ6_RW);         // [BM][TQ6_LD] decoded weights of one step
    half2    * Xs = Ws + BM*TQ6_LD;                     // [BN][TQ6_LD] activations of one step (rotated k-pairs)
    __shared__ int Roff[BM];                            // byte offset of the block inside its staged row (0 or 2)

    const int t = threadIdx.x, rg = t % 16, cg = t / 16, lane = t % WARP_SIZE, wid = t / WARP_SIZE;
    const int m0 = blockIdx.x*BM, n0 = blockIdx.y*BN;
    const int nbk = K/QK_K;
    const int b0 = blockIdx.z*sb_per, b1 = min(nbk, b0 + sb_per);

    // warp w stages rows w + 8i: lane j holds words j and 32 + j of the row's aligned-down block copy
    uint32_t raw[2*RPW];
#define TQ6_FETCH(bb) do {                                                                                            \
        _Pragma("unroll") for (int i = 0; i < RPW; ++i) {                                                             \
            const int m = min(m0 + wid + 8*i, M - 1);                                                                 \
            const uint32_t * w = (const uint32_t *) ((uintptr_t) (W + (int64_t) m*nbk + (bb)) & ~(uintptr_t) 3);       \
            raw[2*i]     = __ldg(w + lane);                                                                           \
            raw[2*i + 1] = lane < TQ6_RW - 32 ? __ldg(w + 32 + lane) : 0u;                                            \
        } } while (0)

    float accf[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            accf[i][j] = 0.0f;
        }
    }

    if (b0 < b1) {
        TQ6_FETCH(b0);
    }
    for (int b = b0; b < b1; ++b) {
        __syncthreads();                               // the previous superblock is done with Rs
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            uint32_t * rr = Rs + (wid + 8*i)*TQ6_RW;
            rr[lane] = raw[2*i];
            if (lane < TQ6_RW - 32) {
                rr[32 + lane] = raw[2*i + 1];
            }
        }
        if (t < BM) {
            Roff[t] = (int) (((uintptr_t) (W + (int64_t) min(m0 + t, M - 1)*nbk + b)) & 3);
        }
        if (b + 1 < b1) {
            TQ6_FETCH(b + 1);                          // in flight during this superblock's four steps
        }
        const float sb_scale = 1.0f/1024.0f;
#pragma unroll 1
        for (int st = 0; st < 4; ++st) {
            __syncthreads();                           // Rs staged / the previous step is done with Ws, Xs
            {   // decode: lane j owns k-pair j of the step (g = 2sh + j/16, l = 2*(j%16), +1)
                const int h = st >> 1, sh = st & 1, g = 2*sh + (lane >> 4), lp = lane & 15;
#pragma unroll 2
                for (int r = wid; r < BM; r += 8) {
                    const unsigned char * rb = (const unsigned char *) (Rs + r*TQ6_RW) + Roff[r];
                    const uint32_t qlw = *(const unsigned short *) (rb + 64*h + 2*lane);
                    const uint32_t qhw = *(const unsigned short *) (rb + 128 + 32*h + 2*lp);
                    const int      sc  = (int) (signed char) rb[192 + 8*h + (lp >> 3) + 2*g];
                    const float    d   = __half2float(*(const __half *) (rb + 208))*1024.0f*sc;
                    const int nsh = 4*sh, hsh = 2*g;
                    const int q0 = ((qlw >> nsh) & 0xF) | (((qhw >> hsh) & 3) << 4);
                    const int q1 = (((qlw >> 8) >> nsh) & 0xF) | ((((qhw >> 8) >> hsh) & 3) << 4);
                    Ws[r*TQ6_LD + lane] = m0 + r < M ? __floats2half2_rn(d*(q0 - 32), d*(q1 - 32)) : __float2half2_rn(0.0f);
                }
            }
            {   // activations: small and L2-resident, rotated by one k-pair inside each group of 4
                const int k0 = b*QK_K + st*64;
#pragma unroll
                for (int u = 0; u < XI; ++u) {
                    const int it = t + 256*u, c = it % BN, q = it / BN, n = n0 + c;
                    if (it < BN*8) {
                        const uint4 v = n < N ? __ldg((const uint4 *) (XS + (int64_t) n*K + k0 + 8*q)) : make_uint4(0, 0, 0, 0);
                        *(uint4 *) (Xs + c*TQ6_LD + 4*q) = make_uint4(v.y, v.z, v.w, v.x);
                    }
                }
            }
            __syncthreads();
            float sw[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int n = n0 + cg*TN + j;
                sw[j] = n < N ? S[(int64_t) n*nwin + (b*QK_K)/TQ6_WIN]*sb_scale : 0.0f;
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
            for (int kq = 0; kq < TQ6_KP; kq += 4) {
                uint4 x4[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    x4[j] = *(const uint4 *) (Xs + (cg*TN + j)*TQ6_LD + kq);
                }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint4 w4 = *(const uint4 *) (Ws + (rg + 16*i)*TQ6_LD + kq);
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
#undef TQ6_FETCH
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
static __global__ void tile_q6_K_reduce(const float * __restrict__ P, float * __restrict__ D, const int M, const int N,
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

template <int TN>
static void tile_q6_K_launch(const block_q6_K * W, const __half * XS, const float * S, float * P, const int M, const int K,
                             const int N, const int nwin, const int splits, const int sb_per, cudaStream_t stream) {
    constexpr int BN = 16*TN;
    const size_t smem = (size_t) TQ6_BM*TQ6_RW*sizeof(uint32_t) + (size_t) (TQ6_BM + BN)*TQ6_LD*sizeof(half2);
    const dim3 grid((M + TQ6_BM - 1)/TQ6_BM, (N + BN - 1)/BN, splits);
    tile_q6_K<TN><<<grid, 256, smem, stream>>>(W, XS, S, P, M, K, N, nwin, sb_per);
}

// GGML_CUDA_TILE_Q6K_STATS=1: histogram of the column counts of every q6_K matmul offered to this hook (all widths > 8 on
// Pascal reach it), printed to stderr at exit. Diagnostics only.
static std::atomic<uint64_t> tq6_hist[2][9];   // [taken][bucket]
static const char * tq6_bucket_name[9] = {"9-16", "17-32", "33-48", "49-64", "65-80", "81-128", "129-256", "257-512", ">512"};
static void tq6_stats_print() {
    fprintf(stderr, "tile_q6_K stats (q6_K matmul calls by column count): taken / cuBLAS\n");
    for (int b = 0; b < 9; ++b) {
        fprintf(stderr, "  %-8s %10llu / %10llu\n", tq6_bucket_name[b], (unsigned long long) tq6_hist[1][b].load(), (unsigned long long) tq6_hist[0][b].load());
    }
}
static void tq6_stats_note(const int64_t n, const bool taken) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_TILE_Q6K_STATS"); if (e && e[0] == '1') { atexit(tq6_stats_print); return true; } return false; }();
    if (!on) {
        return;
    }
    const int b = n <= 16 ? 0 : n <= 32 ? 1 : n <= 48 ? 2 : n <= 64 ? 3 : n <= 80 ? 4 : n <= 128 ? 5 : n <= 256 ? 6 : n <= 512 ? 7 : 8;
    tq6_hist[taken][b]++;
}

bool ggml_cuda_tile_q6_K_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_TILE_Q6K"); return !e || e[0] != '0'; }();
    // GGML_CUDA_TILE_Q6K_MIN / _MAX: the column range that takes this path (default 9..80)
    static const int nmin = [] { const char * e = getenv("GGML_CUDA_TILE_Q6K_MIN"); return e ? atoi(e) : 9; }();
    static const int nmax = [] { const char * e = getenv("GGML_CUDA_TILE_Q6K_MAX"); return e ? std::min(atoi(e), 80) : 80; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t nbk = K/QK_K;
    // Pascal only; one weight matrix, contiguous q6_K rows; the aligned-down 53-word staging reads up to 2 bytes past a
    // 4-aligned block but never past the tensor when M*nbk is even (the last block then ends 4-aligned).
    if (!enabled || cc != GGML_CUDA_CC_PASCAL || src0->type != GGML_TYPE_Q6_K || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < nmin || N > nmax || K % QK_K != 0 || (M*nbk) % 2 != 0 ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[1] != (size_t) nbk*sizeof(block_q6_K) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            ((uintptr_t) src0->data % 2) != 0 || (src1->nb[1] % sizeof(float)) != 0 || (dst->nb[1] % sizeof(float)) != 0) {
        if (src0->type == GGML_TYPE_Q6_K && cc == GGML_CUDA_CC_PASCAL) {
            tq6_stats_note(N, false);
        }
        return false;
    }
    tq6_stats_note(N, true);
    const int tn = N <= 16 ? 1 : N <= 32 ? 2 : N <= 48 ? 3 : N <= 64 ? 4 : 5;
    const int nwin = (int) ((K + TQ6_WIN - 1)/TQ6_WIN);
    const int tiles = (int) (((M + TQ6_BM - 1)/TQ6_BM)*((N + 16*tn - 1)/(16*tn)));
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    int splits = std::max(1, std::min((int) nbk/2, (8*nsm + tiles - 1)/tiles));
    const int sb_per = (int) ((nbk + splits - 1)/splits);
    splits = (int) ((nbk + sb_per - 1)/sb_per);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<__half> xs(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float>  sc(ctx.pool(), N*nwin);
    ggml_cuda_pool_alloc<float>  part(ctx.pool(), (int64_t) splits*N*M);
    {
        const dim3 pb(WARP_SIZE, 4), pg((nwin + 3)/4, (unsigned) N);
        tile_q6_K_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, (int64_t) (src1->nb[1]/sizeof(float)), xs.get(), sc.get(),
                                              (int) K, nwin);
    }
    const block_q6_K * W = (const block_q6_K *) src0->data;
    switch (tn) {
        case 1:  tile_q6_K_launch<1>(W, xs.get(), sc.get(), part.get(), (int) M, (int) K, (int) N, nwin, splits, sb_per, stream); break;
        case 2:  tile_q6_K_launch<2>(W, xs.get(), sc.get(), part.get(), (int) M, (int) K, (int) N, nwin, splits, sb_per, stream); break;
        case 3:  tile_q6_K_launch<3>(W, xs.get(), sc.get(), part.get(), (int) M, (int) K, (int) N, nwin, splits, sb_per, stream); break;
        case 4:  tile_q6_K_launch<4>(W, xs.get(), sc.get(), part.get(), (int) M, (int) K, (int) N, nwin, splits, sb_per, stream); break;
        default: tile_q6_K_launch<5>(W, xs.get(), sc.get(), part.get(), (int) M, (int) K, (int) N, nwin, splits, sb_per, stream); break;
    }
    tile_q6_K_reduce<<<std::min<int64_t>(1024, (N*M + 255)/256), 256, 0, stream>>>(part.get(), (float *) dst->data, (int) M, (int) N,
                                                                                  (int64_t) (dst->nb[1]/sizeof(float)), splits);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
