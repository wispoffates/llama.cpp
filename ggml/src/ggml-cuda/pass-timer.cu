#include "pass-timer.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>

// p100-series per-pass timer (GGML_CUDA_PASS_TIMER=<path prefix>, exp/passtimer). One record per meta-backend graph compute (= one pass
// of one context: a verify batch, a draft step, a prefill ubatch) with
//   host: CLOCK_REALTIME at entry (h0) and after both devices issued their last launch (h1)
//   per device: %globaltimer when its stream reached the pass's first launch (g0) and finished its last one (g1), from a 1-thread mark
//   kernel issued at both ends of the device's dispatch, and the LRU fill counter at both marks (f0, f1)
// The mark kernel writes into mapped pinned host memory, so nothing is copied or synchronized for the timer; completed records are
// appended to <prefix>.<pid>.csv on the next pass. A calibration line per device relates %globaltimer to CLOCK_REALTIME.
// Only the main stream is marked: the exclusive cache's write-back stream can still run after g1.
//
// GGML_CUDA_PASS_AB=<feature>[,<feature>...][:period] (default period 1) alternates arm A (as built) and arm B (the listed features
// off) every <period> passes of the same graph signature (node count + output rows), so both arms share one model load, expert-cache
// state and thermal history. Features: shortk2, splitk (mmvf.cu), fillgrid (LRU fill launch geometry, ggml-cuda.cu).

static constexpr int PT_RING   = 1 << 14;
static constexpr int PT_MAXDEV = 8;

struct pt_dev_rec { unsigned long long g0, g1, f0, f1; };   // written by the mark kernel
struct pt_rec     { pt_dev_rec d[PT_MAXDEV]; };
struct pt_meta    { long long pass; int sig, n_out, n_pos, arm; long long h0, h1; std::atomic<unsigned> devs; };

static constexpr int PT_OPRING = 1 << 18;
struct pt_op_rec  { unsigned long long g0, g1; long long n_fill; long long pad; };   // written by the op mark kernel
struct pt_op_meta { std::atomic<long long> pass; int dev, kind, layer, ntok; long long bytes; };

static long long pt_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static __device__ __forceinline__ unsigned long long pt_gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void k_pass_mark(pt_dev_rec * r, const int k, const unsigned long long * fills) {
    const unsigned long long t = pt_gtime();
    const unsigned long long f = fills != nullptr ? *(volatile const unsigned long long *) fills : 0;
    volatile pt_dev_rec * v = r;
    if (k == 0) {
        v->f0 = f;
        v->g0 = t;
    } else {
        v->f1 = f;
        __threadfence_system();
        v->g1 = t;   // last: the host takes g1 != 0 as "record complete"
    }
}

static __global__ void k_op_mark(pt_op_rec * r, const int k, const int32_t * fill_count) {
    const unsigned long long t = pt_gtime();
    volatile pt_op_rec * v = r;
    if (k == 0) {
        v->g0 = t;
    } else {
        v->n_fill = fill_count != nullptr ? *(volatile const int32_t *) fill_count : -1;
        __threadfence_system();
        v->g1 = t;
    }
}

static __global__ void k_pass_calib(unsigned long long * t) {
    *(volatile unsigned long long *) t = pt_gtime();
}

namespace {
struct pt_state {
    bool timer = false, ab = false;
    int  period = 1;
    std::string features;
    std::mutex mtx;
    FILE * f = nullptr;
    pt_rec  * rec  = nullptr;   // mapped pinned host memory (UVA: the same pointer works in kernels)
    pt_meta * meta = nullptr;
    long long next = 0, flushed = 0;
    std::atomic<int> slot{-1};
    std::atomic<int> arm{0};
    std::map<std::pair<int, int>, long long> sig_count;
    std::atomic<unsigned long long *> fills[PT_MAXDEV] = {};
    bool calibrated = false;
    bool ops = false;
    FILE * fo = nullptr;
    pt_op_rec  * oprec  = nullptr;
    pt_op_meta * opmeta = nullptr;
    std::atomic<long long> op_next{0};
    long long op_flushed = 0;

    pt_state() {
        const char * pre = getenv("GGML_CUDA_PASS_TIMER");
        const char * ab_env = getenv("GGML_CUDA_PASS_AB");
        if (ab_env != nullptr && *ab_env) {
            ab = true;
            features = ab_env;
            const size_t c = features.find(':');
            if (c != std::string::npos) {
                period = std::max(1, atoi(features.c_str() + c + 1));
                features.resize(c);
            }
            features = "," + features + ",";
            GGML_LOG_INFO("%s: A/B switch on (GGML_CUDA_PASS_AB): arm B turns off %s every %d pass(es) per graph\n", __func__, ab_env, period);
        }
        if (pre != nullptr && *pre) {
            char path[1024];
            snprintf(path, sizeof(path), "%s.%d.csv", pre, (int) getpid());
            f = fopen(path, "w");
            if (f == nullptr ||
                    cudaHostAlloc((void **) &rec, sizeof(pt_rec) * PT_RING, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_WARN("%s: GGML_CUDA_PASS_TIMER: cannot open %s or allocate the record ring, timer off\n", __func__, path);
                if (f) { fclose(f); f = nullptr; }
                rec = nullptr;
            } else {
                memset((void *) rec, 0, sizeof(pt_rec) * PT_RING);
                meta  = new pt_meta[PT_RING];
                timer = true;
                fprintf(f, "pass,sig,n_out,n_pos,arm,h0_ns,h1_ns,dev,g0_ns,g1_ns,fills\n");
                GGML_LOG_INFO("%s: per-pass timer on (GGML_CUDA_PASS_TIMER) -> %s\n", __func__, path);
                const char * oe = getenv("GGML_CUDA_PASS_TIMER_OPS");
                if (oe != nullptr && atoi(oe) != 0) {
                    snprintf(path, sizeof(path), "%s.%d.ops.csv", pre, (int) getpid());
                    fo = fopen(path, "w");
                    if (fo != nullptr && cudaHostAlloc((void **) &oprec, sizeof(pt_op_rec) * PT_OPRING,
                                                       cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess) {
                        memset((void *) oprec, 0, sizeof(pt_op_rec) * PT_OPRING);
                        opmeta = new pt_op_meta[PT_OPRING];
                        for (int i = 0; i < PT_OPRING; ++i) {
                            opmeta[i].pass.store(LLONG_MAX);
                        }
                        ops = true;
                        fprintf(fo, "pass,dev,kind,layer,ntok,bytes,n_fill,g0_ns,g1_ns\n");
                        GGML_LOG_INFO("%s: op marks on (GGML_CUDA_PASS_TIMER_OPS) -> %s\n", __func__, path);
                    } else {
                        (void) cudaGetLastError();
                        if (fo) { fclose(fo); fo = nullptr; }
                        oprec = nullptr;
                    }
                }
            }
        }
    }
    ~pt_state() {
        if (f) {
            // the process is exiting: write whatever completed
            flush(true);
            fclose(f);
        }
        if (fo) {
            flush_ops(true);
            fclose(fo);
        }
    }

    // op records of passes already written (their end marks ran after every op mark of the pass on that stream)
    void flush_ops(bool force) {
        const long long hi = op_next.load();
        while (op_flushed < hi) {
            const int i = (int) (op_flushed % PT_OPRING);
            pt_op_meta & m = opmeta[i];
            const long long p = m.pass.load(std::memory_order_acquire);
            if (p == LLONG_MAX || (!force && p >= flushed)) {
                break;
            }
            const volatile pt_op_rec & r = oprec[i];
            if (r.g1 != 0) {
                fprintf(fo, "%lld,%d,%d,%d,%d,%lld,%lld,%llu,%llu\n", p, m.dev, m.kind, m.layer, m.ntok, m.bytes, r.n_fill, r.g0, r.g1);
            }
            m.pass.store(LLONG_MAX);
            if ((++op_flushed & 1023) == 0) {
                fflush(fo);
            }
        }
    }

    // CLOCK_REALTIME vs %globaltimer per device: best of 5 launch + sync round trips
    void calibrate() {
        int cur = 0, ndev = 0;
        cudaGetDevice(&cur);
        cudaGetDeviceCount(&ndev);
        for (int d = 0; d < std::min(ndev, PT_MAXDEV); ++d) {
            cudaSetDevice(d);
            cudaStream_t s;
            if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
                (void) cudaGetLastError();
                continue;
            }
            long long best_h0 = 0, best_h1 = 0; unsigned long long best_g = 0;
            for (int k = 0; k < 5; ++k) {
                const long long h0 = pt_now_ns();
                k_pass_calib<<<1, 1, 0, s>>>(&rec[PT_RING - 1].d[0].g0);
                cudaStreamSynchronize(s);
                const long long h1 = pt_now_ns();
                const unsigned long long g = *(volatile unsigned long long *) &rec[PT_RING - 1].d[0].g0;
                if (best_h1 == 0 || h1 - h0 < best_h1 - best_h0) {
                    best_h0 = h0; best_h1 = h1; best_g = g;
                }
            }
            cudaStreamDestroy(s);
            memset((void *) &rec[PT_RING - 1], 0, sizeof(pt_rec));
            fprintf(f, "# calib dev=%d host0_ns=%lld gpu_ns=%llu host1_ns=%lld\n", d, best_h0, best_g, best_h1);
        }
        cudaSetDevice(cur);
        (void) cudaGetLastError();
    }

    // write completed records in order; force: also incomplete ones (exit, or the ring wrapped)
    void flush(bool force) {
        while (flushed < next) {
            const int s = (int) (flushed % PT_RING);
            const pt_meta & m = meta[s];
            const unsigned devs = m.devs.load();
            bool done = m.h1 != 0;
            for (int d = 0; d < PT_MAXDEV && done; ++d) {
                if (devs & (1u << d)) {
                    done = *(volatile unsigned long long *) &rec[s].d[d].g1 != 0;
                }
            }
            if (!done && !force) {
                break;
            }
            for (int d = 0; d < PT_MAXDEV; ++d) {
                if (devs & (1u << d)) {
                    const volatile pt_dev_rec & r = rec[s].d[d];
                    fprintf(f, "%lld,%d,%d,%d,%d,%lld,%lld,%d,%llu,%llu,%llu\n", m.pass, m.sig, m.n_out, m.n_pos, m.arm, m.h0, m.h1, d,
                            r.g0, r.g1, r.f1 - r.f0);
                }
            }
            if ((++flushed & 63) == 0) {
                fflush(f);
            }
        }
    }
};
}

static pt_state & pt() {
    static pt_state s;
    return s;
}

bool ggml_cuda_pass_active() {
    return pt().timer || pt().ab;
}

void ggml_cuda_pass_begin(int sig, int n_out, int n_pos) {
    pt_state & s = pt();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.ab) {
        const long long c = s.sig_count[{sig, n_out}]++;
        s.arm.store((int) ((c / s.period) & 1));
    }
    if (!s.timer) {
        return;
    }
    if (!s.calibrated) {
        s.calibrated = true;
        s.calibrate();
    }
    s.flush(false);
    if (s.next - s.flushed >= PT_RING - 1) {
        s.flush(true);
    }
    if (s.ops) {
        s.flush_ops(s.op_next.load() - s.op_flushed >= PT_OPRING - 4096);
    }
    const int slot = (int) (s.next % PT_RING);
    memset((void *) &s.rec[slot], 0, sizeof(pt_rec));
    pt_meta & m = s.meta[slot];
    m.pass = s.next++;
    m.sig = sig; m.n_out = n_out; m.n_pos = n_pos; m.arm = s.arm.load();
    m.h1 = 0;
    m.devs.store(0);
    m.h0 = pt_now_ns();
    s.slot.store(slot);
}

void ggml_cuda_pass_mark(int device, cudaStream_t stream, int k) {
    pt_state & s = pt();
    const int slot = s.slot.load();
    if (!s.timer || slot < 0 || device >= PT_MAXDEV) {
        return;
    }
    if (k == 0) {
        if (s.fills[device].load() == nullptr) {
            unsigned long long * p = nullptr;
            if (cudaMalloc((void **) &p, sizeof(*p)) == cudaSuccess && cudaMemset(p, 0, sizeof(*p)) == cudaSuccess) {
                s.fills[device].store(p);
            } else {
                (void) cudaGetLastError();
            }
        }
        s.meta[slot].devs.fetch_or(1u << device);
    }
    k_pass_mark<<<1, 1, 0, stream>>>(&s.rec[slot].d[device], k, s.fills[device].load());
}

void ggml_cuda_pass_end() {
    pt_state & s = pt();
    const int slot = s.slot.load();
    if (s.timer && slot >= 0) {
        s.meta[slot].h1 = pt_now_ns();
    }
}

unsigned long long * ggml_cuda_pass_fill_counter(int device) {
    return device < PT_MAXDEV ? pt().fills[device].load(std::memory_order_relaxed) : nullptr;
}

bool ggml_cuda_ab_listed(const char * feature) {
    const pt_state & s = pt();
    return s.ab && s.features.find("," + std::string(feature) + ",") != std::string::npos;
}

int ggml_cuda_ab_arm() {
    return pt().arm.load(std::memory_order_relaxed);
}

bool ggml_cuda_pass_ops_on() {
    return pt().ops;
}

int ggml_cuda_pass_op_begin(int device, cudaStream_t stream, int kind, int layer, long long bytes, int ntok) {
    pt_state & s = pt();
    const int slot = s.slot.load();
    if (!s.ops || slot < 0) {
        return -1;
    }
    const long long n = s.op_next.fetch_add(1);
    const int i = (int) (n % PT_OPRING);
    if (s.opmeta[i].pass.load() != LLONG_MAX) {
        return -1;   // ring full (not flushed yet): drop this op
    }
    memset((void *) &s.oprec[i], 0, sizeof(pt_op_rec));
    pt_op_meta & m = s.opmeta[i];
    m.dev = device; m.kind = kind; m.layer = layer; m.ntok = ntok; m.bytes = bytes;
    k_op_mark<<<1, 1, 0, stream>>>(&s.oprec[i], 0, nullptr);
    m.pass.store(s.meta[slot].pass, std::memory_order_release);
    return i;
}

void ggml_cuda_pass_op_end(int idx, cudaStream_t stream, const int32_t * fill_count) {
    if (idx >= 0) {
        k_op_mark<<<1, 1, 0, stream>>>(&pt().oprec[idx], 1, fill_count);
    }
}
