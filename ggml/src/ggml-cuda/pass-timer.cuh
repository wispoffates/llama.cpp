#pragma once

#include "common.cuh"

// p100-series per-pass timer and in-process A/B switch (see pass-timer.cu). Both are off unless their environment variable is set.

// true when GGML_CUDA_PASS_TIMER or GGML_CUDA_PASS_AB is set: the meta backend then calls begin / mark / end around every pass
bool ggml_cuda_pass_active();
// host, before any device issues the pass: sig = graph node count, n_out = rows of the last node, n_pos = inp_pos length
void ggml_cuda_pass_begin(int sig, int n_out, int n_pos);
// on the device's dispatch thread: k = 0 before its first launch of the pass, k = 1 after its last
void ggml_cuda_pass_mark(int device, cudaStream_t stream, int k);
// host, after every device has issued the pass
void ggml_cuda_pass_end();
// device counter the LRU decide kernel adds its fills to (nullptr while the timer is off)
unsigned long long * ggml_cuda_pass_fill_counter(int device);

// GGML_CUDA_PASS_AB=<feature>[,<feature>...][:period]: true when <feature> is listed (evaluate once per call site)
bool ggml_cuda_ab_listed(const char * feature);
// arm of the pass being issued: 0 = A (as built), 1 = B (listed features off)
int  ggml_cuda_ab_arm();
