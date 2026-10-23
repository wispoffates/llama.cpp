# p100-series

A branch of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for **NVIDIA Tesla P100 16 GB** (Pascal, sm_60: fp16 but no
tensor cores, no bf16), developed and served on two P100s with tensor split. It is upstream plus a linear patch stack, replayed
onto new upstream commits regularly; releases are tags `p100-series-vX.Y`. See [CREDITS.md](CREDITS.md) for where the work comes from
and [CONTRIBUTING.md](CONTRIBUTING.md) to contribute (AI-written PRs are welcome).

Everything here is runtime- or arch-gated: other GPUs take upstream's paths.

Numbers for every step of the stack (speed, accuracy, serving, Flash-Next): [docs/p100/PERFORMANCE.md](docs/p100/PERFORMANCE.md).

## What it adds (highlights)

- **Decode matvecs for sm_60**: staged / cached activations, fp16 matvecs for Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K (one column and the
  2..8-column speculative-verify widths), short-K and long-K 16-bit matvecs, a fused Pascal tile GEMM for 9..80-column verify batches
  and MoE prefill (Q2_0 / IQ2 / IQ1_M / K-quants), gate/up/GLU fusion.
- **Two-GPU tensor split**: one host thread per device, an in-kernel P2P AllReduce (LL protocol for decode-size reductions) that
  also computes the following RMS norm, tied-output splitting.
- **MoE expert cache for models larger than VRAM**: experts in pinned host RAM with an LRU cache per GPU (lich-77's base), prefill
  through the cache pools, exclusive cache, copy-engine assembly, profile-warm start, per-layer slot budgets (`LLAMA_MOE_LRU_SLOTS`).
- **Qwen 3.8 Flash-Next (qwen4exp) on Pascal**: sparse (gathered) flash attention in the tile kernel for QSA layers, reduced-vocab
  MTP draft heads (d2t), topk-moe for pruned expert counts.
- **Fixes**: NaN-safe flash-attention masking, recurrent-state clear with fill, several sm_60 bounds fixes.

Most switches are environment variables (`GGML_CUDA_*`, `GGML_META_*`, `LLAMA_MOE_*`) documented in the commit that adds them.

## Build

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Developed with CUDA 12.4. For two cards: `-sm tensor` and `GGML_CUDA_P2P=1` (needs working PCIe peer access).

## How changes are accepted

A change ships in a release only after a gate on the P100s: `test-backend-ops`, KL divergence against full-F32 GPU references on
chat-formatted sets (prefill, single-token decode and verify widths), every served model's output md5 and draft acceptance, ABBA speed
runs in llama-bench and in serving, and soak tests. Upstream changes that alter Pascal numerics without a gain are reverted
(listed in the release tag messages).

## License

MIT, as upstream. Carried code keeps its authors' notices and git authorship.
