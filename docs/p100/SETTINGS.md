# Bench and serving settings

The exact settings behind every number in [PERFORMANCE.md](PERFORMANCE.md): model files, llama-bench flags, serving command lines and
environment, and the accuracy (KL divergence) sets. Measured 2026-10-23 on `p100-series-v5.28` code; `v5.29` is the same code on a newer
upstream base and gave identical outputs in its gate. Paths are shortened; ports and hosts are omitted.

## Hardware and common rules

- 2x Tesla P100-PCIE-16GB (VBIOS 86.00.41.00.06, 150 W cap) on one PCIe switch with peer-to-peer access, passed through to an 8-vCPU
  Ubuntu 26.04 VM on a Ryzen 7 2700 with 46 GiB RAM. NVIDIA driver 580.178.04; built with the CUDA 12.4 toolchain
  (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60`).
- `GGML_CUDA_P2P=1` in every run (without it the two cards reduce through the host and tensor split is ~20 % slower).
- Fans on automatic control; every run waits until both GPUs are at or below 65 °C; one warm-up run per model before measuring.
- Steps are run forward and then in reverse order (mirrored), so slow drift cancels out; a result is the mean of both passes.
- Every step of the ladder, stock llama.cpp and Kmic-68's fork used the same flags below. Flags a build doesn't know would be dropped and
  listed; none were.

## llama-bench (speed tables)

Common: `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3`, plus per model:

| model | extra flags |
|---|---|
| gemma4 26B-A4B | `-sm tensor -ub 1536 -b 2048 -d 0,16384` |
| Swift 27B | `-sm tensor -ts 49.3/50.7` |
| qwen36 35B-A3B | `-sm tensor -d 0,16384` |
| ornith 35B | `-sm tensor` |
| glimmer 30B | `-sm tensor` |
| nemotron 30B-A3B | `-sm layer` (tensor split is not implemented for this architecture) |

The "what each feature is worth today" table re-runs gemma4, Swift and qwen36 (depth 0) on v5.28 with one environment switch off at a
time; the base run is repeated at the start and the end.

## Serving (serving tables)

Each model runs with its production `llama-server` command line and environment (below), one server per measurement:

- **Decode:** three requests at temperature 0 with `cache_prompt: false` after one warm-up request: a chat prompt (512 tokens out), a
  pull-request triage prompt (384) and a code-edit prompt (384). The tables report decode tokens/s, the mean of the three; draft
  acceptance and an md5 of the output are recorded too.
- **Prefill:** one request each with ~1K, ~5-6K and ~11-13K prompt tokens (the count depends on the tokenizer) from a fixed corpus of
  pull-request texts with a short completion; prompt tokens/s.

### gemma4 26B-A4B (MoE)

- Model: `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (UD-Q4_K_XL, QAT; 14.25 GB) sha256 `a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891`
- Vision projector: `mmproj-gemma4qat-F16.gguf` sha256 `d00f211a7d4f7fb19bd9b75d8e9342eccffb5920b08fd9562167560fdfcc5dd1`

```bash
GGML_CUDA_P2P=1 \
llama-server --model gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --mmproj mmproj-gemma4qat-F16.gguf \
  --ctx-size 458752 --kv-unified --n-gpu-layers 99 --split-mode tensor --flash-attn on --parallel 3 \
  --batch-size 2048 --ubatch-size 1536 --jinja --reasoning-budget 4096 --n-predict 8192 \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2
```

### Swift 27B (dense), MTP + ngram-mod speculation

- Model: `Swift-1.5-Qwen3.8-27B-Q6_K.gguf` (Q6_K; 23.86 GB) sha256 `cd2eb6597026ce4eb86100e830be24f01461f84b89d4708634c2ec3d43b67fa8`
- MTP draft: `mtp-Qwen3.8-27B-Q4_0.gguf` (Q4_0; 1.37 GB) sha256 `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e`
- Vision projector: `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` sha256 `daa1116c9422fa390cc8688495da0e91781f92841dfc3b31a378ff252571745a`

```bash
GGML_CUDA_P2P=1 \
llama-server --model Swift-1.5-Qwen3.8-27B-Q6_K.gguf --mmproj mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' --model-draft mtp-Qwen3.8-27B-Q4_0.gguf \
  --ctx-size 24576 --n-gpu-layers 99 --fit off --flash-attn on --split-mode tensor --parallel 2 --kv-unified --jinja \
  --spec-type draft-mtp,ngram-mod --spec-ngram-mod-n-max 63 --tensor-split 49.3,50.7
```

Since 2026-10-24 (after these measurements) production also sets `LLAMA_MTP_DRAFT_VOCAB=98304`: the MTP draft scores the first 98,304
token ids instead of the default 81,920 (+0.7 % decode on average, +1.5 % on triage, identical output).

### qwen36 35B-A3B (MoE)

- Model: `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` (UD-Q5_K_XL; 26.59 GB) sha256 `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`
- Vision projector: `mmproj-qwen36-F16.gguf` sha256 `8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f`

```bash
GGML_CUDA_P2P=1 GGML_CUDA_FUSE_ADD_NORM=0 \
llama-server --model Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf --mmproj mmproj-qwen36-F16.gguf --no-mmproj-offload \
  --ctx-size 163840 --n-gpu-layers 99 --split-mode tensor --flash-attn on --parallel 2 --kv-unified --jinja
```

### ornith 35B (MoE)

- Model: `Ornith-1.5-35B-Q5_K_M.gguf` (Q5_K_M; 25.35 GB) sha256 `91df97de5845100e850b4b5ec5ff35695382020b880fad6f7f51787b3a953bd0`
- Vision projector: `mmproj-Ornith-1.5-35B-BF16.gguf` sha256 `1921a36a85aee56cd2abd27f46701802c9d85a33474792e600df6c3b282a135d`

```bash
GGML_CUDA_P2P=1 \
llama-server --model Ornith-1.5-35B-Q5_K_M.gguf --mmproj mmproj-Ornith-1.5-35B-BF16.gguf --no-mmproj-offload \
  --ctx-size 163840 --n-gpu-layers 99 --split-mode tensor --flash-attn on --parallel 2 --kv-unified --jinja
```

### glimmer 30B

- Model: `Muse-Glimmer-30B-UD-Q6_K_XL.gguf` (UD-Q6_K_XL; 26.27 GB) sha256 `fb5f80d110c4fa932cc652e70873c0bd12c0954009038aa675e65086104c2739`
- Vision projector: `mmproj-Muse-Glimmer-30B-Q8_0.gguf` sha256 `01ff73c95108e1754a4c145176c6d3ba44338942285cb87dcac7f4f193192ea2`

```bash
GGML_CUDA_P2P=1 \
llama-server --model Muse-Glimmer-30B-UD-Q6_K_XL.gguf --mmproj mmproj-Muse-Glimmer-30B-Q8_0.gguf \
  --ctx-size 32768 --n-gpu-layers 99 --fit off --flash-attn on --split-mode tensor --parallel 2 --kv-unified --jinja
```

### nemotron 30B-A3B (MoE, layer split)

- Model: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf` (UD-Q4_K_XL; 25.51 GB) sha256
  `112bf957489a497c18f60bf8bd44ee1dfa05e87368b8ed7f68998a4e38f275c9`

```bash
GGML_CUDA_P2P=1 \
llama-server --model NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf \
  --ctx-size 262144 --n-gpu-layers 99 --flash-attn on --parallel 2 --kv-unified --jinja
```

(no `--split-mode`: the default layer split; tensor split is not implemented for `nemotron_h_moe`.)

### Qwen 3.8 Flash-Next (512 experts, 68 GB; experts in host RAM)

- Model: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` + `-00002-of-00002.gguf` (IQ2_XS; 39.8 + 28.4 GB) sha256
  `ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b` / `e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb`
- MTP heads: `mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k-conv.gguf` (frequency-ranked 65k rows, converted for the reduced-vocabulary loader)
  sha256 `4166a416e7b0b8b821dcac203e18819de82a03982b04235dd0bb4dbac5b9e932` (before conversion:
  `282764bf3ce11b1ff6c715d65b37d7d690eb782734127429d2af6bd72b4c48b9`); `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (full vocabulary)
  sha256 `b9880220df29fc224bbce408c867cd5d9c021263b754033ea624b669e374f4ec`

**p100-series (v5.28 / v5.29), production:**

```bash
GGML_CUDA_P2P=1 LLAMA_MOE_LRU_EXCLUSIVE=1 GGML_CUDA_LRU_DMA=1 LLAMA_MOE_LRU_PREFETCH=1 GGML_CUDA_LRU_ASYNC=1 \
LLAMA_MOE_LRU_PROFILE=profile-flashnext.txt LLAMA_MOE_LRU_SLOTS=slots-flashnext.txt \
llama-server --model Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --model-draft mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k-conv.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 \
  --ctx-size 16384 --n-gpu-layers 99 --fit off --flash-attn on --split-mode tensor -lzm on --no-repack \
  --n-cpu-moe 999 --moe-lru-slots 256 -ub 512 -b 2048 --parallel 1 --jinja
```

- Tensor split is required: the expert cache and the Flash-Next kernels are built and measured for `-sm tensor` only.
- Experts go to host memory with `--n-cpu-moe 999` (not `-ot`); the model is memory-mapped (default load mode).
- `LLAMA_MOE_LRU_PROFILE` (warm start) and `LLAMA_MOE_LRU_SLOTS` (per-layer slot budgets) are data files from our traces; without them the
  cache starts cold with 256 uniform slots (a few % slower decode, slower first request).
- "No speculation" rows: the same line without `--model-draft` and the `--spec-*` flags.

**Comparison setups in the Flash-Next table** (same prompts, `--ctx-size 16384 -ub 512 -b 2048 --parallel 1 --jinja`, `GGML_CUDA_P2P=1`):

| setup | flags that differ |
|---|---|
| stock llama.cpp | `--split-mode layer --n-cpu-moe 999 --moe-cache-mib 14000` (upstream's own MoE cache; it refuses tensor split), MTP head Q8_0 |
| lich-77 LRU cache base (step R9) | `--split-mode tensor --n-cpu-moe 999 --moe-lru-slots 256 --load-mode none`, no MTP (the MTP draft crashes at that commit) |
| Kmic-68 fork | `--split-mode layer --flash-attn off --n-cpu-moe 36`, no MTP (its loader cannot read the standalone MTP head; flash attention on aborts in the QSA layers; fewer CPU expert layers run out of VRAM) |

## Accuracy (KL divergence tables)

`llama-perplexity ... -ngl 99 --fit off -fa 1 --kl-divergence-base <ref> --kl-divergence`, against full-F32 GPU reference logits
computed once on the same text. The chat sets are chat-formatted text (with `LLAMA_PPL_PARSE_SPECIAL=1`); glimmer uses `wiki.test.raw`.

| set | flags |
|---|---|
| prefill | `-c 2048 --chunks 4` |
| decode | `-c 1024 --chunks 3 -b 1024 -ub 1` (one token at a time) |
| verify4 | `-c 1024 --chunks 3 -b 1024 -ub 4` (the speculative-verify width) |
| glimmer prefill | `-f wiki.test.raw -c 2048 --chunks 3` |

Split flags as in production: `-sm tensor` (Swift adds `-ts 49.3,50.7`), nemotron layer split.

## Builds

- p100-series and every ladder step: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release`, CUDA 12.4,
  gcc 15.2 (Ubuntu 26.04 toolchain).
- Kmic-68's fork (tip `d90ec36a3`): the same plus `-DGGML_CUDA_FA_QUANTS=all` (its documented flag) and a one-line `#include <algorithm>`
  in `src/llama-context.cpp` so it builds with gcc 15; no other change.
