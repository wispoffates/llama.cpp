# Contributing to p100-series

This file covers this fork only. To contribute to llama.cpp itself, follow upstream's
[CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md); its rules on AI-written PRs are stricter.

## AI is welcome

AI-written code, commit messages, PR descriptions and review replies are all fine here. Many of this branch's own commits were
written with Claude Code. We judge a PR by what it changes and how it was measured, not by who or what typed it.

Saying how AI was involved is appreciated but not required. An `Assisted-by:` trailer in the commit message is enough, for example
`Assisted-by: Claude Opus 5.5 (Claude Code)`. Autonomous agents are welcome too. Please make sure someone is reachable to answer
questions about the PR.

## What fits

- Speed or correctness work for Pascal GPUs (sm_60 P100 first; sm_61 P40 / P4 / GTX 10xx results are useful too)
- Multi-GPU serving on Pascal: tensor split, AllReduce, P2P, host-side dispatch
- The MoE expert cache (models larger than VRAM) and the Qwen 3.8 Flash-Next path
- Measurements: a benchmark or KL divergence run on your hardware is a real contribution, even without code
- Bug reports with the model, quant, flags, GPU, CUDA version and the log

Changes that make sense for every GPU belong upstream. Send them to ggml-org/llama.cpp and we'll pick them up at the next sync.

## What a PR needs

1. **Gating.** New paths are arch- or runtime-gated so other GPUs are unchanged. A new kernel or behaviour gets an environment
   switch (`GGML_CUDA_*`, `GGML_META_*`, `LLAMA_MOE_*`) so it can be turned off for an A/B test.
2. **Correctness.** `test-backend-ops` passes for the ops you touched on sm_60. Add cases for new shapes. If you claim the output is
   unchanged, show it (greedy output md5 before/after). If numerics change, show KL divergence against a reference.
3. **Numbers.** Before/after on Pascal: `llama-bench` prompt and generation speed at least, serving numbers if it matters there.
   Name the GPU, CUDA version, model + quant and flags, and alternate the runs (A B B A) so warm-up and clocks don't decide it.
4. **Small commits.** One change per commit, with the why and the numbers in the message. The stack is replayed onto upstream
   regularly, so keep diffs to upstream files small and avoid unrelated reformatting. No merge commits; rebase onto `p100-series`.
5. **Attribution.** Code from elsewhere keeps its original author. Name the source with `Cherry-picked-from:`, `Ported-from:` or
   `Co-authored-by:` trailers, and make sure its license allows it (MIT, Apache-2.0, ...). Nothing from repositories without a
   license; findings and ideas from them are fine with a mention. See [CREDITS.md](CREDITS.md).

No Pascal card? Open the PR anyway and say so. The maintainer runs the gate on 2x P100 when time allows.

## Code style

Follow upstream's [coding guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md#coding-guidelines) and
naming rules. Match the surrounding code.

## License

MIT, as upstream. By contributing you agree your contribution is licensed under it.
