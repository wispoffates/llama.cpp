# Credits

The `p100-series` branch is upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) plus a linear stack of patches
for NVIDIA Tesla P100 (Pascal, sm_60) GPUs. Much of it builds on other people's work. Code we carry keeps its original author
in git; ported or adapted pieces name their source in `Cherry-picked-from:`, `Ported-from:` and `Co-authored-by:` trailers.
Commits written by the maintainer with AI help carry an `Assisted-by:` trailer.

## Code carried in this branch

| Source | What we carry | License |
|---|---|---|
| [Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp) `p100-optimizations` (Kaden) | the sm_60 matvec rebuild (staged activations, vdr/geometry, cached q8_1 activation), fp16 Q6_K verify matvecs, cuBLAS ALGO6 prefill, the sampler top-k prefilter, MTP catch-up and q4_0 draft head -- split into individual commits authored by Kaden, each linking its original | MIT |
| [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp) | GP100 fp16 matvec (patches 0007, 0008), recurrent-state clear with fill (0014), activation buffer sizing (0017) | MIT |
| [lich-77/llama.cpp](https://github.com/lich-77/llama.cpp) | GPU-resident MoE expert LRU cache and its tensor-split support (the base of our expert cache) | MIT |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | meta-backend offload of host-weight ops and split-aware expert copies | MIT |
| drrros, [ggml-org/llama.cpp#29963](https://github.com/ggml-org/llama.cpp/pull/29963) | pipeline parallelism with MoE experts in host memory | MIT |
| wrj97, [ggml-org/llama.cpp#29509](https://github.com/ggml-org/llama.cpp/pull/29509) | draft KV kept out of context checkpoints | MIT |
| other open upstream PRs | carried until merged, each with its original author and PR number in the message | MIT |

## Ideas and measurements we learned from (no code copied)

- [Niko1221/Strata](https://github.com/Niko1221/Strata) -- Flash-Next on consumer hardware: route-tail expert skip, QSA memory tricks (MIT)
- [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) -- fused up/gate and P2P reduce ideas, register-resident row sums (#2721)
- [shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches) -- a measured, rebased patch series for P100; our maintenance model
- [apollo-mg/Project-Apollo](https://github.com/apollo-mg/Project-Apollo), [vmelnic/pascal-llm](https://github.com/vmelnic/pascal-llm) -- P100 profiling findings and prior art (these repos have no license; we used findings only)
- [JVrachnis/pascal-llm-serving](https://github.com/JVrachnis/pascal-llm-serving) -- per-type sm_60 matvec costs
- [thefallentree/llama.cpp-gp100](https://github.com/thefallentree/llama.cpp-gp100) -- sm_60 patch ideas

## Projects we watch

Checked at every upstream sync for patches worth testing on Pascal:
[Kmic-68/llama.cpp](https://github.com/Kmic-68/llama.cpp), [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp),
[shinbunbun/llama-cpp-p100-patches](https://github.com/shinbunbun/llama-cpp-p100-patches), [poisonxa16/pxa](https://github.com/poisonxa16/pxa),
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp),
[ifm-ai/llama.cpp](https://github.com/ifm-ai/llama.cpp) (K2 Horizon), [thefallentree/llama.cpp-gp100](https://github.com/thefallentree/llama.cpp-gp100),
[JVrachnis/pascal-llm-serving](https://github.com/JVrachnis/pascal-llm-serving), [vmelnic/pascal-llm](https://github.com/vmelnic/pascal-llm),
[apollo-mg/Project-Apollo](https://github.com/apollo-mg/Project-Apollo), [Niko1221/Strata](https://github.com/Niko1221/Strata);
and for an RDNA3 desktop card: [gencer-altan/tulpar.cpp](https://github.com/gencer-altan/tulpar.cpp),
[stew675/llama-cpp-rdna-boosts](https://github.com/stew675/llama-cpp-rdna-boosts), [apollosenvy/kernel-anvil](https://github.com/apollosenvy/kernel-anvil),
[nasone32/llama.cpp-RDNA3-7900xtx-opt](https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt),
[victoralcazardev/rx7900xtx-local-llm](https://github.com/victoralcazardev/rx7900xtx-local-llm).

Thank you to all of them, and to the ggml / llama.cpp authors.
