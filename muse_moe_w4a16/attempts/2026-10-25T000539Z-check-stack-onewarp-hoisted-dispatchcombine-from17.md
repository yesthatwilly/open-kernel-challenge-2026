# muse_moe_w4a16 · 2026-10-25T000539Z · check · stack-onewarp-hoisted-dispatchcombine-from17

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T000511Z-stack-onewarp-hoisted-dispatchcombine-from17.md` (branch `hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine-from17`).

- **Job:** `0fde9013de97404d` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `8f0cad7`

## Goal
Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches, unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_TOKENS 32 -> 16, BSM=8 in the _align_parallel warm-up; scored 1.3250), so tokens 17..32 use the 33-program parallel align. That range sits under the CUDA dispatch rewrite's (> 32 tokens) and above the one-warp change (<= 8), so all four are run-time disjoint.

## What changed, and where
Built in the main thread 2026-10-25 ~00:12Z (reset priority 4 variant, user-specified; self-reviewed). `git apply -3` of the four done/neutral patches, all clean. ONE EDIT beyond the patches: the CUDA dispatch path's gate was `ALIGN_FUSED_MAX_TOKENS < num_tokens <= CDC_MAX_TOKENS`, so lowering ALIGN_FUSED_MAX_TOKENS to 16 (align-parallel-from-17-tokens) would have moved 17..32 tokens onto the CUDA kernels, which their self-check never covered (33/128/256/257). Added `CDC_MIN_TOKENS = 32` and gated the CUDA path on it, so 17..32 tokens run the Triton `_align_parallel` as align-parallel-from-17-tokens intended and the CUDA path keeps exactly its audited range. The CUDA path's PATH CHOSEN text still prints ALIGN_FUSED_MAX_TOKENS as its lower bound (cosmetic).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the four parents; the gate edit is a config-only comparison.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 unchanged.
- [x] Activations BF16.
- [x] n/a: no attention in this problem.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Reset priority 4 variant (user-requested), claimed with --skip-review on that instruction.

## Result (after the run)
- job `0fde9013de97404d` (check), status done, passed True, tests 5/5
- speedup 1.3237x, 95% CI [1.3039, 1.3436], score 10159.421883427603 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine-from17.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine-from17.patch`; branch `hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine-from17`
