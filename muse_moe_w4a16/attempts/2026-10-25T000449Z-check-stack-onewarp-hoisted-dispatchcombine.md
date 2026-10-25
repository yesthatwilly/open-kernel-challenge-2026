# muse_moe_w4a16 · 2026-10-25T000449Z · check · stack-onewarp-hoisted-dispatchcombine

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T000436Z-stack-onewarp-hoisted-dispatchcombine.md` (branch `hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine`).

- **Job:** `7e5ab95a413f4ebe` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `795e035`

## Goal
Built on master. Stack the three moe results of 2026-10-24 that each matched the 1.3278 best within noise, applied unchanged from their done/ patches: align-small-one-warp-tiny-m (1-warp _align_small_kernel at T<=8, scored 1.3287), post-norm-combine-hoisted-slot-loads (all 8 slot loads before the reduction, bs<32, scored 1.3282) and cuda-dispatch-combine-pdl-verify (CUDA align + PDL post-norm combine at 32<T<=512, scored 1.3283). Their shapes and kernels are disjoint except the post-norm combine, which the hoisted change edits only at bs<32 and the CUDA rewrite replaces only above 32 tokens. Tests whether sub-noise wins add up, as prefill's stack showed (1.059 -> 1.0744).

## What changed, and where
Built in the main thread 2026-10-25 ~00:10Z (reset priority 4, user-specified stack; self-reviewed). Worktree on master; `git apply -3` of the three done/neutral patches (align-small-one-warp-tiny-m, post-norm-combine-hoisted-slot-loads, cuda-dispatch-combine-pdl-verify), all clean, no edits. Every parent's base already carried align-parallel-per-expert-verify. Run-time disjoint: the one-warp launch is `_align_small` at n <= 64 (T <= 8); the hoisted loads are the Triton post-norm combine at bs < 32; the CUDA TU replaces `_align_parallel` and the SGLang combine only at 32 < M <= 512, behind its own prepare() self-check (combine gate row 8e-3 / global 4e-3, as audited) with fallback to master.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the three parents, each passed the judge.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 unchanged.
- [x] Activations BF16.
- [x] n/a: no attention in this problem.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Reset priority 4 (user-requested stack, 2026-10-24 ~23:50Z); claimed with --skip-review on that instruction.

## Result (after the run)
- job `7e5ab95a413f4ebe` (check), status done, passed True, tests 5/5
- speedup 1.3176x, 95% CI [1.2979, 1.3374], score 10206.541902305229 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/stack-onewarp-hoisted-dispatchcombine.patch`; branch `hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine`
