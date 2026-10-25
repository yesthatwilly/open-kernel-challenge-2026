# muse_moe_w4a16 · 2026-10-25T003435Z · check · post-norm-pdl-launch-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T183649Z-post-norm-pdl-launch-small-m.md` (branch `hyp/muse_moe_w4a16/post-norm-pdl-launch-small-m`).

- **Job:** `5202cf15936c4492` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `a53a567`

## Goal
num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel body is otherwise unchanged and every other launch is master's. Launch the post-norm with programmatic dependent launch so its launch latency and prologue overlap the tail of the w2 Marlin grid inside the CUDA graph. (1) In _post_norm_combine_kernel add a constexpr USE_PDL; after the static loads that do not depend on Marlin (offs, mask, nw = norm weight + 1) and before the first load of ids/topk_weights/y, do 'if USE_PDL: tl.extra.cuda.gdc_wait()'. (topk_weights and local_ids are written before Marlin, but keep every data-dependent load after the wait for simplicity.) No gdc_launch_dependents call (nothing after it in the graph needs it). (2) Launch with launch_pdl=True (Triton launcher kwarg) when USE_PDL. (3) prepare(): set state['pdl_ok'] by a self-test: build a tiny y/ids/tw case, capture in a torch.cuda.CUDAGraph a plain elementwise kernel writing y followed by the PDL post-norm launch, replay 3 times, and compare with the USE_PDL=False launch (torch.equal). Any exception (kwarg unknown to this Triton, gdc_wait missing, capture error) or mismatch -> pdl_ok False. Also warm up/compile both variants for every num_tokens < 32 there. kernel_fn passes USE_PDL=state['pdl_ok']. Print one line '[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)' in prepare so the run log says which path was measured.

## What changed, and where
Only muse_moe_w4a16/submission.py changed (+68/-5).
- `_post_norm_combine_kernel` gains constexpr `USE_PDL` (default False); `tl.extra.cuda.gdc_wait()` sits after the norm-weight load and before the first ids/topk_weights/y load. No gdc_launch_dependents.
- New `_launch_post_norm(...)` wrapper passes `launch_pdl=True` only when use_pdl. kernel_fn (num_tokens < 32 only) calls it with `state["pdl_ok"]`; every other launch is master's.
- New `_pdl_self_test(device, post_norm_w)`, in-process, no child process: warms the PDL variant, captures a CUDAGraph of an elementwise `y.copy_` plus the PDL launch, replays 3 times, torch.equal vs the plain launch. Whole body in try/except Exception, returns (False, reason) on any error or mismatch, never raises.
- prepare() stores `pdl_ok`, compiles both variants for tokens 1..31 (only if the self-test passed; warmup also guarded), prints `[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)`.
- Deviation: none. The self-test uses seeded synthetic data from torch.Generator, not any input.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn has no sync/.item()/prints; the PDL flag is a prepare-time bool, the branch is on shape only (checked in diff).
- [x] Nothing derived from `data` is cached across calls: state adds only pdl_ok (self-test on synthetic data) and the contiguous post_norm_w weight.
- [x] Weights: n/a: no weight repack change; post_norm_w is the same BF16 weight made contiguous in prepare().
- [x] Activations BF16 where SGLang keeps BF16; unchanged (n/a for KV/queries).
- [x] Attention: n/a: MoE problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: self-test is in-process, no subprocess/os/sys use.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: "OK with warnings" (only host-sync warnings in prepare()/self-check code, outside kernel_fn's call graph). py_compile passed; __pycache__ removed.

## Risks / what could fail
medium
Mechanism parity with the running cuda-dispatch-combine-pdl-verify (PDL on its CUDA combine at >32), here on the Triton combine at <32: shape-disjoint. Marlin never triggers, so the implicit trigger at its block exits applies; the win is the launch/prologue overlap of the last w2 wave only. Must read the PATH CHOSEN line before judging: pdl off means the run measured master. Risk medium: Triton version on the judge (torch 2.14.1) should have launch_pdl and tl.extra.cuda.gdc_wait, but if not the fallback keeps it a master-equivalent run, not a failure. Kernel time in the profile will not move; judge on the shape median vs master's bs=1/8/16 rows. Overlaps post-norm-combine-hoisted-slot-loads (same kernel, different mechanism); keep them as separate runs for attribution, then restack. Replaced at bs=1 by the running bs1 fused rewrite if it lands.

## Result (after the run)
- job `5202cf15936c4492` (check), status done, passed True, tests 5/5
- speedup 1.3222x, 95% CI [1.3024, 1.3421], score 10170.987287590442 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/post-norm-pdl-launch-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/post-norm-pdl-launch-small-m.patch`; branch `hyp/muse_moe_w4a16/post-norm-pdl-launch-small-m`
