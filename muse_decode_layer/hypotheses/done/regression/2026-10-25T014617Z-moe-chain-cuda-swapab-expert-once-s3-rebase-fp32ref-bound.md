---
problem: muse_decode_layer
slug: moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound
created: 2026-10-25T014617Z
base: master
area: w13 marlin + silu-mul + w2 marlin (T>32)
change: >
  Stage 3 of the expert-once swap-AB MoE chain (moe_route_list + moe_w13_silu_swapab +
  moe_w2_swapab replacing moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32). Stage
  2 (moe-chain-cuda-swapab-expert-once-fix-inproc, check 4ebd055adab34ecd) built sm_90a-only, ran
  every canary and both full T=40 chains with NO 715 fault, then turned itself OFF on its own 1e-2
  row bound against Marlin (y row rel 1.07e-2; act row rel 8.79e-3, out global rel 6.00e-3,
  finite, ids equal, repeat identical), so it measured only Marlin, on a stale base (e65eda8, no
  dense chain). REWRITE_READ_2026-10-25 '01:36Z harvest', LESSONS 2026-10-25. Changes, nothing
  else: (1) REBASE onto current decode master (stack-align-par-sidestream-sliding1, submission
  blob 4da94677). Because depends_on a rewrite stage makes claim start the worktree at stage 2's
  commit 1177e31 on the stale base, the implementer's FIRST step in the worktree is: git checkout
  master -- muse_decode_layer/submission.py, then git apply -3 of
  hypotheses/done/regression/moe-chain-cuda-swapab-expert-once-fix-inproc.patch, merging conflicts
  by hand so master's dense chain (dl_ kernels at T<=DL_MAX_TOKENS), parallel align and
  side-stream MoE dispatch all survive; the new path keeps its OWN gate constant (tokens > 32) and
  launches on the side stream master uses for MoE dispatch. Verify the rebase by diffing against
  master: only the MoE large-T path and its prepare() block may differ. (2) RECALIBRATE the
  self-check. Error source: the stage-2 check compared two approximate kernels against each other.
  Both dequantize the same quantized weights and accumulate in fp32, but they round differently:
  the new chain rounds the silu*mul activation to bf16 before w2 (act row rel 8.79e-3 already sits
  in the w2 input), folds the scale with __hmul2 in bf16, and reduces in a different order. Marlin
  carries its own error of the same order, so the y row difference to Marlin is roughly
  sqrt(e_new^2 + e_marlin^2) and can exceed 1e-2 when each kernel alone is under it. Replace the
  Marlin-relative gate with an fp32 reference: in prepare(), dequantize the same expert weights to
  fp32 once (prepare-time only, synthetic inputs, nothing cached for the timed call), compute w13
  -> silu*mul -> w2 -> weighted sum in fp32 for T=40/128/256, and measure BOTH Marlin's and the
  new chain's error to it (global rel L2 and max row rel, near-zero rows measured against a tenth
  of the mean row as the judge does). Enable the path only if new global <= max(1.5 x Marlin
  global, 5e-3) and new max row <= max(1.5 x Marlin max row, 1e-2), AND under absolute caps of
  global <= 1.5e-2 and row <= 3e-2. The caps sit below the judge's tolerance: the judge allows
  3-15% relative L2 on the layer outputs with any row up to 5x its bound (>= 15%), and the MoE
  output is only part of the residual update, so a MoE-output error under 1.5e-2 / 3e-2 stays well
  inside the tightest 3% layer bound. Also print the old Marlin-relative numbers for continuity.
  (3) Print the T=128/256 per-kernel CUDA-event timings (new chain vs Marlin) BEFORE the enable
  decision, whatever the decision is (prepare undercounts the judge profile by 20-25%: rank only).
  KEEP the full canary ladder unchanged: sm_90a image check (capability (9,0), -gencode
  compute_90a/sm_90a only, binaryVersion/ptxVersion 90 for each kernel, dyn-smem attribute print),
  the 1-CTA dequant8 + mma tile micro canary, per-kernel synthetic launches at the smallest T
  above the gate with unbuffered launching/ok prints and torch.cuda.synchronize() after each, then
  the T=40/128/256 self-check; keep plain cp.async.cg, __hmul2 and the sm_90a-only build from
  stage 2. Print [moe-cuda-large-t] ON|OFF with the fp32-ref errors of both kernels, the bounds
  used and PATH CHOSEN, compactly and late (log tail is ~15 KB). First measurement is an okc
  check, never a scored submit.
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: Marlin w13+w2+act_and_mul 254/271 us (sliding/full) vs MoE floor ~101 us; expert-once swap-AB target <= Marlin (M2), ~150 us (M3 ~1.5x floor)
  - shapes: "*load=verify_32 *"
    direction: faster
    why: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
  - shapes: "*load=decode_*"
    direction: flat
    why: "T<=32 outside gate (dense chain / master): control; rebase removes stage 2's stale-base +4-15%"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8 outside gate: control"
risk: high
depends_on: moe-chain-cuda-swapab-expert-once-fix-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32
review: proceed
status: done
review_at: 2026-10-25T034534Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T034535Z
base_sha: 1177e314e5b0c1b2d21bb03967270350659d0f15
base_blob: 451c3c09efdc6fc55db461ae20f27cd8e1cd4e9c
branch: hyp/muse_decode_layer/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound
worktree: .worktrees/muse_decode_layer/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T035159Z
impl_commit: f75681f6130e914b9d18cf614cd177c9d63c48ff
submission_sha256: cdcac6439df93fb054336b23ab4c5cec6367baeb93e5a66a54cc8deef36d4988
norm_sha256: 2e18883291be972ece9c4588d8b6421f45d470803b492f037078ff59c7576269
submission_blob: aa4d099296999ecc3e96238423bded7c90dd656e
lint_warnings: 52
attempt_file: 2026-10-25T035159Z-check-moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound.md
prepared_at: 2026-10-25T035159Z
job: 51d047cbc5ff450a
dispatched_at: 2026-10-25T035207Z
outcome: regression
verdict_detail: regression
speedup: 1.0637372150195272
speedup_ci: [1.0477811567942343, 1.0796932732448201]
harvested_at: 2026-10-25T043358Z
best_at_harvest: 1.2680313318618024
promotion: null
---

# moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound

## Change

Stage 3 of the expert-once swap-AB MoE chain (moe_route_list + moe_w13_silu_swapab + moe_w2_swapab replacing moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32). Stage 2 (moe-chain-cuda-swapab-expert-once-fix-inproc, check 4ebd055adab34ecd) built sm_90a-only, ran every canary and both full T=40 chains with NO 715 fault, then turned itself OFF on its own 1e-2 row bound against Marlin (y row rel 1.07e-2; act row rel 8.79e-3, out global rel 6.00e-3, finite, ids equal, repeat identical), so it measured only Marlin, on a stale base (e65eda8, no dense chain). REWRITE_READ_2026-10-25 '01:36Z harvest', LESSONS 2026-10-25. Changes, nothing else: (1) REBASE onto current decode master (stack-align-par-sidestream-sliding1, submission blob 4da94677). Because depends_on a rewrite stage makes claim start the worktree at stage 2's commit 1177e31 on the stale base, the implementer's FIRST step in the worktree is: git checkout master -- muse_decode_layer/submission.py, then git apply -3 of hypotheses/done/regression/moe-chain-cuda-swapab-expert-once-fix-inproc.patch, merging conflicts by hand so master's dense chain (dl_ kernels at T<=DL_MAX_TOKENS), parallel align and side-stream MoE dispatch all survive; the new path keeps its OWN gate constant (tokens > 32) and launches on the side stream master uses for MoE dispatch. Verify the rebase by diffing against master: only the MoE large-T path and its prepare() block may differ. (2) RECALIBRATE the self-check. Error source: the stage-2 check compared two approximate kernels against each other. Both dequantize the same quantized weights and accumulate in fp32, but they round differently: the new chain rounds the silu*mul activation to bf16 before w2 (act row rel 8.79e-3 already sits in the w2 input), folds the scale with __hmul2 in bf16, and reduces in a different order. Marlin carries its own error of the same order, so the y row difference to Marlin is roughly sqrt(e_new^2 + e_marlin^2) and can exceed 1e-2 when each kernel alone is under it. Replace the Marlin-relative gate with an fp32 reference: in prepare(), dequantize the same expert weights to fp32 once (prepare-time only, synthetic inputs, nothing cached for the timed call), compute w13 -> silu*mul -> w2 -> weighted sum in fp32 for T=40/128/256, and measure BOTH Marlin's and the new chain's error to it (global rel L2 and max row rel, near-zero rows measured against a tenth of the mean row as the judge does). Enable the path only if new global <= max(1.5 x Marlin global, 5e-3) and new max row <= max(1.5 x Marlin max row, 1e-2), AND under absolute caps of global <= 1.5e-2 and row <= 3e-2. The caps sit below the judge's tolerance: the judge allows 3-15% relative L2 on the layer outputs with any row up to 5x its bound (>= 15%), and the MoE output is only part of the residual update, so a MoE-output error under 1.5e-2 / 3e-2 stays well inside the tightest 3% layer bound. Also print the old Marlin-relative numbers for continuity. (3) Print the T=128/256 per-kernel CUDA-event timings (new chain vs Marlin) BEFORE the enable decision, whatever the decision is (prepare undercounts the judge profile by 20-25%: rank only). KEEP the full canary ladder unchanged: sm_90a image check (capability (9,0), -gencode compute_90a/sm_90a only, binaryVersion/ptxVersion 90 for each kernel, dyn-smem attribute print), the 1-CTA dequant8 + mma tile micro canary, per-kernel synthetic launches at the smallest T above the gate with unbuffered launching/ok prints and torch.cuda.synchronize() after each, then the T=40/128/256 self-check; keep plain cp.async.cg, __hmul2 and the sm_90a-only build from stage 2. Print [moe-cuda-large-t] ON|OFF with the fp32-ref errors of both kernels, the bounds used and PATH CHOSEN, compactly and late (log tail is ~15 KB). First measurement is an okc check, never a scored submit.

## Expected

- `*load=verify_16 *` faster: Marlin w13+w2+act_and_mul 254/271 us (sliding/full) vs MoE floor ~101 us; expert-once swap-AB target <= Marlin (M2), ~150 us (M3 ~1.5x floor)
- `*load=verify_32 *` faster: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
- `*load=decode_*` flat: T<=32 outside gate (dense chain / master): control; rebase removes stage 2's stale-base +4-15%
- `*load=verify_1 *` flat: T=8 outside gate: control

## Risk

The design is still untimed against Marlin; this stage's job is to get it enabled and timed. Risks: a 3-way rebase conflict with master's parallel align / side-stream dispatch; the 715 fix is unattributed (three changes at once) so a rebase must not reintroduce cache-hint cp.async or a multi-arch build; a looser bound must still be justified by the printed fp32-ref numbers (if the new chain's error exceeds 1.5x Marlin's against fp32, that is a real accuracy defect to fix, not a bound to raise). claim records base_sha as stage 2's commit; the harvester must compare against current master's BEST.

## Milestones

Rewrite stage 3 of `moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T034534Z by an adversarial subagent (model opus).

**Verdict: proceed.** The fp32-reference recalibration is the right fix, and the absolute caps sit below the judge's bound (global 2x margin, row 5x). However, the max(..., 5e-3)/max(..., 1e-2) floors could pass a chain with 2-3x Marlin's error. The stage-2 numbers are already larger than rounding alone explains, so make the 1.5x-Marlin ratio the binding gate. The rebase recipe is also wrong: the stage-2 patch is a delta over stage 1.

### Justification
Rewrite stage 3, judged on whether the path gets enabled and timed against Marlin at verify_16/32 (Marlin w13+act+w2 254/271 and 443/427 us vs floor ~101/109 us). The design still has no timing, so the milestone is readable only if the path turns ON. Bound soundness, checked against reference.py/checks.py. The judge compares the residual update (stream - a1*a2*stream_in) against FP32, with tol = clamp(2.5 x the e4m3-query error, 0.03, 0.15) and any row <= 5x tol. The update = a2*b1*rms(attn) + b2*rms(f), where f = unpooler(moe)*(1+post_ffn_norm). The unpooler is linear and the RMS only removes the radial error component. So the MoE relative error passes roughly 1:1 into the FFN term, and the 'MoE output is only part of the residual update' argument gives no guaranteed attenuation. The real margins are global cap 1.5e-2 vs the 3% floor (2x, before master's own error adds in quadrature) and row cap 3e-2 vs 15% (5x). That is adequate, not 'well inside'. The judge's tests on the okc check are the final arbiter (milestone 1). Weakest link: the evidence suggests a real accuracy gap, not just a mis-set bound. (i) Stage 2 failed TWO of its bounds, not one: out global rel 6.00e-3 > its own 5e-3 as well as y row 1.07e-2 > 1e-2 (log line 54-55; the hypothesis and harvest mention only the row). (ii) The stated error source does not hold up: Marlin's SGLang path also rounds the w13 output and silu*mul to bf16 before w2. The fused chain keeps h in fp32 and rounds act once, so it should be at least as accurate on act. Pure bf16 rounding differences predict ~2-3e-3, but stage 2 measured act 8.8e-3 (max row) and out 6.0e-3 (global), which is 2-4x more. That points at something like the __hmul2 scale fold or a global-scale rounding, and the fp32 reference will say which kernel carries it. (iii) Because of that, the floors are the loosening risk. If Marlin's fp32 error is ~2e-3 global / 4e-3 row, then max(1.5x, 5e-3) and max(1.5x, 1e-2) admit 2.5x Marlin's error, which is exactly the hidden-bug case. The hypothesis's own Risk line says '>1.5x Marlin is a real defect', so enforce that: enable on new <= 1.5 x Marlin (global and max row, both vs fp32), with floors no higher than the bf16-rounding scale (~3e-3 global, ~6e-3 row), and keep the absolute caps as a backstop. Print the ratio. Also gate on the FINAL MoE output (post_expert_norm weighted sum, the tensor the judge's unpooler sees), with y/act printed for diagnosis only. Synthetic randn x is fine with the judge's real weights, but prefer a pre_expert_norm-shaped x, because LESSONS says random self-checks can pass where coherent data fails. Rebase defect: the stage-2 patch's preimage is blob 81d11fe = e65eda8:submission.py (stage-1 impl), which already contains _LT_CPP and the kernels; master's submission.py has no _lt_ code. So 'git checkout master -- submission.py; git apply -3 <stage-2 patch>' brings over only the stage1->stage2 delta, and the stage-1 kernel source lands in conflict or is lost. Use a cumulative 3-way merge instead: base = a449561:muse_decode_layer/submission.py (the parent of stage-1 impl e65eda8), ours = master, theirs = 1177e31. That is `git merge-file`, or apply `git -C <wt> diff a449561 1177e31 -- muse_decode_layer/submission.py` with -3. Then diff against master as the hypothesis says. Memory: dequantize to fp32 one expert at a time (32 experts x w13+w2 in fp32 at once is large) and free it before timing.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1: master moved the T>32 expert_map gather + moe_align onto the side stream (_expert_dispatch) and overlaps it with dense work. The new chain replaces moe_align with moe_route_list. Whether it is still overlapped on the side stream changes the shape time apart from the GEMMs, so judge the replaced kernels' rows, not the shape.
- marlin-block-m-cap-16: the incumbent Marlin at T=128/256 runs with block_m capped at 16. The 254-443 us incumbent rows already reflect that, so compare against the current master profile rows, not older ones.
- post-expert-norm-triton-small-t: small-T only (T<=32), so it is outside this gate. The new chain's own post_expert_norm_weighted_sum must match SGLang's large-T combine numerics, and it shares the fp32-ref check.
- small-t-launch-minimal-dense-chain-pdl: T<=DL_MAX_TOKENS only. Its presence is what the rebase must preserve; it does not confound the T>32 rows.

### Interference and A/B needs
No companion. This is a rewrite stage whose result is read from the replaced kernels' own rows (moe_route_list + moe_w13_silu_swapab + moe_w2_swapab vs moe_align + 2x Marlin + act_and_mul) and the printed prepare timings. The decode_* and verify_1 shapes are in-run controls outside the gate. The next stage is the follow-up.

### Open questions
- If the fp32 reference shows the new chain at >1.5x Marlin, find the cause (scale fold in bf16 via __hmul2, global-scale rounding, act rounding) before raising any bound. If it shows Marlin at the larger error, the stage-2 OFF was a false negative.
- Is stage 2's original queued sibling 'moe-chain-cuda-swapab-expert-once-fix' (child-process design, queue/2026-10-25T001250Z) still queued? The rules reject process starts, so it should be retired, not run.


## Implementation notes

- Rebase (review wins over the spec's `git apply -3` recipe): cumulative `git merge-file` of base a449561, ours master, theirs 1177e31 on scratch copies. Its conflicts were two large pure insertions at one point plus small edits, so the resolution kept ALL of master (dense chain, parallel align, side-stream `_expert_dispatch`, sliding batch-1 path) and carried the chain's section (`# CUDA large-T MoE path` through `_lt_setup`, unchanged kernels and canary ladder) in before `prepare()`. Diff vs master: +1247/-9, only the module docstring, `import os`, that section, the `_lt_setup` call at the end of `prepare()`, `_experts` -> `_marlin_experts` (returns out, y, cache2, local_ids; `_experts` is a thin wrapper returning out, so the dense chain call is unchanged) and the `kernel_fn` large-T branch.
- Dispatch: `use_lt = tokens > 32 and state["moe_cuda_large_t"]` (host flag, shape only). On the side stream: shifted ids, fp32 weights and `moe_route_list`; on the main stream after `wait_stream`: `moe_w13_silu_swapab`, `moe_w2_swapab`, combine. `_lt_experts` was split into `_lt_route` + GEMMs for that. No align runs on the chain path.
- Gate (review): fp32 reference per T=40/128/256, dequantized ONE expert at a time and freed (`_lt_fp32_ref`); x is `pre_expert_norm(randn)` (unit-RMS fallback). Both Marlin and the new chain are measured against it (global rel L2 and max row rel, row reference floored at 0.1 x mean row norm) for act, y and out. Enable only if finite, ids equal, repeat identical and, on the FINAL out, new <= max(1.5 x Marlin, floor 3e-3 global / 6e-3 row) and under caps 1.5e-2 / 3e-2. act and y errors, the ratios, the old Marlin-relative numbers and a sanity check of the fp32 combine are printed only. Deviation from the spec: floors 3e-3/6e-3 (not 5e-3/1e-2), ratio binding. The old hot-expert check is subsumed. Reference freed and `empty_cache()` before timing.
- Timing now prints for T=128/256 before the decision whenever the canaries and the self-check ran, ON or OFF.
- Kept: sm_90a-only build, binaryVersion==90 gate, canary ladder, plain cp.async.cg, __hmul2, flushed announce + synchronize around each prepare launch, loud PATH CHOSEN line.
- Not verifiable here: no GPU. The reference's final norm (rsqrt(mean s^2 + eps) * post_expert_w) is my reading of `weight_first=True`; the printed combine sanity line shows if it is wrong (then the path stays OFF, safely).

## Rules check

- [x] CUDA-graph capturable: kernel_fn additions are launches, allocations by shape and stream waits only; syncs, .cpu(), the build, canaries and the reference are in prepare() (checked in the diff).
- [x] Nothing derived from `data` is cached: only weights-derived `lt_bank`, `lt_ext`, `lt_grid`, flag; ids and routing are per call.
- [x] Weights repacked losslessly in prepare() only (round-trip asserts kept); NVFP4 stays NVFP4.
- [x] Activations BF16, act rounded to bf16 as SGLang's Marlin path does; fp32 only in the prepare-time reference.
- [x] n/a: attention is untouched by this diff.
- [x] No timing patches, harness access, reference import, or child process (grep for subprocess/multiprocessing/fork/Popen: none).
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune added; no host sync in kernel_fn.

okc_lint: "OK with warnings" (warnings are only prepare()-side host-sync and uint8 weight/canary dtypes). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent. **Verdict: dispatch-with-fixes.**

Fixes (in the worktree submission.py):
- `_lt_fp32_ref` combine semantics were wrong. It took the weighted sum first, then one RMSNorm, then multiplied by (1 + w). reference.py `post_expert_norm` / `_experts` (and SGLang's `weight_first=True`) apply the norm per route: z = y * (1 + w), then z * rsqrt(mean z^2 + 1e-8), then the topk-weighted sum over the local routes. Against the old "reference", both kernels would show O(1) error, which fails the 1.5e-2 / 3e-2 caps on every T. The path would always be OFF, so the run would be wasted. Fixed to reference.py's exact order, with fp32 (1 + post_expert_norm) as the judge uses (not the bf16 effective weight), and updated the docstring. okc_lint OK (prepare-side warnings only), py_compile OK, no __pycache__ left.

Checked, no change needed:
- kernel_fn: the large-T branch is reachable only in the non-dense-chain branch with tokens > 32 and the prepare() flag set. Master's dl chain (T <= 32), parallel align, `_expert_dispatch` and side-stream overlap are unchanged on every shape the chain does not take. `_experts` is a pure wrapper.
- Graph capture: launches use getCurrentCUDAStream, there is no host sync, allocations depend on shape only, and record_stream is guarded by the capture state.
- Memory and time: the fp32 dequant runs one expert at a time (peak under 1 GB) and is freed, with empty_cache before timing. Stage 2 compiled in ~31 s, well inside the 300 s prepare budget.
- Gates: binaryVersion==90 and the arch probe, announced and synced canaries.
- Rules: no child process beyond load_inline's build, nothing cached from inputs.

Residual risk: the fp32 reference now matches the judge, but stage 2's ~6e-3 out gap may be real new-chain error. If so, the 1.5x-Marlin ratio keeps the path OFF (a neutral run).

## Result

- job `51d047cbc5ff450a` (check), status done, passed True, tests 13/13
- speedup 1.0637x, 95% CI [1.0478, 1.0797], score 21264.932429373315 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/regression/moe-chain-cuda-swapab-expert-once-fix-inproc.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound.result.json`, `muse_decode_layer/hypotheses/done/regression/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound.patch`; branch `hyp/muse_decode_layer/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 144.87 | 127.64 | 1.1407 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 154.93 | 142.74 | 1.0889 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 232.64 | 209.4 | 1.0853 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 371.92 | 352.9 | 1.0497 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 261.06 | 241.62 | 1.0775 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 546.21 | 530.0 | 1.0419 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 202.38 | 173.54 | 1.1003 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 211.61 | 194.34 | 1.0911 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 446.97 | 743.63 | 0.5986 | lost |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 589.42 | 893.0 | 0.6603 | lost |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 621.68 | 935.94 | 0.6396 | lost |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 899.65 | 1236.5 | 0.7259 | lost |

gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, decode_32, verify_1; lost on sliding verify_16, verify_32; full verify_16, verify_32. Expected hit: 0/4. Noise ±1.5%.

## Follow-ups

- rewrite stage 3 of `moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_decode_layer "<what to fix or tune>" --kind rewrite --depends-on moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.

