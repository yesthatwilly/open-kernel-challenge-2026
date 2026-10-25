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
review: required
status: queued
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
