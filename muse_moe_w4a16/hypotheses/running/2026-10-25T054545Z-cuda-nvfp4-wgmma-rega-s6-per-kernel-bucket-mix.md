---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
created: 2026-10-25T054545Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 6, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit. Use that run's
  judged profile rows to enable each (T bucket, kernel) cell independently: cells are B128/B256 x
  {w13+silu, w2}. Four kernel_fn paths, chosen by T and two prepare booleans per bucket (shape +
  flag only): (a) wgmma w13+silu then wgmma w2; (b) wgmma w13+silu writes h in the route-row
  layout Marlin's w2 consumes as cache2 (sorted by route index, [T*8, I] bf16) then Marlin w2 with
  master's flags and block_size_m; (c) Marlin w13 + act_and_mul into cache2, then wgmma w2 fed
  Marlin's cache2 (s2-s4 already self-checked this exact pairing as 'w2 (fed Marlin cache2)', rel
  3.5e-3); (d) master. Both GEMMs must use one align: when a Marlin kernel runs in the cell, align
  at master's block_size_m and give the wgmma kernel that sorted_ids/block_experts (wgmma consumes
  m-blocks of 16; if master's block is 32 at B256, either reuse 16 with a second align only for
  wgmma or keep (a)/(d) for that bucket; state which in the log). Static table from the s5-forced
  profile: enable a cell where its wgmma kernel row was <= 0.95x the matching Marlin row (w13 vs
  Marlin call 1 + act_and_mul; w2 vs Marlin call 2) in the same profile; then confirm each enabled
  cell in prepare with the LESSONS-173 graph-replay selector at >=1.02x on the judge-like routing,
  else drop to (d). Self-check every enabled mix with the three-state refill-replay check against
  fp32 dequant. Hard rules: CUDA-graph capturable; kernel_fn does no host sync
  (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except
  and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in
  registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes.
  The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may
  hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: keeps only the wgmma kernel(s) that beat their Marlin counterpart in the s5-forced judged profile
  - shapes: bs=256 *
    direction: faster
    why: same, B256 cells
  - shapes: bs=16 *
    direction: flat
    why: "control: T<128 master"
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu and/or Marlin w2 at 128<=T<=512
review: proceed
status: running
review_at: 2026-10-25T054837Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T212857Z
base_sha: 57e77548ba31c2145d575df62752f62684b2f6ad
base_blob: 77f34503659c90a5953f91a93ba6e1a2b02e4613
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix
mode: null
restack_of: null
stale_base: true
---

# cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix

## Change

Stage 6, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit. Use that run's judged profile rows to enable each (T bucket, kernel) cell independently: cells are B128/B256 x {w13+silu, w2}. Four kernel_fn paths, chosen by T and two prepare booleans per bucket (shape + flag only): (a) wgmma w13+silu then wgmma w2; (b) wgmma w13+silu writes h in the route-row layout Marlin's w2 consumes as cache2 (sorted by route index, [T*8, I] bf16) then Marlin w2 with master's flags and block_size_m; (c) Marlin w13 + act_and_mul into cache2, then wgmma w2 fed Marlin's cache2 (s2-s4 already self-checked this exact pairing as 'w2 (fed Marlin cache2)', rel 3.5e-3); (d) master. Both GEMMs must use one align: when a Marlin kernel runs in the cell, align at master's block_size_m and give the wgmma kernel that sorted_ids/block_experts (wgmma consumes m-blocks of 16; if master's block is 32 at B256, either reuse 16 with a second align only for wgmma or keep (a)/(d) for that bucket; state which in the log). Static table from the s5-forced profile: enable a cell where its wgmma kernel row was <= 0.95x the matching Marlin row (w13 vs Marlin call 1 + act_and_mul; w2 vs Marlin call 2) in the same profile; then confirm each enabled cell in prepare with the LESSONS-173 graph-replay selector at >=1.02x on the judge-like routing, else drop to (d). Self-check every enabled mix with the three-state refill-replay check against fp32 dequant. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: keeps only the wgmma kernel(s) that beat their Marlin counterpart in the s5-forced judged profile
- `bs=256 *` faster: same, B256 cells
- `bs=16 *` flat: control: T<128 master

## Risk

Dispatch only after cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced is harvested with wgmma kernel names in the judged profile. If that run shows both kernels winning in both buckets, skip this (land the s5 gated twin instead); if both lose in both buckets, close the wgmma line. Overlaps the held cuda-nvfp4-wgmma-rega-s2-per-kernel-enable / w2-only-midm in intent; this one is grounded on judged rows and supersedes them.

## Milestones

Rewrite stage 6 of `Marlin w13+silu and/or Marlin w2 at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054837Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run this rewrite stage only after s5-copy-safe-forced has been harvested and its judged profile actually contains wgmma rows with a mixed per-cell outcome. Before the s5 data exists it has nothing to stand on, and the prepare-time evidence so far makes a mixed outcome unlikely: both kernels beat Marlin by about the same margin.

### Justification
The mechanism is plausible. In the judged profiles, Marlin's two calls take 306.2 us at bs=128 and 483.2 us at bs=256 (s4, job 4e7f8be895a145ba). The floors are 101.3 and 108.9 us, so there is 3-4.5x of headroom on 2 of the 6 equally weighted loads. A per-cell enable that keeps only the winning kernel is a sound way to stage the rewrite. The expected directions (bs=128/256 faster, bs=16 flat) fit TASK.md: T<128 runs master. The weakest link is that the s6 decision table has no data behind it yet. No judged profile has ever contained a wgmma row: s1 failed to build, s2 fell back, s3 failed, and in s4 the copy-unsafe gate silently ran Marlin. The only per-kernel evidence is s4's prepare timing. At M=128 cold that was w13+silu 385.6 vs Marlin w13+act 611.6 (0.63x) and w2 185.4 vs 310.0 (0.60x); at M=256 it was 457.2 vs 852.0 (0.54x) and 230.2 vs 425.9 (0.54x). Both kernels win by nearly the same ratio, so the most likely s5 outcome is all cells win (s6 is skipped in favour of the s5 gated twin) or all cells lose (the line closes). Only the mixed case gives s6 something to do. Second weakness: the prepare timings ran an all-local routing (1024 local routes at M=128). There Marlin reads ~920 us against 306 us judged, so the routing does not look like the judge's. The LESSONS-173 selector must use the judge-like skewed routing, where per-expert M is much smaller and the two kernels' ratio may differ. Third: path (b) needs a new wgmma epilogue that scatters h into Marlin's cache2 route-row layout, and that code has never run. Path (c), Marlin cache2 into wgmma w2, has been self-checked at rel 3.5e-3. Finally, a single profile's 0.95x threshold sits inside the about ±20% run-to-run noise of a Marlin row (REWRITE_READ). The static table should only nominate cells, and the selector's ≥1.02x graph-replay confirmation should decide.

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets master's block_size_m at T>=128 (16 at M=128, 32 at M=256 per the s4 log). It defines the incumbent Marlin cells that paths (b), (c) and (d) compare against, and it decides whether wgmma can share the Marlin align (16-row m-blocks) at B256. If it were reverted, the incumbent would be slower and the per-cell table would shift towards wgmma. That is not a confound of the s6 result, but the table is only valid for this block_size_m.
- align-parallel-per-expert-verify: master's T>32 align. Mixed paths (b) and (c) need it plus possibly a second align for wgmma at B256 (~4-7 us). That cost lands on the mixed cells and not on (a) or (d), so a mixed cell's gain must clear it.
- w2-marlin-atomic-add-small-m: T<=32 only, so it is disjoint from the 128<=T<=512 cells. No confound beyond the rebase that s4 already did.
- post-expert-norm-triton-small-m / SGLang fused_post_expert_norm at bs>=32: every path feeds the same combine. Path (a)'s wgmma w2 output layout ('fused norm skips remote rows') differs from Marlin's, so check that the combine input layout is identical across (a), (b), (c) and (d), or the combine time becomes a hidden per-path variable.

### Interference and A/B needs
No companion. This is kind: rewrite, judged per cell on the replaced kernels' own profile rows (wgmma w13+silu vs Marlin call 1 + act_and_mul; wgmma w2 vs Marlin call 2) within the same run. That attribution is internal, and no reverted or retuned landed hypothesis would sharpen it. The landed changes that touch this range (block_size_m pick, T>32 align) define the incumbent and are not alternative explanations for a per-kernel row delta. With 5 moe checks left and s5-forced still unrun, a companion would spend a check that s5 needs. The real control is the dispatch gate in the hypothesis's own Risk section: run s6 only if s5's judged rows split. If s5 shows all-win or all-lose, do not run s6.

### Open questions
- Did s5-forced log LAUNCHED = wgmma and produce judged wgmma kernel rows at bs=128 and 256? If not, s6 has no table and should not run.
- Did the per-cell judged ratios actually split across the 0.95x line? The prepare ratios (0.54-0.63 for both kernels) predict they will not. If they do not, this slug should close as superseded by the s5 gated twin or the line closure.
- At B256, does the wgmma kernel get its own 16-row align (an extra ~5-7 us launch) or reuse Marlin's 32-row align? The spec leaves this open. The cost of a second align must be inside the selector's timed graph so the ≥1.02x test sees it.
- Is the prepare selector's 'judge-like routing' the skewed, this-GPU-busiest distribution (per-expert M far below all-local)? s4's prepare timings used 1024 local routes at M=128 and overstated Marlin by about 3x against the judge.
- How are the bs=257 and skewed bs=128 tests bucketed (B256 cell vs B128 cell)? Each must hit an enabled mix that the three-state check has verified.

