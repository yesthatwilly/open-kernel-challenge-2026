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
review: required
status: queued
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
