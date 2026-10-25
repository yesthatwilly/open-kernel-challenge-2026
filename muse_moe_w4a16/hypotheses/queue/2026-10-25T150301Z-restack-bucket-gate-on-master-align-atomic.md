---
problem: muse_moe_w4a16
slug: restack-bucket-gate-on-master-align-atomic
created: 2026-10-25T150301Z
base: master
area: expert GEMM path selection (wgmma vs Marlin) restack
change: >
  Restack only, no new idea. Bring moe master's submission.py up to the code that is on the board:
  cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, impl fc959d9c; check
  7be125e9e1ea42e3 1.3744x, scored 4e4a0e26c85944f2 1.3727x) was built on base 928151c and
  predates master's two promotions align-parallel-per-expert-verify (1a5da3d2) and
  w2-marlin-atomic-add-small-m (9080a71e). Claim restacks it onto current master by 3-way applying
  the parent's patch; if that conflicts, hand-merge so the result keeps BOTH: (1) master's
  parallel per-expert align for verify shapes and master's w2 Marlin atomic-add flag at small M,
  exactly as on master; (2) the bucket-gate's whole wgmma path (per-call cuTensorMapEncodeTiled,
  three-state copy check, prepare-time selector _wg_select with state flags wg_on_128 / wg_on_256,
  kernel_fn branch on T and those flags only). Where the two touch the same function (the Marlin
  path the selector times as _moe_marlin), the selector must time master's current Marlin path,
  including the atomic-add flag and the new align, so it compares against what master actually
  runs. No other change. Keep the first-line marker. Print the PATH CHOSEN B128 / B256 lines as
  the parent did.
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "B256 wgmma path (judge data: 402 vs 457 us Marlin) on top of master's align/atomic-add"
  - shapes: load=decode_*
    direction: flat
    why: "T<128 stays on master's Marlin path"
risk: low
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-bucket-gate-on-master-align-atomic

## Change

Restack only, no new idea. Bring moe master's submission.py up to the code that is on the board: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, impl fc959d9c; check 7be125e9e1ea42e3 1.3744x, scored 4e4a0e26c85944f2 1.3727x) was built on base 928151c and predates master's two promotions align-parallel-per-expert-verify (1a5da3d2) and w2-marlin-atomic-add-small-m (9080a71e). Claim restacks it onto current master by 3-way applying the parent's patch; if that conflicts, hand-merge so the result keeps BOTH: (1) master's parallel per-expert align for verify shapes and master's w2 Marlin atomic-add flag at small M, exactly as on master; (2) the bucket-gate's whole wgmma path (per-call cuTensorMapEncodeTiled, three-state copy check, prepare-time selector _wg_select with state flags wg_on_128 / wg_on_256, kernel_fn branch on T and those flags only). Where the two touch the same function (the Marlin path the selector times as _moe_marlin), the selector must time master's current Marlin path, including the atomic-add flag and the new align, so it compares against what master actually runs. No other change. Keep the first-line marker. Print the PATH CHOSEN B128 / B256 lines as the parent did.

## Expected

- `load=verify_32*` faster: B256 wgmma path (judge data: 402 vs 457 us Marlin) on top of master's align/atomic-add
- `load=decode_*` flat: T<128 stays on master's Marlin path

## Risk

Goal: master == board code. The parent scored 1.3727 on a stale base; this should score at or above it, since master's own promotions were +0.03 over the parent's base.
