---
problem: muse_moe_w4a16
slug: wg2-pingpong-consumers-occ2-b256
created: 2026-10-25T211843Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Consumer schedule only, inside wg2_kernel<G, 64, 3, 2, ORD> as restacked by
  restack-occ2-wg2-b256-two-stream-halves-on-master (B256 only; every other cell stays that base).
  Today each consumer warpgroup runs mbar_wait(full) -> dq_stage (dequant8 x 32 + scale expand,
  pure ALU) -> 8 register-A wgmma -> commit -> wgmma_wait<0> -> arrive(empty) in lockstep with its
  sibling on the same stage, so within a CTA the ALU idles during the wgmma chain and the tensor
  core idles during both dequants; at OCC2 the other CTA on the SM only partly fills those gaps
  (wg2 measured 0.955 vs single CTA). CHANGE: apply the queued
  cuda-nvfp4-wgmma-rega-pingpong-consumers schedule to wg2's consume loop unchanged in substance:
  two named barriers PP_A = 2 and PP_B = 3 (count 256; bar 1 stays the epilogue barrier), WG1:
  [bar.sync PP_A except its first k-stage] dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B;
  wait<0>; arrive(empty). WG2: bar.sync PP_B; dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A;
  wait<0>; arrive(empty); identical item and KS sequences per warpgroup, every bar op executed on
  the dead path, one final balancing sync. Register-A fragments are never rewritten with a group
  in flight (variant A semantics kept). Build it as a template flag PP on wg2_kernel so the
  prepare gate can time a 2x2 of {OCC1 s5, OCC2 wg2} x {PP off, on} on the full _moe_wgmma path,
  cold-L2 graphs, _wg_judge_ids and zipf routings, and print per kernel (w13, w2) and per N bucket
  (16/32/64) kernel-only DRAM TB/s plus the L2->SM ingest rate. Live only if OCC2+PP beats the
  base's OCC2 by >= 1.02x on the worse routing; canary asserts max |PP - noPP| == 0 (only issue
  order changes). First measurement is an okc check (named-barrier miscount hangs).
expected:
  - shapes: bs=256 *
    direction: faster
    why: "base wg2 rows w13 225.3 + w2 141.5 us at ~2 TB/s; diagnosis sizes a consumer fix at 2.8 TB/s as ~-120 us (+6-7% headline). Realistic at OCC2, where TLP already hides part of the gap: -5 to -10% of the 367 us = -18 to -37 us, +0.8-1.7% headline; below the 1.24% margin at the low end, so it may only land stacked with wg2-w13-occ2-nmax128-stg2-hot"
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin; the print-only B128 V64@OCC2+PP row tells whether the 1.067x gap to Marlin (268.5 vs 251.5 us) closes
risk: medium
depends_on: restack-occ2-wg2-b256-two-stream-halves-on-master
priority: 2
kind: primary
controls_for: null
prior: "cuda-nvfp4-wgmma-rega-pingpong-consumers: same schedule, but that queued file builds on master's single-CTA s5 wg_kernel; this one rebases it onto wg2 at OCC2 (the board code) as DIAGNOSIS action 3 asks, and adds the 2x2 OCC x PP diag to separate TLP from intra-CTA overlap. Prefill's experts-rega-pingpong-consumers-all-t lost 1.18-1.47x because there the limiter was activation ingest at 5.8-6 TB/s L2->SM; moe B256 is weight-DRAM bound at ~2 TB/s with small N, so the ingest rate must be printed to show which regime applies"
review: required
status: queued
---

# wg2-pingpong-consumers-occ2-b256

## Change

Consumer schedule only, inside wg2_kernel<G, 64, 3, 2, ORD> as restacked by restack-occ2-wg2-b256-two-stream-halves-on-master (B256 only; every other cell stays that base). Today each consumer warpgroup runs mbar_wait(full) -> dq_stage (dequant8 x 32 + scale expand, pure ALU) -> 8 register-A wgmma -> commit -> wgmma_wait<0> -> arrive(empty) in lockstep with its sibling on the same stage, so within a CTA the ALU idles during the wgmma chain and the tensor core idles during both dequants; at OCC2 the other CTA on the SM only partly fills those gaps (wg2 measured 0.955 vs single CTA). CHANGE: apply the queued cuda-nvfp4-wgmma-rega-pingpong-consumers schedule to wg2's consume loop unchanged in substance: two named barriers PP_A = 2 and PP_B = 3 (count 256; bar 1 stays the epilogue barrier), WG1: [bar.sync PP_A except its first k-stage] dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B; wait<0>; arrive(empty). WG2: bar.sync PP_B; dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A; wait<0>; arrive(empty); identical item and KS sequences per warpgroup, every bar op executed on the dead path, one final balancing sync. Register-A fragments are never rewritten with a group in flight (variant A semantics kept). Build it as a template flag PP on wg2_kernel so the prepare gate can time a 2x2 of {OCC1 s5, OCC2 wg2} x {PP off, on} on the full _moe_wgmma path, cold-L2 graphs, _wg_judge_ids and zipf routings, and print per kernel (w13, w2) and per N bucket (16/32/64) kernel-only DRAM TB/s plus the L2->SM ingest rate. Live only if OCC2+PP beats the base's OCC2 by >= 1.02x on the worse routing; canary asserts max |PP - noPP| == 0 (only issue order changes). First measurement is an okc check (named-barrier miscount hangs).

## Expected

- `bs=256 *` faster: base wg2 rows w13 225.3 + w2 141.5 us at ~2 TB/s; diagnosis sizes a consumer fix at 2.8 TB/s as ~-120 us (+6-7% headline). Realistic at OCC2, where TLP already hides part of the gap: -5 to -10% of the 367 us = -18 to -37 us, +0.8-1.7% headline; below the 1.24% margin at the low end, so it may only land stacked with wg2-w13-occ2-nmax128-stg2-hot
- `bs=128 *` flat: B128 stays Marlin; the print-only B128 V64@OCC2+PP row tells whether the 1.067x gap to Marlin (268.5 vs 251.5 us) closes

## Risk

Medium risk: named barriers have no timeout; hand-simulate KS=16 (w2) and 32 (w13) and items with nvalid < NB. If flat, ALU issue (not serialization) is the limit and producer-scale-expand on wg2 is next. Do not run the master-based cuda-nvfp4-wgmma-rega-pingpong-consumers as well once this is claimed; it measures a kernel that is no longer on the board path.
