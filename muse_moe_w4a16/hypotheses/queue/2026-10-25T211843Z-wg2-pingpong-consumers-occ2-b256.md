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
review: proceed
status: hold
review_at: 2026-10-25T212523Z
review_companions: null
hold_reason: "killed line (X5 2026-10-27): wg2 re-split measured x0.405/0.882 (SINGLE-PASS s1 killed table), ping-pong consumer WGs x1.18-1.53 slower (prefill rega-pp, campaign K5 prior); the T<=8 incumbent is Engine A, not the wg consumer; adversary of cuda-fused-split-i-moe-tiny-t asked to close remaining wg-consumer items"
---

# wg2-pingpong-consumers-occ2-b256

## Change

Consumer schedule only, inside wg2_kernel<G, 64, 3, 2, ORD> as restacked by restack-occ2-wg2-b256-two-stream-halves-on-master (B256 only; every other cell stays that base). Today each consumer warpgroup runs mbar_wait(full) -> dq_stage (dequant8 x 32 + scale expand, pure ALU) -> 8 register-A wgmma -> commit -> wgmma_wait<0> -> arrive(empty) in lockstep with its sibling on the same stage, so within a CTA the ALU idles during the wgmma chain and the tensor core idles during both dequants; at OCC2 the other CTA on the SM only partly fills those gaps (wg2 measured 0.955 vs single CTA). CHANGE: apply the queued cuda-nvfp4-wgmma-rega-pingpong-consumers schedule to wg2's consume loop unchanged in substance: two named barriers PP_A = 2 and PP_B = 3 (count 256; bar 1 stays the epilogue barrier), WG1: [bar.sync PP_A except its first k-stage] dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B; wait<0>; arrive(empty). WG2: bar.sync PP_B; dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A; wait<0>; arrive(empty); identical item and KS sequences per warpgroup, every bar op executed on the dead path, one final balancing sync. Register-A fragments are never rewritten with a group in flight (variant A semantics kept). Build it as a template flag PP on wg2_kernel so the prepare gate can time a 2x2 of {OCC1 s5, OCC2 wg2} x {PP off, on} on the full _moe_wgmma path, cold-L2 graphs, _wg_judge_ids and zipf routings, and print per kernel (w13, w2) and per N bucket (16/32/64) kernel-only DRAM TB/s plus the L2->SM ingest rate. Live only if OCC2+PP beats the base's OCC2 by >= 1.02x on the worse routing; canary asserts max |PP - noPP| == 0 (only issue order changes). First measurement is an okc check (named-barrier miscount hangs).

## Expected

- `bs=256 *` faster: base wg2 rows w13 225.3 + w2 141.5 us at ~2 TB/s; diagnosis sizes a consumer fix at 2.8 TB/s as ~-120 us (+6-7% headline). Realistic at OCC2, where TLP already hides part of the gap: -5 to -10% of the 367 us = -18 to -37 us, +0.8-1.7% headline; below the 1.24% margin at the low end, so it may only land stacked with wg2-w13-occ2-nmax128-stg2-hot
- `bs=128 *` flat: B128 stays Marlin; the print-only B128 V64@OCC2+PP row tells whether the 1.067x gap to Marlin (268.5 vs 251.5 us) closes

## Risk

Medium risk: named barriers have no timeout; hand-simulate KS=16 (w2) and 32 (w13) and items with nvalid < NB. If flat, ALU issue (not serialization) is the limit and producer-scale-expand on wg2 is next. Do not run the master-based cuda-nvfp4-wgmma-rega-pingpong-consumers as well once this is claimed; it measures a kernel that is no longer on the board path.

## Adversarial review

Reviewed 2026-10-25T212523Z by an adversarial subagent (model opus).

**Verdict: proceed.** Go ahead, but only on top of restack-occ2-wg2-b256-two-stream-halves-on-master: wg2_kernel does not exist on master's submission.py, so 'base: master' cannot be built as written. The in-process 2x2 {OCC1,OCC2}x{PP off,on} already acts as the control, so no companion runs are needed. Two things to tighten: raise the 1.02x single-round gate to today's two-round >=1.03 rule, and spell out where the named barriers sit relative to item boundaries.

### Justification
Mechanism: the board's consume_item2 (branch combo-wg-occ2-v64-hot-interleave-align2, wg2_kernel<G,64,3,2,ORD>) runs exactly the chain the hypothesis describes: mbar_wait(full), dq_stage (dequant8 x 8 per warp plus e4m3 scale expand), 8 register-A wgmma, commit, wgmma_wait<0>, then lane0 arrive(empty). The two consumer warpgroups run that chain in step on the same stage. So within one warpgroup the dequant and the tensor-core work cannot overlap. The profile supports a consumer limit rather than a DRAM limit. B256 wg2 rows are w13 225.3 + w2 141.5 = 366.8 us at about 2 TB/s, against a 4.8 TB/s floor (verify_32 358.1 vs SoL 108.9 us). b128-n32-deep-ring refuted bytes in flight as the limiter (1.83 to 1.87 TB/s), and bf16_proj streams similar weights at about 3.5 TB/s (LESSONS 73, 179). The expected directions fit TASK.md: only verify_32 runs wg2, and B128 stays Marlin, so 'flat' is right. The B128 row is print-only and correctly not counted. Prior check: the cited prior (the queued master-based cuda-nvfp4-wgmma-rega-pingpong-consumers) is genuinely different, because it targets the single-CTA s5 kernel and this targets the board's OCC2 wg2. The prefill ping-pong loss (LESSONS 182: 1.18-1.47x slower, which cut L2->SM ingest from 6 to 4.76 TB/s) happened in an activation-ingest-bound regime. Moe B256 streams weights from DRAM with small N, so the regimes plausibly differ, and the hypothesis prints the ingest rate to settle it. It is not on any do-not-try list. WEAKEST LINK: OCC2 bought only 0.955. That already puts 4 independent consumer chains on each SM (2 CTAs x 2 warpgroups), and they are free to drift out of phase. Ping-pong adds no ALU or tensor-core throughput; it only forces an order onto chains that TLP can already interleave. Total dequant ALU work and wgmma work per SMSP are unchanged, and a warpgroup's wgmma spans all 4 SMSPs. If the 4.5% from OCC2 means latency hiding is nearly saturated and the real limit is dequant issue rate, ping-pong will be flat. Worse, it could be slightly negative: WG2 now trails WG1 by about one dequant on each stage, the stage's empty arrive needs both warpgroups, so release is delayed and the STG=3 ring effectively shrinks. The -5 to -10% kernel estimate (+0.8-1.7% headline) leans on DIAGNOSIS's 2.8 TB/s sizing, which is a target, not a measurement. The low end is below the 1.24% margin, as the file admits. The kernel-level 2x2 print can still read a flat or small result, so a flat outcome stays interpretable. That is acceptable.

### Confounders on the current kernel
- restack-occ2-wg2-b256-two-stream-halves-on-master (queued p1, not landed): it is the real base, and wg2_kernel, _c2_select and _c2_checks are absent from master. If the restack's selector sends B256 to OCC1 s5 on that base, then 'beats the base's OCC2 by >=1.02x' compares against a path that is not live. The gate must compare OCC2+PP against the base's CHOSEN path, min(OCC1, OCC2).
- combo-wg-occ2-v64-hot-interleave-align2 (board, carried in by the restack): the ORD item order and the NMAX 64 chunking of hot experts set item lengths (KS) and how often items change over. Ping-pong's per-stage benefit and its per-item barrier hand-offs both depend on these, so a gain or loss here may not carry over to NMAX128/STG2 (wg2-w13-occ2-nmax128-stg2-hot).
- wg-pdl-gather-w13-w2-chain (landed): wg2 calls griddep_wait/griddep_launch, so w2's prologue overlaps the w13 tail. Ping-pong changes the tail shape (WG2 finishes about half a step later), so headline and kernel-only deltas can differ. Read the kernel-only TB/s per kernel, not just the verify_32 row.
- align-two-phase-port-from17 / align-parallel-per-expert-verify (landed): they set sorted_ids/expert block layout and so nvalid < NB items, which only matters for barrier counting on partial items. No timing confound beyond about 1 us.

### Interference and A/B needs
No companion is warranted. The hypothesis already builds PP as a template flag and times {OCC1 s5, OCC2 wg2} x {PP off, PP on} in one prepare process, on the full _moe_wgmma path, with cold-L2 graphs on two routings, and prints kernel-only TB/s per kernel and per N bucket. That in-process A/B separates the ping-pong effect from every landed piece, from the restack and from row noise better than a second okc run would. An ab-revert of wg-pdl-gather would only re-measure tail overlap that the per-kernel print already exposes. A separate run of the master-based pingpong (OCC1) is exactly the 2x2's OCC1 column, so it would be a duplicate; the file rightly says to retire it once this is claimed. If the headline lands within noise, the GATE and PATH CHOSEN lines decide what happened, as LESSONS 182 requires.

### Open questions
- Build base: wg2_kernel is not on master. Must the builder wait for restack-occ2-wg2-b256-two-stream-halves-on-master to land, or branch from its hyp/ branch? Builds run tonight against current master, so this cannot be built as 'base: master'.
- Barrier placement across item boundaries: does 'WG1 skips bar.sync PP_A on its first k-stage' mean the first k-stage of the CTA's whole loop, or of each item? Per item would leave one unmatched PP_A arrive per item, so phases drift or hang. Per loop is the only consistent reading with 'one final balancing sync'. The hand-simulation should also cover the epilogue named_bar(1,256) sitting between WG2's last bar.arrive PP_A and WG1's next bar.sync PP_A, plus a CTA whose item count is zero.
- Gate threshold: today's lesson says a single-round 1.02x prepare selector mis-routed bf16 cells. Should the PP gate require two interleaved rounds >= 1.03 with an IQR bound, compared against the base's chosen path rather than OCC2 alone?
- Stage release: with WG2 half a step behind, empty(s) completes later. Should the diag also print producer stall time, or full-wait time per warpgroup, at STG 3? That would show whether ring shrinkage offsets the overlap. If it does, PP plus STG 4 at NMAX 64 might fit under the OCC2 smem budget.
- The OCC1 s5 + PP cell in the 2x2 means porting PP into master's s5 module as well, which adds a second build surface and its own hang risk. If build time is tight, can that cell be dropped, keeping {OCC2} x {PP off, on} plus OCC1 noPP?

