---
problem: muse_prefill_layer
slug: experts-rega-a5b-desc-splitcommit
created: 2026-10-29T191222Z
base: master
mint_blob: "d8ce4cbf6e2e54cb8294363e025022d9284a0762"
best_at_mint: 1.5362
area: experts rega tile dq_stage / consumer stage loop (w13, w2)
change: >
  Stack the H100-validated A5b+D+SC form on master's landed A3 rega dequant
  (muse_prefill_layer/submission.py master 1b14174f, the rega dq module from
  experts-rega-dq2-dequant-cost-diag), porting the code from takemyjob hopper/rgx2 771f8eb
  (worktree /home/user/git/.takemyjob-wt/rgx2; findings
  campaigns/2026-10-26-expert-bandwidth/research/h100-rega2-1029.md): (1) A4b: a new lossless
  prepare-time re-encoding of the 4-bit code words so nibble->bf16x2 takes 9 ops/word instead of
  11; (2) A5b: scale bytes stored at prepare as packed 7-bit fields (b-8) so each k16 step's scale
  is one LOP3 (exactly A3's scale value; requires every scale byte in [0x08,0x7E], scanned in
  prepare, else keep A3); (3) D: the wgmma smem descriptor computed once per stage (halves
  uniform-pipe instructions); (4) SC: split commit, dequantizing record 1 while record 0's 4
  wgmmas run, no extra registers. Ship it as a new arm in the existing dq module (template DQ
  value) next to A3 and DQ0, raced per class with the same gate as A3 (<= 0.97 vs the dispatched
  A3 path AND vs DQ0 every round, both L2 modes, both routings, replay pass, bit-exact
  torch.equal); if it misses in a class, that class keeps A3. Re-encoded banks are a prepare-time
  lossless repack of weights only (never from a routing); free superseded copies only when no
  class routes to them. Keep the summary lines ([rega-dq] RATIOS/PAIR/ROUTE/VERDICT) at the end of
  prepare inside the 16 KB tail; build in the existing background thread with its 233 s gate.
expected:
  - shapes: load=turn_1*
    direction: faster
    why: "low class: node pair vs A3 0.923-0.925 (L2), 0.954-0.963 (HBM), random 0.916"
  - shapes: load=turn_4*
    direction: faster
    why: "high class: node pair vs A3 0.933-0.948"
  - shapes: load=first_turn*
    direction: faster
    why: "ft class: node pair vs A3 0.936-0.945"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-dq2-dequant-cost-diag: that landed A3 (26 -> 18 dequant SASS); this stacks a new code encoding, 7-bit packed scales, per-stage descriptors and split commit on it (18 -> 14.5, instr/stage 238 -> 198), node x0.92-0.95 vs A3"
target: check
value_gain: 2.0
value_p: 0.6
value_ev: 1.2
value_cost: 400k tokens, 1 judge run
evidence: >
  H100 gpunode06 rgx2 probe (takemyjob hopper/rgx2 771f8eb; research/h100-rega2-1029.md): A5b+D+SC
  pair vs A3 low L2 0.923-0.925 (vs master 0.79), low HBM 0.954-0.963, high 0.933-0.948, ft
  0.936-0.945, low random 0.916; bit-exact 3 classes x 2 seeds x both banks x pop/random; 168 regs
  0 spill; dequant SASS/k16 A3 18 -> A5b 14.5; A3's judge transfer: check 835e1a44 1.5556, scored
  27f0b9af 1.5362 (board)
user_requested: true
review: required
status: queued
---

# experts-rega-a5b-desc-splitcommit

## Change

Stack the H100-validated A5b+D+SC form on master's landed A3 rega dequant (muse_prefill_layer/submission.py master 1b14174f, the rega dq module from experts-rega-dq2-dequant-cost-diag), porting the code from takemyjob hopper/rgx2 771f8eb (worktree /home/user/git/.takemyjob-wt/rgx2; findings campaigns/2026-10-26-expert-bandwidth/research/h100-rega2-1029.md): (1) A4b: a new lossless prepare-time re-encoding of the 4-bit code words so nibble->bf16x2 takes 9 ops/word instead of 11; (2) A5b: scale bytes stored at prepare as packed 7-bit fields (b-8) so each k16 step's scale is one LOP3 (exactly A3's scale value; requires every scale byte in [0x08,0x7E], scanned in prepare, else keep A3); (3) D: the wgmma smem descriptor computed once per stage (halves uniform-pipe instructions); (4) SC: split commit, dequantizing record 1 while record 0's 4 wgmmas run, no extra registers. Ship it as a new arm in the existing dq module (template DQ value) next to A3 and DQ0, raced per class with the same gate as A3 (<= 0.97 vs the dispatched A3 path AND vs DQ0 every round, both L2 modes, both routings, replay pass, bit-exact torch.equal); if it misses in a class, that class keeps A3. Re-encoded banks are a prepare-time lossless repack of weights only (never from a routing); free superseded copies only when no class routes to them. Keep the summary lines ([rega-dq] RATIOS/PAIR/ROUTE/VERDICT) at the end of prepare inside the 16 KB tail; build in the existing background thread with its 233 s gate.

## Expected

- `load=turn_1*` faster: low class: node pair vs A3 0.923-0.925 (L2), 0.954-0.963 (HBM), random 0.916
- `load=turn_4*` faster: high class: node pair vs A3 0.933-0.948
- `load=first_turn*` faster: ft class: node pair vs A3 0.936-0.945

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 rgx2 probe (takemyjob hopper/rgx2 771f8eb; research/h100-rega2-1029.md): A5b+D+SC pair vs A3 low L2 0.923-0.925 (vs master 0.79), low HBM 0.954-0.963, high 0.933-0.948, ft 0.936-0.945, low random 0.916; bit-exact 3 classes x 2 seeds x both banks x pop/random; 168 regs 0 spill; dequant SASS/k16 A3 18 -> A5b 14.5; A3's judge transfer: check 835e1a44 1.5556, scored 27f0b9af 1.5362 (board)
- **Gain if it works:** 2.0% headline; **p** 0.6; **EV** 1.2% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 2.0% < 2.5%; EV 1.2% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge run.

## Risk

Prefill's A3 landing transferred node->judge within 0.01-0.02 (node low 0.84-0.89 vs judge 0.854; high 0.91-0.92 vs 0.909). By the same scaling (A3 pair cuts 14.6/9.1/7.8% gave check +3.65%, scored +2.36%), a further ~6% pair cut per class is ~+1.5-2.4% headline. Low-class HBM-bank ratio is only 4-5% under A3; the low claim rests on the L2 bank (H200 L2/HBM differ). Isolated parts lose (A3+SC alone 1.02-1.06 at low): SC must ride on A5b. Register ring/deferred wait stays refuted (x1.04-1.12).
