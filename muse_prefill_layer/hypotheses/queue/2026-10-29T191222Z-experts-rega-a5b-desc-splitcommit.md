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
review: proceed
status: queued
review_at: 2026-10-29T191802Z
review_companions: null
review_best: 1.536215653197081
review_blob: d8ce4cbf6e2e54cb8294363e025022d9284a0762
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

## Adversarial review

Reviewed 2026-10-29T191802Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion runs: the A4b codes and A5b scales are exact by construction (proved for every word and every in-range byte, not sampled), the change is a lossless prepare-time repack of weights, and the in-prepare A5b-vs-A3 race makes one check readable. One trap must be fixed in the build: a fifth DQ arm raises REGA_DQ_INST from 16 to 20, so the 233 s build gate becomes 285 s > 270 s, the dq phase turns OFF, and A3 is lost too. Drop A1/A2 (or keep the instantiation count at <= 16) so the gate still passes.

### Justification
A (mechanism): plausible and measured on the same SM. rgx2 on the H100 (research/h100-rega2-1029.md) shows A5b+D+SC at 14.5 dequant SASS per k16 (A3 18) and 198 instructions per stage at N32 (A3 238). ncu on low w13 shows -8.4% cycles/SM, -17% issued and uniform instructions 23.4M -> 11.9M. Pair vs A3: low L2 0.923-0.925, HBM 0.954-0.963, high 0.933-0.948, ft 0.936-0.945, from two seeds. These per-SM [cyc] readings are the kind that transferred for A3: node low 0.857-0.888 vs judge in-prepare 0.851, high 0.908-0.923 vs 0.903. The parent's judge RATIOS show the judge's low numbers sat at or below the node's L2 bank, so the low class should land near the L2 figure, not the HBM one. EXACTNESS, proved rather than sampled (my script, scratchpad proof_a5b.py, transcribed from rgx2 dq4_check.py/gen2.py): (1) ENC4 is a bijection on the 32 bit positions (sorted(tab) == range(32)), so it is lossless. Master's dequant8 and dq4(ENC(q)) use only shifts, rotates, AND with constants and OR of disjoint masks (0x81808180 & 0x00400040 == 0; 0x80008000 & 0x01C001C0 == 0). Both maps are therefore OR-homomorphic with f(0) = 0, so equality on the 32 unit vectors, which I checked, proves equality for all 2^32 words. (2) Scale: for every b in [0x08, 0x7E], A3's per-half value is b<<4 + 0x7780, which equals ((b-8)<<4) + 0x7800. Since (b-8)<<4 <= 0x760 occupies bits 4-10 and 0x7800 bits 11-14, that + is an OR, so the A5b form ((b-8)<<4) | 0x7800 is identical. No half carries (max 0x7F60). The four 7-bit fields (bits 4-10, 20-26, 28-31+0-2, 12-18) are disjoint. rotl(w, 8) maps the k-odd fields onto the k-even mask, and its byte order (bytes 2/3 = rows g/g+8 at KK 1) matches dq_scale_pair<KK>. I checked all 119x119 byte pairs at both parities. The real judge bank already passes the precondition (27f0b9af log: SCALES q13/q2 0 of 67,108,864 / 33,554,432 outside [0x08, 0x7E]). The multiply (fma.rn.bf16x2 with -0) sees bit-identical operands, so its output is identical. (3) D and SC leave the k order and the accumulator chain unchanged: the same wgmma sequence r0k0..r1k3, split into two commit groups of the same shape, which PTX orders by default. Every stage buffer and N bucket is hit in a full class run, so torch.equal over all classes enumerates the complete, data-independent set of descriptor values. THE WEAKEST LINK is not exactness. It is the port plus the build gate. rgx2's source derives from rgx_dev.cuh, generated from c63801e7 plus the rgx arms, not from master's d8ce4cbf _rega_dq_src text, so this is a hand port into the dq module's asserted-edit chain. More concretely: _rega_dq_start sets est = REGA_DQ_EST_4INST_S x REGA_DQ_INST / 4 = 40 x 16 / 4 = 160, giving need = 0.1 + 1.3 x 160 + 25 = 233 s (the '233 s gate' the hypothesis cites). Adding the A5b template value with A1/A2 kept gives 5 x 2 x 2 = 20 instantiations, est 200, need 285 > REGA_DQ_BUILD_LIMIT_S 270. That prints '[rega-dq] OFF: budget', the whole dq phase is skipped, and A3 is not routed, so the run regresses to about the 1.50 lineage. Fix: retire A1 and A2. They are dead weight: A3 routed every class, A1 missed the gate at high and ft, and A2 lost to A3 everywhere. That leaves DQ {0, 3, A5b+D+SC} = 12 instantiations (need 181 s), or 16 if the cheaper fallback A4b+D+SC64 is added as a fourth arm (need 233 s, the same as today). Do not tune REGA_DQ_EST_4INST_S down instead: the build's real time (50.1 s for 16) is variable and the 300 s prepare limit is hard. Second-weakest link: ptxas on the judge toolchain could serialize the SC window (C7510/C7512) where cu130 on the node did not. The attrs check catches spills but not serialization, so log ptxas warnings for the new instantiations (the kp/anti-phase -Werror probe form) to tell 'SC serialized' apart from 'SC did not transfer'. EXPECTED: directions and shapes agree with TASK.md. All three rega classes feed turn_1 (T 753), turn_4 (3012) and first_turn (16384). Magnitude: the A5b cut vs A3 is roughly 45% (low), 60% (high) and 75% (ft) of A3's own cut vs master (14.9 / 9.7 / 7.8%), on a smaller post-A3 expert share. Scaled from A3's +3.3-3.65% check gain (+2.36% scored), that gives about +1.5-2.2% check-on-check and +1.0-1.5% scored. value_gain 2.0 is the optimistic end; it does not change the go, since the item is user-requested and H100-derived per today's steer. PRIOR: correctly named (experts-rega-dq2-dequant-cost-diag, landed, CLAIMS:75). The difference is real: a new code encoding, a new scale encoding, a descriptor hoist and a split commit, none of which A3 had. It is not on Do-not-try. The register ring (CLAIMS:68, refuted, re-refuted by rgx2 at x1.04-1.12) is correctly excluded.

### Confounders on the current kernel
- experts-rega-dq2-dequant-cost-diag (A3, CLAIMS:75, the parent): this stacks directly on it and replaces it per class. It is not a confounder for the in-prepare reading, because A3 is an arm in the same interleaved race, so the A5b/A3 per-round ratio isolates the change. For the headline, compare with A3's check 835e1a44 (1.5556), not the scored 27f0b9af (1.5362): check-vs-scored differed by 1.3% on identical bytes.
- experts-rega-w13-fast-silu-epilogue (CLAIMS:51) and the w2 early-loads arm a / w2st (CLAIMS:54, 56): the dq module derives from the fsilu_w2a source master dispatches, and rgx2's tile included both (gen chain c63801e7 + fsilu_w2a), so the node evidence already carries them. A5b touches only the k loop, not the epilogues. No interaction to revert.
- combo-b rega-lo / two-CTA low re-gate (CLAIMS:46, 56): the LO phase is skipped and rega_low_variant is 0 on the judge, so the low class runs master's rega_kernel as in rgx2. No confound, provided the dq phase's 'low only when rega_low_variant == 0' rule stays.
- fa3-full-contiguous-kv-triton-gather-gated / fa3c pins (CLAIMS:43, 56): different kernels. They change the headline's denominator per shape, not the rega rows, so they do not affect attribution.

### Interference and A/B needs
No companion run. (1) The decisive comparison already runs inside the same process: A5b+D+SC vs A3 vs DQ0, interleaved per round, warm and cold L2, pop and random routing, plus a replay pass. The [rega-dq] PAIR and ROUTE lines give the per-class A5b/A3 ratio on the judge's own SM and HBM, which is what a judge-side ab-revert would measure, at no cost in checks. (2) Sub-change attribution (A4b vs A5b vs D vs SC) was isolated on the H100 with [cyc]-type evidence that transferred for A3. A judge control for a ~1-3% sub-part would sit inside the >= +-1.5% CI and could not be read. (3) If a fallback is wanted, race A4b+D+SC64 as a fourth in-prepare arm (it keeps A3's scale bytes), with A1/A2 dropped to keep the count at 16. It is not a separate run. Is one check readable at ~+2%? On the headline only marginally: the CI is at least +-1.5%, same-code checks spread about 0.5%, and check vs scored about 1.3%. On the mechanism, yes: ROUTE per class, PAIR A5b/A3 per round, and the profile's per-shape rega w13/w2 microseconds against 835e1a44's profile. Read it as: routed classes with PAIR <= 0.95 and the rega kernel time down by a matching amount in the profile is a working mechanism, whatever the headline does. A check at or above about 1.575 (+1.3% over 835e1a44) with all three classes routed justifies a scored submit. A ROUTE that keeps A3 in a class is a clean negative for that class.

### Open questions
- Which arms does the module keep? 'A new arm next to A3 and DQ0' reads as if A1/A2 stay, and with them it fails the build gate (20 instantiations, need 285 s > 270 s). Confirm A1/A2 are retired, or an extra A4b+D+SC64 arm replaces them (16 instantiations, the same 233 s).
- Should the route gate be A5b/A3 <= 0.97 every round, as written, or the existing 'lowest passing arm vs master and DQ0'? The existing selection would pick A5b at A5b/A3 = 0.99, so the 0.97-vs-A3 criterion must be coded explicitly. At high, the node round spread is 2-4% around 0.933-0.948. The judge's in-prepare spread for A3 at high was about 0.5%, so the gate should pass, but expect high to be the class most likely to stay on A3.
- Is D's per-stage descriptor (lo = ((slab & 0x3FFFF) >> 4) | 1<<16, plus r*(SLAB_HALF>>4) + 2k, high word 0x40000040) byte-identical to master's make_desc on the module being ported? It was verified only against rgx_dev.cuh. A wrong high word fails torch.equal (safe), but it is cheaper to static-assert or check it in the canary.
- Does the judge's ptxas keep the SC window unserialized? Capture C7510/C7512 warnings for the new instantiations in the summary, so a non-transfer can be attributed.
- Is ft readable? H100 ft medians carried a +-10% round spread for every arm, while the judge's in-prepare ft A3 ratios were stable (0.915-0.923). If ft misses the gate, the headline gain is about two-thirds of the estimate (ft weight 138+48 on first_turn).

