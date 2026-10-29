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
status: done
review_at: 2026-10-29T191802Z
review_companions: null
review_best: 1.536215653197081
review_blob: d8ce4cbf6e2e54cb8294363e025022d9284a0762
stage: harvested
claimed_at: 2026-10-29T191802Z
base_sha: 80cc831d226d5a172f5b894f4e4810edc893a9d2
base_blob: d8ce4cbf6e2e54cb8294363e025022d9284a0762
branch: hyp/muse_prefill_layer/experts-rega-a5b-desc-splitcommit
worktree: .worktrees/muse_prefill_layer/experts-rega-a5b-desc-splitcommit
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T194011Z
impl_commit: 116dea775204710823f505a6ceb3d8a6a526fee3
submission_sha256: 8c34de1978cebdf3e9f0527f2db231f8c9d1501ec54a51d919a2262d485282e5
norm_sha256: ab35c8c29c8c69a933405d3446940bbc72afb84abe1d86a6b0371769973f63bd
submission_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
lint_warnings: 122
attempt_file: 2026-10-29T194011Z-check-experts-rega-a5b-desc-splitcommit.md
prepared_at: 2026-10-29T194012Z
job: da80fdb9bed749a3
dispatched_at: 2026-10-29T194021Z
outcome: gain
verdict_detail: gain
speedup: 1.579232671320759
speedup_ci: [1.5555441812509476, 1.6029211613905703]
harvested_at: 2026-10-29T194713Z
best_at_harvest: 1.536215653197081
promotion: done
mechanism: null
promotion_commit: d9c22d138d4f368d228d70fb6cf64f346accaac3
promotion_attempt: 2026-10-29T194714Z-submit-experts-rega-a5b-desc-splitcommit.md
promotion_job: 9f93484606cd4f9e
promotion_at: 2026-10-29T194727Z
board: "{\"rank\": 230, \"team\": \"yesthatwilly\", \"speedup\": 1.5721, \"first\": false, \"ahead\": 229, \"to_first\": 1.2929}"
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


## Implementation notes

**Mechanism:** a new DQ 5 arm A5b+D+SC (ENC4 code re-encoding + packed 7-bit scales + per-stage wgmma descriptor + split commit) in the rega dq module's rega_kernel (w13 and w2, every N bucket, low/high/ft), raced in prepare next to A3 and DQ0 and routed per class only when <= 0.97 x A3, DQ0 and master every round; A1/A2 retired.

Worktree `muse_prefill_layer/submission.py` only, all inside the dq module (plus the one dispatch line in `_experts_rega`).

- **Build gate (review MUST 1).** A1 and A2 are retired: `REGA_DQ_NAMES {0: DQ0, 3: A3, 5: A5b}`, `REGA_DQ_ARMS (3, 5)`, `REGA_DQ_INST 12` (rega_kernel<G, BIG, DQ> for DQ 0/3/5). `REGA_DQ_EST_4INST_S` stays 40. The est now counts the 12 instantiations plus the 4 of the ptxas probe build (below): 40 x 16 / 4 = 160 s, need 233 s <= 270 s, the same gate as today. No A4b+D+SC64 fourth arm.
- **Device code (ported from takemyjob hopper/rgx2 771f8eb gen2.py, RGX_DQ 7 / RGX_DESC 1 / RGX_SCHED 1 / RGX_SC_MINN 16 = rgx2 arm 6).**
  - `dq5_k<KK>`: the A4b code views (q & M, (q << 6) & M, rotl(q, 10) & M, ((q << 4) & 0x81808180) | ((q >> 4) & 0x00400040), plain SHF, no IMAD.HI) and the one-LOP3 scale (`lop3 0xEA`, `__byte_perm(w, w, 0x2103)` for odd k, 0x78007800 held in a register as `m28 ^ 0x68007800` with `m28 = dq_rtv(KS, 1 << 28)`).
  - `dq5_rec<R>`: one record.
  - `dq_desc_lo` / `dq_desc_at` / `dq_batch_d<NB, R0, R1>`: D.
  - The MODE 0 k loop for DQ 5 is a new `else if constexpr (DQ == 5)` branch with rgx2's split commit, verbatim. DQ 0 and A3 keep master's loop and make_desc. A3's `dq_k` lost only its dead A1/A2 branches.
  - MODE 2/5 run `dq_stage_t<5>`.
- **Banks (MUST 3).**
  - `_rega_dq_a5b_bank` builds A5b's copy from `state['rega_bank']` (the weights' bank) only. Codes are ENC4-encoded through a nibble LUT (`_rega_dq_enc_luts`; it asserts that ENC4 is a bijection), and scale words are packed with `_rega_dq_pack_scales`.
  - Every 8192-record chunk is checked by its inverse with `torch.equal` (decode, unpack); a mismatch raises and that arm goes OFF.
  - The scale scan (`_rega_dq_scales`, [0x08, 0x7E]) gates both A3 and A5b, as before.
  - `state['rega_dq_bank']` is now `{arm: bank}` and keeps only the routed arms' copies, so A3's copy is freed when no class routes A3. Master's bank is never touched. The dispatch reads `state['rega_dq_bank'].get(dq)`.
- **Canary (MUST 3, 4).**
  - `rega_dq_exact` now takes words / A3-permuted words / ENC4 words / scales / packed scales. It writes master's per-k body and the A3 and A5b bodies for all 121,856 cases at both k16 parities, and compares them bitwise.
  - The same launch checks D: `dq_desc_at(dq_desc_lo(slab), r*(SLAB_HALF>>4)+2k)` against `make_desc(slab + r*SLAB_HALF + 32k)` for every 1024-aligned slab with slab + 2*SLAB_HALF <= 2^18 (1800 cases, `REGA_DQ_DESC_EXPECT`). A mismatch, or a count other than 1800, turns A5b OFF.
  - Two static_asserts pin make_desc's high word (0x40000040) and the offset field.
- **Route gate (MUST 2), `_rega_dq_route_pick`, coded explicitly.**
  - A5b passes when its hs/y equal master's dispatched launches on every self-check routing (and equal DQ0, which `_rega_dq_class` enforces), and its pair MODE 0 is <= 0.97 x master AND x DQ0 AND x A3 in every round, both L2 modes, pop and random. Without an A3 timing, A5b misses.
  - A3 keeps its existing gate.
  - The order is A5b, then A3. `_rega_dq_route` runs the two-state replay check for the first passing arm and falls back to A3's replay if A5b's fails; otherwise master.
- **ptxas (MUST 5).**
  - After the main build, the same background thread runs a probe build. It starts only before 120 s of prepare and its module is never loaded.
  - The probe uses the same source and flags plus `-DREGA_DQ_PTXAS_PROBE` (only rega_kernel<G, BIG, 5> is instantiated: 4) and `-Xptxas -v`, linked against a library that does not exist. The build fails at the link after ptxas has run, and the exception text carries ninja's output.
  - `_rega_dq_ptxas_parse` extracts the A5b entries' registers and spill bytes, C7510/C7512 counts and up to 2 warning lines. The result is printed in the phase and as one summary head line.
  - Regs/spill per instantiation (cudaFuncGetAttributes via rega_init) are printed as before; any spill turns that arm OFF.
- **Log (MUST 6).** The summary is still the last echo, capped at `REGA_DQ_SUMMARY_MAX` 3000. PAIR shows A5b vs DQ0 / vs m / vs A3 ('a'), and ROUTE shows the gate's worst rounds. With A1/A2 gone the RATIOS lines are about 40% shorter; a 3-class synthetic summary measured about 2.0 KB before the head lines.
- **Budget (MUST 7).** Same background thread, same deadlines (240 s ft step / 255 s steps / REGA_COPY_DEADLINE_S), two arms timed instead of three, and the A5b bank copy costs about as much as A3's.

**Deviations and choices.**
- The fourth arm (A4b+D+SC64) is not added: it is optional per the review, and 12 instantiations leave margin for the probe.
- ptxas warnings are captured by a link-stopped probe build instead of moe's `-Xptxas -Werror` probe. The -Werror form misses info-level C7510 notes and needs a failing build to show anything at all. It runs in the existing thread under the same 233 s gate.
- `_rega_dq_launch` now raises if an arm is launched without its bank copy (defensive).

### Node validation, H100 gpunode06 GPU 5 (2026-10-29 19:37Z)

The built submission (sha256 8c34de1978ce, byte-identical to the worktree file) ran through guarded hopper-probe. Everything ran as the operator, with no root and no Slurm. The nodejudge venv was used read-only: every cache, tmp and extension dir was redirected into the run dir, which was then removed. The package is takemyjob worktree `~/git/.takemyjob-wt/pfa5b` (branch hopper/pfa5b, uncommitted), `probes/pfa5b/` (the test drives the submission's own functions). Output: `~/git/.takemyjob-wt/pfa5b/prototypes/hopper-probe/runs/20261029-123722-gpunode06/out/pfa5b/pfa5b.txt`.

The bank is synthetic: random codes, scale bytes in [0x08, 0x47], and HBM-resident (not L2-aliased). It uses nodejudge shims for the judge-only sglang muse_spark modules (the dq path does not use them), and 'master' = DQ0 MODE 0 through master's signature.

- **Build:** the main module (12 instantiations) took 40.9 s and the ptxas probe 34.7 s, 75.6 s of wall time in total. The probe's link stop worked: ptxas output was captured, and A5b showed regs 168x4, spill 0, **C7510 0, C7512 0, no ptxas warning**. attrs (cudaFuncGetAttributes): all 12 at r168 l0 o1.
- **Canary:** 121,856 cases, A3 0 and A5b 0 mismatching; D descriptor 0 of 1800 mismatching. The A5b bank copy took 0.11 s with every inverse check passing. Negative control: a 0x07 scale byte is rejected.
- **Exactness:** hs and y of A3 and A5b are torch.equal to master and to DQ0 on random, skewed, one and pop routings, in all three classes (low T 753, high 3012, ft 16384).
- **Pair timing, A5b vs A3 per round (gate <= 0.97):**

  | class | A5b vs A3 | A5b vs master | route pick |
  |---|---|---|---|
  | low | .961-.972 (worst .972) | .865 | **A3** (A5b missed by 0.002) |
  | high | .931-.958 | .854-.890 | **A5b** |
  | ft | .924-.949 | .852-.875 | **A5b** |

  This bank is HBM-resident with random codes, the node's "low HBM" condition (rgx2 got 0.954-0.963 there). The judge's low class sat nearer the L2 figure (0.923-0.925), so low may pass on the judge; if it misses again, low stays on A3, which is the specified gate.
- **Total test time:** 100 s (classes 1.5 / 2.4 / 10.7 s).

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-a5b-desc-splitcommit diff` (1 file).

- [x] CUDA-graph capturable. The only kernel_fn-path change is `_experts_rega`'s `dq_bank = (state.get("rega_dq_bank") or {}).get(dq)`, a dict lookup on a prepare-time int chosen by the token count (a shape). There is no `.item()`/`.cpu()`/`synchronize()`, no allocation on input values and no data-dependent control flow. All new syncs and `.tolist()` (canary, bank copies, probe) are in prepare(). The DQ 5 kernel has the same launch signature and grid as A3.
- [x] Nothing derived from `data` is cached across calls. The A5b bank copy comes from `state['rega_bank']` (the weights) only and never from a routing. The ptxas probe and canary use synthetic words.
- [x] Weights: repacked losslessly in prepare() only, and NVFP4 stays NVFP4. ENC4 is a bit permutation of each 32-bit code word: the same e2m1 bits in new positions, checked as a bijection, with an inverse `torch.equal` per chunk. The scale bytes are the same e4m3 values stored as b - 8 (7-bit fields), with an inverse `torch.equal` per chunk and the [0x08, 0x7E] range scan first. The dequant product is bit-identical to A3's and master's: the exhaustive canary checks it bitwise, then hs/y are compared with `torch.equal` on every self-check routing.
- [x] Activations BF16 where SGLang keeps BF16. The wgmma A fragments are the same bf16 values as master's, with fp32 accumulation, the same k order and the same accumulator chain; split commit only splits one commit group into two. KV cache and queries are untouched.
- [x] n/a: attention is untouched, so it still sees every key (full window / full context).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, network, or subprocess / multiprocessing / os.fork / spawn. The only build is `load_inline` (torch's own ninja), as master's builds. The probe build's deliberate link stop is a missing `-l` library in `extra_ldflags`, and its module is never loaded.
- [x] All work happens inside the timed call, with no output caching or replay. Prepare-time work is the build, canary, bank repack and race; kernel_fn only launches the routed arm.
- [x] sm_90a: the new PTX (wgmma via the existing `Wg<NB>::mma`, lop3, mov.b64) sits inside the module that is already built with TORCH_CUDA_ARCH_LIST=9.0a and REGA_GENCODE. wgmma stays under master's `__CUDA_ARCH_FEAT_SM90_ALL` guards, and mbarrier waits stay bounded (master's `mbar_wait`).
- [x] A build failure, canary miss, spill or gate miss lands on A3 (if it passed its own gate) or master, and never raises from prepare(): `_rega_dq_phase` keeps its try/finally.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):

```
OK with warnings: muse_prefill_layer/submission.py
(warnings only: host-sync lines all "outside kernel_fn's call graph; fine in prepare()" and 5 pre-existing uint8 torch.empty dtype notes; none new in kernel_fn)
```

py_compile OK; __pycache__ removed. Diff: 1 file, +498 / -196 (muse_prefill_layer/submission.py).

## Result

- job `da80fdb9bed749a3` (check), status done, passed True, tests 5/5
- speedup 1.5792x, 95% CI [1.5555, 1.6029], score 245311.4775519447 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.536215653197081, "ci": [1.5131724183991246, 1.559258887995037], "job": "27f0b9af04604842"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-dq2-dequant-cost-diag.result.json
- **outcome: gain** (gain)
- files: `muse_prefill_layer/hypotheses/done/gain/experts-rega-a5b-desc-splitcommit.result.json`, `muse_prefill_layer/hypotheses/done/gain/experts-rega-a5b-desc-splitcommit.patch`; branch `hyp/muse_prefill_layer/experts-rega-a5b-desc-splitcommit`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 709.56 | 661.91 | 1.0109 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1067.11 | 1030.83 | 1.006 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2077.81 | 1936.06 | 1.0265 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3533.58 | 3491.54 | 1.0109 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 11052.78 | 11160.68 | 1.042 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 10545.85 | 10326.33 | 1.0213 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14570.54 | 14556.85 | 1.0065 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18087.08 | 17600.87 | 0.9979 | flat |

gained on full first_turn. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

