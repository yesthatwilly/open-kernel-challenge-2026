---
problem: muse_moe_w4a16
slug: cmbf-combine-fused-w2-epilogue-t8-32
created: 2026-10-29T151134Z
base: master
mint_blob: "691018b5844d1a985d98f59021313b0c1a3ef3b7"
best_at_mint: 1.9521
area: post-expert norm+combine
change: >
  DIAG (print-only race; the board's arms stay routed, so the timed path is master's). Build arm
  CF ("combine fused") of the routed K2 twins: the w2 kernel of v4xs / v4cxs (ea4x_kernels.cuh,
  eak4x::expert_body_x<false, DQ, false>) and of EPLF v4xe / v4cxe (ea4xe_kernels.cuh), as a new
  template parameter CF (separate instantiations; the board's instantiations keep their source
  path and SASS), compiled in its OWN background module (ea4xcf, built like _stk_build_start; a
  compile or admission failure costs only this diag, never ext4x / ext4xe). New entry points
  ea4x_w2cf / ea4xe_w2cf = the w2 signature + tw32 [T,8] fp32, norm_w [4096] bf16, out [T,4096]
  bf16, eps, cfc (int32 [EA3_MAX_T] token counters, zeroed once in prepare; every finisher resets
  its own, as tickets). WHERE THE COMBINE REDUCTION GOES: y is still written by epi<false> exactly
  as today (the bf16 rows the combine reads, so out stays bit-equal). (1) In-loop hook: after
  EVERY epi<false> call (arm s: aligned owner and ticket finisher; EPLF: owner and fixed-role
  finisher), CTA-uniform, append sgrp to a small smem list (<= 16 entries, 64 B past GS_OFF; SMEM
  slack is 2,568 B/CTA under the 2 x (SMEM + 1024) <= 233,472 assert). Nothing new is carried in
  registers across consume / consume2 (arm s compiles to 93, EPLF to 95, cap 96). (2) Zero rows:
  K2 CTA i (i < T), after its existing griddep_wait on K1, writes out[i] = +0 bf16 iff
  local_ids[i, 0..7] are all -1 (ea_combine's no-local-slot store). (3) CTA end (after the loop
  and any EPLF / DEFER finish): one bar_consumers; consumer threads aggregate, per token t =
  pos_r[v_beg[v] + jj] >> 3 over the listed super-groups' routes, the count of completed
  (super-group, local slot) pairs in smem; thread 0 fence.acq_rel.gpu (release: every warp's y
  stores, ordered by the barrier, the CUTLASS GenericBarrier idiom arm s already uses); then one
  thread per touched token atomicAdd(&cfc[t], n) (integer, ordering-insensitive); the CTA whose
  add makes cfc[t] reach NS x n_t (n_t = number of local slots of t from local_ids, NS =
  super-groups per w2 pass) owns token t: fence.acq_rel.gpu (acquire), sets cfc[t] = 0,
  bar_consumers. (4) For each owned token (1-3 per CTA at T=8..32 under EP8, ~1 local slot per
  token), the 8 consumer warps (= 256 threads, exactly ea_combine's THREADS) run
  ea_combine_kernel's body for t: thread tid owns columns tid*8 and 2048 + tid*8, per-slot ss
  summed i = 0..15, xor-shuffle tree 16..1, red[s][warp] in the now-free ring smem, tot summed
  over warps 0..7 from 0.f, rsqrtf(tot / 4096 + eps), acc += g[s] * v * r over local s in slot
  order, bf16 rn store. Register-lean form: slots are not all hoisted (load, ss, then reload with
  __ldcg in the accumulate pass; nw[16] may be recomputed); same per-element fp32 order, so out is
  BITWISE equal to the incumbent's ea_combine. y and ids reads use __ldcg (other SMs wrote them).
  No fp atomics anywhere, no CTA waits or spins on another (last-arriver only, no co-residency
  requirement), no host sync, nothing cached from inputs; one launch per call, graph-capturable,
  kernel_fn branches only on T and prepare-set state. (5) Host: kernel_fn / _ea3_forward_core gain
  a cf flag: when ea['cf_t'][T] is set it calls the CF entry point with out and skips
  ext.ea_combine; CMBF_ROUTE = False in this diag, so cf_t stays empty after prepare. (6) Prepare
  stage _cmbf_stage after _stk_stage (deadline-gated, never raises, restores ea['on']): admission
  (ea4xcf_info regs <= 96 and 0 B spill per instantiation, else the arm is off), canary
  (torch.equal(out_cf, out_inc) and y equal at EA4X_CANARY_T on judge-like and random routings,
  NaN-poisoned y / out, err word 0, cfc all zero after each call), copy check per class at its
  largest T (graph replay on a state copy), then per class 8 / 16 / 32 / 128 a paired race
  (_ea6_race over _ea4x_draws, the routed twin + ea_combine vs the same twin's CF instantiation
  without the combine, RT_ROUNDS rounds plus EA4X_MORE pooled) printing one [cmbf] line per class:
  routed twin, median us inc / cf per draw (p10 / p50 / p90), d = inc - cf in us, ratio worst /
  pooled, rounds, canary, copy, then 'NOT ROUTED (diag)'. T = 1 (v6pe2's early combine) and 256
  are out of scope. Keep the added source <= 12 KB (submission 964,183 B; budget 988,500 B).
expected:
  - shapes: bs=16 *
    direction: flat
    why: "diag: board arms stay routed (CMBF_ROUTE=False), timed path = master. Measured lever: the [cmbf] T16 race d = inc - cf us. Ceiling = the combine's path, judge profile dceed39d ea_combine 2.6 us of 111.5 us row -> <= x1.024 row, <= 0.39% headline (1/6 weight)"
  - shapes: bs=32 *
    direction: flat
    why: "diag, timed path = master. [cmbf] T32: ceiling ea_combine 3.2 us of 140.2 us -> <= x1.023 row, <= 0.38% headline"
  - shapes: bs=8 *
    direction: flat
    why: "diag, timed path = master. [cmbf] T8 (verify_1, v4xs): combine on the T<=8 critical path 2.5-3.8 us (CLAIMS [node-only] T<=8) of 81.2 us -> <= 0.5-0.8% headline"
  - shapes: bs=128 *
    direction: flat
    why: "diag, timed path = master. [cmbf] T128 (verify_16, v4cxs + CUDA combine): ~3 us est. of 232.3 us -> <= 0.2% headline. Sum of the four ceilings ~1.5% headline; the fc T=1 design estimate (fused tail 2.0-2.4 us vs 2.9 us path) says ~0.5-1.2 us/row is captured, i.e. ~0.5% mid"
  - shapes: bs=1 *
    direction: flat
    why: T=1 is v6pe2 (early combine), not touched
  - shapes: bs=256 *
    direction: flat
    why: T=256 hybrid, not touched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "cuda-combine-kernel-b-bs128-only: that was a standalone CUDA combine kernel replacing SGLang's at T=33..191 behind Marlin (expired unrun); this removes the separate combine launch at T=8..32 (+128) by fusing it into the v4x / EPLF stream-K w2 twins. The nearest design, the T=1 fc ticket-fused K2+combine (research/fc-designs-2026-10-28/design-fusedcombine.json), was rejected unbuilt on a 0.5-1.2 us estimate with one grid-wide ticket; here per-token last-arriver counters spread the 1-3 token combines over the CTAs that complete them, and the run is a paired in-prepare measurement in us per class, not a routed bet"
target: diag
value_gain: 1.5
value_p: 0.2
value_ev: 0.3
value_cost: 450k tokens (build two CF instantiations in their own module + stage + audit), 1 judge run (moe has 12 checks left today)
evidence: >
  Judge profile dceed39d (H200, torch profiler): ea_combine 2.6 us (decode 16) / 3.2 us (decode
  32) after ea4x_w13 59.8/73.9 + ea4x_w2 36.2/43.2 us; gpunode06 nsys (runs 20261028-225317/-225725,
  research/h100-t32-1029.md Q3): combine critical path 2.80-2.95 us (dur 12.1-14.5 us, PDL wait
  9.4-11.6), 2.5-2.7 us flat after w2, shares w13 65-66 / w2 31-32 / combine 2%, 0 launch gaps.
  T<=8 combine on the path 2.5-3.8 us (CLAIMS [node-only] T<=8, H100:699-717). The only sizing of
  a fused tail is the T=1 fc design (research/fc-designs-2026-10-28/design-fusedcombine.json,
  never built): release ~0.4 + atomic ~0.35 + acquire ~0.15 + y L2 trip ~0.35 + body -> fused tail
  2.0-2.4 us vs 2.9 us path = 0.5-1.2 us saved (central ~0.8). CLAIMS [shelf] stack3 /
  combine-fused lines (2026-10-29 screen) name '>= 2 us per row' as stack3's reopen.
decision: >
  Read only classes whose canary (bitwise out and y), copy check and err word passed; d = pooled
  median (inc - cf) us on the p50 draw, p10/p90 as spread. (A) d >= 2.0 us at BOTH T16 and T32 ->
  the combine arm is sized at >= ~0.6% headline: mint stack3 (EPF + slot rider + combine fusion on
  stack2) as a --target check with CF routed per class by the EA4X_MARGIN race (incumbent fallback
  on any canary/copy/race miss), and post the combine claim as [lead] with the d values. (B) 1.0
  <= d < 2.0 us at T16/T32 -> combine stays [shelf] with the measured d; stack3 is not minted on
  it (mid stays < 2.5%) unless another arm is measured first; the CF code stays unrouted on its
  branch. (C) d < 1.0 us (or cf slower) at T16 and T32 -> [refuted] combine fused into the v4x w2
  epilogue at T=8..32 (with d), drop it from stack3 (stack3 mid ~1.3% -> stays shelved). T8 / T128
  readings ride along: a class >= 2.0 us is added to stack3's routed classes under (A) only.
  Canary / copy / build failure -> no reading; fix once (never re-roll identical bytes), else
  close unmeasured. The timed path is master's, so no scored run whatever the headline.
instrument: >
  prepare-time paired race of kernel_fn CUDA graphs on the judge H200 (_ea6_race over _ea4x_draws
  p10/p50/p90, RT_ROUNDS + EA4X_MORE pooled rounds): routed twin + ea_combine vs the twin's CF
  instantiation without the combine launch; one [cmbf] line per class 8/16/32/128 with inc/cf
  medians and d in us
user_requested: true
review: proceed
status: done
review_at: 2026-10-29T151704Z
review_companions: null
review_best: 1.952050927455536
review_blob: 691018b5844d1a985d98f59021313b0c1a3ef3b7
stage: harvested
claimed_at: 2026-10-29T151704Z
base_sha: 1da83323d3e8664218809cf1c921c5ffdc92545b
base_blob: 691018b5844d1a985d98f59021313b0c1a3ef3b7
branch: hyp/muse_moe_w4a16/cmbf-combine-fused-w2-epilogue-t8-32
worktree: .worktrees/muse_moe_w4a16/cmbf-combine-fused-w2-epilogue-t8-32
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T155356Z
impl_commit: 42ee20ddf03e82dd4a76c80a7d6185687e5a16ea
submission_sha256: aabff6091f36f05e29239481407f45801210f1b95a52ba54587cadb588beb82d
norm_sha256: db17c1745521add433448cd16960a4ec2ba54603c16adcc9dcec0e81f731efba
submission_blob: 08be8b80f820bab310402d564c9aa1c1f305c5ae
lint_warnings: 198
attempt_file: 2026-10-29T155357Z-check-cmbf-combine-fused-w2-epilogue-t8-32.md
prepared_at: 2026-10-29T155357Z
job: a5aee75176174ce4
dispatched_at: 2026-10-29T155410Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.9674503247838624
speedup_ci: [1.9379385699121043, 2.0041685111626464]
harvested_at: 2026-10-29T162118Z
best_at_harvest: 1.952050927455536
promotion: null
mechanism: null
---

# cmbf-combine-fused-w2-epilogue-t8-32

## Change

DIAG (print-only race; the board's arms stay routed, so the timed path is master's). Build arm CF ("combine fused") of the routed K2 twins: the w2 kernel of v4xs / v4cxs (ea4x_kernels.cuh, eak4x::expert_body_x<false, DQ, false>) and of EPLF v4xe / v4cxe (ea4xe_kernels.cuh), as a new template parameter CF (separate instantiations; the board's instantiations keep their source path and SASS), compiled in its OWN background module (ea4xcf, built like _stk_build_start; a compile or admission failure costs only this diag, never ext4x / ext4xe). New entry points ea4x_w2cf / ea4xe_w2cf = the w2 signature + tw32 [T,8] fp32, norm_w [4096] bf16, out [T,4096] bf16, eps, cfc (int32 [EA3_MAX_T] token counters, zeroed once in prepare; every finisher resets its own, as tickets). WHERE THE COMBINE REDUCTION GOES: y is still written by epi<false> exactly as today (the bf16 rows the combine reads, so out stays bit-equal). (1) In-loop hook: after EVERY epi<false> call (arm s: aligned owner and ticket finisher; EPLF: owner and fixed-role finisher), CTA-uniform, append sgrp to a small smem list (<= 16 entries, 64 B past GS_OFF; SMEM slack is 2,568 B/CTA under the 2 x (SMEM + 1024) <= 233,472 assert). Nothing new is carried in registers across consume / consume2 (arm s compiles to 93, EPLF to 95, cap 96). (2) Zero rows: K2 CTA i (i < T), after its existing griddep_wait on K1, writes out[i] = +0 bf16 iff local_ids[i, 0..7] are all -1 (ea_combine's no-local-slot store). (3) CTA end (after the loop and any EPLF / DEFER finish): one bar_consumers; consumer threads aggregate, per token t = pos_r[v_beg[v] + jj] >> 3 over the listed super-groups' routes, the count of completed (super-group, local slot) pairs in smem; thread 0 fence.acq_rel.gpu (release: every warp's y stores, ordered by the barrier, the CUTLASS GenericBarrier idiom arm s already uses); then one thread per touched token atomicAdd(&cfc[t], n) (integer, ordering-insensitive); the CTA whose add makes cfc[t] reach NS x n_t (n_t = number of local slots of t from local_ids, NS = super-groups per w2 pass) owns token t: fence.acq_rel.gpu (acquire), sets cfc[t] = 0, bar_consumers. (4) For each owned token (1-3 per CTA at T=8..32 under EP8, ~1 local slot per token), the 8 consumer warps (= 256 threads, exactly ea_combine's THREADS) run ea_combine_kernel's body for t: thread tid owns columns tid*8 and 2048 + tid*8, per-slot ss summed i = 0..15, xor-shuffle tree 16..1, red[s][warp] in the now-free ring smem, tot summed over warps 0..7 from 0.f, rsqrtf(tot / 4096 + eps), acc += g[s] * v * r over local s in slot order, bf16 rn store. Register-lean form: slots are not all hoisted (load, ss, then reload with __ldcg in the accumulate pass; nw[16] may be recomputed); same per-element fp32 order, so out is BITWISE equal to the incumbent's ea_combine. y and ids reads use __ldcg (other SMs wrote them). No fp atomics anywhere, no CTA waits or spins on another (last-arriver only, no co-residency requirement), no host sync, nothing cached from inputs; one launch per call, graph-capturable, kernel_fn branches only on T and prepare-set state. (5) Host: kernel_fn / _ea3_forward_core gain a cf flag: when ea['cf_t'][T] is set it calls the CF entry point with out and skips ext.ea_combine; CMBF_ROUTE = False in this diag, so cf_t stays empty after prepare. (6) Prepare stage _cmbf_stage after _stk_stage (deadline-gated, never raises, restores ea['on']): admission (ea4xcf_info regs <= 96 and 0 B spill per instantiation, else the arm is off), canary (torch.equal(out_cf, out_inc) and y equal at EA4X_CANARY_T on judge-like and random routings, NaN-poisoned y / out, err word 0, cfc all zero after each call), copy check per class at its largest T (graph replay on a state copy), then per class 8 / 16 / 32 / 128 a paired race (_ea6_race over _ea4x_draws, the routed twin + ea_combine vs the same twin's CF instantiation without the combine, RT_ROUNDS rounds plus EA4X_MORE pooled) printing one [cmbf] line per class: routed twin, median us inc / cf per draw (p10 / p50 / p90), d = inc - cf in us, ratio worst / pooled, rounds, canary, copy, then 'NOT ROUTED (diag)'. T = 1 (v6pe2's early combine) and 256 are out of scope. Keep the added source <= 12 KB (submission 964,183 B; budget 988,500 B).

## Expected

- `bs=16 *` flat: diag: board arms stay routed (CMBF_ROUTE=False), timed path = master. Measured lever: the [cmbf] T16 race d = inc - cf us. Ceiling = the combine's path, judge profile dceed39d ea_combine 2.6 us of 111.5 us row -> <= x1.024 row, <= 0.39% headline (1/6 weight)
- `bs=32 *` flat: diag, timed path = master. [cmbf] T32: ceiling ea_combine 3.2 us of 140.2 us -> <= x1.023 row, <= 0.38% headline
- `bs=8 *` flat: diag, timed path = master. [cmbf] T8 (verify_1, v4xs): combine on the T<=8 critical path 2.5-3.8 us (CLAIMS [node-only] T<=8) of 81.2 us -> <= 0.5-0.8% headline
- `bs=128 *` flat: diag, timed path = master. [cmbf] T128 (verify_16, v4cxs + CUDA combine): ~3 us est. of 232.3 us -> <= 0.2% headline. Sum of the four ceilings ~1.5% headline; the fc T=1 design estimate (fused tail 2.0-2.4 us vs 2.9 us path) says ~0.5-1.2 us/row is captured, i.e. ~0.5% mid
- `bs=1 *` flat: T=1 is v6pe2 (early combine), not touched
- `bs=256 *` flat: T=256 hybrid, not touched

## Value case

- **Target:** diag: a prepare-only judge diagnostic that sizes a lever.
- **Evidence:** Judge profile dceed39d (H200, torch profiler): ea_combine 2.6 us (decode 16) / 3.2 us (decode 32) after ea4x_w13 59.8/73.9 + ea4x_w2 36.2/43.2 us; gpunode06 nsys (runs 20261028-225317/-225725, research/h100-t32-1029.md Q3): combine critical path 2.80-2.95 us (dur 12.1-14.5 us, PDL wait 9.4-11.6), 2.5-2.7 us flat after w2, shares w13 65-66 / w2 31-32 / combine 2%, 0 launch gaps. T<=8 combine on the path 2.5-3.8 us (CLAIMS [node-only] T<=8, H100:699-717). The only sizing of a fused tail is the T=1 fc design (research/fc-designs-2026-10-28/design-fusedcombine.json, never built): release ~0.4 + atomic ~0.35 + acquire ~0.15 + y L2 trip ~0.35 + body -> fused tail 2.0-2.4 us vs 2.9 us path = 0.5-1.2 us saved (central ~0.8). CLAIMS [shelf] stack3 / combine-fused lines (2026-10-29 screen) name '>= 2 us per row' as stack3's reopen.
- **Gain if it works:** 1.5% headline; **p** 0.2; **EV** 0.3% (bar: gain >= 3.0%, EV >= 1.5%); **below the bar** (gain 1.5% < 3.0%; EV 0.3% < 1.5%), minted because the user described it.
- **Cost:** 450k tokens (build two CF instantiations in their own module + stage + audit), 1 judge run (moe has 12 checks left today).
- **Decision rule:** Read only classes whose canary (bitwise out and y), copy check and err word passed; d = pooled median (inc - cf) us on the p50 draw, p10/p90 as spread. (A) d >= 2.0 us at BOTH T16 and T32 -> the combine arm is sized at >= ~0.6% headline: mint stack3 (EPF + slot rider + combine fusion on stack2) as a --target check with CF routed per class by the EA4X_MARGIN race (incumbent fallback on any canary/copy/race miss), and post the combine claim as [lead] with the d values. (B) 1.0 <= d < 2.0 us at T16/T32 -> combine stays [shelf] with the measured d; stack3 is not minted on it (mid stays < 2.5%) unless another arm is measured first; the CF code stays unrouted on its branch. (C) d < 1.0 us (or cf slower) at T16 and T32 -> [refuted] combine fused into the v4x w2 epilogue at T=8..32 (with d), drop it from stack3 (stack3 mid ~1.3% -> stays shelved). T8 / T128 readings ride along: a class >= 2.0 us is added to stack3's routed classes under (A) only. Canary / copy / build failure -> no reading; fix once (never re-roll identical bytes), else close unmeasured. The timed path is master's, so no scored run whatever the headline.
- **Instrument:** prepare-time paired race of kernel_fn CUDA graphs on the judge H200 (_ea6_race over _ea4x_draws p10/p50/p90, RT_ROUNDS + EA4X_MORE pooled rounds): routed twin + ea_combine vs the twin's CF instantiation without the combine launch; one [cmbf] line per class 8/16/32/128 with inc/cf medians and d in us

## Risk

Why diag, not check: the lever is <= 2.6-3.2 us on 111-140 us rows (<= 0.4% headline per row, ~1.5% over four rows at full capture), under the judge's 1.24% margin and inside one row's run-to-run spread (~1-2 us), so a routed check's headline cannot resolve 1 vs 2 us; a paired in-process race on judge-like draws resolves ~0.3 us and keeps the timed path = board (no stale-board or scored-run question). The CF arm is still built routable (cf_t per class, incumbent fallback) so stack3 flips CMBF_ROUTE only. Risks: (1) the fused tail is serial on the critical CTA: fence + one integer atomic round + acquire + y L2 trip + store ~1.5-2.4 us vs the separate kernel's 2.6-3.2, so d may land at 0.5-1.2 us (the fc estimate) -> outcome C is the most likely; (2) register cap: __launch_bounds__(288, 2) gives 96 regs/thread (2 CTAs x 9 warps put 5 warps on a 16,384-register sub-partition), arm s is at 93 and EPLF at 95, so the hooks must carry nothing across consume2 and the end-of-CTA body must be the register-lean form; admission refuses spill > 0; (3) compile time of the extra module inside prepare's deadlines (EA4X 190-210 s, STK 208-214 s): the stage gates on its own deadlines and prints 'skipped' past them; (4) size: <= 12 KB added under the 988,500 B budget. The hyp.py claim hits (CLAIMS:90 T=1 w2 prefetch, :92 t1 deep ring, :96 v4xr unit order) share identifiers only (_ea6_race, v4xs / v4cxs); none touches the combine.

## Adversarial review

Reviewed 2026-10-29T151704Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. The single diag run can be read: a paired, interleaved in-prepare race (41 reps x 6 pooled rounds per draw, cold L2, baseline subtracted) resolves about 0.2-0.3 us, far finer than the 1.0 / 2.0 us thresholds. The bitwise/determinism argument holds apart from one release-ordering slip that must be fixed in the build. Rule (A) is miscalibrated: d >= 2 us at T16/T32 alone does not lift stack3 to 2.5%.

### Justification
A. Plausible. The incumbent ea_combine (PDL-linked, w loaded before the wait, X0b prefetch) already hides launch/prologue; remaining cost is the post-K2 critical path, 2.5-2.95 us on gpunode06 nsys (h100-t32-1029 Q3), 2.6 / 3.2 us in the judge profile at decode 16 / 32, ~2% of the row (each load 1/6 of the geomean, TASK.md:42). Design matches code: in eakx::expert_body_x, sgrp = v*NS + si with NS = 32 for w2; epi<false> runs exactly once per (sgrp, pass) by the aligned owner (~10341) or ticket/deferred finisher (~10369); y rows pos_r = t*8 + slot match ea_combine's layout, so 'count reaches NS x n_t' detects a fully written token, and a last-arriver owner with ea_combine's per-element fp32 order gives bitwise-equal deterministic output. No fp atomics, no inter-CTA waits. 96-register cap right (STK_REG_CAP 96, __launch_bounds__(288,2)). WEAKEST LINK: the fused tail costs ~1.5-2.4 us against a 2.5-2.9 us path, so d is most likely 0.5-1.4 us; CLAIMS:99 (late CTAs are co-resident-slot CTAs) puts token ownership on the slowest CTAs. Still interpretable: thresholds are far apart relative to resolution. prior difference is real. Resolution: _rt_time interleaves with rotating/flipping order, 256 MB flush, events, empty-graph baseline; 246 samples per arm per draw; SE of median ~0.1 us.

### Confounders on the current kernel
- stack2 / EPLF (CLAIMS:95, :97): T16 reads the EPLF twin at 95/96 registers; if the EPLF CF instantiation spills, T16 has no reading; EPLF and arm s use different finisher hooks.
- engineA-v4x-s1 / v4x-routes (CLAIMS:84, :91): a class falling back to v4/v4c has no CF instantiation, so no reading.
- carrier CUDA combine at class 128 (CLAIMS:69) and X0b combine prefetch/PDL (DISTILL:67): define the incumbent; d measures only the post-K2 path left exposed.
- T=16/32 late-CTA set (CLAIMS:98-99): last-arriver ownership serializes the tail behind the slowest CTAs; a real design cost.

### Interference and A/B needs
No companion run. Print-only diag, paired in-process race, CMBF_ROUTE = False. Remove the codegen confound IN-RUN: add a third race arm 'cf0' = the same CF instantiation with the tail skipped by a runtime flag, still followed by ea_combine (inc vs cf0 = hook/codegen tax; cf0 vs cf = fusion), or print kernel-only w2 timing of twin vs CF-w2 plus regs/SASS size. Print the race's inc absolute us per class next to the judge row (111.5 / 140.2 / 81.2 / 232.3).

### Open questions
- RELEASE-ORDERING SLIP (must fix): thread 0's fence.acq_rel.gpu does not order other threads' atomicAdds; each counting thread must run its own fence.acq_rel.gpu (or atom.add.release.gpu) after bar_consumers, or bar_consumers after thread 0's fence and before the atomics.
- Rule (A) arithmetic: d = 2.0 us at T16 and T32 is ~0.54% headline, below stack3's combine mid 0.75%. Restate (A): stack3 mid = EPF mid + slot-rider mid + sum over classes of (d_c / inc_c)/6 from the race's own inc_c; mint only if >= 2.5%. Use min(d16, d32) for (A), max(d16, d32) for (C), else (B).
- If EPLF CF fails admission, read T16 on v4xs-CF vs v4xs + ea_combine.
- Register-lean tail: do not unroll ea_combine's 8-slot uint4 hoist (64 regs); per-token count aggregation over T ints in free ring smem; sgrp list overflow (> 16) sets the err word.
- T128 ride-along: last CTA may own several tokens serially; do not pool with T8-32.
- Copy check: the graph replayed on a state copy must use its own zeroed cfc buffer.


## Implementation notes

**Mechanism:** print-only diag: ea_combine fused into the routed w2 twins' K2 (v4xs / v4cxs arm s and EPLF v4xe / v4cxe) as per-token last-arriver counters plus ea_combine's body at the CTA end (arm CF, own module ea4xcf), raced as inc / cf0 / cf at T8 / 16 / 32 / 128; the timed path is master's on every row.

What changed (muse_moe_w4a16/submission.py only, +439 lines, 0 deleted; new section `rewrite/lib/cmbf_core.py` after stack_core.py plus four prepare-time hooks):
- **CF module (ea4xcf), its own background build.** `_cmbf_source()` takes `_stk_source()`'s eak..eak4x base and appends: namespace `eacf` (the fused combine), then text copies of the `eak4x` and `eak4xe` namespaces renamed `eak4xcf` / `eak4xecf` with their `__global__` kernels cut and hooks spliced at four anchors (each must occur exactly once or the build raises): the `expert_body_*` signature gains `const eacf::Cf& cf` (plus `static_assert(!W13)`), `eacf::cf_head` before `const int W = ...` (right after `griddep_launch`), `eacf::cf_note` after the loop's `epi<W13>` call (one call site covers the aligned owner, the ticket finisher and EPLF's fixed-role finisher), and `eacf::cf_tail` after the loop (arm s) / after EPLF's end-of-CTA `publish()`. Kernels: `cf_w2s_kernel` (= expert_body_x<false, 2, false>) and `cf_w2e_kernel<PF>` for PF 2 / 1 / 0, all `__launch_bounds__(288, 2)`, dynamic smem = 113,136 + 128 B (the 16-entry super-group list + count, past s_gs; 2 x (113,264 + 1024) = 228,576 <= 233,472). One entry point `ea4x_w2cf` (deviation: the spec named two, ea4x_w2cf / ea4xe_w2cf; one entry with pdl bit 14 = EPLF, PF in bits 12..13 as ea4xe_w2, saves ~1 KB), bit 15 = cf0 (tail off). `ea4xcf_info()` gives ok / regs / lmem / smem / occ per instance. The board's `_EA2_SRC`, `_EA4X_SRC_BODY`, `_STK_SRC_BODY`, `_ea4x_source()` and `_stk_source()` are byte-identical to master (checked by exec'ing both files' builders); kernel_fn, `_ea_forward`, `_ea3_forward_core` and every other function the timed call reaches are unchanged (only `prepare` and `_ea_setup` differ, by the hooks).
- **Device protocol (eacf).** cf_head: thread 0 zeroes the list count; unless cf0, K2 CTA t < T writes out[t] = +0 when topk_ids[t, 0..7] has no local id (K1's local_ids predicate on the input, so no K1 dependency). cf_note: thread 0 only appends sgrp to smem (overflow > 16 sets err 4); nothing is carried in registers across consume / consume2. cf_tail (consumer warps, CTA end): bar_consumers (ring free, all y stores and the list complete); per-token counts of finished (super-group, route) pairs in ring smem (16 entries x 16 routes = one smem atomicAdd per thread); bar_consumers; warp 0, one lane per touched token: `atom.release.gpu.global.add.s32` (review fix 1: every counting lane's RMW is itself a release, after the CTA barriers that order every warp's y stores, the CUTLASS GenericBarrier idiom); the add that makes cfc[t] reach 32 x n_local(t) owns t: that lane `fence.acq_rel.gpu` (acquire), resets cfc[t] = 0, records t (ballot compaction); bar_consumers; then for each owned token comb_one = ea_combine_kernel's body in its fp32 order (per local slot ss over i 0..15 from 0.f, xor tree 16..1, red[s][warp]; one barrier; tot over warps 0..7 from 0.f, rsqrtf(tot / 4096 + eps), acc += g * v * r in slot order, bf16 rn, k = 2i packing). Register-lean (review fix 4): no 8-slot hoist; each local slot's two y uint4 are __ldcg-loaded in pass 1 and reloaded in pass 2, (1 + w) recomputed; red is double-buffered by token parity in ring smem so one barrier per token suffices. No fp atomics, no inter-CTA waits or spins, no host sync.
- **Host (cmbf).** `_cmbf_build_start` (prepare entry, after `_stk_build_start`): a daemon thread first joins the v4x / EPLF / ea6 build threads (bounded by CMBF_BUILD_S = 120 s) and compiles CF only if they are done and it is still before 120 s, so it never competes with ext4x / STK (review fix 6). `_cmbf_stage` (in `_ea_setup` right after `_stk_stage`): joins until CMBF_JOIN_S = 180 s; admission per instance regs <= STK_REG_CAP (96), 0 B lmem, 2 CTAs/SM (arm-s CF, and EPLF CF at the board's K2 PF from e_pf); prints regs/lmem/occ/SASS-instruction-count of the board's K2 s, CF s, the board's K2 e (its PF) and CF e (SASS from the module .so's sm_90 cubin .text via the existing `_es_cubins` / `_es_elf_sections`). Per class 8 / 16 / 32 / 128: twin = the routed arm (v4xs / v4cxs / v4xe / v4cxe); if the routed arm is EPLF and its CF instance is not admitted, the arm-s twin of the same base (v4xs / v4cxs) is raced instead, with the race's inc set to that arm-s twin (review: read T16 on v4xs-CF vs v4xs + ea_combine); classes on any other engine get no reading. Then canary (eager, NaN-poisoned h / y / out and partials, every EA4X_CANARY_T of the class x `_ea_routings` (+ judge draws at 128, the judge p50 draw at T >= 8): twin + ea_combine vs cf, cf, cf0, cf0, out and y bitwise as int16 views, tickets / flags / err / cfc all 0 after each call), copy check (`_ea4x_copy_check` for arm s, `_stk_copy_check` for EPLF, with the CF arm; cfc is zeroed first so the `_wg_copy_tree` state copy clones its own zeroed counters, review fix 5; this state's cfc must be 0 after), then the paired race: `_ea6_race` (kernel_fn graphs, cold L2, baseline-subtracted) of inc, cf0 and cf on `_ea4x_draws` p10 / p50 / p90, 1 + EA4X_MORE calls pooled (6 rounds). The CF cores are registered in `_EA4X_CORES` only for the stage and removed after it, and `extcf` / `cfc` are popped from the state, so after prepare the engine table and state are master's. `_cmbf_final` (prepare's last step) joins a build still running (until 285 s, so no build outlives prepare) and prints the [cmbf] lines last.
- **Output.** `[cmbf] build ...` (build s, start s, regs/lmem/occ/sass of the four kernels, admission); per class two lines: `[cmbf] T<c> <twin> us p10/p50/p90 inc a/b/c cf0 a/b/c cf a/b/c | judge row <111.5 / 140.2 / 81.2 / 232.3> us` and `[cmbf] T<c> d inc-cf p10/p50/p90 tax cf0-inc (p50) fuse cf0-cf (p50) | inc/cf worst/pooled | rounds | canary runs, copy, sums | NOT ROUTED (diag)`; then the VERDICT line.
- **Decision as implemented (review fix 3, restating (A)).** A class reading is valid only with a clean canary, a passing copy check and zero tickets / err / cfc after the race. d_c = pooled p50 (inc - cf) in us. stack3 mid = EPF mid 0.6% + slot-rider mid 0.63% + sum over the valid classes T8 / T16 / T32 of (d_c / inc_c) / 6 x 100%, inc_c the race's own pooled p50 inc. (A) LEAD (mint stack3, CF routed per class by the EA4X_MARGIN race) only if min(d16, d32) >= 2.0 us AND stack3 mid >= 2.5%; (C) REFUTED if max(d16, d32) < 1.0 us; else (B) SHELF. T128 is printed on its own (d and its /6 term), never pooled into the mid. No valid T16 or T32 reading -> VERDICT none (fix once, never re-roll identical bytes, else close unmeasured). Note the arithmetic: at inc ~113 / 154 us, d16 = d32 = 2.0 us adds only ~0.3% + 0.2%, so (A) in practice also needs a large T8 term; (B) is the likely outcome even at d >= 2 us.
- **Deviations / notes.** (1) Spec (5) said kernel_fn / `_ea3_forward_core` gain a cf flag; per the brief's fix 6 (timed path byte-identical) the CF path is a separate prepare-only runner (`_cmbf_run`) dispatched through `_EA4X_CORES` during the stage; CMBF_ROUTE = False is a documented constant and stack3 would add the per-class route (set ea['on'][t] to `<twin>.cf` and keep `extcf` / `cfc`). (2) Zero rows use topk_ids (input) instead of K1's local_ids (same predicate), so no wait on K1 is needed. (3) cf0's combine follows the class's `cmb_t` (ea_combine, else Triton), as the incumbent; the canary compares everything to ea_combine. (4) **Size: +25,624 B (file 989,807 B).** The spec's <= 12 KB / 988,500 B target could not hold two CF instantiations' host code, the canary, copy check, three-arm race, admission/SASS print and verdict; after compaction the file is under the 990,000 B campaign budget (193 B margin) but 1,307 B over 988,500. (5) Cannot compile or run here: the assembled CF source (133 KB) passes g++ -fsyntax-only with the repo's cuda_stub (templates instantiated), anchors resolve once each; register count, PTX (`atom.release.gpu.global.add.s32`) and bitwise equality are only checked on the judge (admission / canary).

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/cmbf-combine-fused-w2-epilogue-t8-32 diff` (one file, +439 / -0):
- [x] CUDA-graph capturable: kernel_fn and every function it reaches are byte-identical to master; the CF runner (prepare-only, graph-captured in the race) has no `.item()` / `.cpu()` / sync / tensor print and branches only on T, the twin and prepare-set state; the three new host-sync warnings (L12373 `_cmbf_sums`, L12469 `.item()` after the copy check, L12510 synchronize) are in prepare-only code.
- [x] Nothing derived from `data` is cached across calls: cfc counters are scratch that each owner resets within the call (and are dropped from the state after the stage); no output or routing is stored.
- [x] Weights: untouched; CF reads the same repacked NVFP4 records (w2r / gs2) as the board's twins.
- [x] Activations BF16 where SGLang keeps BF16: y and out stay bf16; the combine is ea_combine's fp32 math bitwise. n/a for KV / queries (no attention here).
- [x] Attention sees every key: n/a: MoE expert block, no attention.
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network: the only new build is a torch load_inline in a prepare-time thread (as the existing ea4x / stack builds), joined before prepare returns.
- [x] All work happens inside the timed call; no output caching or replay: the timed path is master's (diag, print-only); CF is never routed.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`): rc 0, `OK with warnings` (198 warnings vs master's 195; the 3 new ones are prepare-only host syncs):

```
warn  host-sync    L12373: return [int(v) for v in torch.stack([ea[k].sum() for k in ("tk13" + sx, "tk2" + sx, "err" + sx, "cfc")]).tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L12469: ent["copy"] = (cc[0] and ea["cfc"].sum().item() == 0, cc[1][:40])  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L12510: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/cmbf-combine-fused-w2-epilogue-t8-32/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: ok (__pycache__ removed). cuda_syntax.py (eb-t32d stub): `_EA2_SRC`, `_EB_SRC`, `_ES1_SRC` ok; the assembled CF module source and its generated cpp declarations: g++ -fsyntax-only rc 0.

## Result

- job `a5aee75176174ce4` (check), status done, passed True, tests 5/5
- speedup 1.9675x, 95% CI [1.9379, 2.0042], score 6835.496597088116 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.952050927455536, "ci": [1.922770163543703, 1.9813316913673689], "job": "8418677e467d4fb7"}; base result: muse_moe_w4a16/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cmbf-combine-fused-w2-epilogue-t8-32.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cmbf-combine-fused-w2-epilogue-t8-32.patch`; branch `hyp/muse_moe_w4a16/cmbf-combine-fused-w2-epilogue-t8-32`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 27.34 | 27.29 | 1.0008 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 111.48 | 105.68 | 1.007 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 140.17 | 139.1 | 1.008 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 81.18 | 77.01 | 1.0046 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 232.32 | 231.32 | 1.0338 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 291.06 | 292.19 | 0.9937 | flat |

gained on verify_16 verify_16. Expected hit: 5/6. Noise ±1.7%.

## Follow-ups

- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

