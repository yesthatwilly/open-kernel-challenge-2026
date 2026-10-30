---
problem: muse_prefill_layer
slug: rega-s2-low-rs2-high-stack
created: 2026-10-30T002745Z
base: master
mint_blob: "7569df17e62ecfb6ecf85d12e2033c6611fcaade"
best_at_mint: 1.5721
area: "experts rega tile: S2 2-CTA at low + RS2 (wait<1> per group) at high, both on A5b+D+SC"
change: >
  Stack two shelved, bit-exact rega arms in master's per-class dq race (prefill master = A5b+D+SC,
  1.5721): (1) S2, the 2-CTA/SM low-class tile from experts-rega-low-2cta-a5b (branch
  hyp/muse_prefill_layer/experts-rega-low-2cta-a5b; judge 074a0e02 routed it at low: S2/A5b
  .931-.932, guard .968), restacked from its patch with all its gates; (2) RS2 at the HIGH class
  only (takemyjob hopper/rgx3 2b6d50e; research/h100-rega3-1029.md: wait<1> every group so the
  next stage's dequant runs under the previous stage's second wgmma group; node pair vs A5b+D+SC
  at high 0.959-0.963, w2 0.955; low/ft no gain; check ptxas C7510-C7514 and route only if clean),
  raced at high with the same <= 0.97 vs A5b every round + exact rule. ft keeps A5b+D+SC. Keep the
  build gate: count instantiations (S2 2 + RS2 2 for high) and probe only the new ones so the 233
  s need holds; print the [rega-dq] summary with both arms.
expected:
  - shapes: load=turn_1*
    direction: faster
    why: "S2 at low: judge 074a0e02 S2/A5b .931-.932"
  - shapes: load=turn_4*
    direction: faster
    why: "RS2 at high: node pair 0.959-0.963 vs A5b+D+SC"
  - shapes: load=first_turn*
    direction: same
    why: ft keeps A5b+D+SC
risk: medium
depends_on: experts-rega-low-2cta-a5b
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-low-2cta-a5b: S2 alone read flat on the headline (1.5678 vs 1.5721) while routed; this adds RS2 at high so the stacked gain clears one run's noise"
target: check
value_gain: 1.8
value_p: 0.5
value_ev: 0.9
value_cost: 350k tokens, 1 judge run
evidence: >
  judge 074a0e02f4e94efd: S2 routed at low, S2/A5b 0.932 pop warm, 0.931 cold, 0.885 random, guard
  1348 routes 0.968, vs master 0.743-0.772 (turn_1 rega ~317 us of 643 us); H100 rgx3
  (research/h100-rega3-1029.md): RS2 pair vs A5b+D+SC at high L2 0.963, HBM 0.959, w2 0.955
  (turn_4 rega ~1076 us)
user_requested: true
review: proceed
status: running
review_at: 2026-10-30T003004Z
review_companions: null
review_best: 1.5721154841318719
review_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
stage: dispatched
claimed_at: 2026-10-30T003004Z
base_sha: 7feca0ccdd2513dfd07fd00241f638b1889fb77c
base_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
branch: hyp/muse_prefill_layer/rega-s2-low-rs2-high-stack
worktree: .worktrees/muse_prefill_layer/rega-s2-low-rs2-high-stack
mode: check
restack_of: experts-rega-low-2cta-a5b
stale_base: null
implemented_at: 2026-10-30T003721Z
impl_commit: 6287ccb0d37a8618e12ddbb46ad8928a3d8f2507
submission_sha256: 9c897701e302273f999b636879ec54f8b0abc079e0562a4567d7465057b9fa11
norm_sha256: e440fe0bd1803855232ef79d6b5d09a6cca1f056fce344b9a53cc8d5749e16a9
submission_blob: 32261c5ec0f77dc39d8534fbe2a2cf911409e466
lint_warnings: 125
attempt_file: 2026-10-30T003722Z-check-rega-s2-low-rs2-high-stack.md
prepared_at: 2026-10-30T003722Z
job: 2c9af73f6b0f4e1b
dispatched_at: 2026-10-30T003735Z
---

# rega-s2-low-rs2-high-stack

## Change

Stack two shelved, bit-exact rega arms in master's per-class dq race (prefill master = A5b+D+SC, 1.5721): (1) S2, the 2-CTA/SM low-class tile from experts-rega-low-2cta-a5b (branch hyp/muse_prefill_layer/experts-rega-low-2cta-a5b; judge 074a0e02 routed it at low: S2/A5b .931-.932, guard .968), restacked from its patch with all its gates; (2) RS2 at the HIGH class only (takemyjob hopper/rgx3 2b6d50e; research/h100-rega3-1029.md: wait<1> every group so the next stage's dequant runs under the previous stage's second wgmma group; node pair vs A5b+D+SC at high 0.959-0.963, w2 0.955; low/ft no gain; check ptxas C7510-C7514 and route only if clean), raced at high with the same <= 0.97 vs A5b every round + exact rule. ft keeps A5b+D+SC. Keep the build gate: count instantiations (S2 2 + RS2 2 for high) and probe only the new ones so the 233 s need holds; print the [rega-dq] summary with both arms.

## Expected

- `load=turn_1*` faster: S2 at low: judge 074a0e02 S2/A5b .931-.932
- `load=turn_4*` faster: RS2 at high: node pair 0.959-0.963 vs A5b+D+SC
- `load=first_turn*` same: ft keeps A5b+D+SC

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** judge 074a0e02f4e94efd: S2 routed at low, S2/A5b 0.932 pop warm, 0.931 cold, 0.885 random, guard 1348 routes 0.968, vs master 0.743-0.772 (turn_1 rega ~317 us of 643 us); H100 rgx3 (research/h100-rega3-1029.md): RS2 pair vs A5b+D+SC at high L2 0.963, HBM 0.959, w2 0.955 (turn_4 rega ~1076 us)
- **Gain if it works:** 1.8% headline; **p** 0.5; **EV** 0.9% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.8% < 2.5%; EV 0.9% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Each part alone is ~+1% headline (below one run); stacked ~+1.5-2%. S2's judge read is direct; RS2's is node-only (A5b-family node->judge transfer held within 0.01-0.02).

## Adversarial review

Reviewed 2026-10-30T003004Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: fresh (board 1.5721, blob 7569df17 unchanged); CLAIMS.md:79 S2 [shelf] reopen ('ride in the next prefill candidate') is met in form, but the stack sums to ~1.8% < the 2.5% stack bar. It proceeds only because the user asked for it. Fix the build-gate arithmetic before building: with RS2 the need is ~259 s, not 233 s. Read the result from the per-class [rega-dq] pair ratios, not from the headline.

### Justification
A: Both arms are plausible, bit-exact per-SM schedule changes on the A5b+D+SC tile. S2 has a direct judge read: check 074a0e02 routed it at low with pair S2/A5b .932 warm, .931 cold, .885 random and guard 1348r .945-.959 (gate g.968 <= 1.00). The probe shows 80 regs, 0 spill and C7510/C7512 0. RS2 is node-only (campaigns/2026-10-26-expert-bandwidth/research/h100-rega3-1029.md). wait<1> per group removes the per-stage wgmma-wait barrier at high: w13 cycles 968k -> 940k and tensor 78.7 -> 81.1%. The high pair is 0.963 at L2 and 0.959 at HBM, w2 0.955. At low it is neutral or worse (low hbm 1.045), and at ft it is noise (0.990 / 1.016, +-10% round spread), so high-only routing is right. RS2-2CTA hit C7512, so RS2 must be instantiated only on the 1-CTA, 168-reg high tile, never combined with S2's 80-reg template. WEAKEST LINK: whether RS2 routes at all on the judge. The gate is <= 0.97 vs A5b on every round plus exact. RS2's interval is [0.950-0.976] at L2 and [0.950-0.976] at HBM, and the earlier 2-round val runs read 0.970-0.974, which would FAIL that gate. Treat p(RS2 routes) as ~0.4-0.5. If it does not route, the run is S2 alone, already measured flat (1.5678; its single run also 'lost on full first_turn', a row it does not touch, which shows the ~1-1.5% run noise). SECOND: the build gate claim is wrong. On the S2 branch: est = 40 x (INST + PROBE_INST)/4 and need = used + 1.3 x est + 25. Today 14 + 2 gives need 233. Adding RS2's 2 instantiations (16) plus probing only RS2 (2) gives est 180 and need = used + 259. That passes the 270 limit only while used <= 11 s. The judge started the build at 0.1 s and the real build took 40.5 s of a 160 s estimate, so it passes, but with 11 s of slack, not 37. Probing both S2 and RS2 (16 + 4) gives 285 and the whole dq module goes OFF, losing the landed A5b routing too. Required: point PROBE_INST at RS2's 2 instantiations only, pin S2 by its judge-proven source (no re-probe), log need, and assert need <= 270 in a local dry count. Expected directions match TASK weights: turn_1 rega ~317 of 643 us x ~7% gives ~+1% headline, turn_4 rega ~1076 us x ~4% gives ~+0.7%, ft flat. PRIOR: S2's own run is cited correctly. RS2 has no earlier prefill attempt. It is the deferred wgmma_wait<1> sibling named in CLAIMS.md:71's second line. It is not the refuted anti-phase or K-pair (CLAIMS.md:57-58), because RS2 changes neither warp count nor issue order across WGs. Not a do-not-retry duplicate.

### Confounders on the current kernel
- experts-rega-a5b-desc-splitcommit (CLAIMS.md:76, landed): the base of both arms. RS2 is a modification of its split commit (rolling instead of drained per stage), so any gain is measured relative to it by construction. The race gates vs A5b (b) and master (m), so this is attribution, not confounding.
- experts-rega-w13-fast-silu-epilogue (landed) and the w2st arm a pinned at high/ft (CLAIMS.md:56): the rgx3 node tile is rgx2's generated tile and may not carry master's fast-silu epilogue or the w2 early gs/sid loads. RS2's 0.955-0.963 is vs a node A5b+D+SC that may differ from master's high-class launch. The in-prepare race against master's dispatched launch (m) covers the routing decision. A node figure that does not transfer shows up as RS2 not routing, not as a wrong route.
- rega_lo / REGA_LO_GUARD_ROUTES (landed low-class gate): S2's route depends on the guard routing (~1350 routes) at <= 1.00. That passed at .945-.959 on 074a0e02 and should hold unchanged, since the S2 arm is a byte-identical restack.

### Interference and A/B needs
No companions. The two arms touch disjoint classes (S2 low only, RS2 high only, ft untouched), and the [rega-dq] summary prints per-class pair ratios and the ROUTE decision for each arm inside the same run. Attribution is therefore in the log: a turn_1 move belongs to S2 and a turn_4 move to RS2. A revert companion of A5b is not implementable, since both arms are built on its encoded bank. A control of S2 alone already exists (074a0e02, 1.5678). The headline alone stays ambiguous at ~1.8% expected vs a ~+-1.5% single-run CI. Judge the run on the route lines, the pair ratios, and the turn_1/turn_4 rega us in the profile vs 074a0e02 and da80fdb9. Do not judge it on the board.

### Open questions
- Does master's RS2 port keep the gate's 0.97 every-round rule at high? Given node 0.959-0.976 and val 0.970-0.974, would the user accept a per-class RS2 threshold, or is it better to know it will likely not route? Do not loosen the threshold in this run.
- Is RS2 a BIG=true high-class instantiation pair (w13, w2) only, with no DQ-mode cross product? Any extra template axis puts need at >= 285 s and turns the dq module OFF.
- Should REGA_DQ_EST_4INST_S be recalibrated (the judge built 14 inst in 40.5 s vs est 160 s)? That would restore build slack. It is a separate change and should not ride silently in this stack.


## Implementation notes

**Mechanism:** experts rega dq module: S2 (2-CTA SC tile, restacked byte-for-byte) raced/routed at low, plus RS2 = A5b+D+SC with a rolling split commit (wgmma_wait<1> before each record's dequant, previous stage released after its second group retires) as `rega_kernel<G, false, 5, 2>` on master's 1-CTA 168-reg tile, raced/routed at high only (turn_4 rega rows); ft keeps A5b+D+SC.

- Worktree diff vs HEAD: 475+/95- (S2 restack staged by churn: 336+/88-; RS2 on top, unstaged: 187+/55-), all in `muse_prefill_layer/submission.py`.
- CUDA (REGA_DQ_EDITS): the existing `int S2` template param takes the value 2 for RS2 (no new template axis): `Geo<2> == Geo<0>` (every `S2 ?` became `S2 == 1 ?`, so S2 = 0/1 compile exactly as before; static_assert that SmemS<.,2> is master's smem). consume_item gets an `if constexpr (DQ == 5 && S2 == 2)` MODE-0 loop ported from rgx3 `RGX_SCHED 3, RGX_NG 2` (wait<1>, dq5_rec<0>, fence, 4 wgmma, commit; wait<1>, release prev stage, dq5_rec<1>, fence, 4 wgmma, commit; final wait<0> + fence_acc + release). MODE 2/5 and the epilogue are shared, so RS2 carries master's fast-silu w13, w2st arm a and the non-local zero store (and exactness vs master's dispatched launches checks that). Host: dq 9 accepted; `case 9` launches `<G,false,5,2>` (a BIG table, impossible at high since T <= 4096 gives <= 2078 blocks, falls back to A5b's existing `<G,true,5,0>`, which is also bit-exact, and adds no instantiation). Flag id 9.
- Build gate (review fix): instantiations are 12 + S2 2 + RS2 2 = 16 (`REGA_DQ_INST = 16`). The ptxas probe build (`-DREGA_DQ_PTXAS_PROBE`) now instantiates RS2's 2 only, and S2 moved inside `#if !defined(REGA_DQ_PTXAS_PROBE)` (judge-proven 074a0e02, not re-probed). est = 40 x 18 / 4 = 180 s, need = used + 259 s <= 270 s while used <= 11 s. A local dry count (exec of the file with stubbed imports): info_one main 16 / probe 2; launch tuples {B_,0,0},{B_,3,0},{B_,5,0},{false,5,1},{false,5,2}; assert need(0.1 s) <= 270 passed. The head line logs `need Ns (16+2 inst) <= 270s`.
- Probe parse: entries matched on `Li5ELi2E` (RS2), plus a new `c751x` count of any ptxas line naming C7510-C7514. `_rega_dq_probe_ok`: state ok, 2 entries, 0 spill, c751x 0.
- Race/route: RS2 is raced only at high, with A5b raced, both RS2 attrs ok (master's criteria + 0 local bytes), and A5b's bank (shared, no copy). It is raced even with an unclean probe so the timing is still read, but `_rega_dq_rs2_gate` routes it only when the probe is clean AND A5b passes its own gate AND hs/y are torch.equal to master's dispatched launches on every exactness routing AND pair MODE 0 is <= 0.97 x A5b, x master and x DQ0 in every round and L2 mode on pop and random (threshold not loosened). It then goes first in the order and still needs master's two-state replay check (fresh routings, plan high T=3012) before `routes[high] = 9`; on a replay failure it falls back to A5b. RS2's first launches at high are announced (print, sync, print).
- Prints: ROUTE high carries `RS2 <m>m/<d>d/<b>b<=0.97 eq n/N probe ok|MISS [A5b-miss] pass|miss`. TIME/PAIR lines give RS2 vs DQ0/m/A5b('b') as S2's do. The probe line reads `ptxas probe RS2 ... C7510-C7514 n ... clean|NOT clean`. The attrs line has `RS2 r/l/o`. `_rega_dq_cap` now cuts the SUMMARY head, ROUTE lines, PAIR low/high and the probe line last. A synthetic dry run of `_rega_dq_summary` + cap (all 3 classes, S2 at low, RS2 at high) came to 3281 chars uncapped and 3000 capped, with the head (build/est/need), probe, PAIR high incl. random and every ROUTE line intact. Only the key/SCALES/low RATIOS lines are trimmed.
- Deviations: (1) RS2 reuses the `S2` template int (value 2) instead of a new param, so there is no extra axis. (2) S2 is "pinned" by leaving its template text semantically unchanged and unprobed, not by a source hash: the dq module hash changes anyway with RS2. (3) The cap tiering is a summary-only change. Not done (review: separate change): no REGA_DQ_EST_4INST_S recalibration.
- Biggest risk: node -> judge transfer of RS2 at the 0.97-every-round gate (node val 0.970-0.974; p(route) ~0.4-0.5). Compile risk: ptxas serialising the in-flight register-A wgmma (C7510-C7514). The probe then shows it and RS2 does not route. Build slack is 11 s.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/rega-s2-low-rs2-high-stack diff` (and `diff HEAD` for the staged S2 restack):
- [x] CUDA-graph capturable: kernel_fn's only change is `dq in (..., REGA_DQ_RS2)` on a prepare-time int; no `.item()`/`.cpu()`/sync/print, no allocation or control flow on input values (all new syncs/prints are in prepare()'s dq phase).
- [x] Nothing derived from `data` is cached across calls (RS2 reads A5b's prepare-time bank copy built from the weights only).
- [x] Weights: repacked losslessly in `prepare()` only (A5b's ENC4/packed-scale copy, inverse-checked; RS2 adds no new repack); NVFP4 stays NVFP4.
- [x] Activations BF16 as before; RS2 is bit-exact by construction (same k order and accumulator chain) and gated on torch.equal vs master; KV cache / queries untouched.
- [x] Attention untouched (n/a: change is in the MoE expert GEMMs).
- [x] No timing patches, harness access, module edits, reference import, subprocesses (load_inline in-process only, as master) or network.
- [x] All work inside the timed call; no output caching or replay (routing chosen in prepare from synthetic routings).
- [x] First line `# okc: problem=muse_prefill_layer` kept; no triton.autotune; sm_90a PTX stays under master's `__CUDA_ARCH_FEAT_SM90_ALL` guard with TORCH_CUDA_ARCH_LIST=9.0a set/restored as before; mbarrier waits bounded (master's mbar_wait).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):
```
OK with warnings: .worktrees/muse_prefill_layer/rega-s2-low-rs2-high-stack/muse_prefill_layer/submission.py
(warnings: host-sync / dtype lines, all "outside kernel_fn's call graph; fine in prepare()" or pre-existing uint8 buffers; no errors)
```
py_compile: ok; __pycache__ removed. Size 494706 B.
