# SWEEP-pf: knob variants around PFC1 (muse_prefill_layer), design 2026-11-01 ~20:30Z

This is a read-only design. It made no okc runs and no H100 runs. The only things written are scratchpad files: this
document, a proposed generator, and a dry run of its arms. Every number below was re-read from the raw records:
- master same-code reads: J1 8b89517b 1.7836, J2 d114d2e8 1.7848, J3 c456c4b9 1.7634, D1 af53426a 1.7815;
- PFC1: checks b49ac7e2 1.7795 and 60384917 1.7947, scored ecf82580 1.7821;
- PFK2: 035dec28 1.7618 and 25f3c3f0 1.7893;
- the H100 combo runs: scratchpad combo-prefill_layer.md and audit-combo-prefill_layer.md;
- the lev-prefill H100 knob runs (takemyjob hopper/lev-prefill): knob1 20261031-011239, moevar -012930, seeds
  -014353, pfgate -021952..-024246;
- public docs/H200_MEASURED_FACTS.md §7/§9 and NEGATIVE_RESULTS.md (prefill section).

How I read the steer: keep PFC1's four parts unchanged. They are K = KPERM T<2048, C = 2-CTA DSMEM split merge,
RC = BND_RC_MIN_T 2048, and RR = the pairwise-tree FP32 denominator. Every one of PFC1's rows cleared its floor
(master row min / 1.025) in all 3 reads, so the variants inherit those floors. Change one or two
scheduling-class knobs on top.

## 0. Answer

- **No variant at the knob level reaches +1% over the board (1.8014) in expectation.**
  - The best one, PFC2, has E[H] of about 1.792: +0.46% over the board and +0.76% over master's 4-read 1.7783.
  - Odds for that variant: P(scored read > 1.7836) ≈ 0.74; P(scored read ≥ 1.8014) ≈ 0.22.
  - If two distinct variants each reach GO and both are scored, P(at least one ≥ 1.8014) ≈ 0.38. Each scored run is a
    fresh routing draw. Nonce-only re-rolls of judged code are excluded.
  - Model behind these odds: read sd 0.0105 (pooled, same-code), plus 0.0070 of mean uncertainty.
  - +1% in expectation needs S1/A2, the small-tile converter co-schedule (CLAIMS:67; +2.2-3.8%). That is a build, not
    a knob.
- **Only one knob class is still live: the MoE work-queue order (O2 = MOE_ORDER_SMALL 2 and its relatives).**
  - It is pure scheduling. The output stays bit-identical: moevar shows an order-2 rel-diff of 0.0 on every row it
    measured (t1s, t1f, t4s, t4f, fts).
  - It transfers about 1:1 to the H200.
  - Every other knob the file exposes is either at its optimum already or measured adverse (§5).

### Ranked variants (rank by E[headline gain vs PFC1] x p(gain > 0 on the judge))

| # | name | change vs PFC1 (knobs) | E[ΔH vs PFC1] | p | gain x p | E[H] | P(>board) | P(≥ +1% board) | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **PFC2** | MOE_ORDER_SMALL 1→2 (1 knob) | +0.32% | 0.80 | 0.26 | 1.7918 | 0.74 | 0.22 | **ready**: generator arm exists; r1/r2 regenerate to the audited sha |
| 2 | PFC2L | + MOE_ORDER_LARGE 1→2 (order 2 at every T; 2 knobs) | +0.35% | 0.78 | 0.27 | 1.7924 | 0.76 | 0.24 | needs an H100 first-turn gate + audit (new arm) |
| 3 | PFC2F / PFC2G | O2 with the small-tile spread fraction 4/5→1/1 or 3/5 (2 knobs) | +0.32% + δ | 0.80 (O2) / ~0.6 (δ) | 0.26 + 0.6δ | ≈1.792-1.794 | 0.74-0.79 | 0.22-0.27 | H100 sweep picks f, or drops it; + audit |
| 4 | PFC1H | MOE_LARGE_T 8192→2048 + MOE_ORDER_LARGE 1→2 (order 2 at T≥2048 only; 2 knobs) | +0.08% | 0.85 | 0.07 | 1.7876 | 0.62 | 0.14 | hedge, used only if O2 at T=753 fails the H100 confirmation |

- #1 and #2 are tied within judge resolution: #2's +0.03% is the first-turn rider. #1 ranks first because it can be
  checked now. #2 replaces it only if its H100 first-turn gate passes and an audit clears it.

## 1. Evidence base

**PFC1 on the judge.** These are the 3-read means; the master reference is its 4 same-code reads.
- Loads: turn-1 +0.88%, turn-4 +0.38%, first turn -0.06%, H +0.40%.
- In rho terms: rho_t1 +0.53%, rho_t4 +0.52%, rho_ft +0.06%, rho_all8 +0.29%.
- Rows: t1s +1.80%, t1f -0.73%, t4s +0.20%, t4f +0.84%, ftf0 -0.39%, fts -0.13%, ftf32 +0.44%, ftf64 +0.31%.
- Row-level noise is 0.3-1.2% per read. The routing is redrawn every run (reference.weights_fields): master t1s
  moe_fused ran 243-290 us.
- PFC1's H100 forecast, judge-scaled (K+C x0.9, RC x0.75), is H +0.45%.
- The blend I use is PFC1 = master +0.44% (E[H] 1.7861). That alone gives P(>board) 0.58.

**t1f residual.** The K+C reads pooled (5 reads, PFK2 + PFC1) put t1f at -0.66% vs master. That is -1.2 sigma, against
an H100 forecast of +0.5%.
- The judge profiles show t1f attn_fwd at 372-373 us for both master and PFC1, so CMERGE costs no wave or cluster
  penalty.
- CMERGE is worth about 2 us per call (public FACTS:141), so it cannot explain a 6 us deficit.
- Read: noise or routing. It is not a knob target (see §5).

**O2 on top of PFC1 (PFC2/PFC1 time ratio).** Builder: 3 runs, 12 copies pooled. Audit: interleaved twins, whole
rotate+reverse cycles.

| draw (t1s distinct) | harness | t1s | t1f | t4s | t4f |
|---|---|---:|---:|---:|---:|
| d0 (14; judge replica) | builder / audit | 0.9858 / 0.9885 | 0.9895 / 0.9915 | 0.9980 / 0.9984 | 0.9985 / 0.9982 |
| s33 (17) | builder / audit | 0.9825 / 0.9837 | 0.9907 / 0.9928 | 0.9963 / 0.9960 | 0.9978 / 0.9981 |
| d1 (8) | builder / audit | 0.9995 / 1.0031 | 1.0031 / 1.0061 | 1.0009 / 1.0015 | 0.9996 / 0.9992 |

**O2 alone on P2, 4 judge-like draws (seeds, knob1).** t1s 0.9865 / 0.9946 / 0.9878 / 0.9916; t1f 0.9920 / 0.9943 /
0.9933 / 0.9928; t4s 0.9958 (mean); t4f 0.9985 (mean). It is positive on every judge-like draw, and super-additive
with KPERM at d0 and s33.

**Where d1 sits.** d1 is a weights_fields draw that meets the router_load targets, but it has only 8 distinct experts
at t1s, with tiles {256:4, 128:2, 64:3, 32:2}.
- The P2 t1s row takes 540 us on d1 against 597-608 us on d0, on the H100.
- Every judge read of master t1s is heavier than d1: 566-601 us. The lightest, af53426a, is about 40% of the way from
  d0 to d1.
- So the model weights d1-like behaviour at 25%, and judge-like (d0/s33) at 75%.

**O2 increment, judge estimate.** I take the 75/25 mixture and multiply by 0.9 for transfer.
- Rows: t1s +0.97%, t1f +0.50%, t4s +0.16%, t4f +0.14%.
- Loads: turn-1 +0.80%, turn-4 +0.15%, first turn 0.
- H: +0.32%.

## 2. Per-row predictions (judge; % speed, + = faster)

| row | w | PFC1 vs master: judge 3-read / H100-scaled | #1 PFC2 Δ vs PFC1 | #2 PFC2L Δ vs PFC1 | #3 PFC2F/G Δ vs PFC1 | #4 PFC1H Δ vs PFC1 |
|---|---:|---|---|---|---|---|
| t1s | 46 | +1.80 / +0.86 | +0.97 (judge-like +1.15..+1.75; d1 -0.31..+0.05) | as #1 | #1 + δ_f | 0 (code-identical at T=753) |
| t1f | 16 | -0.73 / +0.63 | +0.50 (+0.72..+1.05; d1 -0.31..-0.61) | as #1 | #1 + δ_f | 0 |
| t4s | 46 | +0.20 / +0.36 | +0.16 (+0.16..+0.40; d1 -0.09..-0.15) | as #1 | #1 + δ_f' | as #1 |
| t4f | 16 | +0.84 / +0.69 | +0.14 (+0.15..+0.22; d1 +0.04..+0.08) | as #1 | #1 + δ_f' | as #1 |
| ftf0 | 16 | -0.39 / 0 | 0 (O2 off at T=16384) | +0.05 | 0 | +0.05 |
| fts | 138 | -0.13 / 0 | 0 | +0.10 (moevar fts O2 MoE -0.15..-0.34%, noisy) | 0 | +0.10 |
| ftf32 | 16 | +0.44 / +0.25 | 0 | +0.05 | 0 | +0.05 |
| ftf64 | 16 | +0.31 / +0.44 | 0 | +0.05 | 0 | +0.05 |
| loads t1 / t4 / ft | | +0.88 / +0.38 / -0.06 | +0.80 / +0.15 / 0 | +0.80 / +0.15 / +0.09 | +0.80+ / +0.15+ / 0 | 0 / +0.15 / +0.09 |
| **H vs PFC1** | | | **+0.32** | **+0.35** | **+0.32 + δ** | **+0.08** |

**Floors** (the rows each variant must hold). These come from master's 4 reads. KILL floor = row min / 1.025, the PFC1
gate's floor; it applies to every variant, on 2-read means and on the scored read. The targeted-row bar is a stricter
guard on 2-read means for the rows each variant changes. Missing it blocks GO (the result is HOLD); it is not a KILL,
because PFC1's own t1f 3-read mean (1.5864) sits below master's t1f min.

| row | master mean / min | KILL floor | targeted-row bar (2-read mean ≥ master min) | PFC1 3-read min |
|---|---|---:|---|---:|
| t1s | 1.8833 / 1.8595 | 1.8141 | #1 #2 #3 | 1.9016 |
| t1f | 1.5980 / 1.5888 | 1.5500 | #1 #2 #3 | 1.5726 |
| t4s | 1.8709 / 1.8498 | 1.8047 | #1 #2 #3 #4 | 1.8668 |
| t4f | 1.5245 / 1.5184 | 1.4814 | #1 #2 #3 #4 | 1.5201 |
| ftf0 | 1.8571 / 1.8324 | 1.7877 | #2 #4 | 1.8370 |
| fts | 1.8936 / 1.8741 | 1.8284 | #2 #4 | 1.8813 |
| ftf32 | 1.6522 / 1.6387 | 1.5987 | #2 #4 | 1.6377 |
| ftf64 | 1.5500 / 1.5440 | 1.5063 | #2 #4 | 1.5422 |

- t1f is the row to watch. PFC1's 3-read mean (1.5864) already sat below master's mean.
- O2 lifts t1f on judge-like draws and lowers it on d1-like ones.

## 3. Variants: change, generator, gates

**Judge gate metric (all variants):** rho = the geomean of per-row `paired_ratio` from `okc status <job> --json`
(result.records[] with event == "benchmark", field `paired_ratio`). It is **NOT the median of `ratios`**.
- rho_all8 is taken over the 8 rows, rho_t1 over {t1s, t1f}, rho_t4 over {t4s, t4f}, and rho_ft over the 4
  first-turn rows. "mean2" is the mean over the 2 check reads.
- References are computed from the same field:
  - master (4 reads): rho_all8 1.7219 (min 1.7098), rho_t1 1.7348 (min 1.7213), rho_t4 1.6888 (min 1.6827),
    rho_ft 1.7323 (min 1.7178);
  - PFC1 (3 reads): rho_all8 1.7270 (min 1.7166), rho_t1 1.7439 (min 1.7316), rho_t4 1.6976 (min 1.6875),
    rho_ft 1.7333 (min 1.7238).
- Per-read sd of rho_all8 is about 0.010, so a 2-read mean has SE about 0.007.
- At these effect sizes the checks are a route check and a harm guard, not a measurement.

**Common to every variant:**
- Route check, from each check's profiles/submission.<job>.txt:
  - t1s and t1f run moe_fused_persistent<0, 1> and no attn_combine2_kernel;
  - t4s and t4f run boundary_kernel<512, false, false, 1> + <512, true, false, 2>.
  - The work-queue order does not show in the profile.
- KILL also on any failed test, timed check, flag or rejection, or on prepare_s > 200 s.
- Landing (unchanged rule): the scored run passes 5/5 tests and every timed check, its headline beats 1.7836, and no
  row is below its KILL floor. Otherwise the variant gets a CLAIMS shelf line with all its reads.
- Compliance: each change is a work-queue order or threshold constant. The math is identical, the output is
  bit-identical to the base, and routing depends on shape only (T). Nothing is timed or cached in kernel_fn,
  load_inline is unchanged, and the attention and FA3 ruling are untouched.

### #1 PFC2 = PFC1 + MOE_ORDER_SMALL 1→2 (order 2 at T=753 and T=3012)

**Build.** `python3 -I campaigns/2026-10-31-public-plus/files/combos/mk_pf-combo.py muse_prefill_layer/submission.py PFC2 <n> files/combos/PFC2-r<n>.py`. The generator arm already exists; master is e5558cfe.
- Dry run today regenerates the audited bytes exactly: r1 3edc3f83df30c02043d4f453519824ee9e30a8ebb1ea6db30c76f5ead2acc697,
  r2 7002e4b5ab8c31536206d2d65dddd7f445e516c2274ccb43c6ed05b815470584.
- r3, for a fresh nonce if okc dedups the scored run: c58cee39654229831a97f6ddaa65200d700120df72f71292aad5aeca62945982.
- These are the same bytes the 11-01 audit passed on regeneration, diff, rules and H100 correctness (0 BAD, step 0
  clean). None was ever judged.

**Reopen basis.** The audit dropped PFC2 on GO-B alone. Two items missed:
- the d1 turn-1 cap: +0.43% against a 0.40% cap, on the 8-distinct draw that is lighter than every judge read;
- the turn-4 band, which it missed on the upside.

The audit called a reopen "a user call". The user's current steer is that call. Quote it verbatim in the attempt file.

**Mechanism.** Big 256/128-row tiles go first, with the 64/32-row tiles spread between them. This shortens the
small-tile tail of the persistent kernel, which matters most when KPERM has shortened every small item. It is
scheduling, so it transfers.

**Judge gate (2 checks PFC2-r1/r2, same hour). rho = geomean of per-row paired_ratio, as defined above, not the
median of `ratios`.**
- KILL if any of these holds:
  - mean2 rho_t1 < 1.7213 (below every master read on O2's target load);
  - mean2 rho_t4 < 1.6827;
  - mean2 rho_all8 < 1.7098;
  - any row's mean2 < its KILL floor.
- GO if mean2 rho_all8 ≥ 1.7270 (PFC1's 3-read mean) AND mean2 rho_t1 ≥ 1.7439 (PFC1's mean) AND every
  targeted-row bar (t1s, t1f, t4s, t4f) holds.
  - Then one scored run of the r1 bytes, or r3 if okc dedups.
  - Predicted: rho_t1 1.757, rho_t4 1.700, rho_ft 1.733, rho_all8 1.731, t1f about 1.594.
  - P(GO on 2 reads | prediction) ≈ 0.5. Most misses land in HOLD, which costs one more check, not in KILL.
    P(KILL | prediction) < 0.05.
- HOLD otherwise. Run one more check, r3, then apply GO on the 3-read means. If it still misses, shelf.

### #2 PFC2L = PFC2 + MOE_ORDER_LARGE 1→2 (order 2 at every T)

**Edit.** Applied to PFC2, it matches exactly once: `"\nMOE_ORDER_LARGE = 1\n"` → `"\nMOE_ORDER_LARGE = 2  # [SW] was 1: order 2 also at T >= MOE_LARGE_T\n"`.
- Proposed generator: `python3 -I files/combos/mk_pf-sweep.py muse_prefill_layer/submission.py PFC2L <n> out.py`.
- Dry run: r1 00e1e775600237962614727bffc5caac063586d49e5e1ccc5bc21952c9517be4, r2 4f000c9a… (§7).

**Prediction.** As #1 on the turn-1 and turn-4 rows, which run code-identical paths. The first-turn rows gain about
+0.05..+0.10%.

**H100 gate (§4; rho_H100 vs PFC2).**
- GO if all hold:
  - rho_H100(ft) ≥ 1.000 on d0 and on s33;
  - the fts row ≥ 0.999 on both;
  - the t1/t4 rows within ±0.25% of PFC2, an A/A check, since the code is identical there.
- KILL if fts < 0.997 on either draw.
- Between: drop the rider and keep #1.

**Judge gate.** It is #1's gate, with rho = geomean of per-row paired_ratio, not the median of `ratios`. Two
additions: KILL if mean2 rho_ft < 1.7178, and GO also needs every first-turn row's targeted-row bar to hold.

### #3 PFC2F / PFC2G = PFC2 with the order-2 spread fraction 4/5 → 1/1 (F) or 3/5 (G)

**What changes.** plan_warp parks the last fifth of the small tiles at the end of the queue.
- F spreads every small tile between the big ones. G parks two fifths.

**Edits.** Each matches exactly once, in the CUDA plan code; PFC1 lines 3099 / 3104 / 3105.
- `return k + min(S, (4 * k * S) / (5 * B));`
- `const int lim = (4 * S) / 5;`
- `const int bb = j < lim ? (5 * B * (j + 1) + 4 * S - 1) / (4 * S) : B;`
- Replace 4→P and 5→Q.

**Permutation check.** I checked the placement is a permutation of 0..B+S-1 for every B, S ≤ 96 and P/Q in {4/5,
3/5, 2/5, 2/3, 9/10, 1/1} (scratchpad sweep-pf/o2perm.py: 0 failures). So the plan stays valid and the output stays
bit-identical.
- These arms change the gather+plan kernel and need their own extension build. moe_fused_persistent is unchanged.
- Dry runs: PFC2F r1 6a55129a…, PFC2G r1 b24e7650… (§7).

**H100 gate (vs PFC2).**
- Adopt the better fraction only if all hold:
  - pooled judge-like rho_H100(t1) ≥ 1.003 with its CI excluding 1, and ≥ 1.003 on at least 3 of the 4 judge-like
    draws;
  - rho_H100(t1, d1) ≥ 0.997;
  - pooled rho_H100(t4) ≥ 0.999.
- Otherwise drop it. Prior δ is about 0 ± 0.3% on turn-1. This is a sweep of an order parameter that was never measured.

**Judge gate.** Identical to #1, with rho = geomean of per-row paired_ratio, not the median of `ratios`.

### #4 PFC1H = PFC1 + MOE_LARGE_T 8192→2048 + MOE_ORDER_LARGE 1→2

This puts order 2 at T=3012 and T=16384 only. T=753 keeps order 1.

**Edits.** Applied to PFC1, each matches exactly once:
- `'\nMOE_LARGE_T = 8192  # token count from which the "large" launch mode/order apply\n'` → `= 2048` (both modes are
  "fused", and MOE_LARGE_T is read only in _moe_forward);
- plus #2's MOE_ORDER_LARGE edit.
- Dry run: r1 19010f31a9ea95b2608ef28716497a5a8337235c7952c4085f836dd13cb15ef0.

**Role.** A routing-robust hedge. It keeps the half of O2 that is non-negative on every draw (turn-4, d1 included) and
drops the T=753 bet.

**H100 gate (vs PFC1).**
- GO if all hold:
  - pooled judge-like rho_H100(t4) ≥ 1.0015;
  - d1 rho_H100(t4) ≥ 0.999;
  - rho_H100(t1) within ±0.25%, an A/A check;
  - rho_H100(ft, d0) ≥ 0.999.
- Dispatch only if #1 fails: either its H100 confirmation (§4: pooled judge-like PFC2/PFC1 rho_H100(t1) < 1.004, or
  d1 < 0.993) or its judge rho_t1 KILL.

**Judge gate.** rho = geomean of per-row paired_ratio, not the median of `ratios`.
- KILL if mean2 rho_t4 < 1.6827, mean2 rho_ft < 1.7178, or any row breaks a floor.
- GO if mean2 rho_t4 ≥ 1.6976 AND mean2 rho_all8 ≥ 1.7219.

## 4. H100 session (one run, gpunode06 GPU 6 only, hopper-probe rules)

**Rules.** Run as the operator through ~/git/takemyjob/prototypes/hopper-probe, under its guard. No root, no Slurm
allocation, node-local /tmp/USER-hopper, and no change to node state.

**Harness.** Copy probes/pfcaudit to a new probe, pfsweep. pfcaudit is untracked in
~/git/.takemyjob-wt/combo-prefill_layer/prototypes/hopper-probe; it already has interleaved twins and whole
rotate+reverse cycles.

**Arms.**
- P2 (null) e5558cfe, PFC1 (control) dc5e02ae, PFC2 3edc3f83, PFC2L, PFC2F, PFC2G, PFC1H.
- Extension builds:
  - PFC1, PFC2, PFC2L and PFC1H share one extension: their knobs are on the Python side.
  - PFC2F and PFC2G each build their own: 3 builds in all.
- Step 0 for the F and G builds: ptxas -v and cuobjdump -res-usage. Plan/gather kernel registers with 0 spill;
  moe_fused_persistent unchanged at REG 168, LOCAL 0; no C7510-C7514 or C7602.

**Mandatory measurement protocol.**
- At least 4 byte-identical module copies per arm: 2 processes x 2 copies, or 1 x 4 if memory allows. That is about
  14 states x ~2 GiB per process.
- Load order shuffled per process, from a seeded permutation; the second process reverses it.
- Copy-pooled statistics: each copy's same-round ratio against the mean of the control arm's copies, pooled over every
  copy and both processes; median plus 95% bootstrap CI.
- GPU-bound CUDA-graph replay: the host enqueues the whole round ahead.
- A 256 MiB read flush before every call.
- Byte-identical copies differ by up to ~1 us from placement: 0.17% at t1s and 0.06% at t4s. So no single-copy result
  counts.
- A/A: the control's copy-vs-copy spread must sit within ±0.25% on the t1 and t4 rows. Otherwise extend the rounds.

**Draws.**
- Judge-like: d0 6110722306183770365, s11 7595436449497252824, s22 1427425565724231695, s33 5096244490851456549.
- Adverse: d1 399423751762061527.
- Rows: t1s, t1f, t4s and t4f at n=28 (whole cycles) on all 5 draws; ftf0, fts, ftf32 and ftf64 at n=14 on d0 and
  s33, for #2 and #4.

**H100 statistic.** rho_H100(X) = the geomean over rows in X of the copy-pooled median of (t_control / t_arm) from the
same round. The control is PFC2 for #2 and #3, and PFC1 for #1-confirm and #4. This is the H100 analog of the judge's
paired_ratio, measured against our own control instead of SGLang.

**#1-confirm (PFC2 vs PFC1).** It extends the existing d0/s33/d1 evidence to s11 and s22.
- CONFIRM if all hold:
  - pooled judge-like rho_H100(t1) ≥ 1.006, and ≥ 1.003 on at least 3 of 4 judge-like draws;
  - d1 ≥ 0.993;
  - pooled judge-like rho_H100(t4) ≥ 1.000.
- It does not block #1's checks: the existing evidence already meets these bars on d0, s33 and d1.
  - t1: d0 1.0125 / 1.0101, s33 1.0136 / 1.0119, d1 0.9987 / 0.9954 (builder / audit).
  - t4: d0 1.0018 / 1.0017.
  - A FAIL switches the next scored run to #4.

**Correctness.**
- check_implementation, eager and graph, on all 8 rows at d0 and d1 for each new arm.
- The 5 TASK tests.
- Outputs vs base under a shared lt_choice, within the master-null spread. Master itself is not bitwise run to run
  because of the atomic gather slots.
- 4k graph replays at t1f and t4f.

**Cost.** About 45-60 GPU-min.

## 5. Not proposed (exhausted or adverse; evidence)

| knob / idea | why not |
|---|---|
| BND_RC_MIN_T other thresholds | At the scored T, every value in (753, 3012] is the same file, so 2048 is optimal. A value ≤ 753 (RC at t1) costs t1s +0.76% and t1f +0.40% time (knob1 bndrc0, d0). A value > 3012 gives up RC's t4 gain, which is worth -0.47% time on t4s and -0.31..-0.38% on t4f (H100 knob1 / seeds). RC is also byte-class (judge x0.75). |
| RR tree off | RR is a latency/ILP lever worth -0.45% time on t1f and -0.4% on t4f (H100, PFC1 vs PFK2; PFK-PFK0 on 5 runs). Turning it off gives that up. |
| KPERM off + FINE_MIN_T 512 at T=753 (PFK-style) | FINE+O2 on P2 is no better than K+C+O2 on judge-like draws. It is the arm that flipped sign on d1 (PFK turn-1 -1.16%), while PFC2 stayed at or above P2 there. Dominated. |
| KPERM at T=3012 (KPERM_MAX_T + FINE_MIN_T 4096) | It loses the fine plan at t4 (public FACTS:117). The T=3012 kernel is tensor-bound (82-85% busy), so an instruction cut has little to work on. KPERM at 16384 cost 0.4-0.5% (public NEG §4). |
| KPERM + FINE at T=753 (moe_fused_persistent<1,1>) | Not a knob: it needs a new instantiation. Maybe +0.1-0.2%; a build for tomorrow. |
| CMERGE off, or only at sliding | It is worth about 2 us per call (FACTS:141). The t1f residual is 1.2 sigma and ~6 us, and the attention time matches master in the judge profiles. EV ≈ 0. |
| Attention splits, ATTN_SPLIT_MODE, stream-K | The t1f ns sweep: ns2 386 us; ns4 396; ns6 407; ns1 730. split0 costs +10% at t1f; sk132/264 cost +3.7/+5.7%. The t4 and ft rows stay at ns1 by wave math. |
| POOL_SM_RESERVE | The T=753 pooler is exactly 6x22 = 132 tiles, one wave. Reserving SMs adds a wave or a bigger tile (~+30% pooler) to save at most ~18 us of router chain. The public default is {} (measured). |
| ROUTER_SERIAL / ROUTER_LT / LT re-pins | rserial costs t1s +0.54% and t1f +0.37% time. LT picks do not carry from the H100 to the judge (Do-not-try). |
| TRITON_PDL, ATTN_PF_MAX_T, COMBINE_NW, CIT, NSLOT 3, MOE_SPLIT_REM | tpdl costs t1s +0.41%. pf ≈ 0 and is byte-class. The combine is bandwidth-bound glue. CIT asserts with KPERM (cit is None) and is off in the public default. NSLOT 3 is a cost only. SPLIT_REM is a measured default. |

## 6. Dispatch order (for the main thread; attempt file and rules checklist before each okc call)

1. **Now: #1 PFC2.**
   - Regenerate r1 and r2 into files/combos/ and verify the sha.
   - Write the attempt files, quoting the user's steer as the reopen basis and the gate above.
   - Send 2 checks; on GO, one scored run.
   - The quota in the ecf82580 record (the latest one) is 17 checks and 9 scored runs left today. It resets at 00:00Z.
2. **In parallel: the §4 H100 session** on GPU 6, followed by an independent audit of mk_pf-sweep.py and any arm that
   passes its gate.
3. **If #1's scored read is below the board, or #1 HOLDs:** the next candidate is whichever of #2 / #3 passed its H100
   gate (2 checks, then scored).
   - #4 is used only on a #1 H100-confirm FAIL or a judge rho_t1 KILL.
   - Never re-score identical code with only a new nonce.

## 7. Files (scratchpad, proposal only; nothing is committed)

- Generator: /tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-pf/files/combos/mk_pf-sweep.py
  (sha256 1b5e96a9f30311303178f5491e9d20cf2a1a5c08d21cd7aa7a48dc93034d2d83).
  - It wraps mk_pf-combo.build (307d4efb), which is unchanged. Arms: PFC2L, PFC2F, PFC2G, PFC1H.
  - Each edit is asserted to match once, and every output passes compile().
  - Its home would be files/combos/, next to mk_pf-combo.py, after an audit.
- Dry-run arms (full sha256), in /tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-pf/gen/:
  - PFC2-r1 3edc3f83df30c02043d4f453519824ee9e30a8ebb1ea6db30c76f5ead2acc697 (audited)
  - PFC2-r2 7002e4b5ab8c31536206d2d65dddd7f445e516c2274ccb43c6ed05b815470584 (audited)
  - PFC2-r3 c58cee39654229831a97f6ddaa65200d700120df72f71292aad5aeca62945982
  - PFC2L-r1 00e1e775600237962614727bffc5caac063586d49e5e1ccc5bc21952c9517be4; r2 4f000c9a16af7efc6b204915d4536a584875eb3458fec60392d6aae4132ef6a9
  - PFC2F-r1 6a55129ad2d0540896a3eaf6aaa0d6a8d727c4482ca28df54a900f0e88a72560; r2 a9a151d669a8755b39aa4a15315f4cc6841b6d15a8c7192bbb0a61b7eea497bc
  - PFC2G-r1 b24e76508b90caedd4d678bd21fd954d7dcf42dd23c9de3984ec2caf5a31fcff; r2 442daf45eae92b8d763450810417c93967bca97aeb31822bf14ec73c7a7e616e
  - PFC1H-r1 19010f31a9ea95b2608ef28716497a5a8337235c7952c4085f836dd13cb15ef0; r2 ac8e2bb7e96975f63552e9a37c108ad9ef912b4fb56e7330eb709fdeb4511a57
- Analysis scripts, in /tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-pf/:
  - rows.py: per-row paired_ratio and the judge headline;
  - rho.py: rho references;
  - prof.py and med.py: judge profile breakdowns;
  - o2perm.py: the spread-fraction permutation check.
