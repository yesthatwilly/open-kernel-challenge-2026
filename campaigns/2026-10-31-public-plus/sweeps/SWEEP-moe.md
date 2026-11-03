# SWEEP-moe: fresh-submission variants on master B4X (muse_moe_w4a16)

Design only, written 2026-11-01 ~20:40Z. Nothing was run on a GPU, nothing in the repo was changed, and no okc command was run.

- **Base:** master 75b795dd, `muse_moe_w4a16/submission.py` = `files/combos/B4X-r3.py`, sha256 `d4bcc1f7…`, 332,377 B.
- **Build split:** d1 runs TINY, v1 runs TINY2, d16 runs MID, d32 runs B4 (WGS + B4FIN), and v16 / v32 run MAIN (WGS).
- **Board:** 2.3867, a high M1a draw. B4X's 4 reads average 2.3850, and its pooled per-row reference is H0 = **2.3816** (below).
- **Gate metric, in every gate below:** rho(row) = geomean over the reads of that row's `paired_ratio` (okc status JSON, `result.records[]` with `event == "benchmark"`, field `paired_ratio`), divided by the row's reference. The headline is the geomean of the 6 rows' `paired_ratio`. **Not** the median of `ratios`.

## 0. Bottom line (ranked by expected judge gain x p)

| rank | variant | knob(s) (one build each) | rows touched | predicted judge effect if it works | E[gain] if works | p | EV | P(one scored read > 2.3867) |
|---|---|---|---|---|---|---|---|---|
| 1 | **V1 T116** | MAIN-only static-split cost: 1-n-tile phase-A stage 112 -> 116 (d32's B4 build keeps the public table) | v16, v32 | v16 -0.8 us (range -0.4 .. -1.2), v32 -1.9 us (range -1.6 .. -2.3) | +0.23% (+0.17 .. +0.30) | 0.55 | **~+0.11%** | ~0.48 (0.57 if it works, 0.38 if null) |
| 2 | **V3 T116+HF** | V1 + HF: TINY/TINY2/MID drop the WS helper's redundant end-of-kernel `__threadfence` | v16, v32 + d1, v1, d16 | V1, plus d1/v1/d16 0 .. +0.3% each (fence class, likely no transfer) | +0.25% | 0.55 / 0.3 | ~+0.12% | ~0.49 |
| 3 | **V2 T120B95** | MAIN-only: 1-n-tile phase-A cost 112 -> 120 **and** phase-B cost 100 -> 95 | v16, v32 | v16 -0.9 us (-0.3 .. -1.5), v32 -2.3 us (-1.8 .. -2.8) | +0.27% | 0.45 | ~+0.08% (larger downside) | ~0.46 |
| 4 | **V4 ZL** (robustness) | WGS `wgs_ranges()` returns empty ranges when a launch has no local route | none by design (MAIN + B4 differ only in a `__noinline__` function) | 0 | 0 | n/a | 0 (lands under the robustness rule) | 0.38 (= B4X) |

**Recommendation:**
- Screen V1, V2 and V3 together with B4X on the H100 (protocol in §5).
- Send **2 checks of the single screen winner**. Default: V1, with ZL riding on it if the zl.py test in §5 confirms the fix.
- On a judge GO, make one scored run.

Every variant is new code. r1, r2 and r3 differ only in the line-2 nonce, as for B4X; this is not a re-roll of judged bytes.

Why only about +0.1 to +0.3%:
- The moe lever space is nearly mined out. The round-2 finder (LEVERS-1031b) found no lever above 0.5%. GAPMAP puts today's engine ceiling near x1.07.
- B4X itself is worth about +0.15%.
- Expected B4X headline: about 2.383 ± 0.011 per read, so P(> 2.3867) is about 0.38 with no change.

## 1. The user's ask, row by row: B4FIN placement on d16 / v16 / v32 is already measured as harm

B4FIN placement was measured on the H100 before B4X shipped (combo-moe report, scratchpad `combo-moe_w4a16.md`; run 20261101-100015-gpunode06, arm B4ALL = `mk_b4x.py --mid 15 --b4max 256`). S = t(master)/t(arm), with 3 processes pooled:

| row | B4FIN placed: R67 [95% CI] | cold | why |
|---|---|---|---|
| d16 | **0.9277** [0.9244, 0.9324] | 0.9596 | The WGS kernel loses to MID at M=16 by 4-7%; plain WGS (W16) is 0.916 / 0.950. MID's final is already a one-pass column-parallel FINCOL. NF (no final at all) reads x1.000 at d16, so a better final has **zero ceiling** there. |
| v16 | **0.9868** [0.9852, 0.9881] | 0.9934 | 512 quarter-row items on 132 CTAs means 3.9 items per CTA in sequence, plus the epilogue partials. That is why the public code limits B4FIN to M ≤ 32. |
| v32 | **0.9879** [0.9868, 0.9898] | 0.9906 | 7.8 items per CTA in sequence. |
| d32 | 1.0078 / 1.0085 / 1.0084 | 1.0080 / 1.0072 | Kept: this is B4X. |

Placement is a pure scheduling lever, and the d32 gain transferred about 1:1 (judge 4-read rho_d32 about +1.1%). The v16/v32 harm should therefore transfer too. **Do not re-screen these arms.**

A row-item "B4R" final (B4FIN partials plus one token per CTA at v16/v32) was considered and **not built**:
- It needs a 6th build. Judge prepare is 215.9-223.5 s for B4X's 5 builds, each extra build adds about 45 s, and the cap is 300 s.
- The alternative, a runtime M ≤ 256 guard inside `wgs_epi_tile`, hits the 96-register codegen trap: FIN2 and PBLL lost 2-5% that way.
- Its ceiling is small. F1 (one L2 pass) read 0.0% / +0.07% at v16 / v32. C2's one-wave final body cost 2.9-4.6 us.

The other arms in the ask:
- **B1XN** is attention and **RBX-P1** is bf16. There are no moe bytes to combine, and their HOLD rows (d1s, v1f; P1) are not moe rows.
- A moe version of RBX's prepare-time race (refit table vs public table) would need a 6th build, which does not fit in prepare. Plan races have not transferred so far (BFR, DL2H).

## 2. Why V1 / V2: the WGS static split has a systematic per-class bias at v16 / v32, and d32 must keep its table

**How MAIN splits work.** MAIN (WGS) gives every CTA an equal share of a cost-weighted stage list (`wgs_range`), with stage cost = `OKC_WGSA[n-tiles-1]` for phase A and `OKC_WGSB[...]` for phase B. Jobs are ordered fat first (experts with more than 8 tokens, in chunks of at most 64 tokens, mostly 2-8 n-tiles), then skinny (≤ 8 tokens, 1 n-tile). Low CTA indices therefore get the fat stages and high indices the 1-n-tile stages.

**What the stamps show.** I re-analysed per-CTA `%globaltimer` stamps already on disk (no new GPU work):
- R67: lev-moe run 20261031-050519-gpunode06 `levmoe-r6t/res.json`, 16 reps per row.
- Full bytes: run 20261031-012835-gpunode06 `levmoe-tlraw`, board bytes.

| row | regime | kappa_A = phase-A time per predicted cost, 1-tile region / fat region (sd) | kappa_B | B_done by CTA-index sextile, first -> last (us) | corr(CTA index, B_done) |
|---|---|---|---|---|---|
| d32 | R67 | 1.031 (0.021) | 0.913 | 119.2 … 119.2 (flat) | -0.04 |
| v16 | R67 | **1.068** (0.018) | 0.926 | 152.6 -> 155.4 | +0.40 |
| v32 | R67 | **1.075** (0.010) | 0.950 | 177.3 -> 182.4 | +0.60 |
| d32 | full bytes | 1.012 | 0.922 | flat | |
| v16 | full bytes | **1.070** (0.021) | 0.937 | 209.7 -> 213.0 (peak 214.5 in sextile 4) | |
| v32 | full bytes | **1.102** (0.028) | 0.962 | 235.1 -> 245.6 | |

The 1-n-tile region runs about 7% (R67) to 10% (cold) over the table in phase A. Phase B compensates only in part, so the CTAs that finish last, and set the grid barrier, are systematically the high-index ones at v16 and v32.

- The x-gather join is flat across the index, so the bias comes from the phase work itself.
- The bias appears in both regimes, so it is not an R67 artefact.
- The stamps are on M1a bytes. B4X's MAIN differs from them only by MD16F's F fence, which is judge-neutral, and has the same tables.

**Model.** Script `sweep-moe/refit_sim2.py`. Per rep it fits the per-CTA phase times to a fat/skinny two-class cost model, keeps each CTA's (A, B) residual pair (A and B residuals correlate at -0.4 to -0.65), permutes the pairs, re-splits under a new table, and reads off the change in max B_done. Results, in us, as [phase-B bias real / phase-B bias an artefact]:

| table (MAIN) | v16 R67 | v32 R67 | v16 cold | v32 cold | d32 R67 if applied there | d32 cold if applied there |
|---|---|---|---|---|---|---|
| T116 (A 116) | -0.39 / -1.20 | -1.64 / -2.34 | -1.07 / -1.69 | -3.10 / -3.23 | +1.27 / -0.13 | **+2.71** / +0.33 |
| T120B95 (A 120, B 95) | -0.26 / -1.46 | -1.77 / -2.83 | -0.99 / -2.07 | -3.77 / -4.03 | (higher) | **+3.56** / +0.87 |

Pessimistic bound: if every CTA keeps its own residual even after its range moves (a worst case, not a real model), T116 reads +0.26 / +0.78 us at v16 / v32 and T120B95 reads +1.7 / +1.4 us. This sensitivity is why the H100 screen decides.

**Consistency with the public results.**
- The public team's NEGATIVE_RESULTS moe #2(d) says a refit per-n-tile table "halved the phase-A duration spread on held-out seeds but was slower by 0.8 to 4.6 us on the d32 judge cases". The model reproduces that d32 harm (+1.3 to +3.6 us).
- d32 shows no systematic gradient, and its kappa is about 1.0.
- The public table is one compromise across bs 32/128/256: "tuner combo_m798 … bs32 +0.45, bs128 +0.94, bs256 +0.17%".
- B4X now gives d32 its own build. V1 and V2 therefore refit **MAIN only**: d32 keeps the public table, and its B4 build preprocesses to unchanged code.
- Lever class: static-split balance and critical path, so it is scheduling and should transfer. Its size depends somewhat on the byte/compute balance; kappa is 1.075 in R67 and 1.10 cold, and the H200 should be about R67.

## 3. Variant specs

**Generator** (written and dry-run tested, CPU only): `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-moe/mk_sweep_moe.py` (sha256 6a6f8e6e…).
- It asserts the input is master B4X (`d4bcc1f7`).
- Edits are anchored, exactly once, inside the `CUDA_SRC` slice; the helper0 copy is edited only for `--hf`.
- It strips ` pp-nonce B4X-r3` and appends the change note plus `pp-nonce <NAME>-r<n>` to line 2. Lines 1 and 3-9 are asserted unchanged, and the nonce arms differ only on line 2. Each file compiles as Python and stays under 1 MB.
- The builder copies it to `campaigns/2026-10-31-public-plus/files/combos/mk_sweep_moe.py` on its branch.

```
git -C /home/user/git/musespark show master:muse_moe_w4a16/submission.py > B4X.py   # sha256 d4bcc1f7...; NOT the public-plus worktree copy (older)
python3 -I mk_sweep_moe.py B4X.py <out> --name T116    --tab 116           --nonces 1,2,3   # V1
python3 -I mk_sweep_moe.py B4X.py <out> --name T116HF  --tab 116 --hf      --nonces 1,2,3   # V3
python3 -I mk_sweep_moe.py B4X.py <out> --name T120B95 --tab 120b95        --nonces 1,2,3   # V2
python3 -I mk_sweep_moe.py B4X.py <out> --name ZL      --zl                --nonces 1,2,3   # V4
#   riders: add --zl to the winner, e.g. --name T116ZL --tab 116 --zl (only if the section-5 zl.py test passes)
```

Dry-run sizes (`sweep-moe/out/`, for reference only; rebuild in the worktree): T116 332,862 B; T120B95 332,902 B; T116HF 333,376 B; ZL 332,644 B. All 8 + 8 `"r"(UNIT_BYTES)` R67 sites are kept, so `probes/b4x/mkr67.py` works unchanged.

### V1 T116 (rank 1): one knob, MAIN only

Exact edit, inside `CUDA_SRC`. The two-line anchor occurs once; the single line also appears inside the helper0 repr, so anchor on both lines:

```
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
```
becomes
```
#if OKC_B4FIN  // [T116] the d32 build keeps the public table (a refit read slower at d32: public NEG moe #2(d))
<the two lines above, unchanged>
#else  // [T116] MAIN (v16 / v32): 1-n-tile phase-A stages run ~7% over the table vs fat stages (H100 R67 per-CTA stamps)
#define OKC_WGSA 116, 135, 140, 158, 173, 185, 200, 207
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#endif
```

**Build effects.**
- `OKC_B4FIN` is defined at the top of CUDA_SRC (`#ifndef … 0`), so MAIN takes the `#else` branch and B4 (`-DOKC_B4FIN=1`) takes the original.
- TINY, TINY2 and MID compile `ALT_SRCS["helper0"]`, which is untouched, so their .so files are byte-identical.
- MAIN's SASS changes only in the packed cost immediates (`WGS_PA`).
- B4's SASS is identical; only the `-lineinfo` line table shifts.
- No Python edit, no new build. Prepare time equals B4X's.

**Predicted judge effect:**
- v16 -0.8 us, +0.46% of the row (range -0.4 to -1.2).
- v32 -1.9 us, +0.9% of the row (range -1.6 to -2.3).
- d1, d16, d32 and v1: 0 by construction.
- Headline +0.23% (+0.17 to +0.30). Pessimistic bound -0.09%.

**Floors (2-check geomean paired_ratio):**
- Touched rows: v16 ≥ 2.6191 and v32 ≥ 2.9094 (0.990 x R0).
- Untouched rows inside [0.970, 1.030] x R0: d1 [2.9458, 3.1280], d16 [1.9814, 2.1040], d32 [1.6306, 1.7314] (B4X's improved d32 must hold: by construction), v1 [2.1830, 2.3180].
- Mean headline ≥ 2.3739.

**Compliance:**
- Only compile-time constants of the static split change. The stream-K partition moves, so only the FP32 summation order of split units changes, which is the same class as any split.
- FP32 accumulate, BF16 out, weights untouched (lossless), kernel_fn untouched (shape-only routing).
- No timing or caching, load_inline only, notices kept, under 1 MB.

### V3 T116+HF (rank 2): V1 plus a rider on the skinny rows

**The HF edit** is `files/moe/mk_hf.py`'s FENCE_OLD -> FENCE_NEW, applied in `ALT_SRCS["helper0"]` and in CUDA_SRC.
- Since MD16F's D6, TINY, TINY2 and MID all compile helper0.
- The CUDA_SRC copy sits inside `#if OKC_WS` (lines 1599-1796), which is compiled out in MAIN and B4.

**Evidence:**
- H100, A/A-corrected: d1 -0.02%, d16 +0.18%, v1 +0.45% (wf_514e7d34-141, 603f5581).
- R67: d1 +0.3%, v1 -0.1%, d16 +0.2% (lev-moe 20261031-050756).
- Earlier verdict: KILLED at its standalone d1 ≥ +0.5% gate (CLAIMS "[node-only] HF").
- As a fence lever it most likely does not transfer, so its EV on top of V1 is about 0 to +0.02%. It rides only because its rows do not overlap V1's.

**Auditor must re-check under D6.** With the TIDX remap (`TIDX = (threadIdx.x + NT) % NTL`), logical tid 0 is physical thread 32, a compute thread. The chain is: helper stores -> WS_JOIN (bar.sync 2, includes the helper) -> CTA_SYNC -> logical tid 0 `__threadfence` + barrier atomicAdd. Re-run the HF stale-read stress on the B4X base.

**Floors:**
- V1's floors.
- d1 rho ≥ 0.990; v1 and d16 rho ≥ 0.985 (their per-read paired sd is 1.30% / 0.90%).
- d32 inside [0.970, 1.030].

### V2 T120B95 (rank 3): two knobs, MAIN only

Same edit as V1, with the `#else` branch set to `OKC_WGSA 120, 135, …` and `OKC_WGSB 95, 104, …`. This is the full phase-A correction plus phase-B compensation.

- **Prediction:** v16 -0.9 us, v32 -2.3 us, headline +0.27%. Pessimistic bound -0.27%. More upside and more downside than V1, and further from the judge-tuned table.
- **Floors:** as V1.

### V4 ZL (rank 4; robustness, speed-neutral)

**The bug** (CLAIMS 11-01 "[unresolved] moe MAIN zero-local-routes hang", lane-c2 run 20261101-034319): master hangs at bs 17, 33, 64 and 257 when no route is local.

**Hypothesised mechanism** (to be verified, §5):
- With nj = 0, `wgs_stage_at` reads the unset `S.jC[0]`. A stale 0 gives `wgs_cost = 0`, and the 0/0 division gives -1, so CTA G-1 gets the range [-1, 0).
- Its x producer then spins on `xready[-1] < npad(garbage)`.

**Edit** (CUDA_SRC, anchor = the first 3 lines of `wgs_ranges`). Insert after `wgs_prefix(S, nj);`:
```
  if (nj == 0) {  // [ZL] no local route in this launch: empty ranges (wgs_stage_at would read the unset S.jC[0])
    S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;
    return;
  }
```

**Effects.**
- `wgs_ranges` is `__noinline__` and runs once per CTA on one producer lane.
- MAIN and B4 SASS should be identical apart from that function. The skinny builds are untouched.
- Output for a zero-local launch is all zeros, which equals the reference.
- The judge never builds such a launch, because its routing makes this GPU the busiest.

**Value.** It removes a real hang. Under the user's rule (audited robustness fixes land even inside speed noise), it makes the fresh file landable on its own merits.
- **Floors:** every row in [0.970, 1.030] x R0, and mean headline ≥ 2.3697.

## 4. Judge gate template (for the auditor; pre-data)

**Metric.** rho(row) = geomean over the 2 checks of that row's `paired_ratio` (okc status `result.records[]` event=benchmark, field `paired_ratio`) ÷ R0(row). This is not the median of `ratios`.

**References R0** (pooled reads of byte-identical row code):
- 7 reads (MD16F 1ab2de4b / 483bcb8e / 3f9df6f8 and B4X e8bb8ad0 / eb8c11a1 / 7776cad9 / 299e8e6e) for every row except d32: d1 3.0369, d16 2.0427, v1 2.2505, v16 2.6456, v32 2.9388.
- d32: 1.6810, from the 4 B4X reads.
- H0 = 2.3816.

**Gate, 2 checks (r1/r2):**
- **VALID:** both checks pass 5/5 tests and 6/6 rows, and every untouched row has rho in [0.970, 1.030].
- **GO:** G = Σ_touched ln(rho)/6 ≥ +0.10%, AND every touched rho ≥ its floor (§3), AND the 2-check mean headline ≥ 2.3739 (H0 - 0.32%).
- **KILL:** G < 0, OR any touched rho < 0.980, OR mean headline < 2.3697.
- **BETWEEN:** shelve. No third check, and no post-hoc subset.
- **On GO:** one scored run of r3. Land if valid (5/5 + 6/6). The board moves only if the read is > 2.3867.

**Power.** For V1, 2-check sd(G) is about 0.12%, so P(GO | works) ≈ 0.86 and P(GO | null) ≈ 0.19. A false GO only spends one scored run on a file whose expected value equals B4X's.

## 5. Builder: H100 screen (gpunode06 GPU 5 only) and correctness

**Harness.**
- Use the existing moe placement harness: takemyjob worktree `/home/user/git/.takemyjob-wt/combo-moe_w4a16`, `prototypes/hopper-probe/probes/b4x/` (`levmoe.py` / `b4.py` / `run-b4x.sh` paired AB with `--board`, `mkr67.py`, `zl.py`).
- Parties: B4X base, a second copy of the base for the A/A arm, T116, T120B95 and T116HF (plus ZL for SASS and A/A only).
- Each party needs **≥ 4 byte-identical module copies**, shuffled load order and copy-pooled statistics.
- Timing: GPU-bound graph replay with the host enqueued ahead, and a read flush before every call.
- All 6 rows: bs 1, 16, 32, 8, 128, 256.

**Regimes.** R67 is primary (`mkr67.py` copies; this is the validated H200 proxy for the WGS rows). Cold full bytes is the sanity check. Byte-identical copies already differ by up to about 1 us from placement, so judge only on pooled copies.

**Gates, pre-registered before the first timing run** (S = t(base)/t(variant), median of paired per-rep ratios, 95% bootstrap CI):
- **VALID:** A/A |S - 1| ≤ 0.3% pooled on v16, v32 and d32. Repeat once if not.
- **T-part GO (V1 / V2 / V3):**
  - R67 S(v32) ≥ 1.006 with CI lo > 1.000, AND
  - R67 S(v16) ≥ 0.998, AND
  - cold S(v16) and S(v32) ≥ 0.997, AND
  - R67 headline gain 0.095·Δt(v16) + 0.08·Δt(v32) ≥ +0.08% (Δt in us, faster positive).
- **T-part KILL:** R67 S(v32) < 1.002, or any touched row < 0.995 in either regime.
- **Choosing between V1 and V2** when both pass: take the larger R67 headline gain; on a tie take V1, the smaller move.
- **HF part (V3):** d1, v1 and d16 each need R67 S ≥ 1.000 pooled and cold ≥ 0.995. Otherwise drop HF, and V3 becomes V1. d1 has a known R67 A/A bias of about 1% (CLAIMS); use swapped party order and ≥ 1.5% before claiming any d1 gain.
- **Untouched rows** (d1, d16, d32 and v1 for V1/V2) are SASS-identical by construction. Assert this rather than time it. Flag, but do not reject on, an untouched row outside ±0.5%.

**Mechanism check** (cheap and high-information). Re-run the lev-moe `tl` stamp mode on the R67 copies of the base and of T116 (rows 128 and 256, 16 reps). Expect:
- the last-minus-first sextile B_done gap to drop from about +2.8 (v16) and +5.1 (v32) us to ≤ +1.5 us;
- corr(index, B_done) to fall below 0.2;
- no rise in the fat-region B_done larger than the gain.

**Correctness for each variant** (B4X audit set):
- check_implementation, eager and graph with fresh inputs, on:
  - the 6 rows x 2 seeds and the 5 TASK tests;
  - bs 9-17, 20, 24, 31, 32, 33, 48, 64, 96, 128, 160, 192, 224, 255, 256, 257, 385 and 386, normal and skewed;
  - hot routings at bs 16, 32, 128 and 256.
- 4000 fresh-input graph replays at bs 128 and 256, with an L2 flush every 4th replay.
- SASS: TINY, TINY2 and MID identical to B4X (V1/V2/V4); B4 identical (V1/V2/V3); MAIN differing only in the cost immediates.
- No spills, 96 registers.
- Prepare time within 5 s of B4X's.

**ZL** (`probes/b4x/zl.py --files <B4X>,<ZL> --bs 17,24,32,33,64,128,257 --timeout 240`):
- B4X must HANG on at least one bs; this reproduces the bug.
- ZL must print OK with max|out| = 0 at every bs. Kill a hung child process only; never touch node state.
- If B4X does not hang, or ZL also hangs, ZL is unverified: drop it and record the result.

## 6. Files

All in `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/sweep-moe/`:
- `mk_sweep_moe.py` (generator);
- `out/` (dry-run files; regenerate in the worktree);
- `refit_sim.py`, `refit_sim2.py`, `refit_grid2.py`, `refit_grid_cold.py`, `refit_d32.py` (refit model; inputs are the lev-moe stamp runs cited in §2);
- `routesim.py` (routing statistics);
- `runs/` (judge JSONs for the 7 reference reads).

Supporting docs:
- the B4FIN placement data: scratchpad `combo-moe_w4a16.md`;
- the round-2 finder (dead ends, latency ceilings): master `campaigns/2026-10-31-public-plus/research/levers-1031b.json`;
- the public negative results: `/home/user/git/okc-muse-spark-kernels/docs/NEGATIVE_RESULTS.md` moe #2.
