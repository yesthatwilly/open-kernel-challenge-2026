# SWEEP-dec2: moe C1 engine ported into decode's T <= 8 MoE (design only, read-only, 2026-11-02 ~01:00Z)

Base for every variant: master DSTK `muse_decode_layer/submission.py` sha256 1c653e3f... (860,349 B). Every edit is one
append-only overlay with count-asserted anchors (mk_swdec.py style); line 2 gets a provenance clause and a nonce. Closed
and not reopened: CG-O / CG-U / O8 / CPUSH-DEC (DX2 84b5da1a), SWEEP-dec V2 / V3 / V4. V1K4 is pending (user override);
the C1 overlay touches disjoint edges (moe + unpooler launch vs gle_o / gle_pr), so it rebases onto V1K4 if that lands.

## 0. Answer

- **C1 found.** Engine: branch `campaign/c1-moe`. Commit eb0eafbb is design (a) + J1-J8 under `#if OKC_C1`. Commit
  016e301e adds the incremental lane-0 refill and drops the per-refill proxy fence. Design:
  `/tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/C1-DESIGN.md`.
- **The moe-side evidence is H100 only, and C1 was never judged.**
  - M0 (takemyjob hopper/c1m0 30d9493; verify hopper/c1m0v ef998c1, run 20261031-195402) was a warm-L2, R50
    microbench with pinned routing:
    - C1-12 at N=1: AB 1.239, K 1.248-1.251 (the GO-TINY bar was 1.25);
    - C1-8T2A4 at N=8: AB 1.142, K 1.125;
    - C1-M at N=16: 1.87 (phase A only).
  - Verdict BETWEEN, no M1. CLAIMS moe :143 [lead], :146 [unresolved].
  - Two judge calibrations bound its value: DX23 r(d1) = 1.026 (skinny rows are co-bound), and MD16F (the +5-7%
    consumer rider D1+D6 read d1 0.9929, v1 1.0027 on the judge, i.e. neutral).
  - **So C1's consumer-rate gain has low judge elasticity at T <= 8.**
- **The port is mechanical.** A dry run in scratch (`c1port/dryport.py` -> `c1port/moe_c1_dry.cu`) placed 14/14 C1
  CUDA hunks on decode's `moe_helper0`:
  - 11 apply verbatim; 3 are re-anchored (the Smem tables, the FINCOL first item, the FINCOL loop);
  - the refill fence and the kernel rename are added; +16,358 B of CUDA.
  - C1 goes into a separate `@moe_c1` source, so master's TINY / TINY2 / MID / B4 / MAIN modules stay byte-identical.
- **New: C1 is worth more as a scheduling enabler than as a faster consumer.**
  - C1-8 with D=2 at M=1 ("C1S") needs ~103 KB smem, 256 threads and 88-92 registers. Master TINY needs 210 KB,
    288 threads and 145-149 registers.
  - That leaves room for the dependent unpooler (sk2 `slice8_r192`: 416 threads, ~115.5 KB) to become resident beside
    the running moe CTA and stream its 6.29 MB before its `griddepcontrol.wait`.
  - The unpooler is d1's post-moe critical path (SKIPF; H200 chain = solo: 3.62 / 3.65 us at d1s).
  - This lever is scheduling / critical-path class, so it transfers.
  - Feasibility hinges on the unpooler's register count (it must be <= 96 per thread). That is unmeasured, so STEP 0
    decides it.
- **Several rates are unmeasured, so rung D0 runs first (§4):**
  - (r1) C1's in-situ MoE delta inside decode's chain;
  - (r2) an R-proxy calibration: the H100 at full bytes cannot see consumer gains at d1 (H100 r(d1) 1.28 vs H200
    1.026), so the proxy must be shown to emulate the H200;
  - (r3) co-residency, measured as f_co, with a padded-smem causal control;
  - (r4) the value of the two scheduling fixes the M0 metric cancelled: F1 (pre-issue from the touched-expert mask)
    and PBE (phase-B weights before the CTA barrier).
- **Ranking by expected judge gain x p.** Headline effect is central H200; p = p(STEP 0) x p(screen) x p(judge | screen).

| rank | variant | config + knobs | rows | central H200 if it works | p | EV (headline) |
|---|---|---|---|---|---|---|
| 1 | **V1 C1S-A4** | c8s + TRIG after phase A + UNP1 = 4 | d1s d1f | d1s -1.3 us, d1f -0.6 us; I +1.6%, H +0.27% | 0.5 x 0.5 x 0.55 = 0.14 | **+0.038%** |
| 2 | **V2 C1S-B4** | c8s + master trigger + UNP1 = 4 | d1s d1f | d1s -0.8, d1f -0.4; I +1.0%, H +0.17% | 0.5 x 0.35 x 0.55 = 0.10 | +0.017% |
| 3 | **V5 stack** | best d1 survivor + V4's t2 | d1 + v1 | V1 + V4: H +0.41% | 0.14 x 0.3 = 0.04 | +0.017% |
| 4 | **V3 C1-12** | c12 at M <= 4 | d1s d1f | d1s -0.4, d1f -0.4; I +0.6%, H +0.10% | 0.25 x 0.4 = 0.10 | +0.010% |
| 5 | **V4 T2** | t2 at 5 <= M <= 8 | v1s v1f | v1s -1.0, v1f -0.7; H +0.14% | 0.2 x 0.35 = 0.07 | +0.010% |

- **Odds.**
  - P(at least one survivor of the H100 screen) ~ 0.5.
  - P(a C1 variant lands on master) ~ 0.15.
  - P(the decode board moves today via this design) ~ 1-2%: it needs a scored read > 2.2220, which is +1.5% over
    master's 4-read 2.1892, and V1's central is +0.27%.
- **If nothing survives the screen:**
  - no decode checks;
  - CLAIMS decode gets "[refuted] C1 port (consumer)" and/or "[refuted] C1S co-residency" with the D0 numbers;
  - the checks go to other boards.

## 1. Inputs

### 1.1 C1 source and moe-side evidence (all read in this pass)

| item | where | content |
|---|---|---|
| engine | campaign/c1-moe eb0eafbb, 016e301e (diff vs 68e99e7d = moe b95f9537, sha c737e460) | +352 lines in moe CUDA_SRC under `#if OKC_C1`: row-owning warps, private per-warp TMA rings over a K-slice-major copy (w[14] / w[15], 4,608-B stages), x in smem (pitch 8,208), J4 phase-B x by TMA after the doneA acquire + `fence.proxy.async.global`, J5 deterministic range-edge fixups, J6 small routing tables, J7 final on 256 threads; `C1_ON = False`, `_stage_order()` |
| M0 numbers | hopper/c1m0 `results/time-20261031-193943-sum.txt`, `tune-...-sel.txt`; hopper/c1m0v `verify-sum-20261031-195402.txt` | N=1: C1-12 AB 1.2391 / K 1.2493; C1-8 1.116 / 1.075; C1-8D2 1.103 / 1.075; INC-D 1.052. N=8: C1-8T2A4 1.141 / 1.123; INC-D 1.068. N=16: C1-M A 1.86. A/A 1.000. Equal-work 78/78 |
| census | `census-20261031-193459.json` | registers: C1-8 / C1-12 / C1-8T2 = 92 / 94 / 92 (R50 builds), 88 (full bytes). INC 147, master TINY 149. C1 stage loop 557 instr per 32 HMMA (the J3 gate of 535 was already exceeded in M0) |
| judge calibration | moe CLAIMS :141 (DX23 r(d1) 1.0256 [1.0188, 1.0343]); :148 MD16F (D1+D6: d1 0.9929, v1 1.0027, S 0.9994) | skinny rows co-bound; a +5-7% consumer rider did not move the judge |
| decode port notes | C1-DESIGN §6 | paste into `moe_helper0`; PDL: x stays after `griddepcontrol.wait`; decode MoE module names hash their source |

### 1.2 H200 path map and per-edge floor forecasts (DX2 84b5da1a GAPDIAG; master's timed path)

| row | edge | chain / solo us | stream floor (4.6 TB/s) | gap | targeted by |
|---|---|---|---|---|---|
| d1s (loc = 3) | moe | 27.50 / 27.10 | 18.5 (3 experts x 28.31 MB; 21.2 at 4.0 TB/s) | 6.3-9.0 (latency: route, A->B, final, tail) | V3 (coupling), V1 / V2 (tail) |
| d1s | un | 3.62 / 3.65 | ~1.2-1.5 post-moe if its weights are already in smem | 2.1-2.4 | **V1, V2** |
| d1f (loc = 3) | moe / un | 27.58 / 27.50; 2.90 / 3.71 | same / ~1.2-1.5 | 6.3-9.0 / 1.4-1.7 | V1-V3 |
| v1s (loc = 11) | moe | 66.27 / 66.78 | 57.9 (~9.4 distinct experts) | ~8.5 (co-bound coupling) | V4 |
| v1s | un | 3.12 / 3.84 | no co-residency possible at M = 8 (§8) | - | - |

- **Value per us (%H):** d1s 0.1776, d1f 0.0618, v1s 0.1130, v1f 0.0393 (DSTK row means).
- **Each judge read draws a new seed per row** (4 distinct seeds per row in the 4 reads), so loc varies read to read.
  That is part of the per-row sd below.

### 1.3 Gate metric and the improved floor forecasts

**Gate metric (stated for every gate below):** rho_row = (geomean over this variant's checks of that row's
`paired_ratio`, read from the okc status JSON `result.records[]` with `event == "benchmark"`) / R0_row. It is **NOT**
the median of the per-rep `ratios` list. Index I = sum(w_r ln rho_r) / sum(w_r) over the touched rows (w = 46 sliding,
16 full). The okc headline is only a no-harm guard: master's 4-read mean is 2.1892, so the guard is >= 2.1564.

**Pools.**
- d1s, d1f, v16f, v32f: DSTK's 4 reads 5a84a880, 437f3be3, 9185f31c, 84b5da1a (DSTK changed these rows).
- The other 8 rows: those 4 plus D0FIX's 3 (0bd3ae2a, 7167c064, 545b80bb). DSTK is SASS-identical to D0FIX on them.
- When V1K4's read lands, its 10 rows other than d1s / d1f join the pools.
- Recomputed from the JSONs by scratch `c1port/floors.py`; the 4-read R0 reproduces SWEEP-dec §1 exactly.

**Floors.** They are noise-scaled per row instead of a flat 3%, with z = 2.5:
- 2-check floor = R0 exp(-2.5 sd sqrt(1/2 + 1/n));
- scored-read floor = R0 exp(-2.5 sd sqrt(1 + 1/n)).
- Quiet rows get tighter guards. Noisy rows (d32, v16, which caused the HSV1 d16s and CPUSH d32s letter-misses on
  SASS-identical code) get looser ones.

| row | pool (n) | R0 | sd/read | 2-check floor | scored-read floor | flat R0/1.03 | us | %H/us |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d1s | DSTK (4) | 2.4917 | 0.37% | 2.4719 (-0.8%) | 2.4662 (-1.0%) | 2.4191 | 68.4 | 0.1776 |
| d1f | DSTK (4) | 2.4731 | 0.65% | 2.4387 (-1.4%) | 2.4288 (-1.8%) | 2.4011 | 73.0 | 0.0618 |
| d16s | +D0FIX (7) | 2.0378 | 0.55% | 2.0154 (-1.1%) | 2.0080 (-1.5%) | 1.9784 | 129.7 | 0.0827 |
| d16f | +D0FIX (7) | 2.1065 | 0.51% | 2.0852 (-1.0%) | 2.0781 (-1.3%) | 2.0452 | 206.8 | 0.0288 |
| d32s | +D0FIX (7) | 1.7391 | 1.76% | 1.6788 (-3.5%) | 1.6591 (-4.6%) | 1.6885 | 151.4 | 0.0646 |
| d32f | +D0FIX (7) | 2.0344 | 2.16% | 1.9482 (-4.2%) | 1.9203 (-5.6%) | 1.9752 | 306.4 | 0.0225 |
| v1s | +D0FIX (7) | 2.4384 | 1.48% | 2.3669 (-2.9%) | 2.3435 (-3.9%) | 2.3674 | 107.1 | 0.1130 |
| v1f | +D0FIX (7) | 2.3865 | 1.13% | 2.3330 (-2.2%) | 2.3154 (-3.0%) | 2.3170 | 115.9 | 0.0393 |
| v16s | +D0FIX (7) | 2.1354 | 2.03% | 2.0503 (-4.0%) | 2.0227 (-5.3%) | 2.0732 | 246.1 | 0.0458 |
| v16f | DSTK (4) | 2.0739 | 1.53% | 2.0061 (-3.3%) | 1.9868 (-4.2%) | 2.0135 | 338.7 | 0.0159 |
| v32s | +D0FIX (7) | 2.2964 | 1.30% | 2.2374 (-2.6%) | 2.2181 (-3.4%) | 2.2295 | 311.4 | 0.0347 |
| v32f | DSTK (4) | 2.2101 | 1.15% | 2.1557 (-2.5%) | 2.1401 (-3.2%) | 2.1458 | 484.2 | 0.0121 |

**Resolution.** The 2-read I(d1s, d1f) has sd ~0.23%, plus ~0.15% from R0. The 4-row I has sd ~0.42%, because v1s
dominates. v1-only claims therefore pre-register 4 reads.

## 2. Base combo: master DSTK + the shared C1 overlay

**Generator.**
- `campaigns/2026-10-31-public-plus/files/combos/mk_c1dec.py` on a new branch `campaign/c1-dec` (worktree
  `.worktrees/c1-dec`, from master; never master or campaign/public-plus directly).
- It refuses any input whose sha256 is not 1c653e3f...

```
python3 mk_c1dec.py <master 1c653e3f .py> <outdir> c1d-h1            # C1D-V1..V4-c1d-h1.py (ship candidates)
python3 mk_c1dec.py <master 1c653e3f .py> <outdir> c1d-h1 --bench    # C1D-BENCH-c1d-h1.py (D0 only, never shipped)
python3 mk_c1dec.py <master 1c653e3f .py> <outdir> c1d-h2 --var V5 --t1 <c8sA4|c8sB4|c12> --t2 t2   # after the screen
```

**CUDA.** `_MA_ALT_SRCS["moe_c1"]` is a copy of `_MA_ALT_SRCS["moe_helper0"]`; `moe_helper0` itself is untouched.
Every replace is count-asserted. A failure prints `[c1] off: <why>` and leaves master's behaviour.

| id | edit | anchor (decode text) |
|---|---|---|
| H0-H13 | the 14 CUDA_SRC hunks of `git diff 68e99e7d 016e301e -- muse_moe_w4a16/submission.py`. H0-H2, H4-H9, H12, H13 verbatim | as in the diff (each count 1 in decode, verified) |
| H3' | C1 Smem block (J6 tables + c1fix / c1bar / c1xb / c1seq / c1cnt / c1claim) + `int c1pre[NW], c1preB[NW];` | `  int unit_next, unit_next2;\n  signed char eid[EMAX];...pos[EMAX];...\n};` |
| H10' | `FSEG` and the FINPF first item on `FNT` threads | `#if OKC_FINPF\n  if (hasp) {` (decode wraps it) |
| H11' | FINCOL loop on `FNT` threads | `  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {` |
| P1 | restore the per-refill `fence.proxy.async.shared::cta` before `c1_expect(bars + sl_, OKC_C1COPY);`. 016e301e dropped it; the verifier listed it as a review item; decode's D0FIX / MD16F form is the precedent | in the issue lambda |
| P2 | **F1** (`-DOKC_C1F1=1`), described below | `#if FAST1_ON\n  // tiny build (M <= 8)` (count 1) |
| P3 | **PBE**, described below | end of `c1_phase<false>` |
| P4 | **TRIG** (`-DOKC_C1TRIG=1`): `griddepcontrol.launch_dependents` right after phase A; the master site after phase B is compiled out | `  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);` and `  asm volatile("griddepcontrol.launch_dependents;\n" ...` (each count 1) |
| P5 | **CO** (`-DOKC_C1CO=1`): `cudaFuncSetAttribute(okc::moe_c1fused<false/true>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared)`, so the SM keeps 228 KB of smem for the unpooler beside moe | `  if (!attr) {` in okc_run |
| P6 | rename `moe_fused` -> `moe_c1fused` (7 occurrences). The judge profile then names the C1 kernel (the VALID evidence), and no symbol is shared with master's modules | whole `moe_c1` string |

**P2: F1.** It runs right after `griddepcontrol.wait`.
- Each warp's lane 0 ORs the touched-expert mask from `p.topk_ids` (M x 8 <= 32 ids: one per lane, then
  `__reduce_or_sync`).
- It inits its own D ring barriers and computes its phase-A stage range from nS = popc(mask) x RB_A, with the
  `muldiv_g` split that run_phase uses.
- It issues its first min(D, s1 - s0) stages from `p.wAk` (blk = e x RB_A + (v & (RB_A - 1)), expert e = `__fns(m, 0,
  jj + 1)`, as FAST1 computes it) and sets `S.c1pre[w]`.
- `c1_init` stops initializing c1bar. `c1_phase<false>` skips the pre-issued stages.
- This is the same 73.7 KB-per-CTA head start master TINY gets from FAST1 = 2. As built, C1 issues nothing until
  route() ends.
- Premise: at M <= 8 every touched expert is exactly one skinny job, in expert order (FAST1's own comment), and C1 has
  no fat jobs and no dyn pool.

**P3: PBE.** After a warp's last phase-A stage, and before the CTA_SYNC:
- lane 0 issues the warp's first D phase-B stages into its now-free private slots (fence first) and sets
  `S.c1preB[w]`;
- `c1_phase<true>` skips them;
- x still waits for doneA (J4), so only weights move.
- It replaces INC's BFLAG phase-B prefetch, which C1 lacks. The M0 metric cancelled the A->B transition, so this loss
  was never measured.

**Python** (same overlay).
- **Y1 defines.**
  - C1_BASE = `("-DOKC_TINY=1", "-DOKC_C1=1", "-DOKC_EPI4=0", "-DOKC_FAST1=0", "-DOKC_L2H=0", "-DOKC_UWARP=0",
    "-DOKC_FINPF=1", "-DOKC_CPF=0", "-DOKC_C1F1=1", "@moe_c1") + ("-DOKC_PRMTSC=1",)`. `_ma_ext` adds `-DOKC_DQ7=1`.
    WS stays 0, as the C1 static_assert requires.
  - **c8s** `("-DOKC_NW=8", "-DOKC_C1D=2", "-DOKC_C1MAXM=1", "-DOKC_C1ACC=2", "-DOKC_C1CO=1", "-DOKC_C1TRIG=0|1")`.
  - **c12** `("-DOKC_NW=12", "-DOKC_C1D=2", "-DOKC_C1MAXM=4", "-DOKC_C1ACC=2")` = M0's C1-12.
  - **t2** `("-DOKC_NW=8", "-DOKC_C1D=2", "-DOKC_C1MAXM=8", "-DOKC_C1ACC=4")` = M0's C1-8T2A4.
- **Y2 `_c1_stage_order(u)`.** It is moe's `_stage_order`: a lossless byte permutation of the unit-order copies
  `st["ma_w"][10]` / `[11]` to `[blk][k][4,096 B codes | 512 B scales]`. It adds 905,969,664 B, appended as w[14] /
  w[15] at the **end** of prepare. Master's allocations therefore keep master's addresses, and the tuner and
  `_check_guards()` run on master's state.
- **Y3 prepare wrap.**
  - A thread compiles the variant's C1 modules while the prior prepare runs (load_inline).
  - Afterwards it requires `len(st["ma_w"]) == 14` and `st["ma_sfx"] == ("-DOKC_PRMTSC=1",)`, then appends the copies.
  - It self-tests against master TINY / TINY2 on synthetic routings: M=1 with 0 / 1 / 3 / 8 local experts; M=4 for
    c12; M=8 with 1 / 8 experts for t2. The bar is out rel-L2 <= 2e-3, with identical handling of non-local routes.
  - It re-runs `_check_guards()`.
  - It prints `[c1] on t1=.. t2=.. trig=.. unp1=.. copies=905969664 selftest rel=..`, or `[c1] off: ..` (master).
- **Y4 `_moe(M, state)` override (shape-only):** c8s at M == 1; c12 at M <= 4; t2 at 5 <= M <= 8, per variant.
  Everything else goes to master's `_moe`. The tuner keeps "agent" (`st["bank"] = None`), so it never reverts a pick.
- **Y5 UNP1.** mk_swdec V2's `_bfp_gemm` wrap, restricted to `name == "un" and m == 1 and st.get("c1_ok")` with
  T1 = c8s. It sets `bfp_SK2_PDL = _C1_UNP1` for that launch and calls `_bfp_pdl(m)` in `finally`. It wraps the DSTK
  `_bfp_gemm` (the latest definition).
- **Y6** line 2: clause + `pp-nonce C1D-<V>-<nonce>`.

**Size and time.**
- About +28 KB, giving ~888 KB (< 1 MB).
- Prepare adds the C1 compiles (overlapped, ~+10-30 s wall) and ~2 s for the copies. Judge prepare ~125-170 s (DX2
  entry 113.5 s; the limit is 300 s).

**Compliance.**
- NVFP4 bytes are only permuted (rules :107).
- FP32 accumulation; BF16 x and inter. Only the FP32 K-summation order changes (M0: farther from fp64 on 30/31
  cancellation elements, still FP32).
- No timing or race in kernel_fn; the C1 choice is static by M.
- load_inline only; no subprocess, network or harness access.
- P2-P5 change launch order and placement only:
  - the unpooler reads x after its own `griddepcontrol.wait` (sk2 :136), which waits for the whole moe grid and its
    writes;
  - before the wait it reads only static weights.
- No input-derived state persists across calls.

## 3. Variants

Every judge floor in this section is on rho = (geomean over the checks of the row's `paired_ratio` from okc status
`result.records[]`, `event == "benchmark"`) / R0_row, **NOT** the median of `ratios` (§1.3). The H100 bars are us
deltas (§5).

### V1 C1S-A4: c8s at T = 1 + TRIG after phase A + 4 unpooler stages before the wait (rows d1s, d1f)

- **Flags:** `--var V1` = `--t1 c8s --trig 1 --unp1 4`. T >= 2 stays on master (M 2..4 -> master TINY).
- **Mechanism (scheduling / critical path).**
  - The slim moe CTA plus the max carveout let the unpooler grid become resident at moe's trigger.
  - The trigger now fires once the slowest CTA's phase A is done. The unpooler then issues all 4 of its weight stages
    (49 KB per CTA, 6.29 MB in all) during moe's phase B (~6 us at d1s) and blocks in `griddepcontrol.wait`.
  - After moe completes, only x (8 KB), the escape table, the compute and the store remain.
- **Co-residency budget:**
  - smem 103.3 (Smem ~13.2 + ring 73,728 + x 2 rows 16,416) + 115.5 + 2 x 1 KB reserve = 220.8 KB <= 228 KB;
  - threads 256 + 416;
  - registers 8 x 32 x 96 + 416 x r8(R_un) <= 65,536, which needs R_un <= 96 per thread. R_un is unmeasured; for
    scale, the gle sk2 instances read 80 / 84 (swdec sassd).
- **Predicted (H200).**
  - The unpooler's post-moe time goes from 3.62 to ~1.2-1.5 us (-2.1 to -2.4). Against that: +0 to 1.4 us of HBM
    contention inside phase B, and C1S's own moe delta of +-0.5 (consumer ~1.07x vs decode TINY: M0 C1-8D2 AB
    1.103 / 1.03 for D6).
  - d1s: central -1.3 us (range +0.3 to -2.4). d1f: -0.6 (un chain 2.90).
  - rho_d1s ~1.019, rho_d1f ~1.008, I ~+1.6%, headline ~+0.27%.
  - On the H100 at full bytes the contention costs more (3.0 TB/s), so the screen is conservative.
- **Floors:** touched KILL floors d1s 2.4719 and d1f 2.4387; all 10 other rows >= their 2-check floors (§1.3).
- **Risks.**
  - Registers (STEP 0).
  - Two moe CTAs on one SM: 103 KB allows 2 per SM, so STEP 0 requires 132 distinct `%smid`.
  - Phase-B contention.
  - The unpooler's escape table still loads after the wait (§8).
  - UNP4 without co-residency measured **+0.46 us** at d1s (V2off4 - M; an OOM-truncated process, 20261101-135718).
    UNP4 therefore never ships without f_co.

### V2 C1S-B4: c8s at T = 1 + master trigger position + UNP1 = 4 (rows d1s, d1f)

- **Flags:** `--var V2` = `--t1 c8s --trig 0 --unp1 4`. The module differs from V1 only by `-DOKC_C1TRIG`.
- **Mechanism.** The overlap window is moe's grid barrier + final (~0.5-1 us): no contention, smaller reach.
- **Predicted (H200):** d1s -0.8 (0 to -1.5), d1f -0.4. I ~+1.0%, headline ~+0.17%.
- **Floors:** as V1.

### V3 C1-12: c12 at M <= 4 (rows d1s, d1f)

- **Flags:** `--var V3` = `--t1 c12 --trig 0 --unp1 2`. No co-residency (193 KB smem); the unpooler stays at master's 2.
- **Mechanism (consumer issue rate).**
  - M0 AB 1.239 vs moe INC, about 1.20 vs decode TINY, which already rotates its helper.
  - The H200's d1 moe is co-bound and the +5-7% rider read neutral, so the realized gain is the HBM headroom only.
- **Predicted (H200):** d1s -0.4 (+0.5 to -1.5), d1f -0.4. I ~+0.6%, headline ~+0.10%.
- **H100 ruler:** the calibrated R-proxy (§4 CAL); at full bytes, no harm.
- **Floors:** as V1.

### V4 T2: t2 at 5 <= M <= 8 (rows v1s, v1f; T = 1 stays master)

- **Flags:** `--var V4` = `--t2 t2`.
- **Mechanism (consumer).**
  - M0 N=8: AB 1.142, K 1.125, with pinned routing (one 8-token job).
  - In situ there are ~10 jobs of 1-2 tokens (loc 11), so J4 stages <= 2 job buffers per CTA in phase B.
  - v1s moe is 66.27 us against a ~57.9 us stream floor (co-bound, ~8.5 us gap).
- **Predicted (H200):** v1s -1.0 (+0.5 to -3.0), v1f -0.7. Headline ~+0.14%.
- **Judge:** pre-register 4 reads (v1s sd 1.48%).
- **Floors:** touched KILL floors v1s 2.3669 and v1f 2.3330; the other 10 rows >= their 2-check floors.

### V5 fresh-submission stack: the best d1 survivor + t2 (rows d1s, d1f, v1s, v1f)

- **Built only if** one of V1 / V2 / V3 survives the screen **and** V4 survives.
- **Flags:** `--var V5 --t1 <winner> --t2 t2`.
- One stack screen (2 valid processes) checks for interference: the same rows, both parts' bars.
- **Predicted:** V1 + V4 gives headline ~+0.41%, I4 ~+1.2%.
- **Floors:** the union of its parts' floors.

## 4. Rung D0: diagnostic, before any variant is screened (H100 gpunode06 GPU 7 only; no checks)

**Package.**
- takemyjob worktree `~/git/.takemyjob-wt/c1-dec`, branch `hopper/c1-dec`, `prototypes/hopper-probe/probes/c1dec/`.
- It copies `probes/swdec`: 4-copy F/G, virtual knob arms (`--arm NAME=SUB:knob=v`), shared per-row inputs, the
  sassd STEP-0 census, gate / adv / stress.
- It takes c1m0's `sassloop.py` census and R-mode patch logic.
- Commands:
  - `HP_BUNDLE=c1dec bin/hopper-probe build`;
  - `HP_BUNDLE=c1dec bin/hopper-probe run gpunode06 --gpus 7 --tmax 85 c1dec-<mode>`;
  - end with `c1dec-clean`.
- Rules: USER, no root, no Slurm, node-local `/tmp/USER-hopper` only, the guard, never change node state, and run
  only when GPU 7 has no compute process.

**STEP 0** (no timing, ~20 min). Any failure stops the variants it names.
1. Anchors: 14 + 6 edits at count 1. Every non-C1 module has master's module name and is SASS-identical to M (sassd
   lists only NEW `moe_c1fused` kernels).
2. Census of every C1 build:
   - registers <= 96, 0 spill;
   - stage loop <= 560 instr per 32 HMMA (M0 557; P1 adds ~2);
   - `okc_smem_bytes`: c8s <= 105,000 B, c12 <= 200,000, t2 <= 226,000.
3. The unpooler sk2 instance `<1,8,1,1,8,2,4,true,false,2,1,bf16,PK=true>`: registers R_un and static smem. Budget:
   - 256 x r8(R_c8s) + 416 x r8(R_un) <= 65,536;
   - smem_c8s + smem_un + 2,048 <= 233,472;
   - otherwise **V1 / V2 KILL**.
4. Co-residency stamps (bench define `OKC_C1STAMP`; bench-patched sk2 entry stamp), 64 replays at d1s:
   - the moe CTAs sit on 132 distinct `%smid` in every replay;
   - f_co (the share of unpooler CTAs whose entry precedes the exit of the moe CTA on their SM) >= 0.9 for c8sA4 and
     >= 0.5 for c8sB4;
   - the pad control (moe dynamic smem + 20 KB) has f_co = 0.
5. Correctness, C1 vs master moe:
   - M=1 judge-like d1s / d1f draws x 2 seeds + pinned 1 / 3 / 8 experts + 0 local experts;
   - M=2..4 for c12; M=8 v1s / v1f draws + all-8-on-one + 8-distinct for t2;
   - out rel-L2 <= 2e-3; bitwise repeatable; `reference.check_implementation` at every row; self-test prints `on`.

**D0a** (F only; rows d1s, d1f, v1s, v1f, plus d16s as a do-nothing control; ~35 min).
- Subs: M x 4 (master bytes: M, I, K, P) and C x 4 (byte copies of C1D-BENCH, own extension caches). Load order is
  pre-registered and shuffled.
- Virtual arms on each C copy:
  - `inc` (master TINY / TINY2: the in-sub placement control);
  - `c12`, `c12raw` (P2 + P3 off);
  - `c8sB2`, `c8sB4`, `c8sA4`, `c8sA4pad`;
  - `t2`.
- 30 rounds x 8: a 256 MiB `ld.global.cg` read sweep, then one graph replay.

**D0b** (F only; rows d1s, v1s; ~20 min).
- Arms `inc@R100 / R67 / R50 / R33`, `c12@R67`, `c8sB2@R67`, `t2@R67`.
- R-modes are timing-only bench builds, never shipped; their outputs are garbage:
  - INC: the 8 `"r"(UNIT_BYTES)` sites -> `"r"((unsigned)OKC_RUB)` in a bench copy `@moe_rb`, with RUB = 24,576 /
    18,432 / 12,288;
  - C1: `-DOKC_C1COPY=3072 / 2304 / 1536`.

**Output.** Per arm, with the per-copy values and a 95% bootstrap CI:
- D = mean(arm copies) - mean(M copies);
- D_mech = arm - inc (mechanism; inc - M is placement);
- f_co and the R-curve.

**D0 rules (pre-registered).** These are H100 us deltas. No judge rho is involved; that metric, rho = geomean of
per-row `paired_ratio` (not the median of `ratios`), applies only in §6.

| item | rule |
|---|---|
| CAL | R67 is the H200 proxy if D(inc@R33 - inc@R67) at d1s is in [-1.5, -0.2] us. The H200's halving step at d1 is ~-0.7 (DX23 2.6% of ~27 us). If below -1.5, use R50 when its step D(R25 - R50) is in the band. If no proxy qualifies, V3 and V4 cannot be screened and get no checks |
| CO | V1 / V2 need STEP 0 #3 and #4, and D_mech(c8sA4 - c8sA4pad) <= -0.4 us at d1s: the gain must come from co-residency, not the engine |
| advance V1 / V2 | D_mech(c8sA4 or c8sB4, d1s, F) <= -0.6 us |
| advance V3 | D_mech(c12, d1s, R-proxy) <= -0.6 and D_mech(c12, d1s, full F) <= +0.3 |
| advance V4 | D_mech(t2, v1s, R-proxy) <= -1.2 and full F <= +0.3 |
| report only | c12 - c12raw (the value of F1 + PBE); c8sB2 - inc (the slim engine alone); d16s within +-0.3 (it runs no changed code) |
| CLAIMS | every d1 arm has D_mech > -0.3 in both modes and t2 > -0.6 at v1s -> decode CLAIMS "[refuted] moe C1 port at T <= 8" with these numbers |

## 5. H100 screen (survival; the orchestrator's bar; mandatory protocol)

**Arms.**
- 4 byte-identical master copies (M, I, K, P) and 4 byte copies of each candidate file `C1D-<V>-c1d-h1.py`, each
  under its own extension cache.
- Load order is shuffled and rotated per process, with fresh processes and copy-pooled statistics.
- An in-sub control arm on each V copy (`_C1_OFF = True`, giving master's routing and UNP 2) separates placement from
  the mechanism.
- V1 and V2 share one process.

**Metrics.**
- **F** (primary): the 256 MiB read flush, then one graph replay; 30 rounds x 8.
- **G** (corroboration): host enqueued ahead behind a GPU spin, then 16 back-to-back replays; 40 rounds.
- Rows: d1s, d1f, v1s, v1f, d16s, v32s.

**Validity.** A process counts only if the F A/A |mean(M,I) - mean(K,P)| <= 0.20 us at d1s and v1s. Up to 2 void
repeats. Two valid processes are required.

**Survive (in BOTH valid processes).** These are H100 us deltas, not the judge metric. The judge gate (§6) uses rho =
geomean of per-row `paired_ratio` from okc status `result.records[]` (`event == "benchmark"`), **NOT** the median of
`ratios`.
- **D(V - M) at d1s <= -0.80 us** (V1, V2, V3, V5) **or D(V - M) at v1s <= -1.50 us** (V4, V5), with the CI upper
  bound < 0.
- The primary mode is full-bytes F for V1 / V2 / V5's c8s part (scheduling class). For V3 / V4 it is the calibrated
  R-proxy F (consumer class: the H100 at full bytes is byte-bound at d1, H100 r(d1) 1.28 vs H200 1.026), and full F
  must also be <= +0.30 us.
- G has the same sign: <= -0.5 (d1s) or <= -1.0 (v1s).
- No untouched row is worse than +0.30 us F in both processes; the in-sub control is within +-0.3 of M.
- Tests are clean (below).

**KILL at the first valid process:** d1s D > -0.30 (V1-V3), or v1s D > -0.60 (V4).

**Tests** (swdec suite + C1 items):
- 13 TASK tests + 12 rows x 2 seeds through `reference.check_implementation`, with kernel sequences per row (the C1
  kernel only at the touched rows);
- the 102 adversarial cases; C1 vs master moe within the FP32-order envelope; bitwise determinism;
- 16,384 stall-injected fresh-state replays at each touched row with 0 gross;
- V1 / V2 positive control: the unpooler's `griddepcontrol.wait` compiled out, with TRIG = 1, must FAIL;
- compute-sanitizer racecheck + synccheck, one replay each at M = 1, 4, 8;
- a 1,000-replay soak; prepare <= 200 s on the H100.

**Audit (independent, before any check):**
- memory order: J4 acquire + proxy fence; J5 fixup release / acquire and counter reuse; P1; P2 barrier-init
  ownership (c1_init no longer touches c1bar); P3 slot reuse;
- the PDL race argument for P4;
- co-residency evidence;
- the rules checklist.

## 6. Judge gate (2 checks per survivor; never a re-roll of judged bytes)

**Metric:** rho_row = geomean over the checks of the row's `paired_ratio` from okc status `result.records[]`
(`event == "benchmark"`) / R0_row (§1.3). It is **NOT** the median of `ratios`. I = sum(w ln rho) / sum(w) over the
touched rows (46 / 16).

**VALID** requires all of:
- 13/13 tests and 12/12 rows in both checks;
- `[c1] on t1=.. t2=..` with the shipped config printed in both logs;
- the profile shows `moe_c1fused` at the touched rows only.

A VOID read gets at most one re-send of the same bytes.

| variant | GO | KILL | otherwise |
|---|---|---|---|
| V1 / V2 / V3 | I(d1s, d1f) >= +0.6%, both touched rho >= 0.995, every untouched row >= its 2-check floor, mean headline >= 2.1564 | I < +0.2%, rho_d1s < 0.992 or rho_d1f < 0.986 (the 2-check floors), a failed test, or 2+ untouched rows below their floors | HOLD: 2 more reads, decide on the 4-read pool |
| V4 | pre-register **4 reads**: rho_v1s >= 1.008 on the 4-read geomean, rho_v1f >= 0.995, floors as above | rho_v1s < 0.995 after 4 reads, or a failure | shelf |
| V5 | I4 (4 rows) >= +0.7%, each part's touched rows >= 0.995, floors | I4 < +0.2% or a part's KILL | HOLD |

**On GO.**
- Attempt file first, then one scored run of r3 (fresh nonce, code identical).
- Land on master if the scored run is valid, the 3-read pooled I is >= +0.5%, and every untouched row of the scored
  read is >= its scored-read floor (§1.3).
- The board moves only above 2.2220. The standing 11-02 00:00Z rule lands anything above the board.

## 7. Order and time (now ~01:00Z; H100 window to 11:00Z; checks expire 00:00Z)

1. **01:00-03:30** Builder (effort high): `mk_c1dec.py`, the 4 candidates + BENCH, and the probe package. Audit-style
   self-review of P1-P4.
2. **03:30-03:50** STEP 0 on GPU 7. A register or f_co failure kills V1 / V2 immediately; the builder then
   continues with V3 / V4 only.
3. **03:50-04:50** D0a + D0b.
4. **04:50-06:10** Screen: V1 / V2 together, then V3 / V4, in 2 valid processes each (~15-20 min per process, plus
   <= 1 void repeat).
5. **06:10-06:50** Tests.
6. **06:50-07:40** Independent audit.
7. **~07:45** Attempt files + 2 checks for the best survivor.
8. **~08:30** Read; scored run on GO. V5 stack screen ~08:00-08:40 if both parts survived; its checks ~09:30.

**Budget:** <= 6 of 10 decode checks, <= 2 of 5 scored. **Risk:** the 5-hour usage windows; read the reset time from
the limit message.

## 8. Considered and dropped (do not rebuild)

| lever | why |
|---|---|
| C1-12 slim (MAXM 1) / C1-8 D=3 slim | 144 / 140 KB of smem: no room for the 115.5 KB unpooler |
| UNP4 with master TINY | d1s +0.46 us (V2off4 - M) and +0.59 vs off/2 in the same sub (H100 20261101-135718, OOM-truncated, indicative): with no co-residency, the pre-wait stages start only after moe exits |
| C1 + co-residency at T = 8 | phase A's 8 x rows need 65.7 KB of smem (x region 131 KB with 2 phase-B buffers). Reading x from L2 instead is an M2-size redesign |
| C1-M at d16 (MID) | phase-B x pool unresolved (C1-DESIGN M2 item); out of the T <= 8 scope |
| TRIG at kernel start | the unpooler's 6.29 MB would compete with phase A's ramp for ~20 us; not screened |
| unpooler escape table before the wait | an sk2 edit seen by every sk2 user (qkv / o / pooler / unpooler at all T). Follow-up only if V1 survives |
| shrinking master TINY instead of using C1 | TINY uses 288 threads x 145-149 registers (42.9K), which leaves the 416-thread unpooler <= 54 registers per thread. Not reachable without re-tuning the unpooler as well |
| INC-D1 rider | decode already has D6; D1 was judge-neutral on moe (MD16F) |
| consumer claims from R50 / warm-L2 only | M0's metric cancelled the head start, the A->B transition and the HBM; only in-situ D0 numbers count |
| CG-O, CG-U, O8, CPUSH-DEC, SWEEP-dec V2 / V3 / V4 | closed (DX2 84b5da1a, sweep lanes) |

Scratch artifacts from this design:
- `/tmp/claude-1000/-home-user-git-musespark/08efaca7-cd01-4c08-87f6-da5f73e85a3c/scratchpad/c1port/dryport.py`
  and `.../c1port/moe_c1_dry.cu` (the port dry run);
- `.../c1port/c1-cuda.patch` (the C1 CUDA hunks);
- `.../c1port/floors.py` and `.../c1port/r0calc.py` (the R0 and floor tables; input JSONs in `.../c1port/r0` and
  `.../c1port/r0x`).
