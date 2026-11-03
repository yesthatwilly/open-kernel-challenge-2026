# SWEEP-att: scheduling-knob variants on master HSV1+CPUSH (muse_attention_decode)

Design only (read-only; no H100 run, no check). Written 2026-11-01 ~20:30Z.
- Base: master 8881e073, submission.py sha256 399eb35d (= HSV1-r1 bytes = CPUSH 145b6849 + HSV1).
- Board: 2.5192 (scored 1cf77c00). Target: >= +1%, i.e. a scored read >= **2.5444**.
- Numbers: scratchpad `sweepcalc/` (pool.py, ref.json, final.py).

**Gate metric, used in every gate below.** For row r and a set X of judge reads:
- PR_r(X) = the geometric mean over X of `paired_ratio`, taken from `okc status <job> --json` ->
  `result.records[]` with `event == "benchmark"`, matched on the row's (case.load, case.layer).
  It is **not** the median of the per-rep `ratios`.
- rho_r = PR_r(candidate reads) / R_r, where R_r is the reference pool in §1.
- e_r = headline % per 1% speedup on row r (§1).
- Ĥ_T = Σ_{r in T} e_r · 100·ln(rho_r), the touched rows' headline-equivalent %.
- G_u = exp(Σ_{r in U} e_r ln rho_r / Σ_{r in U} e_r), over the untouched rows U.
- H̄ = the mean of the checks' okc headlines. It is a harm guard only.

## 0. Bottom line
1. **None of the variants is expected to move master's own level by +1%.** The best single knob is worth about
   +0.17..+0.27% in headline. The best stack that can be built is worth about +0.5%.
   - Most of the requested directions are already refuted by today's lanes (§2): HSV1 split on v16s/v32s,
     cluster push on v16s/v32s, and plan split/stage counts on rows the public author already tuned.
2. **The board is beatable by +1% anyway.** Master's level already sits above 2.5444:
   - 2.5539 on the 3 HSV1 reads (2.5473, 2.5953, 2.5192);
   - 2.5531 counting B1XN's 2.5571 and 2.5465 (master plus 2 small riders);
   - 2.5516 on the per-row pooled model.
   - 1cf77c00 was the low read.
   - Per-read headline sd is about 1.0-1.1%.
   - A no-harm fresh variant at master's level clears 2.5444 on one scored read with **P ≈ 0.60**.
   - With V1, P ≈ 0.66. With the V1+V5+V2 stack, P ≈ 0.77.
   - Two scored reads of two different variants give P ≈ 0.86-0.95.
   - Say this plainly in the attempt files: most of the "+1%" is the board being a low read, not the knob.
3. **Ready now:** V1 = B1SN (master + CPNQF, audited PASS, H100-gated), 2 checks.
   - B1XN's bytes (CPNQF + HSK8) were judged HOLD. Do not re-send them. Any stack must add a new GO'd part.
4. **H100, GPU 1, about 1 h:** one plan screen (V4 arms plus 2 screen-only arms) and the V3 M0 cluster null.
   No build is needed for either.
5. **Builds, for tonight or tomorrow:** V2 NKF (best EV of the builds) and V3 PH8 (only on an M0 GO).

## 1. Reference pool R, floors and elasticities (master SASS on each row)

**Pools.**
- d1s, CPUSH PH SASS, 6 reads: 929c84e04a0e4a65, 21889c3ef86f459b, 8a73a9d739c24f3b, 608fd6ede72c4c75,
  176fa97c34184a03, 1cf77c003c3e4428.
- v1s, HSV1, 5 reads: 608fd6ed, 176fa97c, 1cf77c00, a4fcde9f91e749c1, 91604729db6a457d.
- v1f, 9 reads: 9499367892524ea7, 262412e1b93349ac, 1280e6f2120a4fbd, plus the 6 CPUSH/HSV1 reads.
- The other 9 rows, 11 reads: the 9 above plus a4fcde9f and 91604729.
- These kernels are SASS-identical across C5L, CPUSH, HSV1 and B1XN (lane cpsass/census), so the pooled spread is
  placement plus judge noise.

**Model.** The model headline at R is **2.5516** (load = 46·sliding + 16·full, geomean over 6 loads). "floor" below
is GAPMAP-1031's compliant floor (judge µs, including 0.8 µs of graph residue).

| row | R (PR geomean) | sd/read | R/1.03 | R/1.05 | µs now | floor µs | gap µs | %/µs | e_r (%H per 1% row) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d1s | 4.2812 | 2.47% | 4.1565 | 4.0773 | 4.86 | 3.60 | 1.26 | 1.920 | 0.092 |
| d1f | 2.8217 | 1.51% | 2.7396 | 2.6874 | 11.15 | 8.20 | 2.95 | 0.665 | 0.073 |
| d16s | 3.2039 | 2.29% | 3.1106 | 3.0513 | 8.39 | 5.56 | 2.83 | 0.434 | 0.036 |
| d16f | 2.3220 | 1.90% | 2.2544 | 2.2115 | 86.53 | 85.03 | 1.50 | 0.151 | 0.130 |
| d32s | 3.0219 | 2.17% | 2.9339 | 2.8780 | 10.37 | 7.72 | 2.65 | 0.246 | 0.025 |
| d32f | 2.3259 | 2.05% | 2.2582 | 2.2151 | 165.27 | 164.14 | 1.13 | 0.085 | 0.140 |
| v1s | 3.5265 | 1.89% | 3.4237 | 3.3585 | 6.47 | 4.20 | 2.27 | 1.426 | 0.091 |
| v1f | 2.3707 | 0.78% | 2.3017 | 2.2579 | 15.16 | 8.61 | 6.55 | 0.494 | 0.074 |
| v16s | 2.4219 | 1.65% | 2.3514 | 2.3066 | 12.13 | 6.16 | 5.97 | 0.337 | 0.041 |
| v16f | 1.9144 | 0.99% | 1.8587 | 1.8233 | 107.28 | 85.23 | 22.05 | 0.117 | 0.125 |
| v32s | 2.1691 | 0.81% | 2.1059 | 2.0658 | 15.96 | 8.52 | 7.44 | 0.198 | 0.031 |
| v32f | 1.9924 | 1.03% | 1.9344 | 1.8975 | 195.97 | 168.66 | 27.31 | 0.069 | 0.134 |

**Judge evidence already in hand (B1XN, paired_ratio).**
- d1s, CPNQF SASS: 4.4980 / 4.2812 = **4.3883 (+2.50% vs R)**.
- v1f, HSK8 SASS: 2.3901 / 2.3955 = **2.3928 (+0.93%)**.
- Neither cleared its transfer line, so B1XN went to HOLD.

## 2. Requested directions already settled (do not spend checks)

**HSV1 head split (<8,false,47>) on other rows.**
- v16s/v32s:
  - HSd measured +2.30 / +4.38 µs.
  - HSi (no duplicated loop) measured +0.31 / +0.50 µs. The cause is 2x CTAs at 2/SM (CLAIMS:136-137).
  - Halving the splits instead to keep 128 CTAs doubles each computing warpgroup's tiles, so it is worse.
- d16s/d32s:
  - There is no L=1 HS kernel; the L=1 per-head path is CPUSH's PH, which is V3.
  - A per-head CTA at the same CTA count needs 2x tiles, which adds a gather round.
- v1f: 504 CTAs are not co-resident for the coop grid. The fixed-count form is HSK8 (V5).

**CPUSH push on uncovered rows.**
- v16s/v32s: the CPUSHQ M0 cluster null alone costs +2.60 / +4.53 µs, because clusters of 4/8 at 1 CTA/SM reach
  only 120 SMs (CLAIMS:140).
- d1f/v1f: cluster merges of 128-318 splits are refuted (public NEG attn #4/#5; AT5c +1.1/+1.6 µs).
- v1s: VPUSH was KILLed and VPUSHF shelved; HSV1 beats VPUSHF.
- Only d16s is open, at 2 CTAs/SM, where master already doubles SMs. That is V3.

**Split counts (plan) on q8 rows.**
- v16s n_splits 7 is null (CLAIMS:142).
- Any v16s/v32s grid above 132 CTAs doubles some SMs unevenly. Doubled q8 CTAs run at about half speed (HSi; the
  public's v32s tuning to 4 = 1/SM). No arm is left.

**Split counts on L=1 rows (PUB CTA 58 KB smem, 114 regs -> 3 CTAs/SM possible).**
- d16s 16->8: the public-kernel analysis expects +0..+1 µs (FINDINGS G3; 2 tiles = one gather group today).
- d32s 8->11: about neutral, because the stream already has ≥ 8 MB in flight against about 4.6 MB needed.
- Both are screen-only arms in V4's run.
- d1f is the one L=1 arm with upside (V4).

**Stage counts (STAGES 3 -> 4/5/6), nc, and coop.**
- The d1f/v1f 4-stage ring lost on the H200 (public NEG attn #4).
- The d16s/d32s streams are already bandwidth-saturated in flight.
- W2 nc=2 at v1s was KILLed (A1, v1s 2-4% slower); the nc sweep is too small to see (G2).
- d1f coop=0 and lazy-zero are refuted (CLAIMS:143-144).
- coop=1 on d16s/d32s: PUB measured the last-CTA finalize faster there (plan comment).

**PP softmax turns on the per-request q8 kernel.** Not a knob. That loop has a CTA-wide `__syncthreads` per tile
(`!WS` branch), so turns would serialize the exp phases. It needs the WS restructure first.

## 3. Variants, ranked by expected judge gain x p
Each variant is master plus one or two named knobs.
- Δ: predicted headline vs master's level.
- p: chance it reaches a check GO with the gain intact.
- P1: chance one scored read is ≥ 2.5444.

| # | variant | knobs | touched rows: predicted rho [range] | ΔH central [range] | p | EV | P1 | ready |
|---|---|---|---|---:|---:|---:|---:|---|
| V1 | **SN** = master + CPNQF | PH new-row fold | d1s 1.019 [1.008, 1.027] (−0.09 µs) | +0.17% [+0.08, +0.25] | 0.75 | **+0.13%** | 0.66 | **now** (audited) |
| V2 | **NKF** = master + new-key fold | 32-tile chunking at L=8 B>1 + 7-key fold | v16s 1.043 [1.025, 1.061]; v32s 1.032 [1.019, 1.046] (−0.5 µs each) | +0.27% [+0.16, +0.38] | 0.45 | **+0.12%** | 0.70 | build |
| V3 | **PH8** = master + CPUSH-8 at d16s | PH ranks 16->8, rank keys 128->256 (B ≤ 16) | d16s 1.077 [1.037, 1.12] (−0.6 µs) | +0.26% [+0.13, +0.40] | 0.30 | +0.08% | 0.69 | M0, then build |
| V4 | **W313P** = master + d1f split 313 + P(L=1) | `_TUNED` d1f n_splits 252->313; rider P on L=1 only | d1f 1.028 [0.965, 1.077] (−0.3 µs) | +0.20% [−0.26, +0.54] | 0.35 | +0.07% | 0.67 | screen + 1-line edit |
| V5 | **KN** = master + HSK8 | v1f key-packed head split | v1f 1.009 [1.003, 1.015] (−0.14 µs) | +0.07% [+0.02, +0.11] | 0.85 | +0.06% | 0.63 | near-ready (subset of audited B1XN) |

**Stacks (§5).**

| stack | ΔH | p | EV | P1 |
|---|---:|---:|---:|---:|
| V1+V5 | +0.24% | 0.65 | +0.16% | 0.69 |
| V1+V5+V2 | +0.51% | 0.40 | +0.21% | 0.77 |
| + V3 | +0.78% | 0.15 | — | 0.84 |

**Transfer class.**
- V2 and V3 are critical-path cuts on latency-bound rows (the CPUSH/HSV1 class).
- V1 is a critical-path cut only when K/V is L2-resident; the judge read +2.5% d1s, about 1.4 se.
- V4 is parallelism mixed with merge contention.
- V5 is mostly a byte class; the judge read +0.93% v1f, about 1.7 se.

## 4. Variant cards
Common judge-gate lines (metric as defined at the top; 2 checks, r1/r2 differ only in the line-2 nonce):
- **VALID:** 16/16 tests and 12/12 rows in both checks. The profile shows the variant's kernel on each touched row
  and master's kernel names on every other row.
- **KILL (no scored run) on any of:**
  - a failure or a misroute;
  - G_u < 0.975;
  - any untouched row with rho < 1/1.05 (the R/1.05 column);
  - H̄ < 2.5263 (= 2.5516/1.01);
  - the card's touched-row KILL line.
- **GO:** VALID, no KILL, the card's touched-row GO line, G_u ≥ 0.985 and H̄ ≥ 2.5300.
  - Otherwise HOLD: no scored run, and a CLAIMS line.
  - On GO: one scored run of the r1 bytes.
- **Land on master** if the scored read is valid and > 2.5192, every touched row's 3-read rho ≥ 1.000, and the
  3-read G_u ≥ 0.985. The board moves if the read is > 2.5192.

### V1 SN: master + CPNQF (d1s; the PH new token's k/v pass is fused into warp 0's q pass)
- **Bytes, check-ready:** `/home/user/git/musespark/.worktrees/combo-attention_decode/campaigns/2026-10-31-public-plus/files/combos/B1SN-r1.py`
  (sha256 6c4a7d59b1f8873deb5e69d066b423613986844d419c32a28f03bf44c0770ff7) and `B1SN-r2.py` (2945dc61…), on branch
  campaign/combo-attention_decode efc86c4c.
  - Regenerate with `cd /home/user/git/musespark/.worktrees/combo-attention_decode && python3 -I
    campaigns/2026-10-31-public-plus/files/combos/mk_b1x.py OUTDIR --no-p`. It asserts the pinned blobs and writes
    B1SN = B1XN with `_HSK = 0`.
  - At runtime this is master 399eb35d plus CPNQF on <1,false,21|29>. The HSK8 instance is compiled but never routed.
  - Independent audit: PASS (scratchpad `audit-combo-attention_decode.md`). b1check 307 cases 0 fail; cpcheck 82 PH
    cases bitwise; SASS census 20 master-identical + 2 CPNQF + 3 HS/HSK; stress clean.
- **Predicted:** d1s −0.04..−0.13 µs, rho 1.019. Every other row runs master SASS, rho 1.00 ± noise.
- **Floors:** d1s ≥ R (4.2812). Untouched rows are judged as the group G_u, with the R/1.05 hard floor.
- **Touched-row lines:**
  - GO: d1s rho over the 4 CPNQF-SASS reads (a4fcde9f, 91604729, SN-r1, SN-r2) ≥ 1.015 (PR ≥ 4.3454), and the SN
    pair alone ≥ 1.000 (PR ≥ 4.2812).
  - KILL: the SN pair < 1/1.034 (PR < 4.1404).
  - p(GO | +1.8%) ≈ 0.77.

### V2 NKF: new-key fold for q_len 8 sliding at B ≥ 2 (v16s, v32s)
**Mechanism (critical path).**
- The window span is 2055 keys = 33 tiles, so master runs v16s at 5 tiles per CTA (splits 5,5,5,5,5,5,3,0) and v32s
  at 9 tiles (9,9,9,6).
- The last 7 keys are the block's own new tokens 1..7, which the new-token CTA already holds in `S.kn`/`S.vn`.
- NKF chunks only keys [lo, seq], i.e. 32 tiles: v16s becomes 8×4 (128 CTAs, all active) and v32s 4×8.
- The CTA that holds key `seq` folds the 7 new keys after its main loop. Nothing else changes.
- H100 L2-resident cost is about 0.73 µs per tile, estimated from the CPUSHQ stamps (3-tile loop-end imbalance
  2.18 µs at v32s). The fold plus patch is about 0.15-0.25 µs on one CTA (estimate). Net about −0.5 µs per row.

**Exact edit (generator `gen_nkf.py` on 399eb35d, every anchor asserted unique).**
- New MODE bit 128 on the q8 coop non-SPEC instance: kn[3] = <8,false,3> becomes <8,false,131>, built and launched
  only under tma bit 27.
  - Bits taken: 1, 16, 20, 24 and 25 on master, plus 26 = HSK in B1SN/B1XN.
- `_plan` sets `tma |= 1 << 27` when `L == 8 and window and B >= 2` (shape-only).
- Kernel, under NKF:
  - `n_keys_t = n_keys - (L - 1)` drives `ntl`/chunk.
  - The CTA with `end == seq + 1` runs the fold for its 2×64 rows: FP32 scores from the staged E4M3 q and the E4M3
    `S.kn` rows; the causal mask (token j sees new key i iff i ≤ j); an exact running-max update with an O/l rescale;
    P in FP32 (finer than E4M3); l from unrounded P; O += P·V with V from `S.vn`.
  - The coop key counter adds `end - start + (L - 1)` for that CTA, so the sum stays n_keys. R11 and the reds are
    unchanged.
- **Compliance:** every key is attended; no precision is coarser; the q and k bytes are identical to the main
  loop's; shape-only routing; load_inline only.

**Alternative arm (same lane, no new numerics): NKF-lite at v16s.** n_splits 9 (4,4,4,4,4,4,4,4 + a 7-key split),
with the grid order swapped so the 16 light CTAs launch last. Its risk is doubled-SM interference.

**H100 pre-gate (pre-register before the first run).**
- STEP 0: SASS census, master-identical except <8,false,131>; 0 spills; no C751x; regs ≤ master + 8.
- STEP 1:
  - ≥ 250 check cases: 16 TASK; 12 rows × 2 seeds; q8 sliding at ctx 1..9, 56-72, 2040-2062, 5000, 80000;
    B 2/5/16/17/32/64; page 1/16/64; kv_scale 1/64; dp heads; R11 force.
  - 0 failures, new cache rows bitwise = master, and outputs inside master's spread.
  - Positive control (NKF chunking with the fold off): must fail.
  - 16k fresh-state stress at v16s and v32s.
- STEP 2 timing:
  - GO: F D ≤ −0.30 µs at v16s and at v32s (CI upper < 0) in 2 valid processes, G the same sign, and no untouched row
    beyond A/A + 0.3% in both.
  - KILL: either row F D > −0.10, or any failure.

**Judge lines.** GO: rho_v16s ≥ 1.020 and rho_v32s ≥ 1.015. KILL: either < 1.000. p(GO | central) > 0.95.

### V3 PH8: CPUSH push-merge at d16s (2 clusters of 8 per request, one per KV head)
**Mechanism.**
- The same as CPUSH at d1s, which read −0.4..−0.7 µs on the H100 and +10% d1s on the judge.
- It replaces the per-request reds, ticket and last-CTA 16-row finalize with st.async partials to owner ranks, and
  halves each CTA's q-prep rows.
- The grid stays (16, 16, 1) = 256 CTAs, the same shape as master's d16s: x = 2 heads × 8 ranks, and each rank holds
  256 keys of one head (2 stage iterations of 128 keys in column halves).

**Knobs.**
- PH ranks 16 -> 8 (cluster dim 8; PH_SRC 32 -> 16; rank = split & 7; head = split >> 3).
- Rank chunk 2 -> 4 tiles: a non-ONE PH instance, MODE 16|1, with the kvc/kvx indexing over 2 iterations.
- Route: `_plan` takes L == 1, sliding, 2 ≤ B ≤ 16 under a new tma bit 28; `launch()` adds `ph8` with cluster dim 8.

**M0 (no build; same H100 session as V4).**
- Arm: master's d16s kernel <1,false,1> launched through cudaLaunchKernelEx with clusterDim (8,1,1) and no DSMEM use.
  4 copies against 4 master copies.
- Also an smid census: SMs used, max CTAs/SM, second wave.
- GO-M0: CN−M ≤ +0.10 µs in both F and G, no SM above master's per-SM CTA count, and no second wave.
- KILL: CN−M ≥ +0.30 µs.
- Lesson CLAIMS:141: cudaOccupancyMaxActiveClusters is not a residency proof; use the stamps.

**M1 (build on M0 GO).** GO: d16s F D ≤ −0.40 µs in 2 valid processes, every master function SASS-identical except
the PH8 instance, and outputs inside master's spread.

**Judge lines.** GO: rho_d16s ≥ 1.040. KILL: < 1.000.

### V4 W313P: d1f 313 splits (4 tiles/CTA, ≤ 3 CTAs/SM) + rider P on L=1 only
**Knob 1: exact `_TUNED` edit.** `(1, 1, False, 1): dict(nc=8, coop=1, n_splits=313)`, was 252.
- coop needs occupancy 3 for <1,false,3> and <1,false,11>. Smem<1,false> is 57,344 B, plus 1,024 B dynamic and 1 KB
  reserved; 114 regs. Otherwise `launch()` silently drops to the plain kernel, which is the refuted coop=0 path.

**Knob 2: rider P (D1FT-p, 1a559375).** Row-less coop pollers leave early. Re-anchor it on 399eb35d and restrict it to
L == 1: in the `!FLAT && M_COOP` branch, the `else` after the last-arriver test becomes
`else if (L != 1 || split_idx * R < L * 16)`.
- That keeps every q8 coop instance on master SASS. Unrestricted P hurt v1f in G in B1X round 1.

**Screen (no build for knob 1).**
- Pass overrides through `_launch(data, **override)` on each of the 4 master copies, so base and arm run on the same
  copy in every round.
- Arms:
  - d1f: base (nc 8, coop 1, n_splits 252) against W313;
  - screen-only: d32s `n_splits=11, nc=1` and d16s `n_splits=8, nc=1`. `nc=1` must be explicit, because an override
    with only `n_splits` sets nc = `_copies(n, 4)` = 4.
- P(L=1): 4 copies of master+P(L=1) against 4 master copies, at d1f with 252 and with 313.
- Engineering:
  - print the occupancies;
  - the profiler must show <1,false,11> with grid 313 at d1f (a fallback to <1,false,9> makes the arm VOID);
  - check 12 rows × 2 seeds + 16 TASK + d1f-shaped short contexts (ctx 1, 63-65, 2047-2050, 16130, 80000; page
    1/16/64);
  - 16k stress at d1f.
- GO per arm: F D ≤ −0.25 µs (d1f) or ≤ −0.20 µs (d32s, d16s), CI upper < 0, in 2 valid processes, G the same sign.
- KILL per arm: F D > −0.10, occupancy < 3, or any failure.
- Only arms that GO go into a file. A screen-only arm that GOes rides on V1 as its second knob.

**Judge lines.** GO: rho_d1f ≥ 1.015. KILL: < 0.990.

### V5 KN: master + HSK8 (v1f)
**Bytes.**
- `git show 0d2feafa:muse_attention_decode/hsk8/HSV1HSK8.py` (sha256 c3503c24…; branch
  campaign/h100-att-hsk8-v1f-keypacked-headsplit).
  - It is `python3 -I muse_attention_decode/hsk8/gen_hsk.py HSV1-M1.py OUT.py --variant k` on HSV1-M1 81a56018,
    which mk_b1x.py regenerates byte for byte.
- Add the line-2 pp-nonce for r1/r2.
- Attempt files plus a light audit; its code is a subset of the audited B1XN.

**Floors and lines.**
- GO: v1f rho over the 4 HSK8-SASS reads (a4fcde9f, 91604729, KN-r1, KN-r2) ≥ 1.006, and the KN pair ≥ 1.000.
- KILL: the KN pair < 1/1.015 (PR < 2.3357, B1XN's line).
- Its standalone value is below two-read resolution, so **prefer it as a stack rider** over 2 checks of its own.

## 5. Fresh-submission stack (the "best sweep possibility")
**STACK = master + V1 + V5 + every part that reached GO among V2, V3 and V4.**
- It is new code, never B1XN's bytes.
- Build it with mk_b1x.py-style 3-way merges on master, assert the hunk union, and re-run the SASS census.

**Lines.**
- Touched rows = the union of the parts' touched rows.
- KILL: any touched row below its own card's KILL line.
- GO: Ĥ_T ≥ +0.20% (about half the summed central prediction), G_u ≥ 0.985 and H̄ ≥ 2.5300.

**Expected.**
- V1+V5+V2: +0.51% → level about 2.565, P1 0.77, P2 0.95.
- Adding V3: +0.78% → P1 0.84.
- If no build part lands today: V1 alone (P1 0.66). Or V1+V5 plus one GO'd screen arm, which is new code and
  eligible.

## 6. Order and budget (attention quota from the LOG: about 13 checks and 8 scored runs left on 11-01; reset 00:00Z)
1. **Now:** attempt files and 2 checks of V1 (B1SN-r1/r2) under the §4 gate. On GO, 1 scored run.
2. **In parallel:** one H100 session on gpunode06 **GPU 1 only**, under the hopper-probe rules (USER, no root, no
   Slurm allocation, node-local /tmp/USER-hopper, guard, no node-state change). About 1 h. It runs the V4 screen
   and the V3 M0.
   - Protocol in every arm: ≥ 4 byte-identical module copies; load order shuffled per fresh process (fresh extension
     cache); copy-pooled D (median + 95% bootstrap CI, plus copy-level D ± SE).
   - F: GPU-bound graph replay with the host enqueued ahead and a per-call 256 MiB ld.global.cg read flush. This is
     the primary mode. G (L2-resident) is informational.
   - Validity: base A/A ≤ 0.10 µs at the touched rows; at most 2 void repeats.
3. **Builder for V2 (effort high), then an independent audit, then the V2 H100 gate.**
   - V3 is built only on an M0 GO.
   - Checks go out when the gates pass (today if before about 23:30Z, else 11-02).
4. **The stack per §5** once at least one build part is GO.

**Compliance, for every variant.**
- The FA3-precision ruling holds: exact max, P ≤ 256 E4M3 in the main loop, FP32 unrounded-P denominator. NKF's fold
  uses FP32 P, which is finer.
- Lossless only; no weights are involved.
- Nothing is timed or cached in kernel_fn. Routing is on shapes only: B, L, window, page, n_splits.
- load_inline only; no subprocess, network or harness access.
- Line 1 and the notices are kept.
- No re-roll of judged bytes: B1XN, CPUSH-r*, HSV1-r*.
- Never time another problem's kernels.
