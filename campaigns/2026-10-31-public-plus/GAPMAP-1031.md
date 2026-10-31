# GAPMAP-1031: attention, prefill and moe gap map (judge)

Judge pass at effort max for gap-map workflow wf_a3de48ce-c2f (task w7xx5jphg), run 2026-10-31 from 20:03Z to 20:27Z (stamps from the workflow journal; the judge wrote "20:05Z to 21:00Z"). Key tables of the three finder reports are in the appendix; the full reports are in `research/gapmap-1031/finder-{attention,prefill,moe}.md`.

**What changed after the finders ran**
- Attention C5L landed at 19:45Z: scored 1280e6f2 at 2.4880, master c5990d50.
- pf-tax returned NONE.
- PQ8 and R11C were killed.
- DC5L was killed.

The judge used these results throughout.

**Method**
- Boards were refetched at 20:10Z with `okc leaderboard <p> --json`; quota with `okc problems`. Both are read-only.
- The judge made no H100 runs. No finder number was disputed in a way that one short run could settle.
- Nothing in musespark was edited, and no check or submit was sent.
- "Verified" means the judge recomputed the number from the judge JSON, the TASK weights or the named run output (§1.3).

## 0. Verdict

**1. No problem reaches 10th place on the mapped mechanisms by 11-04 under our compliance ruling.**

| problem | 10th needs | mapped mechanisms reach |
|---|---|---|
| attention | x1.170 | x1.085 if every item lands; x1.025 p-weighted |
| prefill | x1.128 | about x1.078 realistic, now that pf-tax found nothing |
| moe | x1.087 | today's engine tops out near x1.070 |

**2. Physics does not rule 10th out anywhere. The gaps are structural.**
- Compliant floors: attention 3.27, prefill 2.23 (practical), moe 2.84.
- attention: per-request latency rows hold about 80% of the log gap.
- prefill: about 80 µs per small-T row goes to FP4→BF16 conversion of small MoE tiles. It runs in its own phase while the tensor pipes sit idle.
- moe: the skinny-row consumer is already at its own floor.

**3. Only one candidate in the three maps comes near the /hypothesize value bar.** That is prefill S1, co-scheduling the small-tile conversion under the big tiles. It is worth +3.8–4.2% if the H100 prize holds on the H200, at p 0.35–0.4.

**4. Keep decode first.**
- The overall board tracks decode rank. Prefill does not count toward it.
- Attention savings reach decode rows weakly (DC5L).
- From these maps, phase A gets the moe C1 gate (testbed, decode-portable) and, if the user agrees, prefill S1.
- Attention goes to phase B.

**5. Tonight: 5 diagnostic checks plus 1 reserve (§3).**
- kernel_fn stays byte-identical to master.
- Results are read from the saved job log, not from `log_tail`.
- None of them is ever scored.

## 1. Cross-problem table

### 1.1 The table

| | muse_attention_decode | muse_prefill_layer | muse_moe_w4a16 |
|---|---|---|---|
| Board, rank (20:10Z) | 2.4880, 296 of 1195 | 1.7836, 187 of 754 | 2.3867, 118 of 1252 |
| 10th / top | 2.9098 Lurrobert (x1.170) / 3.2221 tjiang-fair (x1.295) | 2.0125 gauravjain14 (x1.128) / 2.1551 GodlyDonuts (x1.208) | 2.5931 Fangzhou66 (x1.087) / 2.7578 dolanzhao (x1.156) |
| Compliant-floor headline | **3.27** with all 12 rows at FLOOR (3.2–3.35 for trip latency 0.40–0.55 µs) | **2.23** practical, at the judge's own best rates; 2.67 staged hardware floor | **2.84** stream+chain; 2.75 at 4.0 TB/s |
| Gap from board to floor | x1.31 | x1.25 (x1.50) | x1.19 |
| Gap to 10th | x1.170 | x1.128 | x1.087 |
| Where the gap sits | 8 per-request rows: ~80% of the log gap after C5L. Flat q8 ~4.7%, flat q1 ~0. | Small-T loads (t1, t4): MoE small tiles run in a separate phase with the tensor pipes idle, a ~80 µs prize per row. T=16384: GEMMs and MoE are at the power-capped limit. | d1/v1/d16 are at today's consumer floor. d32/v16/v32 are co-bound by bytes and consumer, plus ~12% structure. |
| Today's structure plus mapped items | ~2.70 if S1+S2+S3 all land (x1.085, rank ~51); ~2.55 p-weighted | ~1.92 realistic (x1.078); ~1.99 stretch without the gray exp-split | 2.55 engine ceiling (x1.070); ~2.58 with every chain fix |
| Best structural candidate | **S2**: B=1 latency path (d1s, v1s, d1f). Next: **S3**, the end of the flat-row kernel. | **S1**: a converter warpgroup runs the ≤64-row tiles under the big tiles (t1s, t1f, t4s, t4f). | **C1**: skinny engine with no A→B handoff (d1, v1, d16). Next: **C2**, end game moved to a second kernel. |
| Gain if it works × p = EV, own board | S2: +1.9% (−0.5 µs/row) to +3.8% (−1.0 µs/row) × 0.3 = **0.6–1.1%**. S3: +2.4% × 0.35 = 0.8%. | +3.8% (75 µs prize) to +4.2% (H100's 82 µs) × 0.35–0.4 = **1.3–1.7%** | C1: +2.0% planning (6.6% upper) × 0.3 = **0.6%**. C2: +0.8% × 0.5 = 0.4%. |
| Decode port, if the saving reaches the rows | S2 ≈ +0.35%; S3 ≈ +0.2% (the q8 part is 0, per DC5L) | none; prefill is not on the overall board | C1 ≈ +0.9% at planning (up to ~3%); C2 ≈ +0.5% |
| Transfer class | latency (the H100 L2-resident proxy is 1:1 on these rows); scheduling | compute/issue overlap. The judge's small-T MoE equals the H100's (t1s 285 vs 280–297 µs). | compute/issue plus latency; gate on R67/R50 |
| Build size | 200–400 lines (S2); 150–300 (S3) | 400–600 lines of CUDA | 900–1200 (C1); 150–250 (C2) |
| Rank if the best candidate lands | S2 +1.9% → 278; +3.8% → 243 | +3.5% → 54; +4.2% → 48 | C1 +2.0% → 57; upper +6.6% → 22 |

**How the decode-port column was computed.**
- Decode per-µs weights come from LEVERS-1031b's decode time map: d1s 0.177, v1s 0.114, d16s 0.083, d32s 0.067, d1f 0.062, v16s 0.046, v1f 0.040, v32s 0.035, d16f 0.029, d32f 0.023, v16f 0.016, v32f 0.012 %/µs.
- These are applied to the matching kernels in decode's scored DPP profile, 9d726d84.
  - moe_fused takes 23.0 / 22.1 / 78.7 / 50.7 / 59.8 / 72.1 µs at d1s / d1f / v1s / v1f / d16s / d16f. That is 28–60% of those rows' kernel time.
  - attn_kernel takes 4.9 / 7.3 / 11.1 µs at d1s / v1s / d1f.
- **C1:** a 4% cut of decode's skinny MoE gives +0.94%.
- **S2:** −1 µs on d1s, v1s and d1f attention gives +0.35%.

### 1.2 Rank cliffs (20:10Z boards, our entry excluded)

| headline gain | +1% | +2% | +3% | +5% | +7.5% | +10% |
|---|---:|---:|---:|---:|---:|---:|
| attention rank (now 296) | 287 | 275 | 265 | 169 | 69 | 36 |
| prefill rank (now 187) | 114 | 65 | 56 | 45 | 36 | 21 |
| moe rank (now 118) | 77 | 57 | 45 | 30 | 16 | 7 |

- **Prefill:** the cluster of public-file entries sits at 1.80–1.82. The public prefill reads 1.8167. A +2% gain jumps 120 places.
- **Attention:** rank barely moves until about +4–5% (2.59–2.61). It then climbs fast through entries up to ~2.67; the public attention file reads 2.653.
- **Overall board:** we are 111 of 1764. The API gives ranks only, and the top 50 get serving priority.
  - On 10-26, rank by decode alone matched the overall order with Spearman 0.996 over 632 teams (campaigns/2026-10-26-decode-first-rewrite/PLAN.md:44-47).
  - Prefill does not count toward the overall board (muse_prefill_layer/TASK.md:33).

### 1.3 What the rankings rest on (verified)

**Attention**
1. **Row times.** J1 14cec684, J2 ac0335f9 and J3 d933848f were recomputed from `result.records`.
   - The headlines reproduce exactly: 2.4729 / 2.5236 / 2.4515.
   - Row means match the finder's table to 0.01 µs.
   - Load = 46 × sliding + 16 × full; geomean over 6 loads (TASK.md:35-50).
2. **Floor arithmetic** on the 3-read row means (base 2.4822):
   - all rows at FLOOR: 3.269
   - per-request rows at FLOOR only: 3.046
   - v16f and v32f at FLOOR only: 2.639
   - 10th needs ln 0.159, which is 57.7% of the floor's log gap; the top needs 94.8%.
   - C5L's check-mean rows (v16f 106.0, v32f 194.4 µs) give x1.016.
3. **Proxy.** The H100 L2-resident d1s kernel is 4.51 µs against the judge's 4.6 µs (run 20261031-121900, gapatt-time-l2.json). The LEVERS-1031b attention time map has the per-request q8 L2-resident times matching H200 cold within ~7%.
4. **Latency.** The latbench run (20261031-123054) measured 0.379 µs per dependent cold trip (kernel) and 0.396 (graph). The empty kernel is 0.99 µs. Cold code costs +0.06 µs at 4 KB and +0.11 µs at 16 KB.
5. **Epilogue share** at v1s / v1f / v16s / v32s: removing the whole epilogue saves −19% / −37..−39% / −31..−34% / −36..−37% (LEVERS-1031b time map, H100 L2-resident).
6. **Tail shape.** The two H100 modes disagree:
   - L2-resident v32f (186.0 µs, run 123449): 48 warps for 29 of 40 bins, then a **40-warp plateau for about 33 µs** at 52% issue against 71%, then a drain.
   - Read-flush v32f (254.5 µs, run 123831): a gradual decline from 48 to 15 warps over the last ~6 bins, with DRAM falling from 86% to 2%.
   - S3 must attribute the tail before building (§2.3).

**Prefill**
1. **Row times.** J1 8b89517b, J2 d114d2e8 and J3 c456c4b9 were recomputed: 1.7836 / 1.7848 / 1.7634; the 3-read mean is 1.7773. Weights are 46/16/46/16/16/138/16/16 (TASK.md:35-42).
2. **Judge MoE kernel times (J1).** moe_fused_persistent takes 285.0 µs at t1s (47.3% of the row) and 788.4 µs at t4s.
3. **Prize, H100 run 20261031-125243 (gmpf-diag).**
   - t1s: all experts 297.5, big tiles only 215.6, small tiles only 110.4 → prize 81.8 µs.
   - t1f prize 78.0; t4s prize 82.2.
   - Small tiles with no X/W loads take 99.9 µs (0.90 of normal). On 33 CTAs they take 312.4 µs (x2.83).
4. **Scenario value** on the 3-read means. Capturing 75% of the prize on the four small-T rows gives:

   | prize per row | headline |
   |---|---|
   | 45 µs | x1.022 |
   | 60 µs | x1.030 |
   | 70 µs | x1.035 |
   | 75 µs | x1.038 |
   | 82 µs | x1.042 |

   The full prize on every row gives x1.057.
5. **pf-tax** (CLAIMS node-only; campaign/pf-tax 0aa805a3): the tax is +0.94% at most, and no compliant arm beats P2. The finder's S3a is therefore 0.

**Moe**
1. **Row times.** eaf35636 and f4e16629 were recomputed: 2.38673 and 2.38669. Each row has weight 60.
2. **Byte scaling**, run 20261031-125106. Paired-median ratios of R50 against R67: d1 1.020, v1 1.028, d16 1.042 (the ratio of medians is 1.008), d32 1.129, v16 1.111, v32 1.080. R33 against R50 is at most 1.03.
3. **NW** (run 123110), R67NW against R67: d1 1.043, v1 1.025, the rest 0.99–1.00.
   **NF** (run 123701), R67 minus R67NF: 3.6 / 4.0 / 6.5 µs at d32 / v16 / v32.
4. **Engine ceiling.** 2.3867 × exp(Σ ln(judge/R33)/6) over d32, v16 and v32 = 2.553.
5. **Judge against the H100 consumer floor.** The judge's d1 and v1 (25.0 / 63.3 µs) already run below the H100's R33 times (25.8 / 66.6 µs). So "at the consumer floor" depends on the H200 clock; DX1 measures it.

## 2. Tomorrow's phase A

### 2.1 Position

**Decode first stays right.**
- The overall board tracks decode rank. Prefill is not on it.
- DC5L cut the flat q8 kernel by 3.4–4.2 µs, yet decode's v16f and v32f rows did not move: 337.2/337.6 and 492.3/491.7 µs against DPP's 337–342 and 482–496 (LOG 19:41Z).
- B=1 attention carries about 10x lower per-µs weight in decode.
- Nothing in these three maps outranks a decode build that clears the bar. Decode's slot comes from GAPMAP-1031b.

**Attention should not go first.**
- Its gap to 10th is the largest, and its transfer record is the best (PP, C5L).
- But no attention candidate clears the value bar: best EV is 0.6–1.1%.
- The 8 per-request rows holding about 80% of its gap would each need about −30%.
- Its rank barely moves until about +5%.
- Attention goes to phase B with one builder. The H100 L2-resident proxy is accurate on its rows, so it is cheap to develop there.

**Prefill should go ahead of attention**, and ahead of where the standing phase-B order puts it (moe > bf16 > attention > prefill).
- S1 is the only candidate here within reach of the value bar.
- It is the class that transfers: compute overlap on the same SMs. The judge's small-T MoE time equals the H100's.
- +2% crosses the public-file cluster (rank 187 → 65).
- But it does nothing for the overall board, and it is a 400–600-line change to a persistent warp-specialized kernel. It is gated by D1 and S0.
- **Needs the user's OK**, because the standing phase-A scope is decode plus moe. Default without an answer: the standing order holds.

**Moe keeps the testbed slot, but only as a gate.**
- Its map has no item that clears the bar; summed EV is about +1.0% including the decode port.
- But C1 is the only item in these maps that can move decode's expert block at T ≤ 16.
- Run C1's M0 microbenchmark (3 h or less). Build only on GO, and only if DX2 does not show the skinny rows to be byte-bound.

**Hand-offs to decode's plan (GAPMAP-1031b):**
- DX2's verdict on the skinny-row regime applies to decode's MoE at T ≤ 16.
- DX1's clock reading applies to every consumer-bound prediction.
- Each attention port is worth at most about +0.35% on decode.

### 2.2 Build A1: moe C1 gate, then C1 (the testbed slot)

**Precondition.** DX2's r(d1) < 1.10. At ≥ 1.10 the skinny rows are byte-bound: drop C1 and reopen skinny byte cuts instead.

**Design panel (effort max).** The question: can a skinny NVFP4 consumer (N ≤ 16) reach at least 1.25x weights per µs over today's TINY/TINY2/MID consumer? Today's runs 18.44 instructions per weight word against a decode minimum of about 10–11. It must keep BF16 activations, exact dequant and FP32 accumulation.

Designs to compare:
- (a) Row-split warps over K-streamed 128×256 stages with mma.sync m16n8k16 and no cross-warp reduction.
- (b) (a) plus the K-fused A→B step: W2 K-slices repacked in prepare (a lossless permutation), with FP32 red.add into ybuf. The red.add traffic must stay under the d1 handoff gain of 1.6 µs.
- (c) Control: an instruction diet on today's engine, in the style of KPERM3. KPERM3 cut 8.3% of instructions per word for +1.25% on the board.

Inputs:
- the moe finder's C1;
- LEVERS-1031b's moe dead ends (NW16, LLA, NS4/NS5, FACQ, PAIR, PBLL, FIN2/F1);
- a read-only diff of decode's MoE builds against moe's (§4 Q8), which fixes the port path.

**Stages and kill rules.** Build and audit run at effort high.

| stage | test (H100) | GO | KILL |
|---|---|---|---|
| M0 | warm-L2 consumer microbenchmark at N = 1 / 8 / 16, R50 mode | at least 1.25x weights/µs | below 1.15x |
| M1 | TINY replacement at d1 | R50 at least +8% and R67 at least +5%; 0 failures in 600 checked replays; SASS of the other builds unchanged | R67 below +3% |
| M2 | TINY2 and MID (v1, d16) | the same gates as M1 | the same as M1 |

**Audit and checks.**
- Independent audit: outputs match the incumbent apart from red.add summation order (the same order class as master's atomics). Also adversarial bs 1–17, graph capture and replay, and prepare ≤ 280 s.
- 2 moe checks. The auditor pre-registers the gate: headline mean ≥ 2.3867 × 1.012, and no row slower than its board row by more than 3%.
- The decode port becomes a decode phase-A item the day after a moe GO.

### 2.3 Build A2: prefill S1, the small-tile converter co-schedule

**Preconditions.** PF-D1 returns GO (§3, check #2), the S0 headroom check passes, and the user OKs a phase-A slot.

**Design panel (effort max).**
1. A fourth "converter" warpgroup for tiles of ≤ 64 rows, against folding small items into the consumers' idle issue slots.
2. setmaxnreg budget: 40 / 96 / 168 / 168 registers, about 60.4K of 64K.
3. Shared memory: NSLOT drops from 4 to 3 (213 of 227 KB today; submission.py:2061-2063), plus 2 × 16 KB converter buffers. First measure big-only at NSLOT 3; it must stay within +2%.
4. Bit-identity of SS-WGMMA m64n32/64k16 against the incumbent path: the same conversion, the same k16 order, FP32 accumulation.
5. A two-queue plan by tile rows, plus a prepare-time race per T class against the incumbent on skewed routing (prefill CLAIMS:83 rule), with the incumbent as fallback.
6. The new kernel stays CUDA-graph-capturable.

**Stages and kill rules.**

| stage | test | GO | KILL |
|---|---|---|---|
| S0 (≤ 1 h, H100) | ncu of gmdiag's big-only arm with sm__inst_executed_pipe_{alu,fma,fp16} and issue-active (§4 Q3) | issue-active ≤ 75% and the conversion pipes ≤ 50% busy in the big phase | otherwise; record a node-only CLAIMS line |
| S1 | small-only converter path at t1s | ≤ 105 µs (incumbent 110.4) and bit-identical | — |
| S2 | both roles together on the D1 harness | t1s all ≤ 235 µs and t4s all ≤ 755 µs; fts no worse than +0.5% | t1s gains less than 30 µs |
| S3 | independent audit | bit-identity on TASK tests plus skewed and adversarial routings; capture; prepare ≤ 280 s | — |
| S4 | 2 prefill checks, gate pre-registered | turn-1 load ≥ +4% against the 3-read control, and headline mean ≥ 1.7836 × 1.015 | — |

At S1: a bit difference against the incumbent gets one fix round, then KILL.

**Closest prior work.**
- prefill CLAIMS:86: MOE_ORDER_SMALL=2, sequential within each SM; R2 measured only −6.4 µs.
- CLAIMS:79 / LESSONS:180,185: 2 CTAs per SM lost under skew. S1 keeps popular experts on full slabs.
- CLAIMS:70/71: the same diagnosis on the old engine.
- The zero-change SM split (R5): −3.8 µs at t1s, +6.1 at t1f, +20.1 at t4s.

### 2.4 Phase B: attention S3, then S2

One builder at effort high, on master c5990d50 (C5L), using the H100 L2-resident proxy.

**S3: end of the flat-row kernel**
- **M0 (≤ 1 h):** per-CTA entry, segment and exit `%globaltimer` stamps plus smid, in both L2-resident and read-flush modes.
  - GO if at least 60% of the tail is imbalance plus barrier/finalize, and check #3's intercept is ≥ 8 µs (or 4–8 µs and M0 agrees).
  - KILL if the tail is the fixed 40-warp phase (producer exit or consumer drain) rather than imbalance.
- **M1:** per-request last-arriver finalize, with no grid barrier.
  - GO at ≤ −3 µs on v16f in a paired L2-resident A/B.
  - KILL above −1.5 µs.
- **M2:** cost-weighted partition.
- Closest refuted item: AT6 (attention CLAIMS:88). S3 differs by removing the barrier and by sizing its partition from stamps.

**S2: B=1 latency path**
- **M0:** phase stamps at d1s, v1s and d1f. GO if at least 1.0 µs of non-chain time is left at d1s; check #5 may answer this first.
- **M1:** cut the largest phase. KILL above −0.4 µs at d1s on the proxy.
- It must avoid the refuted B=1 variants: AT2W4, AT3, AT3c, AT5c and AT8G (attention CLAIMS:104-118).

### 2.5 Parked or held

- **attention S1** (DSMEM q8 split merge): shelf.
  - LEVERS-1031b §3 says every compliant variant of the per-request q8 epilogue measured neutral or worse. PQ8 died today on the same rows. AT5c added a cluster level and lost.
  - S1 must name its difference from all of these before any build. The judge's p is 0.2 (EV ≈ 0.6%).
- **prefill S2** (boundary passes fused into GEMM epilogues): park.
  - It needs a custom BF16 GEMM at ≥ 0.97x cuBLAS. CLAIMS:77 records 0.53–0.86x, and LESSONS:133 records an epilogue fusion that lost 1–3%.
  - Judge's p is 0.2, giving EV 0.5–1.0%.
- **moe C2** (end game in a second kernel): a phase-B/C build only if DX1 shows E(256) ≥ 10 µs and E(128) ≥ 6 µs. It replaces the B4 shelf item. C3 rides on C2's check.
- **prefill exp-split** (S3b: about 15% of exp2 on the FMA pipe as a polynomial, about +1.1%): it changes the exp implementation. It needs the user's ruling before anything is built; the shelf line is prefill CLAIMS "full-layer FA3 with a fraction f of exp2 on the FMA pipe".

## 3. Tonight's diagnostic checks: 5 plus 1 reserve

Checks left at 20:20Z: attention 12, prefill 12, moe 12. All three reset at 00:00Z.

**Rules for every diagnostic check**
1. **No timed-path changes.** kernel_fn, its call graph and the timed extension stay byte-identical to master; the first GAPDIAG line prints the sha256 of the timed sources.
   - For moe and prefill, the diagnostic lives at the end of `prepare()`.
   - Attention has no prepare (rules/technical-rules.llms.txt:98), so its diagnostic is one guarded module-level function. Import-time GPU races have precedent in our earlier attention files.
2. **Clean up.** try/except/finally restores every `st` entry and global, re-zeroes attention's `_WS` in finally, frees tensors, then synchronizes and calls `empty_cache()`.
   - Inputs are synthetic (attention) or prepare's own weights (moe, prefill).
   - Nothing reaches the timed path. No timing patches, harness access, subprocess, network or reference import.
3. **Timing-only builds** (R50, debug) get distinct extension names, are unreachable from kernel_fn, and their outputs are never used.
4. **Output** is at most about 4 KB of `GAPDIAG` lines, printed last and flushed.
   - **Read them from the log that `okc status <job> --json` saves** (`output.log`; the judge keeps the last 16,000 bytes).
   - `result.log_tail` holds only 1–2 KB of profiler USDT lines (checked on 14cec684 and eaf35636), so the finders' "read log_tail" would return nothing.
5. **Time budgets.**
   - moe: skip the diagnostic if prepare is already past 150 s, and skip the runs if compiles finish after 260 s. Master's prepare is 163.6–168.2 s, and each pair of extra builds adds about 50 s.
   - attention: the import-time diagnostic takes at most 120 s.
   - prefill: at most 5 s.
6. **Never scored.** The check's headline is one more read of master's bytes. Record it in RESULTS.tsv as a control marked "diag".
7. **Compliance first.** Each check needs an attempt file with the rules checklist and an independent audit before dispatch.
8. **Validity control.** Each diagnostic has one. A void diagnostic gets at most one re-send, from the reserve.

**The checks**

| # | quota | diagnostic | carries | validity control | decides for tomorrow |
|---|---|---|---|---|---|
| 1 | moe | **DX2 + DX3** | TINY-R50 and MAIN-R50 timing-only builds (the 8 TMA sites set to 18432 B, as in probes/gapmoe/sub_R50.py); 40 interleaved paired reps at d1, d32, v16 and v32 (graphs, 256 MiB read flush, events); read ceilings at 73/244/325/444/649/703 MB and 2 GiB; smmap | master-arm medians within ±10% of the judge rows (25.0 / 125.8 / 174.6 / 209.2 µs) | whether A1 runs; whether skinny byte cuts stay closed for moe and decode's MoE; C3(a); the moe floors |
| 2 | prefill | **D1**, with D2 and D3 riding | gmdiag.d1, d3epi and d3glue (takemyjob hopper/gapmap-prefill b1cb10a; drop the `m.` prefixes), run after attn_choice | "all experts" within ±10% of J1's moe_fused at t1s (285.0 µs) and t4s (788.4 µs) | whether A2 runs (and where); how S2 is costed; the glue rider |
| 3 | attention | **D1 intercept + D3 gather** | master's kernel_fn on synthetic v16f, v32f, d16f and d32f, cold (read sweep, **not** a copy flush) and L2-resident; cold at ctx 40k/80k/120k for (16,8) and (16,1); gather variants on two 704 MB pools, each moving ≥ 650 MB | cold v16f/v32f within ±8% of the C5L judge rows (105.3–106.7 / 193.0–195.8 µs); d16f/d32f within ±8% of 86.0 / 164.1; cold ÷ L2-resident ≥ 1.3 at d16f/d32f | the scope of S3; whether flat q1 is closed; the HBM floors |
| 4 | moe | **DX1** | TINY and MAIN with `OKC_DEBUG 1` (assert exactly one replacement) plus `clock64` in DBG; bs 1/32/128/256, 12 reps | stamped spans within ±10% of #1's master arms | C2; the H200 in-kernel clock, which scales every consumer-bound prediction |
| 5 | attention | **D2 latency calibration** | latbench chain at n = 0/1/2/3/4/8; 4/16/64 KB SASS cold vs hot; d1s, v1s, d1f and v1f cold, and as the second of two back-to-back graphs on disjoint inputs (code warm, data cold) | synthetic d1s cold kernel within ±10% of 4.6 µs | the scope and order of S2 |
| 6 | reserve | one re-read of #1, #2 or #3 | — | — | only when the deciding number lands in its ambiguous band or a validity control fails once |

**Pre-registered rules**

**#1 (moe DX2 + DX3).** Let r(row) = master time ÷ R50 time.

| result | action |
|---|---|
| r(d1) ≤ 1.03 | Skinny rows are consumer-bound on the H200. Close skinny byte cuts for moe and for decode's MoE at T ≤ 16. A1 runs. |
| 1.03 < r(d1) < 1.10 | A1 runs at half value; byte cuts stay closed. Reserve re-read allowed. |
| r(d1) ≥ 1.10 | Skinny rows are byte-bound. Drop C1 and reopen skinny byte levers for moe and decode. |
| r(v32) ≥ 1.08 | C3(a), one HBM pass per hot expert, becomes C2's rider. |
| r(v32) ≤ 1.03 | Close C3(a) and every WGS byte lever, including S6's reopen. |
| DX3: ceilings at 444 and 703 MB ≥ 4.40 TB/s | The moe floors stand (2.835). |
| DX3: ceilings ≤ 4.10 TB/s | Recompute: the WGS gaps shrink by at least 7%, and C2/C3 drop in priority. |
| DX3: Spearman ≥ 0.5 and p90/median ≥ 1.03 | A rate-weighted static split goes on the lever list. Otherwise the slow-SM idea is closed. |

**#2 (prefill D1, with D2 and D3).** "Prize" = all experts − max(big only, small only). "No-load ratio" = small with X/W loads skipped ÷ small.

| result | action |
|---|---|
| prize ≥ 75 µs at both t1s and t4s, and no-load ratio ≥ 0.80 | GO for A2 (about +3.8% at 75% capture). Phase A if the user agrees. |
| prize 45–75 µs | A2 becomes a phase-B build (+2.2–3.8%). Reserve re-read allowed. |
| prize 30–45 µs | Shelf. |
| prize < 30 µs at t1s | KILL S1, with a node-only CLAIMS line: the H100 prize does not transfer. |
| no-load ratio < 0.65 | The H200 small set is load-bound: re-plan S1 as a bytes-in-flight design. |
| D2 rider: the unpooler shape hides ≥ 60% of the epilogue traffic | Cost S2 at x1.049. It stays parked behind its own cuBLAS-parity milestone. |
| D2 rider: both shapes hide ≤ 30% | Cost S2 at x1.024. |
| D3 rider: gather or combine ≤ 0.78 of the H200 copy ceiling | A phase-C bandwidth rider (about +0.2–0.3%). |
| D3 rider: both ≥ 0.85 of the ceiling | The glue-bandwidth question is closed. |

The finder's GO threshold was 60 µs. That is about +3.0% at 75% capture, an EV near 1.05%, below the value bar. 75 µs ties the GO to EV ≈ 1.5% at p 0.4.

**#3 (attention D1 + D3).** Fit cold time = a + b·MB over the three contexts; a is the fixed cost per call.

| result | action |
|---|---|
| a (v16f shape) ≥ 8 µs | S3's tail work goes ahead in phase B. |
| a ≤ 4 µs | S3 is dropped after its M0 stamps. |
| 4 µs < a < 8 µs | M0 decides. Reserve re-read allowed. |
| r = cold ÷ L2-resident at v16f/v32f | Recorded for the floor model only. If r ≥ 1.15, add a prefetch-depth arm to S3's M1. |
| D3: best gather variant ≥ 1.08 × the control's TB/s | A flat-q1 gather rewrite becomes a phase-B candidate (about +2.2% at −8% on d16f/d32f) and a lead for decode, where attention is 41–54% of the d16f/d32f rows' kernel time. |
| D3: best variant ≤ 1.03 × control | Close flat q1 and fix the HBM floors at the measured rate. |

**#4 (moe DX1).** E = final − B_done (median).

| result | action |
|---|---|
| E(256) ≥ 10 µs and E(128) ≥ 6 µs | C2 is a phase-B build. |
| E(256) < 6 µs | Drop C2; B4 stays shelved. |
| joined(256) ≥ 6 µs | C3(b), no global x gather, joins C2. |
| in-kernel clock ≤ 1.80 GHz | The consumer is power-throttled. Keep the H100 R33 floors and weight consumer levers (C1, KPERM-style) by 1.1. |
| in-kernel clock ≥ 1.95 GHz | No throttling. The H200's consumer floor sits about 7% under the H100's R33: recompute the 2.553 ceiling. |

**#5 (attention D2).**

| result | action |
|---|---|
| trip ≥ 0.55 µs | Recompute the floors and scale S2's value by (4.6 − new floor) ÷ 1.8. |
| cold − warm-code ≥ 0.5 µs at d1s or v1s | S2's first build is hot-path compaction. |
| cold − warm-code ≤ 0.2 µs | S2 targets the merge chain only. |
| d1s(cold) − (empty + 2·trip) < 1.0 µs | Drop S2. |
| d1s(cold) − (empty + 2·trip) ≥ 2.0 µs | S2 is the second attention build in phase B. |

**Build order and staffing**
- Build #2 first (it is nearly paste-ready and was dry-run in R7/R8) together with #1. Then #3, then #4 and #5.
- One builder per problem (at most 3 agents) and one auditor per file.
- Anything not audited and dispatched by 23:40Z is dropped.

**Rejected**
- **Using attention D1's r as a gate** (r ≥ 1.08 → scale consumer-lever predictions by l2res/cold). The record contradicts it:
  - C5L's L2-resident ratio of 0.943 / 0.937 read −6.2 / −6.6% on the judge.
  - PP's −4..−5% read −5.4 / −9 µs.
  - Flat q8 is consumer-bound for practical purposes. r is recorded, not used as a gate.
- **Prefill D2 as a phase-A trigger.** cuBLAS addmm absorbing a C read is a weak proxy for S2's real risk, a custom GEMM at cuBLAS parity. It rides along in #2 for costing only.
- **Separate checks for DX3, prefill D2 and prefill D3.** They ride in #1 and #2 at no extra check.
- **DX1 and DX2 in one check.** Four extra builds take prepare to about 270 of the 300 s limit.
- **A glue/GEMM stream co-run on the H200.** The H100 answered it (−3%), and the cause, SM residency, is the same on the H200.
- **Default re-reads.** None beyond the one reserve.
- **Decode and bf16 diagnostics.** GAPMAP-1031b owns them.

## 4. What none of the finders could answer, and the cheapest answer

| # | open question | why it matters | cheapest answer |
|---|---|---|---|
| Q1 | The H200's graph-mode timeline for any row | Every "where the time goes" here is an H100 graph timeline plus per-class transfer ratios. The judge's profile pass is one eager call (LEVERS-1031b instrument note). | In-kernel `%globaltimer` stamps from a prepare/import diagnostic on the H200. DX1 (#4) is the template; one check per problem. |
| Q2 | The H200's in-kernel SM clock under load | The judge's moe d1/v1 already run under the H100's R33 "floor" (25.0 vs 25.8, 63.3 vs 66.6 µs), so the skinny-floor claim depends on clock. | DX1's clock64/globaltimer line (#4). Add the same one-line probe to every stamping build. |
| Q3 | Can the prefill big-tile phase host the small-tile conversion? (issue headroom) | It sets S1's p (0.35 or 0.45). The morning ncu (lev-prefill 20261031-013401) has only tensor and XU metrics for moe_fused. | One H100 ncu of the big-only arm with the ALU/FMA/FP16 pipes and issue-active, about 15 min. This is A2's S0. |
| Q4 | Do kernel savings reach decode's rows? | The decode-port column assumes they do on serial kernels. DC5L shows flat attention rows may not. | Wave 2's H100 graph-mode nsys of decode v16f/v32f (running). For B=1 rows the judge shows no kernel overlap at d1s/d1f (LEVERS-1031b), so likely yes. |
| Q5 | The cause of the flat-row tail: a 40-warp plateau for ~33 µs when L2-resident, a gradual 48→15 decline when cold | Decides whether a finalize or partition change can touch it | S3's M0 per-CTA stamps on the H100, 1 h or less |
| Q6 | The H200's ~+10% excess over the H100 L2-resident proxy on flat q8 (after C5L: judge ~105/~194 µs against proxy ~96/~174) | It sets the flat-q8 floor | #3 sizes it (r). Attributing it needs an H200 stamp-and-clock build of the q8 kernel; not worth a check tonight. |
| Q7 | The judge's real routings (prefill distinct-expert counts, moe draws per seed) | The replica error in S1's prize and in moe's floors | Not observable. The ±10% validity controls in #1 and #2 bound it. |
| Q8 | Do decode's MoE builds share moe's TINY / TINY2 / MID code? | C1's decode port | A 15-minute read-only diff of the two files' MoE sources |
| Q9 | Do the leaders' attention scores need precision outside our ruling? | Whether 10th is reachable compliantly in practice | Not answerable without their code. The compliant floor of 3.27 shows physics does not exclude it. |
| Q10 | Is the prefill exp-split compliant? | About +1.1% prefill | A user ruling; there is nothing to measure until then |

## 5. Judge corrections to the finder reports

1. **Attention board.** It is 2.4880 since 19:45Z (C5L, 1280e6f2). The finder's "C5L +1.6% on top" is banked: +0.6% scored, +2.0% on the check mean. S3's tail gains are unchanged in µs; re-costed on post-C5L load times they give +2.4%.
2. **Prefill S3a is dead.** pf-tax found at most +0.94% and no compliant arm beat P2. Without it, realistic prefill is about 1.92 and the stretch is about 1.99 (the stretch excludes the gray exp-split). 10th place is out of reach on mapped items under the current ruling.
3. **Attention S1.** It must state its difference from LEVERS-1031b §3, from PQ8 (killed today) and from AT5c. The judge sets p at 0.2. S2's planning value is −0.5 µs per row (+1.9%); −1.0 µs (+3.8%) is the upper bound.
4. **Diagnostic read-back.** The GAPDIAG lines are in the saved job log (16,000-byte tail), not in `result.log_tail`.
5. **Attention D1's r-rule** is superseded by the PP and C5L transfers (§3, Rejected).
6. **Moe d16 byte elasticity is unresolved.** The run's paired-median ratios give 1.042 (the ratio of medians gives 1.008), against the report's 1.029. R50's interval [1.387, 1.461] covers R67's. The conclusion (skinny rows ≪ WGS rows) stands.
7. **Moe 10th place** is now Fangzhou66 at 2.5931 (a3fb6ef4); it needs x1.0865.
8. **Prefill D1's GO threshold** moves from 60 to 75 µs to tie it to the value bar (§3, #2).
9. **Attention D-designs.** Use a read sweep, never `copy_`, for cold timing. The copy flush adds 7.4–8.6 µs to H100 flat rows (attention finder, method finding 1). This belongs in LESSONS.

## Appendix A. Attention finder: key tables (as reported)

Elasticity e_i = 100·w_i/(6·T_load) %/µs. Share of log gap = (w_i·t_i/T_load)/6 · ln(t_i/FLOOR_i). FLOOR = max(HBM, chain, MUFU, tensor) + 0.8 µs of graph residue. These are pre-C5L rows.

| row | w | judge µs (yours): J1 / J2 / J3, then mean | kernel µs (judge profile, J1/J2/J3) | kernel, grid × block | H100 L2-res kernel (R1) | MB | HBM floor at 4.0 (4.8) TB/s | chain floor | **FLOOR** (yours) | gap µs | gap × | %/µs | share % |
|---|---:|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| d1s | 46 | 5.36 / 5.60 / 5.43 → **5.47** | 4.6/4.6/4.6 | <1,F,13> 32×128 | 4.51 | 0.5 | 0.1 (0.1) | 2.80 | **3.60** | 1.87 | 1.52 | 1.800 | 4.11 |
| d1f | 16 | 10.65 / 10.85 / 11.20 → **10.90** | 9.9/9.8/9.9 | <1,F,11> 252×128 | 9.95 | 20.8 | 5.2 (4.3) | 7.40 | **8.20** | 2.70 | 1.33 | 0.626 | 1.94 |
| d16s | 46 | 8.57 / 8.33 / 8.89 → **8.60** | 7.0/7.1/7.1 | <1,F,1> 16×16×128 | 7.36 | 8.6 | 2.2 (1.8) | 4.76 | **5.56** | 3.04 | 1.55 | 0.433 | 1.62 |
| d16f | 16 | 87.26 / 83.75 / 86.98 → **86.00** | 85.9/81.3/84.7 | <1,T,0> 264×256 | 61.76 | 332.9 | 83.2 (69.4) | 84.23 | **85.03** | 0.97 | 1.01 | 0.151 | 0.15 |
| d32s | 46 | 10.67 / 10.01 / 10.39 → **10.35** | 9.2/9.1/9.0 | <1,F,1> 8×32×128 | 9.23 | 17.3 | 4.3 (3.6) | 6.92 | **7.72** | 2.64 | 1.34 | 0.247 | 0.75 |
| d32f | 16 | 167.00 / 159.67 / 165.74 → **164.14** | 168.1/160.1/166.6 | <1,T,0> 264×256 | 117.97 | 665.8 | 166.5 (138.7) | 167.46 | **164.14** (at ceiling) | 0 | 1.00 | 0.086 | 0.00 |
| v1s | 46 | 6.69 / 7.02 / 7.28 → **7.00** | 6.5/6.3/6.4 | <8,F,15> 33×256 | 6.69 | 0.6 | 0.1 (0.1) | 3.40 | **4.20** | 2.80 | 1.67 | 1.361 | 4.86 |
| v1f | 16 | 14.85 / 15.08 / 15.37 → **15.10** | 14.3/14.5/14.7 | <8,F,11> 252×256 | 14.94 | 20.9 | 5.2 (4.3) | 7.81 | **8.61** | 6.49 | 1.75 | 0.473 | 4.01 |
| v16s | 46 | 12.46 / 12.30 / 11.88 → **12.21** | 11.2/11.5/11.4 | <8,F,3> 8×16×256 | 11.42 | 9.4 | 2.4 (2.0) | 5.36 | **6.16** | 6.05 | 1.98 | 0.324 | 2.71 |
| v16f | 16 | 113.71 / 111.59 / 112.81 → **112.70** | 112.4/112.7/112.4 | <8,T,0> 264×384 | 102.77 | 333.7 | 83.4 (69.5) | 84.43 | **85.23** | 27.48 | 1.32 | 0.113 | 3.55 |
| v32s | 46 | 15.81 / 15.95 / 16.01 → **15.92** | 15.0/14.8/15.3 | <8,F,3> 4×32×256 | 15.41 | 18.9 | 4.7 (3.9) | 7.72 | **8.52** | 7.41 | 1.87 | 0.190 | 1.90 |
| v32f | 16 | 207.28 / 204.83 / 206.15 → **206.09** | 205.5/205.2/205.0 | <8,T,0> 264×384 | 186.09 | 667.4 | 166.9 (139.0) | 167.86 | **168.66** | 37.43 | 1.22 | 0.066 | 2.73 |

| Group | Rows | Share of the log gap (pre-C5L) | What bounds it on the H200 |
|---|---|---|---|
| B=1 per-request | d1s, v1s, d1f, v1f | 14.9% | latency chain |
| B>1 per-request sliding | d16s, d32s, v16s, v32s | 7.0% | 3-trip chain plus the q8 epilogue |
| flat q8 | v16f, v32f | 6.3% | consumer plus tail, at 2.97 / 3.24 TB/s |
| flat q1 | d16f, d32f | 0.15% | HBM, at the ceiling |

| row | judge kernel | H100 read flush (R5) | copy flush (R2) | L2-res (R4) | ncu DRAM read rate (R1) | issue active | XU (MUFU) | achieved / theoretical occupancy | top stalls (cycles per issue) | tail deficit: cold DRAM (R5) / L2-res issue (R4) |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| v32f | 205.2 | 254.5 | 261.9 | 186.0 | 2.58 TB/s, 77% | 48.8% | 35.0% | 36.7 / 37.5% | long_sb 5.41, barrier 1.63, wait 1.53 | 21.6 µs (8.5%) / 18.6 µs (10.0%) |
| v16f | 112.5 | 135.9 | 144.4 | 102.2 | 2.41 TB/s, 73% | 48.8% | 34.1% | 36.2 / 37.5% | long_sb 4.90, barrier 1.90, wait 1.52 | 17.4 µs (12.8%) / 12.7 µs (12.4%) |
| d32f | 164.9 | 242.2 | 250.8 | 117.6 | 2.75 TB/s, 82% | 29.7% | 5.3% | 25.0 / 25.0% | long_sb 9.57, wait 1.31 | 11.0 µs (4.6%) / 10.0 µs (8.5%) |
| d16f | 84.0 | 124.1 | 132.3 | 61.8 | 2.64 TB/s, 79% | 29.4% | 5.2% | 24.7 / 25.0% | long_sb 9.44, wait 1.32 | 9.6 µs (7.7%) / 7.7 µs (12.5%) |
| d1s | 4.6 | 5.02 | 5.18 | 4.51 | 0.09 TB/s | 10.5% | 1.7% | 6.3 / 18.75% | barrier 2.28, wait 1.37, short_sb 1.31, long_sb 1.17, no_instr 1.04, imc_miss 0.69 | n/a |
| v1s | 6.4 | 7.02 | 6.94 | 6.69 | 0.08 TB/s | 23.2% | 7.1% | 12.5 / 25% | barrier 2.43, wait 1.25, short_sb 1.22, no_instr 0.91 | n/a |

H100 runs are on takemyjob branch hopper/gapmap-attention (57553eb): 20261031-121900 / 122655 / 123054 / 123449 / 123831-gpunode06.

## Appendix B. Prefill finder: key tables (as reported)

| row | w | judge µs J1 / J2 / J3 (mean) | SGLang | tensor / attn / HBM floor µs (MB) | floor B | gap | practical | gap | load share | ln-share vs floor B | headline if this row alone reaches practical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t1s | 46 | 601.1 / 582.1 / 584.7 (589.3) | 1110.4 | 290 / 6 / 167 (668) | 332 | x1.78 | 394 | x1.50 | 0.641 | 0.123 | +8.3% |
| t1f | 16 | 963.7 / 941.5 / 941.0 (948.7) | 1515.5 | 281 / 232 / 172 (688) | 550 | x1.73 | 676 | x1.40 | 0.359 | 0.065 | +3.7% |
| t4s | 46 | 1842.7 / 1801.6 / 1812.3 (1818.8) | 3399.5 | 1167 / 24 / 237 (949) | 1283 | x1.42 | 1513 | x1.20 | 0.616 | 0.072 | +3.7% |
| t4f | 16 | 3251.4 / 3235.8 / 3271.5 (3252.9) | 4965.9 | 1141 / 926 / 264 (1057) | 2160 | x1.51 | 2652 | x1.23 | 0.384 | 0.052 | +2.5% |
| ftf0 | 16 | 9937.3 / 9621.1 / 9904.8 (9821.1) | 18225.5 | 6652 / 514 / 552 (2208) | 7586 | x1.29 | 8976 | x1.09 | 0.084 | 0.007 | +0.2% |
| fts | 138 | 8990.8 / 8964.6 / 9197.4 (9050.9) | 17097.3 | 6332 / 128 / 581 (2322) | 6881 | x1.32 | 8106 | x1.12 | 0.668 | 0.061 | +2.4% |
| ftf32 | 16 | 13228.1 / 12498.2 / 13069.7 (12932.0) | 21371.6 | 6497 / 2568 / 561 (2245) | 9486 | x1.36 | 11455 | x1.13 | 0.111 | 0.011 | +0.4% |
| ftf64 | 16 | 16299.0 / 15826.8 / 16091.1 (16072.3) | 24945.0 | 6341 / 4622 / 570 (2282) | 11384 | x1.41 | 13932 | x1.15 | 0.137 | 0.016 | +0.6% |

J1 kernel split in µs (GEMM / MoE / attention / glue): t1s 209 / 285 / 39 / 69; t1f 270 / 328 / 746 (attn_fwd 373 + combine2 373, overlapping through PDL; real span ~380) / 72; t4s 771 / 788 / 62 / 225; t4f 857 / 799 / 1511 / 221; ftf0 4077 / 3994 / 874 / 1080; fts 4133 / 3622 / 310 / 1107; ftf32 4091 / 3869 / 4205 / 1084; ftf64 4078 / 3617 / 7283 / 1075.

| scenario (as reported, before pf-tax returned NONE) | x headline | about |
|---|---|---|
| S1 alone (realistic / at the full measured prize) | 1.041 / 1.057 | 1.856 / 1.886 |
| S2 with 35% hidden (H100 proxy) / 70% / 90% | 1.024 / 1.049 / 1.064 | |
| S3a: compliance tax recovered / plus exp-split (gray) | 1.014 / 1.025 | |
| **Realistic:** S1 + S2 at 35% + S3a + small-T knobs | **1.093** | **1.95** |
| **Stretch:** S1 at the full prize + S2 at 70% + attention ×0.88/×0.78 + knobs | **1.153** | **2.06** |
| Everything the map finds, including the gray exp-split and S2 at 90% | 1.185 | 2.11 |
| Practical floor / floor B | 1.257 / 1.503 | 2.234 / 2.671 |

Judge note: with S3a at 0 (pf-tax NONE), realistic becomes ~x1.078 (~1.92), and the stretch without the exp-split is ~x1.115 (~1.99).

H100 runs are on takemyjob branch hopper/gapmap-prefill (b1cb10a): R1 20261031-121653, R2 123049, R3 123813, R5 124409, R6 124708, R7 125243, R8 125840-gpunode06.

## Appendix C. Moe finder: key tables (as reported)

| row (bs) | w | judge us: eaf35636 / f4e16629 (mean) | near-master us: b3510dcc / 99b14fed / 0882294e | SGLang us | row ratio | judge-profile moe_fused us | MB moved | floor @4.0 / @4.8 TB/s / judge SOL | H200 size-matched ceiling → stream+chain floor | FLOP floor | gap us (x) | share of headline gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 (1) | 60 | 25.19 / 24.87 (25.03) | 26.05 / 25.06 / 24.91 | 77.3 | 3.088 | 29.4 / 23.2 | 73 | 18.3 / 15.2 / 11.4 | 3.76 → 21.9 | 0.3 | 3.1 (1.141) | 12.8% |
| decode 16 (16) | 60 | 91.33 / 91.87 (91.60) | 91.41 / 97.36 / 91.79 | 186.4 | 2.034 | 77.1 / 84.2 | 325 | 81.3 / 67.8 / 52.5 | 4.38 → 76.7 | 2.6 | 14.9 (1.194) | 17.2% |
| decode 32 (32) | 60 | 127.90 / 123.73 (125.82) | 125.21 / 124.46 / 126.70 | 209.7 | 1.667 | 119.2 / 143.6 | 444 | 110.9 / 92.4 / 70.8 | 4.47 → 101.8 | 5.1 | 24.0 (1.236) | 20.5% |
| verify 1 (8) | 60 | 63.49 / 63.11 (63.30) | 67.45 / 63.67 / 68.70 | 142.2 | 2.247 | 49.8 / 57.0 | 244 | 60.9 / 50.8 / 37.3 | 4.31 → 59.0 | 1.4 | 4.3 (1.073) | 6.8% |
| verify 16 (128) | 60 | 173.49 / 175.70 (174.59) | 174.98 / 178.49 / 178.51 | 464.5 | 2.661 | 171.9 / 183.4 | 649 | 162.2 / 135.2 / 101.3 | 4.52 → 147.4 | 19.4 | 27.2 (1.185) | 16.4% |
| verify 32 (256) | 60 | 209.59 / 208.76 (209.17) | 212.55 / 210.75 / 213.51 | 617.9 | 2.954 | 197.2 / 218.8 | 703 | 175.8 / 146.5 / 108.9 | 4.53 → 159.4 | 39.5 | 49.8 (1.312) | 26.3% |

Headline at each floor: stream+chain 2.835 (x1.188), 4.0 TB/s 2.748 (x1.151), 4.8 TB/s 3.30. Consumer-plus-structure floor, with TMA bytes cut to 33% (R33, H100 at 1830 MHz): d1 25.8, d16 85.9, d32 107.0, v1 66.6, v16 155.1, v32 184.7 µs.

Per-CTA stamps in R67 (µs, H200-like):

| row (build) | start chain (routed / x gathered) | A_done | A→B (A_done med → Bgo) | B_done | B-end spread + barrier | final | span |
|---|---|---|---|---|---|---|---|
| v32 (WGS) | 3.3 / 8.4 (5.1–10.0) | 119.8 (112.4–130.3) | hidden by ring | 177.9 (169.5–186.2) | 9.1 | 6.8 | 193.8 |
| d32 (WGS) | 2.0 / 4.4 | 83.7 (78.3–88.7) | hidden by ring | 121.3 (118.4–126.3) | 5.8 | 1.0 (max 4.8) | 128.1 |
| d16 (MID) | 2.8 | 53.9 (52.5–55.6) | 2.4 | 82.0 (81.0–84.1) | 3.1 | 0.4 | 85.5 |
| d1 (TINY) | 2.6 | 12.3 (11.1–12.7) | 1.9 | 19.3 (18.6–19.6) | 1.4 | 0.4 | 21.1 |

ncu per row. Each cell gives master, with R67 in brackets:

| row | time us | DRAM read MB (draw) | TB/s, % of H100 ceiling | issue active | ALU / FMA-heavy | tensor (gmma or hmma) | warps/SM, eligible per scheduler | top stalls per issue (master) |
|---|---|---|---|---|---|---|---|---|
| v32 | 230.9 [181.1] | 609.8 (20 experts = 566.2 + **43.5 re-read**) | 2.64, 85% [2.25, 73%] | 41.7% [55.0%] | 33.4/28.5 [43.6/37.2] | 8.1% avg, 44% max [10.6/57] | 18, 0.89 [1.21] | long-scoreboard 2.98, not-selected 1.15, gmma 1.13, barrier 0.92 |
| d32 | 173.9 [129.2] | 456.9 (16 = 453.0) | 2.63, 85% [2.37, 77%] | 40.6 [54.3] | 32.4/27.7 [44.5/38.1] | 8.0 [11.0] | 18, 0.88 [1.19] | long-scoreboard 4.40, not-selected 1.16, gmma 1.05 |
| d16 | 111.0 [83.0] | 283.3 (10 = 283.1; the split 16-token expert's second read came from L2) | 2.55, 84% [2.28, 74%] | 41.1 [53.1] | 38.0/31.1 [49.9/40.8] | 13.0 [17.1] | 9, 0.62 [0.81] | long-scoreboard 1.53, wait 0.62, barrier 0.54 |
| d1 | 30.3 [24.5] | 56.7 (2 = 56.6) | 1.87 [1.54] | 31.9 [40.7] | 28.4/24.7 [37.1/32.2] | 9.3 [12.1] | 9, 0.46 [0.60] | long-scoreboard 2.08, barrier 1.01 |

Ablations, R67: NW (no phase-B dependency waits): d1 x1.040, v1 x1.020, other rows ~x1.000. NF (no final pass): d32 x1.029 (3.6 µs), v16 x1.025 (4.0), v32 x1.035 (6.5), v1 x1.009, d16 x1.000. Byte scaling, R50 against R67: d1 x1.019, v1 x1.018, d16 x1.029, d32 x1.126, v16 x1.107, v32 x1.077.

H100 runs are on takemyjob branch hopper/gapmap-moe (47352b6, 559e3aa): 20261031-122316 / 123110 / 123701 / 124042 / 125106-gpunode06.
