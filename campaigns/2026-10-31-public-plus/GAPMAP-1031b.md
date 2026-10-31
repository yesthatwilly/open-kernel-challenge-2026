# GAPMAP-1031b: decode and bf16 gap map (judge)

This is the judge pass at effort max for gap-map wave 2: workflow wf_07670be7-65b, task wmzbualcz, run 2026-10-31 from 20:33Z to 20:52Z (stamps from the workflow journal; the judge wrote "to about 21:05Z"). Wave 1 (attention, prefill, moe) is GAPMAP-1031.md. The key tables from the two finder reports (decode: muse_decode_layer; bf16: muse_bf16_proj) are in the appendix; the full reports are in `research/gapmap-1031b/finder-{decode,bf16}.md`.

**Method**
- **Read-only.** Nothing in musespark was edited. No okc check or submit was sent. No attempt files or memory were written.
- **No H100 runs.** The one H100 dispute (§1.2 #6) is settled by both finders' existing ncu tables. The open question that remains (§4.1) is sized as tomorrow's M0.
- **"Verified"** means the judge recomputed the number from the judge JSONs (`benchmarks[].ratios`, `median_ns`, `baseline_ns`, `case.seed`), from the TASK weights, or from the named run output.
- **Leaderboard positions** are taken from the brief (10-31, ~18:40Z) and were not refetched.

## 0. Verdict

1. **The DC5L question is answered, and the LOG's reading of it does not hold.**
   - In decode's graph, attention at v16f and v32f runs serially between two nvjet GEMMs that are not PDL-launched, on one stream. There is no overlap and no PDL wait (H100 nsys, report.txt:237-248 and :289-300). A kernel saving therefore reaches the row.
   - On the judge's pooled paired ratios it did: v32f moved +0.93% against a predicted +0.86%.
   - DC5L's whole headline value is +0.10%. That is about a fifteenth of one read's ±1.5% interval.
   - Untouched rows read low in the DC5L jobs (d1s −3.2%, v1s −2.6%), and those set the KILL.
   - The same arithmetic applies to DPP. Its kernel saving was real and reached its rows, but it was worth about +0.26% of the headline. Most of DPP's +2.0% board step was row noise. The board (2.2220) sits about 0.4% above the 3-read level (2.2136).
2. **Value per µs matters for tomorrow, not reach.**
   - One µs at d1s is worth 0.177% of the headline; one µs at v32f is worth 0.012%. That is a factor of 15.
   - Full-attention rows cannot move decode's headline. The sliding rows at T≤32 can.
3. **Decode cannot reach 10th (2.375) by 11-04 at central estimates.**
   - 10th needs ×1.073 over the 3-read level, which is 25% of the log gap to the 2.934 compliant floor.
   - Every mapped item landing at central gives +5.9% (2.345). The p-weighted stack gives about +1.6%.
   - P(≥ 2.375) is about 0.05–0.1, not the finder's 0.32.
   - The decode top-50 cut (2.261, ×1.021) is within reach if DS1 plus one more item land.
4. **The largest judge-evidenced decode gap is one neither finder ranked: the T≤8 dense stream rate.**
   - The bf16 board runs decode's own T=1 and T=8 projection shapes, from the same 12-bit bytes, at about 4.6 TB/s on the same H200.
   - Decode's chain runs them at 1.3–3.3 TB/s.
   - In judge eager kernel time that is about 11.5 µs per T=1 row and 7.5 µs per T=8 row. About 7 µs and 4 µs of it is on the graph critical path.
   - This is phase-A build 1 (**DS1**).
5. **bf16 deserves no phase-A build.**
   - Its best candidate, lossless 12-bit at m=128, is +2.4% at mid efficiency with p ≈ 0.2. The H100 proxy loses 1.13–1.16×, and BF2's decode-issue non-transfer is unresolved.
   - Its role tomorrow is donor (the cgemv family, for DS1) and, if BF-DX2 passes, testbed for the shared m=128 12-bit GEMM.
6. **No decode candidate clears the /hypothesize bar** (≥ 2.5% if it works, and gain × p ≥ 1.5%). DS1 goes first because it is the cheapest judge-evidenced item on the critical path, and S1's stream rate depends on it.
7. **Tonight: 3 diagnostic checks plus 2 reserves (§3).** The bf16 finder's proposal to spend bf16's remaining checks timing other problems' kernels is rejected as a gray area.

## 1. Cross-problem table

### 1.1 The table

| | muse_decode_layer | muse_bf16_proj |
|---|---|---|
| Board (rank) / 3-read level | 2.2220 (110) / 2.2136 | 1.5657 (87) / 1.5487 |
| Heaviest rows: judge sol vs our 3-read mean | d1s 40.0 vs 68.2 µs (×1.70; 4.4% of the log gap)<br>v32s 143.8 vs 311.8 (×2.17; 3.9%)<br>v16s 126.6 vs 244.6 (×1.93; 3.6%) | v16 pooler 14.64 vs 20.49 µs (14.7% of the gap to F_12)<br>v16 unpooler 14.64 vs 19.12 (11.4%)<br>v16 qkv 8.42 vs 13.82 (10.5%) |
| Compliant-floor headline | **2.934** (12-bit dense, 4.0 TB/s, 3/4 µs chain); 2.710 with BF16 dense | **1.739** at F_12; 1.879 at F_ent; 1.627 at F_raw |
| Gap from board to floor | ×1.320 (×1.325 from the level) | ×1.111 (×1.123) |
| Gap to 10th | ×1.069 to 2.375 (×1.073 from the level), 25% of the log gap; top-50 cut 2.261 needs ×1.021 | ×1.023 to 1.602 (×1.034); top 1.652 needs ×1.067 |
| Value per µs | d1s 0.177, v1s 0.114, d16s 0.083, d32s 0.067, d1f 0.062 … v16f 0.016, v32f 0.012 %/µs | per load: d1 0.485, v1 0.458, d16 0.350, d32 0.324, v16 0.251, v32 0.204 %/µs |
| Best structural candidate | **DS1**: T≤8 dense at the H200 stream rate (d1s, d1f, v1s, v1f). Conditional: S1-M1 (§2). | **A**: lossless 12-bit m=128 GEMM (v16 pooler, unpooler, qkv) |
| Gain if it works × p = EV | DS1 +1.64% (range +0.81% for qkv only to +2.42%; +4.3% only if the whole eager gap were on-path) × 0.45 ≈ **0.7%**<br>S1 +3.36% × 0.2 ≈ 0.7% | A +2.44% (+1.46% to +3.39%) × 0.2 ≈ **0.5%**<br>A + B +3.15% |
| Transfer class | Per-SM stream rate (bytes in flight). The target rate is already measured on the judge (bf16 board). The H100 shows a 24%-per-byte version in ncu. Not the byte-cut class of S6 and DL2H. | Byte cut (bandwidth class) plus 12-bit decode issue (BF2 class). Neither has transferred. |
| Build size | M1 150–250 lines; full DS1 400–600 lines; +20–40 KB of 158,639 B free; packed copies +28 MB (qkv) and +50 MB (pooler) of GPU memory | 600–900 lines of CUDA plus ~150 of Python; +35–45 KB; prepare +35–40 s |

### 1.2 What the rankings rest on (verified)

**Decode**

1. **Row times.** The three DPP reads (eb4e07ad, f078d673, 9d726d84) were recomputed from `benchmarks[]`.
   - Headlines: 2.1826 / 2.2367 / 2.2220. The 3-read level is 2.2136.
   - Row means match the finder's table to 0.1 µs.
   - Load = 46 × sliding + 16 × full; the headline is the geomean over 6 loads (TASK.md:39-52).
2. **Floor.**
   - The finder's per-row floors reproduce 2.9339. 10th needs 25.0% of the log gap.
   - Spot check at d1s: 28.3 (qkv) + 12.6 (o) + 50.3 (pooler) + 8.4 (router) + 6.3 (unpooler slice) + 70.8 (2.5 experts × 28.31) = 176.7 MB. At 4.0 TB/s plus the 3 µs chain that is 47.3 µs.
   - ncu bytes agree: 28.4 / 12.8 / 58.9 / 6.3 MB.
3. **Value per µs.** Reproduced on all 12 rows.
4. **DC5L reach.** Pooled paired ratios from the per-rep `ratios`. DPP n = 180 at d1 and 60 elsewhere; the other arms n = 120 / 40.

   | row | D0 (2 reads) | DL2H (2; same code as D0 on these rows) | DPP (3) | DC5L (2) |
   |---|---|---|---|---|
   | d1s | 2.4908 | 2.5509 | 2.5396 | 2.4585 |
   | v1s | 2.4165 | 2.4633 | 2.4960 | 2.4311 |
   | v16f | 1.9910 | 2.0427 | 2.0686 | 2.0759 |
   | v32f | 2.1308 | 2.1410 | 2.1972 | 2.2177 |

   - **DPP over the D0 family:** v16f +2.7% and v32f +3.2%. Its kernel saving was −8.0 / −10.6 µs, about 2.3% / 2.1% of those rows. The saving reached its rows.
   - **DC5L over DPP:** v32f +0.93% against +0.86% predicted. v16f +0.35% against +1.0% predicted, inside noise.
   - **Identical code moves rows 2–3% between job pairs.** DL2H vs D0 at v16f is +2.6%; DC5L vs DPP at d1s is −3.2%.
   - Each job draws its own seed per row (`case.seed` differs in all 9 JSONs), so routing differs from job to job.
   - "yours" is SGLang's median divided by the paired ratio, so it inherits SGLang drift (v32f SGLang 1052–1095 µs). That is why the µs columns looked flat.
5. **H100 graph mode** (run 20261031-130215-gpunode06, gapdec-prof/report.txt).
   - v16f (:237-248) and v32f (:289-300): one stream, 9 kernels, 100% path presence, cross-stream overlap 0.00. PDL wait is 0.29 and 0.48 µs per row.
   - Attention sits between the nvjet qkv and the nvjet o. The gaps are 0.08 / 0.39 µs at v16f and 0.02 / 0.31 µs at v32f.
   - d1s (:197-207): span 96.45 µs, 8 kernels, all on the path. The qkv is cp 15.47 µs with zero wait.
   - Layer A/B (gapdec-ab/log.txt):

     | row | DC5L vs master | A/A (master vs master) |
     |---|---|---|
     | v16f | 0.9990 [0.9917, 1.0060] | 1.0008 |
     | v32f | 0.9974 [0.9861, 1.0067] | 1.0017 |

6. **New: dense-rate cross-check.** Same shapes, same 12-bit bytes, same judge hardware.

   | edge | decode judge eager µs: finder 5-profile mean; DPP-score | bf16 judge eager µs, 3 reads | H100 graph cp / duration at d1s |
   |---|---|---|---|
   | T=1 qkv | 10.3; 10.7 | cgemv 7.1 / 7.0 / 7.0 | 15.47 / 15.28 (first kernel, no wait) |
   | T=1 o + b1 glue | 6.3; 6.3 | cgemv 4.1 / 4.0 / 4.0 | 4.95 / 8.13 |
   | T=1 pooler + router | 18.9; 19.7 | cgemv 12.8 / 13.2 / 12.9 + rgemv 2.5 / 2.4 / 2.5 | 25.88 / 26.70 |
   | T=1 unpooler slice | 4.7; 4.7 | cgemv 2.2 / 2.3 / 2.2 | 3.38 / 5.20 |
   | T=8 qkv | 9.4; 9.3 | sk2<1,8,3> 7.4 / 7.3 / 7.5 | — |
   | T=8 pooler + router | 19.5; 19.1 | smallm2 13.4 / 13.4 / 13.6 + rr_small 3.5 / 3.6 / 3.6 | — |

   - **Kernel sums.** T=1: 40.2 vs 28.7 µs. T=8: 40.0 vs 32.5 µs.
   - **On-path deficit.** Scaling decode's eager times by the H100 cp/duration ratio (o 0.61, pooler+router 0.97, unpooler 0.65, qkv 1.0) gives about 7 µs at T=1 (qkv 3.3, pooler+router 2.9, unpooler 0.8) and about 4 µs at T=8.
   - **The H100 sees a smaller version.** In ncu (both finders: eager, cache flushed, gpunode06), decode's pooler+router takes 29.25 µs for 58.9 MB (2.09 TB/s). bf16's cgemv pooler takes 20.32 µs for 50.5 MB (2.61 TB/s). That is 24% more time per byte. The same sk2<2,32,3,1,4,2,4,1> qkv instantiation reads 15.87 µs in decode and 14.05 µs in bf16.
   - This contradicts levers-1031b.json:25 ("On the H100, sk2 and cgemv tie").
   - **Why cgemv streams faster.** Every thread issues all of its weight loads before any math (bf16 submission.py:4355, `lw[ROWS][PER]` and `cw[ROWS][PER]`), on 1152–2048 CTAs of 544 threads. That puts far more bytes in flight than sk2's 96–256-CTA ring.

**bf16**

1. **Three reads recomputed** (109f577b, a9547928, 280a3622): 1.5657 / 1.5365 / 1.5442. The level is 1.5487. v16 rows: 20.49 / 19.12 / 13.82 µs. Every weight is 62 (TASK.md:29-60).
2. **F_12 headline:** 1.7394 (finder 1.7392), using min(F_12, incumbent) per row.
3. **Byte check, v16 pooler.**
   - F_raw: 2.33 + (67.1 + 2.1 + 1.05 MB) / 4.57 TB/s = 17.7 µs.
   - F_12 (12-bit weights, 50.3 MB): 14.0 µs.
4. **H100 proxy** (run 20261031-132458-gpunode06, gmbf-proxy/full.txt), pooler:

   | arm | µs | vs incumbent |
   |---|---|---|
   | incumbent | 26.06 | 1.000 |
   | 12-bit stream | 21.06 | 0.808 |
   | raw stream | 25.63 | 0.983 |
   | 12-bit + decode | 24.50 | 0.940 |
   | MMA only | 14.94 | 0.573 |
   | 12-bit + decode + MMA | 29.55 | 1.134 |
   | raw + MMA | 29.62 | 1.136 |

   - ptxas reports serialized wgmma (C7513, C7514, C7520).
   - Raw + MMA equals 12-bit + decode + MMA, so the proxy's MMA path is its limiter. Its "everything" arm says nothing about A.

### 1.3 Where the judge changes the finders' rankings

- **Decode S1: p 0.35 → 0.2.**
  - The public team's in-file megakernel "never won" (decode submission.py:10846-10847).
  - The arch-mirror skeleton cost 17–19 µs empty (KERNEL-COUNTS.md:32).
  - Real glue-launch removal saves 1.2–2.0 µs per launch (public NEGATIVE_RESULTS.md:31), not the stand-in's 3.2 µs. Stand-ins over-predicted by 6.6× there (:32).
  - S1's stream part is DS1.
- **Decode S2: folded into moe.**
  - Its MoE part is wave 1's C2 / unit-end work.
  - Its glue part is mostly off-path at d16/d32: the public premise check found split-K and combine exposed for 0.00 µs on 6 of 6 rows (NEGATIVE_RESULTS.md:50-53).
- **Decode S3 (p 0.3 → 0.15) and bf16 A are one mechanism on two boards.** They are merged as **D12**.
- **The decode finder's "+4.69% shared moe unit-end fix"** is not reproduced by any script in the finder's folder. Wave 1's verified decode port for moe C1+C2 is about +1.4% at planning (GAPMAP-1031 §1.1); use that.
- **bf16 A: p 0.4 → 0.2.** BF2's H100 0.82–0.88 became 1.12–1.29 on the judge (bf16 CLAIMS:78). The cause was decode issue, which is SM-clock-bound, and the proxy reproduces it.

## 2. Tomorrow's phase A (decode first; moe stays the testbed)

**Ground truth for every build (graph mode)**
- Every decode row runs on one stream, and every kernel is on the critical path in all 24 arm-rows.
- At T≤32 the chain is PDL with no launch gaps. At T=128/256 the nvjet GEMMs are not PDL-launched, which leaves 0.8–1.6 µs of launch gaps per row.
- A µs removed from any kernel comes off the row. The only question is which µs are worth the most, and the answer is the T≤32 sliding rows.
- At d1s on the H100, the path splits into:
  - dense: 49.7 µs (52%);
  - moe: 37.8 µs (39%);
  - attention, route and finish: 9.0 µs (9%).

**Common base**
- All decode builds sit on D0FIX once it lands (correctness first; LOG:286).
- DC5L rides on the next carrier as a bit-exact rider (its audit passed on all 4 arms). It gets no gate of its own, because no number of reads can resolve +0.1%.

### Build 1: DS1, the T≤8 dense chain at the H200 stream rate (rows d1s, d1f, v1s, v1f)

**Why it goes first**
- The deficit is measured on the judge (§1.2 #6).
- It sits on the critical path of the highest-value-per-µs rows.
- A judge-proven donor kernel already exists in our own bf16 file.
- It reopens the parked LEVERS-1031 #7 ("decode cgemv": qkv only, +0.45% × 0.2), with stated differences:
  - it covers all four T≤8 edges;
  - it is valued on on-path numbers;
  - the H100 is not blind to it (24% per-byte deficit in ncu).

**Design panel (effort max): scope**
1. **Kernel choice per edge.** For qkv, o + b1, pooler + router and the unpooler slice, at T=1 and T=8, the panel chooses between:
   - bf16's judge-proven kernels: cgemv<2,512,8192,2> and its siblings, cg_compress (bf16 :3209), v18s_gemv, smallm2_gemv;
   - an sk2 with more bytes in flight.

   PUBF:81 ("the qkv sk2 GEMV saturates near 96 CTAs") covers only T=16/32 qkv.
2. **The qkv-to-attention interface at T=1.** Decode's qkv writes 0.30 MB at T=1 (ncu), against a ~5 KB BF16 output, so it hands attention FP32 partials. The panel decides whether the new qkv keeps that protocol or attention's input stage changes. ATT7's FA3-ruling code stays byte-identical.
3. **The router at T≤8 stays exact and stays inside the pooler stream.**
   - Exact means: FP32 normed x; FP32 weights, or hi+lo with FP32 accumulation; ≤ 7.6e-6 against F.linear; top-8 identical (PUBF:87).
   - A separate FP32 router launch cost 5.5 µs at d1 (PUBF:93).
4. **PDL.** cgemv loads all of its weights up front. The panel decides whether to keep the pre-wait stream (2 stages today; PUBN:31) by issuing the weight loads before `griddepcontrol.wait`.
5. **Memory.** A second packed layout costs +28 MB (qkv) and +50 MB (pooler). Extra allocation alone moves unchanged d1/v1 code by 0.5–1.3 µs (PUBF:65-66). Prefer one layout that both T=1 and T=8 can read.
6. **Ring hand-backs.** Every ring slot hand-back keeps D0FIX's `fence.proxy.async` (CLAIMS:116).

**Stages**
- **M0 (tonight):** DEC-DX1 (§3).
- **M1:** the T=1 qkv only, moved to a cgemv-class kernel with `launch_dependents`.
  - H100 gate (from LEVERS-1031 #7): in-layer d1s/d1f no more than 0.3 µs slower; the other rows within ±0.3 µs.
  - Judge: 2 checks.
- **M2:** the T=1 and T=8 pooler+router, fused, at ≥ 4.0 TB/s on the H200.
- **M3:** the o and the unpooler slice. T=16/32 only if DEC-DX1 shows the same in-chain gap there.

**Kill rules** (to be pre-registered in the attempt files)
- **M0 KILL:** on the H200 at T=1, the cgemv qkv run standalone is not at least 1.5 µs faster than the incumbent qkv run standalone, **and** the incumbent has no in-chain penalty of 1.5 µs or more. DS1 is then closed.
- **M1 GO**, all of:
  - the judge eager T=1 qkv kernel is ≤ 8.5 µs in both reads (the incumbent reads 10.3–10.7);
  - no other d1 or v1 kernel is 0.5 µs or more slower;
  - the pooled d1s/d1f paired ratio is at least +2% over DPP's.
- **M1 KILL:** the qkv kernel reads ≥ 9.5 µs in either read.
- **M2 and M3:**
  - Every route is raced in prepare against the incumbent on the H200's graph-timed `_after_attention` chain (SUB:10715-10821). A route is taken only at ≤ 0.97.
  - The headline is a no-harm guard only (≥ board − 1.5%), never the GO.

**Value, transfer and size**
- **Value:** central +1.64%, at −5 µs per T=1 row and −2.5 µs per T=8 row. M1 alone is worth +0.81% at full capture. p 0.45, so EV ≈ 0.7%.
- **Transfer:** the judge already shows the target rate on these exact shapes, so the gate is an H200 race, not an H100 ratio.
- **Size:** M1 150–250 lines; DS1 in full 400–600 lines; +20–40 KB.

### Build 2: moe C1 with a decode port stage (wave 1's testbed; no new decode design)

- MoE is the second on-path block at d1s and v1s: judge 25.7 / 57.9 µs, H100 cp 37.8 / 99.2 µs, at 2.2–2.6 TB/s.
- Scope, stages and gates are as GAPMAP-1031 §2: the C1 M0 microbench, then a build on GO.
- **Decode stage:** port only after moe's own judge GO.
  - Gate: pooled d1s, v1s and d16s paired ratios each at least +1% over 2 reads.
  - The headline is a no-harm guard only.
- **Decode value:** about +0.9% at planning, up to ~3% (GAPMAP-1031 §1.1), × p 0.3.

### Build 3: conditional, decided by tonight's diagnostics

**If DEC-DX1(c) passes the S1-M0 rule, build S1-M1.**
- **What it is:** the T≤8 FFN tail (route, moe, unpooler, finish) as one kernel with epoch counters. No grid barriers, no cooperative launch, the FP32 router input as master has it, and a fence before every hand-back.
- **Panel scope:**
  - the counter-reset scheme: use epoch values, because a memset reset costs 3.2 µs and a self-reset 5.0 µs (PUBF:51);
  - keep attention serial in M1, since attention slows 50–80% next to a GEMM (PUBF:77).
- **Value:** +0.8% at −2 µs per T≤8 row; +3.36% at the finder's central −8 µs.
- **Kill rules:**
  - H100 in-layer d1s/v1s is not at least 2.0 µs faster;
  - or the judge's pooled d1s paired ratio gains less than +2%.

**Otherwise, if BF-DX2 passes, build D12: the shared 12-bit m=128 GEMM.**
- Build it in bf16 first (one kernel per row gives the cleanest judge read), then port it to decode's v16 rows (the GEMM part of finder S3).
- **Stage 1, on the H100 only:**
  - fix the wgmma serialization (MMA alone at ≥ 75% of peak);
  - hide the decode (the decode run at ≤ 1.05× the stream run).
- **Kill rule:** the H200 in-prepare race reads above 0.95 on the v16 pooler.

**If neither passes**, the slot goes to DS1 M2/M3.

**bf16, plainly: no phase-A build.**
- A's EV is about 0.5%.
- B (+0.70%) and C (+1.04% at p 0.25) are below the bar.
- bf16 is DS1's donor, and D12's testbed if BF-DX2 passes.

## 3. Tonight's diagnostic checks (6 at most)

**Rules common to all three**
- `kernel_fn` stays byte-identical to master.
- The diagnostic is appended after master's prepare. For decode, that means wrapping `prepare` once more at the end of the file (it ends at SUB:13886).
- Timing uses the file's own graph timer and read flush (decode `_graph_time`, SUB:10649).
- Output is tagged prints. Every buffer is freed and `torch.cuda.empty_cache()` is called. `st` is never touched.
- The diagnostic skips itself, with a printed reason, if more than 160 s have passed by the time it starts.
- The checks are never scored. Results are read from the saved job log. The check's own row timings are not evidence for anything.
- Each check gets a builder and an independent audit, and the usual attempt file and rules checklist.

**Dispatch:** in the order DEC-DX1, BF-DX2, BF-DX1, by the 23:40Z cutoff. Wave 1's diagnostic builders already hold GPUs and agent slots.

**DEC-DX1 (decode).** 1 read, plus 1 reserve if its internal A/A is above 1%.
- **What it measures:**
  - (a) Each T=1 and T=8 dense kernel standalone vs inside the `_after_attention` chain: 32 ABBA pairs, cold.
  - (b) cgemv<2,512,8192,2> on decode's own qkv weights (a copy of cg_compress), standalone, against the incumbent qkv at T=1.
  - (c) A port of gapskel: an empty 8-kernel PDL chain on T=1 grids, vs a 132-CTA persistent kernel with P = 1/3/8 epoch hand-offs, vs one grid barrier, vs the null graph.
  - (d) Print `st["moe_pick"]` and `st["gluev2"]`.
- **Pre-registered rules:**

  | outcome | action |
  |---|---|
  | (b) ≤ incumbent − 1.5 µs | DS1-GO: M1 as specified |
  | incumbent in-chain − standalone ≥ 1.5 µs at T=1 | DS1-PIVOT: the panel takes the in-chain cause first (PDL placement, co-resident early CTAs) |
  | neither | DS1-KILL |
  | persist(8) − null ≤ 3.0 µs and ≤ 0.5 × (chain(8) − null) | S1-M0 passes: S1-M1 eligible as build 3 |
  | otherwise, persist(3) − null ≤ 2.0 µs | S1-M1 only |
  | neither | S1 closed through 11-04 |

- **Decides:** the shape of DS1's M1 and the choice of build 3.

**BF-DX2 (bf16).** 1 read, plus 1 reserve.
- **What it measures:** the p12 proxy (commit b666749), run in prepare against `kernel_fn`'s m=128 rows. S = 12-bit stream ÷ incumbent; D = (12-bit + decode8) ÷ 12-bit stream. For the pooler, unpooler and qkv.
- **Pre-registered rules:**

  | outcome | action |
  |---|---|
  | S ≤ 0.80 (pooler and unpooler) and D ≤ 1.10 | D12 eligible as build 3 |
  | S ≤ 0.80 and D > 1.10 | D12 goes to H100 decode-hiding work (phase B) |
  | S ≥ 0.90 or D ≥ 1.30 | kill bf16 A and decode S3's 12-bit part; CLAIMS refuted line (BF2-class non-transfer) |

- **Decides:** build 3; opens or closes the m=128 12-bit line on both boards.

**BF-DX1 (bf16).** 1 read; its arms are paired inside one prepare.
- **What it measures:**
  - a pure-read ladder from 6.3 to 67 MB, with 3 launch configs each;
  - the empty-kernel floor;
  - `kernel_fn` anchors at pooler m=1/32/128, qkv m=32 and unpooler m=16;
  - torch.mm sustained TF/s at 8192³, the v32 shapes, and 2-, 4- and 8-way K-split variants.
- **Pre-registered rules:**
  - anchor(m=32) − t(same bytes) < 1.0 µs for both the pooler and qkv → kill C; ≥ 1.5 µs → C gets an H100 prototype;
  - v32 qkv anchor ≥ its floor + 2.5 µs → B goes to phase B; otherwise drop B;
  - t(28.3 MB) and t(58.7 MB) set DS1's M1 and M2 targets;
  - the fitted (t0, BW_H200) becomes the transfer model.
- **Decides:** C, B, and DS1's targets.

**Total:** 3 checks plus 2 reserves = 5.
- Quota at 20:31Z: decode 11 (D0FIX ×2 and ET3FIX ×2 already planned); bf16 16 (BF16FIX ×2 planned).
- Unspent checks expire at 00:00Z.

**Rejected, with reasons**
- **Decode finder's D1** (DC5L on the H200; ~40 min build, 1 check): it decides a +0.1% item. The graph structure already answers reach, and v32f moved as predicted.
- **The moe distinct-expert fit in decode finder's D2:** an engine question for the moe testbed. Wave 1's moe DX checks cover it on moe's quota.
- **The TMA read ladder in decode finder's D2:** the same hardware measurement as BF-DX1's ladder. Run it once.
- **Second reads of BF-DX1:** its arms are already paired inside one prepare.
- **Spending bf16's remaining ~12 checks timing other problems' kernels:**
  - using one problem's check cap to measure another problem's kernels is at least a gray area;
  - it needs the user's explicit ruling;
  - it is not needed, because wave 1's diagnostics run on their own quotas.

## 4. What neither finder answered, and the cheapest way to answer it

1. **Why decode's T≤8 dense kernels take 24–45% more time per byte than the bf16 board's kernels on the same shapes and bytes.**
   - Candidate causes:
     - kernel family (a ring vs all loads up front);
     - split-K partial writes (0.30 MB at the T=1 qkv);
     - the fused FP32 router;
     - early-launched PDL successors sharing SMs;
     - epilogue work.
   - **Cheapest answer:**
     - One H100 run of about 15 minutes, with no judge check: time both files' kernels standalone at T=1/8/32 in one process on one GPU, with ncu on each pair.
     - Then DEC-DX1 on the H200.
     - A free first clue: the identical sk2<2,32,3,...> instantiation reads 15.87 vs 14.05 µs; a SASS diff costs nothing.
2. **Whether the ±2–3% row swings between job pairs on identical code are routing draws or prepare-race picks.**
   - Each job draws its own seed per row.
   - Master's `_tune` (SUB:10715) picks the moe build and glue per T in prepare and prints nothing (its log is off).
   - **Cheapest answer:**
     - DEC-DX1(d) prints the picks.
     - From then on, every decode check prints them (print-only, in prepare), and the row ratios are regressed on the picks.
     - If the picks vary between jobs, pinning the race to its most common pick is a free cut in variance.
3. **Whether 12-bit decode can share an SM with a tensor pipe that is ≥ 75% busy, on the H200** (bf16 A, decode S3).
   - Today's proxy cannot say, because its wgmma is serialized.
   - **Cheapest answer:** fix the serialization in p12 on the H100 first. BF-DX2 measures only S and D.
4. **Whether the decode finder's +4.69% moe figure holds.**
   - No script reproduces it.
   - **Cheapest answer:** wave 1's moe DX checks tonight, read against GAPMAP-1031 §1.3 (moe items 2–5).
5. **S1's real boundary saving on the H200.**
   - The H100 stand-in over-predicts (6.6× in PUBN:32).
   - **Cheapest answer:** DEC-DX1(c) measures the skeleton cost on the H200.

## 5. Proposed record corrections (the judge edits nothing)

- **CLAIMS:117 (DC5L) and LOG:270-279.**
  - Replace "the flat q8 kernel's saving does not reach the decode row time" with: "Reach is structural (one serial stream; report.txt:237-300). v32f's pooled paired ratio moved +0.93% against +0.86% predicted. DC5L's headline value (+0.10%) is below what two reads can resolve."
  - Reopen: as a bit-exact rider.
- **LOG:111-113 and LESSONS (DPP).**
  - DPP's kernel saving was real and reached v16f/v32f (+2.7% / +3.2% pooled), but it was worth only about +0.26% of the headline. The "3–4× amplification" applies to kernel time.
  - Decode gates should be built from per-row pooled paired ratios of the rows a change touches. Use the headline only as a no-harm guard, unless the expected headline effect is above ~1.5%.
- **GAPMAP-1031 §1.1:** "the q8 part is 0, per DC5L" should read: the q8 part reaches the rows, and is worth 0.012–0.016% per µs.
- **levers-1031b.json:25:** "on the H100, sk2 and cgemv tie" is contradicted by today's ncu (§1.2 #6).

## Appendix A: decode finder key tables (muse_decode_layer)

**A.1 Per-row gap.**
- Judge µs is the 3-read mean of eb4e07ad / f078d673 / 9d726d84.
- The kernel split is the mean of 5 eager profiles.
- The floor uses 12-bit dense weights at 4.0 TB/s, plus a 3/4 µs chain.
- The share column is the row's share of the headline log gap.

| row | w | judge µs (3 reads) | judge sol | judge kernel split, µs | floor (MB → µs) | gap µs (x) | %/µs | share % |
|---|---|---|---|---|---|---|---|---|
| d1s | 46 | 68.2 (69.1/67.7/67.8) | 40.0 | moe 25.7, pool+rt 18.9, qkv 10.3, o 6.3, attn 5.1, unpool 4.7, route 2.7, glue 1.9 | 177 → 47.3 | 20.9 (1.44) | 0.177 | **4.41** |
| d1f | 16 | 74.5 (75.2/74.7/73.7) | 44.2 | moe 26.1, pool+rt 18.5, attn 11.0, qkv 10.4, o 6.2, unpool 4.7, route 2.6, glue 1.8 | 198 → 52.4 | 22.1 (1.42) | 0.062 | 1.63 |
| d16s | 46 | 129.9 (130.7/127.5/131.4) | 76.7 | moe 67.6, pool+rt 18.5, unpool 15.4, qkv 10.3, attn 7.7, glue 6.7, o 6.2, route 1.9 | 373 → 96.3 | 33.6 (1.35) | 0.083 | 3.14 |
| d16f | 16 | 204.1 (202.2/202.7/207.2) | 137.5 | attn 89.3, moe 65.5, pool+rt 17.8, unpool 14.7, qkv 10.1, o 8.1, glue 7.2, route 2.1 | 698 → 177.4 | 26.7 (1.15) | 0.029 | 0.87 |
| d32s | 46 | 147.8 (149.3/147.0/147.0) | 84.5 | moe 85.6, pool 18.3, unpool 16.6, qkv 11.5, attn 9.5, glue 7.5, o 6.9, router 5.4, route 2.0 | 425 → 109.2 | 38.5 (1.35) | 0.067 | 2.84 |
| d32f | 16 | 294.5 (304.2/281.7/297.5) | 208.1 | attn 167.4, moe 80.0, pool 17.7, unpool 15.9, qkv 11.3, o 8.3, glue 7.6, router 5.3, route 2.1 | 1073 → 271.4 | 23.1 (1.09) | 0.023 | 0.59 |
| v1s | 46 | 105.7 (109.9/101.7/105.5) | 62.3 | moe 57.9, pool+rt 19.5, qkv 9.4, attn 7.3, o 6.0, unpool 5.1, glue 5.0, route 1.9 | 320 → 82.9 | 22.8 (1.28) | 0.114 | 2.95 |
| v1f | 16 | 115.7 (115.5/117.5/114.0) | 66.4 | moe 56.1, pool+rt 20.3, attn 15.4, qkv 9.6, o 6.2, glue 5.4, unpool 5.3, route 1.9 | 340 → 88.0 | 27.7 (1.32) | 0.040 | 1.25 |
| v16s | 46 | 244.6 (251.9/240.2/241.7) | 126.6 | moe 154.5, o+unpool 29.0, pool 25.0, qkv 16.6, attn 12.4, glue 9.1, route 2.7 | 693 → 177.2 | 67.4 (1.38) | 0.046 | 3.60 |
| v16f | 16 | 339.2 (338.5/337.2/341.8) | 188.7 | moe 155.0, attn 109.3, o+unpool 29.0, pool 24.8, qkv 16.8, glue 9.2, route 2.8 | 1017 → 258.3 | 80.9 (1.31) | 0.016 | 1.50 |
| v32s | 46 | 311.8 (314.4/316.5/304.4) | 143.8 | moe 193.2, pool 31.6, unpool 26.0, qkv 20.6, attn 15.9, glue 14.0, o 8.8, route 3.5 | 796 → 216.6 | 95.2 (1.44) | 0.035 | **3.86** |
| v32f | 16 | 489.4 (496.4/490.1/481.6) | 267.2 | attn 200.1, moe 190.6, pool 31.7, unpool 26.0, qkv 20.4, glue 13.6, o 9.2, route 3.5 | 1445 → 378.7 | 110.7 (1.29) | 0.012 | 1.56 |

**A.2 Gap by component.** Eager kernel time minus floor, times %/µs. The total exceeds 28.2% because the eager durations include PDL waits.

| row class | dense | moe | attention | glue + route | total |
|---|---|---|---|---|---|
| T≤8 | 5.4% | 2.6% | 2.5% | 2.1% | 12.6% |
| T=16/32 | 3.4% | 3.4% | 1.0% | 1.7% | 9.5% |
| T=128/256 | 4.0% | 3.4% | 1.7% | 0.8% | 9.9% |
| all rows | 12.7% | 9.4% | 5.2% | 4.7% | 32% |

**A.3 d1s, H100 graph mode.** Span 96.45 µs (report.txt:197). Columns are cp / wait µs, then ncu MB, TB/s, DRAM %, tensor %.

| # | kernel | cp | wait | ncu |
|---|---|---|---|---|
| 1 | qkv sk2 (96 CTAs) | 15.47 | 0 | 28.4 MB, 1.98 TB/s, 59%, 6.5% |
| 2 | attn<1,0,13> (32 CTAs) | 5.76 | 0.67 | 0.58 MB, latency-bound |
| 3 | o sk2_gle | 4.95 | 3.15 | 12.8 MB, 1.37 TB/s, 41% |
| 4 | pool+router sk2_gle | 25.88 | 0.80 | 58.9 MB, 2.09 TB/s, 62% |
| 5 | _gle_route | 1.64 | 2.37 | — |
| 6 | moe_fused<0> (TINY) | 37.76 | 1.33 | 85.0 MB, 2.17 TB/s, 65% |
| 7 | unpool sk2 | 3.38 | 1.95 | 6.3 MB, 0.90 TB/s, 27% |
| 8 | _sl_cluster_finish | 1.62 | 0.77 | — |

**A.4 d1s synthetic stand-in** (run 20261031-132306-gpunode06, gapdec-skel/log.txt:4-21). Treat these as bounds: stand-ins over-predicted real gains 6.6× in PUBN:32.

| variant | µs |
|---|---|
| one uninterrupted 176.7 MB stream | 67.65 |
| 5 stream phases, persistent kernel with counter hand-offs | 72.59 |
| same 5 phases as a PDL kernel chain | 85.25 |
| full 8 phases, PDL chain | 83.92 |
| full 8 phases, persistent kernel | 79.97 |
| full 8 phases, no PDL | 87.44 |
| null graph replay | 5.44 |

**A.5 Finder candidates and the judge's adjustment**

| candidate | finder gain | finder p | judge |
|---|---|---|---|
| S1: persistent T≤8 kernel | +2.05% (counters) / +3.36% (central) / +6.14% (hidden latency) | 0.35 | p 0.2; its stream part is DS1; build only S1-M1, and only on DEC-DX1 |
| S2: persistent FFN tail, T=16/32 | +2.08% | 0.4 | folded into moe C1/C2 (wave 1) |
| S3: dense GEMMs and glue, T≥128 | +2.53% | 0.3 | merged into D12, p 0.15; build only on BF-DX2 |
| shared moe unit-end fix | +4.69% | 0.4 | not reproduced; use wave 1's ≈ +1.4% |
| finder stack | 2.404 central; 2.526 with moe; P(≥ 2.375) ≈ 0.32 | — | all items central +5.9% → 2.345; P ≈ 0.05–0.1 |

## Appendix B: bf16 finder key tables (muse_bf16_proj)

**B.1 Per-row table.**
- Net µs is the 3-read mean of 109f / a954 / 280a.
- Floors are t0 2.33 µs + bytes / 4.57 TB/s (fitted on the judge's m=1 rows).
- `*` means the incumbent already streams the 12-bit code.

| row | judge net µs | sol | H200 kernel (eager ×3) | F_raw | F_12 | gap to F_12 | share |
|---|---|---|---|---|---|---|---|
| d1 qkv_gate | 8.74 | 7.87 | cgemv<2,512,8192,2> 7.1/7.0/7.0 | 10.61* | 8.54 | 0.20 | 0.8% |
| d1 o | 5.17 | 3.50 | cgemv<2,64,1024> 4.1/4.0/4.0 | 6.01* | 5.09 | 0.08 | 0.3% |
| d1 pooler | 13.25 | 13.99 | cgemv<2,512,8192> 12.8/13.2/12.9 | 17.04* | 13.36 | 0 | 0 |
| d1 unpooler | 3.55 | 1.75 | cgemv<2,128,4096,4> 2.2/2.3/2.2 | 4.17* | 3.71 | 0 | 0 |
| d1 router | 3.65 | 1.75 | rgemv<1,1> 2.5/2.4/2.5 | 4.18* | 4.18 | 0 | 0 |
| d16 qkv_gate | 9.73 | 7.93 | sk2<1,16,3> 8.4/8.7/8.5 | 10.68* | 8.61 | 1.12 | 3.4% |
| d16 o | 5.67 | 3.56 | sk2<1,16,2> 4.0/4.4/4.4 | 6.07* | 5.15 | 0.52 | 1.6% |
| d16 pooler | 13.88 | 14.06 | dd_cap2 13.6/13.1/13.1 | 17.12* | 13.45 | 0.43 | 1.3% |
| d16 unpooler | 13.78 | 14.06 | sk2<1,16,2,2,2> 14.2/12.8/13.8 | 17.12* | 13.45 | 0.33 | 1.0% |
| d16 router | 4.58 | 1.81 | rr_small<2,16,8> 4.2/4.1/4.2 | 4.23 | 4.23 | 0.35 | 1.1% |
| d32 qkv_gate | 10.74 | 8.00 | sk2<2,32,3> 9.9/9.9/9.5 | 10.75* | 8.68 | 2.05 | 5.9% |
| d32 o | 5.97 | 3.62 | sk2<2,32,2> 4.8/4.8/5.1 | 6.14* | 5.22 | 0.75 | 2.1% |
| d32 pooler | 15.01 | 14.14 | sk2<2,32,…,5> 15.2/15.2/14.5 | 17.21* | 13.53 | 1.48 | 4.2% |
| d32 unpooler | 14.35 | 14.14 | sk2<2,32,…,3> 14.4/14.2/13.9 | 17.21* | 13.53 | 0.82 | 2.3% |
| d32 router | 5.36 | 1.86 | sk2<2,32,1,1,8> 4.7/4.7/4.7 | 4.29 | 4.29 | 1.06 | 3.0% |
| v1 qkv_gate | 9.07 | 7.90 | sk2<1,8,3> 7.4/7.3/7.5 | 10.64* | 8.57 | 0.49 | 2.0% |
| v1 o | 5.24 | 3.53 | v18s_gemv 4.7/4.8/4.8 | 6.04* | 5.12 | 0.12 | 0.5% |
| v1 pooler | 13.70 | 14.02 | smallm2_gemv 13.4/13.4/13.6 | 17.08* | 13.40 | 0.29 | 1.2% |
| v1 unpooler | 4.06 | 1.76 | sk2<1,8,1> 3.4/3.3/3.3 | 4.19* | 3.73 | 0.33 | 1.3% |
| v1 router | 4.30 | 1.78 | rr_small<1,16,8> 3.5/3.6/3.6 | 4.20 | 4.20 | 0.09 | 0.4% |
| **v16 qkv_gate** | 13.82 | 8.42 | qb_kernel<64,36,128> 13.9/13.1/13.3 | 11.19 | 9.12 | **4.69** | **10.5%** |
| v16 o | 6.87 | 3.99 | ol_projection<128,128> 6.5/6.3/6.5 | 6.53 | 5.61 | 1.26 | 2.8% |
| **v16 pooler** | 20.49 | 14.64 | nvjet 64x64 coopB 21.7/19.5/20.4 | 17.72 | 14.05 | **6.44** | **14.7%** |
| **v16 unpooler** | 19.12 | 14.64 | nvjet 128x64 1x2 18.8/18.1/18.8 | 17.72 | 14.05 | **5.08** | **11.4%** |
| v16 router | 6.09 | 2.21 | rk_kernel 4.6/4.6/4.6 | 4.66 | 4.66 | 1.43 | 3.1% |
| v32 qkv_gate | 17.85 | 9.77 | nvjet 72x64 TNN 20.4/17.3/19.8 | 13.82 | 13.82 | 4.03 | 7.3% |
| v32 o | 8.97 | 4.48 | rl_early_projection<256,64> 8.8/9.1/9.0 | 7.44 | 7.44 | 1.53 | 2.7% |
| v32 pooler | 24.56 | 17.37 | nvjet 128x64 NNN 24.6/24.2/24.1 | 22.76 | 22.76 | 1.80 | 3.2% |
| v32 unpooler | 23.30 | 17.37 | nvjet 128x128 coopA 23.9/23.4/23.0 | 22.76 | 22.76 | 0.54 | 0.9% |
| v32 router | 7.09 | 2.68 | rk_kernel 6.0/5.9/5.9 | 5.15 | 5.15 | 1.95 | 3.5% |

**B.2 Load totals and floor headlines.**
- Loads at the 3-read mean, F_12 in brackets: d1 34.36 (34.08), d16 47.64 (44.89), d32 51.43 (45.26), v1 36.37 (35.03), v16 66.38 (47.49), v32 81.77 (71.94) µs.
- Headline (SGLang held fixed): 1.5487 now; 1.6273 at F_raw (+5.1%); 1.7392 at F_12 (+12.3%); 1.8790 at F_ent (+21.3%).
- At F_12, by row class: m≤32 +3.96%, m=128 +5.74%, m=256 +2.16%.

**B.3 Heaviest-gap rows on the H100** (run 20261031-130201-gpunode06, graph mode, one kernel node per replay)

| row | kernel (grid × block, regs) | H100 graph kernel µs, median | DRAM read | achieved | tensor % |
|---|---|---|---|---|---|
| v16 pooler | nvjet 64x64 coopB (128 × 384, 168) | 25.26 | 69.2 MB | 2.74 TB/s | 31.5 |
| v16 unpooler | nvjet 128x64 1x2 (128 × 384, 168) | 24.35 | 68.2 MB | 2.80 TB/s | 31.8 |
| v16 qkv | qb_kernel (108 × 256, 90; 24 SMs idle) | 16.24 | 40.1 MB | 2.47 TB/s | 26.3 |
| v32 qkv | nvjet 72x64 (128 × 384, 168) | 19.09 | 42.0 MB | 506 TF/s | 47.7 |

**B.4 Candidate-A feed proxy.** Run 20261031-132458-gpunode06; µs, with the ratio to the incumbent in brackets.

| arm | pooler | unpooler | qkv_gate |
|---|---|---|---|
| incumbent kernel_fn | 26.06 | 25.25 | 17.49 |
| 12-bit stream | 21.06 (0.81) | 20.90 (0.83) | 13.42 (0.77) |
| raw BF16 stream | 25.63 (0.98) | 25.20 (1.00) | 15.81 (0.90) |
| 12-bit + decode8 | 24.50 (0.94) | 24.22 (0.96) | 15.14 (0.87) |
| x ring + wgmma only | 14.94 (0.57) | 14.85 (0.59) | 9.70 (0.55) |
| 12-bit + decode + wgmma | 29.55 (1.13) | 29.23 (1.16) | 18.24 (1.04) |

**B.5 Finder scenarios**, as a headline at the mean level

| scenario | headline | change |
|---|---|---|
| A optimistic (H100 proxy stream ratios) | 1.6011 | +3.39% |
| A mid | 1.5864 | +2.44% |
| A with decode exposed | 1.5713 | +1.46% |
| A mid + B | 1.5975 | +3.15% |
| A mid + B + C at 50% | 1.6141 | +4.22% |
