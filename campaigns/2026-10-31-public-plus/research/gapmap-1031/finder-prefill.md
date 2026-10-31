# Prefill gap map (muse_prefill_layer), 2026-10-31, finder report

**Bottom line.** Today's prefill is not "busy at the floor" everywhere. At T=16384 the GEMMs and the MoE are at the power-capped tensor limit. At T≤3012 (the turn-1 and turn-4 loads, which carry two thirds of the headline) about 80 µs per row is lost in the MoE. The time goes to small-expert tiles that are bound by FP4→BF16 conversion, and they run in a separate phase while the tensor pipes sit idle. 10th place (x1.128) is out of reach with realistic values, which give about 1.95. It is reachable only if two large builds both land near their upper bounds, which gives about 2.06. The top score (2.155) is beyond everything this map finds, which tops out at about 2.11.

**Sources and rules followed**
- **Master bytes:** `submission.py` e5558cfe. The scored run is 8b89517bd87544b6 (J1, 1.7836). The checks d114d2e8da984e8a (J2, 1.7848) and c456c4b92c9b447a (J3, 1.7634) ran the same code; only line 2 differs.
- **Judge profiles:** J1 is `muse_prefill_layer/profiles/submission.8b89517bd87544b6.txt`. J2 and J3 are `P2-r{1,2}.*.txt` in `.worktrees/public-plus/campaigns/2026-10-31-public-plus/files/prefill/profiles/`.
- **H100 work:** gpunode06 GPU 4 only, which had no other process (checked with nvidia-smi before every run). It ran through the hopper-probe guard as the operator. The node run directories were removed: `/tmp/USER-hopper` held only the shared `venv-cu130` afterwards, plus another agent's live run.
- **Code:** takemyjob branch `hopper/gapmap-prefill` (b1cb10a), package `probes/gmpf`, reusing the lever finder's `lpf.py` with judge seeds and router seed 6110722306183770365.
- **Run directories** are under `R = ~/git/.takemyjob-wt/gapmap-pf/prototypes/hopper-probe/runs/`:

| label | directory | what it holds |
|---|---|---|
| R1 | `R/20261031-121653-gpunode06/out` | nsys graph-node critical path, GPU metrics per kernel, MoE item trace, ceilings |
| R2 | `R/…-123049-…` | MoE additivity on real routing |
| R3 | `R/…-123813-…` | attention ncu, master vs the public file |
| R5 | `R/…-124409-…` | zero-change MoE SM split |
| R6 | `R/…-124708-…` | glue/GEMM co-run and glue ceiling |
| R7 | `R/…-125243-…` | D1 dry run with load-free anatomy |
| R8 | `R/…-125840-…` | D2 dry run |
| NCU0 | `~/git/.takemyjob-wt/lev-prefill/…/runs/20261031-013401-gpunode06/out/lpf-ncu/details.csv.gz` | the morning lever finder's ncu |

Nothing on musespark was touched: no edits, no checks, no attempt files, no memory writes.

## 1. Per-row table

**How the headline is computed** (`TASK.md:46`): it is the geomean over 3 loads.
- The three-read means give 1.7773 (the board shows 1.7836).
- Turn 1 is at 1.781; t1s is 64% of that load.
- Turn 4 is at 1.738; t4s is 62%.
- First turn is at 1.814; fts is 67%.

**Floors.** All are H200, BF16 tensor at 989 TF/s. Hopper has no FP4 tensor path, and activations must stay BF16.
- **Experts:** 100.66 MFLOP per local route. Route counts come from the judge-seed replica: t1s 1370, t1f 1290, t4s 5567, t4f 5314, ftf0 33269, fts 30127, ftf32 31750, ftf64 30215.
- **Attention:** whichever is larger of FP8 FLOPs at 1979 TF/s and exps at 132 SMs × 16/clk × 1.98 GHz.
- **HBM at 4.0 TB/s:** dense weights 197 MB, plus 28.31 MB per distinct expert, plus KV, plus the minimum activation I/O (12 B per feature).
- **Floor B** is staged: the sum of each stage's max(compute, bytes), plus an 18 µs graph chain.
- **The practical floor** uses the judge's own best rates: cuBLAS at 816 TF/s, MoE at 853 TF/s, attention at 85% of the MUFU limit at 1.8 GHz, plus 24 µs.

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

**Judge kernel split for J1, in µs (GEMM / MoE / attention / glue):**
- t1s: 209 / 285 / 39 / 69
- t1f: 270 / 328 / 746 / 72. The attention figure is attn_fwd 373 plus combine2 373, which overlap through PDL; the real attention span is about 380.
- t4s: 771 / 788 / 62 / 225
- t4f: 857 / 799 / 1511 / 221
- ftf0: 4077 / 3994 / 874 / 1080
- fts: 4133 / 3622 / 310 / 1107
- ftf32: 4091 / 3869 / 4205 / 1084
- ftf64: 4078 / 3617 / 7283 / 1075

J2 and J3 agree within ±5% per class.

**Headline levels:**
- The ln-shares add up to the floor-B headline of 2.671.
- The practical-floor headline is 2.234.
- The TASK SOL headline is 3.217. It counts only 75% of routes, at 4.8 TB/s.

## 2. H100 graph-mode timeline on the three heaviest rows (fts, t1s, t4s)

Each row is a CUDA graph with a cold L2, 7 profiled replays. Sources are the R1 nsys report, metrics and MoE trace, the R2/R7 additivity runs, and the R3/NCU0 ncu data.

**fts** (span 9298.8 µs; graph gaps only 11.5 µs). The critical path runs:
qkv 760 → qk_prep 27 → attn_prep 4 → attn_fwd 299 → o 333 → bnd1 357 → [router 177 + route 67, in parallel with pooler 1469] → gather 158 → moe 3672 → combine 155 → unpooler 1328 → bnd2 622.
Over the whole span, tensor is active 79.2% and SMs 98.8%.

| kernel | tensor % | DRAM R+W % | verdict |
|---|---|---|---|
| qkv / o / router / pooler / unpooler | 94.4 / 95.4 / 92.8 / 97.4 / 98.4 | — | **At the limit.** Clock is power-capped at 1.50–1.59 GHz. cuBLAS alone at the same shapes gives 722–808 TF/s (R1 ceilings). |
| moe_fused | 92.7 (ncu 92.9) | 17 | **At the limit.** 256-row items take 40.4 µs, which is 6.6 TF/s per SM, and are 94% of SM-time. Padding is 1.7% (30688 rows for 30163 routes). |
| bnd1 / bnd2 | ~1 | 90 / 89 | **At the HBM limit:** 0.96 and 0.99 of the copy ceiling (R6). |
| gather / combine | ~1–2 | 67 / 70 | **Structure:** only 0.75 and 0.72 of the copy ceiling (R6). |
| attn_fwd v8 | 24.7 (MUFU 58.8%, R3) | 9 | **Structure:** 10.35 waves of 1366 CTAs, each running only 17 key tiles. The MUFU floor at 1.82 GHz is 140 µs. |
| pooler, while sharing | | | +91–99 µs against running alone (1378, R1 ceilings), because router and route take SMs. This is still cheaper than running the router serially (244 µs). |

The glue (1.3 ms) runs with the tensor pipes idle, and it cannot be hidden by running it on another stream. Forking a 1.88 GB copy against the T=16384 unpooler hid −3% (R5/R6), because the GEMM's CTAs leave no room on the SM.

**t1s** (span 587.8 µs; graph gaps 0.39 µs; PDL wait 30.9 µs). The critical path runs:
qkv 40.4 → preps 5.9 → attn 22.5 + combine2 4.3 → o 21.7 → bnd1 23.3 → [router 23.4 + route 42.9, in parallel with pooler 88.8, which blocks 23.5] → gather 11.4 → moe 280 → combine 9.9 → unpooler 67.8 → bnd2 23.9.
Tensor is active 62.0% of the span and SMs 94.5%.

- **moe (48% of the row) is structure.** It runs in two serial phases.
  - Big tiles (≥128 rows) are tensor-bound from 0 to 135 µs.
  - The small tiles come next, with the tensor pipes idle: GEMM1 at 135–178 µs and GEMM2 at 244–266 µs. These are the ≤64-row tiles of 8 rare experts carrying 155 routes.
  - A 32-row item takes 12.6 µs; a 256-row item takes 42.7 µs.
  - **Additivity (R7):** all 297.5 µs; big alone 215.6; small alone 110.4. The overlap prize is all − max(big, small) = **81.8 µs**. The other rows: t1f 78.0, t4s 82.2. The prize was replicated 4 times at 81.7–82.2.
  - **Load-free anatomy (R7):** I ran with moe flags 3, which skip the X and W loads (`submission.py:2799`). The small set then takes 99.9 µs, 0.90 of its normal time. On 33 CTAs it takes 312 µs, 2.83x longer. **So the small tiles are bound per SM by FP4 conversion, not by HBM.** The public author's own comment says the same: `:2068-2070` "32, 64 (small: weight-conversion-bound)".
- **pooler:** 88.8 µs against 69 alone; sharing SMs costs +19.
- **o projection:** 58–60% tensor. K=1024 gives a short main loop, and 258 CTAs make 1.95 waves.
- **qkv:** 78–83% tensor (126 CTAs on 132 SMs).
- **unpooler:** 83–84% tensor, near the limit for M=753.
- **bnd1 / bnd2:** a 0.43-wave tail, with SMs active only 57–88%.
- **attention:** 27 µs against a 6 µs MUFU floor. It is latency-bound: 0.97 wave, MUFU 41% (NCU0).
- Launch gaps are negligible; per-kernel ramp and tail cost about 32 µs of idle SM-time.

**t4s** (span 2037 µs; gaps 12 µs). The critical path runs:
qkv 159 → attn 68 → o 79 → bnd1 101 → [router 45 + route 60, in parallel with pooler 307] → gather 33 → moe 855 → combine 32 → unpooler 272 → bnd2 101.
Tensor is active 73.8% of the span.
- **moe:** 86.1% tensor. The prize is 82.2 µs (all 817.0, big 734.8, small 119.0), and padding is 8.5%.
- **pooler:** 307 against 253–258 alone, +49 from sharing.
- **bnd1/bnd2:** at the HBM limit (0.85 DRAM).
- **attention:** MUFU 58% (1.94 waves).
- **qkv and unpooler:** at the limit (91–94%).

**Transfer to the H200.**
- Tensor-bound kernels match the judge within ±3%: moe at fts 3622 on the judge against 3672–3781 here; qkv 782 against 760; pooler 1486 against 1469.
- Glue runs 1.2–1.34x faster on the judge: bnd1 280 against 357, bnd2 465 against 622, gather 132 against 158, combine 127 against 155.
- The judge's MoE at t1s (285/290/278) and t4s (788/789/761) matches the H100. That fits a phase bound by conversion, which is compute and should be the same on both GPUs.

## 3. Can we reach 10th place compliantly?

10th is 2.0125, which needs x1.1283 over 1.7836. The top is 2.1551, which needs x1.2083. The arithmetic below is load-weighted row deltas applied to the three-read means (scen4).

| scenario | x headline | about |
|---|---|---|
| S1 alone (realistic / at the full measured prize) | 1.041 / 1.057 | 1.856 / 1.886 |
| S2 with 35% hidden (H100 proxy) / 70% / 90% | 1.024 / 1.049 / 1.064 | |
| S3a: compliance tax recovered (×0.93 on full rows, ×0.85 sliding) / plus exp-split (gray) | 1.014 / 1.025 | |
| **Realistic:** S1 + S2 at 35% + S3a + small-T knobs | **1.093** | **1.95** |
| **Stretch:** S1 at the full prize + S2 at 70% + attention ×0.88/×0.78 + knobs | **1.153** | **2.06** |
| Everything this map finds, including the gray exp-split and S2 at 90% | 1.185 | 2.11 |
| Practical floor / floor B | 1.257 / 1.503 | 2.234 / 2.671 |

**Verdict.**
- 10th is not ruled out by the hardware: both floors sit above it.
- On realistic values (about 1.95) we fall about 3% short.
- Reaching 10th compliantly needs S1 at about 95% of its measured prize, S2 hiding at least about 60% of the boundary traffic, and full-row attention at about ×0.88. S1 and S2 are the two largest builds.
- The top score is beyond what this map finds (2.11), even counting the gray exp-split.

## 4. Structural candidates

**S1: co-schedule the MoE's conversion work at T<8192 (t1s, t1f, t4s, t4f)**
- **What replaces what.** `moe_fused_persistent<0/1>` keeps its producer and its 2 consumer warpgroups for tiles of 128 rows or more. I add a fourth "converter" warpgroup for tiles of 64 rows or fewer. It converts their FP4 weight k-blocks to BF16 in a 2 × 16 KB shared-memory buffer and issues their SS-WGMMA (m64n32/64k16) while the consumers work on big tiles. The plan splits into two queues by tile rows.
  - Register budget: 128 threads × 40 + 128 × 96 + 256 × 168 = 60.4K, under 64K.
  - Shared memory is 213 of 227 KB today (4 × 42 KB ring + 44 KB staging, `:2061-2063`), so NSLOT drops from 4 to 3.
- **Why existing knobs can't do it.** Order 2 (interleave) gains only −6.4/−5.6/−0.8 µs (R2). A zero-change SM split gains −3.8 at t1s and loses +6.1 at t1f and +20.1 at t4s (R5). The small set is bound per SM, so it needs about 40 SMs.
- **Expected gain.** At 75% of the prize (t1s −60, t1f −55, t4s −60, t4f −60, first-turn rows −20): x1.041. At the full prize: x1.057.
- **Transfer class.** Compute: CUDA-core conversion overlapped with tensor work. The H100 predicts it well: the judge's MoE equals the H100's on the small-T rows. Confidence is medium-high on the prize and medium on how much of it fits in shared memory and registers.
- **Compliance.** Conversion and k16 order are identical; I need to check bit-identity between SS and RS WGMMA. There is no precision change, and it stays graph-safe.
- **Build size:** about 400–600 lines of CUDA, which keeps the file under about 330 KB.
- **Milestones:**
  1. A small-only converter path, at or below 105 µs, bit-identical.
  2. Both roles running concurrently on the D1 harness: t1s at or below 235 µs, t4s at or below 755. Kill if t1s gains less than 30 µs.
  3. A prepare-time race against the incumbent on skewed routing (the CLAIMS:83 rule). The incumbent is the fallback.
  4. 2 checks.
- **Closest CLAIMS.**
  - CLAIMS:86 (the knob version, MOE_ORDER_SMALL=2): sequential within each SM.
  - CLAIMS:79 and LESSONS:180/185 (2 CTAs per SM at the low class): lost under skew because popular experts' slabs were split. S1 keeps popular experts on full slabs.
  - CLAIMS:68: attacked bytes in flight, not conversion issue.
  - CLAIMS:70/71 (lead): the same diagnosis on the old kernel; now measured on the foundation kernel.

**S2: fuse the boundary passes into the GEMM epilogues (o projection + bnd1, unpooler + bnd2)**
- **What replaces what.** The cuBLAS o projection and unpooler plus `boundary_rc` become persistent BF16 WGMMA GEMMs launched in clusters that own full rows: 16 CTAs × N=512 cover the 8192 columns of each 64-row block. RMS partials are reduced in DSMEM, and a warp-specialized epilogue (s0 read, h/hidden/s2 write) overlaps the next block's main loop.
- **Rows:** all. The largest is the boundary pair at 745 µs per first-turn row; t4s has 151 and t1s 39.
- **Why fusion is the only way.** The boundaries already run at 0.96–0.99 of the copy ceiling, and stream overlap hides −3% (R6).
- **Expected gain.** The H100 proxy (R8) adds 268 MB of epilogue traffic to a cuBLAS GEMM at T=16384 and costs +52–61 µs, hiding only 31–41%. At 35% hidden that is x1.024; a custom overlapped epilogue at 70% gives x1.049, and 90% gives x1.064.
- **Transfer class.** Mixed. It hides HBM traffic under tensor time, and the H200's headroom here is unknown, so D2 measures it.
- **Compliance.** The math is the same: FP32 residual and norms, BF16 h/hidden. Only the summation order changes.
- **Risk.** Matching cuBLAS's 94–98% tensor use. CLAIMS:77 records custom dense GEMMs at 0.53–0.86x of cuBLAS, and LESSONS:133 records an epilogue fusion that lost 1–3%.
- **Build size:** large, about 800–1200 lines of CUDA.
- **Milestones:**
  1. A custom o projection at or above 0.97x cuBLAS at T = 16384, 3012 and 753; stop otherwise.
  2. D2 on the H200.
  3. Fused o projection + bnd1 at or below 0.85x of (o + bnd1) at fts.
  4. Unpooler + bnd2.
  5. A race per T class with cuBLAS + boundary as the fallback, then checks.
- **Closest prior work.** The incumbent byte diet is BND_RECOMPUTE (`:145-146`). There is no prior claim on overlapping boundary traffic with GEMM main loops.

**S3: attention exp throughput**
- **The tax, measured at the kernel level (R3):** at identical inputs, master's attention is slower than the public file's by 6.8% at t1f (371 µs vs 348), 6.7% at t4f (1430 vs 1340), 4.9% at ftf0 (856 vs 816), 7.4% at ftf64 (7120 vs 6630) and 1.5% at fts (302 vs 297). MUFU use is 69–75% against 72–81% for the public file. At ftf64 the exp efficiency is 70.6% against 75.9%.
- **The three parts:**
  - (a) The pf-tax lane (`campaign/pf-tax` 0aa805a3, K2 arms, in flight) owns recovering the tax: x1.014.
  - (b) An exp-split: put about 15% of the exps on the FMA pipe as a Cody-Waite/minimax polynomial with error at or below ex2.approx's. That is ×0.92 more on full rows, +1.1% on the headline. It changes the exp implementation, so **it needs your ruling before any check.**
  - (c) A persistent scheduler for the sliding-window variant (variant 8), taking fts/t4s MUFU use from 59% to about 72%: +0.3%.
- **Transfer class.** Compute and issue, the class that amplified 3–4x on the judge (PP 14cec684, DPP 9d726d84). Confidence is high on direction.
- **Build size:** (b) about 150 lines plus an exhaustive accuracy script; (c) about 200 lines.
- **Fallback:** the existing `attn_choice` per class.
- **Closest CLAIMS.** CLAIMS:78 (shelf): same idea, but its base was a rebuilt FA3 that ran 1.31–1.34x slower. That blocker is gone on the foundation kernel.

**Closed by this map:**
- The dense GEMMs and MoE at T=16384 are at the limit.
- The boundaries are at the HBM floor.
- Stream overlap of glue under GEMMs hides −3%.
- A zero-change MoE SM split and MOE_ORDER 2 gain under about 6 µs.

## 5. H200 diagnostic checks

Each check runs only in `prepare()`, at the end, after `attn_choice`. `kernel_fn` stays byte-identical. `st` is restored in `finally` blocks and buffers are freed. Only aggregate lines are printed. Each check adds under 1 s to prepare. The reference code is `probes/gmpf/gmdiag.py` (b1cb10a): drop the `m.` prefixes and it pastes into master. Each check has been dry-run on the H100.

**D1: does S1's prize hold on the H200?** (`gmdiag.d1`, about 30 minutes for a builder)
- **What it does.** It builds synthetic routings with `_routes_from_counts` (`:1028`) using the replica's per-expert counts (t1s, t1f, t4s). It times graphs with a cold L2 for: all experts, big only (>64 rows), small only at 132, 66 and 33 CTAs, small with flags 2 and 3 (`st["moe_flags"]`, `:2799`), and big with flags 3. It adds an item trace via `st["moe_dbg"]` (`:401`). Output is `GMD1 …` lines.
- **H100 reference (R7):**
  - Prize 81.8 / 78.0 / 82.2 µs.
  - Small with no loads ÷ small: 0.90 / 0.91 / 0.94.
  - Small on 66 CTAs ÷ small on 132: 1.53.
  - 32-row items: 12.6 µs, or 11.7 with no loads.
- **Pre-registered rule:**
  - **GO S1 to phase A** if the H200 prize is at least 60 µs at both t1s and t4s and the no-load ratio is at least 0.80.
  - **KILL S1** (node-only) if the prize at t1s is under 30 µs.
  - If the no-load ratio is under 0.65, the H200 small set is load-bound. Re-plan S1 as a bytes-in-flight design.
  - Between 30 and 60 µs, re-cost at 0.75 × prize; phase A only if that clears +2.5% on the headline.

**D2: can a tensor-bound GEMM at T=16384 absorb boundary-sized epilogue traffic on the H200?** This is S2's premise (`gmdiag.d3epi`, about 20 minutes).
- **What it does.** It compares `torch.mm` against `addmm` with beta=1 (which also reads a [T,N] BF16 C) and against BF16→FP32 output, at K=1024 and 4096 with N=8192. It reports the copy ceiling and the fraction hidden.
- **H100 reference (R8):** about 31–41% hidden.
  - o projection: +58.1 µs for 268 MB (34% hidden).
  - unpooler: +61.1 µs (31%) with the extra read, +52.0 µs (41%) with FP32 output.
  - Copy ceiling 3.03 TB/s.
- **Rule:**
  - If the unpooler shape hides at least 60% → S2 goes to phase A, costed at x1.049.
  - If both shapes hide 30% or less → S2 is costed at x1.024 and parked behind S1.
  - Anything in between → S2 goes to phase A only if S1 is killed.
- **Caveat:** cuBLAS may pick a different kernel for `addmm`. The rule's margins are wide to allow for that.

**D3: glue bandwidth against the H200 copy ceiling** (`gmdiag.d3glue`, about 30 minutes)
- **What it does.** At T=16384, using SCEN counts (`:1068`), it measures copy and read ceilings, then `moe_gather_plan`, `_combine_kernel`, and `boundary_rc` modes 1 and 2, each in its own graph. It reports TB/s and the ratio to the copy ceiling.
- **H100 reference (R6):** gather 0.75, combine 0.72, bnd1 0.96, bnd2 0.99, against a 3.03 TB/s copy.
- **Rule:**
  - If gather or combine is at or below 0.78 of the H200 ceiling, a bandwidth pass on it becomes a phase-C rider. Expected −(1 − r/0.92) × judge time, about −30 µs per first-turn row each, or +0.2–0.3% on the headline. S2's boundary share is re-costed.
  - If both are at or above 0.85, close the glue-bandwidth question.

**Not worth spending a judge check:** glue/GEMM stream co-run. It is answered on the H100 (−3%, R5/R6), and the reason is SM residency, which the H200 shares.

**Honest gaps**
- **No H200 timeline.** The judge profile is a single eager call, so there is no H200 graph-mode timeline. I use the H100 graph runs and the measured per-class transfer between the two GPUs.
- **Routing is a replica.** I cannot observe the judge's real distinct-expert counts. The replica's MoE times match the judge within ±5%.
- **Capture fractions are estimates.** S1's and S2's fractions are design estimates, not measurements.
- **ftf32 was not profiled.** ftf0 and ftf64 bracket it.
- **H100 timing noise.** Whole-layer spans vary ±3–5% between runs (fts 9298–9535 µs, t1s 588–607 µs). The comparisons here are back-to-back within one process on a dedicated GPU, which is why there are no A/A pairs.
- **D1–D3 are not yet judge-ready.** They are dry-run reference code, not a built check file.
