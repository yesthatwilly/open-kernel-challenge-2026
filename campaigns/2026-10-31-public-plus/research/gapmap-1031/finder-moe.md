# muse_moe_w4a16 gap map: finder report, 2026-10-31, about 20:00Z

**Scope.** Master moe is public alygg77@6cae49a plus M1a NOFENCE: `muse_moe_w4a16/submission.py`, 326,400 B, sha c737e460. The board is 2.3867. I did not change musespark.

**H100 work.** Branch `hopper/gapmap-moe` in takemyjob (commits 47352b6 and 559e3aa), package `prototypes/hopper-probe/probes/gapmoe/`. I ran 7 times on gpunode06 GPUs 3/4/5, taking each GPU only when it was idle (the run script refuses a busy GPU), and never used GPUs 0, 2 or 7. hopper-probe removed each node run dir, and `gapmoe-clean` removed the build cache. Run-dir names use workstation time (UTC−7): `122014` is 19:20Z and `125106` is 19:51Z. Every comparison below is an interleaved, paired run, so whatever else was running on the node affects both arms equally.

**Main finding.** On the H200 the small-batch builds (d1, v1, d16) are limited by the in-SM consumer (dequant and MMA), not by HBM. The large-batch rows are limited by both.
- At H200 bytes-to-compute balance (the R67 proxy), cutting the TMA bytes further from 67% to 50% barely moves the skinny builds: d1 x1.019, v1 x1.018, d16 x1.029.
- The WGS rows still speed up when bytes drop: d32 x1.126, v16 x1.107, v32 x1.077 (run `20261031-125106`, 40 paired reps).
- This explains today's transfer record: S6/S6M byte cuts did not transfer, while issue and scheduling levers did.

---

## 1. Per-scored-row table (judge, floors, gap)

**Judge reads of master bytes:**
- `eaf356366ca5459c`: check, sfo3c, 2.38673, `prepare_s` 165.6.
- `f4e16629be334b14`: scored, sfo3, 2.38669, `prepare_s` 168.2.

These are two separate runs on different nodes, not a cached duplicate.

**Near-master reads** (public bytes, same kernel apart from the read-back fence): `b3510dcc6553442b` 2.3648, `99b14fedc99142e9` 2.3761, `0882294eec144bea` 2.3670.

**How the headline is computed.** Each load is one shape with weight 60, so the headline is the geometric mean of the 6 row ratios (TASK.md:42).

**Judge profile.** The profile is one eager call. On every row and in both reads, one kernel (`okc::moe_fused<…>`) takes 100%. The phase split is in §2.

| row (bs) | w | judge us: eaf35636 / f4e16629 (mean) | near-master us: b3510dcc / 99b14fed / 0882294e | SGLang us | row ratio | judge-profile moe_fused us | MB moved¹ | floor @4.0 / @4.8 TB/s / judge SOL | H200 size-matched ceiling² → stream+chain floor³ | FLOP floor⁴ | gap us (x) | share of headline gap⁵ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 (1) | 60 | 25.19 / 24.87 (25.03) | 26.05 / 25.06 / 24.91 | 77.3 | 3.088 | 29.4 / 23.2 | 73 | 18.3 / 15.2 / 11.4 | 3.76 → 21.9 | 0.3 | 3.1 (1.141) | 12.8% |
| decode 16 (16) | 60 | 91.33 / 91.87 (91.60) | 91.41 / 97.36 / 91.79 | 186.4 | 2.034 | 77.1 / 84.2 | 325 | 81.3 / 67.8 / 52.5 | 4.38 → 76.7 | 2.6 | 14.9 (1.194) | 17.2% |
| decode 32 (32) | 60 | 127.90 / 123.73 (125.82) | 125.21 / 124.46 / 126.70 | 209.7 | 1.667 | 119.2 / 143.6 | 444 | 110.9 / 92.4 / 70.8 | 4.47 → 101.8 | 5.1 | 24.0 (1.236) | 20.5% |
| verify 1 (8) | 60 | 63.49 / 63.11 (63.30) | 67.45 / 63.67 / 68.70 | 142.2 | 2.247 | 49.8 / 57.0 | 244 | 60.9 / 50.8 / 37.3 | 4.31 → 59.0 | 1.4 | 4.3 (1.073) | 6.8% |
| verify 16 (128) | 60 | 173.49 / 175.70 (174.59) | 174.98 / 178.49 / 178.51 | 464.5 | 2.661 | 171.9 / 183.4 | 649 | 162.2 / 135.2 / 101.3 | 4.52 → 147.4 | 19.4 | 27.2 (1.185) | 16.4% |
| verify 32 (256) | 60 | 209.59 / 208.76 (209.17) | 212.55 / 210.75 / 213.51 | 617.9 | 2.954 | 197.2 / 218.8 | 703 | 175.8 / 146.5 / 108.9 | 4.53 → 159.4 | 39.5 | 49.8 (1.312) | 26.3% |

**Footnotes:**

1. **MB moved.** The mean number of distinct local experts per row (200 seeds, lev-moe run `20261031-010412-gpunode06` levmoe-bd) is 2.58 / 11.48 / 15.65 / 8.60 / 22.84 / 24.69. Each expert is 28.311552 MB (w13 + w2 E2M1 codes plus e4m3 scales), and x/out are added. On the profiled draws, DRAM bytes equal touched experts × 28.31 MB within 0.1–1%, except v32 (+43.5 MB, see §2).
   - The codes are uniform random (nvfp4.py:109-115), so they cannot be compressed. The public team measured scale entropy at 3.71 bits and at most about 5% saving (H200_MEASURED_FACTS.md:108).
2. **H200 size-matched read ceiling.** These are alygg77's measured H200 NCU values (`/home/user/git/okc-muse-spark-kernels/docs/H200_MEASURED_FACTS.md:32-34`: 66 MiB 3.63–3.81, 128 MiB 4.19, 256 MiB 4.33, 512 MiB 4.52, 1–2 GiB 4.51–4.61 TB/s), interpolated on log size.
   - The judge SOL (reference.py:125-138) credits only 0.75 × distinct experts, so it sits about 25% below a reachable floor.
3. **Launch/dependency chain floor in graph mode.** Each call is one graph node (§2), and the judge subtracts the launch (records `replay_overhead` 3.7–4.3 us).
   - The minimal in-kernel chain is: topk read and route (about 1 us), the grid barrier (1.0–1.7 us, H200_MEASURED_FACTS.md:51), and a one-pass final (0.5 us or more).
   - I used 2.5 us for M ≤ 32 and 4.0 us at v16/v32.
4. **FLOP floor.** Routes (2.6 / 25.6 / 49.9 / 13.7 / 190.4 / 387.8) × 100.7 MFLOP at 989 TFLOP/s BF16 dense. It never binds.
5. **Share of the headline gap.** Each row's share is ln(gap)/Σ ln(gap); all weights are equal.
   - With the @4.0 floor instead, the gaps are 1.370 / 1.126 / 1.135 / 1.039 / 1.077 / 1.190 and the shares are d1 37.2%, d16 14.1%, d32 14.9%, v1 4.6%, v16 8.7%, v32 20.5%.

**Headline at each floor:** stream+chain 2.835 (x1.188), @4.0 TB/s 2.748 (x1.151), @4.8 TB/s 3.30.

**Headline sensitivity:** 1 us is worth 0.666 / 0.182 / 0.132 / 0.263 / 0.095 / 0.080 % headline (d1 / d16 / d32 / v1 / v16 / v32).

**Row noise** across the 5 reads of near-identical bytes: v1 63.1–68.7 us, d16 91.3–97.4, d32 123.7–127.9.

**A second floor binds on the H200: the consumer plus structure floor.** This is the time with TMA bytes cut to 33% (R33, H100 clock 1830 MHz, run `20261031-125106`):
- d1 25.8, d16 85.9, d32 107.0, v1 66.6, v16 155.1, v32 184.7 us.
- The judge's d1 (25.0), v1 (63.3) and d16 (91.6) are already at or within about 7% of it.
- So the stream floor in the table cannot be reached on those rows without a faster consumer.

---

## 2. H100 graph-mode timeline: v32, d32 and d16 (the heaviest gap shares), with d1 as a note

**Graph level.** nsys with cuda-graph-trace=node, 30 replays per row, fresh routing per replay, a 256 MiB flush before each (run `20261031-122316-gpunode06/out/gapmoe-nsys/nsys/st_cuda_gpu_trace.csv.gz`):
- Every row runs exactly one kernel node per replay: `moe_fused`, grid 132, so 1 CTA per SM.
  - Skinny builds: 288 threads, 149–152 registers, 215 KB smem.
  - WGS builds: 576 threads, 96 registers, 231 KB smem.
- Flush end to kernel start is 3.7–4.1 us, which is the graph launch. There are no inter-kernel gaps and no extra launches.
- Kernel medians, H100 / R67 (us): d1 36.4/28.0, d16 114.6/85.9, d32 168.2/122.0, v1 95.5/67.6, v16 236.5/171.4, v32 272.2/203.5.
- R67 against the judge: d32 −3%, v16 −2%, v32 −3%, d16 −6%, v1 +7%, d1 +12%. R67 is a good H200 proxy for the WGS rows and a looser one for the skinny rows.

All idle time is therefore inside the kernel.

**Per-CTA stamps.** These are `%globaltimer` stamps on the same master bytes (c737e460) and the same R67 file (de370466), from lev-moe runs `20261031-012835` (tlraw, master), `20261031-045221` (r3t, R67TLAB, rows 1/8/16, including credit and Bgo) and `20261031-050519` (r6t, R67, rows 32/128/256). The table is R67 us (H200-like), median of reps, with min–max over 132 CTAs:

| row (build) | start chain (routed / x gathered) | A_done | A→B (A_done med → Bgo) | B_done | B-end spread + barrier | final | span |
|---|---|---|---|---|---|---|---|
| v32 (WGS) | 3.3 / 8.4 (5.1–10.0) | 119.8 (112.4–130.3) | hidden by ring | 177.9 (169.5–186.2) | 9.1 | 6.8 | 193.8 |
| d32 (WGS) | 2.0 / 4.4 | 83.7 (78.3–88.7) | hidden by ring | 121.3 (118.4–126.3) | 5.8 | 1.0 (max 4.8) | 128.1 |
| d16 (MID) | 2.8 | 53.9 (52.5–55.6) | 2.4 | 82.0 (81.0–84.1) | 3.1 | 0.4 | 85.5 |
| d1 (TINY) | 2.6 | 12.3 (11.1–12.7) | 1.9 | 19.3 (18.6–19.6) | 1.4 | 0.4 | 21.1 |

**Ablations.** Judge-style, interleaved, paired, 40 reps. Each arm is timing-only with invalid output. The NF and R-scaling arms change only the cited lines; NW removes the 4 skinny doneA polls and the WGS doneAs poll (one example of each cited).

- **NW** removes all phase-B dependency waits (polls at submission.py:837, 1006, 1621, 2015). Run `20261031-123110`. R67 results: d1 x1.040 [1.033, 1.051], v1 x1.020, d16 x0.999, d32 x0.999, v16 x1.000, v32 x1.001.
  - The A→B handoff only costs time at d1 and v1. The 2.4 us gap at d16 is not recoverable.
- **NF** skips the final pass after the barrier (inserted at :2649). Run `20261031-123701`. R67 results: d32 x1.029 (3.6 us), v16 x1.025 (4.0 us), v32 x1.035 (6.5 us), v1 x1.009, d16 x1.000.
- **Byte scaling** (TMA bytes per unit at 67%, 50% and 33%). Run `20261031-125106`. R50 against R67: d1 x1.019, v1 x1.018, d16 x1.029, d32 x1.126, v16 x1.107, v32 x1.077. R33 against R50 is at most x1.028 everywhere.
- **Per-SM HBM rate** (smmap: 1 CTA per SM, 4 MB each, 8 reps, run `123701`):
  - Per-SM time p10/med/p90/max is 180.3/181.9/183.1/184.5 us.
  - The rank correlation between consecutive reps is 0.02–0.20.
  - So there is no stable slow-SM set; the B-end spread comes from work and scheduling, not hardware.
- **H100 size-matched read ceiling** (LDG.128, 264×1024 threads, cold L2, graph, overhead 3.4 us subtracted; run `122316` gapmoe-bw): 73 MB 2.86, 244 MB 3.03, 325 MB 3.06, 443 MB 3.08, 647 MB 3.09, 699 MB 3.10, 2 GiB 3.11 TB/s.

**ncu.** One eager call per row, cold cache, clock-control none (run `122316/out/gapmoe-ncu/ncu-raw.csv.gz`). Draw stats are from run `124042`. Each cell gives master, with R67 in brackets.

| row | time us | DRAM read MB (draw) | TB/s, % of H100 ceiling | issue active | ALU / FMA-heavy | tensor (gmma or hmma) | warps/SM, eligible per scheduler | top stalls per issue (master) |
|---|---|---|---|---|---|---|---|---|
| v32 | 230.9 [181.1] | 609.8 (20 experts = 566.2 + **43.5 re-read**) | 2.64, 85% [2.25, 73%] | 41.7% [55.0%] | 33.4/28.5 [43.6/37.2] | 8.1% avg, 44% max [10.6/57] | 18, 0.89 [1.21] | long-scoreboard 2.98, not-selected 1.15, gmma 1.13, barrier 0.92 |
| d32 | 173.9 [129.2] | 456.9 (16 = 453.0) | 2.63, 85% [2.37, 77%] | 40.6 [54.3] | 32.4/27.7 [44.5/38.1] | 8.0 [11.0] | 18, 0.88 [1.19] | long-scoreboard 4.40, not-selected 1.16, gmma 1.05 |
| d16 | 111.0 [83.0] | 283.3 (10 = 283.1; the split 16-token expert's second read came from L2) | 2.55, 84% [2.28, 74%] | 41.1 [53.1] | 38.0/31.1 [49.9/40.8] | 13.0 [17.1] | 9, 0.62 [0.81] | long-scoreboard 1.53, wait 0.62, barrier 0.54 |
| d1 | 30.3 [24.5] | 56.7 (2 = 56.6) | 1.87 [1.54] | 31.9 [40.7] | 28.4/24.7 [37.1/32.2] | 9.3 [12.1] | 9, 0.46 [0.60] | long-scoreboard 2.08, barrier 1.01 |

**Hardware limit versus structure, per row (R67, H200-like):**
- **v32.** About 88% of the span is the A+B stream, running at about 89% of the size-matched rate and limited by HBM and the consumer together.
  - Structure costs about 24 us (12.5%): start 8.4 us (route 3.3, x gather partly hidden by the 3-slot ring), B-end spread and barrier 9.1, final 6.5–6.8.
  - There is also a redundant pass. A hot expert with more than 64 tokens is re-streamed once per 64-token job (OKC_FT_T 64 at submission.py:244, WGS_NT at :1823). On the profiled draw that added 43.5 MB of DRAM (7.1%). Over 40 reps it averages +1.42 passes per call (jobs 25.80 against 24.38 distinct).
- **d32.** Structure is start 4.4 us, B-end and barrier 5.8, final 3.6 (from NF; the B4 shelf item recovers about 30% of it), about 13.8 us in total (11%). The rest is stream time that is still byte-elastic (R50 x1.126).
- **d16.** The chain is route 2.8, A→B 2.4 (not recoverable per NW), barrier 3.1 and final 0.4. The phases are consumer-bound at H200 balance (R50 x1.029): 2 compute warps per scheduler with 0.81 eligible and 53% issue. The skinny engine itself is the limiter.
- **d1.** The chain is 6.3 of 21.1 us (30%): route 2.6 (the first units are already issued before routing by FAST1, :2424), A→B 1.9 (1.6 us recoverable per NW), barrier 1.4, final 0.4. The rest is consumer-bound (R50 x1.019).

---

## 3. Leaderboard: is 10th place reachable under our compliance ruling?

`okc leaderboard muse_moe_w4a16 --json`, read at about 19:05Z:
- Ours: 2.3867, rank 118 of 1250.
- 10th: zzz0906, 2.5899 (`c3f36d48b0bc4b42`). Reaching it needs x1.0851, which is −7.8% on the mean row time.
- Top: dolanzhao, 2.7578 (`850d69d2713343c4`). Reaching it needs x1.1555.

**Against the ideal compliant floors:**
- Stream+chain floor, headline 2.835: 10th needs 47.5% of the log gap, the leader 84%.
- 4.0 TB/s floor, headline 2.748: 10th needs 58% of the log gap; the leader is beyond it (102%).

**Against what the current architecture can reach:** the consumer-bound floors (R33) are the binding H200 limit.
- d1, v1 and d16 are already at their floors, so they have about 0% room with today's consumer.
- Even if d32, v16 and v32 dropped to their floors (107.0 / 155.1 / 184.7 us, as if bandwidth were infinite), the headline would be 2.3867 × exp([ln(125.82/107.0) + ln(174.59/155.1) + ln(209.17/184.7)] / 6) = **2.553**.
- Adding every chain fix in §4 at its upper bound (about +1–1.5%) gives about 2.58–2.59.
- So 10th place needs every lever to land at its upper bound at the same time.

**Verdict.**
- With the public kernel, 10th place is not reachable under our ruling. A realistic plan ends around 2.42–2.45.
- Physics alone would allow it (floor 2.835), but only with a consumer that is at least 15–20% cheaper per weight on every row.
- For reference, the public team's KPERM3 (−8.3% instructions per weight word) bought +1.25% on the board (H200_MEASURED_FACTS.md:115). Today's consumer runs at 18.44 instructions per word against a decode minimum of about 10–11.
- The leader's 2.758 implies about 96% of the stream ceiling on average, including d1 at 73 MB. That is not credible for a BF16-dequant consumer that is consumer-bound or co-bound.
- One unverified explanation: a kernel that converts E2M1 to E4M3 (exact) and runs FP8 MMA with FP8 activations would roughly halve the dequant work and escape this limit. That is outside our ruling (TASK.md:77: activations stay BF16).

---

## 4. Structural candidates

### C1. A new skinny engine with no A→B handoff (rows d1, v1, d16)

**What replaces what.** It replaces the TINY, TINY2 and MID builds. Today's skinny unit is 16 rows × full K, with 8 warps splitting K, a cross-warp reduction every 36 KB unit, an inter publish, a fence, a doneA credit and a poll (:837/:1006/:1621). The new engine has two parts:
- (a) Row-split warps over K-streamed stages. Each warp owns 16 rows; mma.sync m16n8k16 with N ≤ 16; 128-row × 256-K stages of 18 KB through the existing TMA ring; one epilogue per 128-row block and no cross-warp reduction.
- (b) A K-fused A→B step per CTA. Each CTA owns a contiguous range of (expert, 16-inter-column group) and computes gate/up and SwiGLU for those columns. It then applies the matching W2 K-slice and does an FP32 red.add of the 4096 × n partial into ybuf.
- W2 is repacked once in prepare as [expert][K-group][4096 rows], a lossless byte permutation.
- The grid barrier and the FINCOL final stay as they are.

**Expected gain.**
- Upper bound, if the consumer reached the H200 stream floor: d1 25.0 → 21.9, v1 63.3 → 59.0, d16 91.6 → 76.7, which is **+6.6% headline**.
- Measured parts: removing the handoff is worth d1 x1.040 and v1 x1.020 (NW), about +1.0%. The rest needs at least 15% more consumer throughput (from the R33 floors above).
- Planning number: +2.0%, with p ≈ 0.35, so **EV about +0.7%**.
- The red.add traffic must stay below the d1 handoff gain of 1.6 us. At d1 it is about 21 KB per CTA; at d16 about 46 KB per CTA.

**Transfer class.** Compute/issue plus latency, which is the class that binds these rows on the H200 and has transferred before. The plain H100 is HBM-bound on these rows (R100 → R67 is x1.28–1.39), so gate only on R67/R50.

**Compliance.** BF16 MMA on exactly dequantized NVFP4, with FP32 accumulation. The red.add order is nondeterministic, which is the same class as master's existing skinny sum-of-squares atomics and WGS stream-K red.add. Weights are only repacked; x and inter stay BF16.

**Build.** About 900–1200 lines of CUDA plus about 40 of Python. It reuses the DQ7 decode, TMA ring, routing and final. The compile count is unchanged.

**Milestones:**
- M0: a warm-L2 consumer microbench at N = 1/8/16. GO if it reaches at least 1.25x weights/us over the current skinny consumer.
- M1: a TINY replacement. Gate: R50 at least +8% and R67 at least +5% at d1, and 0 failures in 600 checked replays.
- M2: TINY2 and MID.
- Fallback: the incumbent build, chosen per M in `kernel_fn`.

**Closest prior items:**
- NW16, a round-2 finder dead end (`campaigns/2026-10-31-public-plus/research/levers-1031b.json`): 16 warps with 4 K chunks each gave R67 d1 −9%, v1 −7%, d16 −4.9%. It made the cross-warp K reduction bigger; C1 removes it.
- LLA, a fence-free handoff (R67 v1 −1.75%). It doubled the phase-B x bytes; C1 has no inter round trip at all.
- EPF on our old engine (CLAIMS.md:110, refuted).

### C2. Move the end-game out of the WGS kernel into a second kernel (rows d32, v16, v32)

**What replaces what.** Today's in-kernel grid barrier and final (:2613–2710) run one token per CTA with two L2 passes over ybuf and two CTA syncs per token. At d32 only 32 of the 132 CTAs do any final work.
- `moe_fused` would stop after its phase-B share, keeping the flag reset that the next launch needs.
- A second kernel, `moe_final`, launches in the same `okc_run` with the programmatic dependent launch (PDL) attribute; `launch_dependents` is already issued at :2612.
- It waits on `griddepcontrol.wait`, then runs 2 CTAs per SM, one token per CTA, keeping the y rows in registers so there is a single L2 pass.

**Expected gain.**
- The NF ceiling is 3.6 / 4.0 / 6.5 us at d32 / v16 / v32.
- New costs: the PDL hand-off (0.42–0.70 us, H200_MEASURED_FACTS.md:109) plus a one-pass final (about 1–2 us). The in-kernel barrier goes away (about 0.8 us: barrier stamp minus the latest B_done).
- Net: −2.0 / −2.3 / −4.3 us, which is +0.26 + 0.22 + 0.34 = **+0.8%** (upper bound +1.4%). With p ≈ 0.5, **EV about +0.4%**. It replaces the B4 shelf item.

**Transfer class.** Latency (end-of-kernel serialization). The H200 proxy is more sensitive here than the plain H100 (NF x1.025–1.035 against x1.017–1.027).

**Compliance.** Same FP32 sum of squares, rsqrt and weighted sum, with BF16 output. The sum of squares becomes a fixed tree, which is deterministic. The second kernel lives in the same load_inline module. Risk: the generation and reset logic for the ss copies has to move, which needs a 10k-replay soak as M1a had.

**Build.** About 150–250 lines of CUDA and no new compile.

**Milestones:**
- M0: the SASS of the main-loop kernel must not change. The round-2 finder's FIN2 and PBLL lost 2–5% from codegen changes at the 96-register cap; C2 removes code from that kernel rather than adding it.
- M1: R67 gates of v32 at least +2%, v16 and d32 at least +1%, two runs each.
- M2: a soak plus adversarial bs 17–1000, then 2 judge checks.
- Fallback: a compile switch back to the in-kernel final.

**Closest prior items:** B4 (CLAIMS.md:140, shelf); FIN2 and F1 (finder dead ends: in-kernel final edits gave R67 d32 −3.5%, v16 −2.2%); the old engine's per-token last-arriver combine fused into the w2 epilogue (CLAIMS.md:102, refuted).

### C3. WGS feed riders: one pass per hot expert, and no global x gather (v32, plus a little on v16 and d32)

**What replaces what.**
- (a) Jobs of more than 64 tokens currently re-stream the expert's weights once per job. Instead, run all of an expert's jobs over the same smem stages: the producer holds a slot until the last job has its codes in registers. That gives one HBM pass per expert.
- (b) Drop the global x gather (`wgs_gather`, :2027) and its `xready` poll (:2012). Instead, the x-producer warp copies the x rows it needs for each stage directly from `hidden_states`.

**Expected gain.**
- (a) At v32 the redundant passes add about 4.4% bytes. WGS rows respond to bytes with elasticity about 0.29 (R67 → R50: x1.077 for −25% bytes), so about −2.7 us, or +0.2%.
- (b) The exposed gather time is 1–3 us of the 2.4–5.1 us gather (routed → gathered), worth +0.1–0.25%.
- Total about +0.3–0.5%, **EV about +0.2%**. This is a rider for C2's check.

**Transfer class.** (a) is bandwidth, which DX2 decides. (b) is latency.

**Compliance.** Nothing new.

**Risk.** (a) changes code inside `wgs_seg`, which carries the codegen risk seen with PBLL.

**Closest prior items:** MO2 (CLAIMS.md:130, refuted on the old engine; it relied on L2 hits rather than a single pass by construction); x-copy rungs (CLAIMS.md:53, old Engine A, T ≤ 8).

**Totals.** Summed EV is about +1.3%; the sum of upper bounds is about +8%. No single candidate clears the 2.5% bar on its own. C1 holds the largest pool and depends on DX2.

---

## 5. H200 diagnostic checks (prepare-only; `kernel_fn` and every timed build byte-identical to master)

All three need a guard: skip if prepare has passed 150 s on entry, and skip the runs if compilation finishes after 260 s. The judge's prepare for master is 163.6–168.2 s, and each extra build adds about 40–46 s, or about 50 s for two built in parallel threads. Print tagged `GAPDIAG` lines only. Nothing is cached into the timed path; diagnostic tensors are freed. There is no harness patching, no subprocess and no network.

### DX2. Byte-elasticity race on the H200 (highest value; the H100 cannot answer it)

**Builds.** TINY-R50 and MAIN-R50: the same `CUDA_SRC`, with the byte count at the 8 TMA expect_tx/copy sites reduced from the unit size to 18432. These are exactly the sites in `probes/gapmoe/sub_R50.py`. They are timing-only and cannot be reached from `kernel_fn`.

**Race.**
- Rows: d1 (TINY), and d32/v16/v32 (MAIN).
- 40 reps per row, judge-style: shared static inputs refilled from `_gen_input` each rep, both arms captured as CUDA graphs, order alternated, a 256 MiB flush before each replay, CUDA events.
- Print: `GAPDIAG r50 bs= master_us= r50_us= ratio= [q25,q75]`.

**Decision rule.**
- ratio(d1) ≤ 1.03: C1 goes to phase A (M0), and byte cuts for M ≤ 16 are closed.
- ratio(d1) ≥ 1.10: drop C1 and reopen M ≤ 16 byte cuts.
- ratio(v32) ≥ 1.08: C3(a) is in.
- ratio(v32) ≤ 1.03: drop C3(a) and every WGS byte lever, including S6's reopen condition.

### DX1. Stamp timeline and in-kernel SM clock on the H200

**Builds.**
- `CUDA_SRC.replace("#define OKC_DEBUG 0", "#define OKC_DEBUG 1", 1)`, asserting exactly one hit. Use renamed extension names, the same way levmoe's `load_sub(dbg=True)` produced runs `012835`, `045221` and `050519`.
- Extend `DBG(k)` by one line to also store `clock64()` at `dbg[G*8 + blockIdx.x*8 + k]`.
- Compile TINY-debug and MAIN-debug.

**Runs.**
- bs 1, 32, 128 and 256, 12 reps each.
- Each rep: 2 warm-up calls, zero the stamp buffer, flush, one eager `okc_run(state["w"], scratch + [dbg], …)`.
- Per CTA, the clock is Δclock64 / Δglobaltimer from start to final.
- Print: `GAPDIAG tl bs= routed= joined= Adone=min/med/max Bdone=min/med/max barrier= final= clockMHz=`.

**Decision rule.**
- Let E = final − B_done (median).
- E(256) ≥ 10 us and E(128) ≥ 6 us: C2 goes to phase A. E(256) < 6 us: drop C2.
- Joined at bs 256 ≥ 6 us: C3(b) is in.
- Median in-kernel clock ≤ 1.80 GHz: the consumer is power-throttled. Weight C1 and any instruction diet up by 1.1x. At ≥ 1.95 GHz there is no throttling.
- alygg77 measured 1.69–1.73 GHz in this kernel (H200_MEASURED_FACTS.md:99).

### DX3. H200 size-matched read ceiling and per-SM map (no moe build; can ride in DX1 or DX2)

**Build.** One small load_inline module, about 20 s to compile:
- the LDG grid-stride and per-SM `rd_sm` kernels from `probes/gapmoe/gapmoe.py` (`BW_SRC`);
- plus master's own `BW_SRC` TMA modes 17 and 19.

**Runs.**
- Tensor views of 73 / 244 / 325 / 444 / 649 / 703 MB and 2 GiB, flushed, graph-replayed 20 times, minus the empty-graph overhead (2 × one minus two).
- smmap: 4 MB per SM, 8 reps.

**Decision rule.**
- Ceilings at 444 and 703 MB of at least 4.40 TB/s: the §1 floors and the 2.835 headline stand.
- Ceilings of at most 4.10 TB/s: recompute; the WGS gaps shrink by at least 7%, and C2 and C3 lose priority.
- A stable slow-SM set (Spearman ≥ 0.5 and p90/median ≥ 1.03): a per-SM rate-weighted static split goes on the lever list. Otherwise it is closed; on the H100 the Spearman was 0.02–0.20 and p90/median 1.007.

---

**Files and runs:**
- Probe package: `/home/user/git/.takemyjob-wt/gapmap-moe/prototypes/hopper-probe/probes/gapmoe/`
  - harness `gapmoe.py`, runner `run-gapmoe.sh`
  - variant files `sub_NW.py`, `sub_NF.py`, `sub_R50.py`, `sub_R33.py`, `sub_R67NW.py`, `sub_R67NF.py`
- Runs, under `/home/user/git/.takemyjob-wt/gapmap-moe/prototypes/hopper-probe/runs/`:
  - `20261031-122316-gpunode06` (bandwidth ceiling, nsys, ncu)
  - `20261031-123110-gpunode06` (NW ablation)
  - `20261031-123701-gpunode06` (smmap, NF ablation)
  - `20261031-124042-gpunode06` (draw stats, cache clean)
  - `20261031-125106-gpunode06` (R-scaling, clean)
  - `20261031-122014-gpunode06` (build; its first bandwidth attempt is invalid: the launch was not captured in the graph)
- Judge JSON and parsers: `/tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/okc/`, with `table.py`, `ncu.py`, `nsys.py` and `ab.py` in the parent directory.
