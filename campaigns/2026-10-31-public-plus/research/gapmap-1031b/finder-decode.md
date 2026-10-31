# Decode gap map (muse_decode_layer, master = public-C D0 + DPP, sha256 2a0d1906…). Returned 20:3xZ, before the 22:00Z timebox

**The DC5L question is answered.** In graph mode the attention kernel is fully on the critical path at v16f and v32f, so any saving it makes reaches the row. On the judge's paired-speedup column, v32f did move by the expected amount (+0.93%, predicted +0.86%). The KILL was set by rows DC5L does not touch: d1s and v1s read 2.6–3.2% low in both DC5L jobs. DC5L's predicted headline effect is +0.11%, against a ±1.5% interval, so a two-read headline gate could never have detected it.

**10th place (2.375) is reachable under our compliance ruling, but only through structural rewrites, not levers.** The compliant floor is 2.93. The three decode-side candidates at their central estimates stack to 2.40, and to 2.53 with the moe fix that decode shares with the moe problem. Weighted by probability, the outlook is about 2.33, with roughly a 0.3 chance of reaching 2.375.

**How this was done:**
- The musespark repo was not modified. No okc check or submit was run, and no memory was written.
- H100 work ran on gpunode06 GPU 5 only, under hopper-probe and its guard. The runner removed my node run dirs (verified).
- The probe package is **uncommitted** at `/home/user/git/.takemyjob-wt/gapmap-dec/prototypes/hopper-probe/probes/gapdec/`, on branch `hopper/gapmap-decode`.
- H100 results are in `/home/user/git/.takemyjob-wt/gapmap-dec/prototypes/hopper-probe/runs/20261031-130215-gpunode06/out/` (prof, ab, ncu) and `…/runs/20261031-132306-gpunode06/out/gapdec-skel/` (skeleton stand-in). A first skel attempt, `…/20261031-132134-gpunode06`, failed to build on a C++17 flag; the rerun used C++20.
- Analysis scripts are in `/tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/ana/` (rows.py, prof.py, floor2.py, roles15.py, scen.py).

Citation keys used below:
- **P** = `/home/user/git/musespark/.worktrees/public-plus/campaigns/2026-10-31-public-plus/files/decode/profiles/` (judge `.txt` and `.json` files)
- **SUB** = master `muse_decode_layer/submission.py`
- **REF** = `muse_decode_layer/reference.py`
- **LOG**, **KC** = `campaigns/2026-10-31-public-plus/{LOG,KERNEL-COUNTS}.md`
- **CL** = `muse_decode_layer/hypotheses/CLAIMS.md`
- **PUBF**, **PUBN** = `/home/user/git/okc-muse-spark-kernels/docs/{H200_MEASURED_FACTS,NEGATIVE_RESULTS}.md`
- **H** = the H100 runs dir above

## 1. Per-row gap table

**Judge time** is "yours", the mean of three judge reads of master code: eb4e07ad (DPP-r1), f078d673 (DPP-r2) and 9d726d84 (DPP-score, which is master's bytes). r1 and r2 differ from master only in the line-2 nonce comment. The three reads are listed in that order in brackets.

**Kernel split** is the judge's eager profile: the mean of 5 profiles (3 DPP plus 2 DC5L, whose code is identical outside attention). Attention uses the DPP profiles only (P/*.txt, parsed by prof.py). The moe time in a single profile call swings with that call's routing (v1s ranges 45–79 µs), which is why it is averaged.

**Floor** (floor2.py, run with 1.5 bytes per weight). It is the sum over phases of max(bytes / 4.0 TB/s, FLOPs / peak), plus a graph chain term.
- Dense weights use the lossless 12-bit code master already ships (`bfp_COMPRESS`, SUB:8639; `bfp_pack_compressed`, SUB:9066). That is 1.5 B per weight for qkv, o, pooler and unpooler. The router stays at 4 B per weight as BF16 hi+lo.
- Experts: 28.31 MB each, times the mean distinct count backed out of the judge's sol (d1 2.50, v1 7.5, d16 7.53, d32 9.0, v16 18.43, v32 21.33; REF:681-697).
- KV: 260 B per token (FP8 K+V plus an int32 page index), at the mean length. The agentic spread has mean = ctx (REF:153-178).
- Activations: T × 98,304 B.
- Compute: BF16 at 989 TF/s, FP8 at 1979 TF/s, MUFU at 16 per clock per SM. Compute only binds for the dense GEMMs at T=256.
- Chain term: 3 µs at T≤32 and 4 µs at T≥128. For reference, the judge's own replay overhead fields in all 9 job JSONs show about 5.0 µs for a one-kernel graph and about 6.1 µs for a two-kernel graph, so each extra serial kernel costs about 0.9–1.2 µs; the subtracted overhead is 3.8–4.5 µs.

**Share** is each row's slice of the headline gap: the load's log-gap ln(Y/F)/6, split in proportion to w·(y−f). The shares sum to 28.2%. The floor headline is **2.934**; with uncompressed BF16 dense weights it would be 2.710.

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

**Value per microsecond.** One µs at d1s is worth 14.8 times one µs at v32f, so the sliding rows dominate.

**Gap by component** (eager kernel time minus floor, times %/µs; roles15.py and comp15.py):

| row class | dense | moe | attention | glue + route | total |
|---|---|---|---|---|---|
| T≤8 | 5.4% | 2.6% | 2.5% | 2.1% | 12.6% |
| T=16/32 | 3.4% | 3.4% | 1.0% | 1.7% | 9.5% |
| T=128/256 | 4.0% | 3.4% | 1.7% | 0.8% | 9.9% |
| **all rows** | **12.7%** | **9.4%** | **5.2%** | **4.7%** | 32% |

The total exceeds 28.2% because eager kernel durations include PDL waits.

**Achieved H200 rates** on actual bytes (12-bit weights):
- sk2 dense: qkv 2.46–2.75, o 1.83–2.03, pool+router 2.75–3.17, full unpooler 3.03–3.27, unpooler slice 1.34 TB/s.
- moe: 2.7–3.8 TB/s, depending on the routing sample.
- Full attention: 3.98 TB/s at d32f, 3.04 at v16f, 3.33 at v32f.
- The public team's H200 facts agree: d1s 73.5 µs against a 44.6 µs byte floor (PUBF:73); WGS moe is "boundary-bound, not rate-bound", with quiet stretches at 4.39–4.64 TB/s (PUBF:99-100).

## 2. H100 graph-mode timeline

**Method.**
- nsys with `--cuda-graph-trace=node` over cold-L2 CUDA-graph replays (256 MiB flush before each), 16 reps × 2 input states, master and DC5L, all 12 rows. Reduced with nj_profile (H/…130215…/out/gapdec-prof/report.txt).
- ncu per kernel: eager, cache flushed per kernel, `--clock-control none` (gapdec-ncu/raw.csv.gz).
- Whole-layer A/B, 40 rounds (gapdec-ab/log.txt).
- Router seed fitted to the judge's distinct-expert targets. Distinct experts per state 0: d1 3, d16 8/5, d32 6/9, v1 9/8, v16 19/21, v32 22/20 (gapdec-prof/log.txt:5-16).

**Every row runs on one stream, and every kernel is on the critical path.** Path presence is 100%, "blocked by off-path" is 0.00 and cross-stream overlap is 0.00 for all 24 arm-rows (report.txt:81-313).
- **Hidden:** only the kernel prologues that PDL hides. At T≤32 the chain has no launch gaps, and the early-launched next kernels wait 11.0–15.2 µs per row.
- **Not hidden at T=128/256:** the cuBLAS nvjet kernels are not PDL-launched, so there are 0.78–1.60 µs of launch gaps per row and at most 0.5 µs of PDL wait.

**d1s** (span 96.45 µs, report.txt:197). Columns: cp / wait µs; ncu MB, TB/s, DRAM %, tensor %.

| # | kernel | cp | wait | ncu: MB, TB/s, DRAM, tensor |
|---|---|---|---|---|
| 1 | qkv sk2 (96 CTAs) | 15.47 | 0 | 28.4 MB, 1.98 TB/s, 59%, 6.5% |
| 2 | attn<1,0,13> (32 CTAs) | 5.76 | 0.67 | 0.58 MB, 2.4% DRAM (latency-bound) |
| 3 | o sk2_gle | 4.95 | 3.15 | 12.8 MB, 1.37, 41% |
| 4 | pool+router sk2_gle | 25.88 | 0.80 | 58.9 MB, 2.09, 62% |
| 5 | _gle_route | 1.64 | 2.37 | — |
| 6 | moe_fused<0> (TINY) | 37.76 | 1.33 | 85.0 MB, 2.17, 65% |
| 7 | unpool sk2 | 3.38 | 1.95 | 6.3 MB, 0.90, 27% |
| 8 | _sl_cluster_finish | 1.62 | 0.77 | — |

**v1s** (164.1 µs, report.txt:276) and **d16s** (187.6 µs, :172) have the same structure:
- qkv 15.0 / 15.5 µs at about 2.0 TB/s.
- attn 9.1 / 11.0 µs.
- pool+router 24.8 / 23.7 µs at 2.15 / 2.33 TB/s.
- moe 99.2 / 103.2 µs at 2.61 / 2.54 TB/s, 78% / 76% DRAM.
- The glue and boundary kernels take 3.9–4.3 µs each on 16–32 CTAs (33% MUFU in the boundary).

**v32s** (390.2 µs, :302): nvjet qkv 27.5 → attn 16.5 → nvjet o 9.8 → boundary 7.8 → nvjet pool 38.5 → graft route 3.5 → moe<1> 246.3 → nvjet unpool 31.0 → boundary 8.0.
- The nvjet grids are 128 CTAs, tensor-pipe use is 39–62% and DRAM 44–73%. That is one partial wave, saturating neither tensor nor bandwidth.
- moe reads 657 MB at 2.65 TB/s (79% DRAM).

**v16s** (324.2 µs, :250): same structure. The qkv nvjet grid is only 96 CTAs (tensor 25.6%); moe takes 199.3 µs.

**v16f and v32f: what is on the critical path** (spans 448.0 and 621.0 µs, :237 and :289):
- Attention takes cp 140.2 µs (31% of the span) and 265.3 µs (43%).
- It is preceded by the nvjet qkv GEMM (gap 0.00) and followed by the nvjet o GEMM (gap 0.39 and 0.31 µs). There is no PDL on either edge, and attention's wait is 0.00 / 0.10 µs.
- ncu: v16f attention is 71% DRAM-active, 35% MUFU, about 36% warps active, 80 registers, 264 CTAs.

**So nothing in the graph absorbs an attention saving.** The evidence:
- **H100, master vs DC5L:** the layer A/B gives v16f 0.9990 [0.9917..1.0060] and v32f 0.9974 [0.9861..1.0067]. The A/A arm (master vs master) reads 1.0008 / 1.0017 (gapdec-ab/log.txt:7,9). That is a layer change of −0.33 / −1.55 µs, against a cold kernel-alone saving of −0.63 / −0.60 µs (`/home/user/git/.takemyjob-wt/dc5l-aud/prototypes/hopper-probe/runs/20261031-121440-gpunode06/out/dc5laud-kcold.txt`). The H100 is blind to the size of the saving, as already known, but not to whether it reaches the row.
- **Judge, pooled paired ratios** (the metric the headline uses; per-row 60-ratio sets from P/*.json):
  - v16f: DPP 2.0686 [2.0482, 2.0817] vs DC5L 2.0759 [2.0591, 2.0875], **+0.35%**.
  - v32f: DPP 2.1972 [2.1805, 2.2078] vs DC5L 2.2177 [2.2086, 2.2311], **+0.93%**, predicted +0.86% from the 4.2 µs kernel saving.
  - Raw medians moved −2.7 µs on both rows.
- **Why the µs column looked flat:** "yours" is SGLang's median divided by the paired ratio, so it inherits SGLang drift (SGLang's own v32f time ran 1052–1095 µs across the 5 jobs).
- **What actually sank the headline:** unchanged code read low in the DC5L jobs, d1s −3.2% and v1s −2.6% (2.4585 vs 2.5396; 2.4311 vs 2.4960).
- **Size of the effect:** DC5L's predicted headline gain is +0.11%, so the gate at 2.2000 / 2.1887 (LOG:270-272) was decided by noise. CL:117 can be updated from this.

**Synthetic stand-in for d1s** (same bytes as the real chain, latency phases modeled as dependent DRAM hops; H/20261031-132306-gpunode06/out/gapdec-skel/log.txt:4-21). Public data says stand-ins over-predict real gains by about 6.6x (PUBN:32), so treat these as bounds.

| variant | µs |
|---|---|
| one uninterrupted 176.7 MB stream | 67.65 |
| 5 stream phases, persistent kernel with counter hand-offs | 72.59 (about 1.2 µs per hand-off) |
| same 5 phases as a PDL kernel chain | 85.25 (about 3.2 µs per boundary) |
| full 8 phases, PDL chain | 83.92 |
| full 8 phases, persistent kernel | 79.97 (−3.95 µs) |
| full 8 phases, no PDL | 87.44 |
| null graph replay | 5.44 |

- The three latency phases cost +7.4 µs serialized (79.97 vs 72.59).
- A one-batch register run-ahead before each hand-off was *worse*, by +1.3–2.0 µs (lines 9 and 11).

## 3. Leaderboard: can we reach 10th place under our compliance ruling?

- **Distance:** 10th is 2.375. That is ×1.0689 over the board (2.2220) and ×1.0729 over the three-read mean (2.2136).
- **Compliant floor:** 2.934 (2.710 with uncompressed BF16 dense weights). 10th therefore needs about **24–25% of the log-gap** to the floor.
- **What the ruling costs decode:** about 1%, and it is not the constraint.
  - The public raw decode was published at 2.2058 [2.173–2.239] (okc-muse-spark-kernels README.md:33).
  - Our compliant D0 read 2.1708, 2.2060 and 2.1793 (mean 2.1854).
  - DPP's 2.2220 already exceeds the public raw.
- **Realistic ceilings** (scen.py):

  | scenario | headline |
  |---|---|
  | every stream at 3.6 TB/s, 12-bit dense, 6/8 µs chain | 2.588 |
  | 3.4 TB/s, 8/10 µs chain | 2.417 (about where the top of the board sits) |

- **Candidates stacked** (exact headline formula):

  | stack | headline | odds |
  |---|---|---|
  | S1 + S2 + S3, central | 2.404 | |
  | plus the shared moe unit-end fix | 2.526 | |
  | probability-weighted (p = 0.35, 0.4, 0.3, 0.4) | 2.33 | P(≥ 2.375) ≈ 0.32 |

- **Verdict:** reachable only by landing at least two of the structural items. Levers will not get there: today's lever rounds topped out at ≤ 2%.

## 4. Top structural candidates

Gains below use the exact headline formula on the three-read mean (scen.py).

**S1: one persistent kernel for the T≤8 rows (d1s, d1f, v1s, v1f).**
- **What it replaces:** the 8–9 PDL kernels (qkv, attn, o, b1, pool+router, route, moe TINY/TINY2, unpool slice, finish).
- **Design:** one launch of 132 CTAs, each with an SMEM ring fed by a single TMA producer that streams weights in layer order across phase boundaries. Phases hand off through per-phase epoch counters (publish+acquire about 0.9 µs, PUBF:51), not grid barriers. Routing is computed redundantly per CTA, as FAST1 does.
- **Rows and gain:**

  | case | per-row saving | headline |
  |---|---|---|
  | counters only (stand-in −3.95 µs) | −5 µs | **+2.05%** |
  | central | −8 µs | **+3.36%** |
  | attention, route and finish hidden under the weight stream (stand-in shows 7.4 µs available) | −14 µs | **+6.14%** |

- **Transfer:** the launch and hand-off part is the scheduling/latency class, which transferred 3–4x amplified (LOG:113). The stream-rate part is the bandwidth class, which does not transfer (LOG:52, 79-82), so it needs D2.
- **Risks:**
  - Verify attention slows 50–80% next to a co-running GEMM (PUBF:77).
  - The public team's post-attention megakernel "never won" (SUB:10846-10848).
  - The arch-mirror skeleton cost 17–19 µs empty (KC:32).
- **Compliance:** carry over the FP32 router input at T≤8 (PUBF §5), ATT7's FA3 math verbatim, and `fence.proxy.async` before every ring slot hand-back (the latent race, CL:116). Epoch counters carry no activations.
- **Size:** about +40–70 KB, against 158,639 B free (master is 841,361 B).
- **Milestones**, each raced per T in prepare with the incumbent chain as fallback:
  - M0: empty skeleton passes the D3 gate.
  - M1: FFN tail (route, moe, unpool, finish) in one kernel, target −5 µs per row.
  - M2: attention block (qkv, attn, o, b1).
  - M3: join, with a TMA-ring run-ahead.
- **Closest claims, and what differs:**
  - CL:66 W6-A glue folds: those were last-arriver single-CTA tails; S1 distributes the work.
  - CL:67 trigger-first PDL: S1 uses no PDL triggers.
  - PUBN:29-37: the public team says "Prefetch inside one persistent kernel that owns the L2 traffic was not measured."

**S2: persistent FFN-tail kernel for T=16/32 (d16s, d16f, d32s, d32f).**
- **What it replaces:** pool+router → route → moe → unpool → b2.
- **Design:** the router runs first; top-k is distributed into the pooler's tail; moe consumes routing from published counters (no early PDL triggers); the unpooler stream runs as a phase inside the moe kernel during the phase-B end shadows; b2 becomes a distributed final pass.
- **What it targets:** the moe ramp and end (public loss budget: ramp 3–6, end 7–11 µs, PUBF:99), two glue launches, the route kernel and the unpooler ramp.
- **Gain:** central −9 / −11 µs per row gives **+2.08%** (range 1.12–3.40%).
- **Transfer:** scheduling class.
- **Compliance:** the ET3 race class (CL:115-116) means it must sit on D0FIX and be counter-gated.
- **Size:** +25–40 KB.
- **Milestones:** router-first plus distributed route; then moe consuming counters plus the unpooler phase; then the WGS unit-end release decoupling.
- **Shared with the moe problem:** a "unit end is consumer work" fix (PUBF:101) gives decode **+4.69%** at the row savings assumed in scen.py. The moe wave-1 map owns it; port it by copy.
- **Closest claims:** CL:40 (L2 prefetch of o/pooler) and PUBN:18-27 (side streams). S2 differs because it is single-kernel concurrency, which PUBN:27 says was not covered.

**S3: dense GEMMs and glue at T≥128 (v16s, v16f, v32s, v32f).**
- **What it replaces:** 4 nvjet kernels, 2 `_graft_row_boundary` kernels and `_graft_route`.
- **Design:** a persistent 132-SM stream-K 12-bit RS-wgmma GEMM (bfp already has RS-wgmma for m≥9, SUB:8651), with b1 and b2 folded into the o and unpooler epilogues via per-row sum-of-squares atomics plus a counter.
- **Why there is room:** nvjet runs on 96–128 CTAs, tensor 22–62%, DRAM 44–76% (ncu). The public team's cuBLASLt sweep found at most 1.3 µs per GEMM (PUBF:82), which is why this is a custom kernel rather than a plan pick.
- **Gain:** central −20 / −25 µs per row gives **+2.53%** (range 0.98–3.92%).
- **Transfer:** tensor-pipe and grid fill transfer moderately. The 12-bit byte cut is the bandwidth class.
- **Compliance:** BF16 stays lossless (exact escape check, SUB:9066-9100). The accumulation order changes, which needs a rel-L2 check.
- **Size:** +30–50 KB.
- **Milestones:** boundary folds with nvjet kept; then the 12-bit pool/unpool GEMM raced against nvjet in prepare; then qkv and o.
- **Closest claims:** CL:97 (bulk12 shelf, T=8..32) and the bf16 BFR kill (LOG:81-82). S3 is not a cuBLAS plan pick.

## 5. H200 diagnostic checks for tonight

All three share the same shape:
- **Insertion:** append `_gd_parent = prepare; def prepare(w): st = _gd_parent(w); _gapdiag(st); return st` after SUB:13886. `kernel_fn` (SUB:13033) stays byte-identical, and the diagnostic never mutates `st`.
- **Timing:** use the file's own `_graph_time` (SUB:10649, read-sweep flush), so no harness is touched.
- **Output:** print `[gapdiag-Dn] …` lines, which come back in `log_tail`.
- **Guard:** skip the diagnostic if `time.monotonic()` shows more than 150 s since prepare began. Master's cold prepare took about 115 s on the H100.
- **Cleanup:** `del` everything and call `torch.cuda.empty_cache()`. Public data shows that layout shifts alone move d1 and v1 by 0.5–1.3 µs (PUBF §3).
- **The check's own rows are not evidence for anything.**

**D1: DC5L reach on the H200** (about 40 min to build). Decides whether DC5LF gets scored.
- **Setup:** compile the DC5L ATT7 source (the mk_decode_c5l.py edits applied as string replaces) as a second module and launch it through `_launch7(…, _mod=…)` (SUB:10960). Build synthetic v16f/v32f inputs the way `_attn_selftest` does (SUB:11062).
- **Measure:** 64 alternating ABBA pairs, cold, of (a) the kernel alone and (b) the nvjet qkv → attention → nvjet o chain.
- **Rule:**

  | outcome | action |
  |---|---|
  | kernel Δ ≤ −2.0 µs on both rows and chain Δ / kernel Δ ≥ 0.7 | DC5LF gets 2 checks after D0FIX lands, gated on the pooled v16f/v32f paired ratio (≥ +0.6%, interval excluding 0), not on the headline |
  | kernel Δ > −1.0 µs | close DC5L as refuted |
  | kernel Δ ≤ −2.0 µs but chain Δ / kernel Δ < 0.4 | freeze attention-only decode levers until an H200 stamp study |

**D2: H200 stream-rate ladder** (about 45 min). Sizes S1 and S2 on the H200, which the H100 cannot do.
- **Setup:** time cold standalone graphs of the incumbent `_bfp_gemm` qkv, o, pool+router and unpool at m = 1, 8, 16, 32; `moe_fused` at T = 1, 8, 16, 32 with synthetic routing forced to n = 1, 2, 3, 4, 6, 8, 10, 12 distinct experts (`_calib_bias`, SUB:10675), fitted as t = a + b·n; a size-matched TMA read probe from 6.3 to 600 MB; and the `_after_attention` chain at T = 1, 8, 16, 32, as `_tune` already does (SUB:10715).
- **Rule for S1:** if the non-rate time in the T=1 chain (the sum of kernel time minus bytes/ceiling, plus the latency kernels) is ≥ 12 µs, S1 goes to phase A; below 6 µs, S1 is closed.
- **Rule for S2:** if the moe fit at T=16/32 gives a ≥ 8 µs with b ≥ 4.0 TB/s, S2 goes to phase A. If b ≤ 3.4 TB/s, the expert consumer rate is the limiter, so the work goes to the moe consumer instead.

**D3: H200 skeleton cost** (about 40 min). This is S1's M0 gate.
- **Setup:** port gapskel.py (`/home/user/git/.takemyjob-wt/gapmap-dec/prototypes/hopper-probe/probes/gapdec/gapskel.py`) to load_inline. Time an empty 8-kernel PDL chain with the T=1 grids (96/32/256/128/10/132/128/32) against a persistent 132-CTA kernel with P = 1, 3, 8 epoch-counter hand-offs, against one grid barrier, against the null graph.
- **Rule:**

  | outcome | action |
  |---|---|
  | persist(8) − null ≤ 3.0 µs and ≤ 0.5 × chain(8) | S1 M0 passes |
  | otherwise, persist(3) − null ≤ 2.0 µs | build M1 only |
  | neither | close S1 |

D2 and D3 can share one check; D1 should be its own check. That uses 2 of the roughly 11 decode checks.
