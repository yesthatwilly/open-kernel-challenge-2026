# muse_attention_decode gap map: master 0f134f6f, board 2.4729 (finder report, 2026-10-31 ~19:50Z)

**How to read this.** The headline speedup is the geomean over 6 loads of SGLang's time divided by ours. One load = 46 × the sliding row + 16 × the full row.
- **Elasticity** is how much the headline moves per µs saved on a row: e_i = 100·w_i / (6·T_load) %/µs.
- **Share of log-gap** is a row's first-order part of the headline gap: (w_i·t_i / T_load)/6 · ln(t_i / FLOOR_i).

**Judge reads of master code:**
- J1 = `14cec6846e404ea3`: the scored run of the exact master bytes (sha256 0f134f6f).
- J2 = `ac0335f9bc614f65` (PP-r1) and J3 = `d933848f272a429c` (PP-r2). Their bodies are byte-identical to master; only line 2 differs (campaign/public-plus `muse_attention_decode/attempts/2026-10-31T103704Z-submit-pp-softmax-turns.md:4`).
- The leaderboard comes from `okc leaderboard muse_attention_decode --json`, fetched at about 19:13Z.

**H100 runs.** All on gpunode06 GPU 6, through the hopper-probe guard. Package `gapatt`, branch hopper/gapmap-attention at 57553eb, worktree /home/user/git/.takemyjob-wt/gapmap-att. Run dirs are under `prototypes/hopper-probe/runs/`:

| Run | Directory | What it measured |
|---|---|---|
| R1 | 20261031-121900-gpunode06 | 12 rows cold and L2-resident (`gapatt-all/gapatt-time-l2.json`); ncu on 7 rows (`summary.txt`) |
| R2 | 20261031-122655-gpunode06 | nsys, 40 bins, copy flush |
| R3 | 20261031-123054-gpunode06 | latbench: cold DRAM trip latency and cold code fetch |
| R4 | 20261031-123449-gpunode06 | nsys, 40 bins, L2-resident |
| R5 | 20261031-123831-gpunode06 | nsys, 40 bins, read-only flush (judge-like) |

- Inputs used J1's own case seeds.
- The node run dirs were removed by unstage; I checked at 19:42Z.
- Nothing in musespark was edited. No okc check or submit was run.

## 1. Per-row map

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

**Time split by kernel.** Every row is one `attn_kernel` launch, which is 100% of the judge profile (KERNEL-COUNTS.md:13). "Yours" minus the kernel is 0.6–1.5 µs. That remainder is graph-replay cost the judge's 4.16 µs subtraction leaves in (J1 `replay_overhead`).

**Floor model.** FLOOR = max(HBM, chain, MUFU, tensor) + 0.8 µs of graph residue.
- **HBM floor.** Bytes = attended tokens × (256 B of FP8 K and V for both KV heads + 4 B page-table entry), plus qkv, out and the new rows, at 4.0 TB/s achieved. The judge's d32f mean is 4.04 TB/s and d16f is 3.96 (this table); the public ceiling is 3.8–4.1 (LEVERS-1031b:223).
- **Tensor floor.** QK and PV in E4M3 at 1979 TFLOP/s: at most 42 µs (v32f). Never binding.
- **MUFU floor.** One ex2 per score at 16/clk/SM and 1.98 GHz: v16f 39 µs, v32f 78 µs. Below HBM.
- **Chain floor** (per-request rows, graph mode): 1.0 µs empty kernel, plus 0.40 µs per dependent cold DRAM trip, plus at least 0.2 µs (L=1) or 0.4 µs (L=8) of compute, plus one merge round of 0.8 µs (L=1) or 1.2 µs (L=8), plus streaming time.
  - Trip counts: 2 at B=1 (the public SPEC build fetches seq and page indices in one trip, submission.py:2491-2496); 3 at B>1.
  - Measured on the H100 in R3: 0.379–0.396 µs per dependent cold trip, 0.99 µs empty kernel. The 0.8/1.2 µs merge rounds are estimates.

**Where the gap sits.** The share column sums to 28.3% = ln(3.27/2.48).

| Group | Rows | Share of the log-gap | What bounds it on the H200 |
|---|---|---|---|
| B=1 per-request | d1s, v1s, d1f, v1f | 14.9% | latency chain |
| B>1 per-request sliding | d16s, d32s, v16s, v32s | 7.0% | 3-trip chain plus the q8 epilogue |
| flat q8 | v16f, v32f | 6.3% | consumer plus tail, at 2.97 / 3.24 TB/s |
| flat q1 | d16f, d32f | 0.15% | HBM, at the ceiling |

- **78% of the gap is in the eight per-request (latency) rows.**
- For all eight, the H100 L2-resident kernel equals the H200 judge kernel within 0.98–1.045×. They are bound inside the SM and L2, not by DRAM, and the H100 L2-resident mode is a 1:1 proxy for them.

## 2. H100 graph-mode timeline (nsys, plus ncu with cache-control all; R1–R5)

**Graph level.** Each replay is one kernel node.
- The gap from flush end to kernel start is 1.3–2.45 µs median (minimum 1.15–1.44) (R1/R5 `nsysq.txt`).
- There are no other gaps, no serialized launches and no extra passes.
- Everything below is inside the kernel.

| row | judge kernel | H100 read flush (R5) | copy flush (R2) | L2-res (R4) | ncu DRAM read rate (R1) | issue active | XU (MUFU) | achieved / theoretical occupancy | top stalls (cycles per issue) | tail deficit: cold DRAM (R5) / L2-res issue (R4) |
|---|---:|---:|---:|---:|---|---:|---:|---|---|---|
| v32f | 205.2 | 254.5 | 261.9 | 186.0 | 2.58 TB/s, 77% | 48.8% | 35.0% | 36.7 / 37.5% | long_sb 5.41, barrier 1.63, wait 1.53 | 21.6 µs (8.5%) / 18.6 µs (10.0%) |
| v16f | 112.5 | 135.9 | 144.4 | 102.2 | 2.41 TB/s, 73% | 48.8% | 34.1% | 36.2 / 37.5% | long_sb 4.90, barrier 1.90, wait 1.52 | 17.4 µs (12.8%) / 12.7 µs (12.4%) |
| d32f | 164.9 | 242.2 | 250.8 | 117.6 | 2.75 TB/s, 82% | 29.7% | 5.3% | 25.0 / 25.0% | long_sb 9.57, wait 1.31 | 11.0 µs (4.6%) / 10.0 µs (8.5%) |
| d16f | 84.0 | 124.1 | 132.3 | 61.8 | 2.64 TB/s, 79% | 29.4% | 5.2% | 24.7 / 25.0% | long_sb 9.44, wait 1.32 | 9.6 µs (7.7%) / 7.7 µs (12.5%) |
| d1s | 4.6 | 5.02 | 5.18 | 4.51 | 0.09 TB/s | 10.5% | 1.7% | 6.3 / 18.75% | barrier 2.28, wait 1.37, short_sb 1.31, long_sb 1.17, no_instr 1.04, imc_miss 0.69 | n/a |
| v1s | 6.4 | 7.02 | 6.94 | 6.69 | 0.08 TB/s | 23.2% | 7.1% | 12.5 / 25% | barrier 2.43, wait 1.25, short_sb 1.22, no_instr 0.91 | n/a |

**Hardware limit vs structure, per heavy row (H200 judge kernels):**

- **d32f (164.9 µs) and d16f (84.0 µs): hardware-bound body.** They average 4.04 / 3.96 TB/s (84% / 83% of 4.8).
  - H100 body DRAM runs at 87.0–87.5% of peak (R5).
  - The structure is the tail. CTAs finish unevenly: SM-active min/max is 0.856 / 0.875 (R1 ncu). The merge plus coop finalize follows.
  - That is 4.6–7.7% on the H100, about 4–6 µs per row on the H200 if it scales with the kernel.
  - One wave (264 CTAs = 2 per SM), K/V read once, no extra launches.
- **v32f (205.2 µs): not HBM-bound** (3.24 TB/s).
  - The consumer proxy (L2-res) is 186.0 µs: a body at 71% issue plus an 18.6 µs tail. In the tail, issue falls 71% → 52% while warps in flight drop 48 → 40, then drain (R4).
  - The judge sits 19 µs (+10%) above the proxy. This is a memory-side interaction that is not yet measured; D1 measures it.
  - No hardware floor binds: MUFU 78, FP8 tensor 42, HBM 167 µs.
  - Structure: 29% idle issue in the body (long_scoreboard and barrier stalls), about 10% tail, exact-max bookkeeping (C5L −6.3% in the proxy, LOG:170-172, :252-253), and the +10% H200 excess.
- **v16f (112.5 µs):** the same picture. Proxy 102.2 µs = body + 12.7 µs tail (12.4%); the judge is +9.5% over the proxy; 2.97 TB/s.
- **d1s (4.6 µs): latency-bound.**
  - The chain floor is 2.8 µs, leaving about 1.8 µs of structure.
  - 32 of 132 SMs, 6% achieved occupancy; barrier stalls dominate.
- **v1s (6.4 µs):** the chain floor is 3.4 µs, leaving about 3.0 µs of structure. The epilogue is 19% of the kernel (round-2 time map, `research/levers-1031b.json`, attention `judge_time_map`).

**Method findings:**
1. **The copy flush is not judge-like.** `fdst.copy_(fsrc)`, used by levattn, atttax and our earlier probes, leaves about 50 MB of dirty L2.
   - The next kernel's first 30–45 µs then carries 18–24% DRAM write bandwidth (R2 "DRAM Write").
   - That inflates H100 cold flat-row times by 7.4–8.6 µs (copy vs read flush: v32f 261.9/254.5, v16f 144.4/135.9, d32f 250.8/242.2, d16f 132.3/124.1).
   - The judge reads its buffer, which leaves clean lines. A/B ratios are mostly unaffected, but absolute cold times are not judge-like.
   - Use a read sweep: `torch.sum(buf, 0, keepdim=True, out=)`. Suggested LESSONS line.
2. **Cold code fetch is not a lever.** 4, 16 and 64 KB of straight-line SASS cost 0.06, 0.11 and 0.06 µs cold vs hot. Only i-cache overflow costs anything: 128 KB → 6.3 µs (R3). The ncu `no_instruction` and `imc_miss` stalls at d1s are small in time.
3. **The judge regime per row class.** L2-res H100 ≈ H200 for the per-request rows (0.98–1.045×). For flat q8, the H200 is 1.09–1.10× the L2-res proxy. For flat q1, the H200 is HBM-bound.

## 3. Leaderboard: is 10th place reachable under the compliance ruling?

**Board** (~19:13Z):
- 1st: tjiang-fair 3.2221 (`7d1f783733cf49c6`).
- 10th: Lurrobert 2.9098 (`652c08c4dc2e4ab9`; screened 2.8355, pending).
- Ours: 2.4729, rank 296. The 3-read row-mean base is 2.4822.
- 10th needs ×1.1723 over that base (ln 0.159).

**The compliant floor headline is about 3.27** (all 12 rows at FLOOR; range 3.2–3.35 for trip latency 0.55–0.40 µs).
- The FA3 ruling does not raise the floors. The exact-max consumer fits under HBM if its issue efficiency reaches about 70% at C5L's instruction count: 54.3M warp instructions at v16f (d8948831 commit message).
- MUFU (39/78 µs) and FP8 tensor (21/42 µs) floors sit far below HBM (83/167 µs).
- **So 10th is not excluded by compliance.**

**What 2.91 takes:**
- **58% of every row's log-gap.** Closing 50% gives 2.85; closing 60% gives 2.93.
- Or all eight per-request rows 29.8% faster (27.1% faster if C5L lands at its L2-res ratio, which alone gives 2.522).
- **v16f/v32f alone cannot get there.** It would take −48%; their HBM floors allow only −24% / −18%, which tops out at 2.64.
- d16f/d32f contribute about 0. Per-request rows at floor give 3.05.

**Compliance tax context.**
- Today's measured tax vs the published row is 7.7% (LOG:168).
- The published (non-compliant) public rows on today's SGLang times give 2.634.
- Our rows with only the public's v16f/v32f give 2.562.
- So even with zero tax, 10th needs about +10.5% from per-request structure.

**Leaders.** The top (3.22) is at about 95% of the floor's log-gap. Flat rows cannot beat their HBM floors, so the leaders' per-request kernels must be within about 10% of the chain floors (or use precision shortcuts our ruling excludes, such as BF16 partials or no R11).

**Arithmetic verdict:**
- All three §4 candidates landing at stated values: S1 +3.2 + S2 +3.8 − 0.8 (overlap at v1s) + S3 +2.3 + C5L +1.6 ≈ **+10% → ~2.72**.
- p-weighted, about +3–4% → ~2.56.
- **10th is reachable in principle (floor 3.27), but not with the mapped mechanisms by Nov 4.** It needs near-floor rewrites of all eight per-request kernels (about −30% each).

## 4. Structural candidates (ranked)

**S3. Flat-row end of kernel: d16f, d32f, v16f, v32f (with C5L on the q8 body).**
- **What it replaces.** Today 264 CTAs take static equal key shares. Their finish times spread over 15–30 µs: warps in flight fall 48 → 15 (R5), and in R4 they step 48 → 40 then drain. The last-segment FP32 reds land in a burst (q8: 32 KB per segment), then a grid barrier and a sliced finalize. Replace with:
  - a per-request last-arriver finalize, reusing the per-request ticket logic at submission.py:2242-2290, so there is no grid barrier;
  - then a cost-weighted partition (segment fixed cost from stamps);
  - optionally, tile stealing for the last ≤ 2 tiles inside a request.
- **Gain if the tail halves**, scaled to judge kernels:

  | Row | Change | Elasticity | Headline |
  |---|---|---|---|
  | v16f | −7.0 µs | 0.113 | 0.79% |
  | v32f | −10.3 µs | 0.066 | 0.68% |
  | d16f | −3.2 µs | 0.151 | 0.49% |
  | d32f | −3.8 µs | 0.086 | 0.33% |
  | **Total** | | | **+2.3%** |

  C5L adds +1.6% on top (checks 94993678/262412e1 are in flight, LOG:258).
- **Transfer class:** scheduling (balance, barrier removal), the class that transferred today (PP, DPP). The H200 tail size is uncertain; D1's intercept measures it. Confidence medium.
- **Compliance:** the arithmetic is unchanged. FP32 merge order is already order-nondeterministic (LEVERS-1031b:51). Exact max, E4M3 P and the FP32 denominator are untouched. No caches.
- **Build:** about 150–300 lines.
- **Milestones, with fallback to master's plan per row:**
  - **M0 (≤ 1 h):** per-CTA entry, segment and exit globaltimer plus smid stamps, in L2-res and read-flush modes. GO if at least 60% of the tail is imbalance plus barrier/finalize.
  - **M1:** last-arriver finalize. GO at ≤ −3 µs on v16f in the L2-res A/B.
  - **M2:** cost-weighted partition.
- **Closest CLAIMS:** AT6 (CLAIMS:88, refuted), a boundary-charged partition on our old fvg kernel. S3 differs in two ways: it is driven by the measured finish-time spread, and it removes the grid barrier. A fixed charge alone is only M2.
- **p ≈ 0.4 → value × p ≈ 0.9%.**

**S2. B=1 latency path: d1s, v1s, d1f.**
- **The gap.** Judge kernels are 4.6 / 6.4 / 9.9 µs against chain floors of 2.8 / 3.4 / 7.4. The L2-res proxy reproduces the kernels (4.51 / 6.69 / 9.95), so the gap is SM/L2 structure, not DRAM.
- **What it replaces.** Build dedicated B=1 kernels whose critical path is: SPEC trip, then K/V trip, then compute, then ONE merge round. Specifically:
  - the gate, new rows and q-prep fully under the K/V trip;
  - red plus one acq_rel ticket, with the finalize distributed over the last arrivers (the AT5b pattern, CLAIMS:105);
  - fewer inter-phase barriers (barrier is the top stall: 2.28 / 2.43 cycles per issue, R1).
- **Gain:** −1.0 µs each on d1s, v1s, d1f gives 1.80 + 1.36 + 0.63 = **+3.8%**; −0.5 µs each gives +1.9%.
- **Transfer class:** latency. The proxy is accurate on exactly these rows, so prediction confidence is high.
- **Compliance:** no arithmetic change (fixed-reference FP32 merge plus R11; the SPEC fallback stays).
- **Build:** about 200–400 lines (new MODE bits plus routing).
- **Milestones, with fallback to master's builds per row:**
  - **M0:** phase stamps on the H100 proxy, plus D2 on the H200. Gate: at least 1.0 µs of non-chain time at d1s.
  - **M1:** cut the largest phase.
- **Closest CLAIMS:**
  - AT8F (CLAIMS:119; −1.3 µs entry-load hoist on our old line). The public kernel already does this (SPEC plus q issue order, submission.py:1316-1330), so S2 must target another phase.
  - AT2W4 (:118), AT3 and AT3c (:106/:110), and AT5c (:104) are refuted variants.
- **p ≈ 0.3 → value × p ≈ 1.1%.**

**S1. Per-request q8 split merge through DSMEM clusters: v16s, v32s; two-level at v1s, v1f.**
- **What it replaces.** Today each split pushes a 32 KB FP32 partial with `red.global`, then a ticket, a coop barrier, a finalize read from L2, and workspace zeroing. Replace with one cluster per request (v32s 4 CTAs, v16s 8; v1s 2×16, v1f 16×16). The FP32 partials are reduce-scattered on-chip and finalized in the cluster. Only the leaders' slices (16× fewer bytes) take the global R11 reds at v1s/v1f.
- **Gain if the epilogue halves.** The epilogue is 19% / 37–39% / 31–34% / 36–37% of v1s / v1f / v16s / v32s (round-2 time map; the proxy is accurate here):

  | Row | Change | Elasticity | Headline |
  |---|---|---|---|
  | v1s | −0.6 µs | 1.361 | 0.82% |
  | v1f | −2.7 µs | 0.473 | 1.30% |
  | v16s | −1.8 µs | 0.324 | 0.60% |
  | v32s | −2.7 µs | 0.190 | 0.52% |
  | **Total** | | | **+3.2%** |

  v16s and v32s alone give +1.1%.
- **Transfer class:** L2/latency, high predictive confidence. Execution risk is high.
- **Compliance:** FP32 partials and the FP32 merge are kept (BF16 partials stay out, LEVERS-1031b:226). R11 is kept on the global leg.
- **Build:** about 350–500 lines; the file's cluster launch path already exists.
- **Milestones, with fallback by per-row routing:**
  - **M0:** epilogue stamps. Gate: at least 50% of the epilogue is in the global-merge path.
  - **M1:** v32s cluster-4. GO at ≤ −1.5 µs on the L2-res proxy.
  - **M2:** v16s cluster-8.
  - **M3:** v1f and v1s two-level.
- **Closest CLAIMS:**
  - AT5c (CLAIMS:104, refuted: a cluster-of-2 pre-merge at v1f cost +1.1–1.5 µs). It ADDED a level in front of the reds; S1 deletes the global path at ≤ 8 splits.
  - PERHEAD (:48, replaced by AT2 :101) was L=1 with 4 KB partials, 8× smaller than q8's.
  - LESSONS:175: on the old base, cluster C=4 won −17.6% at d32s, and C=8 lost +7% at d16s.
- **p ≈ 0.3 → value × p ≈ 1.0%.**

All three port to decode's attention, which runs the same kernels at about 10× lower elasticity (`levers-1031b.json`, "Decode ATT7").

## 5. H200 diagnostic checks for tonight's spare checks

Attention has no `prepare()` (rules/technical-rules.llms.txt:98), so "prepare-only" here means import time. Each design:
- appends a `_gapdiag()` block to master's bytes and calls it at module level under `if torch.cuda.is_available():` with try/except;
- leaves kernel_fn and CUDA_SRC byte-identical to master;
- builds synthetic inputs in-file (no reference import; page 1; page tables from randperm; seq lengths ±7.5%; keys with a shared direction, c2 0.55);
- times its own CUDA graphs with events, each replay preceded by a read-only 256 MiB sweep, net of a null graph;
- frees everything (del, empty_cache);
- prints compact `GAPDIAG ...` lines (under 2 KB) to the log, readable via `okc status <job> --json` → `log_tail`;
- adds about 40–80 s at import, inside the 15-minute run.

There are no timing patches and no subprocess. kernel_fn calls on synthetic data leave `_WS` zeroed (the kernel resets it), as the tests do. Each design takes one check. Suggested: D1 and D3 in one file, D2 in a second. A builder needs under 45 minutes each, reusing `gapatt.py`'s `l2remap` and graph timing and `latbench.py` from branch hopper/gapmap-attention.

**D1. FQ8-REGIME: is flat q8 consumer-bound on the H200, and how big is the flat-row fixed cost?**
- **Runs:** master's kernel_fn on v16f, v32f, d16f and d32f shapes at ctx 80k, cold and L2-resident (page tables remapped onto 65,536 rows, no sweep). Also cold at ctx 40k and 120k for (16,8) and (16,1), to fit time = a + b·bytes.
- **Prints:** `GAPDIAG fq8 <row> ctx=<c> cold= l2res= null= MB=`.
- **Rule (pre-registered).** Let r = cold/l2res at v16f and v32f.
  - **r ≤ 1.04 on both:** the rows are consumer-bound. Consumer cuts transfer 1:1; S3 and C5L go to phase A at full weight.
  - **r ≥ 1.08 on either:** at least 8% of the time is memory-side. S3 M1 becomes the producer and prefetch depth, and consumer-lever predictions are scaled by l2res/cold.
  - **Intercept a ≥ 8 µs (v16-shape):** the tail is worth it, and S3's tail half goes to phase A. **a ≤ 4 µs:** drop the tail half.
  - **Control:** d16f/d32f should show r ≥ 1.3 (HBM-bound).

**D2. LATCAL: what are the latency floors of the per-request rows on the H200?**
- **Runs:**
  - latbench chain with n = 0, 1, 2, 3, 4, 8 dependent cold loads in a 512 MiB cycle;
  - code kernels of 4, 16 and 64 KB SASS, cold vs hot;
  - master on synthetic d1s, v1s, d1f and v1f (B=1, so SPEC hits), timed cold right after the sweep, and timed again as the second of two back-to-back graphs on disjoint inputs (code and constants warm, data cold).
- **Prints:** `GAPDIAG lat trip= empty= code16k= code64k=` and `GAPDIAG b1 <row> cold= warmcode=`.
- **Rule:**
  - **trip ≥ 0.55 µs:** rerun the §1 floors and scale S2's value by (4.6 − floor_new)/1.8.
  - **(cold − warmcode) ≥ 0.5 µs at d1s/v1s:** hot-path compaction becomes S2's first build. **≤ 0.2 µs:** S2 targets the merge chain only (the H100 prior from R3 is about 0.1 µs).
  - **d1s(cold) − (empty + 2·trip) ≥ 2.0 µs:** S2 goes to phase A.

**D3. GATHER: is ~4.0 TB/s the H200's random 128-B row ceiling?**
- **Runs:** two 704 MB pools (K and V) plus an int32 row table, moving at least 650 MB per variant. Variants:
  - LDG.128, 8 threads per row, U = 4, 8, 16 rows in flight;
  - a cp.async.cg 16 B ring of 4, 6 or 8 stages of 64 rows (master's producer shape);
  - K and V of a token issued adjacently vs as separate streams.
  
  The control is master's d32f and d16f kernel_fn on synthetic inputs. The H100 version of this test (R3, 2.15–2.22 TB/s) is not valid: too small, and it used the copy flush.
- **Prints:** `GAPDIAG gather <variant> TB/s=` and `GAPDIAG ctrl d32f us= TB/s=`.
- **Rule:**
  - **Best variant ≥ 1.08 × control:** a flat-q1 gather rewrite gets a phase-A slot, worth about +1.0% at −8% on d16f/d32f.
  - **≤ 1.03:** close the flat q1 line and fix the v16f/v32f HBM floor at the measured rate.

## Artifacts
- **Worktree / branch:** /home/user/git/.takemyjob-wt/gapmap-att, branch hopper/gapmap-attention, commits c6333f4 … 57553eb.
- **Probe:** prototypes/hopper-probe/probes/gapatt/ (`gapatt.py`, `latbench.py`, `nsysq.py`, `ncusum2.py`, `run-gapatt.sh`). Run outputs are in /home/user/git/.takemyjob-wt/gapmap-att/prototypes/hopper-probe/runs/20261031-12{1900,2655,3054,3449,3831}-gpunode06/ (the runs/ dirs are gitignored; only INDEX.tsv is committed).
- **Judge JSON and leaderboard:** /tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/judge/
- **Calculators:** /tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/tools/ (`jrows.py`, `floor3.py`, `scen2.py`, `table1.py`)
