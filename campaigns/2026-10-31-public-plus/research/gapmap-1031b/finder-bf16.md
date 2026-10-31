# muse_bf16_proj gap map, wave 2 (master blob b8c06565, 2026-10-31 19:52–20:35Z)

**Bottom line.** No bf16 change is ready for a phase-A build tonight. Every row runs exactly one kernel, so there is no scheduling or launch lever. Only one big structural gap is left: the m=128 rows (the v16 load) still stream raw BF16 weights. Master uses its lossless 12-bit code only at m ≤ 32. Building a 12-bit GEMM for m=128 (candidate A) is a hard kernel: a naive version lost 13–16% to nvjet on the H100. I recommend spending 4 bf16 checks on the three diagnostics in §5 and the other ~12 on H200 bandwidth diagnostics for the other problems.

**What I did and did not touch.** I made no edits in musespark, ran no okc check or submit, wrote no attempt files and no memory. The probe package is on takemyjob branch `hopper/gapmap-bf16`, commit b666749, worktree `/home/user/git/.takemyjob-wt/gapmap-bf16`, at `prototypes/hopper-probe/probes/gmbf/{gmbf.py,p12.py,run-gmbf.sh}`. All probes ran on gpunode06 GPU 1. Results are pulled back under `.../runs/`. hopper-probe removed each node run dir, and the gmbf-clean runs removed the node cache (`removed /tmp/USER-hopper/gmbf-cache`, runs 20261031-130927, -132336 and -132458).

**Judge evidence: three reads of master's bytes.** The two check files differ from master only in the line-2 comment; I confirmed that with diff.
- 109f577bcfaf4d8a (scored): 1.5657 [1.5422, 1.5892], `/home/user/git/musespark/muse_bf16_proj/profiles/submission.109f577bcfaf4d8a.txt`.
- a95479284a204d87: 1.5365, and 280a3622e2384023: 1.5442, both in `/home/user/git/musespark/.worktrees/public-plus/campaigns/2026-10-31-public-plus/files/bf16/profiles/B0-aa{1,2}.*.txt`.
- Rebuilding the headline from the per-row tables (geomean over 6 loads of ΣSGLang/Σyours, every weight 62) reproduces all three reads exactly. The level from 3-read mean row times is **1.5487**.

## 1. Per-row table

**Floor model** (H200, graph mode, in the judge's net metric): T = t0 + max(B/BW, FLOP/(0.85 × 989.4 TF/s)).
- **Fit:** t0 = 2.33 µs and BW = 4.57 TB/s, fitted on the judge's own four m=1 cgemv rows (12-bit; 6.3/12.6/28.3/50.4 MB → 3.55/5.17/8.74/13.25 µs).
- **H100 check:** the intercept matches the H100 pure-read fit, 2.18 µs + B/3.13 TB/s (lvbf micro, `/home/user/git/.takemyjob-wt/lev-bf16/prototypes/hopper-probe/runs/20261031-005619-gpunode06`).
- **Brief's model:** with 4.0 TB/s + 1.3 µs instead, the floors near 50 MB move by less than 0.6 µs.

**Byte counts.** B = weights + x + output.
- **F_raw:** BF16 at 2 B per weight; router at 4 B per weight, FP32 or hi+lo. The judge's own `sol` uses these bytes at 4.8 TB/s.
- **F_12:** master's 12-bit code, 1.5 B per weight (`/home/user/git/musespark/muse_bf16_proj/submission.py:42-55`, decode8 at :213). The escape share I computed for these N(0, 1/K) weights is 0.7–1.1e-4.
- **F_ent:** the order-0 entropy of the BF16 weights, 10.46 bits per weight (sign 1 + exponent 2.54 + mantissa 6.92); router about 26.5 bits. An 11-bit fixed code would escape 1.2–2.5% of weights, too many to be practical.

**How to read the columns.**
- `*` means the incumbent already beats F_raw because it streams the 12-bit code.
- The gap is measured against min(F_12, incumbent).
- "Share" is that row's part of the total log gap to the all-rows F_12 headline. It is first-order, so the column sums to about 92%.
- Scripts: `/tmp/claude-1000/-home-user-git-musespark/47d181e3-6ab1-4b7a-bc8b-48599bc90f20/scratchpad/an/{parse,floors,scen,table2,entropy}.py`.
- The H100 net column comes from run 20261031-130201-gpunode06 (§2).

| row | judge net µs: mean (109f/a954/280a) | SGLang | judge sol | H200 kernel (eager µs ×3) | H100 net | F_raw | F_12 | F_ent | gap to F_12 | headline if alone | share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d1 qkv_gate | 8.74 (8.72/8.66/8.84) | 14.11 | 7.87 | cgemv<2,512,8192,2> 7.1/7.0/7.0 | 11.38 | 10.61* | 8.54 | 7.75 | 0.20 (1.02x) | +0.10% | 0.8% |
| d1 o | 5.17 (5.24/5.00/5.28) | 6.91 | 3.50 | cgemv<2,64,1024> 4.1/4.0/4.0 | 6.29 | 6.01* | 5.09 | 4.74 | 0.08 (1.02x) | +0.04% | 0.3% |
| d1 pooler | 13.25 (13.31/13.21/13.24) | 21.23 | 13.99 | cgemv<2,512,8192> 12.8/13.2/12.9 | 18.46 | 17.04* | 13.36 | 11.95 | 0 | 0 | 0 |
| d1 unpooler | 3.55 (3.52/3.47/3.65) | 6.99 | 1.75 | cgemv<2,128,4096,4> 2.2/2.3/2.2 | 4.48 | 4.17* | 3.71 | 3.54 | 0 | 0 | 0 |
| d1 router | 3.65 (3.60/3.66/3.68) | 10.35 | 1.75 | rgemv<1,1> 2.5/2.4/2.5 | 4.88 | 4.18* | 4.18 | 3.86 | 0 | 0 | 0 |
| d16 qkv_gate | 9.73 (9.45/9.82/9.93) | 14.12 | 7.93 | sk2<1,16,3,…> 8.4/8.7/8.5 | 12.66 | 10.68* | 8.61 | 7.82 | 1.12 (1.13x) | +0.40% | 3.4% |
| d16 o | 5.67 (5.59/5.81/5.61) | 6.80 | 3.56 | sk2<1,16,2,…> 4.0/4.4/4.4 | 6.91 | 6.07* | 5.15 | 4.80 | 0.52 (1.10x) | +0.18% | 1.6% |
| d16 pooler | 13.88 (14.02/13.77/13.84) | 21.47 | 14.06 | dd_cap2_b_cpssm_gemv 13.6/13.1/13.1 | 18.69 | 17.12* | 13.45 | 12.03 | 0.43 (1.03x) | +0.15% | 1.3% |
| d16 unpooler | 13.78 (13.80/13.77/13.77) | 18.53 | 14.06 | sk2<1,16,2,2,2,…> 14.2/12.8/13.8 | 18.77 | 17.12* | 13.45 | 12.03 | 0.33 (1.02x) | +0.12% | 1.0% |
| d16 router | 4.58 (4.44/4.69/4.62) | 14.91 | 1.81 | rr_small<2,16,8> 4.2/4.1/4.2 | 5.57 | 4.23 | 4.23 | 3.92 | 0.35 (1.08x) | +0.12% | 1.1% |
| d32 qkv_gate | 10.74 (10.67/10.80/10.74) | 14.25 | 8.00 | sk2<2,32,3,…> 9.9/9.9/9.5 | 12.83 | 10.75* | 8.68 | 7.89 | **2.05 (1.24x)** | +0.68% | 5.9% |
| d32 o | 5.97 (5.61/6.27/6.04) | 6.87 | 3.62 | sk2<2,32,2,…> 4.8/4.8/5.1 | 7.02 | 6.14* | 5.22 | 4.87 | 0.75 (1.14x) | +0.25% | 2.1% |
| d32 pooler | 15.01 (14.81/15.24/14.98) | 21.89 | 14.14 | sk2<2,32,…,5> 15.2/15.2/14.5 | 19.31 | 17.21* | 13.53 | 12.12 | 1.48 (1.11x) | +0.49% | 4.2% |
| d32 unpooler | 14.35 (14.25/14.62/14.18) | 18.55 | 14.14 | sk2<2,32,…,3> 14.4/14.2/13.9 | 19.07 | 17.21* | 13.53 | 12.12 | 0.82 (1.06x) | +0.27% | 2.3% |
| d32 router | 5.36 (5.23/5.63/5.21) | 15.33 | 1.86 | sk2<2,32,1,1,8,…> 4.7/4.7/4.7 | 6.24 | 4.29 | 4.29 | 3.98 | 1.06 (1.25x) | +0.35% | 3.0% |
| v1 qkv_gate | 9.07 (8.82/9.19/9.19) | 14.08 | 7.90 | sk2<1,8,3,…> 7.4/7.3/7.5 | 12.46 | 10.64* | 8.57 | 7.78 | 0.49 (1.06x) | +0.23% | 2.0% |
| v1 o | 5.24 (5.15/5.38/5.20) | 6.79 | 3.53 | v18s_gemv 4.7/4.8/4.8 | 6.62 | 6.04* | 5.12 | 4.77 | 0.12 (1.02x) | +0.06% | 0.5% |
| v1 pooler | 13.70 (13.79/13.56/13.74) | 21.40 | 14.02 | smallm2_gemv 13.4/13.4/13.6 | 18.69 | 17.08* | 13.40 | 11.99 | 0.29 (1.02x) | +0.14% | 1.2% |
| v1 unpooler | 4.06 (3.95/4.22/4.02) | 6.92 | 1.76 | sk2<1,8,1,…> 3.4/3.3/3.3 | 4.86 | 4.19* | 3.73 | 3.55 | 0.33 (1.09x) | +0.15% | 1.3% |
| v1 router | 4.30 (3.92/4.66/4.31) | 26.33 | 1.78 | rr_small<1,16,8> 3.5/3.6/3.6 | 5.34 | 4.20 | 4.20 | 3.89 | 0.09 (1.02x) | +0.04% | 0.4% |
| **v16 qkv_gate** | 13.82 (13.44/14.11/13.90) | 16.89 | 8.42 | qb_kernel<64,36,128,…> 13.9/13.1/13.3 | 16.74 | 11.19 | 9.12 | 8.33 | **4.69 (1.51x)** | +1.23% | **10.5%** |
| v16 o | 6.87 (6.88/6.58/7.15) | 7.40 | 3.99 | ol_projection<128,128,…> 6.5/6.3/6.5 | 8.27 | 6.53 | 5.61 | 5.25 | 1.26 (1.23x) | +0.32% | 2.8% |
| **v16 pooler** | 20.49 (20.24/21.16/20.06) | 25.27 | 14.64 | nvjet 64x64_64x13 coopB_NNN 21.7/19.5/20.4 | 25.84 | 17.72 | 14.05 | 12.64 | **6.44 (1.46x)** | +1.71% | **14.7%** |
| **v16 unpooler** | 19.12 (19.27/18.99/19.11) | 19.79 | 14.64 | nvjet 128x64_64x8_1x2 NNT 18.8/18.1/18.8 | 24.91 | 17.72 | 14.05 | 12.64 | **5.08 (1.36x)** | +1.33% | **11.4%** |
| v16 router | 6.09 (6.29/5.75/6.22) | 16.32 | 2.21 | rk_kernel 4.6/4.6/4.6 | 6.93 | 4.66 | 4.66 | 4.34 | 1.43 (1.31x) | +0.36% | 3.1% |
| **v32 qkv_gate** | 17.85 (18.24/17.61/17.70) | 20.04 | 9.77 | nvjet 72x64_64x12 TNN 20.4/17.3/19.8 | 19.62 | 13.82 | 13.82 | 13.82 | **4.03 (1.29x)** | +0.84% | 7.3% |
| v32 o | 8.97 (9.05/8.82/9.04) | 9.37 | 4.48 | rl_early_projection<256,64,…> 8.8/9.1/9.0 | 9.38 | 7.44 | 7.44 | 7.44 | 1.53 (1.21x) | +0.32% | 2.7% |
| v32 pooler | 24.56 (24.59/24.66/24.43) | 30.73 | 17.37 | nvjet 128x64_64x8 NNN 24.6/24.2/24.1 | 27.95 | 22.76 | 22.76 | 22.76 | 1.80 (1.08x) | +0.37% | 3.2% |
| v32 unpooler | 23.30 (23.23/23.37/23.30) | 24.33 | 17.37 | nvjet 128x128_64x6 coopA 23.9/23.4/23.0 | 25.86 | 22.76 | 22.76 | 22.76 | 0.54 (1.02x) | +0.11% | 0.9% |
| v32 router | 7.09 (7.00/7.09/7.19) | 17.55 | 2.68 | rk_kernel 6.0/5.9/5.9 | 8.13 | 5.15 | 5.15 | 4.89 | 1.95 (1.38x) | +0.40% | 3.5% |

**Loads, 3-read mean µs (F_12 in brackets):** d1 34.36 (34.08), d16 47.64 (44.89), d32 51.43 (45.26), v1 36.37 (35.03), v16 66.38 (47.49), v32 81.77 (71.94).

**Headline at the floors** (SGLang times held fixed): 1.5487 now, 1.6273 at F_raw (+5.1%), 1.7392 at F_12 (+12.3%), 1.8790 at F_ent (+21.3%). At F_12 by class: m ≤ 32 rows +3.96%, m=128 rows +5.74%, m=256 rows +2.16%.

**What the table says:**
- **Most rows are near the floor.** 17 of 30 rows are within 1.0 µs of F_12, and every d1 and v1 row is within 0.5 µs. This matches LESSONS.md:47: m ≤ 32 streams cap out, and the headroom is at m=128/256.
- **v16 holds 42.5% of the gap** (pooler 14.7%, unpooler 11.4%, qkv 10.5%, router 3.1%, o 2.8%). v32 holds 17.6% (mostly qkv at 7.3%), and d32 holds 17.5%.
- **Every m=128/256 route streams raw BF16.** These are nvjet via the cuBLASLt plans (:4016/:4040), `qb_kernel` on `state["raw"]["qkv_gate"]` (:4334), `ol_projection`, `rl_early_projection` and `rk`. ncu read bytes confirm it: v16 pooler 69.2 MB, v16 qkv 40.1 MB, v32 o 17.3 MB.
- **v16 pooler and unpooler are bandwidth-bound** at 87–92% of the raw H200 ceiling (F_raw 17.72), with 25% of the bytes compressible.
- **v32 is compute-bound.** nvjet runs the pooler at about 72% of 989 TF on the H200, but v32 qkv runs at only about 56%.

## 2. H100 graph-mode timeline

**Setup.** Run 20261031-130201-gpunode06 on GPU 1 with no other compute app before, during or after, SM clock 1830 MHz, no throttle. nsys traced graph nodes over 30 judge-shaped replays per row: a 256 MiB read flush before each replay, net = raw − (2·one − two) = raw − 3.87 µs. ncu profiled 12 kernels with caches flushed before each kernel and clocks not locked.

**Every replay of all 30 rows has the same shape:**

```
flush end → 3.87–4.03 µs gap → ONE kernel node → +0.37–0.73 µs → end event
```

- The 3.9–4.0 µs gap is the same for an empty graph (3.97 µs). It is the judge's subtracted fixed cost.
- There is one node per replay. There are no PDL edges, no memset or zero-fill nodes, no second kernel and no bubbles.
- **That one kernel is the whole critical path. Nothing is hidden or overlapped**, so there is no launch or overlap lever (KERNEL-COUNTS.md:15 agrees).
- Net time ≈ kernel time + 0.4–0.7 µs. For example, v16 pooler is 25.84 net = 25.26 kernel + 0.58.

**The heaviest-gap rows.** ncu durations are inflated by kernel replay, so the rates below use graph-mode kernel times.

| row | kernel (grid × block, regs) | H100 graph kernel µs med (min–max) | H100 net | H200 net / eager | DRAM read | achieved | tensor pipe % | L2 traffic | issue % |
|---|---|---|---|---|---|---|---|---|---|
| v16 pooler | nvjet 64x64 coopB (1×128 × 384, 168) | 25.26 (24.93–26.24) | 25.84 | 20.49 / 19.5–21.7 | 69.2 MB | 2.74 TB/s, 91% of the H100 67 MB ceiling (23.62 µs, lvbf micro) | 31.5 | 169 MB | 26.1 |
| v16 unpooler | nvjet 128x64 1x2 (2×64 × 384, 168) | 24.35 (24.13–25.79) | 24.91 | 19.12 / 18.1–18.8 | 68.2 MB | 2.80 TB/s | 31.8 | 140 MB | 11.4 |
| v16 qkv | qb_kernel (108 × 256, 90 regs; 24 SMs idle) | 16.24 (16.03–17.57) | 16.74 | 13.82 / 13.1–13.9 | 40.1 MB | 2.47 TB/s; warps active 8.3% | 26.3 | 115 MB | 14.3 |
| v32 qkv | nvjet 72x64 (1×128 × 384, 168) | 19.09 (18.34–20.48) | 19.62 | 17.85 / 17.3–20.4 | 42.0 MB | 506 TF/s (51% of peak) | 47.7 | 118 MB, 2.8× DRAM (x re-read by 32 N-tiles) | 20.7 |

**Other kernels (ncu):**
- v32 pooler and unpooler: nvjet at 59.9% and 61.7% tensor.
- v32 o: rl_early at 42.7% tensor.
- v32 router: rk at 24.4% tensor.
- d32 qkv: sk2 at 13.5% tensor, 13.3% warps active.
- d1 pooler: cgemv reads 50.5 MB at 68.8% warps active.

**H200 time ÷ H100 time per row:**
- d1, d16 and v1 rows: 0.72–0.84. A pure byte model predicts 0.73 at 50 MB.
- d32 qkv, o and router: 0.84–0.86, the worst among m ≤ 32. Their excess is issue and latency, which faster HBM does not shrink.
- v16: 0.77–0.88. v32: 0.87–0.96 (compute-co-limited).

**Candidate-A feed proxy** (`p12.py`; clean run 20261031-132458-gpunode06, GPU 1 alone).
- **What it is:** not a real GEMM. 12-bit planes are decoded with master's decode8 into register-sourced wgmma fragments (m64n128k16). x arrives through a 4-stage bulk-copy ring at the m=128 tile shapes. There is no epilogue, split-K reduce or escape handling.
- **Method:** each arm is raced judge-style against master's kernel_fn, 30 interleaved rounds, cold L2.
- **Cells** show µs, with the ratio to master's kernel in brackets.

| arm | pooler | unpooler | qkv_gate |
|---|---|---|---|
| master kernel_fn | 26.06 | 25.25 | 17.49 |
| 12-bit weights + x ring (stream only) | 21.06 (0.81) | 20.90 (0.83) | 13.42 (0.77) |
| raw BF16 weights + x ring | 25.63 (0.98) | 25.20 (1.00) | 15.81 (0.90) |
| 12-bit + decode8 | 24.50 (0.94) | 24.22 (0.96) | 15.14 (0.87) |
| x ring + wgmma only (no weights) | 14.94 (0.57) | 14.85 (0.59) | 9.70 (0.55) |
| everything (12-bit + decode + wgmma) | 29.55 (1.13) | 29.23 (1.16) | 18.24 (1.04) |

What the proxy shows:
- **The byte cut turns into time when streaming.** 12-bit vs raw stream is 0.82, against a byte ratio of 0.757.
- **decode8 is exposed** at 8 consumer warps per SM with a 2-step register prefetch: +3.4 µs cold, +18% warm (12.22 vs 10.34 µs). cgemv hides the same decode at 4.57 TB/s on the H200, but with 1152–2048 CTAs of 544 threads. So candidate A needs cgemv-level memory parallelism, not more ALU.
- **The compiler serialized every wgmma.** The SASS shows 8 HGMMA and 8 `WARPGROUP.DEPBAR.LE gsb0,0x0` per loop body. Even with no weight loads, the skeleton reaches only 518 TF/s warm (52% of peak). The "everything" row is therefore a pessimistic bound, not a verdict on the idea.

**Note on the earlier proxy runs.** Run 20261031-131443 had invalid stream arms because the checksum was dead code. Runs -132148 and -132203 were two copies running at once on GPU 1 because of a launch mistake on my part. Their cold arms agree with the clean run within 3%.

## 3. Is 10th place reachable under our compliance ruling?

The positions are from the brief (10-31, ~18:40Z); I did not re-read the leaderboard. 10th is 1.602, which is +3.44% over our 3-read mean of 1.5487 and +2.32% over the 1.5657 board read. Top is 1.652, +6.67% over the mean.

**The ruling costs bf16 nothing.** Master is the public file as adopted (CLAIMS.md:91), and lossless 12-bit with exact escape corrections is allowed. The compliant floor headline is 1.739 at F_12 and 1.879 at F_ent, both far above 10th.

**Small levers cannot get there.** Today's tuning rounds found at most 0.5%, and BFR and BRK were killed (CLAIMS.md:92-93; LOG.md:196-199). The m ≤ 32 headroom (+3.96%) is spread over 20 rows, and the largest single row gap is 2.05 µs.

**Paths, by scenario** (`scen.py`):

| scenario | v16 pooler/unpooler, qkv targets | headline (mean level) | change |
|---|---|---|---|
| A optimistic (stream-bound, decode and MMA hidden; H100 proxy ratios 0.81/0.83/0.77) | 15.4/15.4, 10.6 µs | 1.6011 | +3.39% |
| A mid | 16.5/16.5, 11.5 µs | 1.5864 | +2.44% |
| A with decode exposed | 17.8/17.8, 12.3 µs | 1.5713 | +1.46% |
| A mid + B (qkv v32 → 14.5 µs) | | 1.5975 | +3.15% |
| A mid + B + C at 50% | | 1.6141 | +4.22% |

**Verdict.** 10th is reachable only if A works at mid efficiency or better (p ≈ 0.4) together with B. Even then, one scored read has roughly even odds of reaching 1.602, given our reads run −0.8%/+1.1% around the mean. Top place is not credible.

## 4. Structural candidates

**A. Lossless 12-bit weights for the m=128 (v16) GEMMs** — top by value × p (about 1.0%).

*What changes*
- **Replaces** nvjet (v16 pooler and unpooler) and qb_kernel (v16 qkv), optionally ol_projection (v16 o).
- **Layout:** 12-bit planes repacked in prepare into fragment order per 128-row tile and k64 step.
- **Pipeline:** a TMA weight ring, or a 3–4 deep prefetch, to match cgemv's memory parallelism. Dedicated decode warps feed BF16 fragments to an MMA warpgroup issuing wgmma that the compiler does not serialize. x arrives through a 2-D TMA box of 128 tokens × 64 k.
- **Split and reduce:** 128 CTAs (pooler 32 tiles × 4-way K split, unpooler 64 × 2, qkv 18 × 7). Partial sums are reduced in FP32, pushed across the cluster in a fixed order (the BF4d/BF4e pattern). Escapes are applied exactly in rank 0's epilogue, about 73 per tile.

*Impact and risk*
- **Gain:** about +2.4% for mid efficiency (range +1.5 to +3.4%); pooler and unpooler alone give +2.2% at 70% of the floor gap.
- **Transfer class:** bandwidth, which has not transferred well before (LOG.md:113). Those earlier cuts were 5% of bytes or less or L2 policy; this one is 25% on kernels that are bandwidth-bound on both GPUs.
- **What the H100 does and does not predict:** it predicts the stream ratio (0.82 measured). It under-tests issue: on the H200 the tensor pipe must run at about 76% of peak inside an 11.4 µs stream, versus about 54% on the H100.
- **Compliance:** the same lossless scheme master already ships at m ≤ 32, FP32 accumulation, fixed-order reduce, BF16 output, graph-capturable, no caching.
- **Build size:** about 600–900 lines of CUDA plus 150 of Python, adding 35–45 KB (file goes from 382.8 KB to about 425 KB). Compile adds 35–40 s; H100 prepare is 146–183 s today (runs 20261031-005756, -130201, -132458).

*Staged plan, keeping the incumbent as fallback in every cell:*
1. D2 tonight decides go or no-go.
2. On the H100, fix the wgmma serialization (wgmma alone ≥ 75% of peak) and hide the decode (decode run ≤ 1.05× the stream run).
3. A correct pooler v16 kernel: bit-exact against an FP32 reference, probe rows exact, passes the 9 judge tests, and runs ≤ 0.85× nvjet on the H100.
4. Add unpooler and qkv.
5. Judge check with an in-prepare race that routes a cell only where the H200 race shows ≤ 0.95.

*Closest CLAIMS:* BF2 (CLAIMS.md:78, refuted at m=9..32 because decode issue sat on the critical path), BF2-DIAG and BF2R (:89-90), and the raw Triton m=128 split-K kernel that lost (LESSONS.md:55). What differs:
- m=128 has 4–16× more MMA work per weight byte than BF2's m ≤ 32.
- It uses the public decode8 instead of BF2's consumer.
- It moves 25% fewer bytes than nvjet.

The proxy reproduces BF2's specific risk, so step 2 must solve it before any check.

**B. qkv_gate at m=256 (v32) with a decomposition sized for N=2304.**
- **Design:** 18 tiles of 128 rows × 256 tokens × 7-way K split = 126 CTAs in one wave. Operands come from shared memory (m64n256k16), x is multicast within each cluster, and partials are reduced in-cluster in FP32. This replaces nvjet 72x64, which runs at 56% tensor on the H200 with L2 traffic 2.8× DRAM. Weights stay raw: this row is compute-bound, so 12-bit buys nothing.
- **Gain:** 17.85 → about 14.5 µs, **+0.70%**. That is below the margin on its own, so it should ride with A.
- **Transfer:** compute/scheduling class, which transfers conservatively. Precedent: BF5's CUTLASS pingpong went from 0.945–0.96 on the H100 to 0.905–0.909 on the judge (CLAIMS.md:79).
- **Compliance:** clean.
- **Build size:** about 300 lines, or re-instantiate qb at 256 tokens, or port BF5 from tag ours-final-1031; adds 10–15 KB.
- **Plan:** D3, then an H100 race at ≤ 0.85× nvjet, then a judge race. Fallback is nvjet.
- **Closest CLAIMS:** DeepGEMM and the K-split batched cuBLAS approach were refuted (:58, :45). This differs by per-shape tiling and an in-cluster reduce.

**C. d32 rows on register-direct wgmma (m64n32k16 fed by decode8) instead of mma.sync in sk2.**
- **Evidence:** on the H200, sk2 at m=32 costs 1.76 µs more than cgemv at m=1 for the same bytes (pooler 15.01 vs 13.25 µs) and 2.0 µs more for qkv. The d32 H200/H100 ratios are the worst among m ≤ 32 (0.84–0.86).
- **Caveat:** the public "skinny" route already uses this wgmma form for 12-bit at m ≥ 9 (`USE_WGM`, submission.py:1761), and it loses to sk2 in every m=8..32 cell. C must differ by having no shared-memory weight round trip and no wgmma serialization.
- **Gain:** **+1.04%** at 50% of the gap; p ≈ 0.25.
- **Transfer:** issue class, which the H100 under-predicts.
- **Build size:** about 300 lines.
- **Closest CLAIMS:** :73 (shelf) and :88 (VT, refuted). This changes the MMA instruction class, not launch values.

## 5. H200 diagnostic checks for tonight

Common rules for all three:
- `kernel_fn` stays byte-identical to master; the diagnostics live in prepare only.
- They print tagged lines and free every buffer; nothing goes into `state`.
- No harness patches; prepare must stay under 300 s.
- The judge's profile pass is a single eager call (LEVERS-1031b.md:65), so each diagnostic must graph-time its own numbers.

**D1 — H200 stream ceiling, launch floor and anchors** (builder about 30 min).
- **Build:** port `lvbf.py`'s `step_micro` and its tools (copied at `probes/gmbf/lvbf.py`; tested on the H100 in run 20261031-005619) into prepare, after master's prepare runs.
- **What it times:** pure reads at 6.29/8.39/12.58/16.78/28.31/37.75/50.33/67.11 MB, three launch configs each; empty-kernel times; and anchors from kernel_fn at (pooler,1), (pooler,32), (qkv,32), (unpooler,16), (pooler,128).
- **Output tags:** `GMBW`, `GMLAUNCH`, `GMANCHOR`. Cost: about +35 s compile and 5 s run.
- **Decision rules:**
  - Fit t0 and BW. If anchor(pooler,1) ≤ 1.05 × t(50.33 MB), cgemv is at the ceiling.
  - Then if anchor(m=32) − t(same bytes) < 1.0 µs for pooler and qkv, **kill C**. If it is ≥ 1.5 µs, C gets an H100 prototype.
  - In every case, (t0, BW_H200) becomes the transfer model for decode and moe bandwidth levers. A lever whose gain does not survive rescaling from (2.18 µs, 3.13 TB/s) gets no check.

**D2 — Candidate-A feed proxy on the H200** (builder about 30 min).
- **Build:** copy `p12.py` (commit b666749) into prepare, about +34 s compile.
- **What it times:** race the kernel_fn m=128 rows for pooler, unpooler and qkv against four proxy arms: 12-bit stream only, raw stream, 12-bit + decode8, and x ring + wgmma only.
- **Output tag:** `GMP12`.
- **Ratios:** S = 12-bit stream ÷ master's kernel; D = (12-bit + decode8) ÷ 12-bit stream. On the H100, S was 0.81/0.83/0.77 and D was 1.16/1.16/1.13.
- **Decision rules:**
  - If S ≤ 0.80 for pooler and unpooler, A's ceiling holds: proceed to step 2 of A's plan.
  - If additionally D ≤ 1.10 on the H200, the decode is already hidden: **A goes to phase A tomorrow**.
  - If S ≥ 0.90, **kill A** and record it as a bandwidth non-transfer.

**D3 — H200 dense-GEMM ceiling at the v32 shapes** (builder about 20 min, torch only).
- **What it times:** torch.mm at 8192³ for sustained dense TF/s; the v32 shapes; qkv at m=256 as 2-, 4- and 8-way K-split batched matmuls (diagnostic only); kernel_fn v32 anchors.
- **Output tag:** `GMTF`.
- **Decision rules:**
  - Floor for v32 qkv = 2.33 + 9.66e9 / TF_sus. If the anchor exceeds that floor by ≥ 2.5 µs, **B goes to phase A**; otherwise drop B.
  - If the v32 pooler and unpooler anchors are within 1.0 µs of 2.33 + 17.18e9 / TF_sus, close those two rows.
  - If any K-split variant beats the incumbent by ≥ 10%, that confirms B's decomposition on the judge.

**Spending.** Combine D1 and D3 in one prepare (about 220 s) and read it twice: 2 checks. Read D2 twice: 2 checks. That is 4 bf16 checks. **The remaining ~12 are better spent on H200 bandwidth diagnostics for the other problems**: D1's harness with their NVFP4 GEMV and gle_pr streams, specified by those gap maps. They are not worth spending on bf16 tuning or plan races, given the BFR/BRK/DL2H/S6 record.

## Caveats

- **Eager kernel times run warm on some rows.** The judge's eager profile sometimes reads at or below the graph net time (d1 pooler eager 12.8–13.2 µs vs net 13.21–13.31).
- **Floors are assumptions.** They assume 85% of tensor peak and fit the H200 stream model on only four rows.
- **The proxy omits parts of a real kernel.** It has no epilogue, reduce or escapes, and its wgmma path is serialized.
- **The leaderboard numbers come from the brief**; I did not re-read them.
