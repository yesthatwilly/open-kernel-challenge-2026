<!-- Research report, copied verbatim from a read-only Explore subagent (Opus) run in the planning session.
     Task: NVFP4 expert-GEMM forensics and engine design. Finished 2026-10-26T03:56:26.476Z. Saved by campaign 2026-10-26-decode-first-rewrite. Do not edit; add notes to ../FINDINGS.md. -->

# NVFP4 (W4A16) expert-GEMM forensics and a new engine design for muse_moe_w4a16

## 1. Summary

**Problem facts**
- H = I = 4096. Per expert: w13 is [8192, 4096] and w2 is [4096, 4096]. One expert is 25.17 MB of e2m1 codes plus 3.15 MB of e4m3 scales, **28.31 MB** (0.5625 B/element).
- The floors in `reference.speed_of_light_ns` use 0.75 × router_distinct. Backing that out gives the mean number of touched local experts per call:

| shape | T | mean touched experts | bytes per call |
|---|---|---|---|
| decode_1 | 1 | 2.6 | 73 MB |
| verify_1 | 8 | 8.4 | 238 MB |
| decode_16 | 16 | 11.9 | 337 MB |
| decode_32 | 32 | 16.0 | 453 MB |
| verify_16 | 128 | 22.9 | 648 MB |
| verify_32 | 256 | 24.6 | 696 MB |

- Judge-like routings are skewed. At T=128 the hottest expert has 44–65 routes. At T=256 one expert gets all 256 tokens (`judge-rebuild (draw 4)`: touched 23, max 256, 453 local routes).
- Routing changes every repetition. Board decode_16 runs span p10 128 µs to p90 190 µs.

**Every CUDA NVFP4 kernel in the repo uses the same dequant**
- It is master's `dequant8` plus a `cvt`-based scale path: wg/wg2, prefill rega, fsi, rowowner, swap-AB and fused bs1 all carry it.
- Cost: about 26–31 instructions per lane per k16 (8 elements). That is ~3.5 issue slots per element, of which ~1.7–2.2 are INT-ALU ops and 0.375 are conversions.
- At 4 TB/s that needs **~72–93% of the SM's INT32 ALU pipe for dequant alone**. So master's sequence cannot reach 4 TB/s however well it is scheduled; its practical ceiling is about 3.2–3.6 TB/s.

**Measured rates**

| kernel family | weight-stream rate |
|---|---|
| wg family (s5, 1 CTA/SM) | 1.8–2.2 TB/s |
| wg2 (2 CTAs/SM) | 2.1–2.5 TB/s |
| fsi | 1.6–1.9 TB/s |
| rowowner (mma.sync) | ≈ Marlin (gate 0.88–0.97× of master) |
| swap-AB, fused bs1, Triton | below Marlin |
| Marlin (judge profiles, estimated) | ~2.0–2.5 TB/s |
| bf16 streams with a trivial consumer (contrast) | 3.4–3.6 TB/s |

**Why the wg consumer is slow.** The evidence localises it to a serialized per-stage critical path, about 1150–1200 clk per 9216 B stage per SM:
- deeper rings were flat;
- two CTAs per SM gave +15–17%;
- forcing the two consumer warpgroups to alternate (prefill ping-pong) was 1.18–1.53× slower;
- moving the scale expansion into the producer was 1.03–1.13× slower;
- the overlap fix (double-buffered A with `wgmma.wait_group<1>`, "variant B") was only ever measured in a grid-starved one-expert canary. That result does not count.

**Measurement confounders** — they bias every prepare-side TB/s for short kernels:
- Every moe gate and timer flushes with `flush.zero_()` on 256 MB. That writes, leaving ~50 MB of dirty L2 lines that the measured kernel must write back. The judge reads its flush buffer instead.
- Many "kernel-only" numbers are eager single launches, which include launch and host time (`wg_gemm` also encodes 4 tensor maps on the host per call).
- Example: the bf16 pure read printed 2.54 TB/s in prepare, but the judge profile shows 64 MB in 19.0 µs = 3.37 TB/s.

**Proposal**
1. A cheaper exact dequant: integer-only scale construction, plus a code bit layout where 3 of 4 pairs cost one shift and one LOP3. This takes INT-ALU per element from ~1.9–2.2 down to ~1.1–1.25, with zero conversions. Exactness is proved in §3.
2. Engine A (T ≤ 32 routes per expert):
   - a CTA-shared 18 KB 1-D bulk ring, 2–3 CTAs/SM, ≥128 KB in flight per SM;
   - mma.sync swap-AB, with each warp owning 16 rows over the full K;
   - stream-K balancing with a no-spin ticket fix-up;
   - PDL chain w13 → w2 (w2 weights prefetched before `griddepcontrol.wait`) → combine.
3. Engine B (T = 128/256):
   - wgmma register-A with double-buffered A and `wait_group<1>`;
   - BK = 64 stages so N ≤ 64 fits 104 registers at 2 CTAs/SM;
   - a 1-CTA/SM N = 128/256 variant that reads hot experts once;
   - hot items first, and a stream-K split of the last wave.

**Expected (estimates, judge, clean L2)**

| shape | current µs | expected µs |
|---|---|---|
| decode_1 | 46.4 | 26–29 |
| verify_1 | 104.5 | 73–84 |
| decode_16 | 150.9 | 101–117 |
| decode_32 | 199.1 | 134–156 |
| verify_16 | 323.2 | 215–245 |
| verify_32 | 358.1 | 235–265 |

That takes the moe headline from 1.40 to roughly 1.8–2.1. In the decode KPI the Marlin pair rows shrink by the same proportion (decode_1 42.5 µs → ~22–25 µs).

**First action:** the one-run microbenchmark ladder in §6.

---

## 2. Forensics

### 2a. The shared dequant, per lane per k16 step (8 weight elements)

Source: `/home/user/git/musespark/muse_moe_w4a16/submission.py:1057-1082` (helpers) and `:1123-1140` (`dq_stage`). The same code is copied verbatim in the wg2, fsi, rowowner (`sc4`/`deq<I>`), swap-AB and fused-bs1 branches, and in prefill `muse_prefill_layer/submission.py:2705-2722`.

**Codes:**
```
for i in 0..3: v = (q & 0x80008000) | ((q & 0x70007000) >> 6);   // LOP3, SHF.R, LOP3   -> 3 INT
               a[i] = fma.rn.bf16x2(v, s_row, -0);                 // HFMA2.BF16          -> 1 FMA
               q <<= 4;                                            // SHL/IMAD.SHL (x3)
```
That is 12 INT + 3 shifts + 4 HFMA2.

**Scales**, two rows (g, g+8) per k16:
```
cvt.rn.f16x2.e4m3x2 (1 per k16, amortised)   // conversion pipe
__half22float2 -> 2x HADD2.F32
2x FMUL (* 2^119)
2x cvt.rn.bf16x2.f32 (bf16x2_bcast)          // F2FP, conversion pipe
```
That is 7 instructions, 3 of them conversions.

**Loads:** LDS.128 for codes (0.25 per k16) and LDS.64 for scales (0.125–0.25 per k16).

**MMA issue and address math:** wgmma plus `make_desc`, about 3–4 INT per k16; or mma.sync plus a B-fragment LDS/ldmatrix.

**Total:** ≈ 26–31 instructions per 8 elements, i.e. 3.3–3.9 per element. INT ≈ 13.5–17.5 per 8 elements (1.7–2.2 per element). Conversions 0.375 per element. The block scale is applied per element by the bf16x2 FMA. The global scale (× 2^7 bias) is applied once in the fp32 epilogue (`acc * gs[e]`, `submission.py:1181-1201`).

### 2b. Per-kernel structure

| kernel (source) | load path; weight bytes in flight per SM | smem staging | MMA form; wait | occupancy | grid fill at scored shapes | reductions / spins | NMAX and re-reads |
|---|---|---|---|---|---|---|---|
| **SGLang Marlin pair** (baseline incumbent at T<128 and B128) | cp.async pipeline (Marlin internal) | Marlin tiles | mma.sync | Marlin heuristic, workspace 4 blocks/SM | block_size_m 8/16 m-blocks | FP32 reduce; `use_atomic_add` on w2 at T≤32 (removed the bs=1 slow mode); w13 atomic at T=1 is routing-gated | one m-block per 8/16 routes, so hot experts are re-streamed per m-block at T=16/32/128 (LESSONS 62) |
| **master s5 `wg_kernel`** (`submission.py:1229-1366`) | 1 thread: 1-D `cp.async.bulk` of a 9216 B record pair plus 2 TMA 2D slab boxes per stage; 4 stages, so ≤37 KB/SM of weights | weights in ring; slab 2×16 KB per stage; stmatrix epilogue tile | wgmma m64nNk16 RS (N = 16/32/64/128 per chunk); 8 per stage per consumer warpgroup; **`wait_group<0>` right after commit** (`:1170`); both consumer warpgroups consume the same stage in lockstep | 384 threads, 168 regs at launch, setmaxnreg 40/232 (`:1315/:1339`), 1 CTA/SM, smem 187–204 KB + 8 KB | persistent 132; items = chunks × 64 (w13) or × 32 (w2). At T=256 judge: 1536/132 = 11.6 waves (w13), 768/132 = 5.8 (w2) | none | chunks cut every 8 blocks (NMAX 128); a 256-route expert becomes 2 chunks (28 MB read twice) |
| **wg2 V64** (board branch, `combo-wg-occ2-v64-hot-interleave-align2:submission.py:3678-3693, 3796-3876, 3878-4030`) | same, 3 stages × 9216 B × 2 CTAs = 55 KB/SM | stage 25.6 KB; smem 87,040 / 95,232 B | same RS wgmma, `wait<0>` (`:3824`) | `__launch_bounds__(384,2)`, 80 regs, setmaxnreg 32/104, 2 CTAs/SM, grid 264 | T=256 judge: w13 1664/264 = 6.3 waves (makespan 7, ~90%); **w2 832/264 = 3.15 (makespan 4, ~79%)** | none | NMAX 64: the 256-route expert becomes 4 chunks, "weight read twice 85 MB" (mostly L2 hits: V64s/V64 = 0.997), dequantized 4× |
| **deep-ring n32s10 / n64s7** (b128-n32-deep-ring) | 10 × 9216 = 92 KB/SM, or 7 stages = 64.5 KB | slab 2×4 KB / 2×8 KB | s5 consumer | 168 regs, 1 CTA/SM | B128 | none | NMAX 32/64: judge-like T=128 splits 1–2 experts (28–85 MB re-read) |
| **s3 variant B** (`cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix:submission.py:702-760`) | s5 | s5 | `a0/a1` double buffer, **`wait<1>`** (`:742`) | s5 | measured only on a one-expert canary: 64 w13 CTAs / 32 w2 CTAs active, ~2350 clk/stage | none | — |
| **prefill rega** (`muse_prefill_layer/submission.py:2677-2760`, the same consumer) | s5 | s5 | s5 `wait<0>` | s5; V64/V32 2 CTAs/SM variants in experts-rega-low-class-two-ctas-per-sm | T=753: 32 chunks × 64 | none | multiple chunks per expert (L2 re-reads) |
| **fsi midt** (fused split-I; `cuda-fused-split-i-moe-midt-cluster-reduce:submission.py:4333-4770`) | bulk + TMA; KC256: 3 stages × 36.9 KB = 110 KB/SM, KC128: 5 × 18.4 KB | h kept in smem (bf16 SWIZZLE_128B, `fence.proxy.async`); fp32 partial-y tile 128×32 double-buffered | wgmma m64n32k16 RS, `dq_stage` + 8 wgmma + **`wait<0>` per NT tile** (`:4708, :4762`) | 384 threads, no setmaxnreg, 121–156 regs, 1 CTA/SM, 186–210 KB | 368–928 units on 120–132 CTAs (2.8–7.7 waves) | atomicAdd(float4) of fp32 partials into y32 (CL1); CL4 cluster DSMEM reduce with cluster mbarriers | block-32 align, read once at T≤32 |
| **rowowner M/F** (`nvfp4-rowowner-bulk-stream-tiny-t-rewrite:submission.py:4285-4760`) | **per-warp** private ring D=4 × 2304 B; refilled only after consume (`:4605-4607`), so ~3 records (6.9 KB) in flight per warp, **~55 KB/SM** | 2304 B records, x/h slab in smem | M: mma.sync m16n8k16, 2 accumulator sets, tokens padded to 8; F: fp32 FFMA after bf16→fp32 unpack | 128 threads, `launch_bounds(128,3)`, 96–128 regs, grid 264; w2 at T=4 is 1 CTA/SM (168 KB) | units = 16-row tiles over full K: w13 n_act×512, w2 n_act×256. Judge T=1 with 2–3 experts gives 1024–1536 w13 units over 1056 warps, makespan 1–2 | none (device work map, no spins) | read once (T≤4) |
| **swap-AB mma.sync persistent** (`cuda-nvfp4-swapab-mma-persistent-small-m-inproc:submission.py:425-753`) | all 256 threads issue cp.async.cg 16 B; 4 stages × (16 KB codes + 2 KB scales + 8·NT×512 B slab), ~54 KB/SM | swizzled slab, ldmatrix.x4 B | mma.sync m16n8k16, NT n-tiles; **`cp_wait` + `__syncthreads` every stage** (`:637-638`) | 256 threads, 80–128 regs, 1 CTA/SM, 88–139 KB | T=1: ~666 items (≈2.6 experts × 64 tiles × SK4) on 132 CTAs; items are only 4 stages long | split-K 4/8 (w13/w2 at NT=1), fp32 workspace + per-tile ticket, fixed-order reduce | BSM = 8·NT |
| **fused bs1 persistent** (`cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc:submission.py:414-503, 581-616`) | cp.async 16 B, 8–10 stages × 18 KB = 144–180 KB/SM | 18 KB stages, x/h in smem | mma.sync, token in column 0, even/odd accumulators | 256 threads, 72 regs, 158–195 KB, 1 CTA/SM, grid 132 | R×64 phase-1 + R×128 phase-2 items | **grid-wide `ld.acquire` spin** on phase-1 counters (`:794`), tickets, last-CTA combine | — |
| **tiny-m-bucket16** | s5 path, N16 only | s5 | wgmma N16 `wait<0>` | s5 | split-K 2 (w13) / 4 (w2) when items < 132 | ticket fix-up | — |
| **Triton split-K GEMV / w13 GEMV** (REWRITE_READ §5) | Triton loads; 8 KB smem, so the K loop is not pipelined | — | `tl.dot` | 211 regs, grid 264 | — | split-K 4/8 with ticket | arithmetic dequant, ~9–15 INT/FP ops per weight |
| **decode moe-chain swap-AB expert-once** (T≥33) | — | 221 KB smem | mma.sync | 148 regs, 1 CTA/SM | — | — | — |
| **bf16 bulk_kernel** (contrast; `combo-rowowner-reduce-free-m1-qkv-o-plus-naive:muse_bf16_proj/submission.py:2172-2300`) | 1 producer warp, 1-D bulk 16 KB chunks, 4–6 stages, 64–96 KB per CTA, 1–2 CTAs/SM | 16 KB weight chunks + x | ldmatrix A/B + mma.sync, about 3 instructions per 256 elements | 160 threads | N/R CTAs | intra-CTA 4-way K split reduced in smem only | — |
| **decode kernel P** (bf16, `muse_decode_layer/submission.py:272-300`) | cp.async 16 B, 3 × 16 KB in flight per CTA × 2 CTAs = 96 KB/SM, PDL weight prefetch | — | mma.sync | `launch_bounds(128,2)`, 216 CTAs in one wave | SPLIT_K 3 | fixed-order ticket | — |

### 2c. Limiter table

| kernel | measured TB/s or µs | binding limiter | evidence | fixable how |
|---|---|---|---|---|
| SGLang Marlin pair | Judge: 35.4 µs @T=1, 156.4 @T=16, 176.8 @T=32, 113.0 @T=8, 260.6–306.7 @T=128, 483.2 @T=256. Using mean touched experts that is ≈2.1, 2.1, 2.6, 2.1, 2.5, 1.4 TB/s | per-m-block re-stream of hot experts; generic cp.async pipeline; bs=1 split-reduce slow mode (cut by w2 atomic) | `archive/combo-wg-occ2-v64-hot-interleave-align2/31c27e4b8fc84bd2.profile.txt`; s4 profile (483.2 µs); LESSONS 62–66 | replace with Engine A (T≤32) and Engine B (T≥128) |
| master s5 wg (1 CTA/SM) | prepare (zero_ flush, 600 MB): w13 1.83–2.03, w2 1.74–2.19. Judge: 238.4 + 110.9 µs @T=128, 271.0 + 126.1 @T=256 (≈1.75–1.86) | **per-stage serial critical path** ≈1150–1200 clk per 9216 B per SM: `mbar_wait(full)` → `dq_stage` (~210–250 instructions per warp, INT-heavy) → 8 wgmma → `wait<0>` → arrive. Both warpgroups lockstepped on the same stage, so the ALU idles during the MMA chain and vice versa; only 2 consumer warps per SMSP | `submission.py:1160-1175`; deep-ring log (37 → 92 KB/SM in flight: 1.83 → 1.87 TB/s); producer-scale-expand spec quotes "~1150 clk per 9216-byte stage per SM" | `wait<1>` with double-buffered A **measured at full grid**; 2 CTAs/SM; cheaper dequant (§3); stream-K tail |
| wg2 V64 (2 CTAs/SM) | prepare: w13 2.14–2.38, w2 2.28–2.52 (judge-like 1.98/1.97 with 85 MB re-read). Judge: 225.3 + 141.5 µs @T=256 | same consumer; TLP hides part of the MMA wait (gate 0.955). w2 last-wave quantisation (~79%). Hot expert ×4 dequant (NMAX 64). Possible L2→SM ingest at N64: 9.2 KB weights + 16 KB slab per stage, about 6.5 TB/s at 2.35 TB/s of weights | board log lines 26/55; `[wg2-occ2] select` lines; prefill LESSONS 180 ("~6.4 TB/s L2->SM at bucket 64") | `wait<1>` within 104 regs (BK=64); stream-K tail (queued `wg2-stream-k-tail-floor-g-over-r`); hot experts read once on the 1-CTA N≤256 path |
| deep ring n32s10 / n64s7 | 1.74–2.13 (w13), 1.65–2.16 (w2) | consumer-bound (bytes in flight are not the limiter at 1 CTA/SM) | `archive/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring/6759e36bbd4648fc.log` | n/a (diagnostic) |
| s3 variant B (`wait<1>`) | single-expert canary only: 57.4 → 60.3 µs (N16), 85.7 → 91.7 µs (N128) | **confounded**: only 64/32 CTAs active, ~2350 clk/stage, latency-starved; says nothing about the full-grid consumer | `archive/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix/3bdec0e58d684c17.log:71-79` | re-measure at full grid on a judge-like routing (ladder R3) |
| prefill rega (same consumer) | weights 1.98–2.03 (1 CTA/SM) → 2.29–2.31 (2 CTAs/SM); L2→SM including slab 3.7–6.1 TB/s | at N128 / T≥3012 the tensor core plus activation ingest bind; at N32 the same ~2 TB/s consumer cap | prefill logs: ping-pong 1.18–1.53× **slower**, scale-expand 1.03–1.13× slower, MC2 multicast 1.3–1.9× slower, 2 CTAs/SM 0.875–0.887× | do not serialize dequant across warpgroups; do not add producer-side latency per stage; no cluster multicast |
| fsi midt (fused split-I) | 1.58–1.88 (CL1), 1.07–1.37 (CL4); gate 0.45–0.81× of master | same `wait<0>` consumer per NT tile; h round-trip through smem; CL4 cluster barrier and DSMEM reduce stall the stream; 1 CTA/SM; 2.8–7.7 waves | `archive/cuda-fused-split-i-moe-midt-cluster-reduce/6fe2a63f490f49a3.log` (DIAG/GATE) | drop the cluster reduce; row ownership (Engine A) is preferable to split-I with fp32 y partials |
| rowowner M (mma.sync) | eager single launch, zero_ flush, T=1 with 8 experts: w13 2.12 (151 MB), w2 1.71 (75.5 MB). Normalised for ~50 MB dirty write-back, both ≈2.8 TB/s of total HBM traffic. T=4: 2.28/1.25. Gate 0.884/0.966× master @T=1, 0.69/0.71× @T=4 | **bytes in flight**: ~3 × 2304 B per warp, ~55 KB/SM (bf16 needs ≥96 KB/SM for 3.4–3.6 TB/s); tiny 2.3 KB copies; refill after consume; full master dequant; w2 at T=4 is 1 CTA/SM | `archive/nvfp4-rowowner-bulk-stream-tiny-t-rewrite/c08dbd737e29442d.log`; branch `:4291, :4600-4607` | CTA-shared 18 KB stages, 4–6 deep, 2–3 CTAs/SM, stream-K; keep its device work map, PDL prefetch and zero-route handling |
| rowowner F (CUDA-core FFMA) | 1.12/0.98 (T=1), 0.79/0.41 (T=4) | per element per token: bf16→fp32 unpack (INT) + FFMA; ALU/FMA-bound | same log | dead; mma.sync wins |
| swap-AB mma.sync persistent | judge T=1: 44.7 + 31.2 µs (≈1.1 / 0.8 TB/s); 1.7–2× slower than Marlin at T≤32 | `__syncthreads` per stage across 8 load+compute warps; 4-stage split-K items with fp32 workspace and ticket; ~54 KB/SM; padding | REWRITE_READ harvest table (job fecc63ef567b4dc2) | dead in this form |
| fused bs1 persistent | judge fast mode 37.7 µs for the whole MoE; 41–59% of replays +23 µs | grid-wide `ld.acquire` spin between phases, so the judge loop is bimodal | REWRITE_READ §4, harvests (7a691942, cd0f6b08) | never spin across CTAs; two kernels + PDL |
| tiny-m-bucket16 (wgmma N16) | 46.0 vs 45.5 µs (T=1 skewed); 1.14–1.5× slower T=2..7 | wg consumer at N16 plus split-K ticket | DIAGNOSIS; `archive/cuda-nvfp4-wgmma-rega-tiny-m-bucket16/3df5c69750884600.log` | Engine A |
| Triton GEMV ×2 | 84.1 vs 41.1 µs @bs=1; 345 µs @bs=8/16 | unpipelined K loop (8 KB smem), 211 regs, arithmetic dequant | REWRITE_READ §5 | dead |
| decode swap-AB expert-once (T=128) | judge 405 + 207 µs vs Marlin ~273–315 | mma.sync MMA-bound at large N; 1 CTA/SM | `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound/51d047cbc5ff450a.profile.txt` | wgmma for N ≥ 64 |
| bf16 bulk / kernel P (contrast) | pure read 2.54 TB/s prepare (zero_ flush, eager) **vs judge 19.0 µs / 64 MB = 3.37 TB/s**; kernel P 3.62 TB/s (graph, back-to-back, read-only) | trivial consumer; ≥96 KB/SM in flight | `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild/571ea532ad7a410d.log`; `/home/user/git/musespark/muse_decode_layer/submission.py:2863-2895` | template for Engine A's load path |

### 2d. Measurement confounders (read every prepare TB/s with these in mind)

1. **Write-flush.**
   - moe uses `flush.zero_()` of 256 MB before every sample: `submission.py:2350` (`_wg_med_cold`), `:2641` (selector), `:655`; the rowowner `_ro_rounds` and `_ro_kernel_tbps` do the same.
   - About 50 MB of dirty L2 must be written back during the measured kernel. That is +66% HBM traffic for a 75 MB kernel and +8% for a 600 MB kernel.
   - The judge reads its flush buffer, so it carries no such penalty.
2. **Eager single launches.**
   - These include host and launch gaps; `wg_gemm` encodes 4 CUtensorMaps on the host per call.
   - bf16 example: prepare showed the master path at 29.34 µs; the judge profiled the same kernel at 19.0 µs.
3. **Single-call profiles.** The judge profile is one routing draw per shape, so Marlin's ±20% between runs is routing variance (REWRITE_READ caveat).
4. **Earlier conclusions drawn in the wrong regime.**
   - "Consumer is not the limiter" (variant B) came from a grid-starved canary.
   - "Bytes in flight is not the limiter" (deep ring) holds only for the wg consumer at 1 CTA/SM. It does not hold for the per-warp rowowner ring.

---

## 3. Instruction budget and the cheapest exact dequant

### Per-SM capacities

These are approximate CUDA Programming Guide values for compute capability 9.0, at 1.98 GHz with 132 SMs. They are not measured on the judge.

| resource | lane-ops per clk per SM |
|---|---|
| warp-instruction issue (4 SMSP × 32) | 128 |
| INT32 ALU (LOP3 / SHF / PRMT / IADD3 / ISETP) | 64 |
| FMA pipe (FFMA / FMUL / HFMA2.BF16) | 128 |
| IMAD (FMA-heavy half) | 64 |
| "all other conversions" (F2F, F2FP, `cvt.e4m3x2`) | 16 |
| shared memory | 128 B/clk |
| dense bf16 tensor | ~2048 FMA/clk |

**Required element rate:** 4 TB/s = 15.3 B/clk/SM. At 0.5625 B/element that is **27.2 elements/clk/SM** (30.6 counting code bytes only). At 3.5 TB/s it is 23.8.

| per 8 elements (one lane-k16) | master `dequant8` + cvt scales | proposed |
|---|---|---|
| code extraction | 12 INT (4 × LOP3/SHF/LOP3) + 3 shifts | 6 INT + 5 IMAD (layout below) |
| scale × code | 4 HFMA2 | 4 HFMA2 |
| scale construction | 1 CVT + 2 HADD2.F32 + 2 FMUL + 2 F2FP (3 conversions) | 1 PRMT + 1 IMAD (or 2 + 2 if `.H0_H0` is not folded) |
| LDS (codes + scales) | 0.375 | 0.375 |
| MMA issue / B fragment / address | ~2–5 | ~2–4 |
| **issue per element** | 3.3–3.9 | 2.4–2.9 |
| INT-ALU per element | 1.7–2.2 | 0.9–1.2 |
| at 4 TB/s: issue | 90–106 / 128 (70–83%) | 65–79 / 128 (51–62%) |
| at 4 TB/s: INT ALU | 46–60 / 64 (**72–93%**) | 24–33 / 64 (38–52%) |
| at 4 TB/s: conversions | 10.2 / 16 (64%) | 0 |
| headroom left for loads, MMA, address, barriers, epilogue | ~0.8–1.4 issue/el; ALU nearly saturated | ~1.8–2.3 issue/el; ~half the ALU free |

The MMA itself is cheap. With mma.sync m16n8k16 swap-AB it is one instruction per 256 elements per warp, which at 4 TB/s is ~218 FMA/clk/SM (~11% of tensor peak) at N=8 and ~44% at N=32.

### Code bit layout

This is a lossless bijection on the 32 bits of each lane word, applied once in prepare. Define the bf16x2 target set P = {6,7,8,15, 22,23,24,31}: sign at bits 15/31, magnitude (e1 e0 m) at bits 8..6 / 24..22.

| pair | source bits | extraction | cost |
|---|---|---|---|
| pair0 | P | `v0 = q & 0x81C081C0` | 1 LOP3 |
| pair1 | P−3 = {3,4,5,12,19,20,21,28} | `v1 = (q<<3) & 0x81C081C0` | IMAD.SHL + LOP3 |
| pair2 | P−6 = {0,1,2,9,16,17,18,25} | `v2 = (q<<6) & 0x81C081C0` | IMAD.SHL + LOP3 |
| pair3 | leftover {10,11,13,14} (+16): (e1,e0) at (11,10), m at 13, s at 14 | `v3 = ((q>>3)&0x01800180) \| ((q>>7)&0x00400040) \| ((q<<1)&0x80008000)` | 3 shifts (IMAD.HI / IMAD.SHL) + 3 LOP3 |

Four rotations of P cannot tile 32 bits: it reduces to tiling Z16 by {0,1,2,9}, and the difference-set argument shows there is no solution. So one pair must always cost about 6 operations. Total: 11 ops instead of 15, and ~6 INT instead of 12–15.

### Exactness proof

The 2^126 bias is split as code 2^−126, scale 2^+119, epilogue 2^+7.

1. **Code.** For c = (s, e1, e0, m) with E = 2e1 + e0, the bf16 bits `B = s<<15 | E<<7 | m<<6` equal v(c) · 2^−126 exactly.
   - E ≥ 1: 2^(E−127)(1 + m/2) = 2^−126 · 2^(E−1)(1 + m/2).
   - E = 0: subnormal, 2^−126 · m/2 = 2^−126 · v (v = 0.5·m).
   - ±0 maps to ±0.
2. **Scale (integer, no cvt).** For an e4m3 byte b with sign 0, exponent F ∈ [1,15], mantissa M, and b ≠ 0x7F: `S = (b<<4) + (239<<7)` is bf16 of sf · 2^119. The value b<<4 places F at the exponent and M<<4 in the mantissa; +239 lifts the exponent to F + 239 ≤ 254 with no carry, giving 2^(F−7)(1 + M/8) · 2^119.
   - For the bf16x2 broadcast: `PRMT(w, RZ, 0x4040)` gives [b,0,b,0], then `IMAD(t, 16, 0x77807780)` gives {S, S}. No carry between halves, since 0x7F0 + 0x7780 < 2^16.
   - Better: one PRMT/IMAD pair builds {S(g), S(g+8)}, then HFMA2 uses `.H0_H0` / `.H1_H1` operand selection (written in PTX as `mov.b32 {lo,hi}`).
   - Precondition, checked once in prepare: every scale byte lies in [0x08, 0x7E]. The generator clamps sf to [8, 448], so F ∈ [10, 15] and the check always passes. Otherwise fall back to master's cvt path. Marlin's own repack (`prepare_moe_nvfp4_layer_for_marlin`, per vLLM lineage) likewise re-encodes positive scales so its in-kernel scale expansion is integer-only. That is recalled, not visible in this repo.
3. **Product.** `fma.rn.bf16x2(B, S, −0)` equals v · sf · 2^−7 exactly.
   - The significand product (2 or 3)(8 + M)/16 has at most 6 significant bits, within bf16's 8.
   - The magnitude lies in [0.5·2^−6·2^−7 = 2^−14, 6·448·2^−7 ≈ 21], all normal, so nothing rounds.
   - Subnormal B inputs (c = ±0.5) are not flushed: bf16 FMA has no `.ftz`, and master's judge self-checks show 3.9e-3 rel-L2 against fp32 versus Marlin's 5–6e-3.
4. **Epilogue.** `acc · (gs · 2^7)` in fp32. Scaling gs by 2^7 is exact.
   - The result equals Σ (e2m1 · sf · gs) · x up to fp32 accumulation order. bf16 × bf16 products are exact in fp32.
   - The reference instead rounds sf·gs to fp32 first; the difference is within an ulp.
   - Weights are stored as the same codes and bytes, only permuted. **Lossless.**

**PRMT LUT, rejected.**
- A code nibble cannot be used directly as a PRMT selector: bit 3 (the sign) turns on PRMT's sign-replicate mode.
- Per 4 elements a LUT needs LOP3 (mask `0x7777`) + 2 table PRMTs + 2 interleave PRMTs + 2 LOP3 sign merges = **1.75 INT/el**, versus ~0.75–1.0 for the bit trick.
- Tensor-side scaling (codes unscaled into the MMA, scale applied to fp32 outputs) costs the same instruction count at N=8 and risks fp32 underflow of 2^−126 × x products. Rejected.

---

## 4. Engine A: small M (≤32 routes per expert; moe bs ≤ 32, decode T ≤ 32)

**Structure:** three kernels chained with PDL.
- **K1, w13 + SiLU.**
  - Each CTA builds the device work map from `topk_ids` (rowowner's shuffle/ballot pattern). K1 replaces `_align_small`.
  - It writes `local_ids` in master's [T, 8] layout and zeros `ss[T*8]`.
- **K2, w2 + per-route sum of squares.**
- **K3, post-norm combine.**

**Load path: a CTA-shared 1-D bulk ring.** Do not use per-warp rings, and keep register-direct streaming only as a ladder variant.
- 1 producer warp: lane 0 issues one 18,432 B `cp.async.bulk` per stage (4 warps × 16 rows × 512 k, contiguous after repack). Lanes 0..n_e−1 issue 1 KB x/h row segments for the expert's routes, as the bf16 kernel does.
- 4 consumer warps per CTA, 160 threads.
- Ring 4–6 stages; 2–3 CTAs/SM; **≥128 KB of weights in flight per SM**. The bf16 evidence is 3.4–3.6 TB/s at 96–128 KB/SM. Register-direct LDG.128 would need ~64 regs/lane for 8 × 16 B in flight and ties depth to occupancy.

**Consumer: mma.sync m16n8k16 swap-AB.**
- Weights are A, 16 rows per warp over the **full K** (row ownership).
- Tokens are B: N = 8 × ⌈n_e/8⌉, up to 4 n-tiles. The dequantized A fragment is reused across n-tiles.
- 2 accumulator sets per n-tile break the MMA dependency chain.
- B fragments come from `ldmatrix.x2/x4` with per-lane row addresses; padding rows point at a zero row.
- FMA is rejected: the F variant measured 0.47–0.55× of M.

**Prepare-time repack (lossless).**
- Layout: [expert][64-row group][512-k chunk][warp][16 rows × 512 k record = 4096 B codes + 512 B scales].
- w13 16-row tiles hold 8 gate + 8 up rows of the same 8 h-columns, so SiLU is fused in registers (lane holds c0 = gate, c2 = up for token 2·t4).
- Codes use the §3 bit layout; scale bytes are permuted to [b(g,s), b(g,s+1), b(g+8,s), b(g+8,s+1)].

**Balance: stream-K, not split-I.**
- Each CTA takes a contiguous slice of the flattened (group, k-chunk) list: 8 chunks per group (K = 4096).
- At most 2 partial groups per CTA: partials go to an fp32 workspace with a per-group ticket; the last arriver adds and runs the epilogue and resets the ticket. **No spinning.** This is the decode kernel P / swap-AB ticket pattern, proven on the judge.
- Without stream-K, T=1 with 2.6 experts gives 1331 w13 tiles over 264 CTAs → makespan 6 vs 5.04 average (84%).
- Split-I with `red.add` of fp32 y was rejected: fsi was 1.07–1.88 TB/s, y-partial traffic grows as I-chunks × routes × 16 KB, and it needs a zeroed y32.

**Epilogues.**
- K1: h = bf16(silu(bf16(g·gs) ) · bf16(u·gs)), matching master's `silu_mul` parity.
- K2: y bf16 [route, H], plus Σ (bf16(y)·(1+w))² over the CTA's rows → one `red.global.add.f32` into `ss[route]` per CTA span.
- K3 (PDL, ~2–3 µs): out[t] = Σ_s g·y·(1+w)·rsqrt(ss/H + 1e-8); exact zeros for tokens with no local route.

**PDL.**
- K2 issues its first ring stages **before** `griddepcontrol.wait`, since w2 weights are known from `topk_ids`; only h waits.
- K1 calls `launch_dependents` after its last weight issue. Overlap occurs only as K1 CTAs exit (smem is full at 2–3 CTAs/SM).
- In decode_layer, K1 itself must wait on the router.

**Expected (estimates).** Kernel-only 3.0–3.5 TB/s on the judge (clean L2), plus ~5 µs fixed (ramp, w13→w2 gap, combine, prologue):

| shape | bytes | expected µs | current µs |
|---|---|---|---|
| decode_1 | 73 MB | 26–29 | 46.4 |
| verify_1 | 238 MB | 73–84 | 104.5 |
| decode_16 | 337 MB | 101–117 | 150.9 |
| decode_32 | 453 MB | 134–156 | 199.1 |

Hot experts at T=32 read once (N ≤ 32), unlike Marlin's per-m-block re-streams.

## 5. Engine B: large M (T = 128/256; hot experts up to 256 routes)

**Consumer.**
- wgmma m64nNk16 RS with **BK = 64 stages** (4608 B weights + N × 128 B slab).
- A double buffer `a[2][4][4]` (32 regs). Per stage: `wait full[s]` → dequant into `a[s&1]` (§3 sequence) → `wgmma.fence` → 4 wgmma → commit → **`wgmma.wait_group<1>`** (retires stage s−1) → arrive `empty[s−1]`.
- The A buffer being written is never in flight (PTX-legal). The warpgroups stay concurrent; do not ping-pong them.

**Occupancy cells.**
- N ≤ 64: 2 CTAs/SM, setmaxnreg 32/104 (acc ≤ 32 + A 32 + ~25 ≈ 90 regs).
- N = 128/256: 1 CTA/SM, 232-reg consumers (acc 64/128 + 32 + ~25). Ring 4–5 × (4.6 KB + 16/32 KB slab) ≤ 183 KB.

**NMAX.** NMAX ≥ max routes (256) in the 1-CTA cell, so hot experts are **read and dequantized once** (V64 reads the T=256 hot expert 4×).
- The hot expert is MMA- and ingest-bound: 25.8 GFLOP at T=256, i.e. 96 tiles of ~45 µs each on one SM each, about 25–35 µs of full-GPU time.
- Issue hot items first (LPT); cold items stream on the remaining SMs. Run the hot (1-CTA) and cold (2-CTA) kernels as concurrent graph branches, or run cold experts (n_e ≤ 32) on Engine A.

**Tail and other levers.**
- Stream-K split of the last wave (w2 at V64 is ~79% efficient today).
- PDL chain gather/align → w13 → w2 → combine (master's call sites).
- No cluster multicast (prefill MC2 was 1.3–1.9× slower).
- BM-256 tiles were neutral in prefill.

**Expected (estimates).** Cold experts at 2.9–3.3 TB/s with the hot MMA mostly overlapped:

| shape | bytes | expected µs | current µs |
|---|---|---|---|
| verify_16 | 648 MB | 215–245 | 323.2 |
| verify_32 | 696 MB | 235–265 | 358.1 |

---

## 6. Microbenchmark ladder (one `okc check`, printed from prepare)

**Setup.**
- One extra load_inline module holds all rungs as template instantiations `<PATH, DQ, MMA, ACT, EPI>`, so there is one ~40 s compile. It runs after master's setup, with its own deadline (start ≤ 200 s, skip on deadline).
- Bank: real repacked weights for 8 local experts (226.5 MB, more than L2).
- Each sample is a CUDA graph [read-flush kernel (256 MB, sum into one word); rung kernel], timed with events. Subtract [read-flush; empty kernel]. Report median and p10/p90 over 31 reps.
- `clk_stage_cta = t_us·1980·(CTAs/SM·132)/stages_total`.

**Exact print format.** One line per rung, `key=value`, tag `[ladder]`:
```
[ladder] env sm=132 clk_mhz=1980 bank_mb=226.5 flush=read256 timer=graph reps=31
[ladder] R0 flush=read|zero_ rung=R1 t_us=58.1|70.4 dirty_penalty_us=12.3 dirty_penalty_pct=21.2
[ladder] R1 stream path=bulk1d|cpasync16|ldg128 stage_b=18432 ring=4 ctas_sm=2 warps=1p+4c inflight_kb_sm=144 t_us=58.10 p10=57.9 p90=58.6 tbps=3.90 clk_stage_cta=1270 regs=48 spill=0 smem_b=75008 occ=2
[ladder] R2 +dq dq=master|prop path=bulk1d ... tbps=... clk_stage_cta=...
[ladder] R3 +mma mma=sync16816:nt1:acc2|wg64n16:wait0|wg64n16:wait1dbuf dq=prop ... tbps=...
[ladder] R4 +act T=1|8|32 bsrc=smem_ldmatrix|tma_slab ... tbps=... l2sm_tbps=...
[ladder] R5 +epi kernel=w13_silu|w2_ss ... tbps=...
[ladder] R6 full T=1|8|16|32|128|256 routing=judge|random touched=.. routes=.. maxr=.. k1_us=.. k1_tbps=.. k2_us=.. k2_tbps=.. k3_us=.. path_us=.. master_us=.. ratio=..
[ladder] DIAG R6 T=1 clk_share wait_full=..% dq=..% mma=..% mma_wait=..% epi=..% fixup=..%
[ladder] VERDICT ceiling_tbps=R1max drops R1>R2=..% R2>R3=..% R3>R4=..% R4>R5=..% R5>R6=..% limiter=<first drop >5%> dq_exact=0/1904 mismatches
```

**Rung contents.**
- R1: only XOR-reduces the loaded words.
- R2: dequantizes all codes and XOR-accumulates the bf16x2 results.
- R3: adds the MMA with a constant B.
- R4: adds the real B operand.
- R5: adds the store epilogue.
- R6: the real kernels, with an instrumented instantiation recording per-CTA `clock64` buckets.
- `dq_exact` checks the proposed dequant bitwise against master over 16 codes × 119 valid scale bytes.

## 7. Risks and pitfalls (judge-specific), and what to validate first

**Build**
- Set `TORCH_CUDA_ARCH_LIST=9.0a` and pass only `-gencode arch=compute_90a,code=sm_90a` plus `-std=c++20 -O3 -lineinfo` in `extra_cuda_cflags`.
- Guard wgmma/setmaxnreg with `__CUDA_ARCH_FEAT_SM90_ALL` (else error word 2). Print regs, spill and smem per instantiation.
- A setmaxnreg pool must exactly equal launch regs × threads, or `inc` blocks (`combo` comment at `submission.py:3165`).

**Faults poison the context**
- Run a canary for every new kernel with sync and flushed prints.
- Use only instruction forms already proven on the judge: 1-D `cp.async.bulk` + mbarrier, `mma.sync` bf16, `fma.rn.bf16x2` −0, `cvt.rn.f16x2.e4m3x2`, wgmma RS, setmaxnreg, `griddepcontrol`, `ldmatrix`/`stmatrix`, `atomicAdd(float4*)`.
- Avoid `createpolicy` / `L2::cache_hint` (the 715 suspects).
- No child processes; `cuobjdump`-style SASS checks are impossible.

**State copies**
- The judge prepares 2 states and copies them by tensor-cloning. Keep no module-global device pointers or tensor maps; Engine A's 1-D bulk needs no maps.
- Per-state workspace, tickets, `ss` and error words. Tickets are reset by the finisher.
- Prove with the s5 A / deepcopy / tensor-clone graph check.

**Graph safety**
- Device-side work maps; branch only on shape and prepare flags; zero-local routings must produce exact zeros without hangs.

**No cross-CTA spins** (fused bs1 bimodality). Tickets of the "last arriver works" kind only.

**Gates**
- Use a read-only flush and graph replays.
- Use ≥2 routings (judge-like via `_wg_judge_ids`, plus random/zipf), two interleaved rounds, and an IQR bound (LESSONS 164).
- Correctness against an fp32-dequant reference with the judge's 1.5% / 5× row bounds, forcing every path: one-expert, all-local, skewed, zero-local, each n-tile and N bucket.

**Rules**
- Codes and e4m3 bytes may only be permuted; no bf16 scale tables in global memory.
- h and y stay bf16 (SGLang parity).
- Final review reruns nearby shapes (T = 2..7, 9..15, 17..31, 33), so no cliffs.

**Validate first, in this order**
1. R0/R1 ceiling with the read-flush.
2. `dq_exact` = 0 mismatches.
3. R2 for prop vs master, the drop relative to R1.
4. R3 sync-MMA at T=1/8 and wgmma `wait<1>` at full grid.
5. R6 at T=1 vs Marlin on the judge-like routing.

## 8. Open questions
1. What is the real DRAM ceiling on the judge (clean L2) for 18 KB 1-D bulk stages versus cp.async 16 B versus LDG.128 at 2–3 CTAs/SM? Is ~3.6 TB/s the practical cap, as the bf16 evidence suggests?
2. What is the sm_90 throughput of F2FP/F2F and `cvt.e4m3x2`, and does ptxas map half→float to HADD2.F32? This sets how much the current scale path really costs.
3. Does ptxas fold `mov.b32 {lo,lo}` into HFMA2.BF16 `.H0_H0`?
4. How large is the zero_ dirty-L2 penalty in practice (R0)? It retroactively rescales every prepare TB/s in the repo.
5. Does `wait<1>` with double-buffered A lift the wg consumer at full grid, and does BK=64 fit 104 regs without spills?
6. Is mma.sync with 4 n-tiles enough for decode_32 hot experts, or is wgmma N32 needed there?
7. For hot experts: is a 1-CTA N=256 read-once path or NMAX 128 with L2-shared chunks better on the judge?
8. What are the exact per-shape `router_load` / `router_distinct` values? They live in the evaluator's `task.yml`; only its hash is in the okc JSON.
9. In decode_layer, can any MoE work start before the router finishes (PDL only for setup), and how should Engine A share SMs with the attention tail?
