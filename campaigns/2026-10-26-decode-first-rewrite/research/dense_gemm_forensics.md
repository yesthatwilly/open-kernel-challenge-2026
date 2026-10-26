<!-- Research report, copied verbatim from a read-only Explore subagent (Opus) run in the planning session.
     Task: dense BF16 GEMM forensics and design. Finished 2026-10-26T03:46:12.574Z. Saved by campaign 2026-10-26-decode-first-rewrite. Do not edit; add notes to ../FINDINGS.md. -->

# Dense BF16 GEMMs in muse_bf16_proj and muse_decode_layer: limits, failed attempts, design, and decode port map

## 1. Summary

**Current state of bf16_proj**
- The board incumbent is job `741072dc`, scoring 1.2566.
- `FINDINGS.md`, `INDEX.md` and `BEST.json` are stale. They predate 7 newer harvests from 19:31–19:38 local time.
- Only two of those newer runs changed a kernel on a scored row:
  - `pooler-k-halves-fp32-partials-restack-m128-m256` (de95f817, check, 1.2683).
  - Its confirm run (caaa76cc, 1.2590).
- Despite the slug, both of those routed "cand a", which uses **bf16 partial sums**. The fp32-partials candidate lost every gate (0.93–0.99), so this is a rules risk (see §7).
- The other ~1.26 headlines put their new path back on master at their own gate, so their rise over 1.2566 is batch drift: 38daa28c, 566a79e4, 0124ea8f, 37ed8722.

**Where the time is**
- The m≤32 cells sit 0.5–3 µs above a practical floor. I define that floor as about 3.95 TB/s of DRAM plus about 0.8 µs of cold-start cost.
- The m=128/256 cells hold the largest absolute gaps:

| cell | row (µs) | floor (µs) | gap (µs) |
|---|---|---|---|
| pooler m=256 | 27.98 | 17.37 | 10.6 |
| qkv m=256 | 19.46 | 9.77 | 9.7 |
| pooler m=128 | 22.85 | 14.64 | 8.2 |
| qkv m=128 | 15.55 | 8.42 | 7.1 |
| router m=256 | 9.72 | 2.68 | 7.0 |
| router m=128 | 7.61 | 2.21 | 5.4 |

**Why every custom m≥128 kernel lost**
- The cause is the split-K decomposition and reduction layer, not the wgmma mainloop.
- CUTLASS stage 2's DataParallel diagnostic ran **6.86 TFLOP/s per SM**, about 85–92% of an SM's peak (qkv, splits=1: 18 tiles of 256×128×8192 in 78.27 µs).
- Its SplitK path, in contrast, added **10–32 µs per call**. All three CUTLASS patches set `ReductionMode::Deterministic` and also add an `initialize_workspace` memset node.
- The hand-written swap-AB kernel lost in the mainloop itself:
  - Clusters of 4 or more fit only 120 SMs, so qkv was forced down to 72 CTAs.
  - The configs that did fit had no x multicast.
- DeepGEMM never ran:
  - `bf16_gemm_nt` on this image is bitwise cuBLASLt.
  - The raw sm90 kernel failed to instantiate on the template parameter name `BLOCK_K_`.

**Recommended design (bf16)**
- One kernel family:
  - One M-tile covering all tokens × 128 weight rows.
  - x loaded by TMA and multicast across a 2-CTA cluster.
  - Static split-K that keeps the grid at ≤132 CTAs.
  - A deterministic reduction with no memset and no extra graph node.
- Estimate: **+3.5–4.5% on the bf16 headline**, about 1.31 from 1.2566.

| cell | today (µs) | estimate (µs) |
|---|---|---|
| qkv m=256 | 19.5 | ~15 |
| pooler m=256 | 28 | ~23.5 |
| qkv m=128 | 15.6 | ~12 |
| pooler m=128 | 22.9 | ~19 |

**m≤32**
- The tile, stage, split and TMA knobs are spent.
- The one untested combination: tile-contiguous weight repack + 1-D bulk/TMA loads + wgmma swap-AB + DSMEM split-K.
- The only kernel ever measured at 3.97 TB/s (the bulk m=1 kernel) is also the only one with the repack.

**Decode, which is the KPI**

| lever | headline gain |
|---|---|
| T=1 ports: bulk qkv, swap-AB o, ouns unpool, bulk-style P (about −7 µs per shape) | ≈ +1.0% |
| Rewrite of the P (pooler+router) stream at T=8..32 | +0.7–1.3% |
| Boundaries folded into neighbouring kernels at T≤8 (speculative) | +0.5–1% |
| Custom m=128/256 GEMM | ≈ +0.9% |
| Router at T≥128 | 0–0.5% |

**Two measurement flaws**
- Every prepare-time gate flushes L2 with `flush.zero_()`, a write, while the judge flushes with a read. This inflates ≥37 MB streams by 2–4.8 µs.
- bf16 profile kernel times for L2-resident weights are warm. Decode sees them cold: swap-AB qkv is 9.6 µs in the bf16 profile but 12.7–13.0 µs inside decode.

## 2. bf16_proj cell table

- **Incumbent and row** come from 741072dc (`/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/combo-rowowner-reduce-free-m1-qkv-o-plus-naive/741072dca5fc4c9d.profile.txt`).
- **Eff.** is (W + x + out bytes) / row; for m≥128 it also shows TFLOP/s.
- **Best measured** is the best different kernel ever run. "Same kernel" means the best row was drift on the incumbent kernel.
- **Practical** ≈ max(bytes/3.95 TB/s, FLOP/~800 TFLOP/s) + ~0.8 µs.

| m (load) | layer | incumbent kernel | row | SGL | floor | eff. | best measured (kernel, job, row) | row−floor | practical |
|---|---|---|---|---|---|---|---|---|---|
| 1 (dec_1) | qkv | `bulk_kernel<16,8192,8,4,2>` row-owner A2, repacked W, 2 CTA/SM (7.4 warm) | 10.54 | 13.60 | 7.87 | 3.58 TB/s | same (864e8094, paired 1.309); prior swapab16 12.46 (89fabef7) | 2.67 | ~10.4 |
| 1 | o | `_bf16_swapab_kernel` BT16 full-K (4.1 warm) | 6.32 | 7.10 | 3.50 | 2.66 | same 5.76–5.80 (drift); bulk A3 tied at the gate (0.986); oun cluster +14% (3f3330f7) | 2.82 | ~5.1 |
| 1 | pooler | `bulk_kernel<32,8192,8,6,1>` (16.9 = 3.97 TB/s) | 17.72 | 21.49 | 13.99 | 3.79 | 17.28 `bulk<32,8192,8,0>` (571ea532) | 3.73 | ~17.8 |
| 1 | unpool slice | `ouns_kernel` cluster-8 DSMEM split-K (4.4) | 5.01 | 6.85 | 1.75 | 1.68 | same 4.72 | 3.26 | ~3.5–4 |
| 1 | router | `router_kernel<8,8>` cluster-8 DSMEM, no memset (3.7) | 5.17 | 10.42 | 1.75 | 1.63 | same 4.85 | 3.42 | ~3.5–4 |
| 16 (dec_16) | qkv | swapab BT16 SPLIT_K 4 ticket (9.5 warm) | 12.64 | 13.86 | 7.93 | 3.01 | same | 4.71 | ~10.4 |
| 16 | o | swapab BT16 full-K (4.0) | 6.21 | 6.88 | 3.56 | 2.75 | same 5.78 | 2.66 | ~5.1 |
| 16 | pooler | swapab BT16 SPLIT_K 2 (19.9) | 19.46 | 21.60 | 14.06 | 3.47 | same 18.72 (84f17adc) | 5.40 | ~17.9 |
| 16 | unpool full | cuBLAS nvjet 64x16 2x1 TNT (18.5) | 18.82 | 18.85 | 14.06 | 3.59 | 17.97 `_pooler_skinny` (635c2693, old base, 1.033 paired); swapab 18.48 (05782267) | 4.75 | ~17.9 |
| 16 | router | `_router_swapab_kernel` + torch.zeros (3.4+0.7) | 5.51 | 15.10 | 1.81 | 1.57 | same 5.15; cluster m=16 5.64 lost (ef8a8bca) | 3.70 | ~4 |
| 32 (dec_32) | qkv | swapab BT32 SPLIT_K 4 (11.7) | 13.71 | 14.16 | 8.00 | 2.80 | same; SPLIT_K 8 15.64 lost (a61778a0); w128 split-K 7 lost 7% (c7138fe0) | 5.70 | ~10.5 |
| 32 | o | swapab BT32 full-K (4.5) | 6.47 | 6.88 | 3.62 | 2.69 | same 6.07 | 2.85 | ~5.2 |
| 32 | pooler | swapab BT32 SPLIT_K 2 (20.7) | 19.86 | 21.89 | 14.14 | 3.42 | 19.48 nvjet 64x16 via Lt preference (5ed9ace1; per-call toggle banned) | 5.72 | ~18.0 |
| 32 | unpool full | swapab BT32 full-K (18.3) | 18.49 | 18.72 | 14.14 | 3.67 | same 17.96 | 4.34 | ~18.0 |
| 32 | router | swapab + memset (3.6+0.8) | 5.95 | 15.22 | 1.86 | 1.50 | same 5.55 | 4.08 | ~4 |
| 8 (ver_1) | qkv | swapab BT16 SPLIT_K 4 (9.6) | 12.35 | 13.84 | 7.90 | 3.07 | same; A2 row-owner at m=8 lost its gate 0.949/0.959 (0124ea8f) | 4.45 | ~10.4 |
| 8 | o | swapab BT16 (3.9) | 5.90 | 6.53 | 3.53 | 2.87 | same 5.69 | 2.37 | ~5.1 |
| 8 | pooler | swapab BT16 SPLIT_K 2 (19.6) | 19.31 | 21.41 | 14.02 | 3.49 | same 18.75; bulk at m=8 +5% (571ea532) | 5.29 | ~17.9 |
| 8 | unpool slice | ouns (4.8) | 5.54 | 7.02 | 1.76 | 1.53 | same 4.88 | 3.78 | ~4 |
| 8 | router | `router_kernel<8,8>` (4.2) | 5.49 | 26.37 | 1.78 | 1.55 | same 4.85 | 3.71 | ~4 |
| 128 (ver_16) | qkv | cuBLASLt plan nvjet 40x64 64x16 4x2 TNN, single node (15.5) | 15.55 | 16.93 | 8.42 | 2.60 / 311 TF | same 15.51; CUTLASS one-tile gate 0.559 (d4edd38a) | 7.12 | ~11 |
| 128 | o | nvjet 128x64 1x2 TNT (5.7) | 7.68 | 7.68 | 3.99 | 2.49 / 280 | Lt NN 128x64 NNT 7.24 (37ed8722); CUTLASS 0.546 | 3.69 | ~5.7 |
| 128 | pooler | NN `torch.mm` → nvjet splitK NNT 20.8 + splitKreduce 3.0 | 22.85 | 25.65 | 14.64 | 3.07 / 376 | **K-halves bmm, bf16 partials 21.36 (de95f817: 19.6 + add 1.4)** | 8.21 | ~18.6 |
| 128 | unpooler | NN nvjet 128x64 NNT (19.3) | 18.88 | 19.95 | 14.64 | 3.72 / 455 | same 18.54 | 4.24 | ~18.6 |
| 128 | router | `_router_hilo_kernel` 64x64 SPLIT_K 16 + memset (5.0+0.8) | 7.61 | 16.32 | 2.21 | 1.39 / 71 | same 7.04; wgmma swap-AB tall-cluster gate 0.75–0.78 (38daa28c) | 5.40 | ~5 |
| 256 (ver_32) | qkv | nvjet 96x64 64x10 2x4 TNN (20.0) | 19.46 | 19.66 | 9.77 | 2.22 / 496 | same 19.26; Lt 144x128 split-K 20.29 (37ed8722); K-split bmm 22.17 (9d93c3d0); CUTLASS s1 28.52, s2 gate 0.56 | 9.69 | ~12.9 |
| 256 | o | nvjet 128x128 2x1 TNT (8.3) | 9.44 | 9.41 | 4.48 | 2.28 / 455 | same 9.01; CUTLASS s1 11.47, s2 0.50 | 4.96 | ~6.5 |
| 256 | pooler | NN nvjet 128x64 NNT (28.8) | 27.98 | 29.63 | 17.37 | 2.62 / 614 | **K-halves bf16 partials 26.75 (ce808e5a: 128x128 NNT 24.7 + add 2.1)**; 26.98 (caaa76cc) | 10.61 | ~22.3 |
| 256 | unpooler | nvjet 128x128 2x1 TNT (24.8) | 24.18 | 24.14 | 17.37 | 3.04 / 710 | same 23.81; CUTLASS s1 26.52 | 6.81 | ~22.3 |
| 256 | router | hilo 64x64 SPLIT_K 8 + memset (6.9+0.8) | 9.72 | 17.66 | 2.68 | 1.32 / 110 | same 9.03; tall-tile config lost in the judge (9.99, 864e8094) | 7.04 | ~5.5–6.5 |

**Headline value of 1 µs on one cell** (each load has 5 cells at weight 62; the score is a geomean over 6 loads):

| load | %/µs |
|---|---|
| dec_1 | 0.372 |
| ver_1 | 0.343 |
| dec_16 | 0.266 |
| dec_32 | 0.258 |
| ver_16 | 0.230 |
| ver_32 | 0.184 |

**Routing order in `kernel_fn`** (`/home/user/git/musespark/muse_bf16_proj/submission.py` L3165–3212):
1. ouns, for the unpooler slice at m≤8.
2. bulk, at m==1, with a per-cell flag.
3. swapab16, at m≤16.
4. The older Triton paths, now shadowed.
5. swapab, at 17..32.
6. NN `torch.mm`: pooler/unpooler at 65..128 (`nn_ok`), and pooler at 129..256 (`nn_p256_ok`).
7. The qkv m==128 Lt plan.
8. cuBLAS dispatch.
9. Router: cluster (m≤8), then swapab plus memset (≤32), then hilo plus memset.

The selector is `_cell_gate` (L2648): graph replay, a 256 MB **write** flush, 41 interleaved reps, the empty-graph baseline subtracted, then a profiler assertion on the kernel name and a two-state A/B self-check.

## 3. Limiter by cell class

**m=1**
- These cells are pure DRAM streams.
- The bulk row-owner reaches 3.97 TB/s in-kernel (pooler 16.9 µs). Every other design reached ≤3.7 TB/s.
- pooler and qkv sit within about 0.2–0.5 µs of a practical floor.
- o (16.8 MB) runs at 2.66 TB/s by row, so it is latency/ramp-bound: about 1.5 µs of cold first-byte and launch cost on a 4.3 µs stream.
- The unpooler slice and the router (8.4 MB) are dominated by fixed cost (row about 5 µs against a 2.1 µs stream). Deleting the memset node gave −0.4 to −0.5 µs (LESSONS 04:36Z).

**m=8**
- swapab runs wgmma m64n16k16, row-major weight in 512 B segments, and a split-K ticket reduce (qkv split 4 over 144 CTAs at 2 CTA/SM; pooler split 2 over 128 CTAs).
- It reaches 3.07–3.49 TB/s.
- The row-owner design scales linearly with m because x is fetched in 512 B pieces per token. Evidence:
  - qkv A2 at m=8 had a gate ratio of 0.95.
  - pooler bulk at m=8 lost 5%.

**m=16/32**
- Same kernels as m=8, plus x traffic of 25–50% of the weight bytes into each CTA.
- qkv is worst at 2.80–3.01 TB/s. Its split-4 ticket tail and 144 CTAs on 132 SMs hurt, and SPLIT_K 8 added +14%, "all in the reduce tail".
- Ten or more redesigns (Triton, CUDA gemv, bulk-TMA, lossless) all landed at 3.4–3.95 TB/s. Lossless compression is dead: the decode alone runs at 1.74 TB/s compressed.

**m=128**
- These cells are memory-bound with poor CTA fill.
- qkv: the cuBLASLt pick is a 40x64 tile, 116 CTAs, wgmma at M=40, giving 311 TF/s.
- pooler: split-K plus a separate splitKreduce node (3.0 µs) gives 3.07 TB/s.
- unpooler: 3.72 TB/s, almost no headroom.
- o: K=1024 gives only 16 k-blocks, so the kernel is fill/epilogue-dominated.
- In every tokens-on-M Triton or CUTLASS design, x is re-read from L2 by every weight tile (about 2× the weight bytes; REWRITE_READ §1).

**m=256**
- These cells are compute-plus-quantization bound.
- qkv: N=2304 maps to 96x64 tiles, 96 CTAs (73% of SMs), giving 496 TF/s.
- pooler: 128x64 tiles mean heavy x re-reads, giving 614 TF/s.
- unpooler: 128x128 2x1 reaches 710 TF/s, which is near practical.
- o: K=1024, 455 TF/s.

**Router hi+lo**
- m≤8: the cluster kernel. m=16/32: swapab plus memset. m≥128: hilo plus memset.
- At m=256, hilo (64x64 tiles, SPLIT_K 8, 128 CTAs) re-reads W through 4 M-tiles and x through 4 N-tiles. That is about 49 MB of L2→SM traffic for 8.4 MB of weights, plus 2 MB of fp32 atomics, plus a 0.8 µs memset node.
- The narrow N=256 forces a trade between x re-reads and deep split-K reductions. Three custom attempts lost: tiled cluster (b7b2b55d), wgmma swap-AB tall cluster (38daa28c, no-reduce 6.6–10 µs against master 7.1–9.3), and tall-tile Triton (864e8094).

**Unpooler slice (n=1024, m≤8)**
- 8.4 MB, latency-bound: ouns runs 4.4–4.8 µs in-kernel, about 1.9 TB/s.
- It already issues all loads at once (64 KB per CTA) and reduces through DSMEM. The remaining lever is the cold-start cost, which only PDL prefetch can hide (decode only).

## 4. m=128/256: why custom kernels lost, the design to build, and DeepGEMM/CUTLASS

### 4a. Evidence per attempt

| attempt (job) | design | measured | cause |
|---|---|---|---|
| `pooler-unpooler-m128-wgmma-splitk-semaphore` (d34fd76f) | Triton 128x128x64, tokens on M, split 4/2 | ~26 µs warm mainloop, above nvjet's whole kernel | x K-slice re-read per CTA ≈ 2× W bytes |
| `pooler-unpooler-m128-tma-wspec-rewrite` (61da9b91) | Triton TMA + warp_specialize | pooler 255 regs / **110 spills**; mainloop 27.6/27.4 µs warm; rows 31.69/25.68 vs 25.55/19.79 | same tiling; Triton warp-spec spills |
| `bf16-verify-streamk-cutlass-wgmma-rewrite` s1 (f5b1361a) | CUTLASS 4.5 128x128x64 cooperative, cluster 2x1, StreamK (qkv, pooler) / DataParallel (o, unpooler) | judge 28.52 / 11.47 / 34.28 / 26.52 vs 19.62 / 9.01 / 30.53 / 24.50; warm diag 343 / 495 / 408 / 608 vs cuBLAS 484 / 511 / 316 / 647 TF/s | two 128-row M-tiles, Deterministic StreamK fixup, Memset node (+0.9 µs kernel, about +2.3 µs row) |
| `bf16-verify32-cutlass-coop-256x128-splitk-stage2` (92c42836) | 256x128x64 cooperative, cluster 1x2 (x multicast), SplitK 7/4/2/2, Deterministic | gate (prepare, net): qkv **43.68 vs 24.38**; o 18.43 vs 9.18; pooler 0.727×; unpooler 0.762×. **DataParallel diag: qkv splits=1 78.27 µs → 6.86 TFLOP/s/SM**; o splits=1 15.01 (64 CTAs) | the mainloop is fine; the SplitK fixup (Deterministic, `ReductionMode::Deterministic` in all 3 patches) plus `initialize_workspace` memset costs 10–32 µs and grows with workspace bytes (8–13.5 MB of fp32 partials), not with link count |
| `bf16-verify16-cutlass-m128-onetile-splitk-rewrite` (d4edd38a) | 128x128x64 cooperative, cluster 1x2, SplitK 7/2/4/2, Deterministic | gate: qkv 31.87 vs 17.82 (0.559); o 14.18 vs 7.74; pooler 35.01 vs 27.33; unpooler 30.82 vs 23.71 | same |
| `bf16-verify-swapab-wgmma-cluster-dsmem-rewrite` (3a7f42bc) | hand-written swap-AB m64nTk16, TMA, x multicast in pairs, DSMEM split-K | **no-reduce** qkv m=256 49.25–53.47 vs cuBLAS 28.7; unpooler m=128 (1,2) 46.98 vs 29.06; DSMEM reduce itself only 1.8–4.1 µs | clusters ≥4 at about 197 KB smem fit only 120 SMs (size 4: 30 clusters, size 7–8: 15, size 2: 66), so qkv ran 72 CTAs; the (1,S) configs that fit have no multicast; the consumer releases stages one k-block late (`wait_group<1>`) with 4×64-K stages; no ncu |
| `bf16-verify-deepgemm-bf16-gemm-nt-route` (2d522d47) | `deep_gemm.bf16_gemm_nt` | "warm+check 0.0 s", rel L2 bitwise 1.66e-3, profile shows nvjet | the API is cuBLASLt on the 0.2.1.post1+muse build |
| `…-deepgemm-sm90-kernel-asserted-rewrite` (4c224beb) / `…-explicit-instantiation-stage2` (566a79e4) | raw `deep_gemm::sm90_bf16_gemm_impl` | VOID: regex parse failed; then "unknown template parameter 'BLOCK_K_'" | name matching only; its own model predicted 19.7–39.3 µs (no split-K) |
| cuBLAS reshapes (26a8be34, 37ed8722, 9d93c3d0) | Lt algo / plan, K-split bmm | qkv m=256 Lt pick: +6.8% / 1.07 in prepare but −14% / −4% in the judge; o128 NN −0.44 µs; K-split bmm 22.17 | the prepare gate mis-ranks (see §7) |
| router m≥128 (b7b2b55d, 38daa28c, 864e8094) | mma.sync tiled cluster, wgmma tall cluster, Triton tall tile | 17.3 vs 12.5 (eager); 9.15–9.50 vs 7.14 and 12.0–12.8 vs 9.2; 9.99 vs 9.72 | narrow N: x re-reads vs reduction traffic |

**What these runs establish**
- A CUTLASS mainloop at about 85–92% of SM peak exists and builds and runs on the judge. The image has CUTLASS 4.5.0 at `/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include`, built in 49 s with `TORCH_CUDA_ARCH_LIST=9.0a` and `-std=c++20`.
- What is broken is the global-workspace Deterministic fixup and the memset node.
- An extra graph node costs about 1.5–2.5 µs per row at these sizes.

### 4b. Design: "one M-tile, x-multicast, static split-K, on-chip/last-arriver reduce" (sm_90a)

**Orientation and tile**
- TN layout: A = x [m,k] (tokens on wgmma M), B = W [n,k] K-major. Swap-AB is only needed at m<64.
- CTA tile BM = m_pad (128 or 256: all tokens in one tile, so W leaves DRAM exactly once), BN = 128 weight rows, BK = 64 (128 B rows, 128B swizzle).

**Warps, registers and smem**
- One producer warpgroup at setmaxnreg 40, with one elected thread issuing TMA.
- Two consumer warpgroups at setmaxnreg 232. Budget: 2×128×232 + 128×40 = 64,512 ≤ 65,536.
- m=256: each consumer warpgroup owns 128 tokens × 128 rows, i.e. 2× m64n128k16 per k16 and 128 fp32 accumulator registers.
- m=128: each owns 64×128, 64 accumulator registers.
- Stages: m=256 uses 4 × (32 KB x + 16 KB W) = 192 KB; m=128 uses 6 × 32 KB = 192 KB.
- Keep 2 wgmma groups in flight, and release a stage after `wait_group(1)` of the following group.

**TMA**
- W: a 2-D box of 128×64 over a prepare()-time **tile-contiguous lossless repack**, so each box is one contiguous 16 KB block with long DRAM bursts.
- x: a box issued half by each CTA of a cluster (2,1,1) along N with `.multicast::cluster` mask 0b11.
- At m=256, L2→SM ingest is about 32 KB per 4.19 MFLOP. That is about 6.6–6.9 TB/s at 6.5–6.9 TFLOP/s per SM, which is under the ≥8 TB/s that cuBLAS 128x128 at 710 TF/s implies.
- Without multicast the need is 9.5 TB/s, so multicast is mandatory.

**Grid**
- One wave at 1 CTA/SM. Cluster-2 allows 66 clusters, so it fits. No persistence is needed.

| layer | weight tiles × splits | CTAs | notes |
|---|---|---|---|
| qkv | 18 × 7 | 126 | k-blocks per split 18/18/18/18/18/19/19 |
| pooler | 32 × 4 | 128 | |
| unpooler | 64 × 2 | 128 | |
| o | 64 × 2 | 128 | K-slice 512 |

**Reduction** (deterministic, no memset, no extra node; per-state fp32 workspace in fragment-major layout plus self-resetting int counters)
- S=2: the first CTA to arrive stores its partial with coalesced float4 writes and does a release-increment. The last CTA acquires, adds the partner's partial into its registers, rounds once to bf16, stores, and resets the counter.
- S≥3: every CTA stores its partial and arrives. Each then spins on `ld.acquire.gpu` until the count reaches S, then reduces 1/S of the fragment range in fixed split order. A departure counter resets both counters.
- Co-residency is guaranteed because the grid is ≤132 at 1 CTA/SM, and the kernel issues no `griddepcontrol.launch_dependents` before the reduce.
- The fp32 sum is rounded once, the same numerics as cuBLAS split-K.

**Epilogue and PDL**
- bf16 output staged through smem and written with a TMA store.
- For decode: issue the W TMA loads for stages 0..S−1 before `griddepcontrol.wait`; issue x loads after it.

**Expected rows (judge, cold)**

| cell | today (µs) | estimate (µs) | breakdown |
|---|---|---|---|
| qkv m=256 | 19.46 | **14.5–15.5** | compute 11.8 + fill 1 + reduce of 16 MB partials ~2.5 |
| pooler m=256 | 27.98 / 26.75 (K-halves) | **23–24.5** | |
| unpooler m=256 | 24.18 | **22.5–23.5** | |
| o m=256 | 9.44 | **7–7.5** | |
| qkv m=128 | 15.55 | **11.5–12.5** | |
| pooler m=128 | 22.85 / 21.36 | **18.5–19.5** | |
| unpooler m=128 | 18.88 | ~18.5–19 | no gain; keep NN unless the repack lifts DRAM efficiency |
| o m=128 | 7.68 | **6–6.5** | |

That is about 12 µs at 0.184%/µs plus about 8 µs at 0.230%/µs, i.e. **+3.5–4.5% on the bf16 headline**.

**Staging plan**
1. Cheapest diagnostic (one check): rebuild stage 2 with `ReductionMode::Nondeterministic` (parallel fixup), relax the "run2 bit-identical" self-check to a rel-L2 bound, switch to a read flush, and print splits=1 vs split-S again. If qkv m=256 drops to ≤17 µs net, the fixup diagnosis is confirmed and CUTLASS's `CollectiveMma` can be kept with a custom scheduler and reduction.
2. Write the custom kernel described above.
3. Add the tile-contiguous W repack.

### 4c. DeepGEMM and CUTLASS launchability

- **CUTLASS already launches.** Stages 1, 2 and m128 all built, passed their self-checks, and showed `cutlass::device_kernel` rows. Do not reuse its StreamK SplitK with Deterministic fixup.
- **DeepGEMM can be made to launch** by positional explicit instantiation of `sm90_bf16_gemm_impl`:
  - Template parameter order: kMajorA, kMajorB, SHAPE_M=0 (runtime M), SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M/N/K, the three swizzles, stages, kNumTMAThreads, kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, GemmType::Normal, kWithAccumulation, cd_dtype_t.
  - Host side: three `cute::TmaDescriptor`s and a cluster launch.
  - Its include tree carries CUTLASS 4.2.1.
- **It is not worth more spend.** The kernel has no split-K, so at m=128 it either under-fills 132 SMs or re-reads W and x through 64-wide tiles. Its own model predicted 19.7–39.3 µs, which is at or above cuBLAS on every cell.

## 5. m≤32: what remains and the best design per cell

**What remains, per class**
- **Graph nodes.** The only extra nodes left are the router memsets at m=16/32/128/256 (0.7–0.8 µs kernel). Deleting a node gave −0.4 to −0.5 µs at m≤8. The DSMEM cluster router at m=16 lost (5.64 vs 5.51), so a ticket-based memset-free swap-AB is the remaining option.
- **Split-K tails.** The qkv split-4 ticket costs about 1–2 µs at m=8..32. Replace it with cluster-2 DSMEM along K (66 clusters fit) and drop the workspace and ticket.
- **Launch ramp and cold first bytes.** About 0.8–1.5 µs per kernel. o, the unpooler slice and the router are already all-in-flight. Only PDL prefetch hides the rest, which applies to decode, not bf16.
- **DRAM efficiency.** The only 3.97 TB/s design is bulk: repacked contiguous 16 KB chunks, 1-D `cp.async.bulk`, no split. Everything else reads row-major 128–512 B segments at 3.4–3.7 TB/s.

**Best design per cell**
- **pooler / qkv / unpooler-full at m=8..32 (new):**
  - wgmma swap-AB m64nNk16 (N = 8/16/32 tokens).
  - Weights repacked tile-contiguous and streamed by a producer warp with 1-D bulk copies (16–32 KB) through a 4–6 stage ring.
  - Split-K via cluster-2 DSMEM; 128–132 CTAs.
  - x needs no multicast at m≤32: L2 ingest is about 1.5× DRAM, under the ceiling.

  | cell | today m=8/16/32 (µs) | estimate (µs) |
  |---|---|---|
  | pooler | 19.3 / 19.5 / 19.9 | ~17.5–18 |
  | qkv | 12.4 / 12.6 / 13.7 | ~10.8–11.3 |
  | unpooler full (m=16/32 only) | 18.8 / 18.5 | ~17.5–18 |

  - If the 3.9 TB/s rate materializes, that is **+3–4% on bf16**. Probability is moderate given the "local minimum" history, so make the first measurement an A/B of the same kernel on the row-major vs the repacked weight.
- **pooler, qkv at m=1:** keep the bulk row-owner. Using all 132 SMs instead of 128 is marginal.
- **o at m=1..32:** keep swapab. An all-in-flight repacked bulk with a balanced grid may give −0.3 to −0.5 µs.
- **unpooler slice and router at m≤8:** keep the cluster kernels; ≤0.3 µs is left.
- **router at m=16/32:** a memset-free swapab gives about −0.4 µs per cell.

## 6. Port map to muse_decode_layer

Sources: the profile `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/combo-measured-rows-slide-batched-preattn-t8-tail/a903b9d004d2452d.profile.txt` and `submission.py` `kernel_fn` L7301.

**Rules for estimating**
- Use bf16 **row** deltas (SGLang row minus ours). Do not use profile deltas: the bf16 qkv/o profile times are warm.
- Evidence: decode's ported `_swapab_qkv_kernel` ran 12.7–13.0 µs cold, against 9.6 µs in the bf16 profile (b0057d2e).

**Headline value of 1 µs on both layers of a load (decode)**

| load | %/µs |
|---|---|
| dec_1 | 0.139 |
| ver_1 | 0.091 |
| dec_16 | 0.072 |
| dec_32 | 0.057 |
| ver_16 | 0.036 |
| ver_32 | 0.025 |

| T | decode today (in-profile µs) | bf16 winners decode lacks | est. saving / shape | headline |
|---|---|---|---|---|
| 1 | sliding: qkv nvjet 128x8 splitK 12.1 + reduce 1.8, o nvjet 64x8 7.0; full: `dl_qkv<1>` 13.3 (fused norm/RoPE/fp8/store), `dl_o<1>` 8.2 (includes PDL wait); P `dl_pooler_router<1>` 21.5; `dl_unpool` 6.1 | bulk A2 qkv (row 13.60→10.54); swapab o (7.10→6.32); bulk-style P (repacked 4608×8192, hi/lo pairs summed in-CTA, ≈19.5); ouns unpool (≈4.4–5.0) | qkv −3.1 (full: −2..−3 if `dl_qkv` is rebuilt on the bulk mainloop with rows permuted so RoPE pairs are CTA-local and the head norm uses a cluster-4 DSMEM sum), o −0.8, P −2, unpool −1.1..−1.7 → **≈ −7** | **≈ +1.0%** |
| 8 | qkv nvjet 128x8 splitK 12.8/13.7 + reduce; o nvjet 64x8 7.0/7.7; P<1> 22.5/23.8; `dl_unpool` 6.6 | swapab16 qkv/o (decode-measured −2 µs kernel sum; rows flat in noise); ouns m=8 (7.02→5.54) | −3.2..−4.3; P rewrite −3 more (speculative) | +0.3–0.4% (+0.27%) |
| 16 | qkv nvjet 64x16 splitK 12.1; o + unpool nvjet 64x16 26.3 (2 calls); P<2> 24.6 | swapab16 qkv (−1.2) / o (−0.7); unpool full at parity | −1.9; P: swapab-style wgmma ~22 (−2.5) / repacked ~19.6 (−5) | +0.14% (+0.18–0.36%) |
| 32 | qkv nvjet 64x32 splitK 12.6; o + unpool 27.1; P<4> 26.6/26.8 (2.84 TB/s) | swapab qkv/o/unpool −0.45/−0.41/−0.23 | −1.1; P −4.4..−7 | +0.06% (+0.25–0.4%) |
| 128 | qkv nvjet 96x64 splitK 14.6 + reduce; o + unpool 128x64 TNT 28.5 (2 calls); pooler 64x64 TNT 26.2/22.8; router: SGLang 2× tss splitK 10.7 + 2× reduce + add on the side stream; `vectorized_layer_norm` 6.2 | Lt qkv plan (−1.4), Lt NN o (−0.4), NN unpool (−1.1), NN/K-halves pooler (−1.5..−4.8; K-halves has a rules risk), hilo router (row −8.7, on-path share unknown) | −4.4..−7.7 (+ router 0..−8); custom §4 kernel −12..−15 | +0.16–0.28% (+0–0.3%); custom +0.45% |
| 256 | qkv 96x64 22.8 (19.5 in bf16: clock/power?); o + unpool 34.4; pooler 128x64 TNT 30.8; router tss 13.2 ×2 | NN/K-halves pooler (−2.8..−4), hilo router | −2.8..−4 (+ router); custom §4 kernel −15..−18 | +0.07–0.1%; custom +0.4–0.45% |

**Decode-specific fusions (estimates)**
1. **Router + pooler sharing x at T≥128.**
   - Append the 512 hi/lo rows as 4 extra weight tiles (36 tiles × split 4 = 144 > 132, so use split 3 or stream-K balance).
   - Sum hi+lo in the epilogue, using the interleave decode's P kernel already uses: `[hi16|lo16|hi16|lo16]`.
   - This removes about 5 side-stream nodes at the cost of +12.5% weight bytes (≈ +2 µs).
   - The queued `ffn-front-t128-t256-concat-route` does this through a cuBLAS fp32-output GEMM. Caution: cuBLAS fp32-output tss kernels were about 2.7 µs slower in K-halves candidate b.
2. **Pre-expert RMSNorm at T=128/256** (`vectorized_layer_norm`, 6.2 µs): fold it into kernel R-style routing as already done at T≤32, or into a second counter phase of the pooler epilogue. About −4 µs, worth +0.1–0.15%.
3. **Boundaries at T≤8** (`muse_spark_norm_residual` 3.4–3.7 µs, two per layer):
   - attn_boundary into the o epilogue, or redundantly into the P prologue (each P CTA recomputes both row norms from a and s, about 48 KB of extra L2 reads per token).
   - ffn_boundary into the unpool epilogue. Only 1024 columns are nonzero, so this needs a two-phase grid reduction over 128 co-resident CTAs and a self-resetting counter.
   - About −5 µs at T=1 (+0.7%) and −4 at T=8 (+0.36%). Speculative: LESSONS says fusion pays only when it removes nodes into a PDL chain of the problem's own kernels.
4. **PDL weight prefetch before `griddepcontrol.wait`.**
   - P already prefetches 3 stages (10 MB of 75.5 MB).
   - Extend it to the full smem ring plus `cp.async.bulk.prefetch.L2`. This needs a decode-owned predecessor that triggers `launch_dependents` early, which is another reason for fold 3.
   - Restrict L2 warm-up to T=1: the Triton side-stream warm-up regressed 10–19% at T≥8, where attention is bandwidth-bound (96ee63e1).

## 7. Risks and open questions

**Risks**
- **K-halves precision.** Both winning K-halves runs used bf16 partial sums (rel L2 2.45e-3 vs 1.66e-3), which is lower precision than SGLang's fp32 accumulation. The rules-safe fp32-partials candidate lost every gate. Promoting de95f817 or caaa76cc carries review risk.
- **Prepare gates are biased.** They flush L2 with writes (`flush.zero_()`, `submission.py` L2666/2808/1488/1527; decode L4913/6596), while the judge reads.
  - ≥37 MB streams are inflated by 2–4.8 µs: pooler m=128 master 27.3 µs in prepare vs a 22.85 row; qkv m=256 23.4 vs 19.46.
  - The dirty-L2 write-back explanation is likely but unproven: inflation tracks stream size, but o m=128 (16.8 MB) showed none.
  - This compresses ratios toward 1 and is a plausible cause of the qkv m=256 flips.
  - Fix: flush with a read reduction.
- **Spin-wait reductions** need grid ≤132 at 1 CTA/SM and no early `launch_dependents`, or PDL dependents can deadlock.
- **Two judge state copies** need per-state workspaces, counters and TMA descriptors (moe `g_maps` lesson).
- **Clusters >2** at about 200 KB smem fit only 120 SMs.
- **Build and faults.**
  - Build requirements:
    - sm_90a only: `TORCH_CUDA_ARCH_LIST=9.0a`.
    - `-std=c++20`.
    - `#error` on `!__CUDA_ARCH_FEAT_SM90_ALL`.
    - CUTLASS compiles in about 49 s; the bf16 run limit is 10 min.
  - A fault poisons the CUDA context and child processes are banned. Use canaries and `-lineinfo`, and make the first measurement a check.
- **Nondeterministic CUTLASS fixup** fails bit-identical self-checks and gives up determinism.
- **Drift.** Rows drift ±1.5–2.5%, and cuBLAS m≥128 rows swing more (pooler m=128 read 23.9 vs 25.7 across runs). Read paired ratios plus kernel names.
- **Decode clocks and power.** qkv m=256 runs 22.8 µs in decode vs 19.5 standalone, so compute-bound gains may shrink.
- **Final review uses nearby shapes** (m=130/200/512/1000): every m-band gate must stay correct and fast there.

**Open questions**
1. Is the bulk kernel's 3.97 TB/s due to the contiguous repack or to the reduce-free row ownership? This is one A/B, and it decides the §5 design.
2. Does Nondeterministic SplitK, or no memset node, recover the 10–32 µs? How much of it is the `initialize_workspace` memset? The archived logs are 16 KB tails; the pooler and unpooler diagnostic lines were cut.
3. What is the true L2→SM ceiling on the judge (≥8 TB/s implied) and the sustained SM clock under 128-SM wgmma load? Together these set the m=256 floor at about 750–850 TFLOP/s.
4. At T≥128 in decode, is the side-stream router on the critical path? The verify profiles show no overlap marker.
5. Why are decode's in-profile GEMMs slower than bf16's on identical kernels (T=128 qkv 14.6 vs 12.7)? Clock, L2 state, or PDL-wait accounting?
6. Does decode's P reach the bf16 swapab rate (about 3.4 TB/s) just by switching mma.sync to wgmma and BLOCK_K 128 to 256 (75.5 MB in ~22 µs vs 24.6–26.8 µs today)?
7. Prefill is out of scope but bounded: large-T nvjet runs at about 740–820 TFLOP/s, so there is no headroom. At T=753 only o (112x128, ~325 TF/s, 38.8 µs) and the router (tss splitK, ~250 TF/s, 25.1 µs) are weak.

## Key evidence files

- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify32-cutlass-coop-256x128-splitk-stage2/92c42836caf54987.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify16-cutlass-m128-onetile-splitk-rewrite/d4edd38aeed44334.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify-streamk-cutlass-wgmma-rewrite/f5b1361aaa0e48b5.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify-swapab-wgmma-cluster-dsmem-rewrite/3a7f42bc7f544886.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2/566a79e403754c76.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify-deepgemm-bf16-gemm-nt-route/2d522d47ca5c4fc9.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/router-m128-m256-wgmma-swapab-tall-cluster-stage2/38daa28c6a704810.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/qkv-rowowner-a2-widen-m8-m16/0124ea8fcaf14b3c.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole/37ed8722bc9b402d.log`
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/archive/pooler-k-halves-fp32-partials-restack-m128-m256/de95f81782374124.log` and the `-cf/caaa76cc288d47c7.log` confirm run
- `/home/user/git/musespark/muse_bf16_proj/hypotheses/done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch` (L173, `RM::Deterministic`)
- `/home/user/git/musespark/muse_bf16_proj/profiles/baseline.1ccc6ba13da94194.txt`
- `/home/user/git/musespark/muse_decode_layer/hypotheses/archive/qkv-o-swapab16-t2-t16-no-fork/b0057d2e6cef4d5a.profile.txt`
- `/home/user/git/musespark/muse_decode_layer/submission.py` (`dl_pooler_router_kernel` L395–545)
- `/home/user/git/musespark/hypotheses/LESSONS.md`
