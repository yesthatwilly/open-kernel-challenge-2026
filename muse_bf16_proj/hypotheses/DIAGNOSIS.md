---
diagnosed_at: 2026-10-25T203424Z
problem: muse_bf16_proj
best: 1.2566x [1.2378,1.2755], job 741072dca5fc4c9d (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, scored, master 36c26199)
---

# muse_bf16_proj diagnosis

## Diagnosis

**Wrong bottleneck (#4).** The loads are weakest at m=128/256: verify_32 is 1.107x and verify_16 is 1.192x, against 1.33 / 1.22 / 1.19 / 1.55 for the other four. Those two loads hold the largest absolute gaps to their floors: 90.8 µs vs a 49.7 µs floor, and 72.6 vs 43.9. Four cuBLAS rows there still sit at SGLang parity: o 7.68 / 9.44, qkv m=256 19.46, unpooler m=256 24.18. Even so, today's last three runs went to m=1 cells.

**Local minimum and repetition (#1, #2).** The other main spend went to areas that are already settled:
- The m≤32 pooler/unpooler weight stream has been redesigned more than ten times. Every design lands at 3.4-3.95 TB/s, so pooler rows sit at 17.7-19.9 µs against a 14.0 floor. Slugs: pooler-triton-skinny-m32, pooler-triton-splitk2-blockn64-semaphore(-control-blockk256), pooler-splitk2-tma-weight-stream, bf16-small-m-cuda-gemv-rewrite, pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite / -s2, three bf16-lossless-exp4-* runs, unpooler-full-m16-swapab-route.
- Router m≤32 has been tuned seven times and holds at 5.2-5.9 µs. Slugs: router-splitk32-*, router-splitk-semaphore-no-memset, router-swapab-wgmma-all-m, router-cuda-cluster-dsmem-splitk-rewrite, router-cluster-dsmem-s2-flat-in-m-occupancy, restack-router-cluster-dsmem-m8.

**Build-churn wall (#3).** At m≥128, five of the custom-kernel runs never beat cuBLAS in the timed path:
- bf16-verify-deepgemm-bf16-gemm-nt-route: the API was bitwise cuBLAS.
- bf16-verify-deepgemm-sm90-kernel-asserted-rewrite: "header parse failed ... VOID RUN".
- bf16-verify-swapab-wgmma-cluster-dsmem-rewrite and router-m128-m256-cluster-dsmem-tiled-rewrite: each lost its own gate, so master ran.
- bf16-verify-streamk-cutlass-wgmma-rewrite: lost.

Every m≥128 win so far came from the lighter abstraction, i.e. cuBLAS reshapes:
- the NN layout (pooler m=128 −10%)
- K-halves bmm (pooler m=256 −10.7%)
- a single-node cuBLASLt plan (qkv m=128 −10.5%)

**Unstacked wins (#7).** pooler-k-halves-batched-gemm-m65-m256 won pooler verify_32 29.94→26.75 and verify_16 23.2→21.43. It was never landed, and master's submission.py has no K-halves path.

**Ignored recommendations (#6).** Three were never minted:
- The DeepGEMM sm90 stage 2. Its header `sm90_bf16_gemm.cuh` is present in the judge's include tree (log 4c224beb).
- A test of the qkv row-owner above m=1.
- A correction to LESSONS after the B-cell result.

## Strategy

**Pivot.** Move the spend from the m≤32 streams and the small-m router, which are at a 3.4-3.95 TB/s local minimum, to the m=128/256 band. There the cuBLAS incumbent runs at 455-711 TFLOP/s and 2.9 TB/s with 25-41 µs per load left to its floors. Start with per-cell cuBLAS reshapes behind the proven model selector, then move to a CUTLASS kernel for the remaining ceiling.

## Actions (priority ordered; /hypothesize mints these first)

1. **Restack pooler K-halves at m=128 and m=256.**
   - **What:** in kernel_fn, for layer == 'pooler' at cfg m ∈ {128, 256} only, port the `[khalf]` path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 onto master 36c26199. That path is bmm over two K-halves of the NN repack, then an add.
   - Put it behind master's `_cell_gate` model selector, which requires ≥1.02, plus the profiler kernel-name assertion.
   - Offer two candidates to the selector: (a) bf16 partials, as ran; (b) fp32 partials (`out_dtype=torch.float32`, add, cast). (b) keeps one rounding like cuBLAS's own split-K, so it is the rules-safe form. Route (a) only if (b) loses and a review clears it.
   - **Why:** job ce808e5a gave pooler m=256 as nvjet 128x128 NNT 24.7 + add 2.1 µs, row 26.75 against master's NNT 128x64 28.8 µs, row 27.98 (741072dc). Pooler m=128 was 20.1 + 1.5, row 21.43, against master's splitK NNT 20.8 + splitKreduce 3.0, row 22.85.
   - **Impact:** −1.2 µs on verify_32 (0.184 %/µs) and −1.4 µs on verify_16 (0.230 %/µs), about +0.5% headline.
   - **Prior:** pooler-k-halves-batched-gemm-m65-m256 measured on the 67b46a57 base. The only difference is the rebase onto the combo master and the fp32-partials candidate.

2. **One per-cell cuBLAS reshape selector for all eight m∈{128,256} cells.**
   - **What:** generalise master's B-cell machinery (`_lt_prepare`, `_cell_gate`, `_cell_kernels`) from qkv m=128 to the cells (qkv_gate, o, pooler, unpooler) × (128, 256). Per cell the candidates are:
     - master;
     - the top-N cuBLASLt heuristic plans at split-K 1 and at split-K >1 with an fp32 workspace;
     - for qkv/pooler/unpooler, a K-split bmm with S ∈ {2, 4} and fp32 partials (not o, whose K is only 1024);
     - NN vs TN weight layout.
   - Route a candidate only at ≥1.02 against master, in graph replay with a 256 MB flush, interleaved, with the empty-graph baseline subtracted. Print one PATH CHOSEN line per cell.
   - This one hypothesis subsumes three queue items: qkv-k-split-batched-gemm-m65-m256, unpooler-k-halves-batched-gemm-m65-m128 and the held combo-cublaslt-single-node-m128-m256-plus-naive.
   - **Why:**
     - The B cell's interleaved gate predicted 0.919, and the judge gave 0.895 (qkv m=128 17.38→15.55, 741072dc). The selector carries over.
     - The backend toggle (5ed9ace1) showed that Lt changes the kernel per cell: it won qkv m=128 (single node 16.24) and lost qkv m=256 (22.08 vs 19.5). Only per-cell choice pays.
     - Never-targeted rows: o m=128/256 at 7.68/9.44 (SGLang parity, floors 4.0/4.5, nvjet with 2.0 µs of row-minus-kernel on m=128) and qkv m=256 at 19.46, which is 497 TFLOP/s against a 9.8 floor.
   - **Impact:** a realistic −0.5 to −2 µs on each of the 6-7 cells, about +0.5-1.0% headline.
   - **Prior:** bf16-cublaslt-algo-pick-in-prepare-m128-m256. Its selector timed candidates without interleaving against the incumbent, used no empty-graph baseline and set no margin, and its qkv m=256 pick flipped in the judge.

3. **Dispatch the two queued CUTLASS rewrites against the post-1/2 master: bf16-verify16-cutlass-m128-onetile-splitk-rewrite, then bf16-verify32-cutlass-coop-256x128-splitk-stage2.**
   - **What:** as written, with one new requirement: the gate's incumbent is whatever actions 1/2 routed per cell.
   - **Why:** this is the ceiling argument.
     - At m=128 pooler/unpooler are memory-bound at 2.9-3.5 TB/s. One M-tile per CTA fetches W once; REWRITE_READ found that the Triton m=128 mainloops re-read the x K-slice at about 2x the weight bytes.
     - At m=256, stage 1's 128-row tiles fetched W twice (qkv 25.3, pooler 31.0 µs, f5b1361a).
     - At about 700 TFLOP/s / 4 TB/s, verify_32 would go 90.8→~78 µs and verify_16 72.6→~60 µs.
   - **Impact:** up to +2-4% headline. That is the largest ceiling left, and cuBLAS reshapes cannot reach it.
   - **Prior:** bf16-verify-streamk-cutlass-wgmma-rewrite (stage 1: scaffold proven, the tile shape was wrong).

4. **Widen the qkv_gate bulk row-owner (A2) to m=8 and m=16.**
   - **What:** gate qkv_gate at cfg m == 8 onto the existing `bulk_kernel<16, 8192, 8, 4, 2>`, which is the NT=8 instance already compiled. Add an NT=16 instance for m == 16. Each gets its own `_cell_gate` (≥1.02) and canary.
   - **Why:** A2 took qkv decode_1 from 12.67 to 10.54 µs (1.16x, 741072dc). verify_1 (12.35), decode_16 (12.64) and decode_32 (13.71) still run the SPLIT_K 4 ticket swap-AB. The m-linear loss of the row-owner design was measured only at 1 CTA/SM on the pooler (571ea532: +1 µs at m=8, +6 µs at m=16). A2 runs at 2 CTAs/SM.
   - **Impact:** if m=8 reaches ~11.0 µs, −1.3 µs at 0.343 %/µs (+0.45%). For m=16, ~+0.25%. The gate decides, so there is no downside risk.
   - **Prior:** the m=8 row of pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild (lost 5%). The difference is qkv's 2-CTA/SM instance and a split-K-4 incumbent.

5. **DeepGEMM sm90 BF16 stage 2: fix the kernel discovery.**
   - **What:** in the scaffold of bf16-verify-deepgemm-sm90-kernel-asserted-rewrite, replace the regex header parse that failed ("no __global__ template kernel found") with an explicit instantiation of the kernel named in `deep_gemm/include/.../sm90_bf16_gemm.cuh`. The header is present per log 4c224beb. Alternatively, call deep_gemm's own sm90 JIT entry, not `bf16_gemm_nt`, whose output was bitwise cuBLAS (2d522d47 STEP A). Assert the kernel name in the profiler and use the same per-cell selector.
   - **Why:** a tuned persistent TMA/wgmma kernel for m=128/256 that already exists on the judge image is cheaper than finishing a hand CUTLASS kernel. Both earlier runs measured nothing.
   - **Impact:** the same ceiling as action 3 if it runs. It is a fallback route to that ceiling.
   - **Prior:** bf16-verify-deepgemm-sm90-kernel-asserted-rewrite (VOID: the kernel never compiled).

The router at m≥128 (verify_16/32 rows 7.61/9.72 µs against floors 2.2/2.7) is already covered by router-hilo-tall-tile-m65-m256 (running) and router-m128-m256-wgmma-swapab-tall-cluster-stage2 (queued). Harvest those before minting more router work.

## Do not try

- `bf16-lossless-exp4-weight-stream-rewrite`, `-occupancy-dequant-once-stage2`, `-decode-only-throughput-probe`, and queued `bf16-lossless-exp4-warpspec-persistent-stage3`: the entropy decode alone runs at 1.74 TB/s compressed (1.021x the yardstick; revive needed ≤0.85). The kill gate missed by 2.1x (40.3 vs 19.0 µs). Retire stage 3.
- Retuning m≤32 pooler or full-unpooler streams (`pooler-triton-skinny-m32`, `pooler-triton-splitk2-blockn64-semaphore` and its `-control-blockk256`, `pooler-splitk2-tma-weight-stream`, `unpooler-full-triton-skinny-m32` and its control, `unpooler-full-m16-swapab-route`), plus the held `unpooler-full-m16-swapab16-blockt16`, `pooler-m32-tma-splitk-extend` and `pooler-skinny-tiled-weight-layout`. Every tile, stage, split or TMA variant lands at 3.4-3.95 TB/s, ±1 µs. Retire the held ones.
- The bulk row-owner on pooler/unpooler at m≥8 at 1 CTA/SM (`pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild`): per-token FMA is serialised behind one CTA per SM (+6/+25 µs at m=16/32). The judge saw +43%/+145%.
- Triton m=128 mainloops (`pooler-unpooler-m128-wgmma-splitk-semaphore`, `pooler-unpooler-m128-tma-wspec-rewrite`), plus the held `qkv-gate-o-m128-wgmma-splitk-semaphore`, which is the same design. They are stuck at 26-27.6 µs warm, above nvjet's whole kernel, and warp specialisation spills 110 registers. Retire the held one.
- The swap-AB wgmma + cluster-DSMEM design at m≥128 (`bf16-verify-swapab-wgmma-cluster-dsmem-rewrite`): its own no-reduce mainloop measured 42-94 µs against master's 24.6-37.7 µs in the cold-L2 gate, so the loss is in the mainloop, not the tail. Do not stage-2 it; the CUTLASS line (action 3) owns this band.
- `bf16-small-m-cuda-gemv-rewrite`: no target row beat the incumbent, and the newer Triton swap-AB beats it everywhere (REWRITE_READ §2).
- Router m≤32 retuning (`router-splitk32-small-m-all-loads-in-flight` and its stages3 control, `router-splitk-semaphore-no-memset`, `router-swapab-wgmma-all-m`, `router-cluster-dsmem-s2-flat-in-m-occupancy` at m=16/32): settled at 5.2-5.9 µs. The cluster kernel at m=16 was 5.64 against swap-AB's 5.51.
- Deeper split-K or two CTAs per SM on small-K / m≤32 swap-AB:
  - `o-swapab-m32-splitk2-two-ctas-per-sm`: +40.8%.
  - `qkv-swapab-m32-splitk8-stages3`: +14%, all of it in the reduce tail.
  - `qkv-swapab-w128-splitk7-one-wave-m32`: qkv m=32 lost 7%.
- Per-call `preferred_blas_library` toggles (`bf16-cublaslt-backend-toggle`) are forbidden by review. Use plans held in state, as the B cell does.
- FMA (no tl.dot) GEMV (`unpooler-slice-triton-gemv`, 10x slower). BLOCK_N=16 skinny tiles (`qkv-gate-triton-skinny-m32`). A full-K router grid of 8 CTAs (`router-triton-hilo-fused`).
- The held `router-triton-hilo-fused-control-hybrid-cublas-above-m32` and `router-hilo-addmm-epilogue-control-inplace-out` are superseded, because master's router has no cuBLAS path. Retire both.

## LESSONS edits (proposed; churnsubs applies them)

- **correct:** `cuBLAS vs cublasLt backend preference changes nothing on this image (bf16-cublaslt-backend-toggle).` → It changes the kernel per cell. qkv m=128 went from nvjet 96x64 splitK + splitKreduce (17.38) to one nvjet 48x64 node (16.24). qkv m=256 went to 72x64 (22.08 vs 19.5). The headline was flat because the cells cancelled (5ed9ace1). A per-cell plan in state pays (741072dc qkv m=128 −10.5%).
- **correct:** `Prepare-time algo picking is a dead knob.` (bf16-cublaslt-algo-pick-in-prepare-m128-m256) → A pick timed without interleaving against the incumbent, with no empty-graph baseline and no margin, did not carry over. The interleaved model selector with a ≥1.02 margin did: B cell predicted 0.919, judge 0.895 (741072dc).
- **correct:** `The naive stack (0ee20555, 1.2448) carried the same pooler/router wins but lost o decode_16 (0.966), and the cuBLASLt plans added nothing measurable.` → The Lt plan gained qkv verify_16 in both combos: 17.38→15.7 (0ee20555) and →15.55 (741072dc).
- **correct:** `A row-owner layout that serialises per-token FMA work behind one CTA per SM only suits m=1.` → Add: "measured on pooler/unpooler at 1 CTA/SM; the 2-CTA/SM qkv instance (bulk_kernel<16,8192,8,4,2>, 1.16x at m=1) is untested at m≥8".
- **merge:** the bf16 m≤32 stream lines (`Triton full-K weight streams: BLOCK_N=64, BLOCK_K=256, 4 stages ...`, `bf16 pooler at m<=32: TMA weight loads (1.169x) and BLOCK_K=256 cp.async (1.162x) ...`, `bf16 router: SPLIT_K 32 at m<=32 was flat ...`) → one line: "bf16 m≤32 DRAM weight streams cap at 3.4-3.95 TB/s across Triton, CUDA gemv, bulk-TMA row-owner and lossless designs; tile, stage, TMA and split knobs are settled; the remaining headroom is at m=128/256."
