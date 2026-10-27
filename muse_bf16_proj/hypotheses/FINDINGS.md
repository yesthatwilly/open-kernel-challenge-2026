# muse_bf16_proj findings

Regenerated 2026-10-27T111650Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Current best

1.2735x [1.2544,1.2926] from job `bc147c5b87bc465d` (submit), master commit `8f233e88d4`, hypothesis pooler-khalf-m128-fp32-partials-triton-castadd

Today's clean-context diagnosis: `muse_bf16_proj/hypotheses/DIAGNOSIS.md` (ranked actions and its own Do-not-try list).

## By area

### bf16 GEMM dispatch
- **sglang-gemm-dispatch-parity** → gain (gain), 1.0022x — gained on router decode_16, decode_32, verify_16, verify_32; o verify_32. Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], 
- **bf16-cublaslt-backend-toggle** → neutral (within-ci), 1.1505x — gained on qkv_gate decode_1, verify_16; pooler decode_32; lost on o decode_1, verify_1; unpooler decode_1, verify_1; router verify_1; qkv_gate verify_32. A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') imme

### bf16 GEMM dispatch (m<=16, swap-AB Triton)
- **bf16-m16-swapab-wgmma-extend** → neutral (unproven-vs-best), 1.2115x — gained on qkv_gate decode_1; o decode_1, decode_16, verify_1; pooler decode_1; unpooler decode_1, verify_1; lost on router decode_16. Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kerne

### bf16 GEMM dispatch (m<=32, lossless-compressed weight stream)
- **bf16-lossless-exp4-weight-stream-rewrite** → regression (regression), 0.7071x — gained on router decode_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1, verify_16; pooler decode_1, decode_16, decode_32, verify_1; unpooler decode_1, decode_16, decode_32, verify_1; router decode_1, verify_16. Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: every m<=32 bf16 shape is a pure weight stream (FINDINGS: pooler 19.0-20.5 us kernel

### bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream)
- **bf16-small-m-cuda-gemv-rewrite** → regression (regression), 1.1667x — gained on pooler verify_16; lost on o decode_1, verify_1; router decode_1; qkv_gate verify_1; pooler verify_1; unpooler verify_1. Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; at

### bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection)
- **bf16-cublaslt-algo-pick-in-prepare-m128-m256** → neutral (within-ci), 1.1879x — lost on qkv_gate decode_1, decode_32, verify_1, verify_32; router verify_1. Built on master (658d63b). For the four bf16 layers at exactly cfg m == 128 and cfg m == 256 (loads verify_16 and verify_32; 8 shapes), replace _bf16_gemm_dispa

### bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector)
- **bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole** → neutral (within-ci), 1.2525x — gained on unpooler verify_1; router verify_1, verify_32; lost on o decode_16; router decode_16; qkv_gate verify_32. DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise master's cell-B machinery (_LT_CPP candidates/run with MuseLtDesc(n, k, m

### bf16 GEMM dispatch (m=129..256, CUTLASS sm90 cooperative split-K)
- **bf16-verify32-cutlass-coop-256x128-splitk-stage2** → neutral (within-ci), 1.2483x — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Stage 2 of bf16-verify-streamk-cutlass-wgmma-rewrite (job f5b1361aaa0e48b5), but ported onto CURRENT master (blob 353e3b19, restack-three-cell-wins-router-unpoo

### bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma)
- **bf16-verify-streamk-cutlass-wgmma-rewrite** → regression (regression), 1.1635x — gained on qkv_gate decode_16, decode_32; router decode_16, decode_32, verify_1; o decode_32; pooler verify_16; lost on qkv_gate decode_1, verify_1, verify_32; o verify_32; pooler verify_32; unpooler verify_32. Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: at m=256 (verify_32, the load furthest from its floor: 92.9 us vs 51.8 floor) all fo

### bf16 GEMM dispatch (m=17..32, swap-AB Triton)
- **bf16-m32-swapab-wgmma-triton** → neutral (unproven-vs-best), 1.1984x — gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_32, verify_1, verify_16; pooler decode_32, verify_1; router verify_1, verify_16. Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-

### bf16 GEMM dispatch (m=65..128, CUTLASS sm90 one-M-tile split-K)
- **bf16-verify16-cutlass-m128-onetile-splitk-rewrite** → neutral (within-ci), 1.2423x — gained on router verify_1; lost on qkv_gate decode_1, decode_16, verify_32; o verify_1, verify_16; pooler verify_32. Built on current master (blob 353e3b19). This is a new line for the m=128 band, which the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (m=129..256 on

### bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)
- **bf16-verify-deepgemm-bf16-gemm-nt-route** → regression (regression), 1.2027x — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; router decode_16, decode_32. Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=1

### bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
- **bf16-verify-deepgemm-sm90-kernel-asserted-rewrite** → neutral (unproven-vs-best), 1.2221x — gained on unpooler verify_1; router verify_1; lost on router decode_16, verify_16; o decode_32. Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT 
- **bf16-verify-deepgemm-sm90-explicit-instantiation-stage2** → neutral (unproven-vs-best), 1.2606x — gained on qkv_gate decode_1, decode_32, verify_16; pooler decode_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32; o decode_32; lost on o verify_16, verify_32. DIAGNOSIS Action 5. Built on master 36c26199, porting stage 1's [dg] scaffold (patch done/neutral/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite.patch: TORCH

### bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K)
- **bf16-verify-swapab-wgmma-cluster-dsmem-rewrite** → neutral (within-ci), 1.2155x — gained on router decode_32, verify_32; lost on o decode_1, decode_16, decode_32. Built on master (blob a8bb42a). WHY: the m=128/256 rows are the last ones on cuBLAS and the furthest from their floors in absolute us (master profile 1602a4e947

### bf16 GEMM dispatch (m>=128, weight layout)
- **bf16-m128-nn-weight-layout-probe** → neutral (unproven-vs-best), 1.1580x — gained on pooler verify_16; unpooler verify_16; lost on o decode_1; router verify_1, verify_16; qkv_gate verify_32. Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 

### bf16 GEMM dispatch (shape-gated Triton routing, m<=16)
- **bf16-shape-gated-triton-winners-combo** → neutral (unproven-vs-best), 1.1760x — gained on qkv_gate decode_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neu

### combination (naive stack + bulk row-owner widened to qkv_gate/o m=1)
- **combo-rowowner-reduce-free-m1-qkv-o-plus-naive** → neutral (unproven-vs-best), 1.2566x — gained on qkv_gate decode_1, verify_16; pooler decode_1, verify_16. Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gate

### combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node)
- **combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt** → neutral (within-ci), 1.2448x — gained on pooler decode_1; router decode_32; qkv_gate verify_16; lost on o decode_16. Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins + pooler-nn-layout-widen-m256). NAIVE TOP-ROW STACK: the only two cells where a

### lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)
- **bf16-lossless-exp4-occupancy-dequant-once-stage2** → neutral (beats-base-not-best), 0.8870x — gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1; unpooler decode_1, decode_16, decode_32, verify_1; router verify_1. Build on the stage-1 code of bf16-lossless-exp4-weight-stream-rewrite (job 542f2b169c934706; codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-ro
- **bf16-lossless-exp4-decode-only-throughput-probe** → regression (regression), 1.1904x — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1. A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn routes NOTHING new (byte-identical master dispatch, so the headline shoul

### o GEMM
- **o-triton-skinny-m32** → neutral (within-ci), 1.0057x — gained on qkv_gate decode_1, decode_16; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1; lost on o decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=8192, k=1024, 16.8 MB): in kernel_fn route o at m <= 32 to the same ke

### o GEMM (m=17..32, swap-AB)
- **o-swapab-m32-splitk2-two-ctas-per-sm** → regression (regression), 1.1860x — lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (1

### o GEMM + unpooler GEMM (latency-bound small streams)
- **o-unpooler-slice-cuda-cluster-dsmem-rewrite** → regression (regression), 1.1710x — gained on unpooler decode_1, verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB, floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are th

### pooler GEMM
- **pooler-triton-skinny-m32** → neutral (within-ci), 0.9983x — gained on pooler decode_1, decode_16, verify_1, verify_16; lost on qkv_gate decode_1; unpooler decode_1; pooler decode_32. For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams t
- **pooler-triton-splitk2-blockn64-semaphore** → neutral (unproven-vs-best), 1.1664x — gained on pooler decode_1, decode_16, verify_1, verify_16; qkv_gate decode_32; lost on router verify_1, verify_16; o verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf
- **pooler-triton-splitk2-blockn64-semaphore-control-blockk256** → neutral (within-ci), 1.1620x — gained on pooler decode_1; lost on qkv_gate decode_16, verify_1; router verify_1, verify_16, verify_32. From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each C
- **pooler-splitk2-tma-weight-stream** → neutral (unproven-vs-best), 1.1693x — gained on pooler decode_1, decode_16, decode_32, verify_1; router decode_16; lost on qkv_gate verify_16. From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32: _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2,

### pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)
- **pooler-k-halves-fp32-partials-restack-m128-m256** → neutral (unproven-vs-best), 1.2683x — gained on unpooler decode_1; router decode_1, verify_32; qkv_gate decode_16, decode_32; pooler verify_16; lost on router decode_32. DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 (done/n
- **pooler-k-halves-fp32-partials-restack-m128-m256-cf** → neutral (within-ci), 1.2590x — gained on o decode_32; router decode_32, verify_1; pooler verify_32; lost on o decode_1; unpooler decode_1; router decode_1; qkv_gate decode_16, decode_32, verify_16. Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one t

### pooler GEMM (m=129..256, cuBLAS NN layout)
- **pooler-nn-layout-widen-m256** → neutral (within-ci), 1.2394x — gained on unpooler decode_32; pooler verify_32; lost on router decode_32, verify_1; unpooler verify_1. Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = 

### pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)
- **pooler-k-halves-batched-gemm-m65-m256** → neutral (within-ci), 1.2470x — gained on qkv_gate decode_1, decode_16; router verify_1; pooler verify_16, verify_32; lost on router decode_16. Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only (scored rows verify_16 m=128 and verify_32 m=256; no test uses this ba

### pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel)
- **pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite** → regression (regression), 1.1828x — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32. Built on master (blob a8bb42a). WHY: pooler (n=4096, k=8192) and the full unpooler (n=8192, k=4096) are 67.1 MB pure weight streams at m<=32 with a 14.0-14.1 us
- **pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild** → regression (regression), 1.0331x — gained on qkv_gate decode_1; o decode_1; pooler decode_1; lost on pooler decode_16, decode_32; unpooler decode_16, decode_32, verify_1; router verify_1, verify_16; o verify_16. Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built on its commit (8ba8624, claim bases a rewrite-on-rewrite on the p

### pooler GEMM + unpooler GEMM (m=128)
- **pooler-unpooler-m128-wgmma-splitk-semaphore** → regression (regression), 1.1158x — gained on router decode_16; lost on qkv_gate verify_1; o verify_1; unpooler verify_1, verify_16; router verify_1, verify_16; pooler verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 
- **pooler-unpooler-m128-tma-wspec-rewrite** → regression (regression), 1.1639x — gained on router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; pooler verify_16; unpooler verify_16; router verify_16. Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler} at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m ma

### pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
- **pooler-unpooler-m128-nn-layout-gated** → regression (regression), 1.1953x — gained on pooler verify_16; unpooler verify_16; router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1; pooler decode_16. Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master) to the only rows it won. Measured per
- **restack-pooler-unpooler-nn-layout-m128** → neutral (unproven-vs-best), 1.2282x — gained on unpooler decode_1, verify_16; router verify_1; pooler verify_16; lost on router decode_16, verify_16; o decode_32. Built on master (b99b1dd, the landed restack). Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the re

### qkv GEMM
- **qkv-gate-triton-skinny-m32** → regression (regression), 0.9238x — lost on qkv_gate decode_1, decode_16, decode_32, verify_1. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32 (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at
- **qkv-gate-triton-splitk4-blockn64-semaphore** → neutral (within-ci), 1.1431x — gained on qkv_gate decode_1, decode_16, verify_1; router decode_16; pooler verify_16; lost on qkv_gate decode_32; o verify_1; unpooler verify_1; router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvje

### qkv GEMM (m<=16, swap-AB Triton)
- **qkv-swapab16-five-stage-ring-m8-m16** → regression (regression), 1.2507x — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master blob 2bcfe7d4. qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf16_swapab_kernel through _swapab16_forwa

### qkv GEMM (m<=32, swap-AB)
- **qkv-swapab-w128-splitk7-one-wave-m32** → neutral (within-ci), 1.2229x — gained on o verify_32; lost on router decode_16, decode_32, verify_1; qkv_gate decode_32, verify_1; o decode_32. Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, check 84f17adc3f074a33, 1.2366x). WHY: at m<=32 qkv_gate (n=2304, k=

### qkv GEMM (m=17..32, swap-AB)
- **qkv-swapab-m32-splitk8-stages3** → regression (regression), 1.1963x — gained on pooler verify_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 st

### qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)
- **qkv-k-split-batched-gemm-m65-m256** → regression (regression), 1.2410x — gained on unpooler verify_1; router verify_1; lost on qkv_gate decode_1, decode_16, verify_32; o decode_16; router decode_16, decode_32; pooler verify_32. Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only, which covers the scored rows verify_16 m=128 and verify_32 m=256. T

### qkv GEMM (m=8/16, bulk row-owner kernel)
- **qkv-rowowner-a2-widen-m8-m16** → neutral (unproven-vs-best), 1.2603x — gained on router decode_1, verify_1, verify_32; unpooler verify_1; lost on router decode_16. DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; de

### restack (m<=16 swap-AB GEMMs + router swap-AB)
- **bf16-restack-m16-swapab-router-swapab-m32** → neutral (unproven-vs-best), 1.2215x — gained on o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1. Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma

### router GEMM
- **router-hilo-at-m8** → gain (gain), 1.0219x — gained on unpooler verify_1; router verify_1. In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX
- **router-hilo-addmm-epilogue** → neutral (within-ci), 1.0256x — gained on o verify_16; pooler verify_32. In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc +=
- **router-triton-hilo-fused** → regression (regression), 0.9402x — gained on router decode_1, decode_16, decode_32, verify_1; qkv_gate verify_1; o verify_1, verify_16; unpooler verify_1; lost on router verify_16, verify_32. Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch 
- **router-triton-hilo-fused-control-splitk-all-m** → gain (gain), 1.1510x — gained on router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; lost on qkv_gate decode_1. Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Tr
- **router-splitk32-small-m-all-loads-in-flight** → neutral (within-ci), 1.1427x — gained on pooler verify_16; lost on router decode_1, decode_16, decode_32, verify_1. Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + 
- **router-splitk-semaphore-no-memset** → neutral (unproven-vs-best), 1.1516x — gained on qkv_gate decode_1, verify_1; pooler verify_16; lost on router decode_1, decode_16, decode_32, verify_16. Built on master (dba4ffd). For the router at m <= 32 only (buckets (16,32,16) and (32,32,16): grid (1, 8, 16) = 128 CTAs, unchanged tiles, BLOCK_K=128, num_stag
- **router-splitk32-small-m-all-loads-in-flight-control-stages3** → neutral (unproven-vs-best), 1.1518x — gained on unpooler decode_1; router decode_1, decode_16, decode_32, verify_1; qkv_gate decode_16; lost on o verify_16; pooler verify_16; router verify_16. From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPL
- **router-swapab-wgmma-all-m** → neutral (within-ci), 1.1951x — gained on router decode_1, decode_16, decode_32, verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router verify_16. Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M
- **router-cuda-cluster-dsmem-splitk-rewrite** → regression (regression), 1.1897x — gained on router decode_1, verify_1; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1, verify_16, verify_32; unpooler decode_1, verify_1; router decode_16, decode_32, verify_16. Built on master (8b9ad43, blob a8bb42a); the restack is not landed and is not needed. WHY: router rows at m<=32 are 5.4-6.2 us against a 1.8-1.9 us floor (8.4 M

### router GEMM (m<=32, cluster DSMEM kernel)
- **router-cluster-dsmem-s2-flat-in-m-occupancy** → neutral (within-ci), 1.2058x — gained on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1, verify_1; router decode_32; lost on router verify_1; pooler verify_16. Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4). Stage 1 won rows at m=1/8 only by deleting the memset node: its

### router GEMM (m<=8)
- **restack-router-cluster-dsmem-m8** → neutral (within-ci), 1.2199x — gained on router decode_1, verify_1; qkv_gate decode_16; lost on o decode_32, verify_16; router verify_16. Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel fr

### router GEMM (m=65..256)
- **router-m128-m256-cluster-dsmem-tiled-rewrite** → neutral (within-ci), 1.2189x — gained on unpooler verify_1; lost on o decode_32. Built on master (the router at m > 32 is identical on master and the restack). Router rows at m=128/256 are 7.3/9.6 us against floors of 2.2/2.7 (3.3-3.6x, the 

### router GEMM (m=65..256, CUDA wgmma swap-AB cluster kernel)
- **router-m128-m256-wgmma-swapab-tall-cluster-stage2** → neutral (unproven-vs-best), 1.2612x — gained on qkv_gate decode_1, decode_16, decode_32, verify_1, verify_16; pooler decode_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1; lost on router verify_32. Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port its load_inline scaffold, self-check, cold-L2 timing table and PAT

### router GEMM (m=65..256, hilo tile config)
- **router-hilo-tall-tile-m65-m256** → neutral (within-ci), 1.2539x — gained on unpooler decode_1; router decode_1, verify_1; lost on router decode_16, verify_32. Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound

### shape-gated restack: router GEMM m<=8 + unpooler-slice GEMM m<=8 + pooler/unpooler GEMM m=65..128
- **restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128** → neutral (unproven-vs-best), 1.2366x — gained on unpooler decode_1, verify_1; router decode_1, decode_32, verify_1; pooler verify_16; lost on o verify_32. Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check 1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked al

### unpooler GEMM
- **unpooler-slice-triton-gemv** → regression (regression), 0.8801x — gained on o verify_1; lost on unpooler decode_1, verify_1. For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call i
- **unpooler-full-triton-skinny-m32** → neutral (within-ci), 1.0053x — gained on qkv_gate decode_1, decode_16, verify_1; unpooler decode_16, verify_1; o verify_1; lost on unpooler decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** → neutral (within-ci), 0.9990x — gained on o verify_16; lost on qkv_gate decode_16; unpooler decode_32. From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num
- **unpooler-slice-triton-splitk8-semaphore** → neutral (unproven-vs-best), 1.1546x — gained on unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only,
- **unpooler-full-m16-swapab-route** → regression (regression), 1.1885x — gained on router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; pooler decode_1, decode_16, verify_1; router decode_1, decode_16, decode_32, verify_1; unpooler verify_1. Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (c

### unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)
- **unpooler-k-halves-batched-gemm-m65-m128** → neutral (within-ci), 1.2452x — gained on unpooler verify_1; router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router decode_16; pooler verify_16. Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=1

### unpooler GEMM (sliced n=1024, m<=8)
- **restack-unpooler-slice-cluster-dsmem-m8** → neutral (unproven-vs-best), 1.2260x — gained on unpooler decode_1, verify_1; router verify_1; lost on o verify_16; router verify_16. Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-

### unpooler GEMM (sliced n=1024, m=1, bulk12 12-bit row-owner)
- **b12-unpooler-slice-m1-r16-rowowner** → neutral (unproven-vs-best), 1.2737x — gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024 rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. C

## Open follow-ups

- bf16-lossless-exp4-occupancy-dequant-once-stage2: rewrite stage 2 of `Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32`: read the replaced kernels' own us per shape (profile top_kernels) against the inc
- bf16-verify-swapab-wgmma-cluster-dsmem-rewrite: rewrite stage 1 of `cuBLAS nvjet at m=128/256 for qkv_gate (13.0+3.0 splitKreduce / 19.4 us), pooler (25.7 / 29.5 us) and unpooler m=128 (20.4 us)`: read the replaced kernels' own us per shape (profile top_kernels) again
- bf16-verify-deepgemm-sm90-kernel-asserted-rewrite: rewrite stage 1 of `cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check whi
- router-cluster-dsmem-s2-flat-in-m-occupancy: rewrite stage 2 of `_router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor
- router-m128-m256-cluster-dsmem-tiled-rewrite: rewrite stage 1 of `_router_hilo_kernel + torch.zeros memset at router m=65..256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reach
- bf16-verify32-cutlass-coop-256x128-splitk-stage2: rewrite stage 2 of `cuBLAS nvjet at m=129..256 (verify_32) for qkv_gate (nvjet 96x64 2x4 TNN 19.4 us), o (nvjet 128x128 TNT 8.1 us), pooler (nvjet 128x64 TNT 29.7 us) and unpooler (nvjet 128x128 TNT 24.8 us)`: read the r
- bf16-verify16-cutlass-m128-onetile-splitk-rewrite: rewrite stage 1 of `cuBLAS at m=65..128 (verify_16): qkv_gate nvjet 96x64 splitK 12.7 + cublasLt splitKreduce 3.1 us, o nvjet 128x64 TNT 5.5 us, pooler NN nvjet 128x64 splitK 20.2 + splitKreduce 3.0 us, unpooler NN nvjet
- router-m128-m256-wgmma-swapab-tall-cluster-stage2: rewrite stage 2 of `_router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbe
- bf16-verify-deepgemm-sm90-explicit-instantiation-stage2: rewrite stage 2 of `cuBLAS nvjet (+splitKreduce) for qkv_gate, pooler, unpooler at m=128/256`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milest

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end.

- **router-hilo-addmm-epilogue** [router GEMM] neutral, within-ci: In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accum
- **pooler-triton-skinny-m32** [pooler GEMM] neutral, within-ci: For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kerne
- **bf16-cublaslt-backend-toggle** [bf16 GEMM dispatch] neutral, within-ci: A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers 
- **o-triton-skinny-m32** [o GEMM] neutral, within-ci: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=
- **unpooler-full-triton-skinny-m32** [unpooler GEMM] neutral, within-ci: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** [unpooler GEMM] neutral, within-ci: From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, gri
- **qkv-gate-triton-splitk4-blockn64-semaphore** [qkv GEMM] neutral, within-ci: For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16
- **router-splitk32-small-m-all-loads-in-flight** [router GEMM] neutral, within-ci: Change only the two small-m buckets of _router_config and the num_stages they launch with;
- **pooler-triton-splitk2-blockn64-semaphore-control-blockk256** [pooler GEMM] neutral, within-ci: From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one co
- **bf16-cublaslt-algo-pick-in-prepare-m128-m256** [bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection)] neutral, within-ci: For the four bf16 layers at exactly cfg m == 128 and cfg m == 256 (loads verify_16 and ver
- **router-swapab-wgmma-all-m** [router GEMM] neutral, within-ci: Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, B
- **restack-router-cluster-dsmem-m8** [router GEMM (m<=8)] neutral, within-ci: 2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch hyp/muse_b
- **qkv-swapab-w128-splitk7-one-wave-m32** [qkv GEMM (m<=32, swap-AB)] neutral, within-ci: 2366x). WHY: at m<=32 qkv_gate (n=2304, k=8192, a 37.7 MB weight, cold L2 every replay) is
- **pooler-nn-layout-widen-m256** [pooler GEMM (m=129..256, cuBLAS NN layout)] neutral, within-ci: 2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] 
- **pooler-k-halves-batched-gemm-m65-m256** [pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)] neutral, within-ci: Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only 
- **router-hilo-tall-tile-m65-m256** [router GEMM (m=65..256, hilo tile config)] neutral, within-ci: Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo
- **unpooler-k-halves-batched-gemm-m65-m128** [unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)] neutral, within-ci: Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8
- **combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt** [combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node)] neutral, within-ci: Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins + pooler-
- **pooler-k-halves-fp32-partials-restack-m128-m256** [pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)] neutral, not confirmed (pooled 1.2636332647989572): DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from b
- **bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole** [bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector)] neutral, within-ci: DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise mas
- **router-triton-hilo-fused** [router GEMM] regression, regression: Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the 
- **unpooler-slice-triton-gemv** [unpooler GEMM] regression, regression: For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight
- **qkv-gate-triton-skinny-m32** [qkv GEMM] regression, regression: Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <=
- **pooler-unpooler-m128-wgmma-splitk-semaphore** [pooler GEMM + unpooler GEMM (m=128)] regression, regression: For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 65 <= m <= 128 may t
- **o-swapab-m32-splitk2-two-ctas-per-sm** [o GEMM (m=17..32, swap-AB)] regression, regression: o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BL
- **qkv-swapab-m32-splitk8-stages3** [qkv GEMM (m=17..32, swap-AB)] regression, regression: qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192,
- **unpooler-full-m16-swapab-route** [unpooler GEMM] regression, regression: On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through t
- **pooler-unpooler-m128-nn-layout-gated** [pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)] regression, regression: Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master)
- **bf16-lossless-exp4-decode-only-throughput-probe** [lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)] regression, regression: A one-run kill/revive gate for the held lossless line, not a pipeline stage: kernel_fn rou
- **bf16-verify-deepgemm-bf16-gemm-nt-route** [bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)] regression, regression: At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master p
- **qkv-k-split-batched-gemm-m65-m256** [qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)] regression, regression: Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 onl
- **qkv-swapab16-five-stage-ring-m8-m16** [qkv GEMM (m<=16, swap-AB Triton)] regression, regression: qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf1

## Queue and in flight

- queue: combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16 [combination (pooler K-halves m=128/256 + qkv row-owner m=8/16 + per-cell Lt plans m=128/256)] p2 hold
- queue: pooler-khalf-m128-fp32-partials-triton-castadd [pooler GEMM (m=128, K-split as a 2-batch cuBLAS GEMM)] p1 hold
- queue: b12-qkv-m1-r32-one-cta-per-sm-readflush-gate [qkv GEMM (m=1, bulk12 12-bit row-owner CTA shape)] p1 hold
- queue: lt-nn-layout-cells-o-m128-qkv-m128 [o GEMM + qkv GEMM (m=128, cuBLASLt plan on an NN weight copy)] p2 queued
- queue: lt-config-sweep-m256-beyond-heuristic [bf16 GEMM dispatch (m=256, cuBLASLt configs beyond the heuristic)] p3 queued
- 8 stale holds expired to `done/expired/` (not run)
