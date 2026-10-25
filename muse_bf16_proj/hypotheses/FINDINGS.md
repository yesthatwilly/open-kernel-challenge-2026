# muse_bf16_proj findings

Regenerated 2026-10-25T043406Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Current best

1.2215x [1.2032,1.2398] from job `1e90292c7b954d96` (check), master commit `b99b1dd350`, hypothesis bf16-restack-m16-swapab-router-swapab-m32

Where the time goes (per scored shape, top kernels):

- layer=qkv_gate load=decode_1 m=1: yours 12.2µs vs SGLang 13.8µs (SoL 7.9µs) · _bf16_swapab_kernel 9.5µs 100.0%
- layer=o load=decode_1 m=1: yours 5.9µs vs SGLang 6.7µs (SoL 3.5µs) · _bf16_swapab_kernel 4.1µs 100.0%
- layer=pooler load=decode_1 m=1: yours 18.8µs vs SGLang 21.1µs (SoL 14.0µs) · _bf16_swapab_kernel 18.6µs 100.0%
- layer=unpooler load=decode_1 m=1: yours 6.2µs vs SGLang 7.3µs (SoL 1.7µs) · _bf16_swapab_kernel 4.3µs 100.0%
- layer=router load=decode_1 m=1: yours 5.4µs vs SGLang 10.4µs (SoL 1.8µs) · _router_swapab_kernel 3.1µs 80.2%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.8µs 19.8%
- layer=qkv_gate load=decode_16 m=16: yours 13.1µs vs SGLang 14.1µs (SoL 7.9µs) · _bf16_swapab_kernel 10.1µs 100.0%
- layer=o load=decode_16 m=16: yours 6.2µs vs SGLang 6.9µs (SoL 3.6µs) · _bf16_swapab_kernel 4.4µs 100.0%
- layer=pooler load=decode_16 m=16: yours 19.5µs vs SGLang 21.6µs (SoL 14.1µs) · _bf16_swapab_kernel 19.6µs 100.0%
- layer=unpooler load=decode_16 m=16: yours 18.5µs vs SGLang 18.7µs (SoL 14.1µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 19.3µs 100.0%
- layer=router load=decode_16 m=16: yours 5.5µs vs SGLang 14.8µs (SoL 1.8µs) · _router_swapab_kernel 3.4µs 82.9%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 17.1%
- layer=qkv_gate load=decode_32 m=32: yours 14.2µs vs SGLang 14.6µs (SoL 8.0µs) · _bf16_swapab_kernel 11.8µs 100.0%
- layer=o load=decode_32 m=32: yours 6.3µs vs SGLang 6.9µs (SoL 3.6µs) · _bf16_swapab_kernel 4.8µs 100.0%
- layer=pooler load=decode_32 m=32: yours 19.7µs vs SGLang 21.8µs (SoL 14.1µs) · _bf16_swapab_kernel 20.7µs 100.0%
- layer=unpooler load=decode_32 m=32: yours 18.1µs vs SGLang 18.6µs (SoL 14.1µs) · _bf16_swapab_kernel 18.7µs 100.0%
- layer=router load=decode_32 m=32: yours 6.2µs vs SGLang 15.6µs (SoL 1.9µs) · _router_swapab_kernel 3.6µs 83.6%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 16.4%
- layer=qkv_gate load=verify_1 m=8: yours 12.5µs vs SGLang 14.0µs (SoL 7.9µs) · _bf16_swapab_kernel 9.9µs 100.0%
- layer=o load=verify_1 m=8: yours 6.1µs vs SGLang 6.7µs (SoL 3.5µs) · _bf16_swapab_kernel 4.2µs 100.0%
- layer=pooler load=verify_1 m=8: yours 18.9µs vs SGLang 21.1µs (SoL 14.0µs) · _bf16_swapab_kernel 19.7µs 100.0%
- layer=unpooler load=verify_1 m=8: yours 6.0µs vs SGLang 7.1µs (SoL 1.8µs) · _bf16_swapab_kernel 4.4µs 100.0%
- layer=router load=verify_1 m=8: yours 5.5µs vs SGLang 26.2µs (SoL 1.8µs) · _router_swapab_kernel 3.0µs 81.7%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 18.3%
- layer=qkv_gate load=verify_16 m=128: yours 17.0µs vs SGLang 17.2µs (SoL 8.4µs) · nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN 12.6µs 80.2%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 3.1µs 19.8%
- layer=o load=verify_16 m=128: yours 7.0µs vs SGLang 7.2µs (SoL 4.0µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 5.4µs 100.0%
- layer=pooler load=verify_16 m=128: yours 25.0µs vs SGLang 24.9µs (SoL 14.6µs) · nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 24.5µs 100.0%
- layer=unpooler load=verify_16 m=128: yours 20.0µs vs SGLang 20.0µs (SoL 14.6µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 20.3µs 100.0%
- layer=router load=verify_16 m=128: yours 7.3µs vs SGLang 16.3µs (SoL 2.2µs) · _router_hilo_kernel 5.0µs 85.7%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.8µs 14.3%
- layer=qkv_gate load=verify_32 m=256: yours 19.7µs vs SGLang 19.7µs (SoL 9.8µs) · nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 19.5µs 100.0%
- layer=o load=verify_32 m=256: yours 9.3µs vs SGLang 9.4µs (SoL 4.5µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.5µs 100.0%
- layer=pooler load=verify_32 m=256: yours 31.2µs vs SGLang 31.2µs (SoL 17.4µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 30.7µs 100.0%
- layer=unpooler load=verify_32 m=256: yours 24.4µs vs SGLang 24.4µs (SoL 17.4µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 24.3µs 100.0%
- layer=router load=verify_32 m=256: yours 9.6µs vs SGLang 17.7µs (SoL 2.7µs) · _router_hilo_kernel 6.9µs 90.4%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 9.6%

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

### bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma)
- **bf16-verify-streamk-cutlass-wgmma-rewrite** → regression (regression), 1.1635x — gained on qkv_gate decode_16, decode_32; router decode_16, decode_32, verify_1; o decode_32; pooler verify_16; lost on qkv_gate decode_1, verify_1, verify_32; o verify_32; pooler verify_32; unpooler verify_32. Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: at m=256 (verify_32, the load furthest from its floor: 92.9 us vs 51.8 floor) all fo

### bf16 GEMM dispatch (m=17..32, swap-AB Triton)
- **bf16-m32-swapab-wgmma-triton** → neutral (unproven-vs-best), 1.1984x — gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_32, verify_1, verify_16; pooler decode_32, verify_1; router verify_1, verify_16. Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-

### bf16 GEMM dispatch (m=65..256, DeepGEMM bf16)
- **bf16-verify-deepgemm-bf16-gemm-nt-route** → regression (regression), 1.2027x — lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; router decode_16, decode_32. Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=1

### bf16 GEMM dispatch (m>=128, weight layout)
- **bf16-m128-nn-weight-layout-probe** → neutral (unproven-vs-best), 1.1580x — gained on pooler verify_16; unpooler verify_16; lost on o decode_1; router verify_1, verify_16; qkv_gate verify_32. Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 

### bf16 GEMM dispatch (shape-gated Triton routing, m<=16)
- **bf16-shape-gated-triton-winners-combo** → neutral (unproven-vs-best), 1.1760x — gained on qkv_gate decode_1; o decode_1, decode_16; pooler decode_1, decode_16, verify_1; unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neu

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

### qkv GEMM (m=17..32, swap-AB)
- **qkv-swapab-m32-splitk8-stages3** → regression (regression), 1.1963x — gained on pooler verify_16; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1, decode_16, decode_32, verify_1. Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 st

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

### router GEMM (m<=8)
- **restack-router-cluster-dsmem-m8** → neutral (within-ci), 1.2199x — gained on router decode_1, verify_1; qkv_gate decode_16; lost on o decode_32, verify_16; router verify_16. Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel fr

### unpooler GEMM
- **unpooler-slice-triton-gemv** → regression (regression), 0.8801x — gained on o verify_1; lost on unpooler decode_1, verify_1. For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call i
- **unpooler-full-triton-skinny-m32** → neutral (within-ci), 1.0053x — gained on qkv_gate decode_1, decode_16, verify_1; unpooler decode_16, verify_1; o verify_1; lost on unpooler decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** → neutral (within-ci), 0.9990x — gained on o verify_16; lost on qkv_gate decode_16; unpooler decode_32. From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num
- **unpooler-slice-triton-splitk8-semaphore** → neutral (unproven-vs-best), 1.1546x — gained on unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only,
- **unpooler-full-m16-swapab-route** → regression (regression), 1.1885x — gained on router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; pooler decode_1, decode_16, verify_1; router decode_1, decode_16, decode_32, verify_1; unpooler verify_1. Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (c

## Open follow-ups

- router-hilo-addmm-epilogue: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-hilo-addmm-epilogue`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- pooler-triton-skinny-m32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-triton-skinny-m32`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- bf16-cublaslt-backend-toggle: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-cublaslt-backend-toggle`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- o-triton-skinny-m32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: o-triton-skinny-m32`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- unpooler-full-triton-skinny-m32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: unpooler-full-triton-skinny-m32`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- unpooler-full-triton-skinny-m32-control-blockm16-all-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: unpooler-full-triton-skinny-m32-control-blockm16-all-m`) that adds a further change in the same area is the only way to learn more (ident
- pooler-triton-splitk2-blockn64-semaphore: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-triton-splitk2-blockn64-semaphore`) that adds a further change in the same area is the only way to learn more (identical co
- qkv-gate-triton-splitk4-blockn64-semaphore: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: qkv-gate-triton-splitk4-blockn64-semaphore`) that adds a further change in the same area is the only way to learn more (identical code is
- router-splitk32-small-m-all-loads-in-flight: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-splitk32-small-m-all-loads-in-flight`) that adds a further change in the same area is the only way to learn more (identical code i
- router-splitk-semaphore-no-memset: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-splitk-semaphore-no-memset`) that adds a further change in the same area is the only way to learn more (identical code is n
- unpooler-slice-triton-splitk8-semaphore: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: unpooler-slice-triton-splitk8-semaphore`) that adds a further change in the same area is the only way to learn more (identical cod
- pooler-triton-splitk2-blockn64-semaphore-control-blockk256: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-triton-splitk2-blockn64-semaphore-control-blockk256`) that adds a further change in the same area is the only way to learn more (i
- pooler-splitk2-tma-weight-stream: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-splitk2-tma-weight-stream`) that adds a further change in the same area is the only way to learn more (identical code is ne
- router-splitk32-small-m-all-loads-in-flight-control-stages3: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-splitk32-small-m-all-loads-in-flight-control-stages3`) that adds a further change in the same area is the only way to learn
- bf16-shape-gated-triton-winners-combo: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-shape-gated-triton-winners-combo`) that adds a further change in the same area is the only way to learn more (identical code 
- bf16-m128-nn-weight-layout-probe: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-m128-nn-weight-layout-probe`) that adds a further change in the same area is the only way to learn more (identical code is ne
- bf16-m32-swapab-wgmma-triton: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-m32-swapab-wgmma-triton`) that adds a further change in the same area is the only way to learn more (identical code is never 
- bf16-cublaslt-algo-pick-in-prepare-m128-m256: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-cublaslt-algo-pick-in-prepare-m128-m256`) that adds a further change in the same area is the only way to learn more (identical code 
- bf16-m16-swapab-wgmma-extend: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-m16-swapab-wgmma-extend`) that adds a further change in the same area is the only way to learn more (identical code is never 
- router-swapab-wgmma-all-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-swapab-wgmma-all-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- bf16-restack-m16-swapab-router-swapab-m32: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-restack-m16-swapab-router-swapab-m32`) that adds a further change in the same area is the only way to learn more (identical c
- bf16-lossless-exp4-occupancy-dequant-once-stage2: rewrite stage 2 of `Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32`: read the replaced kernels' own us per shape (profile top_kernels) against the inc
- restack-pooler-unpooler-nn-layout-m128: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-pooler-unpooler-nn-layout-m128`) that adds a further change in the same area is the only way to learn more (identical code
- restack-router-cluster-dsmem-m8: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: restack-router-cluster-dsmem-m8`) that adds a further change in the same area is the only way to learn more (identical code is never re-r

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: pooler-skinny-tiled-weight-layout [pooler GEMM] p4 hold
- queue: router-triton-hilo-fused-control-hybrid-cublas-above-m32 [router GEMM] p1 hold
- queue: router-hilo-addmm-epilogue-control-inplace-out [router GEMM] p1 hold
- queue: qkv-gate-o-m128-wgmma-splitk-semaphore [qkv GEMM + o GEMM (m=128)] p2 hold
- queue: pooler-m32-tma-splitk-extend [pooler GEMM (m=17..32)] p3 hold
- queue: bf16-lossless-exp4-warpspec-persistent-stage3 [lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)] p1 hold
- running: bf16-verify-swapab-wgmma-cluster-dsmem-rewrite [bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K)] p1 running
- running: bf16-verify-deepgemm-sm90-kernel-asserted-rewrite [bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)] p1 running
- running: restack-unpooler-slice-cluster-dsmem-m8 [unpooler GEMM (sliced n=1024, m<=8)] p1 running
- running: router-cluster-dsmem-s2-flat-in-m-occupancy [router GEMM (m<=32, cluster DSMEM kernel)] p2 running
- running: router-m128-m256-cluster-dsmem-tiled-rewrite [router GEMM (m=65..256)] p2 running
