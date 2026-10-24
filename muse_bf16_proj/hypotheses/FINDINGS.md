# muse_bf16_proj findings

Regenerated 2026-10-24T115933Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Current best

1.1433x [1.1261,1.1604] from job `240eb22ea0b6487e` (submit), master commit `dba4ffd594`, hypothesis router-triton-hilo-fused-control-splitk-all-m

Where the time goes (per scored shape, top kernels):

- layer=qkv_gate load=decode_1 m=1: yours 14.4µs vs SGLang 14.1µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 10.8µs 86.7%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 13.3%
- layer=o load=decode_1 m=1: yours 6.7µs vs SGLang 6.8µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 5.0µs 100.0%
- layer=pooler load=decode_1 m=1: yours 21.2µs vs SGLang 21.3µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.3µs 92.1%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 7.9%
- layer=unpooler load=decode_1 m=1: yours 7.1µs vs SGLang 7.0µs (SoL 1.7µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0µs 73.1%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.5µs 26.9%
- layer=router load=decode_1 m=1: yours 5.8µs vs SGLang 10.3µs (SoL 1.8µs) · _router_hilo_kernel 3.3µs 81.0%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.8µs 19.0%
- layer=qkv_gate load=decode_16 m=16: yours 13.8µs vs SGLang 13.7µs (SoL 7.9µs) · nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT 8.1µs 83.0%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 17.0%
- layer=o load=decode_16 m=16: yours 6.8µs vs SGLang 6.8µs (SoL 3.6µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 4.5µs 100.0%
- layer=pooler load=decode_16 m=16: yours 21.3µs vs SGLang 21.5µs (SoL 14.1µs) · nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 19.3µs 91.6%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.8µs 8.4%
- layer=unpooler load=decode_16 m=16: yours 18.4µs vs SGLang 18.3µs (SoL 14.1µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 18.3µs 100.0%
- layer=router load=decode_16 m=16: yours 6.2µs vs SGLang 14.9µs (SoL 1.8µs) · _router_hilo_kernel 3.5µs 83.1%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 16.9%
- layer=qkv_gate load=decode_32 m=32: yours 14.3µs vs SGLang 14.2µs (SoL 8.0µs) · nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT 10.2µs 85.1%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.8µs 14.9%
- layer=o load=decode_32 m=32: yours 6.6µs vs SGLang 6.7µs (SoL 3.6µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 4.8µs 100.0%
- layer=pooler load=decode_32 m=32: yours 22.0µs vs SGLang 22.0µs (SoL 14.1µs) · nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 20.2µs 91.3%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.9µs 8.7%
- layer=unpooler load=decode_32 m=32: yours 18.7µs vs SGLang 18.7µs (SoL 14.1µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 18.5µs 100.0%
- layer=router load=decode_32 m=32: yours 6.5µs vs SGLang 15.4µs (SoL 1.9µs) · _router_hilo_kernel 4.1µs 85.3%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 14.7%
- layer=qkv_gate load=verify_1 m=8: yours 14.0µs vs SGLang 14.1µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.2µs 82.6%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 2.0µs 17.4%
- layer=o load=verify_1 m=8: yours 6.3µs vs SGLang 6.4µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 4.1µs 100.0%
- layer=pooler load=verify_1 m=8: yours 21.3µs vs SGLang 21.5µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.9µs 91.9%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 8.1%
- layer=unpooler load=verify_1 m=8: yours 7.0µs vs SGLang 7.1µs (SoL 1.8µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.1µs 72.3%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 27.7%
- layer=router load=verify_1 m=8: yours 5.7µs vs SGLang 26.1µs (SoL 1.8µs) · _router_hilo_kernel 3.3µs 81.9%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.7µs 18.1%
- layer=qkv_gate load=verify_16 m=128: yours 17.0µs vs SGLang 17.0µs (SoL 8.4µs) · nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN 12.9µs 80.8%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 3.1µs 19.2%
- layer=o load=verify_16 m=128: yours 7.0µs vs SGLang 7.1µs (SoL 4.0µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 5.3µs 100.0%
- layer=pooler load=verify_16 m=128: yours 25.2µs vs SGLang 24.6µs (SoL 14.6µs) · nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 25.2µs 100.0%
- layer=unpooler load=verify_16 m=128: yours 19.7µs vs SGLang 19.8µs (SoL 14.6µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 20.0µs 100.0%
- layer=router load=verify_16 m=128: yours 7.0µs vs SGLang 15.9µs (SoL 2.2µs) · _router_hilo_kernel 4.9µs 86.5%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.8µs 13.5%
- layer=qkv_gate load=verify_32 m=256: yours 19.5µs vs SGLang 19.5µs (SoL 9.8µs) · nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 19.0µs 100.0%
- layer=o load=verify_32 m=256: yours 9.5µs vs SGLang 9.6µs (SoL 4.5µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.4µs 100.0%
- layer=pooler load=verify_32 m=256: yours 30.5µs vs SGLang 30.9µs (SoL 17.4µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 29.5µs 100.0%
- layer=unpooler load=verify_32 m=256: yours 24.4µs vs SGLang 24.4µs (SoL 17.4µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 24.8µs 100.0%
- layer=router load=verify_32 m=256: yours 9.3µs vs SGLang 17.4µs (SoL 2.7µs) · _router_hilo_kernel 7.0µs 86.6%, void at::native::vectorized_elementwise_kernel<4, at::nat... 1.1µs 13.4%

## By area

### bf16 GEMM dispatch
- **sglang-gemm-dispatch-parity** → gain (gain), 1.0022x — gained on router decode_16, decode_32, verify_16, verify_32; o verify_32. Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], 
- **bf16-cublaslt-backend-toggle** → neutral (within-ci), 1.1505x — gained on qkv_gate decode_1, verify_16; pooler decode_32; lost on o decode_1, verify_1; unpooler decode_1, verify_1; router verify_1; qkv_gate verify_32. A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') imme

### o GEMM
- **o-triton-skinny-m32** → neutral (within-ci), 1.0057x — gained on qkv_gate decode_1, decode_16; o decode_1, decode_16, verify_1; unpooler decode_1; router decode_1; lost on o decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'o' at m <= 32 (n=8192, k=1024, 16.8 MB): in kernel_fn route o at m <= 32 to the same ke

### pooler GEMM
- **pooler-triton-skinny-m32** → neutral (within-ci), 0.9983x — gained on pooler decode_1, decode_16, verify_1, verify_16; lost on qkv_gate decode_1; unpooler decode_1; pooler decode_32. For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams t
- **pooler-triton-splitk2-blockn64-semaphore** → neutral (unproven-vs-best), 1.1664x — gained on pooler decode_1, decode_16, verify_1, verify_16; qkv_gate decode_32; lost on router verify_1, verify_16; o verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf

### pooler GEMM + unpooler GEMM (m=128)
- **pooler-unpooler-m128-wgmma-splitk-semaphore** → regression (regression), 1.1158x — gained on router decode_16; lost on qkv_gate verify_1; o verify_1; unpooler verify_1, verify_16; router verify_1, verify_16; pooler verify_16. Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 

### qkv GEMM
- **qkv-gate-triton-skinny-m32** → regression (regression), 0.9238x — lost on qkv_gate decode_1, decode_16, decode_32, verify_1. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32 (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at
- **qkv-gate-triton-splitk4-blockn64-semaphore** → neutral (within-ci), 1.1431x — gained on qkv_gate decode_1, decode_16, verify_1; router decode_16; pooler verify_16; lost on qkv_gate decode_32; o verify_1; unpooler verify_1; router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvje

### router GEMM
- **router-hilo-at-m8** → gain (gain), 1.0219x — gained on unpooler verify_1; router verify_1. In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX
- **router-hilo-addmm-epilogue** → neutral (within-ci), 1.0256x — gained on o verify_16; pooler verify_32. In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc +=
- **router-triton-hilo-fused** → regression (regression), 0.9402x — gained on router decode_1, decode_16, decode_32, verify_1; qkv_gate verify_1; o verify_1, verify_16; unpooler verify_1; lost on router verify_16, verify_32. Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch 
- **router-triton-hilo-fused-control-splitk-all-m** → gain (gain), 1.1510x — gained on router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; lost on qkv_gate decode_1. Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Tr
- **router-splitk32-small-m-all-loads-in-flight** → neutral (within-ci), 1.1427x — gained on pooler verify_16; lost on router decode_1, decode_16, decode_32, verify_1. Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + 
- **router-splitk32-small-m-all-loads-in-flight-control-stages3** → neutral (unproven-vs-best), 1.1518x — gained on unpooler decode_1; router decode_1, decode_16, decode_32, verify_1; qkv_gate decode_16; lost on o verify_16; pooler verify_16; router verify_16. From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPL

### unpooler GEMM
- **unpooler-slice-triton-gemv** → regression (regression), 0.8801x — gained on o verify_1; lost on unpooler decode_1, verify_1. For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call i
- **unpooler-full-triton-skinny-m32** → neutral (within-ci), 1.0053x — gained on qkv_gate decode_1, decode_16, verify_1; unpooler decode_16, verify_1; o verify_1; lost on unpooler decode_32; pooler verify_16. Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route
- **unpooler-full-triton-skinny-m32-control-blockm16-all-m** → neutral (within-ci), 0.9990x — gained on o verify_16; lost on qkv_gate decode_16; unpooler decode_32. From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num
- **unpooler-slice-triton-splitk8-semaphore** → neutral (unproven-vs-best), 1.1546x — gained on unpooler decode_1, verify_1; lost on router verify_1, verify_16. Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only,

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
- unpooler-slice-triton-splitk8-semaphore: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: unpooler-slice-triton-splitk8-semaphore`) that adds a further change in the same area is the only way to learn more (identical cod
- router-splitk32-small-m-all-loads-in-flight-control-stages3: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-splitk32-small-m-all-loads-in-flight-control-stages3`) that adds a further change in the same area is the only way to learn

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: pooler-skinny-tiled-weight-layout [pooler GEMM] p4 hold
- queue: router-triton-hilo-fused-control-hybrid-cublas-above-m32 [router GEMM] p1 hold
- queue: router-hilo-addmm-epilogue-control-inplace-out [router GEMM] p1 hold
- queue: router-splitk-semaphore-no-memset [router GEMM] p4 hold
- queue: qkv-gate-o-m128-wgmma-splitk-semaphore [qkv GEMM + o GEMM (m=128)] p2 hold
- queue: bf16-shape-gated-triton-winners-combo [bf16 GEMM dispatch (shape-gated Triton routing, m<=16)] p1 hold
- running: pooler-triton-splitk2-blockn64-semaphore-control-blockk256 [pooler GEMM] p1 running
- running: pooler-splitk2-tma-weight-stream [pooler GEMM] p3 running
- running: bf16-m128-nn-weight-layout-probe [bf16 GEMM dispatch (m>=128, weight layout)] p3 running
