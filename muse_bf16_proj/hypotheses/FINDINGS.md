# muse_bf16_proj findings

Regenerated 2026-10-24T032013Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Current best

1.0307x [1.0152,1.0462] from job `9aa98e889be140eb` (submit), master commit `f9dc05da5a`, hypothesis router-hilo-at-m8

Where the time goes (per scored shape, top kernels):

- layer=qkv_gate load=decode_1 m=1: yours 14.2µs vs SGLang 14.0µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.4µs 84.7%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 15.3%
- layer=o load=decode_1 m=1: yours 6.6µs vs SGLang 6.6µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 5.0µs 100.0%
- layer=pooler load=decode_1 m=1: yours 21.5µs vs SGLang 21.2µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.6µs 91.9%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 8.1%
- layer=unpooler load=decode_1 m=1: yours 7.0µs vs SGLang 6.9µs (SoL 1.7µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0µs 73.1%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.5µs 26.9%
- layer=router load=decode_1 m=1: yours 10.5µs vs SGLang 10.4µs (SoL 1.8µs) · void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv... 4.0µs 52.1%, void at::native::unrolled_elementwise_kernel<at::native::... 2.1µs 27.7%, void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor... 1.6µs 20.2%
- layer=qkv_gate load=decode_16 m=16: yours 14.0µs vs SGLang 14.1µs (SoL 7.9µs) · nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT 8.1µs 83.0%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 17.0%
- layer=o load=decode_16 m=16: yours 6.9µs vs SGLang 6.8µs (SoL 3.6µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 4.6µs 100.0%
- layer=pooler load=decode_16 m=16: yours 21.6µs vs SGLang 21.4µs (SoL 14.1µs) · nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 19.2µs 91.8%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 8.2%
- layer=unpooler load=decode_16 m=16: yours 18.6µs vs SGLang 18.6µs (SoL 14.1µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 18.4µs 100.0%
- layer=router load=decode_16 m=16: yours 14.5µs vs SGLang 14.6µs (SoL 1.8µs) · nvjet_sm90_tss_64x16_64x16_4x1_v_bz_splitK_NNT 8.4µs 65.1%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.6µs 28.2%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.9µs 6.7%
- layer=qkv_gate load=decode_32 m=32: yours 14.4µs vs SGLang 14.3µs (SoL 8.0µs) · nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT 9.1µs 83.2%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.8µs 16.8%
- layer=o load=decode_32 m=32: yours 7.0µs vs SGLang 7.0µs (SoL 3.6µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 4.7µs 100.0%
- layer=pooler load=decode_32 m=32: yours 21.9µs vs SGLang 21.9µs (SoL 14.1µs) · nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 20.8µs 91.7%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.9µs 8.3%
- layer=unpooler load=decode_32 m=32: yours 18.6µs vs SGLang 18.6µs (SoL 14.1µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 19.0µs 100.0%
- layer=router load=decode_32 m=32: yours 15.0µs vs SGLang 15.2µs (SoL 1.9µs) · nvjet_sm90_tss_32x64_64x16_4x1_v_bz_splitK_NNN 8.1µs 63.4%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.7µs 28.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 1.0µs 7.8%
- layer=qkv_gate load=verify_1 m=8: yours 14.1µs vs SGLang 14.2µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.4µs 85.2%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 14.8%
- layer=o load=verify_1 m=8: yours 6.4µs vs SGLang 6.5µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 4.0µs 100.0%
- layer=pooler load=verify_1 m=8: yours 20.9µs vs SGLang 21.1µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 19.0µs 92.0%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 8.0%
- layer=unpooler load=verify_1 m=8: yours 6.7µs vs SGLang 6.8µs (SoL 1.8µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0µs 72.0%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 28.0%
- layer=router load=verify_1 m=8: yours 14.2µs vs SGLang 26.1µs (SoL 1.8µs) · nvjet_sm90_tss_64x8_64x16_4x1_v_bz_splitK_NNT 7.9µs 64.6%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.4µs 28.1%, void at::native::vectorized_elementwise_kernel<4, at::nat... 0.9µs 7.3%
- layer=qkv_gate load=verify_16 m=128: yours 17.3µs vs SGLang 16.7µs (SoL 8.4µs) · nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN 12.9µs 80.6%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 3.1µs 19.4%
- layer=o load=verify_16 m=128: yours 7.4µs vs SGLang 7.2µs (SoL 4.0µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 6.0µs 100.0%
- layer=pooler load=verify_16 m=128: yours 25.0µs vs SGLang 24.4µs (SoL 14.6µs) · nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 24.5µs 100.0%
- layer=unpooler load=verify_16 m=128: yours 19.7µs vs SGLang 19.5µs (SoL 14.6µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 20.0µs 100.0%
- layer=router load=verify_16 m=128: yours 16.2µs vs SGLang 16.1µs (SoL 2.2µs) · nvjet_sm90_tss_64x32_64x16_2x4_h_bz_splitK_NNT 8.8µs 64.2%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.8µs 27.9%, void at::native::vectorized_elementwise_kernel<4, at::nat... 1.1µs 7.9%
- layer=qkv_gate load=verify_32 m=256: yours 19.8µs vs SGLang 19.5µs (SoL 9.8µs) · nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 19.6µs 100.0%
- layer=o load=verify_32 m=256: yours 9.3µs vs SGLang 9.1µs (SoL 4.5µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.1µs 100.0%
- layer=pooler load=verify_32 m=256: yours 31.1µs vs SGLang 30.3µs (SoL 17.4µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 29.2µs 100.0%
- layer=unpooler load=verify_32 m=256: yours 24.1µs vs SGLang 23.9µs (SoL 17.4µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 24.7µs 100.0%
- layer=router load=verify_32 m=256: yours 17.7µs vs SGLang 17.6µs (SoL 2.7µs) · nvjet_sm90_tss_64x32_64x16_1x2_h_bz_splitK_NNT 10.5µs 70.7%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.3µs 22.2%, void at::native::vectorized_elementwise_kernel<4, at::nat... 1.1µs 7.1%

## By area

### bf16 GEMM dispatch
- **sglang-gemm-dispatch-parity** → gain (gain), 1.0022x — gained on router decode_16, decode_32, verify_16, verify_32; o verify_32. Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], 

### pooler GEMM
- **pooler-triton-skinny-m32** → neutral (within-ci), 0.9983x — gained on pooler decode_1, decode_16, verify_1, verify_16; lost on qkv_gate decode_1; unpooler decode_1; pooler decode_32. For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams t

### router GEMM
- **router-hilo-at-m8** → gain (gain), 1.0219x — gained on unpooler verify_1; router verify_1. In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX
- **router-triton-hilo-fused** → regression (regression), 0.9402x — gained on router decode_1, decode_16, decode_32, verify_1; qkv_gate verify_1; o verify_1, verify_16; unpooler verify_1; lost on router verify_16, verify_32. Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch 

### unpooler GEMM
- **unpooler-slice-triton-gemv** → regression (regression), 0.8801x — gained on o verify_1; lost on unpooler decode_1, verify_1. For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call i

## Open follow-ups

- pooler-triton-skinny-m32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-triton-skinny-m32`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: bf16-cublaslt-backend-toggle [bf16 GEMM dispatch] p5 queued
- queue: o-triton-skinny-m32 [o GEMM] p4 queued
- queue: pooler-skinny-tiled-weight-layout [pooler GEMM] p4 queued
- queue: unpooler-full-triton-skinny-m32 [unpooler GEMM] p3 queued
- queue: qkv-gate-triton-skinny-m32 [qkv GEMM] p3 queued
- queue: router-triton-hilo-fused-control-hybrid-cublas-above-m32 [router GEMM] p1 queued
- queue: router-hilo-addmm-epilogue-control-inplace-out [router GEMM] p1 queued
- running: router-hilo-addmm-epilogue [router GEMM] p3 running
- running: router-triton-hilo-fused-control-splitk-all-m [router GEMM] p1 running
