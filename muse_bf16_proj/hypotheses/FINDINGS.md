# muse_bf16_proj findings

Regenerated 2026-10-23T221905Z by `churn.py index`. Read this before writing a hypothesis for muse_bf16_proj.

## Current best

0.9021x [0.8885,0.9156] from job `366189f0f1114da1` (check), master commit `ab21d70942`

Where the time goes (per scored shape, top kernels):

- layer=qkv_gate load=decode_1 m=1: yours 14.1µs vs SGLang 13.9µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.6µs 84.8%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 15.2%
- layer=o load=decode_1 m=1: yours 7.0µs vs SGLang 7.0µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 5.1µs 100.0%
- layer=pooler load=decode_1 m=1: yours 21.4µs vs SGLang 21.6µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 18.8µs 92.3%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 7.7%
- layer=unpooler load=decode_1 m=1: yours 6.9µs vs SGLang 6.8µs (SoL 1.7µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.3µs 73.8%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.5µs 26.2%
- layer=router load=decode_1 m=1: yours 10.5µs vs SGLang 10.4µs (SoL 1.8µs) · void dot_kernel<float, 128, 0, cublasDotParams<cublasGemv... 4.0µs 52.7%, void at::native::unrolled_elementwise_kernel<at::native::... 2.0µs 26.8%, void reduce_1Block_kernel<float, 128, 7, cublasGemvTensor... 1.6µs 20.5%
- layer=qkv_gate load=decode_16 m=16: yours 14.3µs vs SGLang 14.2µs (SoL 7.9µs) · nvjet_sm90_tst_64x16_64x16_4x1_v_bz_splitK_TNT 7.8µs 82.5%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.7µs 17.5%
- layer=o load=decode_16 m=16: yours 6.7µs vs SGLang 6.6µs (SoL 3.6µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 4.5µs 100.0%
- layer=pooler load=decode_16 m=16: yours 21.4µs vs SGLang 21.6µs (SoL 14.1µs) · nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 19.8µs 91.9%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.8µs 8.1%
- layer=unpooler load=decode_16 m=16: yours 18.5µs vs SGLang 18.6µs (SoL 14.1µs) · nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 18.5µs 100.0%
- layer=router load=decode_16 m=16: yours 26.2µs vs SGLang 14.8µs (SoL 1.8µs) · sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta... 9.3µs 46.9%, sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta... 8.4µs 42.0%, void at::native::unrolled_elementwise_kernel<at::native::... 2.2µs 11.1%
- layer=qkv_gate load=decode_32 m=32: yours 14.5µs vs SGLang 14.4µs (SoL 8.0µs) · nvjet_sm90_tst_64x32_64x16_4x1_v_bz_splitK_TNT 8.4µs 82.4%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.8µs 17.6%
- layer=o load=decode_32 m=32: yours 6.6µs vs SGLang 6.6µs (SoL 3.6µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 4.8µs 100.0%
- layer=pooler load=decode_32 m=32: yours 22.1µs vs SGLang 22.1µs (SoL 14.1µs) · nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 20.2µs 91.3%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.9µs 8.7%
- layer=unpooler load=decode_32 m=32: yours 18.9µs vs SGLang 18.7µs (SoL 14.1µs) · nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 18.8µs 100.0%
- layer=router load=decode_32 m=32: yours 19.7µs vs SGLang 15.1µs (SoL 1.9µs) · void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_tn... 11.3µs 65.4%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.4µs 19.8%, void at::native::unrolled_elementwise_kernel<at::native::... 2.6µs 14.8%
- layer=qkv_gate load=verify_1 m=8: yours 14.3µs vs SGLang 14.1µs (SoL 7.9µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 9.4µs 85.2%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 14.8%
- layer=o load=verify_1 m=8: yours 6.9µs vs SGLang 6.9µs (SoL 3.5µs) · nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT 4.6µs 100.0%
- layer=pooler load=verify_1 m=8: yours 21.7µs vs SGLang 21.6µs (SoL 14.0µs) · nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 19.4µs 91.3%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.9µs 8.7%
- layer=unpooler load=verify_1 m=8: yours 7.1µs vs SGLang 7.0µs (SoL 1.8µs) · nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0µs 71.8%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 1.6µs 28.2%
- layer=router load=verify_1 m=8: yours 26.6µs vs SGLang 26.6µs (SoL 1.8µs) · sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta... 9.3µs 47.1%, sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize32x32x8_sta... 8.3µs 41.9%, void at::native::unrolled_elementwise_kernel<at::native::... 2.2µs 11.0%
- layer=qkv_gate load=verify_16 m=128: yours 17.0µs vs SGLang 17.1µs (SoL 8.4µs) · nvjet_sm90_tst_96x64_64x10_4x2_h_bz_splitK_TNN 12.8µs 80.4%, void cublasLt::splitKreduce_kernel<32, 16, int, float, __... 3.1µs 19.6%
- layer=o load=verify_16 m=128: yours 7.8µs vs SGLang 7.5µs (SoL 4.0µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 5.9µs 100.0%
- layer=pooler load=verify_16 m=128: yours 25.4µs vs SGLang 24.7µs (SoL 14.6µs) · nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 25.1µs 100.0%
- layer=unpooler load=verify_16 m=128: yours 20.1µs vs SGLang 20.0µs (SoL 14.6µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 20.7µs 100.0%
- layer=router load=verify_16 m=128: yours 29.1µs vs SGLang 16.3µs (SoL 2.2µs) · void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_t... 18.6µs 71.6%, void cublasLt::splitKreduce_kernel<32, 16, int, float, fl... 3.7µs 14.3%, void at::native::unrolled_elementwise_kernel<at::native::... 3.7µs 14.1%
- layer=qkv_gate load=verify_32 m=256: yours 20.4µs vs SGLang 19.9µs (SoL 9.8µs) · nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TNN 19.6µs 100.0%
- layer=o load=verify_32 m=256: yours 9.7µs vs SGLang 9.3µs (SoL 4.5µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 8.4µs 100.0%
- layer=pooler load=verify_32 m=256: yours 30.9µs vs SGLang 30.7µs (SoL 17.4µs) · nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 30.9µs 100.0%
- layer=unpooler load=verify_32 m=256: yours 24.6µs vs SGLang 24.5µs (SoL 17.4µs) · nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 24.8µs 100.0%
- layer=router load=verify_32 m=256: yours 45.8µs vs SGLang 17.2µs (SoL 2.7µs) · sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_sta... 32.4µs 71.7%, void at::native::unrolled_elementwise_kernel<at::native::... 7.0µs 15.4%, sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_sta... 5.8µs 12.8%

## By area

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-gemm-dispatch-parity [bf16 GEMM dispatch] p1 queued
