# muse_bf16_proj: same-day public control (B0-aa1, B0-aa2)

Reads: B0-aa1 `a95479284a204d87` 1.5365 [1.5134, 1.5595] on sfo3c, B0-aa2 `280a3622e2384023` 1.5442 [1.5210, 1.5674] on sfo3

Control mean (geomean) **1.5403**, headline spread s_h = 0.50%. Public published (10-29) row: see RESULTS.md. Our board BEST job `7753d5c7031a4e72`.

| row | w | public 10-29 us | B0-aa1 us | B0-aa2 us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| layer=qkv_gate load=decode 1 m=1 | 62 | 8.83 | 8.66 | 8.84 | 8.75 | 2.1 | 0.991 | 10.53 | 1.204 | void cgemv_kernel<2, 512, 8192, 2>(uns 7.0 | void b12_kernel<16, 8192, 6, 2, 4>(uns 9.2 |
| layer=o load=decode 1 m=1 | 62 | 5.24 | 5.00 | 5.28 | 5.14 | 5.5 | 0.980 | 5.32 | 1.036 | void cgemv_kernel<2, 64, 1024>(unsigne 4.0 | void b12_kernel<64, 1024, 8, 1, 8>(uns 4.1 |
| layer=pooler load=decode 1 m=1 | 62 | 13.36 | 13.21 | 13.24 | 13.22 | 0.3 | 0.990 | 14.62 | 1.106 | void cgemv_kernel<2, 512, 8192>(unsign 13.2 | void b12_kernel<32, 8192, 8, 1, 8>(uns 13.9 |
| layer=unpooler load=decode 1 m=1 | 62 | 3.58 | 3.47 | 3.65 | 3.56 | 5.1 | 0.994 | 4.71 | 1.325 | void cgemv_kernel<2, 128, 4096>(unsign 2.3 | void (anonymous namespace)::ouns_kerne 4.1 |
| layer=router load=decode 1 m=1 | 62 | 3.69 | 3.66 | 3.68 | 3.67 | 0.7 | 0.994 | 3.71 | 1.011 | void rgemv_kernel<1, 1>(float const*,  2.4 | void (anonymous namespace)::router_rv_ 3.6 |
| layer=qkv_gate load=decode 16 m=16 | 62 | 9.65 | 9.82 | 9.93 | 9.87 | 1.1 | 1.023 | 12.20 | 1.236 | void sk2_kernel<1, 16, 3, 1, 4, 2, 5,  8.7 | void (anonymous namespace)::bf1_kernel 8.9 |
| layer=o load=decode 16 m=16 | 62 | 5.45 | 5.81 | 5.61 | 5.71 | 3.6 | 1.048 | 6.14 | 1.075 | void sk2_kernel<1, 16, 2, 2, 1, 4, 3,  4.4 | _bf16_swapab_kernel 4.0 |
| layer=pooler load=decode 16 m=16 | 62 | 13.54 | 13.77 | 13.84 | 13.80 | 0.5 | 1.020 | 17.97 | 1.302 | void dd_cap2_b_cpssm_gemv<1, 2, 1, 2,  13.1 | void (anonymous namespace)::bf1_kernel 18.1 |
| layer=unpooler load=decode 16 m=16 | 62 | 13.64 | 13.77 | 13.77 | 13.77 | 0.1 | 1.010 | 17.35 | 1.260 | void sk2_kernel<1, 16, 2, 2, 2, 2, 3,  12.8 | void (anonymous namespace)::bf1_kernel 17.6 |
| layer=router load=decode 16 m=16 | 62 | 4.5 | 4.69 | 4.62 | 4.66 | 1.5 | 1.035 | 5.48 | 1.176 | void rr_small_kernel<2, 16, 8>(int4 co 4.1 | _router_swapab_kernel 3.4; _router_zero_pdl_kernel 0.8 |
| layer=qkv_gate load=decode 32 m=32 | 62 | 10.78 | 10.80 | 10.74 | 10.77 | 0.5 | 0.999 | 11.97 | 1.111 | void sk2_kernel<2, 32, 3, 1, 4, 2, 4,  9.9 | void (anonymous namespace)::bf1_kernel 10.1 |
| layer=o load=decode 32 m=32 | 62 | 6.09 | 6.27 | 6.04 | 6.16 | 3.8 | 1.011 | 6.46 | 1.050 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  4.8 | _bf16_swapab_kernel 4.5 |
| layer=pooler load=decode 32 m=32 | 62 | 14.87 | 15.24 | 14.98 | 15.11 | 1.7 | 1.016 | 18.20 | 1.204 | void sk2_kernel<2, 32, 2, 2, 2, 2, 5,  15.2 | void (anonymous namespace)::bf1_kernel 18.1 |
| layer=unpooler load=decode 32 m=32 | 62 | 14.46 | 14.62 | 14.18 | 14.40 | 3.1 | 0.996 | 18.40 | 1.277 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  14.2 | _bf16_swapab_kernel 18.4 |
| layer=router load=decode 32 m=32 | 62 | 5.24 | 5.63 | 5.21 | 5.42 | 7.8 | 1.035 | 5.67 | 1.046 | void sk2_kernel<2, 32, 1, 1, 8, 1, 4,  4.7 | _router_swapab_kernel 3.8; _router_zero_pdl_kernel 0.9 |
| layer=qkv_gate load=verify 1 m=8 | 62 | 9.78 | 9.19 | 9.19 | 9.19 | 0.0 | 0.940 | 11.65 | 1.268 | void sk2_kernel<1, 8, 3, 1, 4, 2, 5, t 7.3 | void (anonymous namespace)::b12m_tma_k 11.4 |
| layer=o load=verify 1 m=8 | 62 | 5.52 | 5.38 | 5.20 | 5.29 | 3.5 | 0.958 | 6.50 | 1.230 | void v18s_gemv<1, 1, 1, 2, 0, 0, 256,  4.8 | _bf16_swapab_kernel 4.0 |
| layer=pooler load=verify 1 m=8 | 62 | 13.55 | 13.56 | 13.74 | 13.65 | 1.3 | 1.007 | 15.99 | 1.171 | void smallm2_gemv<1, 1, 1, 1, 0, 0, 10 13.4 | void (anonymous namespace)::b12m_tma_k 16.0 |
| layer=unpooler load=verify 1 m=8 | 62 | 4.11 | 4.22 | 4.02 | 4.12 | 4.9 | 1.003 | 5.24 | 1.271 | void sk2_kernel<1, 8, 1, 1, 8, 2, 4, t 3.3 | void (anonymous namespace)::ouns_kerne 4.6 |
| layer=router load=verify 1 m=8 | 62 | 4.38 | 4.66 | 4.31 | 4.48 | 7.9 | 1.023 | 4.88 | 1.088 | void rr_small_kernel<1, 16, 8>(int4 co 3.6 | void (anonymous namespace)::router_ker 3.5 |
| layer=qkv_gate load=verify 16 m=128 | 62 | 13.79 | 14.11 | 13.90 | 14.01 | 1.5 | 1.016 | 16.18 | 1.155 | void qb_e12dcf82a592343f::qb_kernel<64 13.1 | nvjet_sm90_tst_40x64_64x16_4x2_h_bz_TN 16.0 |
| layer=o load=verify 16 m=128 | 62 | 6.96 | 6.58 | 7.15 | 6.86 | 8.4 | 0.986 | 7.41 | 1.079 | void ol_projection<128, 128, 4, 3, 1,  6.3 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TN 5.3 |
| layer=pooler load=verify 16 m=128 | 62 | 19.94 | 21.16 | 20.06 | 20.61 | 5.3 | 1.034 | 21.48 | 1.042 | nvjet_sm90_tst_64x64_64x13_1x1_v_bz_co 19.5 | nvjet_sm90_tss_128x64_64x8_1x2_h_bz_NN 21.5; _add2_cast_kernel 2.0 |
| layer=unpooler load=verify 16 m=128 | 62 | 19.07 | 18.99 | 19.11 | 19.05 | 0.6 | 0.999 | 19.09 | 1.002 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NN 18.1 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NN 18.9 |
| layer=router load=verify 16 m=128 | 62 | 5.77 | 5.75 | 6.22 | 5.99 | 7.8 | 1.038 | 7.10 | 1.187 | rk_kernel(CUtensorMap_st, CUtensorMap_ 4.6 | _router_hilo_kernel 5.2; _router_zero_pdl_kernel 0.8 |
| layer=qkv_gate load=verify 32 m=256 | 62 | 17.27 | 17.61 | 17.70 | 17.65 | 0.6 | 1.022 | 19.27 | 1.091 | nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TN 17.3 | nvjet_sm90_tst_96x64_64x10_2x4_h_bz_TN 19.2 |
| layer=o load=verify 32 m=256 | 62 | 9.26 | 8.82 | 9.04 | 8.93 | 2.4 | 0.964 | 9.28 | 1.039 | void rl_early_projection<256, 64, 5, 3 9.1 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_T 8.2 |
| layer=pooler load=verify 32 m=256 | 62 | 24.36 | 24.66 | 24.43 | 24.54 | 0.9 | 1.007 | 27.34 | 1.114 | nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NN 24.2 | _ZN7cutlass13device_kernelINS_4gemm6ke 28.1 |
| layer=unpooler load=verify 32 m=256 | 62 | 23.2 | 23.37 | 23.30 | 23.34 | 0.3 | 1.006 | 24.05 | 1.030 | nvjet_sm90_tst_128x128_64x6_1x1_v_bz_c 23.4 | nvjet_sm90_tst_128x128_64x6_2x1_v_bz_T 24.6 |
| layer=router load=verify 32 m=256 | 62 | 7.35 | 7.09 | 7.19 | 7.14 | 1.4 | 0.972 | 9.04 | 1.266 | rk_kernel(CUtensorMap_st, CUtensorMap_ 5.9 | _router_hilo_kernel 7.0; _router_zero_pdl_kernel 0.9 |
