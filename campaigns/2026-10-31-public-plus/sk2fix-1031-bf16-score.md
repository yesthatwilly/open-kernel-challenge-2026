# muse_bf16_proj: same-day public control (BF16FIX-score)

Reads: BF16FIX-score `d1f5f46c65fa466a` 1.5522 [1.5289, 1.5755] on sfo3

Control mean (geomean) **1.5522**, headline spread s_h = nan%. Public published (10-29) row: see RESULTS.md. Our board BEST job `109f577bcfaf4d8a`.

| row | w | public 10-29 us | BF16FIX-score us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| layer=qkv_gate load=decode 1 m=1 | 62 | 8.83 | 8.52 | 8.52 | nan | 0.964 | 8.72 | 1.024 | void cgemv_kernel<2, 512, 8192, 2>(uns 7.0 | void cgemv_kernel<2, 512, 8192, 2>(uns 7.1 |
| layer=o load=decode 1 m=1 | 62 | 5.24 | 5.22 | 5.22 | nan | 0.996 | 5.24 | 1.004 | void cgemv_kernel<2, 64, 1024>(unsigne 4.2 | void cgemv_kernel<2, 64, 1024>(unsigne 4.1 |
| layer=pooler load=decode 1 m=1 | 62 | 13.36 | 13.17 | 13.17 | nan | 0.986 | 13.31 | 1.011 | void cgemv_kernel<2, 512, 8192>(unsign 13.0 | void cgemv_kernel<2, 512, 8192>(unsign 12.8 |
| layer=unpooler load=decode 1 m=1 | 62 | 3.58 | 3.60 | 3.60 | nan | 1.004 | 3.52 | 0.980 | void cgemv_kernel<2, 128, 4096>(unsign 2.7 | void cgemv_kernel<2, 128, 4096, 4>(uns 2.2 |
| layer=router load=decode 1 m=1 | 62 | 3.69 | 3.51 | 3.51 | nan | 0.951 | 3.60 | 1.027 | void rgemv_kernel<1, 1>(float const*,  2.5 | void rgemv_kernel<1, 1>(float const*,  2.5 |
| layer=qkv_gate load=decode 16 m=16 | 62 | 9.65 | 9.45 | 9.45 | nan | 0.979 | 9.45 | 1.001 | void sk2_kernel<1, 16, 3, 1, 4, 2, 5,  8.4 | void sk2_kernel<1, 16, 3, 1, 4, 2, 5,  8.4 |
| layer=o load=decode 16 m=16 | 62 | 5.45 | 5.47 | 5.47 | nan | 1.004 | 5.59 | 1.022 | void sk2_kernel<1, 16, 2, 2, 1, 4, 3,  4.4 | void sk2_kernel<1, 16, 2, 2, 1, 4, 3,  4.0 |
| layer=pooler load=decode 16 m=16 | 62 | 13.54 | 13.37 | 13.37 | nan | 0.987 | 14.02 | 1.049 | void dd_cap2_b_cpssm_gemv<1, 2, 1, 2,  13.3 | void dd_cap2_b_cpssm_gemv<1, 2, 1, 2,  13.6 |
| layer=unpooler load=decode 16 m=16 | 62 | 13.64 | 13.42 | 13.42 | nan | 0.984 | 13.80 | 1.028 | void sk2_kernel<1, 16, 2, 2, 2, 2, 3,  13.2 | void sk2_kernel<1, 16, 2, 2, 2, 2, 3,  14.2 |
| layer=router load=decode 16 m=16 | 62 | 4.5 | 4.66 | 4.66 | nan | 1.036 | 4.44 | 0.953 | void rr_small_kernel<2, 16, 8>(int4 co 4.2 | void rr_small_kernel<2, 16, 8>(int4 co 4.2 |
| layer=qkv_gate load=decode 32 m=32 | 62 | 10.78 | 10.35 | 10.35 | nan | 0.960 | 10.67 | 1.031 | void sk2_kernel<2, 32, 3, 1, 4, 2, 4,  9.8 | void sk2_kernel<2, 32, 3, 1, 4, 2, 4,  9.9 |
| layer=o load=decode 32 m=32 | 62 | 6.09 | 5.91 | 5.91 | nan | 0.970 | 5.61 | 0.949 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  4.9 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  4.8 |
| layer=pooler load=decode 32 m=32 | 62 | 14.87 | 14.77 | 14.77 | nan | 0.993 | 14.81 | 1.003 | void sk2_kernel<2, 32, 2, 2, 2, 2, 5,  14.8 | void sk2_kernel<2, 32, 2, 2, 2, 2, 5,  15.2 |
| layer=unpooler load=decode 32 m=32 | 62 | 14.46 | 14.48 | 14.48 | nan | 1.002 | 14.25 | 0.984 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  14.0 | void sk2_kernel<2, 32, 2, 2, 2, 2, 3,  14.4 |
| layer=router load=decode 32 m=32 | 62 | 5.24 | 5.81 | 5.81 | nan | 1.109 | 5.23 | 0.899 | void sk2_kernel<2, 32, 1, 1, 8, 1, 4,  5.1 | void sk2_kernel<2, 32, 1, 1, 8, 1, 4,  4.7 |
| layer=qkv_gate load=verify 1 m=8 | 62 | 9.78 | 9.29 | 9.29 | nan | 0.949 | 8.82 | 0.950 | void sk2_kernel<1, 8, 3, 1, 4, 2, 5, t 7.6 | void sk2_kernel<1, 8, 3, 1, 4, 2, 5, t 7.4 |
| layer=o load=verify 1 m=8 | 62 | 5.52 | 5.17 | 5.17 | nan | 0.937 | 5.15 | 0.995 | void v18s_gemv<1, 1, 1, 2, 0, 0, 256,  4.8 | void v18s_gemv<1, 1, 1, 2, 0, 0, 256,  4.7 |
| layer=pooler load=verify 1 m=8 | 62 | 13.55 | 13.68 | 13.68 | nan | 1.009 | 13.79 | 1.009 | void smallm2_gemv<1, 1, 1, 1, 0, 0, 10 13.4 | void smallm2_gemv<1, 1, 1, 1, 0, 0, 10 13.4 |
| layer=unpooler load=verify 1 m=8 | 62 | 4.11 | 4.24 | 4.24 | nan | 1.032 | 3.95 | 0.932 | void sk2_kernel<1, 8, 1, 1, 8, 2, 4, t 3.5 | void sk2_kernel<1, 8, 1, 1, 8, 2, 4, t 3.4 |
| layer=router load=verify 1 m=8 | 62 | 4.38 | 4.22 | 4.22 | nan | 0.963 | 3.92 | 0.928 | void rr_small_kernel<1, 16, 8>(int4 co 3.6 | void rr_small_kernel<1, 16, 8>(int4 co 3.5 |
| layer=qkv_gate load=verify 16 m=128 | 62 | 13.79 | 13.58 | 13.58 | nan | 0.985 | 13.44 | 0.990 | void qb_e12dcf82a592343f::qb_kernel<64 13.2 | void qb_e12dcf82a592343f::qb_kernel<64 13.9 |
| layer=o load=verify 16 m=128 | 62 | 6.96 | 6.84 | 6.84 | nan | 0.982 | 6.88 | 1.006 | void ol_projection<128, 128, 4, 3, 1,  7.2 | void ol_projection<128, 128, 4, 3, 1,  6.5 |
| layer=pooler load=verify 16 m=128 | 62 | 19.94 | 19.67 | 19.67 | nan | 0.987 | 20.24 | 1.029 | nvjet_sm90_tst_64x64_64x13_1x1_v_bz_co 19.9 | nvjet_sm90_tst_64x64_64x13_1x1_v_bz_co 21.7 |
| layer=unpooler load=verify 16 m=128 | 62 | 19.07 | 18.63 | 18.63 | nan | 0.977 | 19.27 | 1.035 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NN 18.0 | nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NN 18.8 |
| layer=router load=verify 16 m=128 | 62 | 5.77 | 5.99 | 5.99 | nan | 1.038 | 6.29 | 1.050 | rk_kernel(CUtensorMap_st, CUtensorMap_ 4.6 | rk_kernel(CUtensorMap_st, CUtensorMap_ 4.6 |
| layer=qkv_gate load=verify 32 m=256 | 62 | 17.27 | 17.08 | 17.08 | nan | 0.989 | 18.24 | 1.067 | nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TN 18.4 | nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TN 20.4 |
| layer=o load=verify 32 m=256 | 62 | 9.26 | 9.05 | 9.05 | nan | 0.978 | 9.05 | 1.000 | void rl_early_projection<256, 64, 5, 3 9.0 | void rl_early_projection<256, 64, 5, 3 8.8 |
| layer=pooler load=verify 32 m=256 | 62 | 24.36 | 24.38 | 24.38 | nan | 1.001 | 24.59 | 1.008 | nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NN 24.6 | nvjet_sm90_tst_128x64_64x8_1x1_v_bz_NN 24.6 |
| layer=unpooler load=verify 32 m=256 | 62 | 23.2 | 23.18 | 23.18 | nan | 0.999 | 23.23 | 1.002 | nvjet_sm90_tst_128x128_64x6_1x1_v_bz_c 23.1 | nvjet_sm90_tst_128x128_64x6_1x1_v_bz_c 23.9 |
| layer=router load=verify 32 m=256 | 62 | 7.35 | 6.89 | 6.89 | nan | 0.937 | 7.00 | 1.016 | rk_kernel(CUtensorMap_st, CUtensorMap_ 5.9 | rk_kernel(CUtensorMap_st, CUtensorMap_ 6.0 |
