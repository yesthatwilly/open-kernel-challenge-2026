# muse_moe_w4a16: same-day public control (M0-aa1, M0-aa2)

Reads: M0-aa1 `b3510dcc6553442b` 2.3648 [2.3293, 2.4003] on sfo3, M0-aa2 `99b14fedc99142e9` 2.3761 [2.3404, 2.4117] on sfo3c

Control mean (geomean) **2.3704**, headline spread s_h = 0.47%. Public published (10-29) row: see RESULTS.md. Our board BEST job `476d3a0e9dc74d0e`.

| row | w | public 10-29 us | M0-aa1 us | M0-aa2 us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| load=decode 1 bs=1 | 60 | 24.64 | 26.05 | 25.06 | 25.55 | 3.9 | 1.037 | 26.92 | 1.053 | void okc::moe_fused<false>(okc::Params 23.7 | void eak6::ea6_kernel<true, 3, 2, true 17.8; void eak6::ea6_kernel<false, 2, 2, tru 12.7; eak6::ea6_cmb_kernel(__nv_bfloat16 con 2.9 |
| load=decode 16 bs=16 | 60 | 92.27 | 91.41 | 97.36 | 94.38 | 6.3 | 1.023 | 99.92 | 1.059 | void okc::moe_fused<false>(okc::Params 77.8 | void eak4x::ea4x_w13_kernel<3, false>( 67.6; void eak4x::ea4x_w2_kernel<3, false>(_ 40.2; eac::ea_combine_kernel(__nv_bfloat16 c 3.0 |
| load=decode 32 bs=32 | 60 | 123.96 | 125.21 | 124.46 | 124.83 | 0.6 | 1.007 | 134.17 | 1.075 | void okc::moe_fused<false>(okc::Params 116.0 | void eak4x::ea4x_w13_kernel<4, false>( 71.2; void eak4x::ea4x_w2_kernel<4, false>(_ 42.6; eac::ea_combine_kernel(__nv_bfloat16 c 3.2 |
| load=verify 1 bs=8 | 60 | 63.61 | 67.45 | 63.67 | 65.56 | 5.8 | 1.031 | 73.03 | 1.114 | void okc::moe_fused<false>(okc::Params 69.3 | void eak4x::ea4x_w13_kernel<3, false>( 45.3; void eak4x::ea4x_w2_kernel<3, false>(_ 27.5; eac::ea_combine_kernel(__nv_bfloat16 c 3.1 |
| load=verify 16 bs=128 | 60 | 175.24 | 174.98 | 178.49 | 176.74 | 2.0 | 1.009 | 213.97 | 1.211 | void okc::moe_fused<true>(okc::Params) 169.8 | void eak4x::ea4x_w13_kernel<4, false>( 131.8; void eak4x::ea4x_w2_kernel<4, false>(_ 71.1; eac::ea_combine_kernel(__nv_bfloat16 c 12.7 |
| load=verify 32 bs=256 | 60 | 207.76 | 212.55 | 210.75 | 211.65 | 0.9 | 1.019 | 253.13 | 1.196 | void okc::moe_fused<true>(okc::Params) 218.6 | void eak4x::ea4x_w13_kernel<3, false>( 104.7; void eak4x::ea4x_w2_kernel<3, false>(_ 76.9; void (anonymous namespace)::wg_kernel< 52.4 |
