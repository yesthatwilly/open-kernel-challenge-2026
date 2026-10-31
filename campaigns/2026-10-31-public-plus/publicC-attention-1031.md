# muse_attention_decode: public-C arm reads (A0-r1, A0-r2)

Reads: A0-r1 `5b75167625f94994` 2.4361 [2.3995, 2.4853] on sfo3, A0-r2 `3ba805db4ac841f7` 2.4377 [2.4011, 2.4857] on sfo3

Control mean (geomean) **2.4369**, headline spread s_h = 0.07%. Public published (10-29) row: see RESULTS.md. Our board BEST job `4db5f1395ced4b34`.

| row | w | public 10-29 us | A0-r1 us | A0-r2 us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| layer=sliding load=decode 1 batch=1 | 46 | 5.19 | 5.36 | 5.36 | 5.36 | 0.0 | 1.032 | 6.43 | 1.200 | void (anonymous namespace)::attn_kerne 4.6 | void (anonymous namespace)::sliding_s2 6.0 |
| layer=full load=decode 1 batch=1 | 16 | 10.95 | 10.75 | 10.82 | 10.78 | 0.6 | 0.985 | 15.08 | 1.399 | void (anonymous namespace)::attn_kerne 9.8 | void (anonymous namespace)::c320_attn_ 11.4; void (anonymous namespace)::c320_combi 2.8 |
| layer=sliding load=decode 16 batch=16 | 46 | 7.91 | 8.94 | 8.41 | 8.68 | 6.0 | 1.097 | 10.71 | 1.234 | void (anonymous namespace)::attn_kerne 6.9 | void (anonymous namespace)::sliding_s2 8.4 |
| layer=full load=decode 16 batch=16 | 16 | 83.65 | 87.62 | 87.01 | 87.32 | 0.7 | 1.044 | 96.00 | 1.099 | void (anonymous namespace)::attn_kerne 85.9 | void (anonymous namespace)::splitkv_at 93.4; _pre_attention_kernel 2.0 |
| layer=sliding load=decode 32 batch=32 | 46 | 9.7 | 10.60 | 10.70 | 10.65 | 0.9 | 1.098 | 14.81 | 1.390 | void (anonymous namespace)::attn_kerne 9.4 | void (anonymous namespace)::sliding_at 12.9 |
| layer=full load=decode 32 batch=32 | 16 | 159.93 | 165.57 | 166.07 | 165.82 | 0.3 | 1.037 | 175.40 | 1.058 | void (anonymous namespace)::attn_kerne 166.1 | void (anonymous namespace)::splitkv_at 169.8; _pre_attention_kernel 2.3 |
| layer=sliding load=verify 1 batch=1 q_len=8 | 46 | 6.77 | 7.22 | 7.05 | 7.14 | 2.3 | 1.054 | 9.41 | 1.319 | void (anonymous namespace)::attn_kerne 6.4 | void (anonymous namespace)::sliding_s1 8.1 |
| layer=full load=verify 1 batch=1 q_len=8 | 16 | 14.25 | 15.20 | 15.26 | 15.23 | 0.4 | 1.069 | 23.67 | 1.554 | void (anonymous namespace)::attn_kerne 14.5 | void (anonymous namespace)::fvg_kernel 19.8; _pre_attention_kernel 1.4 |
| layer=sliding load=verify 16 batch=16 q_len=8 | 46 | 11.77 | 11.93 | 12.22 | 12.07 | 2.4 | 1.026 | 19.57 | 1.621 | void (anonymous namespace)::attn_kerne 11.4 | void (anonymous namespace)::slidewin_v 13.0; void (anonymous namespace)::sw_combine 7.6; _pre_attention_kernel 2.8 |
| layer=full load=verify 16 batch=16 q_len=8 | 16 | 99.98 | 117.20 | 118.01 | 117.60 | 0.7 | 1.176 | 131.43 | 1.118 | void (anonymous namespace)::attn_kerne 116.9 | void (anonymous namespace)::fvg_kernel 129.2; _pre_attention_kernel 2.8 |
| layer=sliding load=verify 32 batch=32 q_len=8 | 46 | 14.82 | 15.86 | 16.01 | 15.93 | 1.0 | 1.075 | 25.24 | 1.584 | void (anonymous namespace)::attn_kerne 15.1 | void (anonymous namespace)::slidewin_v 18.6; void (anonymous namespace)::sw_combine 13.2; _pre_attention_tiled_kernel 2.5 |
| layer=full load=verify 32 batch=32 q_len=8 | 16 | 182.13 | 213.59 | 214.26 | 213.93 | 0.3 | 1.175 | 240.72 | 1.125 | void (anonymous namespace)::attn_kerne 214.0 | void (anonymous namespace)::fvg_kernel 239.8; _pre_attention_kernel 4.2 |
