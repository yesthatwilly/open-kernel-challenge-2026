# muse_attention_decode: same-day public control (GD-ATT-D13)

Reads: GD-ATT-D13 `66308d87a316442b` 2.5092 [2.4715, 2.5468] on sfo3c

Control mean (geomean) **2.5092**, headline spread s_h = nan%. Public published (10-29) row: see RESULTS.md. Our board BEST job `1280e6f2120a4fbd`.

| row | w | public 10-29 us | GD-ATT-D13 us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| layer=sliding load=decode 1 batch=1 | 46 | 5.19 | 5.38 | 5.38 | nan | 1.036 | 5.25 | 0.976 | void (anonymous namespace)::attn_kerne 4.6 | void (anonymous namespace)::attn_kerne 4.6 |
| layer=full load=decode 1 batch=1 | 16 | 10.95 | 11.24 | 11.24 | nan | 1.026 | 11.14 | 0.991 | void (anonymous namespace)::attn_kerne 9.7 | void (anonymous namespace)::attn_kerne 9.8 |
| layer=sliding load=decode 16 batch=16 | 46 | 7.91 | 8.45 | 8.45 | nan | 1.069 | 8.28 | 0.979 | void (anonymous namespace)::attn_kerne 7.0 | void (anonymous namespace)::attn_kerne 7.0 |
| layer=full load=decode 16 batch=16 | 16 | 83.65 | 86.64 | 86.64 | nan | 1.036 | 87.28 | 1.007 | void (anonymous namespace)::attn_kerne 83.8 | void (anonymous namespace)::attn_kerne 86.8 |
| layer=sliding load=decode 32 batch=32 | 46 | 9.7 | 10.16 | 10.16 | nan | 1.047 | 10.52 | 1.035 | void (anonymous namespace)::attn_kerne 8.9 | void (anonymous namespace)::attn_kerne 9.1 |
| layer=full load=decode 32 batch=32 | 16 | 159.93 | 166.33 | 166.33 | nan | 1.040 | 166.23 | 0.999 | void (anonymous namespace)::attn_kerne 168.0 | void (anonymous namespace)::attn_kerne 166.1 |
| layer=sliding load=verify 1 batch=1 q_len=8 | 46 | 6.77 | 6.91 | 6.91 | nan | 1.021 | 7.12 | 1.030 | void (anonymous namespace)::attn_kerne 6.5 | void (anonymous namespace)::attn_kerne 6.3 |
| layer=full load=verify 1 batch=1 q_len=8 | 16 | 14.25 | 15.16 | 15.16 | nan | 1.064 | 15.40 | 1.016 | void (anonymous namespace)::attn_kerne 14.5 | void (anonymous namespace)::attn_kerne 14.4 |
| layer=sliding load=verify 16 batch=16 q_len=8 | 46 | 11.77 | 11.96 | 11.96 | nan | 1.016 | 11.98 | 1.001 | void (anonymous namespace)::attn_kerne 11.4 | void (anonymous namespace)::attn_kerne 11.1 |
| layer=full load=verify 16 batch=16 q_len=8 | 16 | 99.98 | 107.14 | 107.14 | nan | 1.072 | 108.13 | 1.009 | void (anonymous namespace)::attn_kerne 105.3 | void (anonymous namespace)::attn_kerne 111.4 |
| layer=sliding load=verify 32 batch=32 q_len=8 | 46 | 14.82 | 15.79 | 15.79 | nan | 1.066 | 15.78 | 1.000 | void (anonymous namespace)::attn_kerne 15.0 | void (anonymous namespace)::attn_kerne 15.4 |
| layer=full load=verify 32 batch=32 q_len=8 | 16 | 182.13 | 195.72 | 195.72 | nan | 1.075 | 198.03 | 1.012 | void (anonymous namespace)::attn_kerne 196.2 | void (anonymous namespace)::attn_kerne 197.7 |
