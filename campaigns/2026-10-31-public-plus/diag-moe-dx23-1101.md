# muse_moe_w4a16: same-day public control (GD-MOE-DX23)

Reads: GD-MOE-DX23 `5d1bc475d2a54baf` 2.3992 [2.3632, 2.4352] on sfo3

Control mean (geomean) **2.3992**, headline spread s_h = nan%. Public published (10-29) row: see RESULTS.md. Our board BEST job `f4e16629be334b14`.

| row | w | public 10-29 us | GD-MOE-DX23 us | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| load=decode 1 bs=1 | 60 | 24.64 | 25.13 | 25.13 | nan | 1.020 | 24.87 | 0.989 | void okc::moe_fused<false>(okc::Params 22.8 | void okc::moe_fused<false>(okc::Params 23.2 |
| load=decode 16 bs=16 | 60 | 92.27 | 91.36 | 91.36 | nan | 0.990 | 91.87 | 1.006 | void okc::moe_fused<false>(okc::Params 104.2 | void okc::moe_fused<false>(okc::Params 84.2 |
| load=decode 32 bs=32 | 60 | 123.96 | 124.76 | 124.76 | nan | 1.006 | 123.73 | 0.992 | void okc::moe_fused<false>(okc::Params 124.5 | void okc::moe_fused<false>(okc::Params 143.6 |
| load=verify 1 bs=8 | 60 | 63.61 | 62.85 | 62.85 | nan | 0.988 | 63.11 | 1.004 | void okc::moe_fused<false>(okc::Params 97.0 | void okc::moe_fused<false>(okc::Params 57.0 |
| load=verify 16 bs=128 | 60 | 175.24 | 169.99 | 169.99 | nan | 0.970 | 175.70 | 1.034 | void okc::moe_fused<true>(okc::Params) 174.7 | void okc::moe_fused<true>(okc::Params) 183.4 |
| load=verify 32 bs=256 | 60 | 207.76 | 207.85 | 207.85 | nan | 1.000 | 208.76 | 1.004 | void okc::moe_fused<true>(okc::Params) 215.7 | void okc::moe_fused<true>(okc::Params) 218.8 |
