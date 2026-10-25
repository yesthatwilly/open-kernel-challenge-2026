---
problem: muse_bf16_proj
slug: pooler-unpooler-m128-nn-layout-gated
created: 2026-10-25T011407Z
base: master
area: pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
change: >
  Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on
  the older dba4ffd master) to the only rows it won. Measured per shape in that run: pooler
  verify_16 m=128 25.21 -> 22.86 us (1.120x) and unpooler verify_16 m=128 19.65 -> 19.15 (1.043x);
  it LOST qkv_gate verify_32 m=256 (0.941x), and the other m=128/256 rows were flat. Master's
  incumbent on those two rows is cuBLAS TN through _bf16_gemm_dispatch_impl: pooler
  nvjet_sm90_tst_64x64_64x13_2x1 25.7 us kernel (row 25.55, ~2.6 TB/s against a 14.6 us floor),
  unpooler nvjet_sm90_tst_128x64_64x8_1x2 20.4 us (row 19.79). Change: in prepare(), for pooler
  and unpooler ONLY, store a lossless transposed copy state['pooler_t'] =
  W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() ([K, N] bf16
  row-major, same values and dtype; the originals stay for every other path; +134 MB once,
  untimed). In kernel_fn, add one branch placed just before master's cuBLAS fallback: if layer in
  {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n equals the full weight width, return
  torch.mm(x, state[layer + '_t']) (a contiguous bf16 [m, n] with fp32 accumulation inside cuBLAS,
  the same math). Gate only on layer and cfg m/n. Everything else is byte-identical to master: m
  <= 32 Triton paths, the router, qkv_gate and o at every m, and pooler/unpooler at m >= 129 (so
  m=256 stays TN, where the probe showed nothing, and the m=130 and m=1000 unpooler tests stay on
  master). The attempt file records the nvjet kernel name and us for the two target rows from the
  profile.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: NN probe measured 25.21 -> 22.86 us (1.12x); master row 25.55, nvjet 64x64 kernel 25.7 at ~2.6 TB/s vs 14.6 floor
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: NN probe measured 19.65 -> 19.15 (1.04x), small; needs the per-shape row, may read flat
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m >= 129 keeps TN
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m >= 129 keeps TN
  - shapes: layer=pooler load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=unpooler load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: bf16-m128-nn-weight-layout-probe
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-unpooler-m128-nn-layout-gated

## Change

Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master) to the only rows it won. Measured per shape in that run: pooler verify_16 m=128 25.21 -> 22.86 us (1.120x) and unpooler verify_16 m=128 19.65 -> 19.15 (1.043x); it LOST qkv_gate verify_32 m=256 (0.941x), and the other m=128/256 rows were flat. Master's incumbent on those two rows is cuBLAS TN through _bf16_gemm_dispatch_impl: pooler nvjet_sm90_tst_64x64_64x13_2x1 25.7 us kernel (row 25.55, ~2.6 TB/s against a 14.6 us floor), unpooler nvjet_sm90_tst_128x64_64x8_1x2 20.4 us (row 19.79). Change: in prepare(), for pooler and unpooler ONLY, store a lossless transposed copy state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() ([K, N] bf16 row-major, same values and dtype; the originals stay for every other path; +134 MB once, untimed). In kernel_fn, add one branch placed just before master's cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n equals the full weight width, return torch.mm(x, state[layer + '_t']) (a contiguous bf16 [m, n] with fp32 accumulation inside cuBLAS, the same math). Gate only on layer and cfg m/n. Everything else is byte-identical to master: m <= 32 Triton paths, the router, qkv_gate and o at every m, and pooler/unpooler at m >= 129 (so m=256 stays TN, where the probe showed nothing, and the m=130 and m=1000 unpooler tests stay on master). The attempt file records the nvjet kernel name and us for the two target rows from the profile.

## Expected

- `layer=pooler load=verify_16 m=128` faster: NN probe measured 25.21 -> 22.86 us (1.12x); master row 25.55, nvjet 64x64 kernel 25.7 at ~2.6 TB/s vs 14.6 floor
- `layer=unpooler load=verify_16 m=128` faster: NN probe measured 19.65 -> 19.15 (1.04x), small; needs the per-shape row, may read flat
- `layer=pooler load=verify_32 m=256` flat: m >= 129 keeps TN
- `layer=unpooler load=verify_32 m=256` flat: m >= 129 keeps TN
- `layer=pooler load=decode_*` flat: m <= 32 untouched
- `layer=unpooler load=decode_*` flat: m <= 32 untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Low. Pure dispatch, no kernel. The probe's gain came from one run against an older master; LESSONS says cuBLAS rows at m>=128 swing between runs (pooler m=128 nvjet 23.9 vs 25.7 us across runs), so a per-shape gain under ~7% needs a repeat before it counts. The headline effect is small (~+0.5-0.7%: one load of six), so judge on the two rows.
