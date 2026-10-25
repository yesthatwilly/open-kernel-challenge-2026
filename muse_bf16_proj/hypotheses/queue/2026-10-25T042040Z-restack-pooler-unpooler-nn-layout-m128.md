---
problem: muse_bf16_proj
slug: restack-pooler-unpooler-nn-layout-m128
created: 2026-10-25T042040Z
base: master
area: pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout)
change: >
  Built on master (b99b1dd, the landed restack). Apply the exact change of branch
  hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the restack: in
  prepare(), store state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] =
  W_unpooler.t().contiguous() (lossless [K, N] bf16 copies; the originals stay). In kernel_fn add
  one branch just before the `if layer != 'router'` cuBLAS fallback: if layer in {pooler,
  unpooler} and 65 <= cfg m <= 128 and cfg n is the full width -> return torch.mm(x, state[layer +
  '_t']). m >= 129 keeps TN; m <= 32 keeps the restack's swap-AB paths. Nothing else changes.
  Record the nvjet NN kernel names for the two target rows.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: TN nvjet 64x64 25.7 us (row 25.55) -> NN splitK 20.4 + reduce 3.0, row 22.98 on master (-10.1%, z -29)
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: row 19.79 -> 19.20 on master (-3%, z -9); small
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m>=129 keeps TN
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m>=129 keeps TN
  - shapes: layer=* load=decode_*
    direction: flat
    why: m<=32 untouched
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
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# restack-pooler-unpooler-nn-layout-m128

## Change

Built on master (b99b1dd, the landed restack). Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the restack: in prepare(), store state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() (lossless [K, N] bf16 copies; the originals stay). In kernel_fn add one branch just before the `if layer != 'router'` cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n is the full width -> return torch.mm(x, state[layer + '_t']). m >= 129 keeps TN; m <= 32 keeps the restack's swap-AB paths. Nothing else changes. Record the nvjet NN kernel names for the two target rows.

## Expected

- `layer=pooler load=verify_16 m=128` faster: TN nvjet 64x64 25.7 us (row 25.55) -> NN splitK 20.4 + reduce 3.0, row 22.98 on master (-10.1%, z -29)
- `layer=unpooler load=verify_16 m=128` faster: row 19.79 -> 19.20 on master (-3%, z -9); small
- `layer=pooler load=verify_32 m=256` flat: m>=129 keeps TN
- `layer=unpooler load=verify_32 m=256` flat: m>=129 keeps TN
- `layer=* load=decode_*` flat: m<=32 untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Kill: pooler verify_16 row not <= 24.0 us (6% under restack) with the NN kernel in the profile. +134 MB device memory, untimed. Estimated headline +0.003 over the restack.
