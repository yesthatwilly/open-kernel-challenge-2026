---
problem: muse_decode_layer
slug: sliding-batched-triton-b16-band
created: 2026-10-25T042121Z
base: master
area: FA3 call (sliding decode, batch 2..16)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Re-apply decode's harvested
  sliding-fused-batched-decode-port (done/neutral patch: the Triton _sliding_fused_kernel batched
  path with 16 splits, ported from attention's landed sliding-deep-splits-on-cuda-splitkv), but
  narrow its config gate from 'sliding, q_len 1, batch > 1' to sliding, q_len 1, page 1, 2 <=
  batch <= 16 (sliding decode_16 and the batch-8/9 tests). batch 17+ (sliding decode_32) stays on
  master FA3 so that band is left for sliding-cluster-c4-decode32-port and the two can later stack
  without overlap. No other change; apply the patch hunks onto master and resolve any conflict in
  kernel_fn by keeping both master's full-decode split-KV route and this sliding route.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "the port moved sliding decode_16 217.1 -> 209.1 us (-3.7%, gained) on its base; master row ~205-210 -> ~200. Weight 46. Kill: not faster than master by z>=2"
  - shapes: "*layer=sliding load=decode_32*"
    direction: flat
    why: outside the narrowed gate
  - shapes: "*load=decode_1 *"
    direction: flat
    why: outside the gate
  - shapes: "*layer=full*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-batched-triton-b16-band

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Re-apply decode's harvested sliding-fused-batched-decode-port (done/neutral patch: the Triton _sliding_fused_kernel batched path with 16 splits, ported from attention's landed sliding-deep-splits-on-cuda-splitkv), but narrow its config gate from 'sliding, q_len 1, batch > 1' to sliding, q_len 1, page 1, 2 <= batch <= 16 (sliding decode_16 and the batch-8/9 tests). batch 17+ (sliding decode_32) stays on master FA3 so that band is left for sliding-cluster-c4-decode32-port and the two can later stack without overlap. No other change; apply the patch hunks onto master and resolve any conflict in kernel_fn by keeping both master's full-decode split-KV route and this sliding route.

## Expected

- `*layer=sliding load=decode_16*` faster: the port moved sliding decode_16 217.1 -> 209.1 us (-3.7%, gained) on its base; master row ~205-210 -> ~200. Weight 46. Kill: not faster than master by z>=2
- `*layer=sliding load=decode_32*` flat: outside the narrowed gate
- `*load=decode_1 *` flat: outside the gate
- `*layer=full*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

Its earlier run's verify_32 'loss' was on untouched paths (drift).
