---
problem: muse_decode_layer
slug: w2-marlin-atomic-add-t1-port
created: 2026-10-25T042121Z
base: master
area: w2 marlin (T=1)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m
  (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100
  -> 10/100 replays). In _experts (the two moe_wna16_marlin_gemm calls), at num_tokens == 1 ONLY
  (config: every decode_1 shape), build a separate kwargs dict for the SECOND call (w2,
  size_n=size_k=hidden) with use_atomic_add=True and use_fp32_reduce=False; the w13 call and every
  num_tokens > 1 call keep `common` unchanged. Because w2 then accumulates into its C buffer, give
  that path a cache3 that is zeroed (a dedicated slice with .zero_() inside kernel_fn, graph-safe)
  unless sglang's wrapper already zeroes C with atomic add (read the wrapper; do not double-zero).
  Prepare self-check at T=1 vs master's path: rel L2 vs an fp32-dequant reference no worse than
  1.5x master's own, finite, else keep master and print PATH CHOSEN.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "Marlin (2 calls) 40.3-40.9 us at T=1, same w2 shape as moe bs=1 (w2 K-slice fp32 reduce replaced by atomics): expect -3..4 us per decode_1 shape (-2.5%), weights 46+16; also check runs_ns mode counts. Kill: both decode_1 shapes within +-1% (z<2)"
  - shapes: "*load=decode_16*"
    direction: flat
    why: T>1 keeps master kwargs (moe bs 8/16/32 flat)
  - shapes: "*load=decode_32*"
    direction: flat
    why: T>1 keeps master kwargs
  - shapes: "*load=verify_*"
    direction: flat
    why: T>1 keeps master kwargs
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# w2-marlin-atomic-add-t1-port

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100 -> 10/100 replays). In _experts (the two moe_wna16_marlin_gemm calls), at num_tokens == 1 ONLY (config: every decode_1 shape), build a separate kwargs dict for the SECOND call (w2, size_n=size_k=hidden) with use_atomic_add=True and use_fp32_reduce=False; the w13 call and every num_tokens > 1 call keep `common` unchanged. Because w2 then accumulates into its C buffer, give that path a cache3 that is zeroed (a dedicated slice with .zero_() inside kernel_fn, graph-safe) unless sglang's wrapper already zeroes C with atomic add (read the wrapper; do not double-zero). Prepare self-check at T=1 vs master's path: rel L2 vs an fp32-dequant reference no worse than 1.5x master's own, finite, else keep master and print PATH CHOSEN.

## Expected

- `*load=decode_1 *` faster: Marlin (2 calls) 40.3-40.9 us at T=1, same w2 shape as moe bs=1 (w2 K-slice fp32 reduce replaced by atomics): expect -3..4 us per decode_1 shape (-2.5%), weights 46+16; also check runs_ns mode counts. Kill: both decode_1 shapes within +-1% (z<2)
- `*load=decode_16*` flat: T>1 keeps master kwargs (moe bs 8/16/32 flat)
- `*load=decode_32*` flat: T>1 keeps master kwargs
- `*load=verify_*` flat: T>1 keeps master kwargs

## Risk

Shares decode_1 with running stack-route-topk-pdl-unpool-nf1 and dense-chain T1 gate: attribute by the Marlin row and runs_ns modes.
