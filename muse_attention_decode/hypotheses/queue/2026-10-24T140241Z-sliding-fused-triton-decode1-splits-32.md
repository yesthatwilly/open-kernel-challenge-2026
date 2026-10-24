---
problem: muse_attention_decode
slug: sliding-fused-triton-decode1-splits-32
created: 2026-10-24T140241Z
base: master
area: sliding-layer attention (batch-1 decode, split depth)
change: >
  Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only
  shape master routes to _sliding_fused_kernel) use S = 32 splits instead of _num_splits(1, 2) =
  16, so each program owns ceil(2047/32) rounded up to 64 = 64 old keys, exactly one BLOCK_N=64
  tile, and the grid is (1, 2, 32) = 64 CTAs instead of 32 CTAs of two serial tiles. To keep the
  combine compact, the two tl.static_range(S) loops in the last-ticket combine may stay static (32
  iterations) or become a runtime range(S) when S > 16 behind a constexpr branch; the S <= 16
  instantiation must compile unchanged. Workspace and counters already scale with S from config
  (ws_acc 64 x 16 x 64 fp32 = 256 KB). Leave _num_splits for other batches, BLOCK_N, num_warps=4,
  num_stages=2 and everything else as master.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "incumbent 13.12 us shape / 10.5 us kernel against a 0.1 us floor (512 KB); the measured cost of a program is ~1.5-1.8 us per dependent page-table+gather tile (sliding-single-launch-fused-attention: 2/8/16 tiles -> 13.5/23.9/38.5 us), so one tile per program should save ~1.5 us, partly paid back by a 32-way instead of 16-way combine (+~0.3 us); sliding decode_1 is 58% of the decode_1 load at weight 46, so ~1.2 us is ~5% on that load"
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
  - shapes: "*load=decode_16* *load=decode_32* *load=verify_*"
    direction: flat
    why: outside the batch-1 decode gate
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-fused-triton-decode1-splits-32

## Change

Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits instead of _num_splits(1, 2) = 16, so each program owns ceil(2047/32) rounded up to 64 = 64 old keys, exactly one BLOCK_N=64 tile, and the grid is (1, 2, 32) = 64 CTAs instead of 32 CTAs of two serial tiles. To keep the combine compact, the two tl.static_range(S) loops in the last-ticket combine may stay static (32 iterations) or become a runtime range(S) when S > 16 behind a constexpr branch; the S <= 16 instantiation must compile unchanged. Workspace and counters already scale with S from config (ws_acc 64 x 16 x 64 fp32 = 256 KB). Leave _num_splits for other batches, BLOCK_N, num_warps=4, num_stages=2 and everything else as master.

## Expected

- `*layer=sliding load=decode_1*` faster: incumbent 13.12 us shape / 10.5 us kernel against a 0.1 us floor (512 KB); the measured cost of a program is ~1.5-1.8 us per dependent page-table+gather tile (sliding-single-launch-fused-attention: 2/8/16 tiles -> 13.5/23.9/38.5 us), so one tile per program should save ~1.5 us, partly paid back by a 32-way instead of 16-way combine (+~0.3 us); sliding decode_1 is 58% of the decode_1 load at weight 46, so ~1.2 us is ~5% on that load
- `*layer=full*` flat: untouched
- `*load=decode_16* *load=decode_32* *load=verify_*` flat: outside the batch-1 decode gate

## Risk

Cheap single-knob probe of split depth on the one shape the fused kernel owns today; it also calibrates how far deep splitting goes before the combine and the atomics cost more than a tile, which informs sliding-fused-triton-decode-batched-deep-splits (B>1, S=16; disjoint shapes, the two can be 3-way merged). Risk: 64 partials of fresh tiles may be dominated by the fixed seq/page-table/gather/ticket latency chain, so the gain may be under 1 us (within ~1.5% per-shape drift); read the kernel row in the profile.
