---
problem: muse_attention_decode
slug: triton-flash-decode-deep-pipeline
created: 2026-10-23T225858Z
base: master
area: FA3 call (Triton pipeline)
change: >
  Stacked on triton-flash-decode-full-both-heads: change only the memory-level parallelism of its
  kernel 1 on full decode steps, leaving the math, the row gather and kernel 2 as landed. Three
  coupled settings: num_stages=4 (from the landed default, 2 or 3) so Triton's software pipeline
  keeps four 64-token K and V blocks (4 x 2 x 8 KB) of cp.async gathers in flight per program;
  BLOCK=128 tokens per iteration (16 KB per K block, 16 KB per V block) if the landed kernel used
  64; and num_splits raised from clamp(ceil(2*132 / B), 1, 128) to clamp(ceil(6*132 / B), 1, 256)
  (at batch 16: 50 splits = 800 programs; at batch 32: 25 splits = 800 programs; at batch 1: 256)
  with num_warps=4, so that three or more programs are resident per SM (shared memory per program
  at 4 stages x 32 KB = 128 KB is too much, so use BLOCK=64 with num_stages=4 = 64 KB, three
  programs per SM) and the combine reads proportionally more partials. The mechanism under test is
  bytes in flight per SM: FA3's persistent warp-specialized kernel keeps one CTA per SM with a
  two-stage pipeline of 8 KB K blocks, about 32 KB in flight per SM and ~4 MB device-wide, which
  at the random-row DRAM latency of a page_size-1 cache bounds throughput near the measured ~1.7
  TB/s; three resident programs with four stages of 16 KB K+V blocks put ~190 KB in flight per SM,
  ~25 MB device-wide. Expose BLOCK, num_stages and the split multiplier as module-level constants
  so a tune companion can bisect.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "388 us vs 126 us floor: if latency-bound on random 128-byte rows, more bytes in flight is the lever; the combine over 25 splits of 16 x 66 fp32 per request is under 1 MB"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 201 us vs 63 us, same mechanism, 50 splits
  - shapes: "*layer=full load=decode_1 page*"
    direction: flat
    why: "256 splits over 80K tokens is ~5 blocks each: pipeline depth cannot help, and the bigger combine may cost a microsecond"
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched; control
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: triton-flash-decode-full-both-heads
priority: 4
kind: primary
controls_for: null
review: required
status: done
hold_reason: "moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)"
outcome: expired
verdict_detail: "hold expired after 33 h: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)"
expired_at: 2026-10-25T211327Z
---

# triton-flash-decode-deep-pipeline

## Change

Stacked on triton-flash-decode-full-both-heads: change only the memory-level parallelism of its kernel 1 on full decode steps, leaving the math, the row gather and kernel 2 as landed. Three coupled settings: num_stages=4 (from the landed default, 2 or 3) so Triton's software pipeline keeps four 64-token K and V blocks (4 x 2 x 8 KB) of cp.async gathers in flight per program; BLOCK=128 tokens per iteration (16 KB per K block, 16 KB per V block) if the landed kernel used 64; and num_splits raised from clamp(ceil(2*132 / B), 1, 128) to clamp(ceil(6*132 / B), 1, 256) (at batch 16: 50 splits = 800 programs; at batch 32: 25 splits = 800 programs; at batch 1: 256) with num_warps=4, so that three or more programs are resident per SM (shared memory per program at 4 stages x 32 KB = 128 KB is too much, so use BLOCK=64 with num_stages=4 = 64 KB, three programs per SM) and the combine reads proportionally more partials. The mechanism under test is bytes in flight per SM: FA3's persistent warp-specialized kernel keeps one CTA per SM with a two-stage pipeline of 8 KB K blocks, about 32 KB in flight per SM and ~4 MB device-wide, which at the random-row DRAM latency of a page_size-1 cache bounds throughput near the measured ~1.7 TB/s; three resident programs with four stages of 16 KB K+V blocks put ~190 KB in flight per SM, ~25 MB device-wide. Expose BLOCK, num_stages and the split multiplier as module-level constants so a tune companion can bisect.

## Expected

- `*layer=full load=decode_32*` faster: 388 us vs 126 us floor: if latency-bound on random 128-byte rows, more bytes in flight is the lever; the combine over 25 splits of 16 x 66 fp32 per request is under 1 MB
- `*layer=full load=decode_16*` faster: 201 us vs 63 us, same mechanism, 50 splits
- `*layer=full load=decode_1 page*` flat: 256 splits over 80K tokens is ~5 blocks each: pipeline depth cannot help, and the bigger combine may cost a microsecond
- `*load=verify_*` flat: untouched; control
- `*layer=sliding*` flat: untouched; control

## Risk

Medium risk: a pure tuning change on a landed kernel, correctness unaffected (reduction order only). Hazards: Triton may refuse num_stages=4 with the gather loads (it pipelines tl.load gathers with cp.async on Hopper, but register pressure for 16 heads x 64 fp32 accumulators plus a 4-deep prefetch may spill); if it does, fall back to num_stages=3 and report that. If the result is flat, the parent kernel is bandwidth- not latency-bound, which is the information this run buys. Eligible only once the parent lands as gain or neutral.
