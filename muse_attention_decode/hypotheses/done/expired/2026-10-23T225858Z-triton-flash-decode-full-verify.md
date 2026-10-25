---
problem: muse_attention_decode
slug: triton-flash-decode-full-verify
created: 2026-10-23T225858Z
base: master
area: FA3 call
change: >
  Stacked on triton-flash-decode-full-both-heads: extend its two-launch Triton flash-decoding pair
  from decode steps to full-attention verify blocks (config window == 0 and q_len == 8), so FA3 is
  no longer called on any full layer. Kernel 1 keeps grid (B, num_splits) (num_splits =
  clamp(ceil(2*132 / B), 1, 128) from shapes) and the whole-row gather, but each program now
  carries 8 x Hq = 128 query rows: it loads Q as [8 tokens x 16 heads, 64] (FP8 query buffer or
  bf16, times sm_scale*qk_scale_factor), and per 64-token block computes two tl.dot's, one per kv
  head, between the 64 rows (8 tokens x 8 heads) of that group and the head's 64-byte half of the
  gathered 128-byte K rows, applies the per-row causal mask t <= seq_lens[b] + j (and t <
  cache_seqlens[b]), runs the online softmax per row, and accumulates P@V from the matching half
  of the V rows. Partials go to a [B, num_splits, 128, 64+2] fp32 scratch; kernel 2, grid (B,
  128), combines the splits and writes bf16 [B*8, Hq, 64] into the buffer rms_norm_gate consumes.
  The new tokens' rows are already in the cache (store before attention, as in the base), so the
  block's own keys are read from the cache like any other and causal-in-block is just t <=
  position. Decode steps keep the landed decode kernels unchanged; sliding layers keep whatever
  the base does.
expected:
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "390 us vs 127 us floor: the same 128-byte-row gather mechanism as the landed decode kernel, now on the shape that dominates the verify_32 load"
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us, same mechanism
  - shapes: "*layer=full load=verify_1 page*"
    direction: flat
    why: "20 MB over 128 splits at 128 query rows: launch-bound; a loss attributes to the extra combine launch, as for the decode kernel"
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched; control
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: high
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

# triton-flash-decode-full-verify

## Change

Stacked on triton-flash-decode-full-both-heads: extend its two-launch Triton flash-decoding pair from decode steps to full-attention verify blocks (config window == 0 and q_len == 8), so FA3 is no longer called on any full layer. Kernel 1 keeps grid (B, num_splits) (num_splits = clamp(ceil(2*132 / B), 1, 128) from shapes) and the whole-row gather, but each program now carries 8 x Hq = 128 query rows: it loads Q as [8 tokens x 16 heads, 64] (FP8 query buffer or bf16, times sm_scale*qk_scale_factor), and per 64-token block computes two tl.dot's, one per kv head, between the 64 rows (8 tokens x 8 heads) of that group and the head's 64-byte half of the gathered 128-byte K rows, applies the per-row causal mask t <= seq_lens[b] + j (and t < cache_seqlens[b]), runs the online softmax per row, and accumulates P@V from the matching half of the V rows. Partials go to a [B, num_splits, 128, 64+2] fp32 scratch; kernel 2, grid (B, 128), combines the splits and writes bf16 [B*8, Hq, 64] into the buffer rms_norm_gate consumes. The new tokens' rows are already in the cache (store before attention, as in the base), so the block's own keys are read from the cache like any other and causal-in-block is just t <= position. Decode steps keep the landed decode kernels unchanged; sliding layers keep whatever the base does.

## Expected

- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor: the same 128-byte-row gather mechanism as the landed decode kernel, now on the shape that dominates the verify_32 load
- `*layer=full load=verify_16*` faster: 205 us vs 63 us, same mechanism
- `*layer=full load=verify_1 page*` flat: 20 MB over 128 splits at 128 query rows: launch-bound; a loss attributes to the extra combine launch, as for the decode kernel
- `*load=decode_*` flat: untouched; control
- `*layer=sliding*` flat: untouched; control

## Risk

High risk only as inherited from the parent kernel; the delta is the M dimension (8 tokens x 16 heads) and the in-block causal mask. Correctness: the 1,048,576-token full verify test (batch 1, page 64) exercises int64 row arithmetic and the page-64 formula; the dp tests (Hq 128, Hkv 16) need the group loop the parent already has; fp32 accumulation over all 80K keys stays (coherent dist). Register pressure is the hazard at 128 query rows x 64 fp32 accumulators per program: use num_warps 8 and BLOCK 64, or split the 8 tokens across two programs per split if Triton spills. Only eligible once the parent lands as gain or neutral; if the parent is flat because the gather, not the row width, is the bottleneck, do not run this.
