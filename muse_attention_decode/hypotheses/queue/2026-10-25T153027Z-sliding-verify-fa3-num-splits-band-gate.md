---
problem: muse_attention_decode
slug: sliding-verify-fa3-num-splits-band-gate
created: 2026-10-25T153027Z
base: master
area: FA3 call (sliding verify split count)
change: >
  Built on current master (no depends_on). Sliding verify at B=16/32 runs FA3 unpaired with
  scheduler_metadata=None, so FA3 picks its own split count on device and launches
  prepare_varlen_num_blocks (1.9/1.8 us), the main kernel (10.7/15.0 us) and FlashAttnFwdCombine
  (4.0/4.2 us) (profile 252abb7f11ec48c3). The window is only 2048 + 7 keys, so a few splits may
  fill the GPU: 32 (B=16) or 64 (B=32) (request, kv-head) m-blocks, each 64 query rows under
  pack_gqa; a combine of ~4 us is 20-28% of FA3's time. num_splits is a CAP under dynamic split
  (LESSONS), so lowering it is effective, and num_splits=1 skips the combine launch. CHANGE: in
  kernel_fn's FA3 branch, for sliding verify only (window != 0 and q_len 8 and not the PSEUDO
  route; config-only), pass num_splits=_SV_SPLITS.get(band, 0) to flash_attn_with_kvcache, band =
  16 for 2 <= B <= 16 and 32 for 17 <= B <= 64; every other call keeps num_splits at its default.
  _SV_SPLITS is an import-time frozen table (MappingProxyType) chosen in prepare: for each band,
  at batch x 80000 agentic lengths, time the full master sliding-verify step after pre-attention
  (FA3 + combine + rms_norm_gate) for num_splits in {0 (auto), 1, 2, 3, 4, 6, 8} as interleaved
  cold-L2 CUDA-graph replays (256 MB flush, empty graph subtracted, median of 31), print every
  number and whether the combine kernel launched, and pick the fastest only if it beats auto by >=
  1.03x and its output matches auto's by rel L2 < 1e-2; otherwise keep auto. Self-check the chosen
  setting on the 16 test-like sliding verify cases at page 1 (B 3/5, kv 0.25, lens < window)
  against master's output. Hold-reason note: fa3-single-split-sliding was held because forcing 1
  split everywhere was expected to lose at verify_1/decode_16; here decode and B=1 are not FA3 any
  more (Triton/cluster/sk routes) and the split count is a measured per-band choice that falls
  back to auto. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine).
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: drop or shrink FlashAttnFwdCombine (4.0 us) at a smaller FA3 cost; shape 22.8 us, floor 1.9
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: combine 4.2 us; 64 m-blocks may need only 2-3 splits; shape 29.2 us
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding load=verify_1 *"
    direction: flat
    why: PSEUDO route, untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-verify-fa3-num-splits-band-gate

## Change

Built on current master (no depends_on). Sliding verify at B=16/32 runs FA3 unpaired with scheduler_metadata=None, so FA3 picks its own split count on device and launches prepare_varlen_num_blocks (1.9/1.8 us), the main kernel (10.7/15.0 us) and FlashAttnFwdCombine (4.0/4.2 us) (profile 252abb7f11ec48c3). The window is only 2048 + 7 keys, so a few splits may fill the GPU: 32 (B=16) or 64 (B=32) (request, kv-head) m-blocks, each 64 query rows under pack_gqa; a combine of ~4 us is 20-28% of FA3's time. num_splits is a CAP under dynamic split (LESSONS), so lowering it is effective, and num_splits=1 skips the combine launch. CHANGE: in kernel_fn's FA3 branch, for sliding verify only (window != 0 and q_len 8 and not the PSEUDO route; config-only), pass num_splits=_SV_SPLITS.get(band, 0) to flash_attn_with_kvcache, band = 16 for 2 <= B <= 16 and 32 for 17 <= B <= 64; every other call keeps num_splits at its default. _SV_SPLITS is an import-time frozen table (MappingProxyType) chosen in prepare: for each band, at batch x 80000 agentic lengths, time the full master sliding-verify step after pre-attention (FA3 + combine + rms_norm_gate) for num_splits in {0 (auto), 1, 2, 3, 4, 6, 8} as interleaved cold-L2 CUDA-graph replays (256 MB flush, empty graph subtracted, median of 31), print every number and whether the combine kernel launched, and pick the fastest only if it beats auto by >= 1.03x and its output matches auto's by rel L2 < 1e-2; otherwise keep auto. Self-check the chosen setting on the 16 test-like sliding verify cases at page 1 (B 3/5, kv 0.25, lens < window) against master's output. Hold-reason note: fa3-single-split-sliding was held because forcing 1 split everywhere was expected to lose at verify_1/decode_16; here decode and B=1 are not FA3 any more (Triton/cluster/sk routes) and the split count is a measured per-band choice that falls back to auto. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=sliding load=verify_16 *` faster: drop or shrink FlashAttnFwdCombine (4.0 us) at a smaller FA3 cost; shape 22.8 us, floor 1.9
- `*layer=sliding load=verify_32 *` faster: combine 4.2 us; 64 m-blocks may need only 2-3 splits; shape 29.2 us
- `*layer=full *` flat: untouched
- `*load=decode_* *` flat: untouched
- `*layer=sliding load=verify_1 *` flat: PSEUDO route, untouched

## Risk

If the gate keeps auto at both bands the run is a free no-op read (the printed table is the result). Weight 46 per row.
