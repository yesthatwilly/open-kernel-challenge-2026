---
problem: muse_decode_layer
slug: fa3-num-splits-full-no-metadata
created: 2026-10-23T225545Z
base: master
area: FA3 call
change: >
  On full-attention layers only (cfg['window'] == 0), stop passing the judge-supplied
  scheduler_metadata to flash_attn_with_kvcache (pass None) and instead pass an explicit
  num_splits chosen on the host from the shape: with 2 kv heads and 132 SMs, num_splits =
  clamp(ceil(132 * 2 / (B * 2)), 1, 128) so B=1 gets 128, B=16 gets 9, B=32 gets 5 (more CTAs than
  SMs by ~2x to cover the latency of page-size-1 gathers over an 80K-row context). q_len=8 verify
  blocks on full layers use the same formula (their metadata is already None). The sliding branch
  and everything outside the FA3 call are unchanged. The gap this targets: decode_32 full costs
  SGLang 623 us against 261 us for the sliding twin, i.e. ~360 us for 655 MB of FP8 KV that the
  4.8 TB/s floor moves in 137 us; decode_16 full is 165 us over its sliding twin for a 68 us floor
  delta. Attribution: full shapes only (weight 16 per load); sliding shapes are the control.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "largest absolute gap: ~360 us of FA3 over a 137 us floor; a split count that oversubscribes the SMs ~2x should raise achieved bandwidth"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 165 us over the sliding twin for a 68 us floor delta
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: same KV volume with 8 queries; metadata was already None here so only num_splits changes
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: B=1 is already split heavily by the metadata heuristic; 128 vs its pick likely a wash
  - shapes: "*layer=sliding*"
    direction: flat
    why: sliding branch untouched; control
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-num-splits-full-no-metadata

## Change

On full-attention layers only (cfg['window'] == 0), stop passing the judge-supplied scheduler_metadata to flash_attn_with_kvcache (pass None) and instead pass an explicit num_splits chosen on the host from the shape: with 2 kv heads and 132 SMs, num_splits = clamp(ceil(132 * 2 / (B * 2)), 1, 128) so B=1 gets 128, B=16 gets 9, B=32 gets 5 (more CTAs than SMs by ~2x to cover the latency of page-size-1 gathers over an 80K-row context). q_len=8 verify blocks on full layers use the same formula (their metadata is already None). The sliding branch and everything outside the FA3 call are unchanged. The gap this targets: decode_32 full costs SGLang 623 us against 261 us for the sliding twin, i.e. ~360 us for 655 MB of FP8 KV that the 4.8 TB/s floor moves in 137 us; decode_16 full is 165 us over its sliding twin for a 68 us floor delta. Attribution: full shapes only (weight 16 per load); sliding shapes are the control.

## Expected

- `*layer=full load=decode_32*` faster: largest absolute gap: ~360 us of FA3 over a 137 us floor; a split count that oversubscribes the SMs ~2x should raise achieved bandwidth
- `*layer=full load=decode_16*` faster: 165 us over the sliding twin for a 68 us floor delta
- `*layer=full load=verify_32*` faster: same KV volume with 8 queries; metadata was already None here so only num_splits changes
- `*layer=full load=decode_1*` flat: B=1 is already split heavily by the metadata heuristic; 128 vs its pick likely a wash
- `*layer=sliding*` flat: sliding branch untouched; control

## Risk

Risk: SGLang's scheduler_metadata is FA3's own tile schedule for this exact step and may already be near-optimal; dropping it can make B=16/32 decode SLOWER if the heuristic FA3 then runs on the host sees seqlen_k as the page table extent and picks worse. This run will be read either way: the per-shape result tells whether the metadata or the split count is what limits FA3 at page size 1. The remaining gap (page-size-1 row gathers through the page table) is not touched by num_splits; if full shapes come back flat, the next hypothesis is a different decode kernel (see trtllm-decode-full-attention). Correctness: only the fp32 combine order changes. Graph-safe: num_splits from config only. The batch 1 ctx 1048576 full q_len=8 test still runs through this branch: 128 splits over 8192 tiles is fine, but check its 15-minute budget.
