---
problem: muse_attention_decode
slug: sliding-decode1-perkvhead-launch-select
created: 2026-10-26T222249Z
base: master
area: sliding-layer attention (batch-1 decode, cluster kernel launch configuration)
change: >
  Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master; that branch
  keeps master's per-kvhead sliding decode_1 path byte for byte). Sliding decode_1 runs
  sliding_cluster_kernel<PERHEAD=true> as 2 clusters of C = 16 (a non-portable cluster size), 9.6
  us of kernel in a 10.81 us shape. Two launch configurations of that same kernel were never
  timed; add them as candidates inside _slide_ph_init's existing step (5) (interleaved cold-L2
  graph replays at B=1 x 80000, median of 31, empty graph subtracted): (a) PERHEAD with C = 8 (2
  clusters of 8, a portable size; each rank owns 256 keys as two pipelined SK=128 sub-chunks; the
  kernel already supports it: NR = max(1, G / C) = 1 row per rank, slide_ph_attn accepts C in {4,
  8, 16}); extend slide_ph_setup's info with occupancy_of<true>(8) and require >= 2 co-resident
  clusters; (b) PERHEAD with C = 16 launched with an extra
  cudaLaunchAttributeClusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread (a new
  trailing int64 arg policy on slide_ph_attn, default 0 = today's launch, attribute added only
  when policy != 0). Each candidate first passes the same eager self-check as PERHEAD C=16 today
  (B=1, lens 1 / 10 / 2047 / 2048 / 2049 / 80000 at kv 0.25 and 2049 at 1.0, vs the FP32 reference
  and vs master C=16, new cache rows compared) and the two-state replay through kernel_fn; route
  the fastest only if it beats today's PERHEAD C=16 by >= 1.03x in two interleaved rounds, else
  keep PERHEAD C=16 exactly. Freeze the choice at import in module globals (_SLIDE_PH_C,
  _SLIDE_PH_POLICY; config-only use in _slide_forward), and print all three timings in the PATH
  CHOSEN line.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: a portable 8-CTA cluster or the Spread policy may shave cluster launch / placement latency off a 9.6 us latency-bound kernel; 0.98%/us, so 0.5 us = +0.5%
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: B=2..16 stays on Triton
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: low
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 4
kind: primary
controls_for: null
prior: "sliding-decode1-cluster-per-kvhead: it split the C=16 cluster per kv head (+9.9% on sliding decode_1); per-kvhead C=8 clusters and the Spread cluster scheduling policy were never timed"
review: required
status: queued
---

# sliding-decode1-perkvhead-launch-select

## Change

Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master; that branch keeps master's per-kvhead sliding decode_1 path byte for byte). Sliding decode_1 runs sliding_cluster_kernel<PERHEAD=true> as 2 clusters of C = 16 (a non-portable cluster size), 9.6 us of kernel in a 10.81 us shape. Two launch configurations of that same kernel were never timed; add them as candidates inside _slide_ph_init's existing step (5) (interleaved cold-L2 graph replays at B=1 x 80000, median of 31, empty graph subtracted): (a) PERHEAD with C = 8 (2 clusters of 8, a portable size; each rank owns 256 keys as two pipelined SK=128 sub-chunks; the kernel already supports it: NR = max(1, G / C) = 1 row per rank, slide_ph_attn accepts C in {4, 8, 16}); extend slide_ph_setup's info with occupancy_of<true>(8) and require >= 2 co-resident clusters; (b) PERHEAD with C = 16 launched with an extra cudaLaunchAttributeClusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread (a new trailing int64 arg policy on slide_ph_attn, default 0 = today's launch, attribute added only when policy != 0). Each candidate first passes the same eager self-check as PERHEAD C=16 today (B=1, lens 1 / 10 / 2047 / 2048 / 2049 / 80000 at kv 0.25 and 2049 at 1.0, vs the FP32 reference and vs master C=16, new cache rows compared) and the two-state replay through kernel_fn; route the fastest only if it beats today's PERHEAD C=16 by >= 1.03x in two interleaved rounds, else keep PERHEAD C=16 exactly. Freeze the choice at import in module globals (_SLIDE_PH_C, _SLIDE_PH_POLICY; config-only use in _slide_forward), and print all three timings in the PATH CHOSEN line.

## Expected

- `*layer=sliding load=decode_1 *` faster: a portable 8-CTA cluster or the Spread policy may shave cluster launch / placement latency off a 9.6 us latency-bound kernel; 0.98%/us, so 0.5 us = +0.5%
- `*layer=sliding load=decode_16 *` flat: B=2..16 stays on Triton
- `*layer=full *` flat: untouched

## Risk

Low prior (per-kvhead's win came from MORE CTAs, which C = 8 halves), so this is mostly a cheap measurement of two never-timed launch knobs on the highest-rate shape (sliding decode_1, 0.98%/us); it routes only on a >= 1.03x two-round win, otherwise the run is v3chain. Decode-portable (the d1s cluster port).
