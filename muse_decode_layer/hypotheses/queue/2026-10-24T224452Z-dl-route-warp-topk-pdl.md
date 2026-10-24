---
problem: muse_decode_layer
slug: dl-route-warp-topk-pdl
created: 2026-10-24T224452Z
base: master
area: routing top-k + MoE align (kernel R, T<=32)
change: >
  Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32
  (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes, nothing else: (a) top-8
  without block barriers: replace the 8-round loop (warp_max_u64 + s_kred + 2 __syncthreads per
  round) with a per-warp top-8: each warp runs 8 rounds of warp_max_u64 over its 32 keys with a
  lane-local taken flag (no block sync), lane r writes its round-r winner to s_cand[warp*8+r]; one
  __syncthreads; warp 0 then picks the global top-8 of the 64 candidates (each lane holds 2, 8
  rounds of warp_max_u64) and writes s_sel[0..7]; one __syncthreads. route_key, tie order (lower
  id wins), weights s/(sum+1e-15), topk_ids, topk_w and local_ids writes unchanged. (b) PDL
  between P and R: in dl_pooler_router_kernel add griddepcontrol.launch_dependents after the main
  K loop of every CTA (before the split-K partial store), and launch R with cudaLaunchKernelEx and
  the programmatic serialization attribute when dl["pdl"] is 1; in R, load norm_w (its 16 values
  per thread) and bias[e] into registers BEFORE griddepcontrol.wait and read ei/logits only after
  it. Also hoist the pad loop: compute each padding slot from s_seg0/s_seg1/s_count by a binary
  search over 32 experts instead of the linear scan. Self-check: the existing _dl_self_check cases
  (routing, near-tie, no-local-slot, align structure, Marlin y, graph replay) must pass unchanged;
  print R CUDA-event time with pdl 0/1 for T 1, 8, 32.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "R 7.6 -> ~3.5 us plus launch overlap with P tail: ~4-5 us of 132.7/142.6 (+3%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: ~4 us of 186.5 (+2%)
  - shapes: "*load=decode_16 *"
    direction: faster
    why: ~4 us of 217 (+1.5-2%)
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~4 us of 247 (+1.5%)
  - shapes: "*load=verify_16 *"
    direction: flat
    why: outside gate
  - shapes: "*load=verify_32 *"
    direction: flat
    why: outside gate
risk: medium
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# dl-route-warp-topk-pdl

## Change

Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32 (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes, nothing else: (a) top-8 without block barriers: replace the 8-round loop (warp_max_u64 + s_kred + 2 __syncthreads per round) with a per-warp top-8: each warp runs 8 rounds of warp_max_u64 over its 32 keys with a lane-local taken flag (no block sync), lane r writes its round-r winner to s_cand[warp*8+r]; one __syncthreads; warp 0 then picks the global top-8 of the 64 candidates (each lane holds 2, 8 rounds of warp_max_u64) and writes s_sel[0..7]; one __syncthreads. route_key, tie order (lower id wins), weights s/(sum+1e-15), topk_ids, topk_w and local_ids writes unchanged. (b) PDL between P and R: in dl_pooler_router_kernel add griddepcontrol.launch_dependents after the main K loop of every CTA (before the split-K partial store), and launch R with cudaLaunchKernelEx and the programmatic serialization attribute when dl["pdl"] is 1; in R, load norm_w (its 16 values per thread) and bias[e] into registers BEFORE griddepcontrol.wait and read ei/logits only after it. Also hoist the pad loop: compute each padding slot from s_seg0/s_seg1/s_count by a binary search over 32 experts instead of the linear scan. Self-check: the existing _dl_self_check cases (routing, near-tie, no-local-slot, align structure, Marlin y, graph replay) must pass unchanged; print R CUDA-event time with pdl 0/1 for T 1, 8, 32.

## Expected

- `*load=decode_1 *` faster: R 7.6 -> ~3.5 us plus launch overlap with P tail: ~4-5 us of 132.7/142.6 (+3%)
- `*load=verify_1 *` faster: ~4 us of 186.5 (+2%)
- `*load=decode_16 *` faster: ~4 us of 217 (+1.5-2%)
- `*load=decode_32 *` faster: ~4 us of 247 (+1.5%)
- `*load=verify_16 *` flat: outside gate
- `*load=verify_32 *` flat: outside gate

## Risk

Medium: a wrong tie order fails routing checks; the near-tie self-check case covers it. The profiler row for R will include its PDL wait when pdl=1, so judge on the printed CUDA-event times and the shape totals.
