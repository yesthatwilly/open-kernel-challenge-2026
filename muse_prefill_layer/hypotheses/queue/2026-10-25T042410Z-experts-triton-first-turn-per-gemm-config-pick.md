---
problem: muse_prefill_layer
slug: experts-triton-first-turn-per-gemm-config-pick
created: 2026-10-25T042410Z
base: master
area: "Triton NVFP4 engine at T > 4096 (first_turn): separate tile config for w13 and w2, picked by prepare timing"
change: >
  Built on master (a81d19c). Today both _nvfp4_moe_gemm calls in _experts_triton use the single T
  > 4096 EXPERT_CONFIGS entry (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4, 8 warps, 2 stages),
  although w13 is N=8192 and w2 is N=4096 (half the weight tiles, so half as many CTAs per token
  block and a different L2 reuse pattern), and that config was only ever measured on w13+w2
  together. Change only the config choice, no kernel code: in prepare(), for w13 and for w2
  separately, compile and time a fixed candidate list on a synthetic T=16384 soft routing through
  master's own align (moe_align_block_size over the 32 local experts) and master's
  zero_nonlocal_w2 flag: (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4/8/16, 8 warps, 2 stages),
  (BLOCK_T 256, BLOCK_W 64, GROUP_SIZE_T 8, 4 warps, 3 stages), (BLOCK_T 128, BLOCK_W 128,
  GROUP_SIZE_T 8, 8 warps, 3 stages). Timing: CUDA-graph replays with a cold-L2 flush, 7
  interleaved alternations against master's config; reject any candidate with n_spills > 4 or a
  rel L2 vs master's output above 1e-3. Pick a candidate only at <= 0.97x master's time for that
  GEMM; print the table and PATH CHOSEN per GEMM, asserting the pick is the fastest printed.
  kernel_fn passes state['cfg_w13'] / state['cfg_w2'] (a shape-only branch); the align still pads
  to the w13 BLOCK_T, so a w2 candidate with a different BLOCK_T is only allowed if it divides it
  (128 into 256: two w2 token tiles per aligned block, the sorted_ids layout is unchanged). T <=
  4096 untouched.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel 8.1-9.2 ms -> -3 to -8% if a pick wins (w2 is the likelier mover: 1580 us of the 4640 us measured in the persistent A/B); first_turn shapes -2 to -5%"
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: Marlin path
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: Marlin path
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-first-turn-per-gemm-config-pick

## Change

Built on master (a81d19c). Today both _nvfp4_moe_gemm calls in _experts_triton use the single T > 4096 EXPERT_CONFIGS entry (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4, 8 warps, 2 stages), although w13 is N=8192 and w2 is N=4096 (half the weight tiles, so half as many CTAs per token block and a different L2 reuse pattern), and that config was only ever measured on w13+w2 together. Change only the config choice, no kernel code: in prepare(), for w13 and for w2 separately, compile and time a fixed candidate list on a synthetic T=16384 soft routing through master's own align (moe_align_block_size over the 32 local experts) and master's zero_nonlocal_w2 flag: (BLOCK_T 256, BLOCK_W 128, GROUP_SIZE_T 4/8/16, 8 warps, 2 stages), (BLOCK_T 256, BLOCK_W 64, GROUP_SIZE_T 8, 4 warps, 3 stages), (BLOCK_T 128, BLOCK_W 128, GROUP_SIZE_T 8, 8 warps, 3 stages). Timing: CUDA-graph replays with a cold-L2 flush, 7 interleaved alternations against master's config; reject any candidate with n_spills > 4 or a rel L2 vs master's output above 1e-3. Pick a candidate only at <= 0.97x master's time for that GEMM; print the table and PATH CHOSEN per GEMM, asserting the pick is the fastest printed. kernel_fn passes state['cfg_w13'] / state['cfg_w2'] (a shape-only branch); the align still pads to the w13 BLOCK_T, so a w2 candidate with a different BLOCK_T is only allowed if it divides it (128 into 256: two w2 token tiles per aligned block, the sorted_ids layout is unchanged). T <= 4096 untouched.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel 8.1-9.2 ms -> -3 to -8% if a pick wins (w2 is the likelier mover: 1580 us of the 4640 us measured in the persistent A/B); first_turn shapes -2 to -5%
- `*load=turn_1*` unchanged: Marlin path
- `*load=turn_4*` unchanged: Marlin path

## Risk

Kill: PATH CHOSEN keeps master's config for both GEMMs (then the shipped code is master and the run is a re-measure; read the table only). Prepare timings undercount the judge by 20-25% (LESSONS), so rank, do not predict. Does not touch the K-loop (the running unmasked-loads hypothesis does).
