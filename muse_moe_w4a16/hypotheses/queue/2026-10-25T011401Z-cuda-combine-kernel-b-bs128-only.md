---
problem: muse_moe_w4a16
slug: cuda-combine-kernel-b-bs128-only
created: 2026-10-25T011401Z
base: master
area: post-expert norm+combine
change: >
  Port ONLY kernel B (the CUDA post-expert RMSNorm + weighted combine) from
  muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch, and use it only
  for 33 <= num_tokens <= 191 (verify_16 bs=128, the bs=33 and skewed-128 tests) under its own
  constants CUDA_COMBINE_MIN_TOKENS = 33 / CUDA_COMBINE_MAX_TOKENS = 191; at num_tokens >= 192
  (bs=256, 257) keep SGLang's fused_post_expert_norm_weighted_sum, because that run measured
  kernel B slower there (6.2 vs 6.0 us profile; 4.75 vs 3.72 us prepare timing) while faster at
  M=128 (3.7 vs 4.9 us profile; 3.08 vs 3.50 us prepare timing). No PDL (it follows Marlin w2,
  which cannot trigger). Align and everything else stay master. Build with load_inline in prepare
  (no sm_90a instructions); self-check against master's combine at T in {33, 57, 128, 191} with
  random/skewed/all-remote routings (rel-L2 <= 5e-3, finite), else master; print '[cuda-combine-b]
  PATH CHOSEN = CUDA|SGLANG for 33..191 (<reason>)'.
expected:
  - shapes: bs=128 *
    direction: faster
    why: combine row 4.9 -> ~3.7 us (about -1.2 us of 318 us); judge on the combine row
  - shapes: bs=256 *
    direction: flat
    why: "control: SGLang combine kept"
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# cuda-combine-kernel-b-bs128-only

## Change

Port ONLY kernel B (the CUDA post-expert RMSNorm + weighted combine) from muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch, and use it only for 33 <= num_tokens <= 191 (verify_16 bs=128, the bs=33 and skewed-128 tests) under its own constants CUDA_COMBINE_MIN_TOKENS = 33 / CUDA_COMBINE_MAX_TOKENS = 191; at num_tokens >= 192 (bs=256, 257) keep SGLang's fused_post_expert_norm_weighted_sum, because that run measured kernel B slower there (6.2 vs 6.0 us profile; 4.75 vs 3.72 us prepare timing) while faster at M=128 (3.7 vs 4.9 us profile; 3.08 vs 3.50 us prepare timing). No PDL (it follows Marlin w2, which cannot trigger). Align and everything else stay master. Build with load_inline in prepare (no sm_90a instructions); self-check against master's combine at T in {33, 57, 128, 191} with random/skewed/all-remote routings (rel-L2 <= 5e-3, finite), else master; print '[cuda-combine-b] PATH CHOSEN = CUDA|SGLANG for 33..191 (<reason>)'.

## Expected

- `bs=128 *` faster: combine row 4.9 -> ~3.7 us (about -1.2 us of 318 us); judge on the combine row
- `bs=256 *` flat: control: SGLang combine kept
- `bs=1 *` flat: control
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Below headline resolution by design; route-each-shape-to-the-faster-engine (LESSONS prefill stage 4). Cheap, low-risk per-shape part to stack later with cuda-align-kernel-a-from17.
