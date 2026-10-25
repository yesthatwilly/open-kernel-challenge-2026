---
problem: muse_attention_decode
slug: full-verify1-sk-two-level-token-group
created: 2026-10-25T054449Z
base: master
area: full-layer attention (batch-1 verify, stream-K two-level kernel, token grouping)
change: >
  On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us
  shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-gate. At B=1 the full
  two-level decode kernel is latency-bound (18.6 us for 20 MB, 0.89 TB/s), so adding rows to each
  K/V tile is nearly free there, unlike B>1. Route full verify B=1 (window == 0, q_len 8, tp8,
  page 1) through the two-level ticket kernel with TG in {2,4,8} token groups (window = 0: lo_j =
  0, sp_j = cache_seqlens-7+j, causal per token on the last tile; R = 8/TG blocks over 80000 keys,
  each K/V tile loaded once and consumed by TG warps of 16 rows each). TG=8 reads the 20 MB once;
  MMA work is 8x decode's, ~2.6 GFLOP, a few us on mma.sync. Two-level ticket groups are per
  block. Gate: interleaved graph replay of each TG x grid {SM, 2xSM} vs master FA3 path (prepare +
  main + combine + norm-gate; pre-attention excluded from both), route at >= 1.05x. Supersedes the
  queued full-verify1-sk-pseudo-decode (that is TG=1 here; include TG=1 in the sweep for the
  record). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: 30.5 us shape, ~27 us replaced; TG=8 reads 20 MB once on the latency-bound two-level grid (decode_1 does it in 18.6 us); M2 shape <= 26 us, M3 <= 23 us
  - shapes: "*layer=full load=verify_16 *"
    direction: flat
    why: B>1 untouched (MMA-bound, wgmma line owns it)
  - shapes: "*layer=sliding *"
    direction: flat
    why: as in the parent stage
risk: high
depends_on: sliding-verify-sk-pseudo-s2-token-group
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate"
review: required
status: queued
---

# full-verify1-sk-two-level-token-group

## Change

On the token-group branch (sliding-verify-sk-pseudo-s2-token-group). Full verify_1 is 30.5 us shape: FA3 15.4 + combine 7.0 + prepare 2.8 + pre-attn 1.5 + norm-gate. At B=1 the full two-level decode kernel is latency-bound (18.6 us for 20 MB, 0.89 TB/s), so adding rows to each K/V tile is nearly free there, unlike B>1. Route full verify B=1 (window == 0, q_len 8, tp8, page 1) through the two-level ticket kernel with TG in {2,4,8} token groups (window = 0: lo_j = 0, sp_j = cache_seqlens-7+j, causal per token on the last tile; R = 8/TG blocks over 80000 keys, each K/V tile loaded once and consumed by TG warps of 16 rows each). TG=8 reads the 20 MB once; MMA work is 8x decode's, ~2.6 GFLOP, a few us on mma.sync. Two-level ticket groups are per block. Gate: interleaved graph replay of each TG x grid {SM, 2xSM} vs master FA3 path (prepare + main + combine + norm-gate; pre-attention excluded from both), route at >= 1.05x. Supersedes the queued full-verify1-sk-pseudo-decode (that is TG=1 here; include TG=1 in the sweep for the record). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_1 *` faster: 30.5 us shape, ~27 us replaced; TG=8 reads 20 MB once on the latency-bound two-level grid (decode_1 does it in 18.6 us); M2 shape <= 26 us, M3 <= 23 us
- `*layer=full load=verify_16 *` flat: B>1 untouched (MMA-bound, wgmma line owns it)
- `*layer=sliding *` flat: as in the parent stage

## Risk

Depends on stage 2's token-group code; if stage 2 fails to build or self-check, this waits. Risk: 8 row groups per tile need 8 warps or 2 groups per warp; register pressure at TG=8 (print spills, refuse a spill).

## Milestones

Rewrite stage 3 of `full verify batch 1 (q_len 8): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + norm-gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
