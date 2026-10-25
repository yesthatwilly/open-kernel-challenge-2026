---
problem: muse_attention_decode
slug: full-decode-sk-s4-deferred-ticket-arrive
created: 2026-10-25T054448Z
base: master
area: full-layer attention (decode 1<B<=64, stream-K ticket epilogue)
change: >
  Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs
  106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.09 TB/s), and 221.5 vs 218.2
  us at B=32 ([sk-diag] in c0f1945a108c4aa2 and f519261b7c2b435f logs): the in-kernel combine
  costs ~6 us at B=16 against the no-ticket kernel, before the separate combine it saves. Where:
  at every segment end inside the tile loop the CTA does a __threadfence + 2 __syncthreads +
  atomicAdd round trip (and the last arrival a serial merge_slots of ~16 slots) while its next
  segment's tiles wait. Change (new template flag DEFER on the single-level ticket path, B>1 only;
  DEFER=false byte-identical): inside the loop a finished non-whole segment only stores its
  partial (ws_acc/ws_ml slot bx + c_b) and records c_b in a small smem list (<= 4 entries; at grid
  264 a CTA holds <= 3 segments for B<=64, assert it); the loop continues streaming with no fence
  or atomic. After the last tile, ONE __threadfence + __syncthreads, then for each recorded
  request draw the ticket (atomicAdd == c_hi - c_lo) and the last arrival merges + finalize_store
  + resets the ticket, exactly as today. Whole segments still finalize in place. Correctness
  argument: the ticket draw still follows the release fence of every partial it counts. Variant
  matrix in prepare: DEFER on/off x grid 2xSM/SM at B=16, B=32, B=64; route DEFER only if it beats
  master's variant 0 by >= 1.02x in interleaved graph replay; B=1 two-level unchanged. RULES
  (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input
  values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes. The judge
  prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
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
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: ticket kernel 110.1 us in profile vs the no-ticket kernel ~6 us faster in prepare; target kernel <= 106 us (floor 63.2)
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: 225.6 us kernel; no-ticket gap ~3 us; target <= 222 us
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: two-level B=1 path untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: medium
depends_on: full-decode1-streamk-s3-two-level-ticket
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue"
review: required
status: queued
---

# full-decode-sk-s4-deferred-ticket-arrive

## Change

Built on master (c3c1b01). Master's <4 stages, ticket> kernel runs 112.3-113.5 us at B=16 vs 106.0-106.7 us for the same stream-K kernel with no ticket (3.07-3.09 TB/s), and 221.5 vs 218.2 us at B=32 ([sk-diag] in c0f1945a108c4aa2 and f519261b7c2b435f logs): the in-kernel combine costs ~6 us at B=16 against the no-ticket kernel, before the separate combine it saves. Where: at every segment end inside the tile loop the CTA does a __threadfence + 2 __syncthreads + atomicAdd round trip (and the last arrival a serial merge_slots of ~16 slots) while its next segment's tiles wait. Change (new template flag DEFER on the single-level ticket path, B>1 only; DEFER=false byte-identical): inside the loop a finished non-whole segment only stores its partial (ws_acc/ws_ml slot bx + c_b) and records c_b in a small smem list (<= 4 entries; at grid 264 a CTA holds <= 3 segments for B<=64, assert it); the loop continues streaming with no fence or atomic. After the last tile, ONE __threadfence + __syncthreads, then for each recorded request draw the ticket (atomicAdd == c_hi - c_lo) and the last arrival merges + finalize_store + resets the ticket, exactly as today. Whole segments still finalize in place. Correctness argument: the ticket draw still follows the release fence of every partial it counts. Variant matrix in prepare: DEFER on/off x grid 2xSM/SM at B=16, B=32, B=64; route DEFER only if it beats master's variant 0 by >= 1.02x in interleaved graph replay; B=1 two-level unchanged. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_16 *` faster: ticket kernel 110.1 us in profile vs the no-ticket kernel ~6 us faster in prepare; target kernel <= 106 us (floor 63.2)
- `*layer=full load=decode_32 *` faster: 225.6 us kernel; no-ticket gap ~3 us; target <= 222 us
- `*layer=full load=decode_1 *` flat: two-level B=1 path untouched
- `*layer=sliding *` flat: untouched

## Risk

Small (~0.3-0.5% headline) but attributable on two weight-16 rows via the kernel row. Composes with v3 fused-preattn later (both are template flags on the same kernel). Risk: deferring arrivals lengthens the time the last request waits for its earliest segments' owners; the gate decides.

## Milestones

Rewrite stage 4 of `full decode 1<B<=64: master splitkv_attn_sk_kernel<4, ticket> single-level in-stream ticket epilogue`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
