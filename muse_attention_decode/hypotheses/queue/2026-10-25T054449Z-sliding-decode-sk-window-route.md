---
problem: muse_attention_decode
slug: sliding-decode-sk-window-route
created: 2026-10-25T054449Z
base: master
area: sliding-layer attention (decode, stream-K windowed kernel)
change: >
  On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is
  already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over
  rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b],
  token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK
  = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB
  at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6
  us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel
  (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is
  _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes;
  the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for
  the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of
  [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >=
  1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv
  0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
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
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: medium
depends_on: sliding-verify-sk-pseudo-decode-s1
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)"
review: required
status: queued
---

# sliding-decode-sk-window-route

## Change

On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b], token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6 us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes; the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >= 1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv 0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=decode_16 *` faster: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
- `*layer=sliding load=decode_32 *` faster: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
- `*layer=sliding load=decode_1 *` flat: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
- `*layer=full *` flat: untouched

## Risk

Kill: if B=16 [pre-attn + sk] is not >= 1.05x faster than Triton in the printed gate, sliding decode stays Triton/cluster. If B=17..32 wins, compare its kernel row with the cluster C=4 row (18.0 us) before a restack picks one. Weight 46 rows: worth ~+1-2% headline per band if it reaches M2.

## Milestones

Rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
