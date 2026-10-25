---
problem: muse_attention_decode
slug: full-decode1-sk-two-level-fused-preattn
created: 2026-10-25T054448Z
base: master
area: full-layer attention (batch-1 decode, stream-K prologue fusion)
change: >
  Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel:
  ~2.6 us of the row is the second graph node and its launch gap. PDL did not close it (plain
  27.74 vs PDL 28.29/31.58 us, c0f1945a108c4aa2) and cluster merges did not beat the two-level
  ticket (24.9 vs 25.0 us, f519261b7c2b435f). Lever: delete the pre-attention node at B=1, as
  stage 3 (full-decode-splitkv-s3-fused-preattn, 825baa03798f431a) did for 1<B<=64 (where it saved
  7.2 us at B=32). On master's kernel add the FUSED_PRE flag taken from the s3-fused-preattn
  branch and instantiate it with the two-level ticket (TWO=true). Split of the prologue work: q
  (16 heads x 64, RMSNorm + interleaved RoPE + e4m3, from the qkv row and norm weights, ~2 KB) is
  built by each CTA into its smem q slot (132 CTAs x 2 KB of L2 reads, negligible); the new K/V
  row (norm+rope K, FP8 K and V) is computed by the last-tile owner, stored to the cache AND used
  from registers/smem for its own last tile, so no other CTA reads it (only the last tile contains
  key seq-1). Keep the old-key gather unchanged. Self-check: cache rows bit-exact vs master's
  pre-attention kernel, output rel L2 vs fp32 reference at 80000/4097/64/63/1 keys, kv 1.0/0.25,
  grids 3/7/SM. Gate: interleaved graph replay of [fused two-level] vs [pre-attn + two-level] at
  B=1, route at >= 1.03x. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync,
  no allocation sized by input values (shape/config only), no data-dependent Python control flow,
  no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8,
  q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static
  global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer
  (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch,
  hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32
  reference with two states and alternating graph replays with refilled inputs (capture once,
  refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path
  for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "22.6 us shape: remove the 1.4 us pre-attention kernel and its gap; target shape <= 21 us (floor 4.3)"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: B>1 stays on master v2 (this run does not route v3 at B>1)
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: medium
depends_on: full-decode-splitkv-s3-fused-preattn
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>"
review: required
status: queued
---

# full-decode1-sk-two-level-fused-preattn

## Change

Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel: ~2.6 us of the row is the second graph node and its launch gap. PDL did not close it (plain 27.74 vs PDL 28.29/31.58 us, c0f1945a108c4aa2) and cluster merges did not beat the two-level ticket (24.9 vs 25.0 us, f519261b7c2b435f). Lever: delete the pre-attention node at B=1, as stage 3 (full-decode-splitkv-s3-fused-preattn, 825baa03798f431a) did for 1<B<=64 (where it saved 7.2 us at B=32). On master's kernel add the FUSED_PRE flag taken from the s3-fused-preattn branch and instantiate it with the two-level ticket (TWO=true). Split of the prologue work: q (16 heads x 64, RMSNorm + interleaved RoPE + e4m3, from the qkv row and norm weights, ~2 KB) is built by each CTA into its smem q slot (132 CTAs x 2 KB of L2 reads, negligible); the new K/V row (norm+rope K, FP8 K and V) is computed by the last-tile owner, stored to the cache AND used from registers/smem for its own last tile, so no other CTA reads it (only the last tile contains key seq-1). Keep the old-key gather unchanged. Self-check: cache rows bit-exact vs master's pre-attention kernel, output rel L2 vs fp32 reference at 80000/4097/64/63/1 keys, kv 1.0/0.25, grids 3/7/SM. Gate: interleaved graph replay of [fused two-level] vs [pre-attn + two-level] at B=1, route at >= 1.03x. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_1 *` faster: 22.6 us shape: remove the 1.4 us pre-attention kernel and its gap; target shape <= 21 us (floor 4.3)
- `*layer=full load=decode_16 *` flat: B>1 stays on master v2 (this run does not route v3 at B>1)
- `*layer=sliding *` flat: untouched

## Risk

Stage 4 of the fused-preattn line (B=1 extension); builds on master code plus the s3-fused-preattn prologue. Risk: the last-tile owner's new-key build serialises a few hundred ns before its last tile; the per-CTA q build adds L2 reads to every CTA's prologue (v3 measured +3.1 us of build at B=16).

## Milestones

Rewrite stage 4 of `full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
