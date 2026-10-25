---
problem: muse_attention_decode
slug: restack-sk-pseudo-cluster-plus-v3-fused-preattn
created: 2026-10-25T054448Z
base: master
area: full-layer attention (decode 1<B<=64 prologue) on top of the sliding restack
change: >
  Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster
  C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the stage-3 fused pre-attention
  prologue from branch hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn (job
  825baa03798f431a): the v3 instantiation of splitkv_attn_sk_kernel that builds the FP8 q (RMSNorm
  + RoPE) and stores the new K/V row in-kernel, removing the _pre_attention_kernel launch, gated
  to full decode 1<B<=64 only. Full decode B=1 keeps master's two-level ticket with the separate
  pre-attention kernel (v3 has no two-level variant). Port v3 as a NEW template flag (FUSED_PRE,
  default false) on master's kernel so variants 0-3 and PSEUDO stay byte-identical (print sk_attrs
  IDENTICAL for the landed variants, as s1 does); v3 keeps its own self-check (cache rows
  bit-exact vs the pre-attention kernel, rel L2 vs fp32 reference) and its own bound constant. Add
  a gate: route v3 per band (B=2..16 from the B=16 timing, B=17..64 from B=32) only if the printed
  interleaved graph-replay time of [v3] beats [pre-attn + v2 ticket] by >= 1.02x (v3 measured
  115.1 vs 118.2 us at B=16 and 220.5 vs 227.7 at B=32 in its diag). This run is the A/B partner
  of restack-sk-pseudo-b1-cluster-c16-c4 dispatched in the same batch: the only difference is v3,
  so the full decode_16/32 rows pair directly. RULES (every path): CUDA-graph capturable;
  kernel_fn does no host sync, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state;
  nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key
  in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and
  rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA
  map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and
  returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route
  self-checks in prepare against an fp32 reference with two states and alternating graph replays
  with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on
  any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing
  case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary,
  sync+print after each self-check launch). Gates route a band only when the printed interleaved
  graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path
  winning by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "v3 measured 110.91 -> 109.55 us (-1.2%) with _pre_attention_kernel gone; must beat the partner run's row"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: v3 measured 226.81 -> 223.11 us (-1.6%); diag saved 7.2 us at B=32
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: same as partner (pseudo-decode B=1)
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: same as partner (cluster C=4)
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: B=1 keeps master two-level ticket
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-sk-pseudo-cluster-plus-v3-fused-preattn

## Change

Exactly restack-sk-pseudo-b1-cluster-c16-c4 (pseudo-decode sliding verify B=1 + sliding cluster C=16 at B=1 and C=4 at B=17..32, on master c3c1b01), PLUS the stage-3 fused pre-attention prologue from branch hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn (job 825baa03798f431a): the v3 instantiation of splitkv_attn_sk_kernel that builds the FP8 q (RMSNorm + RoPE) and stores the new K/V row in-kernel, removing the _pre_attention_kernel launch, gated to full decode 1<B<=64 only. Full decode B=1 keeps master's two-level ticket with the separate pre-attention kernel (v3 has no two-level variant). Port v3 as a NEW template flag (FUSED_PRE, default false) on master's kernel so variants 0-3 and PSEUDO stay byte-identical (print sk_attrs IDENTICAL for the landed variants, as s1 does); v3 keeps its own self-check (cache rows bit-exact vs the pre-attention kernel, rel L2 vs fp32 reference) and its own bound constant. Add a gate: route v3 per band (B=2..16 from the B=16 timing, B=17..64 from B=32) only if the printed interleaved graph-replay time of [v3] beats [pre-attn + v2 ticket] by >= 1.02x (v3 measured 115.1 vs 118.2 us at B=16 and 220.5 vs 227.7 at B=32 in its diag). This run is the A/B partner of restack-sk-pseudo-b1-cluster-c16-c4 dispatched in the same batch: the only difference is v3, so the full decode_16/32 rows pair directly. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_16 *` faster: v3 measured 110.91 -> 109.55 us (-1.2%) with _pre_attention_kernel gone; must beat the partner run's row
- `*layer=full load=decode_32 *` faster: v3 measured 226.81 -> 223.11 us (-1.6%); diag saved 7.2 us at B=32
- `*layer=sliding load=verify_1 *` faster: same as partner (pseudo-decode B=1)
- `*layer=sliding load=decode_32 *` faster: same as partner (cluster C=4)
- `*layer=full load=decode_1 *` flat: B=1 keeps master two-level ticket

## Risk

If both runs land, the partner is the cleaner submit unless this one's full decode_16/32 rows beat it by more than the 1.5% drift band on the kernel rows. Risk: three edits to one TU (PSEUDO + FUSED_PRE on the same kernel); keep each behind if constexpr and print attrs for every instantiation.
