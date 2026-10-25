---
problem: muse_attention_decode
slug: restack-sk-pseudo-b1-cluster-c16-c4
created: 2026-10-25T054448Z
base: master
area: "sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)"
change: >
  Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at
  B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A)
  From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job
  94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as
  is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup
  gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed
  only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at
  B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route.
  (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job
  195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU
  (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode
  (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on
  master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's
  stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS
  2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit
  (B) for module/static globals holding device pointers or state-sized buffers and move any into
  per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state
  refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills;
  extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no
  host sync, no allocation sized by input values (shape/config only), no data-dependent Python
  control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from
  inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window
  attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them:
  no module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0
  by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare
  against an fp32 reference with two states and alternating graph replays with refilled inputs
  (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to
  master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed
  TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)"
  - shapes: "*layer=full *"
    direction: flat
    why: "full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: B=2..16 stays on master Triton (C=8 gated off)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-sk-pseudo-b1-cluster-c16-c4

## Change

Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A) From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job 94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route. (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job 195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS 2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit (B) for module/static globals holding device pointers or state-sized buffers and move any into per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills; extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_1 *` faster: s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)
- `*layer=sliding load=decode_32 *` faster: cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)
- `*layer=sliding load=decode_1 *` faster: cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)
- `*layer=full *` flat: full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)
- `*layer=sliding load=decode_16 *` flat: B=2..16 stays on master Triton (C=8 gated off)

## Risk

Landing candidate. Substituting the two cluster rows into s1's measured loads gives ~1.489x (s1 alone 1.4705; master re-measured 1.432-1.457 across three master-equivalent checks today: 2a3241780c7b4c46, f519261b7c2b435f and c0f1945a108c4aa2 all fell back to master on every row). Do not add sk-pdl-from-preattn: its gate chose plain launch (PDL +0.54 / +3.84 us slower), so that run measured master; its full decode_32 199 us row is below its own 229.6 us kernel row, i.e. a timing artefact, not a gain. Risk: the cluster TU was written before the two-state lesson; the audit above is mandatory.
