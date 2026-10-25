---
problem: muse_decode_layer
slug: dense-chain-s3-t1-q-sk2-o-route-unpool
created: 2026-10-25T054152Z
base: master
area: "whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler)"
change: >
  Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2
  dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on an old base). Built on
  full-decode-sk-ticket-restack-b1-b64, whose two-level stream-K kernel replaces
  FA3+combine+pad/unpad+FusedRMSNormGate at full decode_1; the 04:36Z reviewer noted s2's
  _dl2_attention would otherwise preempt that path, so this stage COMPOSES them instead of
  choosing. Three pieces, each behind its own state flag, selector and fallback, all tokens == 1
  only (T>1 byte-identical to the parent): (A) full decode_1 attention side: dl_qkv_kernel<NF=1>
  (from branch hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate: qkv GEMV + q/k
  RMSNorm + RoPE + FP8 encode + KV store, measured 8.3 us, 4.5 TB/s) writes q in the unpaired
  layout splitkv_attn_sk_kernel<4,true,true> reads, then the sk two-level kernel (norm-gate
  epilogue, 20.1 us), then dl_o_kernel<NF=1> (6.5 us, PDL) into master's residual path; the s2
  measurement was custom Q+O 18.6 vs master attention side sans FA3 27.3 us. Sliding decode_1
  keeps master's _sliding_fused_kernel side (s2 was flat there). (B) from branch
  hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1: per-warp top-8 R with PDL after P (run
  87211e13 selector 87.3 vs 93.0 us summed, R 7.0 -> 5.8 us at T=1), both layers. (C) same branch:
  dl_unpool_kernel at NF=1 (T<=8 code, used here at T=1 only) replacing _gemm(expert_output,
  unpooler[:1024]) + F.pad (6.5 vs 9.2 us); boundaries stay on SGLang
  FusedPostNormResidualPreNorm. Those two runs were on a base without split-KV; hand-merge their
  _DL_CUDA TU edits into master's current dense-chain TU under one new module name, keeping
  master's w2 atomic-add Marlin flags at T=1 untouched. Self-check: Q/O at T=1 full vs master
  incl. KV cache bytes within one FP8 code, R on no-local / one-hot / near-tie, unpool rel L2 <=
  2e-3 with zero columns exactly 0, and the full T=1 chain under alternating-state replays. HARD
  RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except,
  allocation on input values, data-dependent control flow or per-call mutable state; gates read
  config only (layer, tokens/batch, q_len, page); nothing cached from inputs; precision per
  TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES:
  the judge prepares two state copies and rotates them, so tickets, workspaces, device pointers
  and descriptors live in the per-state dict built in prepare(), never in module/static globals.
  Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with
  refilled inputs and compare each to master; a miss turns off only the failing piece (fallback to
  master for it) and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only, print
  regs/smem/local/binaryVersion, canary launch + synchronize with launching/ok prints. Each
  selector prints CUDA-event timing new vs incumbent at T=1 and keeps the incumbent if new is
  slower (LESSONS 03:41Z). First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "parent ~134 us expected; Q/O replaces nvjet qkv+norm/rope/cast/store+nvjet o (-4.7..-8.7 us in s2), R -1.2, unpool -2.7: ~-8..-12 us -> ~123-126. M2: Q+sk2+O chain <= 36 us in prepare timing; kill: shape not 4 us below parent (z>=2)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "R + unpool only (stack run 87211e13: -3.7 us vs its master); 123.9 -> ~120.5. Kill: not below 122.5"
  - shapes: "*load=decode_16*"
    direction: flat
    why: T>1 byte-identical to the parent
  - shapes: "*load=decode_32*"
    direction: flat
    why: T>1 unchanged
  - shapes: "*load=verify_*"
    direction: flat
    why: "verify_1 is T=8: route/unpool gated to tokens == 1 here, so flat"
risk: medium
depends_on: full-decode-sk-ticket-restack-b1-b64
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)"
review: required
status: queued
---

# dense-chain-s3-t1-q-sk2-o-route-unpool

## Change

Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2 dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on an old base). Built on full-decode-sk-ticket-restack-b1-b64, whose two-level stream-K kernel replaces FA3+combine+pad/unpad+FusedRMSNormGate at full decode_1; the 04:36Z reviewer noted s2's _dl2_attention would otherwise preempt that path, so this stage COMPOSES them instead of choosing. Three pieces, each behind its own state flag, selector and fallback, all tokens == 1 only (T>1 byte-identical to the parent): (A) full decode_1 attention side: dl_qkv_kernel<NF=1> (from branch hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate: qkv GEMV + q/k RMSNorm + RoPE + FP8 encode + KV store, measured 8.3 us, 4.5 TB/s) writes q in the unpaired layout splitkv_attn_sk_kernel<4,true,true> reads, then the sk two-level kernel (norm-gate epilogue, 20.1 us), then dl_o_kernel<NF=1> (6.5 us, PDL) into master's residual path; the s2 measurement was custom Q+O 18.6 vs master attention side sans FA3 27.3 us. Sliding decode_1 keeps master's _sliding_fused_kernel side (s2 was flat there). (B) from branch hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1: per-warp top-8 R with PDL after P (run 87211e13 selector 87.3 vs 93.0 us summed, R 7.0 -> 5.8 us at T=1), both layers. (C) same branch: dl_unpool_kernel at NF=1 (T<=8 code, used here at T=1 only) replacing _gemm(expert_output, unpooler[:1024]) + F.pad (6.5 vs 9.2 us); boundaries stay on SGLang FusedPostNormResidualPreNorm. Those two runs were on a base without split-KV; hand-merge their _DL_CUDA TU edits into master's current dense-chain TU under one new module name, keeping master's w2 atomic-add Marlin flags at T=1 untouched. Self-check: Q/O at T=1 full vs master incl. KV cache bytes within one FP8 code, R on no-local / one-hot / near-tie, unpool rel L2 <= 2e-3 with zero columns exactly 0, and the full T=1 chain under alternating-state replays. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only (layer, tokens/batch, q_len, page); nothing cached from inputs; precision per TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so tickets, workspaces, device pointers and descriptors live in the per-state dict built in prepare(), never in module/static globals. Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with refilled inputs and compare each to master; a miss turns off only the failing piece (fallback to master for it) and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only, print regs/smem/local/binaryVersion, canary launch + synchronize with launching/ok prints. Each selector prints CUDA-event timing new vs incumbent at T=1 and keeps the incumbent if new is slower (LESSONS 03:41Z). First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: parent ~134 us expected; Q/O replaces nvjet qkv+norm/rope/cast/store+nvjet o (-4.7..-8.7 us in s2), R -1.2, unpool -2.7: ~-8..-12 us -> ~123-126. M2: Q+sk2+O chain <= 36 us in prepare timing; kill: shape not 4 us below parent (z>=2)
- `*layer=sliding load=decode_1 *` faster: R + unpool only (stack run 87211e13: -3.7 us vs its master); 123.9 -> ~120.5. Kill: not below 122.5
- `*load=decode_16*` flat: T>1 byte-identical to the parent
- `*load=decode_32*` flat: T>1 unchanged
- `*load=verify_*` flat: verify_1 is T=8: route/unpool gated to tokens == 1 here, so flat

## Risk

Three pieces in one stage are attributable by their prepare selector lines and by the sliding decode_1 row (B+C only) vs the full decode_1 row (A+B+C). If the parent did not land, build on its branch anyway (rewrite rule). Composition risk: q layout and FP8 cache encode between dl_qkv_kernel and the sk kernel.

## Milestones

Rewrite stage 3 of `T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
