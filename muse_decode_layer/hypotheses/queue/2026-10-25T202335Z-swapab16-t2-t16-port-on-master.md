---
problem: muse_decode_layer
slug: swapab16-t2-t16-port-on-master
created: 2026-10-25T202335Z
base: master
area: qkv GEMM + o GEMM (2 <= T <= 16)
change: >
  Port ONLY the swap-AB part of qkv-o-swapab16-t2-t16-no-fork onto current master's
  muse_decode_layer/submission.py: the diff between its s3 tip e4b88414 and its impl commit
  3d768883 (git diff e4b88414 3d768883 -- muse_decode_layer/submission.py, +447/-3; saved at the
  hypothesis's notes). It adds the bf16_proj swap-AB wgmma Triton path (_swapab_qkv_kernel and
  _swapab_o_kernel over a shared body, SWAPAB16_CFG qkv_gate row 2304x8192 SPLIT_K 4 and o row
  8192x1024 SPLIT_K 1, ticket split-K reduce, per-STATE fp32 workspace and int32 tickets allocated
  in prepare(), warmed at T=2..16, self-checked against torch.mm at T=2,8,9,16 plus a two-state
  ticket check, any failure turns that layer off and keeps cuBLAS) and routes exactly two calls in
  the T>1 attention helper (qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o,
  dense['o'])) through it when config tokens satisfy 2 <= T <= 16. Master moved since (the 3-way
  apply conflicts around _parent_attention), so place the routing at master's equivalent two call
  sites by hand; T=1 (dense chain s3, on master) and T>=17 keep their master paths; nothing else
  changes.
expected:
  - shapes: "*load=decode_16*"
    direction: faster
    why: "T=16: the parent measured full decode_16 308.7 -> 296.7 us (+5%, in gate); cuBLAS nvjet 64x16 rows ~26 us over 2 calls + splitK 12 us replaced by swap-AB (bf16_proj: qkv -18%, o -10-19% at m<=16)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: T=8 in gate; parent flat within noise here, expected ~3-4 us of ~200 us
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate
  - shapes: "*load=decode_32*"
    direction: flat
    why: T=32 outside the gate
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 outside
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 outside
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "qkv-o-swapab16-t2-t16-no-fork: that run carried the then-unlanded dense-chain-s3 (T=1) rewrite, so its +2.6% headline was mostly s3 (out-of-gate decode_1/32 gains); s3 has since landed, and this ports only the swap-AB part onto current master to measure it alone"
review: required
status: queued
---

# swapab16-t2-t16-port-on-master

## Change

Port ONLY the swap-AB part of qkv-o-swapab16-t2-t16-no-fork onto current master's muse_decode_layer/submission.py: the diff between its s3 tip e4b88414 and its impl commit 3d768883 (git diff e4b88414 3d768883 -- muse_decode_layer/submission.py, +447/-3; saved at the hypothesis's notes). It adds the bf16_proj swap-AB wgmma Triton path (_swapab_qkv_kernel and _swapab_o_kernel over a shared body, SWAPAB16_CFG qkv_gate row 2304x8192 SPLIT_K 4 and o row 8192x1024 SPLIT_K 1, ticket split-K reduce, per-STATE fp32 workspace and int32 tickets allocated in prepare(), warmed at T=2..16, self-checked against torch.mm at T=2,8,9,16 plus a two-state ticket check, any failure turns that layer off and keeps cuBLAS) and routes exactly two calls in the T>1 attention helper (qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o, dense['o'])) through it when config tokens satisfy 2 <= T <= 16. Master moved since (the 3-way apply conflicts around _parent_attention), so place the routing at master's equivalent two call sites by hand; T=1 (dense chain s3, on master) and T>=17 keep their master paths; nothing else changes.

## Expected

- `*load=decode_16*` faster: T=16: the parent measured full decode_16 308.7 -> 296.7 us (+5%, in gate); cuBLAS nvjet 64x16 rows ~26 us over 2 calls + splitK 12 us replaced by swap-AB (bf16_proj: qkv -18%, o -10-19% at m<=16)
- `*load=verify_1 *` faster: T=8 in gate; parent flat within noise here, expected ~3-4 us of ~200 us
- `*load=decode_1 *` flat: T=1 outside the gate
- `*load=decode_32*` flat: T=32 outside the gate
- `*load=verify_16*` flat: T=128 outside
- `*load=verify_32*` flat: T=256 outside

## Risk

Expected headline effect is small (~+0.3-0.5%, one shape at weight 16), below the one-run gain margin (1.24%): this run measures the isolated swap-AB effect on master and uses a scored run that would otherwise be spent on the queue. LESSONS: an earlier decode small-T Triton GEMM lost 2-4.5% on its shapes despite faster kernel sums (possible router side-stream interference, since removed at T<=32). Self-check fallback keeps cuBLAS on any failure, so the correctness risk is low.
