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
review: hold
status: hold
hold_reason: "adversarial review: Hold. The swap-AB in-gate effect (T=8/16) was already measured kernel-by-kernel in the parent's profile b0057d2e6cef4d5a: ~1.5-2 us per in-gate shape, ~+0.3% he"
review_at: 2026-10-25T202557Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T202557Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. The swap-AB in-gate effect (T=8/16) was already measured kernel-by-kernel in the parent's profile b0057d2e6cef4d5a: ~1.5-2 us per in-gate shape, ~+0.3% headline, which one scored run cannot resolve against +-1.5% per-shape noise; the expected full decode_16 +5% belongs to the s3/sk-ticket lineage already on master (296.7 us there without swap-AB).

### Justification
Parent profile rows: verify_1 _swapab_qkv 12.7-13.0 us vs nvjet 128x8 splitK 12.0-12.2 + splitKreduce 1.8 (~1-1.5 us saved); _swapab_o 6.2-6.5 vs nvjet 64x8 6.6-7.0 (~0.5 us); decode_16 ~1 us each for qkv and o. Prepare-time selector qkv 1.066x, o 1.004x. In-gate shapes measured flat (sliding verify_1 191.42->191.58, sliding decode_16 204.68->204.72, full verify_1 +1%). The parent's decode_16 +5% compares against base w2-marlin-atomic-add-t1-port, not s3: s3 alone (34a9be06) shows 298.4 us and master's dl3-piece-a run (933d3f82) 296.7 us with the nvjet row still present. The sibling dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32 is held for the same reason; this is its T=2..16 subset.

### Confounders on the current kernel
- dense-chain-s3-t1-q-sk2-o-route-unpool (landed d5595769): source of the parent's decode_16/decode_1/decode_32 gains credited to swap-AB.
- small-t-launch-minimal-dense-chain-pdl (landed f1f2f35): PDL chain after the attention boundary at T<=32; a non-PDL Triton o kernel may lose overlap (candidate for the ~3 us in-context penalty vs bf16_proj standalone).
- combo-measured-rows-slide-batched-preattn-t8-tail (landed ec89382b): new o-input producers and layouts at sliding decode_16; per-shape drift there (+-7% between identical-path runs) exceeds the effect.
- dl3-piece-a-gate-stride-check-fix (landed 6773ec5c): disjoint gate; read full decode_1 with its PATH CHOSEN line.

### Interference and A/B needs
No companion makes this interpretable: the effect is below noise. If the line is worth more spend, time swap-AB vs cuBLAS in prepare on a captured _parent_attention+boundary segment (enable at >=1.02x), explain the ~3 us in-context penalty first, and fold surviving cells into a combination submit.

### Open questions
- Why does the identical _bf16_swapab_kernel run ~3 us (qkv) / ~2.3 us (o) slower in decode than in muse_bf16_proj while nvjet does not?
- Is the scored slot better spent on a queued p1/p2 item with a resolvable effect?
- Re-mint as an in-context-selector variant folded into a combination submit?

