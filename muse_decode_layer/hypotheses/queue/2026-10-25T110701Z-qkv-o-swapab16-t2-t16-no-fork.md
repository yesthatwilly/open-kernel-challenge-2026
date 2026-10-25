---
problem: muse_decode_layer
slug: qkv-o-swapab16-t2-t16-no-fork
created: 2026-10-25T110701Z
base: master
area: qkv GEMM + o GEMM (2 <= T <= 16)
change: >
  Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend:
  qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sites in muse_decode_layer
  kernel_fn: qkv = _gemm(data['attn_input'], dense['qkv_gate']) (n 2304, k 8192) and attn_output =
  _gemm(o, dense['o']) (n 8192, k 1024), gated on config tokens 2 <= T <= 16 (scored: verify_1 T=8
  and decode_16 T=16, both layer types; tests at batch 8/9 too). T=1 is left to the in-flight
  dense-chain-s3 (which rewrites the T=1 qkv/o side) and T >= 17 keeps cuBLAS (that band is the
  held prefork-qkv-o-swapab-t16-t32, whose side-stream-fork premise is stale). Copy verbatim from
  muse_bf16_proj/submission.py on master: _bf16_swapab_kernel (do_not_specialize M, ticket split-K
  reduce), SWAPAB_BLOCK_W=64, SWAPAB_NUM_WARPS=4, SWAPAB16_BLOCK_T=SWAPAB16_WS_M=16, the qkv_gate
  row (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages) and the o row (8192, 1024, SPLIT_K 1, BLOCK_K
  256, 4 stages) of SWAPAB16_CFG, SWAPAB_FALLBACK_STAGES=3 with the OutOfResources import guard,
  and _swapab16_forward. For attribution give each call its own kernel name in the profile: two
  @triton.jit copies (or thin wrappers over a shared @triton.jit body) named _swapab_qkv_kernel
  and _swapab_o_kernel. prepare(): per STATE (the judge prepares two state copies and rotates
  them) allocate the qkv fp32 split-K workspace [16, 2304]-shaped as bf16 does and a zeroed int32
  ticket vector, store them in state (never module globals; o has SPLIT_K 1 and needs none), warm
  each kernel at every T in 2..16 so the timed call never compiles, print '[swapab16-dl]
  layer=qkv|o T=.. stages=.. n_regs=.. wgmma=.. ON|OFF', and self-check each layer against
  torch.mm on random bf16 inputs at T = 2, 8, 9, 16 with rel L2 <= 1e-2, then once more after
  alternating calls on state 0 and state 1 with refilled inputs (ticket reset check). Any failure
  or OutOfResources at both stage counts turns that layer off (cuBLAS _gemm kept). kernel_fn: a
  branch on cfg tokens and the state flag only; nothing else changes (attention, boundaries, dense
  chain P/R, Marlin, unpooler untouched). WHY NOW: LESSONS records that decode's earlier
  bf16-shape-gated-triton-gemms-small-t won kernel sums at T=8/16 yet lost 2-4.5% on those shapes,
  'possibly interference with the side-stream router overlap (unverified)'.
  small-t-launch-minimal-dense-chain-pdl has since removed the router side stream at T <= 32, so
  this run is also the direct test of that open lesson: per-layer kernel rows faster AND shapes
  faster confirms it; rows faster but shapes flat/slower refutes it.
expected:
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: cuBLAS qkv nvjet_sm90_tst_128x8_64x12_4x1 splitK 13.5 us + splitKreduce (~1.7) and o rows (nvjet 64x8 ~7 us) in profile 3fd3fa1f; bf16 m16 swap-AB gave qkv -18%, o -10-19% at this m -> ~3-4 us of 191/202 us, ~2%; read the _swapab_qkv_kernel / _swapab_o_kernel rows against the cuBLAS rows they replace"
  - shapes: "*load=decode_16*"
    direction: faster
    why: "T=16: nvjet 64x16 2x1 (26.2 us over 2 calls) + 64x16 4x1 splitK 12.2 us in profile 3fd3fa1f; same per-kernel saving, ~3-5 us of 205/309 us (sliding weight 46)"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate (dense-chain-s3 territory)
  - shapes: "*load=decode_32*"
    direction: flat
    why: T=32 outside the gate
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 outside the gate
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 outside the gate
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-o-swapab16-t2-t16-no-fork

## Change

Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend: qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sites in muse_decode_layer kernel_fn: qkv = _gemm(data['attn_input'], dense['qkv_gate']) (n 2304, k 8192) and attn_output = _gemm(o, dense['o']) (n 8192, k 1024), gated on config tokens 2 <= T <= 16 (scored: verify_1 T=8 and decode_16 T=16, both layer types; tests at batch 8/9 too). T=1 is left to the in-flight dense-chain-s3 (which rewrites the T=1 qkv/o side) and T >= 17 keeps cuBLAS (that band is the held prefork-qkv-o-swapab-t16-t32, whose side-stream-fork premise is stale). Copy verbatim from muse_bf16_proj/submission.py on master: _bf16_swapab_kernel (do_not_specialize M, ticket split-K reduce), SWAPAB_BLOCK_W=64, SWAPAB_NUM_WARPS=4, SWAPAB16_BLOCK_T=SWAPAB16_WS_M=16, the qkv_gate row (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages) and the o row (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages) of SWAPAB16_CFG, SWAPAB_FALLBACK_STAGES=3 with the OutOfResources import guard, and _swapab16_forward. For attribution give each call its own kernel name in the profile: two @triton.jit copies (or thin wrappers over a shared @triton.jit body) named _swapab_qkv_kernel and _swapab_o_kernel. prepare(): per STATE (the judge prepares two state copies and rotates them) allocate the qkv fp32 split-K workspace [16, 2304]-shaped as bf16 does and a zeroed int32 ticket vector, store them in state (never module globals; o has SPLIT_K 1 and needs none), warm each kernel at every T in 2..16 so the timed call never compiles, print '[swapab16-dl] layer=qkv|o T=.. stages=.. n_regs=.. wgmma=.. ON|OFF', and self-check each layer against torch.mm on random bf16 inputs at T = 2, 8, 9, 16 with rel L2 <= 1e-2, then once more after alternating calls on state 0 and state 1 with refilled inputs (ticket reset check). Any failure or OutOfResources at both stage counts turns that layer off (cuBLAS _gemm kept). kernel_fn: a branch on cfg tokens and the state flag only; nothing else changes (attention, boundaries, dense chain P/R, Marlin, unpooler untouched). WHY NOW: LESSONS records that decode's earlier bf16-shape-gated-triton-gemms-small-t won kernel sums at T=8/16 yet lost 2-4.5% on those shapes, 'possibly interference with the side-stream router overlap (unverified)'. small-t-launch-minimal-dense-chain-pdl has since removed the router side stream at T <= 32, so this run is also the direct test of that open lesson: per-layer kernel rows faster AND shapes faster confirms it; rows faster but shapes flat/slower refutes it.

## Expected

- `*load=verify_1 *` faster: T=8: cuBLAS qkv nvjet_sm90_tst_128x8_64x12_4x1 splitK 13.5 us + splitKreduce (~1.7) and o rows (nvjet 64x8 ~7 us) in profile 3fd3fa1f; bf16 m16 swap-AB gave qkv -18%, o -10-19% at this m -> ~3-4 us of 191/202 us, ~2%; read the _swapab_qkv_kernel / _swapab_o_kernel rows against the cuBLAS rows they replace
- `*load=decode_16*` faster: T=16: nvjet 64x16 2x1 (26.2 us over 2 calls) + 64x16 4x1 splitK 12.2 us in profile 3fd3fa1f; same per-kernel saving, ~3-5 us of 205/309 us (sliding weight 46)
- `*load=decode_1 *` flat: T=1 outside the gate (dense-chain-s3 territory)
- `*load=decode_32*` flat: T=32 outside the gate
- `*load=verify_16*` flat: T=128 outside the gate
- `*load=verify_32*` flat: T=256 outside the gate

## Risk

Headline effect is small (~0.3-0.6%), inside drift; judge on the two named kernel rows and the paired per-shape ratios. If rows are faster but verify_1/decode_16 are flat or slower, record that the T=8/16 loss was NOT the side-stream (the side stream is gone) and stop porting bf16 small-m GEMMs to this layer. Overlaps the held prefork-qkv-o-swapab-t16-t32 only at T=16; if this lands, retire that file or narrow it to 17..32. Shape-disjoint from dense-chain-s3 (T=1) and sliding-cluster-c4-s3 (attention only), so it can be restacked with either.
