---
problem: muse_decode_layer
slug: bf16-m32-swapab-wgmma-port
created: 2026-10-24T165242Z
base: master
area: pooler GEMM
change: >
  Port muse_bf16_proj bf16-m32-swapab-wgmma-triton (job 1602a4e947e64090, 1.198x there; per layer
  at m=32: pooler 21.55 -> 20.00 us, qkv_gate 14.13 -> 13.69, o 7.04 -> 6.67, unpooler 18.89 ->
  18.48) to decode_layer dense GEMMs at 17 <= T <= 32, which in the scored set is decode_32 (T=32)
  only. Copy verbatim from muse_bf16_proj/submission.py on master:
  SWAPAB_MIN_M/MAX_M/BLOCK_W/BLOCK_T/NUM_WARPS, SWAPAB_CFG (pooler (4096, 8192, split-K 2, BLOCK_K
  256, 4 stages), qkv_gate (2304, 8192, 4, 128, 4), unpooler (8192, 4096, 1, 256, 4), o (8192,
  1024, 1, 256, 4)), SWAPAB_FALLBACK_STAGES, the OutOfResources import, _bf16_swapab_kernel (64
  weight rows on tl.dot M, tokens on N so Triton emits wgmma m64n32k16; fp32 partials + one
  gpu-scope acq_rel ticket per N-tile, last ticket sums in fixed order and self-resets),
  _swapab_forward and _swapab_prepare (adapted only to read weights from state["dense"][name] and
  keep its workspaces/tickets under its own state keys). In kernel_fn route, by tokens and weight
  shape only: qkv = _swapab_forward("qkv_gate", ...) for attn_input, o GEMM = _swapab_forward("o",
  ...), pooler (main stream, still overlapping the side-stream router) = _swapab_forward("pooler",
  ...), and the full-width unpooler (T > 8) = _swapab_forward("unpooler", ...), each when 17 <=
  tokens <= 32, else master _gemm. prepare() calls _swapab_prepare, which allocates the pooler ws
  [2,32,4096] / cnt [64] and qkv ws [4,32,2304] / cnt [36], picks 4 or 3 stages per layer, warms
  every T in 17..32 for each layer on synthetic bf16 inputs, synchronises and zero_()s every cnt.
  The router (side stream, hi/lo torch.mm), all T <= 16 and T >= 128 GEMMs, attention and MoE are
  untouched.
expected:
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "bf16_proj measured -2.8 us of kernels across the four layers at m=32, and each replaced cuBLAS split-K GEMM drops its splitKreduce node (decode_32 profile: splitKreduce 4.0 + 3.8 us rows, ~2-4 us per nvjet+reduce pair in LESSONS), so ~4-7 us off 252 us (sliding) and 536 us (full): +1.5-2.5% sliding, ~+1% full, ~+1.5% on the decode_32 load"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate
  - shapes: "*load=decode_16 *"
    direction: flat
    why: T=16 outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: T=8/128/256 outside the gate; control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-m32-swapab-wgmma-port

## Change

Port muse_bf16_proj bf16-m32-swapab-wgmma-triton (job 1602a4e947e64090, 1.198x there; per layer at m=32: pooler 21.55 -> 20.00 us, qkv_gate 14.13 -> 13.69, o 7.04 -> 6.67, unpooler 18.89 -> 18.48) to decode_layer dense GEMMs at 17 <= T <= 32, which in the scored set is decode_32 (T=32) only. Copy verbatim from muse_bf16_proj/submission.py on master: SWAPAB_MIN_M/MAX_M/BLOCK_W/BLOCK_T/NUM_WARPS, SWAPAB_CFG (pooler (4096, 8192, split-K 2, BLOCK_K 256, 4 stages), qkv_gate (2304, 8192, 4, 128, 4), unpooler (8192, 4096, 1, 256, 4), o (8192, 1024, 1, 256, 4)), SWAPAB_FALLBACK_STAGES, the OutOfResources import, _bf16_swapab_kernel (64 weight rows on tl.dot M, tokens on N so Triton emits wgmma m64n32k16; fp32 partials + one gpu-scope acq_rel ticket per N-tile, last ticket sums in fixed order and self-resets), _swapab_forward and _swapab_prepare (adapted only to read weights from state["dense"][name] and keep its workspaces/tickets under its own state keys). In kernel_fn route, by tokens and weight shape only: qkv = _swapab_forward("qkv_gate", ...) for attn_input, o GEMM = _swapab_forward("o", ...), pooler (main stream, still overlapping the side-stream router) = _swapab_forward("pooler", ...), and the full-width unpooler (T > 8) = _swapab_forward("unpooler", ...), each when 17 <= tokens <= 32, else master _gemm. prepare() calls _swapab_prepare, which allocates the pooler ws [2,32,4096] / cnt [64] and qkv ws [4,32,2304] / cnt [36], picks 4 or 3 stages per layer, warms every T in 17..32 for each layer on synthetic bf16 inputs, synchronises and zero_()s every cnt. The router (side stream, hi/lo torch.mm), all T <= 16 and T >= 128 GEMMs, attention and MoE are untouched.

## Expected

- `*load=decode_32 *` faster: bf16_proj measured -2.8 us of kernels across the four layers at m=32, and each replaced cuBLAS split-K GEMM drops its splitKreduce node (decode_32 profile: splitKreduce 4.0 + 3.8 us rows, ~2-4 us per nvjet+reduce pair in LESSONS), so ~4-7 us off 252 us (sliding) and 536 us (full): +1.5-2.5% sliding, ~+1% full, ~+1.5% on the decode_32 load
- `*load=decode_1 *` flat: T=1 outside the gate
- `*load=decode_16 *` flat: T=16 outside the gate
- `*load=verify_*` flat: T=8/128/256 outside the gate; control

## Risk

Medium: one load only and an expected effect near the 1.5% drift, so judge it on the decode_32 rows and the replaced kernels own us (nvjet 64x32 / 128x32 splitK rows and their splitKreduce), not the headline. Shape-disjoint from bf16-shape-gated-triton-gemms-small-t (running, T<=16) and touches the same four call sites, so the two combine by 3-way merge whatever each scores. The router is deliberately excluded (swap-AB router read 0.984x in bf16_proj).
