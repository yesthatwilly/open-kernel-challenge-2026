---
problem: muse_decode_layer
slug: bf16-shape-gated-triton-gemms-small-t
created: 2026-10-24T140116Z
base: master
area: pooler GEMM
change: >
  Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code
  is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel, _pooler_forward,
  _qkv_splitk_sem_kernel/_qkv_forward, _unpool_forward, _o_skinny_gemm and their constants
  POOLER_* (BLOCK_K 256, SPLIT_K 2), QKV_* (SPLIT_K 4), UNPOOL_* (SPLIT_K 8), O_* (full K, 128
  programs)) to muse_decode_layer's own dense GEMMs, which have exactly bf16_proj's shapes:
  qkv_gate [T,8192]x[2304,8192]^T, o [T,1024]x[8192,1024]^T, pooler [T,8192]x[4096,8192]^T,
  unpooler slice [T,4096]x[1024,4096]^T (rows 0-1023, row stride 4096). Copy the kernels and
  launch helpers verbatim; the area is all four dense-GEMM call sites in kernel_fn (labelled
  pooler GEMM, the largest of them). Routing, by tokens and weight shape only (no data-dependent
  branch), mirroring the combo's gates: qkv = _qkv_forward at 1 <= T <= 16, else _gemm; o =
  _o_skinny_gemm at T <= 16, else _gemm; pooler (main stream, still overlapping the side-stream
  router) = _pooler_forward at T <= 16, else _gemm; unpooler slice = _unpool_forward on
  dense['unpooler'][:UNPOOL_SLICE] at T <= 8 (followed by the existing F.pad all-gather), else
  unchanged. T >= 32 (decode_32, verify_16, verify_32) keeps _bf16_gemm_dispatch_impl everywhere
  as the control. Each path keeps its own fp32 split-K workspace and int32 ticket counter (pooler
  ws [2,16,4096] cnt [64]; qkv ws [4,16,2304] cnt [36]; unpooler ws [8,16,1024] cnt [16]),
  allocated in prepare(); prepare() warms every routed specialization on synthetic bf16 inputs (T
  1..16 for qkv/o/pooler, 1..8 for the unpooler slice), synchronises and zero_()s every cnt, as
  the combo's prepare() does. The router (side stream, hi/lo torch.mm) and everything else are
  untouched.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "T=1, the combo measured per GEMM: pooler 21.2->19.2, qkv 14.4->13.2, unpooler slice 7.1->5.8, o 6.7->6.6 us: ~4.5 us off 173/182 us, +2-3%"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: pooler 21.3->19.6, unpooler 7.0->6.5, qkv 14.0->13.5, o flat: ~2.5-3 us off 238/248 us, ~+1%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "T=16: pooler 21.3->19.5, o 6.8->6.4, qkv flat: ~2-2.5 us off 260/407 us, ~+0.5-1%, near drift"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: cuBLAS kept"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: control
  - shapes: "*load=verify_32 *"
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-shape-gated-triton-gemms-small-t

## Change

Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel, _pooler_forward, _qkv_splitk_sem_kernel/_qkv_forward, _unpool_forward, _o_skinny_gemm and their constants POOLER_* (BLOCK_K 256, SPLIT_K 2), QKV_* (SPLIT_K 4), UNPOOL_* (SPLIT_K 8), O_* (full K, 128 programs)) to muse_decode_layer's own dense GEMMs, which have exactly bf16_proj's shapes: qkv_gate [T,8192]x[2304,8192]^T, o [T,1024]x[8192,1024]^T, pooler [T,8192]x[4096,8192]^T, unpooler slice [T,4096]x[1024,4096]^T (rows 0-1023, row stride 4096). Copy the kernels and launch helpers verbatim; the area is all four dense-GEMM call sites in kernel_fn (labelled pooler GEMM, the largest of them). Routing, by tokens and weight shape only (no data-dependent branch), mirroring the combo's gates: qkv = _qkv_forward at 1 <= T <= 16, else _gemm; o = _o_skinny_gemm at T <= 16, else _gemm; pooler (main stream, still overlapping the side-stream router) = _pooler_forward at T <= 16, else _gemm; unpooler slice = _unpool_forward on dense['unpooler'][:UNPOOL_SLICE] at T <= 8 (followed by the existing F.pad all-gather), else unchanged. T >= 32 (decode_32, verify_16, verify_32) keeps _bf16_gemm_dispatch_impl everywhere as the control. Each path keeps its own fp32 split-K workspace and int32 ticket counter (pooler ws [2,16,4096] cnt [64]; qkv ws [4,16,2304] cnt [36]; unpooler ws [8,16,1024] cnt [16]), allocated in prepare(); prepare() warms every routed specialization on synthetic bf16 inputs (T 1..16 for qkv/o/pooler, 1..8 for the unpooler slice), synchronises and zero_()s every cnt, as the combo's prepare() does. The router (side stream, hi/lo torch.mm) and everything else are untouched.

## Expected

- `*load=decode_1 *` faster: T=1, the combo measured per GEMM: pooler 21.2->19.2, qkv 14.4->13.2, unpooler slice 7.1->5.8, o 6.7->6.6 us: ~4.5 us off 173/182 us, +2-3%
- `*load=verify_1 *` faster: T=8: pooler 21.3->19.6, unpooler 7.0->6.5, qkv 14.0->13.5, o flat: ~2.5-3 us off 238/248 us, ~+1%
- `*load=decode_16 *` faster: T=16: pooler 21.3->19.5, o 6.8->6.4, qkv flat: ~2-2.5 us off 260/407 us, ~+0.5-1%, near drift
- `*load=decode_32 *` flat: control: cuBLAS kept
- `*load=verify_16 *` flat: control
- `*load=verify_32 *` flat: control

## Risk

Expected ~+1% geomean: modest but the four kernels are measured winners at these exact shapes, and each replaces an nvjet+splitKreduce pair (two graph nodes) with one launch. Risks: (1) in this layer the pooler overlaps the side-stream router; a 128-CTA Triton pooler may contend with the router's GEMMs differently than nvjet does; read the pooler's own top_kernels time against the bf16 run. (2) The judge uses a cold L2 before each replay here too, as in bf16_proj, so per-kernel times should carry over. (3) The ticket-semaphore reduce is shared-pattern code that passed in bf16_proj; counters are per path, never shared. If the headline is flat, read the per-GEMM rows (nvjet 128x8/64x8 splitK and splitKreduce rows should disappear at T<=16).
