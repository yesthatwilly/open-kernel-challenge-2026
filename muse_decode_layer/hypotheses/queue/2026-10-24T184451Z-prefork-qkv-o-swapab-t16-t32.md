---
problem: muse_decode_layer
slug: prefork-qkv-o-swapab-t16-t32
created: 2026-10-24T184451Z
base: master
area: qkv GEMM + o GEMM (before the router fork)
change: >
  At tokens == 16 and tokens == 32 only (decode_16 and decode_32, both layer types; config gate),
  route the two dense GEMMs that run BEFORE the router side-stream fork through the bf16 problem
  swap-AB Triton kernel: qkv = _gemm(data["attn_input"], dense["qkv_gate"]) (n 2304, k 8192) and
  attn_output = _gemm(o, dense["o"]) (n 8192, k 1024). The pooler, the router, the unpooler and
  every other T keep master exactly. Copy _bf16_swapab_kernel, SWAPAB_BLOCK_W, SWAPAB_NUM_WARPS,
  SWAPAB_FALLBACK_STAGES and the OutOfResources import guard from muse_bf16_proj/submission.py on
  master, with the qkv_gate and o rows of SWAPAB_CFG ((2304, 8192, 4, 128, 4) and (8192, 1024, 1,
  256, 4)). T=32 uses master BLOCK_T = WS_M = 32; T=16 uses BLOCK_T = WS_M = 16 with the qkv_gate
  and o rows of SWAPAB16_CFG from branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend (same
  kernel, smaller token tile, wgmma m64n16). For attribution give each layer its own kernel name:
  two thin @triton.jit wrappers _swapab_qkv_kernel and _swapab_o_kernel that call the shared body
  (or two copies of the kernel), so the profile shows qkv and o rows separately. prepare():
  allocate per-layer fp32 split-K workspaces and zeroed int32 ticket counters (as bf16
  _swapab_prepare does, qkv and o only, sized for WS_M 32), warm each (layer, T) once, and
  self-check each against torch.mm on random bf16 inputs (rel L2 <= 1e-2); any failure or
  OutOfResources at both stage counts sets that layer flag off and keeps _gemm. Print
  "[prefork-swapab] layer=.. T=.. stages=.. ON|OFF".
expected:
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "bf16 m16 swap-AB measured qkv 18% and o 10-19% faster; qkv (nvjet 64x16 4x1 splitK 12.2 us + splitKreduce ~2 us) -> ~10 us, o ~6 -> ~5 us: -3..-4 us of 230 / 385 us, +1-1.5%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "m32 swap-AB measured qkv +3.6%, o +4.1%: -1 us, flat to +0.5%"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate (restack covers it)
  - shapes: "*load=verify_*"
    direction: flat
    why: T=8/128/256 outside the gate
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# prefork-qkv-o-swapab-t16-t32

## Change

At tokens == 16 and tokens == 32 only (decode_16 and decode_32, both layer types; config gate), route the two dense GEMMs that run BEFORE the router side-stream fork through the bf16 problem swap-AB Triton kernel: qkv = _gemm(data["attn_input"], dense["qkv_gate"]) (n 2304, k 8192) and attn_output = _gemm(o, dense["o"]) (n 8192, k 1024). The pooler, the router, the unpooler and every other T keep master exactly. Copy _bf16_swapab_kernel, SWAPAB_BLOCK_W, SWAPAB_NUM_WARPS, SWAPAB_FALLBACK_STAGES and the OutOfResources import guard from muse_bf16_proj/submission.py on master, with the qkv_gate and o rows of SWAPAB_CFG ((2304, 8192, 4, 128, 4) and (8192, 1024, 1, 256, 4)). T=32 uses master BLOCK_T = WS_M = 32; T=16 uses BLOCK_T = WS_M = 16 with the qkv_gate and o rows of SWAPAB16_CFG from branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend (same kernel, smaller token tile, wgmma m64n16). For attribution give each layer its own kernel name: two thin @triton.jit wrappers _swapab_qkv_kernel and _swapab_o_kernel that call the shared body (or two copies of the kernel), so the profile shows qkv and o rows separately. prepare(): allocate per-layer fp32 split-K workspaces and zeroed int32 ticket counters (as bf16 _swapab_prepare does, qkv and o only, sized for WS_M 32), warm each (layer, T) once, and self-check each against torch.mm on random bf16 inputs (rel L2 <= 1e-2); any failure or OutOfResources at both stage counts sets that layer flag off and keeps _gemm. Print "[prefork-swapab] layer=.. T=.. stages=.. ON|OFF".

## Expected

- `*load=decode_16 *` faster: bf16 m16 swap-AB measured qkv 18% and o 10-19% faster; qkv (nvjet 64x16 4x1 splitK 12.2 us + splitKreduce ~2 us) -> ~10 us, o ~6 -> ~5 us: -3..-4 us of 230 / 385 us, +1-1.5%
- `*load=decode_32 *` faster: m32 swap-AB measured qkv +3.6%, o +4.1%: -1 us, flat to +0.5%
- `*load=decode_1 *` flat: T=1 outside the gate (restack covers it)
- `*load=verify_*` flat: T=8/128/256 outside the gate

## Risk

Mostly an attribution probe for the open LESSONS line: bf16-shape-gated-triton-gemms-small-t lost 2-4.5% at T=8/16 despite faster kernel sums, "possibly interference with the side-stream router overlap". qkv and o run before the fork, so if their own rows get faster and the shape gets faster, the loss came from the pooler/unpooler side; if the rows get faster and the shape does not, the graph-node or ticket overhead is to blame. Overlaps the held bf16-m32-swapab-wgmma-port (qkv/o/pooler at T=32); this is narrower (pre-fork layers only) and adds T=16. The in-judge restack gates its Triton GEMMs to T=1, so no shape overlap.
