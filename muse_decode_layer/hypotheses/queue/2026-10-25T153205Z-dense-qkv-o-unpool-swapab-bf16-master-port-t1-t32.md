---
problem: muse_decode_layer
slug: dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32
created: 2026-10-25T153205Z
base: master
area: qkv GEMM + o GEMM + unpooler GEMM (T <= 32)
change: >
  Port muse_bf16_proj master's swap-AB wgmma GEMMs (commit 040fc7e18f, best 1.2491x submit
  ee4f89f2f62c42d3) to decode's qkv_gate and o GEMMs at every T <= 32, and to the full-width
  unpooler at 9 <= T <= 32. bf16_proj measured these exact weight shapes on the same judge with
  one launch each (no splitKreduce node): qkv_gate 9.7 us at m=1/8, 10.2 at m=16, 11.5 at m=32
  (decode master: nvjet 128x8/64x16/64x32 splitK 12.2-13.1 us PLUS a splitKreduce 1.8 us node); o
  3.9 / 4.3 / 4.4 / 4.7 us at m=1/8/16/32 (decode master: nvjet 64x8 6.8-7.1 us at T=1/8; at
  T=16/32 the 2-call nvjet row 26.4-27.4 us is o + unpooler). Copy verbatim from
  muse_bf16_proj/submission.py on master: SWAPAB_* and SWAPAB16_* constants and configs (the m <=
  16 BLOCK_T=16 path and the 17 <= m <= 32 BLOCK_T=32 path), _bf16_swapab_kernel and its m<=16
  sibling, _swapab_route_ok, _swapab_forward, _swapab_buffers, _swapab_prepare (per-state
  workspaces and self-resetting tickets; warm every m in 1..32 per layer on synthetic bf16 inputs,
  synchronize, zero the tickets), and the OutOfResources stage fallback. Only the layers qkv_gate
  (2304 x 8192), o (8192 x 1024) and unpooler (8192 x 4096, full width at T > 8) are routed; the
  pooler and router stay inside dense-chain kernel P at T <= 32 (so the held
  bf16-m32-swapab-wgmma-port's concern, the swap-AB pooler competing with the side-stream router
  for SMs, no longer exists: there is no side stream at T <= 32 on master). Call sites in
  _parent_attention: qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o, dense['o']) ->
  _swapab_forward when tokens <= 32 and this state's per-(layer, T bucket) flag is set; in
  kernel_fn the T > 8 unpooler _gemm likewise at 9 <= T <= 32. Full decode_1 with dl3 piece A on
  keeps piece A (its own Q/O kernels); T <= 8 unpooler keeps dl3 piece C / the slice GEMM.
  SELECTOR per (layer, T in {1, 8, 16, 32}): the bf16_proj model selector (graph replay, 256 MB
  flush, 41 interleaved reps, empty-graph baseline subtracted, >= 1.02x win, assert enabled ==
  printed winner), timing the GEMM INCLUDING master's splitKreduce node; print '[swapab] PATH
  CHOSEN <layer> T=<t>: swapab X us vs cuBLAS Y us'. Self-check rel-L2 vs an fp32 torch.mm
  reference at every m in 1..32, bound 1e-2, on two alternating states. This fixes the hold
  reasons of bf16-m32-swapab-wgmma-port (now one combined restack of the decode GEMMs with
  per-cell attribution, pooler excluded) and ports the newer bf16 master kernel rather than
  re-running qkv-o-swapab16-t2-t16-no-fork's code (1.3056x, unproven-vs-best; it also covered
  T=2..16 only). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no
  timing, no prints, no try/except, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow and no per-call mutable Python state; gates read config/shape
  and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4
  stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key
  attended; no child processes. The judge prepares TWO state copies and rotates them: keep every
  device pointer, TMA map, ticket array and workspace per state, never in a module global. Every
  new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode
  arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an
  in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any
  miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check
  by capturing once and replaying with refilled inputs while alternating two states. Selectors
  time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps,
  empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS
  by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per
  module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First
  measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: qkv nvjet 12.2 + splitKreduce 1.8 -> ~9.7 us; o 7.1 -> ~3.9 us; ~-7 us on 117.8 us (weight 46)
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: qkv 12.6 + reduce ~1.7 -> 9.7; o 6.9-7.1 -> 4.3; ~-6 us on 190/199 us"
  - shapes: "*load=decode_16*"
    direction: faster
    why: qkv 12.8 + reduce -> 10.2; o + unpooler 2-call row 26.4 -> ~4.4 + unpooler swapab; ~-4..-7 us
  - shapes: "*load=decode_32*"
    direction: faster
    why: "qkv 13.1 + reduce -> 11.5; o 4.7; unpooler bf16 m32 18.5 vs nvjet share of the 26.9 us row; ~-3..-6 us on 236/442 us (sliding decode_32 is master's worst ratio, 1.075x)"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: piece A owns qkv/o there when on; otherwise the same -7 us as sliding
  - shapes: "*load=verify_16*"
    direction: flat
    why: "T=128: untouched"
  - shapes: "*load=verify_32*"
    direction: flat
    why: "T=256: untouched"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32

## Change

Port muse_bf16_proj master's swap-AB wgmma GEMMs (commit 040fc7e18f, best 1.2491x submit ee4f89f2f62c42d3) to decode's qkv_gate and o GEMMs at every T <= 32, and to the full-width unpooler at 9 <= T <= 32. bf16_proj measured these exact weight shapes on the same judge with one launch each (no splitKreduce node): qkv_gate 9.7 us at m=1/8, 10.2 at m=16, 11.5 at m=32 (decode master: nvjet 128x8/64x16/64x32 splitK 12.2-13.1 us PLUS a splitKreduce 1.8 us node); o 3.9 / 4.3 / 4.4 / 4.7 us at m=1/8/16/32 (decode master: nvjet 64x8 6.8-7.1 us at T=1/8; at T=16/32 the 2-call nvjet row 26.4-27.4 us is o + unpooler). Copy verbatim from muse_bf16_proj/submission.py on master: SWAPAB_* and SWAPAB16_* constants and configs (the m <= 16 BLOCK_T=16 path and the 17 <= m <= 32 BLOCK_T=32 path), _bf16_swapab_kernel and its m<=16 sibling, _swapab_route_ok, _swapab_forward, _swapab_buffers, _swapab_prepare (per-state workspaces and self-resetting tickets; warm every m in 1..32 per layer on synthetic bf16 inputs, synchronize, zero the tickets), and the OutOfResources stage fallback. Only the layers qkv_gate (2304 x 8192), o (8192 x 1024) and unpooler (8192 x 4096, full width at T > 8) are routed; the pooler and router stay inside dense-chain kernel P at T <= 32 (so the held bf16-m32-swapab-wgmma-port's concern, the swap-AB pooler competing with the side-stream router for SMs, no longer exists: there is no side stream at T <= 32 on master). Call sites in _parent_attention: qkv = _gemm(attn_input, qkv_gate) and attn_output = _gemm(o, dense['o']) -> _swapab_forward when tokens <= 32 and this state's per-(layer, T bucket) flag is set; in kernel_fn the T > 8 unpooler _gemm likewise at 9 <= T <= 32. Full decode_1 with dl3 piece A on keeps piece A (its own Q/O kernels); T <= 8 unpooler keeps dl3 piece C / the slice GEMM. SELECTOR per (layer, T in {1, 8, 16, 32}): the bf16_proj model selector (graph replay, 256 MB flush, 41 interleaved reps, empty-graph baseline subtracted, >= 1.02x win, assert enabled == printed winner), timing the GEMM INCLUDING master's splitKreduce node; print '[swapab] PATH CHOSEN <layer> T=<t>: swapab X us vs cuBLAS Y us'. Self-check rel-L2 vs an fp32 torch.mm reference at every m in 1..32, bound 1e-2, on two alternating states. This fixes the hold reasons of bf16-m32-swapab-wgmma-port (now one combined restack of the decode GEMMs with per-cell attribution, pooler excluded) and ports the newer bf16 master kernel rather than re-running qkv-o-swapab16-t2-t16-no-fork's code (1.3056x, unproven-vs-best; it also covered T=2..16 only). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: keep every device pointer, TMA map, ticket array and workspace per state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), with -std=c++20 -O3 -lineinfo, a renamed load_inline module, an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x, and assert the enabled choice equals the printed winner. Print compile seconds per module and the cumulative prepare() wall time; the whole prepare must stay under 270 s. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_1 *` faster: qkv nvjet 12.2 + splitKreduce 1.8 -> ~9.7 us; o 7.1 -> ~3.9 us; ~-7 us on 117.8 us (weight 46)
- `*load=verify_1 *` faster: T=8: qkv 12.6 + reduce ~1.7 -> 9.7; o 6.9-7.1 -> 4.3; ~-6 us on 190/199 us
- `*load=decode_16*` faster: qkv 12.8 + reduce -> 10.2; o + unpooler 2-call row 26.4 -> ~4.4 + unpooler swapab; ~-4..-7 us
- `*load=decode_32*` faster: qkv 13.1 + reduce -> 11.5; o 4.7; unpooler bf16 m32 18.5 vs nvjet share of the 26.9 us row; ~-3..-6 us on 236/442 us (sliding decode_32 is master's worst ratio, 1.075x)
- `*layer=full load=decode_1 *` flat: piece A owns qkv/o there when on; otherwise the same -7 us as sliding
- `*load=verify_16*` flat: T=128: untouched
- `*load=verify_32*` flat: T=256: untouched

## Risk

In-layer interference: bf16_proj timed each GEMM alone; in decode the qkv GEMM follows the ffn boundary and the o GEMM follows attention, both on one stream, so standalone gains should carry, but LESSONS records a T=8/16 loss for Triton split-K GEMMs in decode (bf16-shape-gated-triton-gemms-small-t, a different kernel family); the per-cell in-prepare selector is the guard. Read the swapab rows and the missing splitKreduce rows in the profile per shape, not the headline. Overlaps dl3-unpool-nf1-t2-t8 (queued) only in that both touch the unpooler; this one routes the unpooler at T >= 9 only, so they are disjoint.
