---
problem: muse_moe_w4a16
slug: wg-pdl-gather-w13-w2-chain
created: 2026-10-25T153417Z
base: master
area: launch overhead
change: >
  Programmatic dependent launch across master's wgmma chain at 128 <= T <= 512 (verify_32 bs=256
  runs it; verify_16 only if its bucket flag is on). Kernels, tiling, numerics, align and
  post-norm are unchanged. MECHANISM: wg_kernel<w13+silu> runs ~15.5 waves of ~19.5 us items at
  bs=256 (303 us / 2048 items / 132 CTAs) and wg_kernel<w2> ~7.8 waves; the last wave leaves part
  of the SMs idle until the slowest CTA ends, then w2's grid launches, builds its chunk tables,
  inits barriers, setmaxnreg and only then starts filling its weight ring. The w2 WEIGHTS do not
  depend on h, only its B slab (wg_hs) does. CHANGE: (1) launch wg_gather_kernel, wg_kernel<w13>
  and wg_kernel<w2> with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization
  = 1 (capturable: it becomes a programmatic edge in the graph; moe's
  cuda-dispatch-combine-pdl-verify already ran graph+PDL on the judge). (2) In wg_kernel<w13> and
  wg_kernel<w2>: each CTA executes griddepcontrol.launch_dependents when its persistent item loop
  has no further item (producer thread after issuing its last stage; consumers after their last
  epilogue store) so the next grid's CTAs can occupy SMs freed in the tail. (3) In the dependent
  kernels, everything that does not read the predecessor's output runs before griddepcontrol.wait:
  barrier init, the chunk-table prologue (it reads only sorted_ids / block_experts / n_padded,
  produced by the align before the gather, which completed before the w13 grid started),
  setmaxnreg, and the producer's first STG weight bulk copies (bulk_g2s of the first item's weight
  records: weights are constant). The producer then executes griddepcontrol.wait before its first
  B-slab TMA (wg_xs for w13, wg_hs for w2); consumers execute griddepcontrol.wait before their
  first epilogue global store (the w13 kernel's hs stores must not race the previous call's w2
  reads: the wait guarantees the predecessor grid completed and flushed). The post-norm (SGLang
  fused_post_expert_norm_weighted_sum) launches normally after w2 (no PDL; it is not ours). (4)
  Prepare: the s5 self-check and three-state copy check on the PDL launches, refilled graph
  replays compared to the non-PDL path bit for bit (same math, same order: outputs must be
  identical), and the _wg_select bucket selector re-run with PDL so the B128/B256 flags reflect
  it; print 'PDL on' and per-kernel event times with and without PDL. A launch error turns PDL off
  (state['wg_pdl'] = False) and keeps master's launches. BUILD (as master's wg module): own
  load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode
  arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under
  __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy /
  L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes /
  smem (a spill line is a bug: fix before spending the check). No module-global device pointer,
  tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded
  per call from the tensor passed in that call (master's s5 rule; the judge prepares two states
  and copies them by cloning tensors). First launch in-process after a small synthetic canary,
  then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first
  measurement is an okc check, never a scored submit.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "verify_32: master 404.9-407.1 us (w13 303 + w2 148 + small kernels in the profile); the w13 tail plus w2's prologue and ring fill are ~10-20 us of partly idle SMs. Target -2 to -4% on the shape (<= 395 us); read the per-kernel event times with and without PDL in the log"
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin unless the selector flips (PDL alone is not expected to flip it)
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=32 *
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

# wg-pdl-gather-w13-w2-chain

## Change

Programmatic dependent launch across master's wgmma chain at 128 <= T <= 512 (verify_32 bs=256 runs it; verify_16 only if its bucket flag is on). Kernels, tiling, numerics, align and post-norm are unchanged. MECHANISM: wg_kernel<w13+silu> runs ~15.5 waves of ~19.5 us items at bs=256 (303 us / 2048 items / 132 CTAs) and wg_kernel<w2> ~7.8 waves; the last wave leaves part of the SMs idle until the slowest CTA ends, then w2's grid launches, builds its chunk tables, inits barriers, setmaxnreg and only then starts filling its weight ring. The w2 WEIGHTS do not depend on h, only its B slab (wg_hs) does. CHANGE: (1) launch wg_gather_kernel, wg_kernel<w13> and wg_kernel<w2> with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 (capturable: it becomes a programmatic edge in the graph; moe's cuda-dispatch-combine-pdl-verify already ran graph+PDL on the judge). (2) In wg_kernel<w13> and wg_kernel<w2>: each CTA executes griddepcontrol.launch_dependents when its persistent item loop has no further item (producer thread after issuing its last stage; consumers after their last epilogue store) so the next grid's CTAs can occupy SMs freed in the tail. (3) In the dependent kernels, everything that does not read the predecessor's output runs before griddepcontrol.wait: barrier init, the chunk-table prologue (it reads only sorted_ids / block_experts / n_padded, produced by the align before the gather, which completed before the w13 grid started), setmaxnreg, and the producer's first STG weight bulk copies (bulk_g2s of the first item's weight records: weights are constant). The producer then executes griddepcontrol.wait before its first B-slab TMA (wg_xs for w13, wg_hs for w2); consumers execute griddepcontrol.wait before their first epilogue global store (the w13 kernel's hs stores must not race the previous call's w2 reads: the wait guarantees the predecessor grid completed and flushed). The post-norm (SGLang fused_post_expert_norm_weighted_sum) launches normally after w2 (no PDL; it is not ours). (4) Prepare: the s5 self-check and three-state copy check on the PDL launches, refilled graph replays compared to the non-PDL path bit for bit (same math, same order: outputs must be identical), and the _wg_select bucket selector re-run with PDL so the B128/B256 flags reflect it; print 'PDL on' and per-kernel event times with and without PDL. A launch error turns PDL off (state['wg_pdl'] = False) and keeps master's launches. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit.

## Expected

- `bs=256 *` faster: verify_32: master 404.9-407.1 us (w13 303 + w2 148 + small kernels in the profile); the w13 tail plus w2's prologue and ring fill are ~10-20 us of partly idle SMs. Target -2 to -4% on the shape (<= 395 us); read the per-kernel event times with and without PDL in the log
- `bs=128 *` flat: B128 stays Marlin unless the selector flips (PDL alone is not expected to flip it)
- `bs=1 *` flat: control
- `bs=32 *` flat: control

## Risk

Attacks the same tail as wg-stream-k-tail-split (that one splits the last wave's items inside each kernel; this one lets the next kernel fill the idle SMs): run them standalone on master and compare kernel rows, then stack. No overlap with the queued consumer-side lines. Correctness hazard: an early launch_dependents with a missing griddepcontrol.wait before a dependent global read would read stale wg_xs/wg_hs; the bit-identity check against non-PDL replays with refilled inputs is the guard.
