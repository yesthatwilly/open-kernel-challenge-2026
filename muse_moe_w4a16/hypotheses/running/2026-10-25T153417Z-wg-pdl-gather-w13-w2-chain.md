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
review: proceed
status: running
review_at: 2026-10-25T162011Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T162011Z
base_sha: 04b5d4ce29ff4cb29cbed7b24b63a5492c811e97
base_blob: 593b238c16387c5ac3774424ee9557882b9fc46a
branch: hyp/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain
worktree: .worktrees/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T162011Z by an adversarial subagent (model opus).

**Verdict: proceed.** Safe and in-rules if the gather either gets its own griddepcontrol.wait or is launched without the PDL attribute. The idea holds but its size is overstated: PDL can only hide w2's launch gap, prologue and first weight stage, not w13's last-wave tail work. Expect about 0.5-1.5% on bs=256 (3.4 us err on a 404.9 us row) and roughly +0.1-0.2% on the headline, so the scored number cannot confirm it. The readout is the prepare-time per-kernel event times with and without PDL, so no companion is needed.

### Justification
A. Plausible on H200: wg_kernel is persistent, runs one CTA per SM with a static item stride, and runs back to back in a graph, so PDL can overlap w2's launch latency, barrier init, chunk-table prologue, setmaxnreg and first weight bulk_g2s with w13's tail. The tail work itself cannot be overlapped. w2 cannot consume anything until griddepcontrol.wait returns, and that only happens once every w13 CTA has completed and flushed. Early w2 CTAs therefore sit on freed SMs and wait; they do not do useful GEMM work in the tail. The real saving is the inter-kernel gap plus w2's prologue and ring-fill latency, about 2-6 us at bs=256, with smaller savings at gather->w13. Weakest link: the hypothesis's numbers. It quotes w13 303 us + w2 148 us = 451 us, which is more than the 404.9 us shape total. The scored profiles of master's lineage show wg_kernel<true> 237.6-239.2 us and wg_kernel<false> 111.8-119.0 us (s5-bucket-gate job 4e4a0e26/7be125e9, align2 job c05a8f73). The '~19.5 us items, 10-20 us of idle SMs' figure then becomes a reason for a 2-4% target that PDL cannot reach. bs=256 err_ns is 3.3-3.4 us (about 0.8%), and the shape carries 1/6 of the score, so even a real 1% shape gain is about +0.17% on the headline against a CI of about +/-1.5%. The headline and the row can at most fail to refute it. The conjecture can only be confirmed or refuted by the prepare log ('PDL on' plus per-kernel event times with and without PDL on the same graph). The expected directions are consistent: bs=1/16/32/8 run Marlin, and bs=128 runs Marlin in both master profiles (B128 flag off), so they are flat controls. Separate correctness gap: item (1) PDL-launches wg_gather_kernel, but (3) gives the gather no griddepcontrol.wait. Its predecessor is the SGLang expert_map gather / moe_align_block_size, which is not ours and never triggers, so the trigger is implicit at its exit. PTX still only guarantees visibility of the primary's writes to a dependent after griddepcontrol.wait. Also, w13's pre-wait prologue reads block_experts/n_padded, and the claim that 'align completed before the w13 grid started' holds only if the gather waited. Require one of two things: give wg_gather_kernel a griddepcontrol.wait as its first statement, or (simpler, and nearly costless because align never triggers early) launch the gather with a plain <<<>>> launch and put PDL only on the w13 and w2 launches.

### Confounders on the current kernel
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (current BEST, the base of the wgmma path): bs=256 runs wgmma only when the prepare-time _wg_select timing selector sets wg_on_256, and the hypothesis re-runs that selector with PDL. If the flag flips in this run (off, or B128 on), the bs=256/bs=128 rows measure a path switch, not PDL. The log must print the final wg_on_128/wg_on_256 and the path each bucket took; a row move with a flipped flag is not attributable to PDL.
- align-two-phase-port-from17 (on master, Marlin path only): no interaction while bs=256 is on wgmma, because _moe_wgmma still uses SGLang's moe_align_block_size at block 16. If the selector turns B256 off, bs=256 falls to Marlin with align2, which is a different and confounded comparison. Same remedy: log the chosen path.
- align-parallel-per-expert-verify / triton-map-align-on-block-m-restack (align for T>32 on the Marlin path): these do not touch _moe_wgmma's align. They matter only through the same selector-flip route.
- w2-marlin-atomic-add-small-m, post-expert-norm-triton-small-m, marlin-block-m-two-smaller-on-postnorm, sglang-marlin-path-parity: their shapes or paths do not overlap this change (num_tokens <= 32, or Marlin at bs=128); no confounding beyond run-to-run noise on the control rows.

### Interference and A/B needs
No companion. The only area this change touches is three launches on the wgmma path at 128<=T<=512. The landed hypotheses that could interact do so only by flipping the bucket selector, and the run's log resolves that, not a revert run. The hypothesis already builds its own A/B into prepare: the same captured graph is timed with and without PDL, outputs are compared bit for bit, and per-kernel event times are printed. That in-process A/B has far less noise than a second scored run, which would cost a submit and still sit inside the 3.4 us err. A gain or a loss on the bs=256 row can be read through the prepare event times, so the result is interpretable without a companion. Ordering note from the hypothesis's own Risk section: wg-stream-k-tail-split targets the tail work PDL cannot reach, so the two are complementary, not duplicates.

### Open questions
- Where do the 303/148 us per-kernel figures come from? The scored profiles show about 238/112-119 us. Re-derive the target from the profile; a realistic bs=256 expectation is 2-6 us (0.5-1.5%), not 10-20 us.
- Is griddepcontrol.launch_dependents issued by one thread (the producer after its last stage) enough for the CTA on this toolchain? PTX treats the instruction per CTA, but confirm the consumer-side issue is either redundant or harmless.
- With the 3.4 us err and a 1/6 shape weight, is a scored submit the right vehicle for a change whose expected headline effect is about +0.1-0.2%? It is safe and interpretable through the prepare log, but it will not register as a 'gain' under the CI rules even if it works.

