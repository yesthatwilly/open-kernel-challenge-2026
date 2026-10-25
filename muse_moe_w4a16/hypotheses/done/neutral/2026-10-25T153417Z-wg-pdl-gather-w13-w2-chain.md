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
status: done
review_at: 2026-10-25T162011Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T162011Z
base_sha: 04b5d4ce29ff4cb29cbed7b24b63a5492c811e97
base_blob: 593b238c16387c5ac3774424ee9557882b9fc46a
branch: hyp/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain
worktree: .worktrees/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T163022Z
impl_commit: 61ce566d5de32460f7987c73044a9d5aa8c9c4ba
submission_sha256: 17fdbbfb215a068e7ce0327d5db9e64496aecc8fa3c04031b7f669e7c81f5b10
norm_sha256: 378b54dd908fe4c7ae5e77214d510feea756a9a1a6f084188d4cd1ca528c42c3
submission_blob: 72de8f1edeaa3215bb806cdc5ae13791f54cebd2
lint_warnings: 77
attempt_file: 2026-10-25T163022Z-submit-wg-pdl-gather-w13-w2-chain.md
prepared_at: 2026-10-25T163022Z
job: 667b64e3fb1343ec
dispatched_at: 2026-10-25T163042Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3752355517671193
speedup_ci: [1.3546070184906125, 1.395864085043626]
harvested_at: 2026-10-25T163338Z
best_at_harvest: 1.3726662821010693
promotion: done
promotion_job: 667b64e3fb1343ec
hand_landed: true
promotion_commit: f7eac2c3ca666ac6dcd24425163ae410a5ca60f5
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


## Implementation notes

Built 2026-10-25 on base 04b5d4ce (master's muse_moe_w4a16/submission.py is unchanged since; not stale). Diff:  1 file changed, 207 insertions(+), 14 deletions(-). Only submission.py of the worktree was edited.

What changed
- CUDA (own module name muse_moe_w4a16_wgmma_rega_v6_pdl; same sm_90a-only gencode, -O3 -lineinfo, 9.0a arch list): griddep_wait / griddep_launch helpers; wg_kernel producer thread issues the first stage's weight bulk_g2s, then griddepcontrol.wait (once, n == 0), then the B-slab TMAs; consumers execute griddepcontrol.wait once before their item loop (before any epilogue store); every thread executes griddepcontrol.launch_dependents after its role's item loop (the producer warpgroup's other threads exit early, which counts as having triggered). wg_gemm takes an int64 pdl argument: pdl 0 is the old <<<>>> launch, pdl 1 is cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization = 1. No new global or static state (the audit passes the same findings as before).
- Python: _wg_gemm passes pdl = 1 only for which == 1 (w2) and only when state.get('wg_pdl') is true. state['wg_pdl'] is assigned in _wg_setup (prepare) and never in kernel_fn; kernel_fn and _moe_wgmma are otherwise unchanged (no print, no try/except, no mutable state added).
- New prepare-only _wg_pdl_stage (runs after the existing checks pass, before the bucket selector, so the selector times the shipped launches): bit-identity of _moe_wgmma graphs PDL off vs on x2 at T=128 and T=256 over four refilled routings; prepare-time event timings (w13 and w2 alone, gather+w13+w2 chain graph, full wgmma path graph, PDL off vs on, cold L2, 41 interleaved reps); then the existing three-state copy check re-run with PDL on (_wg_copy_check gained a pdl flag). Any mismatch, error word, missing launch marker, failed copy check, exception, spent budget (WG_PDL_DEADLINE_S = 225 s), or T=256 full-path PDL median slower than off by more than 0.3 percent leaves state['wg_pdl'] = False, i.e. master's launches. If no bucket ends on wgmma, wg_pdl is forced False. The stage never raises.
- Final log lines: "PDL ..." lines (bit-identity, event timings, copy check, on/off verdict) and one "FINAL FLAGS wg_on_128=... wg_on_256=... wg_pdl=...; B128 took ..., B256 took ..." line in addition to the existing PATH CHOSEN lines.

Review fixes applied (the review wins over the spec where they differ)
1. Gather without PDL: wg_gather_kernel stays a plain <<<>>> launch. Also deviating from spec point (1): wg_kernel<w13> is launched plainly too (the review says PDL only on the w13 -> w2 boundary), so only the w2 launch carries the attribute. w13 still executes launch_dependents so w2 can start early; its own griddepcontrol.wait is a no-op under a plain launch. The align and gather output are therefore complete (stream order) before w13's pre-wait prologue reads them.
2. state['wg_pdl'] is set once in prepare; kernel_fn only reads it (through _wg_gemm); the per-launch attribute is fine.
3. Printed: wg_on_128, wg_on_256, wg_pdl and the path each bucket took (FINAL FLAGS plus the existing PATH CHOSEN lines), and prepare-time event timings with and without PDL (per kernel alone, chain graph, full path). Caveat: the selector runs once, with the PDL decision applied; a flag flip caused by PDL is not isolated by a second selector run (budget), but the full-path PDL off/on graph timings at T=128 and T=256 are printed beside it.
4. Bit-identity check and three-state copy check fall back to master's launches on any mismatch and never raise (try/except in prepare code only).
5. Per-kernel times corrected: the spec's w13 303 us / w2 148 us are wrong; the scored profile shows wg_kernel<true> (w13+silu) about 238 us (237.6-239.2) and wg_kernel<false> (w2) about 112-119 us. Realistic expectation per the review: PDL hides w2's launch gap, prologue and first weight stage (about 2-6 us at bs=256, 0.5-1.5 percent of the 404.9 us row, roughly +0.1-0.2 percent on the headline); it cannot hide the w13 tail work. The scored number cannot confirm it; the readout is the prepare-time PDL off/on event timings.

Review open questions answered
- launch_dependents: PTX/CUDA treat it per CTA (the dependent launches when every thread of the CTA has executed it or exited), so it is issued by all threads after their role's loop (one executes the loop, the rest exit); not producer-only.
- Consumer-side griddepcontrol.wait: consumers wait once before the loop rather than inside the first epilogue; equivalent in time because the full barriers cannot complete before the producer's wait returns, and it guarantees no consumer store precedes predecessor completion.
- Scored submit as the vehicle: per the caller, this runs as a SCORED submit; the prepare log is the interpretable readout.

Biggest risks: cudaLaunchKernelEx template with a __grid_constant__ struct parameter failing to compile (then the whole wgmma build fails and master's Marlin path ships); PDL edge capture behavior in graphs (guarded by the bit-identity and copy checks).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds no sync, item(), print or input-dependent control flow; the only new work in kernel_fn's path is an int flag read from state and a launch attribute (programmatic edge in the graph); verified in git diff.
- [x] Nothing derived from data is cached: wg_pdl is a prepare-time bool; no input-derived state added.
- [x] Weights: n/a: untouched (no repack change).
- [x] Activations BF16 / KV FP8 / queries: n/a: no numerics changed; outputs are required bit-identical to the non-PDL launches.
- [x] Attention: n/a: MoE problem.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: load_inline only, no process spawn, nothing added to the harness.
- [x] All work inside the timed call; no output caching or replay.
- [x] State safety (judge copies two states): no new module/static/global device state; wg_pdl is a copied bool; the copy check is re-run with PDL on.

okc_lint:
OK with warnings: .worktrees/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain/muse_moe_w4a16/submission.py
py_compile: ok; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent (read-only emulation; no GPU, nothing compiled by nvcc). **Verdict: dispatch-with-fixes.** Rules OK.

Fixes (worktree submission.py only, not committed):
1. **Build-breaking dangling else (category 1/2).** The new `WG_LAUNCH(G_)` macro expanded to a bare `if (pdl == 0) {...} else {...}`, and the call site is `if (which == 0) WG_LAUNCH(true); else WG_LAUNCH(false);`. The `;` after the inner if/else leaves the outer `else` without an `if`. That is a hard compile error: the PDL module could never build, the whole wgmma path would have been lost, and verify_32 bs=256 would have fallen back to Marlin. g++ confirms the error on a reduced copy. Fix: wrap the macro body in `do { ... } while (0)`; the reduced copy then compiles. The `cudaLaunchKernelEx(&cfg, wg_kernel<G_>, mp, ...)` template deduction with the by-value Maps struct and 14 arguments was also checked against a mock of the runtime template, and it compiles.
2. **Master-module build fallback (category 2, requested).** If the PDL module (`muse_moe_w4a16_wgmma_rega_v6_pdl`) fails to build, prepare now builds MASTER's exact module: the same name `muse_moe_w4a16_wgmma_rega_v5`, the same flags and probe-then-plain sequence, and source text derived by reverting the PDL hunks (`_WG_MASTER_REVERT`, each hunk asserted unique). In that case `state['wg_ext_pdl'] = False`, the PDL stage is skipped, `wg_pdl` stays False, and `_wg_gemm` calls the 13-argument `wg_gemm`. A failed PDL probe with no ptxas line is treated as a compile error and goes straight to master's module, so no time is spent on a plain rebuild. The log prints `WG MODULE BUILT = <name> (PDL-capable | master module, no PDL)`. Verified offline via ast: the reverted `_WG_CPP`, `_WG_SRC_HEAD` and `_WG_SRC_BODY` are byte-identical to master's, and `_wg_wgmma_src` is unchanged. The PDL source differs from master in only two ways: in the kernel, the griddepcontrol helpers and their three call sites (the producer wait at n == 0 after the first weight bulk copy, the consumer wait before the item loop, and launch_dependents at the end); on the host, the pdl argument and the launch branch, where the pdl == 0 branch is master's `<<<>>>` launch plus `C10_CUDA_KERNEL_LAUNCH_CHECK`.

Findings (no change needed):
- griddepcontrol: w2 reads nothing from w13 before its wait. Before the wait the producer reads only the constant wbank, and the chunk prologue reads only sorted_ids / block_experts / n_padded. Those come from align, which finished before w13 started because w13 is a plain launch. No consumer global load or store happens before the wait, and the B-slab TMA (wg_hs) comes after it. The only pre-wait global write is the err[1] launch marker, which holds the same value w13 writes, so it is benign. Under graph capture the edge is w13 to w2 (programmatic), and align, gather and w13 stay full edges.
- The attribute is used correctly: `cudaLaunchAttributeProgrammaticStreamSerialization`, `val.programmaticStreamSerializationAllowed = 1`, and it is set on the w2 launch only. The gather and w13 are plain launches, as the review asked. The post-norm is a plain launch after w2.
- launch_dependents: the 127 idle producer-warpgroup threads execute it right after setmaxnreg. Per PTX the first execution in a CTA triggers that CTA, so every w13 CTA triggers almost at once, not "after its item loop" as the comments say. This is harmless: w13 is a persistent grid with one CTA per SM and already fully resident, and smem/registers forbid a w2 CTA from sharing an SM, so w2 CTAs only land on SMs that w13 has vacated. Correctness still rests on the wait, which needs full completion of w13.
- kernel_fn purity: wg_pdl / wg_ext_pdl are assigned only in _wg_setup, _wg_pdl_stage and _wg_copy_check (all prepare). Inside kernel_fn, _wg_gemm only reads prepare-time bools. Nothing depends on input values, and there is no sync or allocation beyond master's.
- The three-state copy check is re-run with pdl=True (A, deepcopy B, clone C, sequence A B C A B), and on failure PDL is turned off. The bit-identity gate cannot fail by construction: the wg kernels, align ordering effects and post-norm are deterministic per route, and remote rows of cache3 are skipped by slot_expert_ids.
- Budget: a failed PDL probe (~40 s) plus master's build (~40 s) still fits WG_SETUP_BUDGET_S = 200 (master's prepare-so-far is 37-49 s).

okc_lint: OK with warnings (prepare-only host syncs, same class as master). py_compile ok; __pycache__ removed.

Residual risk: the PDL module has never been through nvcc. If it still fails to build, master's module now ships, so the risk is a neutral run rather than a verify_32 regression. A PDL capture error on an old driver is caught in prepare and turns PDL off. A sticky device fault in the PDL stage would not be recoverable in-process (same class as master's canary).

## Result

- job `667b64e3fb1343ec` (submit), status done, passed True, tests 5/5
- speedup 1.3752x, 95% CI [1.3546, 1.3959], score 9779.052019648014 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3726662821010693, "ci": [1.3520762878695534, 1.3932562763325853], "job": "4e4a0e26c85944f2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-two-phase-port-from17.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/wg-pdl-gather-w13-w2-chain.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/wg-pdl-gather-w13-w2-chain.patch`; branch `hyp/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 46.77 | 45.09 | 1.0021 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 145.99 | 145.64 | 0.9986 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 203.75 | 198.1 | 0.9994 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 111.85 | 106.53 | 1.0006 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 319.08 | 315.4 | 1.0037 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 404.89 | 400.26 | 1.0186 | flat |

no shape moved outside noise. Expected hit: 3/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: wg-pdl-gather-w13-w2-chain`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

