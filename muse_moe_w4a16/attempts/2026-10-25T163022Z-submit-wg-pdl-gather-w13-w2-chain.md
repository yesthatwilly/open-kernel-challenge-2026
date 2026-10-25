# muse_moe_w4a16 · 2026-10-25T163022Z · submit · wg-pdl-gather-w13-w2-chain

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T153417Z-wg-pdl-gather-w13-w2-chain.md` (branch `hyp/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain`).

- **Job:** `667b64e3fb1343ec` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `61ce566d`

## Goal
Programmatic dependent launch across master's wgmma chain at 128 <= T <= 512 (verify_32 bs=256 runs it; verify_16 only if its bucket flag is on). Kernels, tiling, numerics, align and post-norm are unchanged. MECHANISM: wg_kernel<w13+silu> runs ~15.5 waves of ~19.5 us items at bs=256 (303 us / 2048 items / 132 CTAs) and wg_kernel<w2> ~7.8 waves; the last wave leaves part of the SMs idle until the slowest CTA ends, then w2's grid launches, builds its chunk tables, inits barriers, setmaxnreg and only then starts filling its weight ring. The w2 WEIGHTS do not depend on h, only its B slab (wg_hs) does. CHANGE: (1) launch wg_gather_kernel, wg_kernel<w13> and wg_kernel<w2> with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 (capturable: it becomes a programmatic edge in the graph; moe's cuda-dispatch-combine-pdl-verify already ran graph+PDL on the judge). (2) In wg_kernel<w13> and wg_kernel<w2>: each CTA executes griddepcontrol.launch_dependents when its persistent item loop has no further item (producer thread after issuing its last stage; consumers after their last epilogue store) so the next grid's CTAs can occupy SMs freed in the tail. (3) In the dependent kernels, everything that does not read the predecessor's output runs before griddepcontrol.wait: barrier init, the chunk-table prologue (it reads only sorted_ids / block_experts / n_padded, produced by the align before the gather, which completed before the w13 grid started), setmaxnreg, and the producer's first STG weight bulk copies (bulk_g2s of the first item's weight records: weights are constant). The producer then executes griddepcontrol.wait before its first B-slab TMA (wg_xs for w13, wg_hs for w2); consumers execute griddepcontrol.wait before their first epilogue global store (the w13 kernel's hs stores must not race the previous call's w2 reads: the wait guarantees the predecessor grid completed and flushed). The post-norm (SGLang fused_post_expert_norm_weighted_sum) launches normally after w2 (no PDL; it is not ours). (4) Prepare: the s5 self-check and three-state copy check on the PDL launches, refilled graph replays compared to the non-PDL path bit for bit (same math, same order: outputs must be identical), and the _wg_select bucket selector re-run with PDL so the B128/B256 flags reflect it; print 'PDL on' and per-kernel event times with and without PDL. A launch error turns PDL off (state['wg_pdl'] = False) and keeps master's launches. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Attacks the same tail as wg-stream-k-tail-split (that one splits the last wave's items inside each kernel; this one lets the next kernel fill the idle SMs): run them standalone on master and compare kernel rows, then stack. No overlap with the queued consumer-side lines. Correctness hazard: an early launch_dependents with a missing griddepcontrol.wait before a dependent global read would read stale wg_xs/wg_hs; the bit-identity check against non-PDL replays with refilled inputs is the guard.

## Result (after the run)
- job `667b64e3fb1343ec` (submit), status done, passed True, tests 5/5
- speedup 1.3752x, 95% CI [1.3546, 1.3959], score 9779.052019648014 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3726662821010693, "ci": [1.3520762878695534, 1.3932562763325853], "job": "4e4a0e26c85944f2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-two-phase-port-from17.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/wg-pdl-gather-w13-w2-chain.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/wg-pdl-gather-w13-w2-chain.patch`; branch `hyp/muse_moe_w4a16/wg-pdl-gather-w13-w2-chain`
