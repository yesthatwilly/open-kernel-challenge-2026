---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
created: 2026-10-25T011255Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 2 of the wgmma register-A grouped NVFP4 rewrite (make it numerically correct). Stage 1b
  (job b56d17fda2d54be2) finally BUILT and RAN on the judge: TORCH_CUDA_ARCH_LIST=9.0a gave only
  -gencode compute_90a/sm_90a, load_inline 32.8 s, w13+silu 168 regs/0 local, w2 168 regs/8 B
  local, 187-204 KB smem, cuTensorMapEncodeTiled x8 rc 0, and every canary launch (align,
  wg_gather_kernel, wg_kernel<w13+silu>, wg_kernel<w2>, post-norm) completed with device error
  word 0 (no fault, no mbarrier timeout). But the canary output (M=128, slot 0 of every token on
  local expert 0, 7 remote routes, randn activations, REAL retiled weights) was NOT FINITE, so
  setup raised 'canary failed' and PATH CHOSEN = MARLIN: the 1.3303 score is a master rerun. Build
  on stage 1b's code (claim bases a rewrite-on-rewrite on the parent's commit) and change only
  what follows. (1) LOCALIZE in the canary, each step after its own synchronize + flushed print:
  (a) gather: compare the slab/gathered activation buffer with x[sorted token rows] exactly; (b)
  w13+silu: compare cache2 on the LOCAL route rows only against master's Marlin w13 + silu_and_mul
  on the same inputs; print count of non-finite, first non-finite (row, col), col // BN (N-tile)
  and k-stage if derivable, rel-L2 over finite entries, and the 3 worst N-tiles; (c) w2: feed w2
  the MARLIN reference cache2 (so w2 is judged independently of w13) and compare its y on local
  rows against Marlin w2; same prints; (d) report finiteness of y separately on local rows and on
  remote/sentinel rows, since remote rows are never written (Marlin is_ep convention) and must not
  be read by the norm. (2) FIX the likely causes, all cheap and safe: zero the whole dynamic smem
  ring/slab once at CTA start before the first bulk copy (uninitialized smem read through a wrong
  B descriptor/box is the classic NaN source; bf16 garbage is often NaN/Inf), and zero-fill slab
  rows of padding/sentinel tokens explicitly rather than leaving them unwritten; make cache2/y
  zero for rows the post-norm may touch if (1d) shows remote rows are read. (3) BEFORE DISPATCH,
  without a GPU, emulate offline in numpy (as the swap-AB review did to 1e-15): the wgmma m64nNk16
  register-A fragment mapping of the dequantized weight (it differs from mma.sync m16n8k16: warp w
  of the warpgroup owns rows 16w..16w+15), the B shared-memory matrix descriptor (start address >>
  4, LBO, SBO and swizzle mode must match the swizzle the bulk-copy/TMA wrote), the mbarrier
  expect_tx byte count per stage versus the bytes the bulk copies plus TMA actually deliver now
  that the L2::cache_hint operand was dropped, and the e4m3 block-scale / global-scale
  application; fix any mismatch found. (4) Remove the w2 8-byte spill if trivial (not a
  correctness suspect). Keep tiling (BK 128, 4 stages, N buckets 16/32/64/128, grid 132 x 384
  threads, 1 producer + 2 consumer warpgroups), the num_tokens >= 128 gate with its own bound
  constant, the sm_90a build, the in-process guarded first launch and the Marlin fallback. Log
  PATH CHOSEN; a log without PATH CHOSEN = WGMMA is unmeasured. First measurement is an okc check,
  never a submit.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "M1: canary finite, SELF-CHECK PASSED, PATH CHOSEN = WGMMA, 5/5 tests. M2: wg_gather + wg_kernel x2 rows below master Marlin pair + act_and_mul (292.5 + 3.9 us in best profile; 275.0 + 3.7 in stage 1b fallback run). Floor 101.3 us, M3 <= 170 us"
  - shapes: bs=256 *
    direction: faster
    why: M2 below Marlin 391.4 + 6.4 us (best profile; 425.4 + 5.5 in stage 1b run); floor 108.9 us, M3 <= 220 us
  - shapes: bs=1 *
    direction: flat
    why: "control: gate is num_tokens >= 128"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: proceed
status: running
review_at: 2026-10-25T012041Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T012041Z
base_sha: c89b01f6d4eeee26bad681a66af3e272ffc51f07
base_blob: f09390b45e1e1af24f0ccc3645988b81f97b2bf8
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
mode: null
restack_of: null
stale_base: true
implemented_at: 2026-10-25T013639Z
impl_commit: 570f5e4ac5465931183addaf7a472e01668ced15
submission_sha256: 1ef5716e41db310ac8e17a509ee79b37b2393d829653dc2b6f99e82cb9abf814
norm_sha256: e8ad77e26f3b65404fd680d2f487a15da5857aa59b2a866c9b6253a00c9db963
submission_blob: 1ae7e9cac6dc078a0e71c7a6b0b77ec48b0ed01f
lint_warnings: 42
---

# cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix

## Change

Stage 2 of the wgmma register-A grouped NVFP4 rewrite (make it numerically correct). Stage 1b (job b56d17fda2d54be2) finally BUILT and RAN on the judge: TORCH_CUDA_ARCH_LIST=9.0a gave only -gencode compute_90a/sm_90a, load_inline 32.8 s, w13+silu 168 regs/0 local, w2 168 regs/8 B local, 187-204 KB smem, cuTensorMapEncodeTiled x8 rc 0, and every canary launch (align, wg_gather_kernel, wg_kernel<w13+silu>, wg_kernel<w2>, post-norm) completed with device error word 0 (no fault, no mbarrier timeout). But the canary output (M=128, slot 0 of every token on local expert 0, 7 remote routes, randn activations, REAL retiled weights) was NOT FINITE, so setup raised 'canary failed' and PATH CHOSEN = MARLIN: the 1.3303 score is a master rerun. Build on stage 1b's code (claim bases a rewrite-on-rewrite on the parent's commit) and change only what follows. (1) LOCALIZE in the canary, each step after its own synchronize + flushed print: (a) gather: compare the slab/gathered activation buffer with x[sorted token rows] exactly; (b) w13+silu: compare cache2 on the LOCAL route rows only against master's Marlin w13 + silu_and_mul on the same inputs; print count of non-finite, first non-finite (row, col), col // BN (N-tile) and k-stage if derivable, rel-L2 over finite entries, and the 3 worst N-tiles; (c) w2: feed w2 the MARLIN reference cache2 (so w2 is judged independently of w13) and compare its y on local rows against Marlin w2; same prints; (d) report finiteness of y separately on local rows and on remote/sentinel rows, since remote rows are never written (Marlin is_ep convention) and must not be read by the norm. (2) FIX the likely causes, all cheap and safe: zero the whole dynamic smem ring/slab once at CTA start before the first bulk copy (uninitialized smem read through a wrong B descriptor/box is the classic NaN source; bf16 garbage is often NaN/Inf), and zero-fill slab rows of padding/sentinel tokens explicitly rather than leaving them unwritten; make cache2/y zero for rows the post-norm may touch if (1d) shows remote rows are read. (3) BEFORE DISPATCH, without a GPU, emulate offline in numpy (as the swap-AB review did to 1e-15): the wgmma m64nNk16 register-A fragment mapping of the dequantized weight (it differs from mma.sync m16n8k16: warp w of the warpgroup owns rows 16w..16w+15), the B shared-memory matrix descriptor (start address >> 4, LBO, SBO and swizzle mode must match the swizzle the bulk-copy/TMA wrote), the mbarrier expect_tx byte count per stage versus the bytes the bulk copies plus TMA actually deliver now that the L2::cache_hint operand was dropped, and the e4m3 block-scale / global-scale application; fix any mismatch found. (4) Remove the w2 8-byte spill if trivial (not a correctness suspect). Keep tiling (BK 128, 4 stages, N buckets 16/32/64/128, grid 132 x 384 threads, 1 producer + 2 consumer warpgroups), the num_tokens >= 128 gate with its own bound constant, the sm_90a build, the in-process guarded first launch and the Marlin fallback. Log PATH CHOSEN; a log without PATH CHOSEN = WGMMA is unmeasured. First measurement is an okc check, never a submit.

## Expected

- `bs=128 *` faster: M1: canary finite, SELF-CHECK PASSED, PATH CHOSEN = WGMMA, 5/5 tests. M2: wg_gather + wg_kernel x2 rows below master Marlin pair + act_and_mul (292.5 + 3.9 us in best profile; 275.0 + 3.7 in stage 1b fallback run). Floor 101.3 us, M3 <= 170 us
- `bs=256 *` faster: M2 below Marlin 391.4 + 6.4 us (best profile; 425.4 + 5.5 in stage 1b run); floor 108.9 us, M3 <= 220 us
- `bs=1 *` flat: control: gate is num_tokens >= 128
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

The canary used real weights and randn x, so a non-finite y is a kernel/descriptor/layout bug, not data. If (1) shows y non-finite only on remote rows, the fix is the norm read/zeroing and the GEMMs may already be right. A sticky fault is still unrecoverable in-process; keep -lineinfo and flushed prints.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T012041Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run this one first, as the single stage-2 check. Fold in the diagnostic half of s2-per-kernel-enable: judge w13+silu and w2 independently, and do not abort on the first non-finite value. Also add prepare-time per-kernel timing against Marlin and a compact summary printed last. One check then names the faulty kernel and tensor, and gives M2 evidence for both kernels even if correctness still fails. Hold the two siblings on this result.

### Justification
Design (kind: rewrite): unchanged from stage 1b and still sound. At bs=128 and 256, Marlin is 292.5 and 391.4 us, 95% of each shape (FINDINGS), against floors of 101.3 and 108.9 us. A swap-AB wgmma with register-A dequant and TMA-fed token slabs is the right tensor-core path there. Stage 1b proved the build (sm_90a only, 32.8 s), the TMA maps (rc 0) and every launch (error word 0). The remaining problem is numerical. Reading stage 1b's code (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc) narrows the suspects. (a) The E2M1/E4M3 dequant bit trick, the code/scale retiling order and the A-fragment register order are verbatim from the swap-AB sibling, which passed rel-L2 9.2e-3 on the judge. They are unlikely to cause the bug, though wgmma RS should still be checked to confirm it uses the same per-warp A layout as mma.sync m16k16 (rows 16w..16w+15). (b) The B descriptor (start>>4, LBO 1, SBO 1024, layout type 1 = 128B swizzle, +32 B per k16, 1024-aligned boxes) and expect_tx = 9216 + 2*(16<<b)*128 look right on paper and are exactly what the numpy emulation must confirm. (c) The canary checks only `out`, and cache3 is torch.empty. Any local route whose w2 row is never written (a chunk-coverage or epilogue-scatter bug) is uninitialised caching-allocator memory, which is a very common non-finite source and needs no wrong math at all. wg_hs is torch.zeros, so an unwritten h row shows up as a wrong value rather than a NaN. The canary covers only bucket N=128 with one 8-block chunk: N=16/32/64 and multi-chunk experts have never run. WEAKEST LINK: the spec's fix (2), zeroing the smem ring and the cache2/y rows, can turn a NaN into a finite wrong answer. It masks an expect_tx-short race or an unwritten row instead of fixing it. Finiteness is therefore not the pass criterion: canary and self-check must gate on rel-L2 against Marlin per kernel. Use a SENTINEL fill, not zeros, in the diagnostic pass so 'never written' and 'computed non-finite' can be told apart. SECOND: wgmma_wait<1> keeps one group in flight while the next iteration's dequant rewrites the A registers. That is undefined behaviour for RS wgmma unless ptxas serialises it. ptxas normally does serialise it, with the warning 'wgmma.mma_async instructions are serialized due to non wgmma instructions defining input registers'. The result is correct but slow, so it is an M2 risk rather than a NaN suspect. The build stderr is never shown (the archived log is a ~15 KB tail), so capture it and print a one-line summary of any 'serialized' warning late. Expected directions (bs>=128 faster, bs<128 flat controls) are consistent with TASK.md weights and floors.

### Confounders on the current kernel
- align-parallel-per-expert-verify (master): at bs>=128 the wgmma path bypasses _align_parallel_kernel (4.0/6.9 us) for expert_map gather + moe_align_block_size(16) + wg_gather. Add that delta to the replaced-kernel sum. It is small against ~300-400 us and does not confound M1/M2.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: these set the incumbent Marlin rows (block_size_m 16 at bs=128, 32 at bs=256). Compare against this run's own fallback-free master profile. Marlin row noise between identical-code runs is +/-10-20% (275.0 vs 292.5 us at bs=128, 425.4 vs 391.4 us at bs=256).
- post-expert-norm-triton-small-m: no interaction at bs>=128, because both paths use SGLang's fused norm with slot_expert_ids. It does matter for (1d): the Triton kernel's docstring says remote y rows are uninitialised and skipped, and the fused SGLang kernel is assumed to skip them as well. Verify this rather than assume it.
- sglang-marlin-path-parity: defines the Marlin bank used as the reference in the canary and self-check. A shared quantisation-convention error (global-scale direction) would cancel against Marlin, but the existing fp32-dequant expert check covers that.

### Interference and A/B needs
No ab-revert or ab-tune companion is needed: it is a rewrite, and the fallback keeps master's path, so the landed hypotheses cannot fake a gain. The real interference is between the three stage-2 siblings, which all depend on the same unfixed bug. Running them in parallel spends three checks to learn one fact. Merge the diagnostic part of s2-per-kernel-enable into this run. Do NOT merge its mixed Marlin/wgmma production path: that needs a bridge, and Marlin's block_size_m (16/32) differs from the wgmma align block 16, so it is a stage-3 item only if exactly one kernel passes. Hold s2-fix-gate-from-8 until this run gives M1/M2 at >=128. Required design so the single check pinpoints the fault: (A) never raise on the first non-finite value. Run the canary as a diagnostic that always completes, then fall back. (B) Pre-fill wg_hs, wg_xs pad rows and cache3 with a distinctive finite sentinel (e.g. bf16 -12345.0) for the diagnostic pass only, and report per tensor: rows unwritten (still sentinel), rows non-finite, first bad (row, col, col//BN, tile j, chunk, bucket) and rel-L2 on finite rows. (C) Judge w13+silu (h at sorted slots mapped back to routes, against Marlin w13 + silu_and_mul) and w2 (fed Marlin's cache2 gathered into h_sorted by the existing gather logic with sid instead of sid>>3, against Marlin w2) independently. Do this at the canary M=128 AND at a multi-bucket case: M=256 skewed hits buckets 16/32/64 and multi-chunk experts. (D) If launches stay fault-free, run the full self-check in diagnostic mode even when the canary fails numerically. It has already been shown not to fault, and its per-case h/y/out rel-L2 is the localisation. Cut its graph before/after replay lines to one line per case so the tail is not flooded. (E) Time each wgmma kernel against its Marlin counterpart in prepare at M=128 and 256, using CUDA events and the median of 20. A NaN does not change speed, so M2 evidence arrives even if M1 fails. Discount by the 20-25% prepare undercount noted in LESSONS and treat the number as optimistic if the bug is an early-release race. (F) Print one compact block LAST in prepare: build flags, regs/spill, ptxas 'serialized' yes/no, per-kernel verdict (unwritten/non-finite/rel-L2), per-kernel prepare us vs Marlin, and PATH CHOSEN. The archived log keeps only the last ~15 KB. Do the numpy emulation in (3) before dispatch, as the spec says. If it finds a definite mismatch, fix it and still keep A-F.

### Open questions
- Does SGLang's fused_post_expert_norm_weighted_sum with slot_expert_ids skip slots with local id -1? Read its source offline before the check. If it does, (1d) is moot and the canary's 7 remote routes are harmless.
- Does ptxas serialise the wgmma pipeline (A registers rewritten while one group is in flight)? If it does, M2 needs double-buffered A fragments, or wait<0> before the dequant of stage n+1 overlaps differently. That is the stage-3 performance fix.
- FINDINGS line 66 says stage 1b 'gained on verify_16; verify_32'. That is wrong: it was a Marlin rerun, and it should be corrected so later hypothesis minting does not cite it.
- Prepare budget: build 32.8 s, plus diagnostic canary, full self-check (20-60 s) and per-kernel timing. Keep WG_SETUP_BUDGET_S and log the total seconds.


## Implementation notes

Only `muse_moe_w4a16/submission.py` in the worktree changed; the CUDA kernel source is untouched. Nothing could be run (no GPU, no torch here). The review wins over the spec wherever they differ: the spec's fix (2), zeroing the smem ring and the cache2/y rows, was NOT applied in production because it can turn a NaN into a finite wrong answer. This is a diagnostic-first stage.

What changed (all in prepare()-time Python):
- `_wg_canary` is replaced by `_wg_diag` + `_wg_eval_case`: a non-faulting diagnostic that never raises on a bad value. `wg_xs`, `wg_hs` and cache3 are pre-filled with a bf16 sentinel (-12288.0), so "row never written" (unwritten) differs from "computed non-finite". Per case, each launch is announced with a flushed print and synced after: gather (xs rows exact vs x, pad rows zero), w13+silu (h at sorted slots vs Marlin cache2), w2 fed its OWN h, w2 fed MARLIN's cache2 (judged independently of w13) vs Marlin y, then the fused norm with sentinel vs NaN remote rows (answers open question 1: does it skip slot id -1), then the production `_moe_wgmma` out vs Marlin out. Reported per launch, per N bucket (16/32/64/128, plus "noslot"), per row (first bad route/token/col/N-tile/kind/bucket, up to 4 bad rows, 3 worst N-tiles), with unwritten / partial / non-finite counts and rel-L2 over all rows and over finite rows. Pass/fail is rel-L2 <= 1e-2 against Marlin per kernel (a sentinel or NaN row makes it large or inf), never finiteness. Cases: stage-1b canary (M=128, 1 local route/token), mixed M=128 and mixed M=256 (local loads chosen to hit buckets 16/32/64/128, and multi-chunk experts of 16 and 9 blocks).
- cache3 coverage (review point 4): the sentinel fill reports any local route whose w2 row is unwritten (and any local route without a sorted slot). No kernel fix was made because the cause is unknown until the check shows which rows.
- `_wg_self_check` runs in the same diagnostic mode: per case one line (h rel-L2, w2-fed-Marlin-cache2 rel-L2, own-h w2, out, graph replay), no per-launch or per-replay lines, no early stop on a failed case (only on a device error word). Graph replay and the fp32-dequant expert check are kept.
- Build stderr (review point 6): a probe build with `-Xptxas -Werror` under a separate extension name runs first. load_inline discards a good build's stderr but a failed build's exception carries it, so ptxas warnings (the wgmma 'serialized' one, C7510) are extracted in-process, with no fd redirection and no child process of ours. A passing probe is used as the extension; otherwise the normal build runs (about +33 s, still inside WG_SETUP_BUDGET_S, skipped if under 60 s of budget remain).
- Timing (point 5): `_wg_time_vs_marlin` times wg_gather, w13+silu, w2 against Marlin w13, silu_and_mul, w2 at M=128 and 256 (all 32 local experts hit, uniform), CUDA events, median of 20 samples of 5 eager launches, warm L2. Print only; also runs if correctness failed (skipped on a device error word or past 255 s).
- Summary (point 7): one compact block printed LAST from `_wg_setup` (build flags, ptxas warning, regs/spill/smem, per-kernel verdicts and counts, self-check, timing, total seconds), then the `PATH CHOSEN` line. Enable rule (point 8): WGMMA only if the h kernel and the w2 kernel (fed Marlin's cache2) both pass in every diagnostic case, the self-check passes, and the error word is 0; otherwise Marlin. Scratches are zeroed again at the end (as in stage 1b).

The spec's fix (3), the offline numpy emulation (scratchpad emu/emu.py), found NO mismatch: the retile followed by the lane's A fragment reads (register i = row g + 8 (i&1), k16 half i>>1, nibbles 3-i / 7-i, warp w owns rows 16w..16w+15, same as mma.m16n8k16) reproduces the source codes and the E4M3 scale bytes with 0 mismatches for w13 and w2 shapes; expect_tx = 9216 + 2 * rows * 128 equals the bulk copy plus the two TMA boxes for all four buckets (13312/17408/25600/41984 B); the descriptor (start >> 4, LBO 1, SBO 1024, 128B swizzle, +32 B per k16, 1024-aligned boxes) matches the TMA swizzle; the w13 and w2 stmatrix .trans epilogue mappings are exact. So the NaN is not in those three; the check will say which kernel and tensor. Not done: fix (1)-(2) zeroing (masking, see above), fix (4) the w2 8-byte spill (not trivial, not a correctness suspect), and the open question on the fused norm's source (answered at run time by the NaN-remote test instead). FINDINGS line 66 correction is not in this file's scope.

## Rules check

- [x] CUDA-graph capturable: kernel_fn / _moe_wgmma are unchanged; all new code (diagnostics, sync, .item(), timing) is in prepare()-time functions, never in kernel_fn's call graph (checked against `git diff`).
- [x] Nothing derived from `data` is cached across calls: the diagnostic inputs are synthetic and seeded; the persistent scratches are zeroed after the diagnostic.
- [x] Weights: no change to the lossless retiling or the Marlin repack; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention: n/a, this is a MoE problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no subprocess/fork/Popen/multiprocessing in the file (grep clean); the -Xptxas -Werror probe is an extra load_inline flag only, no fd redirection, no monkeypatching of torch.
- [x] All work happens inside the timed call; no output caching or replay: kernel_fn's two paths are unchanged.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a set and restored around load_inline (both builds); unchanged guards and error word 2; every wait is bounded; any failure lands on Marlin without raising.
- okc_lint: OK with warnings: muse_moe_w4a16/submission.py; py_compile ok; __pycache__ removed. Full lint output has only prepare()-time host-sync warnings (outside kernel_fn's call graph) and two dtype warnings for the uint8 retile buffers (unchanged from stage 1b).

## Pre-flight audit

Audited 2026-10-25 (opus), by reading only (no GPU here). **Verdict: dispatch-with-fixes.**

Checked and found sound:
- The per-kernel gate is rel-L2 <= 1e-2 against Marlin on local routes. A sentinel or non-finite row makes it inf, so it is not a finiteness gate. It is reachable: bf16-rounding differences are about 3e-3.
- The Marlin calls in `_wg_time_vs_marlin` match master's `_moe_marlin`: the same argument order, the two-smaller block_size_m pick, `_align_parallel`, and silu_and_mul's expert_ids.
- The sentinel bookkeeping has no out-of-bounds index: slot_of, blk_bucket, `x[sid>>3]`, `r_c2[sid[real]]`, and the `hs_buf[:used][mask]` writes through the view.
- The all-remote cases (n_padded 0) take the empty paths cleanly.
- kernel_fn and `_moe_wgmma` are unchanged. All new syncs and `.item()` calls are in prepare().
- The only child process is load_inline's own library build.
- There is no ptxVersion==90 gate. The gate is binaryVersion==90, and ptxVersion is only logged.

Budget: stage 1b's whole run took 101.6 s from start to finish. This run's prepare() should take about 110-135 s:
- master prepare: about 20-25 s
- the -Werror probe, if ptxas warns: about 33 s
- the main build: about 33 s
- the diagnostic and the 20-case self-check: 20-40 s
- timing: under 3 s

The second build is gated at under 140 s elapsed, the self-check at 200 s and the timing at 255 s. That keeps prepare() under 300 s even if the builds are slow, so no budget fix was needed.

Fixes applied (submission.py only):
1. `_wg_ptxas_warnings` now also matches `ptxas error` lines, because under -Werror a promoted warning can print that way. It also matches info lines that mention `serialized` or `Performance`, and lists any `serialized` line first. Before this, a failed probe could report 0 warnings and serialization "unknown".
2. A passing -Werror probe no longer claims "no wgmma serialization". The C75xx serialization note can be info-level, which -Werror does not fail and load_inline discards. The summary now says "no warning (info-level note not visible)".
3. Every section now catches, prints and continues: each diagnostic case, the `_wg_diag` call, the `_wg_self_check` call and each self-check case. A Python error marks only that section FAIL. Before, it skipped the self-check and the timing too. The summary still prints last, and wgmma stays off.
4. Log tail: the self-check's separate per-case FAIL line is now part of the case line, saving about 3.6 KB over 20 cases. The verbose per-launch diagnostic (about 9 KB) would otherwise fall out of the ~15 KB tail behind the self-check, so the summary now carries per-case gather, ids and coverage, and part counts. For a failing kernel it also carries the per-N-bucket breakdown, the first bad (route, tok, col, tile, kind, bucket) and the worst tiles. The worst dict is formatted compactly.
5. Every diagnostic launch is now announced and synced: the Marlin reference plus align, the fused-norm pair, and the production `_moe_wgmma` call, which is now traced in the verbose diagnostic.

okc_lint: OK with warnings (prepare()-time host syncs only). py_compile ok. __pycache__ removed.

Residual risk: a sticky device fault in any diagnostic launch still kills prepare, as in stage 1b. Serialization can only be confirmed if ptxas reports it as a warning or error.
