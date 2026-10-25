---
problem: muse_attention_decode
slug: full-verify-wgmma-s4-multitile-fix
created: 2026-10-25T035310Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, multi-tile correctness)
change: >
  Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit
  (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc,
  splitkv_verify_wgmma_kernel<ticket>/<B8>, module muse_full_verify_wgmma_v1) whatever its score;
  rename the module (v2) so it recompiles. Stage 3 built clean (sm_90a, 128/120 regs, 0 local, 2
  CTAs/SM, 91744 B smem) and passed the canary (B=2, grid 4) and case 1 (B=6 lens_a, kv 1.0,
  grid=device) at rel L2 <= 7.9e-3, but failed case 2 (B=6 lens_a, kv_scale 0.25, grid 3) at rel
  L2 1.23, finite, in BOTH ticket and B8 combine, so PATH CHOSEN fell back to master FA3 and the
  kernel was never timed. Every passing launch gave each CTA at most 1 tile; case 2 is the first
  with several (~81) tiles per CTA. (1) BISECT in the prepare self-check, each case after its own
  synchronize + flushed print of rel L2, finiteness and the DEVICE ERROR WORD (bounded mbarrier
  wait timeout sets it; stage 3 never printed it): split case 2 into (lens_a, kv_scale 1.0, grid
  3) and (lens_a, kv_scale 0.25, grid 264); add single-request (B=1, no request crossing) cases
  sized to give exactly 2, 4 and 17 tiles per CTA at kv 1.0 (e.g. grid 1 with 128/256/1088 keys,
  or grid 4 with 8/16/68 tiles); keep the canary, case 1 and original case 2. This one okc check
  must name which variable breaks: multi-tile loop state, request crossing, IDX_TILES page-index
  rotation past 16 tiles, or kv_scale. (2) FIX the most likely cause, the per-CTA multi-tile loop
  state: (a) online-softmax carry across tiles: m and l carried per row in registers, acc (wgmma
  PV accumulator) rescaled by exp2(m_old - m_new) BEFORE the next PV wgmma is issued, after
  wgmma.wait_group drains the previous one, and l rescaled with it; reset m=-inf, l=0, acc=0 only
  at a segment/request boundary, never per tile; (b) the K/V smem ring phase parity: full/empty
  mbarrier phase bit flips each time the slot index wraps the stage count, tracked per slot across
  tiles and segments (not reset per tile or per request), producer and consumer agree on the same
  (iteration / STAGES) & 1; the V^T double buffer index advances per tile, not per segment; (c)
  the FP8 P operand uses FA3 Max_offset across tiles: P = exp2(s*scale_log2 - (m_new - 8)) with
  the same offset folded into the l and acc rescale, so the 2^8 cancels in O/l and in the partial
  merge (LESSONS 2026-10-25); (d) kv_scale is applied once per tile (to s before the max and to
  the PV output), not accumulated into the carry. Q reload on request crossing and page-index
  rotation stay as they are unless the bisect names them. The self-check must still pass all cases
  for the kernel to be chosen (PATH CHOSEN: full verify wgmma v2 vs master FA3 (fallback: <case,
  rel L2, error word>)); when it passes, print a CUDA-event time per kernel at B=16 and 32 vs
  master paired FA3 main+combine+norm-gate. BUILD/SAFETY as stage 3: TORCH_CUDA_ARCH_LIST=9.0a /
  -gencode arch=compute_90a,code=sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo,
  in-process canary, never a child process. Test FP8-P on coherent long-context keys, not only
  random keys. First measurement is an okc check, never a scored submit. Do not touch sliding
  verify or the decode paths (the held s4 sliding-window and verify1-batch1 stages are re-minted
  after this passes).
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "M1: every self-check case passes and PATH CHOSEN = wgmma; M2: replaced FA3 main+combine+norm-gate below master 162.0 us shape (FA3 paired); stage-2 A8 ran at 1.44 TB/s, wgmma target >= 2.4 TB/s"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "M2: below master 300.8 us shape"
  - shapes: "*load=decode_* *layer=sliding* *layer=full load=verify_1 *"
    direction: flat
    why: outside the 1<batch<=64 full verify gate; master code
risk: high
depends_on: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)"
review: proceed
status: running
review_at: 2026-10-25T040253Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T040258Z
base_sha: ff9bf1e4fbe38f9f847a906f5135ecaf60604668
base_blob: 136874e0158f8b76208c4130e523bc91b84160ad
branch: hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix
worktree: .worktrees/muse_attention_decode/full-verify-wgmma-s4-multitile-fix
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T041932Z
impl_commit: 3ee2b893476879190a690392a3295a35afed11bf
submission_sha256: c0b8173eb722f5a655d5e072dbcded9cf8bb90cde652ab077ad34e1b184c3d87
norm_sha256: b4c0abba9555df955e632b86d0dcbe8b2bd85739ce933ff73a04d14abc9455f0
submission_blob: 9bcf522b0dd23a0a8173046d2150473827122206
lint_warnings: 11
attempt_file: 2026-10-25T041932Z-check-full-verify-wgmma-s4-multitile-fix.md
prepared_at: 2026-10-25T041932Z
job: 51b0d5c110f3459c
dispatched_at: 2026-10-25T042009Z
---

# full-verify-wgmma-s4-multitile-fix

## Change

Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc, splitkv_verify_wgmma_kernel<ticket>/<B8>, module muse_full_verify_wgmma_v1) whatever its score; rename the module (v2) so it recompiles. Stage 3 built clean (sm_90a, 128/120 regs, 0 local, 2 CTAs/SM, 91744 B smem) and passed the canary (B=2, grid 4) and case 1 (B=6 lens_a, kv 1.0, grid=device) at rel L2 <= 7.9e-3, but failed case 2 (B=6 lens_a, kv_scale 0.25, grid 3) at rel L2 1.23, finite, in BOTH ticket and B8 combine, so PATH CHOSEN fell back to master FA3 and the kernel was never timed. Every passing launch gave each CTA at most 1 tile; case 2 is the first with several (~81) tiles per CTA. (1) BISECT in the prepare self-check, each case after its own synchronize + flushed print of rel L2, finiteness and the DEVICE ERROR WORD (bounded mbarrier wait timeout sets it; stage 3 never printed it): split case 2 into (lens_a, kv_scale 1.0, grid 3) and (lens_a, kv_scale 0.25, grid 264); add single-request (B=1, no request crossing) cases sized to give exactly 2, 4 and 17 tiles per CTA at kv 1.0 (e.g. grid 1 with 128/256/1088 keys, or grid 4 with 8/16/68 tiles); keep the canary, case 1 and original case 2. This one okc check must name which variable breaks: multi-tile loop state, request crossing, IDX_TILES page-index rotation past 16 tiles, or kv_scale. (2) FIX the most likely cause, the per-CTA multi-tile loop state: (a) online-softmax carry across tiles: m and l carried per row in registers, acc (wgmma PV accumulator) rescaled by exp2(m_old - m_new) BEFORE the next PV wgmma is issued, after wgmma.wait_group drains the previous one, and l rescaled with it; reset m=-inf, l=0, acc=0 only at a segment/request boundary, never per tile; (b) the K/V smem ring phase parity: full/empty mbarrier phase bit flips each time the slot index wraps the stage count, tracked per slot across tiles and segments (not reset per tile or per request), producer and consumer agree on the same (iteration / STAGES) & 1; the V^T double buffer index advances per tile, not per segment; (c) the FP8 P operand uses FA3 Max_offset across tiles: P = exp2(s*scale_log2 - (m_new - 8)) with the same offset folded into the l and acc rescale, so the 2^8 cancels in O/l and in the partial merge (LESSONS 2026-10-25); (d) kv_scale is applied once per tile (to s before the max and to the PV output), not accumulated into the carry. Q reload on request crossing and page-index rotation stay as they are unless the bisect names them. The self-check must still pass all cases for the kernel to be chosen (PATH CHOSEN: full verify wgmma v2 vs master FA3 (fallback: <case, rel L2, error word>)); when it passes, print a CUDA-event time per kernel at B=16 and 32 vs master paired FA3 main+combine+norm-gate. BUILD/SAFETY as stage 3: TORCH_CUDA_ARCH_LIST=9.0a / -gencode arch=compute_90a,code=sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, in-process canary, never a child process. Test FP8-P on coherent long-context keys, not only random keys. First measurement is an okc check, never a scored submit. Do not touch sliding verify or the decode paths (the held s4 sliding-window and verify1-batch1 stages are re-minted after this passes).

## Expected

- `*layer=full load=verify_16*` faster: M1: every self-check case passes and PATH CHOSEN = wgmma; M2: replaced FA3 main+combine+norm-gate below master 162.0 us shape (FA3 paired); stage-2 A8 ran at 1.44 TB/s, wgmma target >= 2.4 TB/s
- `*layer=full load=verify_32*` faster: M2: below master 300.8 us shape
- `*load=decode_* *layer=sliding* *layer=full load=verify_1 *` flat: outside the 1<batch<=64 full verify gate; master code

## Risk

If the bisect shows a fault outside the multi-tile loop state (request crossing, IDX_TILES rotation, kv_scale), the fix attempt may not land but the check still names the cause for stage 5. Held s4 siblings (sliding-window, verify1-batch1-grid-sm) wait on this.

## Milestones

Rewrite stage 4 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound rewrite stage: the bisect cases separate the variables that case 2 confounded, so one check names the faulty variable whether or not the fix lands. However, the four fixes the change lists, (a) to (d), are already implemented in the stage-3 kernel, so the bisect output, not a blind rewrite of the loop, has to drive the fix.

### Justification
Diagnosis: case 2 (B=6, kv 0.25, grid 3, ~81 tiles per CTA) failed at rel L2 1.23, finite, with both combines. Case 1 (grid=device, at most 1 tile per CTA) passed. That places the bug in the main kernel and in something that only happens when a CTA runs several tiles or crosses requests. The harvest supports this, and the bisect plan is the right one: kv 0.25 alone; multi-tile with B=1 at 2/4/17 tiles, so no request crossing and the 17-tile case covers IDX_TILES rotation; plus the original case. WEAKEST LINK: I read splitkv_verify_wgmma_kernel on the s3 branch and the suspects (a) to (d) are already coded as the spec asks. m/l/oacc reset only at segment start (n==0 || lt==0). oacc is rescaled after wg_wait0 of the previous PV, i.e. before the next PV is issued. The full parity is (n/STAGES)&1 and the empty-wait parity ((n-1)/STAGES)&1 is correct for the refill slot. V^T uses buffer n&1 per tile. The FA3 Max_offset (P = 2^(s - (u - 8))) is present, with l and O both carrying the 2^8. k_scale is folded into the per-tile score scale and v_scale is applied in the merge. So the 'most likely cause' named in step (2) is probably not the cause. What I could not rule out by reading: (i) the bounded-wait abort, where flag_s[0] makes every later wait return at once and produces finite garbage of exactly this kind (rel L2 ~1); stage 3 never printed the error word; (ii) the page-index prefetch and request crossing inside a 16-tile meta block (issue_meta uses req_of per tile, which looks right); (iii) start_of/req_of/slot = bx + cb bookkeeping when a CTA spans several requests (the slot mapping is collision-free by my reading); (iv) kv_scale 0.25 with grid=device, which was never run. The milestones are physically plausible. verify is MMA-heavy (64 query rows per kv head = one m64 wgmma), wgmma is the right tensor-core path where mma.sync lost 1.7x (LESSONS 155), and the 2.4 TB/s target is ~1.7x stage 2's 1.44 TB/s. The expected direction (full verify_16/32 faster, everything else flat) matches the gate.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv / sliding-deep-splits-on-cuda-splitkv (master): the s3 TU shares the pre-attention Triton kernel and the B8 combine lineage with the decode split-KV path. A passing self-check only clears the verify kernel. Any decode-row move in this check is master drift (full verify_1 and sliding verify_1 drifted +2-6% in the 03:41Z batch).
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup (master): the incumbent being replaced is paired FA3 main + combine + norm-gate. M2 must compare the wgmma kernel against those three rows in the same check, not against the headline.
- full-decode1-streamk-s3-two-level-ticket (land candidate, not on master): untouched by this stage, but if it lands first the verify TU must be rebased (both add load_inline modules to the same file). No measurement confound.

### Interference and A/B needs
No companion. This is a rewrite stage whose deliverable is a named cause (M1) and, if the self-check passes, the kernel's own profile rows against FA3 (M2). The B=1 multi-tile cases and the split of case 2 are the in-run controls. A separate ab-revert run would add nothing that the per-case prints cannot. The next stage is the follow-up.

### Open questions
- Did case 2 trip the bounded-wait abort (error word 2)? Finite garbage at rel L2 ~1 is exactly what the abort path produces. If so, the bug is a real barrier deadlock or a too-small WAIT_LIMIT at 81 tiles per CTA, not softmax state.
- BUILDER: make the bisect name the failing segment, not just the failing case. For the failing case, copy the per-slot partials (ws_acc O, ws_ml m/l) to host and compare each slot against a torch fp32 reference of that exact (CTA range, request) segment. Print the first bad slot with its CTA, request, tile count, whether it follows a request crossing, and whether its tile index is past 16. Also print the per-case error word. This turns one check into a line-level diagnosis even if no (2) fix lands.
- BUILDER: do not rewrite items (a) to (d) blindly. They read correct on the s3 branch (reset at lt==0, rescale after wg_wait0, Max_offset, parity). Change the loop only where the bisect points, and keep any speculative edit behind a constexpr so the s3 path can be A/B'd inside the self-check.
- When the self-check passes, choose wgmma regardless of the prepare timing (print both). This is a check, and M2 needs the kernel's profile rows. A timing hard gate would hide it, and prepare under-reads the judge by 20-25%.


## Implementation notes

Edited only `.worktrees/muse_attention_decode/full-verify-wgmma-s4-multitile-fix/muse_attention_decode/submission.py` (no GPU here; nothing was run).

**Review wins over the spec.** The review says fixes (a)-(d) are already in the s3 kernel, so I did NOT rewrite the loop. I re-read the kernel independently and agree: m/l/oacc reset only at `n == 0 || lt == 0`; oacc/l rescaled by ex2(m_old - m_new) after the previous PV `wg_wait0` and before the next PV issue; full parity `(n/STAGES)&1`, refill empty parity `((n-1)/STAGES)&1` (checked for aliasing and the initial-phase trick, correct); V^T buffer `n&1`; Max_offset 8 present; k_scale folded into the score scale and v_scale in the merge; slot `bx + cb` is collision-free; the index-block overwrite is ordered after the last read by the empty-barrier chain. No arithmetic or barrier fault is visible from reading, so there is no fix to land blind.

**One review hypothesis is already ruled out by the s3 code.** The s3 self-check returned on a nonzero error word BEFORE computing rel L2, and case 2 failed with a rel L2 of 1.23, so the error word was 0: the bounded-wait abort (word 2) did not fire in case 2. The garbage is real, not a timeout. Also: every s3 passing launch (canary grid 4 over 3 tiles; case 1 on grid 264 over 242 tiles) gave each CTA at most one tile, so loop iterations n >= 1 (ring stages 1-3, V^T buffer 1, oacc/l/m carry, refill with empty wait) were never exercised by any passing case. That, not kv_scale, is the real untested surface.

What changed:
1. Module renamed `muse_full_verify_wgmma_v2` (recompiles). Same sm_90a flags, `__CUDA_ARCH_FEAT_SM90_ALL` guard, `-lineinfo`, TORCH_CUDA_ARCH_LIST=9.0a around load_inline, no child process.
2. Kernel is now `<TICKET, SYNC>`; SYNC (constexpr, diagnostic A/B) adds one `__syncthreads()` at the end of every tile, serializing both warpgroups and the producer side. Instantiated as `<true,false>`, `<false,false>`, `<true,true>`. Default path is byte-for-byte the s3 loop. New mode `ticket_sync` is tried last: if the s3 loop fails but this passes, the fault is a pipeline race and the check also leaves a working (slower) kernel chosen.
3. Self-check runs EVERY case (no early return), prints per case: rel L2, finiteness and the DEVICE ERROR WORD, PASS/FAIL. Cases: canary; case 1 (lens_a kv 1.0 device); lens_a kv 1.0 grid 3; lens_a kv 0.25 grid 264; B=1 single CTA with 2 / 4 / 17 / 34 tiles (34 rotates both index buffers and wraps the ring); B=1 4 tiles kv 0.25; [128,128] grid 1 (request crossing, 4 tiles); lens_a kv 0.25 grid 3 (the s3 failure); stage-2 leftovers (grid 7, 61, lens_b); two coherent-key cases (planted keys along the last query, 0.3x noise: peaked softmax with a long tail for the FP8 P range). B=1 is allowed through the gate only inside the self-check (`_WG_ANYB`).
4. For failing cases (first four, ticket and ticket_sync modes) `_wg_diagnose` copies the per-slot partials (ws_acc O, ws_ml m/l) and compares each (CTA range, request) segment to an fp32 reference of exactly that segment (O/l and l with the 2^8 offset, raw m), prints bad/good counts by tiles per segment, after request crossing, CTA-relative tile >= 16, starts at CTA tile > 0, and the first four bad segments with cta, req, tiles, whether it follows a crossing, tile index past 16, worst row, per kv head error. `_wg_verdict` turns the case table into a BISECT line naming the variable (single-tile math, multi-tile loop state, long-case index/ring rotation, request crossing, kv_scale).
5. Choice: the first mode whose whole self-check (incl. graph replays) passes is chosen regardless of timing. `_wg_time` then prints CUDA-event times (eager kernel_fn, B=16 and 32, ragged ~2-5.8k keys) for wgmma vs master paired FA3, information only. Final line: `PATH CHOSEN: full verify wgmma stream-K v2 (<mode>) vs master FA3` or `... master FA3, paired heads vs wgmma v2 (fallback: <case, rel L2, finite, error word>)` (reasons of all three modes joined).

Deviations from the spec: no loop rewrite (see above); added the SYNC A/B variant and ticket_sync mode (review: keep speculative edits behind a constexpr); added a B=1 34-tile and a [128,128] crossing case.

Biggest risk: compile time grows with a third kernel instantiation, and if the bug is arithmetic in the loop the check yields a diagnosis (named segment) but no wgmma path (falls back to master FA3).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds no `.item()`/sync/alloc on input values; the only new kernel_fn-side code reads module globals (`_WG_KEEP`, `_WG_MODE`) set by the self-check; all syncs and prints are in prepare-time self-check functions (lint: host-sync warnings only outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls (`_WG_KEEP` is only set during the self-check on synthetic data and cleared).
- [x] Weights: n/a: no weight handling touched.
- [x] Activations BF16, KV cache FP8, queries FP8 E4M3: unchanged kernel math.
- [x] Attention sees every key: unchanged; causal mask only inside the 8-token block.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, or child processes (grep for subprocess/multiprocessing/fork/spawn is empty).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] sm_90a-only PTX stays under `__CUDA_ARCH_FEAT_SM90_ALL` with an error-flag `#else`; every mbarrier wait bounded; build/self-check failure lands on master paired FA3, never raises.

okc_lint: `OK with warnings: submission.py` (11 host-sync warnings, all in prepare()-only self-check code, outside kernel_fn's call graph). `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Pre-flight audit

Audited 2026-10-25 (model opus). **Verdict: dispatch-with-fixes.** Rules OK.

Multi-tile bug hunt (loop iterations n >= 1). By reading, again no fault: the FULL parity `(n/STAGES)&1` and the refill EMPTY parity `((n-1)/STAGES)&1` are correct, a barrier can never run two phases ahead, and the noinc FULL count of 256 matches. The page-index block k is written by tile 16(k-1)'s cp.async and covered by that tile's FULL arrival, which every thread waits on at loop 16k-16, before its first read at loop 16k-3. The overwrite of buffer k&1 (issued at loop 16k-15) comes after every thread's last read of block k-2, ordered by the EMPTY chain. V^T buffer n&1 is ordered against PV n-2 by the tile n-1 named barrier. m/l/oacc reset only at segment start, the oacc/l rescale happens after the previous PV wg_wait0, Max_offset is consistent in l, O and the merge, and the slot/owner/req bookkeeping is collision-free. A numpy emulation of the kernel's exact control flow (per-CTA flat tiles, 4-stage ring and double-buffered 16-tile index blocks in issue order, segment reset and store, E4M3 P with offset 8, the ticket/B8 LSE merge) reproduces the fp32 reference at rel L2 4.7e-3 (lens_a grid 264), 8.2e-3 (lens_a grid 3, kv 1.0 and 0.25), 8.8e-3 (B1 34 tiles), 8.2e-3 (2-request crossing) and 7.1e-3 (grid 7). So the stage-3 algorithm and bookkeeping are right as written. Ticket and B8 both gave exactly 1.23e+00, which points to a deterministic hardware-semantics fault that only shows on n >= 1: ring stages 1-3 reuse, V^T buffer 1, or an async-proxy view of a reused stage. It is not arithmetic. The bisect plus DIAG will localize it.

Gate passability. Coherent cases emulate at 1.76-1.81e-2 over 5 seeds at grid 3 (1.3e-2 at device grid), under the 3e-2 self-check threshold. All other cases are under 9e-3. No gate always fails.

Fallback. A build failure, image check, self-check miss or exception in any mode leaves `_WG_MODE`/`_WEXT` None, so the gate is False, `paired` is True, master FA3 runs, and the loud `PATH CHOSEN ... (fallback: ...)` line prints. Every mbarrier wait is bounded, and tickets complete even on abort. The residual risk is a device fault (illegal address), which would poison the in-process context. This is unchanged from s3. Compile: s3's 2 instantiations took 34.5 s. With a third, expect about 50 s plus 41 s for the decode module, well inside the 15-minute budget.

Fix:
1. `SYNC` variant (ticket_sync, A/B only): added `cp_wait_all(); fence_proxy_async(); __syncthreads();` right after the FULL wait. Reason: the only s3 hazard I could not rule out by reading is the ring's cross-thread cp.async (generic proxy) -> wgmma (async proxy) visibility on a REUSED stage. s3 fences that on the consumer side only. The end-of-tile `__syncthreads` alone would not separate that hypothesis from a warpgroup race. Now ticket_sync covers both with the canonical writer-side fence, so "ticket FAIL / ticket_sync PASS" names a pipeline/proxy fault and still leaves a working kernel chosen. The default `<*, false>` path is unchanged. Lint OK (host-sync warnings only, all outside kernel_fn), py_compile OK, __pycache__ removed.

Residual risk: if the fault is a stage/buffer layout error for n >= 1, all three modes fail and the run falls back to master FA3 with a DIAG-localized cause (neutral run, stage 5 needed).
