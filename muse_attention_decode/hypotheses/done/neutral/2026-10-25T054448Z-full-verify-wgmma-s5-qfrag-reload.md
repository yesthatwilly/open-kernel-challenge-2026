---
problem: muse_attention_decode
slug: full-verify-wgmma-s5-qfrag-reload
created: 2026-10-25T054448Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, multi-tile Q-fragment fix)
change: >
  Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job
  51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1-2t with no request
  crossing) is wrong with all 128 rows' running max bad; 1-tile segments pass at any ring stage;
  the fully synchronised variant fails identically, ruling out ring/mbarrier races. The only
  per-segment state not reloaded before the S wgmma of tile lt>=1 is the register-sourced Q
  A-fragment qa[2][4], loaded only at n == 0 || lt == 0 and passed as the {a regs} operand across
  iterations. Fix (a): reload qa from smem/L2 unconditionally before EVERY tile's S wgmma (2 KB),
  with wg_fence/fence_acc ordering around it as for the first tile; keep the rest of s4. Add the
  DIAG from the read (c): for B1-2t, dump S (pre-max) of tile 1 and compare with the fp32
  reference S, printing max abs err, so 'S wrong' vs 'carry wrong' separates in this one check. If
  (a) fixes S but rows still fail, print per-tile mA/lA and oacc error to localise the carry
  (ex2(mA - uA) with Max_offset folded). Drop the SYNC variant. Keep FP8 P with FA3's Max_offset
  (P = exp2(s - (max - 8))) and test on COHERENT long-context data, not random keys (LESSONS).
  Gate: after the full s4 bisect table passes (every multi-tile case, kv 0.25, grid 3, device
  grid, lens_a/lens_b, coherent), time [wgmma ticket], [wgmma b8] vs master paired FA3 path
  (prepare + main + combine + paired norm-gate) at B=16 and B=32 by interleaved graph replay;
  route a band only at >= 1.05x. Print regs/local per variant (refuse a spill, LESSONS prefill
  wgmma). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: incumbent 162.8 us shape (FA3 main 151.6 + combine 6.0 + prepare 3.0); floor 63.4 us; M1 = self-check passes and the kernel is timed, M2 <= 150 us
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: incumbent 302.2 us (FA3 288.0 + combine 7.5); floor 126.8 us; M2 <= 280 us
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: high
depends_on: full-verify-wgmma-s4-multitile-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)"
review: proceed
status: done
review_at: 2026-10-25T054737Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054738Z
base_sha: 3ee2b893476879190a690392a3295a35afed11bf
base_blob: 9bcf522b0dd23a0a8173046d2150473827122206
branch: hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload
worktree: .worktrees/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T084811Z
impl_commit: 69378d744597bc703edd03a127b4b8e260d449d4
submission_sha256: 13bf4cd9d89b5ecfae6ce27e2cea9a3c83aee5824b64c7dd14c003f84621b16f
norm_sha256: 12e83fd1688e4d04e45ce6a14bd8eceb6903de9f1835c9ebeb09ab2b0ffe1653
submission_blob: 8e5ec0c7813b95db7011d9fe859f3835673f935c
lint_warnings: 35
attempt_file: 2026-10-25T084811Z-check-full-verify-wgmma-s5-qfrag-reload.md
prepared_at: 2026-10-25T084811Z
job: 71e40388a1904937
dispatched_at: 2026-10-25T084834Z
outcome: neutral
verdict_detail: beats-base-not-best
speedup: 1.4345479685047922
speedup_ci: [1.4130297489772203, 1.456066188032364]
harvested_at: 2026-10-25T091104Z
best_at_harvest: 1.444132502888963
promotion: null
---

# full-verify-wgmma-s5-qfrag-reload

## Change

Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job 51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1-2t with no request crossing) is wrong with all 128 rows' running max bad; 1-tile segments pass at any ring stage; the fully synchronised variant fails identically, ruling out ring/mbarrier races. The only per-segment state not reloaded before the S wgmma of tile lt>=1 is the register-sourced Q A-fragment qa[2][4], loaded only at n == 0 || lt == 0 and passed as the {a regs} operand across iterations. Fix (a): reload qa from smem/L2 unconditionally before EVERY tile's S wgmma (2 KB), with wg_fence/fence_acc ordering around it as for the first tile; keep the rest of s4. Add the DIAG from the read (c): for B1-2t, dump S (pre-max) of tile 1 and compare with the fp32 reference S, printing max abs err, so 'S wrong' vs 'carry wrong' separates in this one check. If (a) fixes S but rows still fail, print per-tile mA/lA and oacc error to localise the carry (ex2(mA - uA) with Max_offset folded). Drop the SYNC variant. Keep FP8 P with FA3's Max_offset (P = exp2(s - (max - 8))) and test on COHERENT long-context data, not random keys (LESSONS). Gate: after the full s4 bisect table passes (every multi-tile case, kv 0.25, grid 3, device grid, lens_a/lens_b, coherent), time [wgmma ticket], [wgmma b8] vs master paired FA3 path (prepare + main + combine + paired norm-gate) at B=16 and B=32 by interleaved graph replay; route a band only at >= 1.05x. Print regs/local per variant (refuse a spill, LESSONS prefill wgmma). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_16 *` faster: incumbent 162.8 us shape (FA3 main 151.6 + combine 6.0 + prepare 3.0); floor 63.4 us; M1 = self-check passes and the kernel is timed, M2 <= 150 us
- `*layer=full load=verify_32 *` faster: incumbent 302.2 us (FA3 288.0 + combine 7.5); floor 126.8 us; M2 <= 280 us
- `*layer=sliding *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Even a fallback run is informative: the DIAG decides between the Q-fragment and the carry. First timed measurement of this kernel; okc check only.

## Milestones

Rewrite stage 5 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054737Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-5 correctness bisect step for the wgmma verify rewrite. Read against the s4 kernel, the Q A-fragment is the only per-segment input to tile lt>=1's S wgmma that differs from tile 0, and the S-vs-carry DIAG makes even a fallback run informative. Proceed with no companions, but fix how the reload is placed, and the lineage, before building.

### Justification
Mechanism (A). In s4's splitkv_verify_wgmma_kernel<TICKET,SYNC>, qa[2][4] is loaded only under `if (n == 0 || lt == 0)`, and the S wgmmas take it as the register A operand. The S accumulator is reset by scale_d=0 every tile. 1-tile segments pass at any ring stage, so the K descriptor/stage path and the V^T double buffer (n&1, which multi-request 1-tile runs also exercise) are already cleared, and SYNC rules out ring races. Between tile lt==0 and lt>=1 that leaves three differences: the qa reuse, the carried mA/lA/oacc, and the lt-dependent edge mask (lim). The rows go bad on all 128 rows, including unmasked ones, which points at S (qa) or the carry rather than the mask. The DIAG (tile-1 S vs the fp32 reference) separates those two in one check, so the run is interpretable whatever happens. Floors: verify_16 incumbent 162.8 us (FA3 main 151.6) against a 63.4 us floor, and verify_32 302.2 us (288.0) against 126.8 us. M2 (<=150 / <=280 us) only needs about 8% off the FA3 main kernel, a modest and physically reachable target for a single-pass FP8 wgmma kernel at ~2.4x above floor. Sliding and decode shapes are untouched, so flat is right. Weakest link (performance, not correctness): an unconditional per-tile reload 'from smem/L2' done as __ldg just before the S wgmma puts one L2 round trip (~0.5 us) on the critical path of EVERY tile. At ctx 80K, B=16 that is ~20K tiles over a 2*SM grid, ~75 tiles per CTA, i.e. tens of us exposed. That could erase the M2 margin and make the >=1.05x gate refuse a correct kernel. Stage Q in shared memory once per segment (ld.shared per tile), or issue the qa reload before wait_bar(full_bar(st)) so it overlaps the barrier wait. Root-causing also matters: a C++ register array held across iterations should not go stale. Dump SASS (cuobjdump -sass) and the ptxas wgmma-serialization warnings for qa's registers, so the fix is understood and not just worked around.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (master c3c1b01): the s4 branch lacks it (git merge-base confirms; s4's base table shows full decode_1 at 27.8 us vs master's 22.6 us). If s5 is built on s4 as written, the headline and full decode_1 will look worse than BEST from lineage alone, not from the verify change. The verify path is disjoint from the decode stream-K kernels, so rebase s5 onto master (frontmatter already says base: master) or judge strictly on the verify shapes' top_kernels.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup (+ ab-tune-batch-gate): these define the incumbent being replaced at full verify_16/32 (paired FA3 + prepare + combine + paired norm-gate). The interleaved graph-replay gate must time exactly master's paired path, including _paired_norm_gate_kernel, or the >=1.05x routing decision is mis-anchored. That is not a confounder of the measured kernel's own time.
- fused-preattention-triton: it writes the FP8 q (paired layout on full batched layers) that the new kernel reads via the qra/qrb addressing. If s5 sits on master, the q layout the kernel assumes (head-major, per-token HQ*D, unpadded vs paired-padded) must match what master's pre-attention kernel emits for the routed band. The self-check catches a mismatch, but it would show up as a fallback and be misread as the qfrag fix failing.

### Interference and A/B needs
No companions. This is a rewrite stage whose deliverable is the DIAG plus the self-check table; the next stage is the follow-up. A gain or loss on the headline is not the readout: verify_16/32 top_kernels against 151.6/288.0 us and the PATH CHOSEN lines are. The single-run ambiguities (S wrong vs carry wrong; correct-but-slower vs fallback) are already resolved inside the run by the DIAG and the printed paired timings. Attention has 8 checks left for 8 queued hypotheses, so an ab-revert or control would cost a check without adding information. The one lineage issue (s4 lacks c3c1b01) is fixed by rebasing, not by a companion run.

### Open questions
- Did s4's bisect include a 1-tile segment that STARTS mid-request (n == 0, lt >= 1, the stream-K tail case)? If it passed, the lt-based edge mask is cleared and qa is implicated even before s5's DIAG. If it was never tested, add it to the table.
- Why would qa go stale at all? Does the SASS show ptxas reusing or rematerialising qa's registers across the loop (e.g. under register pressure with sacc/oacc/pa at 256 threads), and are there 'wgmma serialized' ptxas warnings? A real register-clobber root cause also threatens pa and the oacc carry.
- After the fix, what are regs/thread and local bytes per variant? Reloading through smem keeps register pressure flat; an L2 __ldg reload before every tile adds latency on the critical path.
- Will s5 be rebased onto master (c3c1b01) so that a promotable result carries master's full decode_1 stream-K kernel?


## Implementation notes

Worktree `.worktrees/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload`, `muse_attention_decode/submission.py` only.

**Rebase onto master (stale_base).** The branch was s3+s4 on 27b43dc; master was 52ca902. `git merge-file` conflicted in 11 hunks (s3 also edited the decode `splits`/sliding predicate that master has since superseded), so I took master's file wholesale and re-applied only the verify-path additions of s4: the `muse_full_verify_wgmma` module block (inserted after `_full_decode_splitkv`), the `_verify_synthetic` ... `_wg_verify_setup` block (before the final `if torch.cuda.is_available():`), the `splitkv_verify` gate/dispatch in `kernel_fn`, one `_wg_verify_setup()` call at the end of the import-time block, and the docstring sentence. Everything outside the verify path (decode stream-K, two-level ticket c3c1b01, sliding fused path, restacks) is byte-identical to master; `git diff master` shows 2 deleted lines (docstring, `paired` predicate), the rest additions.

**The change (stage 5).**
- qa[2][4] is staged in shared memory ONCE per segment (each thread writes its own 8 A-fragment words at `n == 0 || lt == 0`, word-major `u32[8][256]`, 8 KB, `QS_OFF`; SMEM_A grows by 8 KB, 2 CTAs/SM still fit) and RELOADED into registers before EVERY tile via volatile `ld.shared` (8 conflict-free loads), issued before `wait_bar(full_bar(st))` so they overlap the barrier wait. The existing `fence_acc` + `wg_fence` sit between the reload and the S wgmma. (Review: not an L2 `__ldg` on the critical path; applied.)
- SYNC variant dropped (template param, `use_sync` arg, `ticket_sync` mode removed). Its slot is now `DBG` (template bool, a separate instantiation used only by the prepare DIAG; the timed kernels are unchanged apart from the qa reload).
- DIAG (prepare only, after case B1-2t, ticket mode): a second launch of the DBG kernel dumps CTA 0's tile-1 scaled S (pre edge-mask, pre-max) and running mA/mB/lA/lB after tiles 0..3; the host compares S with the fp32 reference (max abs err, entries off, verdict "S correct / S WRONG") and prints per-tile max |mA - ref| and l relative error (quad-summed lane partials), which localises the carry if S is fine. Never launched from the timed path (`_WG_DBG` is None, `use_dbg` 0). Per-tile oacc error is not dumped (per-segment O error from `_wg_diagnose` remains).
- New bisect cases: `mid-1t-B1-3t-grid3` (lens [192], grid 3: CTAs 1 and 2 each own one tile that starts mid-request, n == 0, lt >= 1) and `mid-1t-2req-grid3` (lens [128,128], grid 3: CTA 1's single tile is lt = 1 of request 0, CTA 2 crosses into request 1). `_wg_verdict` names them.
- regs/local per variant: `_wg_image_check` prints regs, local bytes, binary/ptx version, CTAs/SM for `<ticket>` and `<B8>` and the combine kernel (unchanged, still prints; a local-bytes > 0 spill is visible in the log). Module renamed `muse_full_verify_wgmma_v3` (fresh build cache).
- Two-state graph check (brief requirement): `_wg_two_state_check` prepares two states, warms and captures one graph per state, then replays A B A B B A with `_wg_refill` (qkv, k_cache, v_cache refilled in place) and compares each replay with the fp32 reference of its own state. No kernel static or module global holds a state pointer; the only module-level tensors are the pre-existing self-resetting int32 tickets and the error word.
- Routing gate (spec: >= 1.05x): the old eager, non-gating `_wg_time` is replaced by `_wg_time_graph`: wgmma path vs master paired FA3 path (pre-attention + FA3 + paired norm-gate) at B=16 and B=32, each in its own CUDA graph, interleaved replays (alternating order), 256 MB L2 flush before each, empty-graph baseline subtracted, medians of 40. Band routed only if master/wgmma >= 1.05; `_splitkv_verify_gate` consults `_WG_BANDS[16 if batch <= 24 else 32]` (config only). PATH CHOSEN lines per band print both numbers (TIMING line) and the decision; if neither band wins the whole path falls back to master.

**Review items.** (1) smem staging / reload before the barrier wait: done. (2) DIAG tile-1 S vs fp32 only in prepare: done. (3) 1-tile mid-request segment in the bisect table: done (two cases). (4) regs/local printed per variant: done. (5) Rebase onto master incl. c3c1b01: done (see above); master paired path times as the incumbent. (6) cuobjdump / any subprocess: NOT added (no child processes). Open question left unanswered for the user, since SASS cannot be dumped in-process: does `cuobjdump -sass` of splitkv_verify_wgmma_kernel<true,false> show ptxas re-using or rematerialising qa's registers across the loop, or "wgmma serialized" warnings (the true root cause of the stage-4 failure, which also threatens pa / oacc)? The smem reload makes the result independent of that, but does not explain it. (7) Open question "did s4's bisect include a 1-tile mid-request segment": unknown, hence the new cases. (8) Open question "regs/thread after the fix": printed at run time.

**Deviations.** Band mapping (<= 24 / > 24) is my choice. FP8 P / Max_offset, coherent cases, kv 0.25 and grid cases are kept from s4 unchanged. Not built or run (no GPU): the biggest risk is a compile error in the new CUDA (DBG template, ld/st.shared asm) or the extra 8 KB smem dropping occupancy below 2 CTAs/SM (prepare prints CTAs/SM and warns).

## Rules check

- [x] CUDA-graph capturable: kernel_fn path adds no sync/.item()/.cpu()/print; all new syncs/prints are in prepare (self-check, DIAG, timing); lint only warns host-sync outside kernel_fn's call graph.
- [x] Nothing derived from `data` is cached across calls: workspaces are allocated per call from shapes; tickets are self-resetting zero arrays; DIAG/refill buffers are prepare-local; `_WG_BANDS` is a config-only routing decision from synthetic data, not from judge inputs.
- [x] Weights: n/a: attention kernel, no weights.
- [x] Activations/KV: KV cache FP8, q E4M3 at most (FP8 query), fp32 softmax and merges, bf16 out; unchanged from s4.
- [x] Attention sees every key: every 64-token tile of each window is attended; causal mask only (t <= seq - 8 + j).
- [x] No timing patches / harness access / reference import: the prepare timing uses CUDA events on synthetic data only and gates routing on a printed margin; no child processes, no cuobjdump.
- [x] All work inside the timed call, no output caching or replay: yes.
- [x] Judge two states: no device pointer, TMA map or state-sized buffer in any module/static global; two-state alternating graph self-check added.

okc_lint output:
```
OK with warnings: muse_attention_decode/submission.py
(35 warnings, all host-sync in prepare-only code outside kernel_fn's call graph, plus one dtype warning for the uint8 256 MB L2-flush buffer in prepare)
```
py_compile: ok; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (model opus). **Verdict: dispatch-with-fixes.** Rules ok.

What was checked:
- **Rebase.** A `diff` against master (9f004955) shows only additions, plus the docstring sentence and the `paired` predicate (`and not splitkv_verify`). These parts are byte-identical to master: decode stream-K, the two-level ticket (c3c1b01), sliding and the restacks.
- **Two states.** No module or static global holds a state pointer, a TMA map or a state-sized buffer. The only ones are the self-resetting int32 tickets, the int32[1] error word and a C++ `static bool`. The self-check does the two-state alternating graph replay (A B A B B A, with in-place refills).
- **Fallback.** Every miss (build, image check, a case, two-state, DIAG, timing) lands on master's paired FA3 with a PATH CHOSEN line, without raising.
- **kernel_fn.** The verify path has no sync, prints or try/except, and allocates or branches on shapes and config only. The hook, KEEP and DBG globals are None when timed.
- **sm90a guard.** All wgmma and mbarrier code is under `__CUDA_ARCH_FEAT_SM90_ALL`, with an sm_90a-only gencode.
- **Waits.** Every mbarrier wait is bounded by clock64, and the named barriers are reached uniformly.
- **Budgets.** Smem is 99,936 B per CTA, so 2 CTAs/SM fit. Each module compiles in about 35-45 s. s4 fit in 15 min with three self-check modes, and s5 runs two.
- **Gate.** The 3e-2 self-check gate can pass: s4's passing cases measured 5e-3 to 9e-3.

Fixes, both in `_wg_time_graph`:
1. **Timing shape.** The routing timing ran at lengths of about 2-6K tokens, not the scored ctx 80000 (agentic, +/-7.5%). The >= 1.05x band decision was therefore made on a shape the judge never scores. It now uses lengths of 74K-86K.
2. **Agreement check.** The self-check's longest request is 9,000 tokens, but the band-16 route also carries the test `batch 2, ctx 100000, q_len 8`. A bug that only shows at long context would fail the run instead of falling back. A band is now routed only if the wgmma output, on the 80K timing data, also matches master's paired-FA3 output: rel L2 < 3e-2, finite, and the error word clear. The TIMING line prints all of this.

Residual risk: the CUDA has never been built (DBG template, ld/st.shared asm), and if the multi-tile bug is not the Q fragment, every band falls back again; the DIAG localises it either way.

## Result

- job `71e40388a1904937` (check), status done, passed True, tests 16/16
- speedup 1.4345x, 95% CI [1.4130, 1.4561], score 2625.426324311454 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-verify-wgmma-s4-multitile-fix.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-wgmma-s5-qfrag-reload.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-wgmma-s5-qfrag-reload.patch`; branch `hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.27 | 13.23 | 0.9915 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.06 | 22.76 | 1.1826 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.99 | 15.47 | 1.0292 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 122.76 | 111.02 | 1.1047 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.71 | 24.35 | 1.0173 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 258.18 | 229.95 | 1.1299 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.14 | 17.21 | 0.9983 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.35 | 30.13 | 1.0011 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.86 | 22.83 | 1.0096 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.63 | 162.75 | 1.0047 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.03 | 29.09 | 1.0096 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 301.66 | 301.13 | 1.0053 | flat |

gained on full decode_1, decode_16, decode_32. Expected hit: 1/4. Noise ±1.5%.

## Follow-ups

- rewrite stage 5 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-verify-wgmma-s5-qfrag-reload --stage 6`; a slow or failed first cut is the expected start, not a do-not-retry.
- beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-verify-wgmma-s5-qfrag-reload`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

