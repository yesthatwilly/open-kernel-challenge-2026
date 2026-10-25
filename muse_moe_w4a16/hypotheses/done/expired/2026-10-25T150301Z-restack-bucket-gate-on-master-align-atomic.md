---
problem: muse_moe_w4a16
slug: restack-bucket-gate-on-master-align-atomic
created: 2026-10-25T150301Z
base: master
area: expert GEMM path selection (wgmma vs Marlin) restack
change: >
  Restack only, no new idea. Bring moe master's submission.py up to the code that is on the board:
  cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, impl fc959d9c; check
  7be125e9e1ea42e3 1.3744x, scored 4e4a0e26c85944f2 1.3727x) was built on base 928151c and
  predates master's two promotions align-parallel-per-expert-verify (1a5da3d2) and
  w2-marlin-atomic-add-small-m (9080a71e). Claim restacks it onto current master by 3-way applying
  the parent's patch; if that conflicts, hand-merge so the result keeps BOTH: (1) master's
  parallel per-expert align for verify shapes and master's w2 Marlin atomic-add flag at small M,
  exactly as on master; (2) the bucket-gate's whole wgmma path (per-call cuTensorMapEncodeTiled,
  three-state copy check, prepare-time selector _wg_select with state flags wg_on_128 / wg_on_256,
  kernel_fn branch on T and those flags only). Where the two touch the same function (the Marlin
  path the selector times as _moe_marlin), the selector must time master's current Marlin path,
  including the atomic-add flag and the new align, so it compares against what master actually
  runs. No other change. Keep the first-line marker. Print the PATH CHOSEN B128 / B256 lines as
  the parent did.
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "B256 wgmma path (judge data: 402 vs 457 us Marlin) on top of master's align/atomic-add"
  - shapes: load=decode_*
    direction: flat
    why: "T<128 stays on master's Marlin path"
risk: low
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
priority: 1
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T150511Z
review_companions: null
hold_reason: "adversarial review: The premise is false: the parent's submission.py (blob 1774e42c, branch tip = impl fc959d9c) already contains master's align-parallel-per-expert-verify and w2-m"
outcome: expired
verdict_detail: "retired: stale hold closed per the 2026-10-25 moe diagnosis"
expired_at: 2026-10-25T212024Z
---

# restack-bucket-gate-on-master-align-atomic

## Change

Restack only, no new idea. Bring moe master's submission.py up to the code that is on the board: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, impl fc959d9c; check 7be125e9e1ea42e3 1.3744x, scored 4e4a0e26c85944f2 1.3727x) was built on base 928151c and predates master's two promotions align-parallel-per-expert-verify (1a5da3d2) and w2-marlin-atomic-add-small-m (9080a71e). Claim restacks it onto current master by 3-way applying the parent's patch; if that conflicts, hand-merge so the result keeps BOTH: (1) master's parallel per-expert align for verify shapes and master's w2 Marlin atomic-add flag at small M, exactly as on master; (2) the bucket-gate's whole wgmma path (per-call cuTensorMapEncodeTiled, three-state copy check, prepare-time selector _wg_select with state flags wg_on_128 / wg_on_256, kernel_fn branch on T and those flags only). Where the two touch the same function (the Marlin path the selector times as _moe_marlin), the selector must time master's current Marlin path, including the atomic-add flag and the new align, so it compares against what master actually runs. No other change. Keep the first-line marker. Print the PATH CHOSEN B128 / B256 lines as the parent did.

## Expected

- `load=verify_32*` faster: B256 wgmma path (judge data: 402 vs 457 us Marlin) on top of master's align/atomic-add
- `load=decode_*` flat: T<128 stays on master's Marlin path

## Risk

Goal: master == board code. The parent scored 1.3727 on a stale base; this should score at or above it, since master's own promotions were +0.03 over the parent's base.

## Adversarial review

Reviewed 2026-10-25T150511Z by an adversarial subagent (model opus).

**Verdict: hold.** The premise is false: the parent's submission.py (blob 1774e42c, branch tip = impl fc959d9c) already contains master's align-parallel-per-expert-verify and w2-marlin-atomic-add-small-m code verbatim, so the restack's correct output is byte-identical to the code that already scored 1.3727x in 4e4a0e26c85944f2. Running it would spend a scored submit to re-measure identical code. Promote the parent's blob to master directly instead, with no run.

### Justification
A. Content, not git ancestry, decides this. Git ancestry does say the parent sits on 928151c (s4), which is not a master descendant. But the s4/s5 lineage was rebased by content onto master 9080a71 (FINDINGS says 'same s4 base rebased onto master 9080a71'), and 928151c's file already has _align_parallel_kernel and _w2_atomic_self_check. Line-diffing master's submission.py (blob 60dfe35d) against the parent's (1774e42c), only 5 master lines are missing from the parent, and all 5 come from moving master's _forward body into _moe_marlin(x, topk_ids, topk_weights, bank, state, w2_mode) (data['topk_ids'] -> topk_ids, return out -> return out, local_ids, cache2, y; kernel_fn now branches to _moe_wgmma or _moe_marlin). _moe_marlin is master's _forward launch for launch: _align_small for T<=32, _align_parallel above, two-smaller block_size_m at T>=128, the w2 atomic-add/atomic_zero path at T<=W2_ATOMIC_MAX_TOKENS=32 driven by state['w2_mode'] from the same _w2_atomic_self_check, and the same post-norm split. The change paragraph's sub-requirement is also already met. _wg_select times _moe_marlin(..., 'master'), which runs master's new align at T>=128. w2_mode='master' there is exact, because the atomic-add flag only applies at T<=32 and the selector buckets are T>=128. So 'master's current Marlin path' is what the selector already compares against. A 3-way git merge-file with base 928151c conflicts only on hunks where master lacks the wgmma additions, and resolving every one toward the parent gives the parent's file. The expected rows are also wrong: verify_32 'faster' was the parent's gain over the old master. Against the board (the parent's own score) the honest expectation is flat on every shape, with only run-to-run noise around 1.3727 [1.352,1.393]. Weakest link: the stated mechanism ('this should score at or above it, since master's own promotions were +0.03 over the parent's base') assumes the parent lacked those promotions. It did not; the +0.03 is already inside 1.3727.

### Confounders on the current kernel
- align-parallel-per-expert-verify: already present verbatim in the parent (_align_parallel_kernel, _align_parallel, the extended _align_self_check, the warm-up), and used by both _moe_marlin and the _wg_select Marlin timing. Nothing to add, so nothing to confound.
- w2-marlin-atomic-add-small-m: already present verbatim (W2_ATOMIC_MAX_TOKENS, _w2_atomic_self_check, the w2_mode plumbing in _moe_marlin). It applies only at T<=32, which the wgmma path (T>=128) never touches. No interaction.

### Interference and A/B needs
No companion. The run itself is uninformative: identical code (same blob as scored job 4e4a0e26c85944f2) can only reproduce 1.3727 within noise. A companion would cost another scored run to learn nothing more. The project convention says identical code is never re-run, and that applies directly here.

### Open questions
- Can churn promote the parent's blob directly (copy hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate:muse_moe_w4a16/submission.py, blob 1774e42c, onto master as a promote commit citing scored job 4e4a0e26c85944f2) without a run? That achieves this hypothesis's whole goal (master == board code) for zero checks and zero submits.
- If policy requires a run before every promotion, is a re-measure of an already-scored blob allowed, or should the promotion be recorded against 4e4a0e26c85944f2?
- Why was the parent's outcome filed as neutral/unproven-vs-best and not promoted, given it scored 1.3727 above the 1.3439 board? The promotion gate may be comparing against something other than the board.

