---
problem: muse_moe_w4a16
slug: post-norm-pdl-launch-small-m
created: 2026-10-24T183649Z
base: master
area: launch overhead
change: >
  num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel
  body is otherwise unchanged and every other launch is master's. Launch the post-norm with
  programmatic dependent launch so its launch latency and prologue overlap the tail of the w2
  Marlin grid inside the CUDA graph. (1) In _post_norm_combine_kernel add a constexpr USE_PDL;
  after the static loads that do not depend on Marlin (offs, mask, nw = norm weight + 1) and
  before the first load of ids/topk_weights/y, do 'if USE_PDL: tl.extra.cuda.gdc_wait()'.
  (topk_weights and local_ids are written before Marlin, but keep every data-dependent load after
  the wait for simplicity.) No gdc_launch_dependents call (nothing after it in the graph needs
  it). (2) Launch with launch_pdl=True (Triton launcher kwarg) when USE_PDL. (3) prepare(): set
  state['pdl_ok'] by a self-test: build a tiny y/ids/tw case, capture in a torch.cuda.CUDAGraph a
  plain elementwise kernel writing y followed by the PDL post-norm launch, replay 3 times, and
  compare with the USE_PDL=False launch (torch.equal). Any exception (kwarg unknown to this
  Triton, gdc_wait missing, capture error) or mismatch -> pdl_ok False. Also warm up/compile both
  variants for every num_tokens < 32 there. kernel_fn passes USE_PDL=state['pdl_ok']. Print one
  line '[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)' in prepare so the run log says which
  path was measured.
expected:
  - shapes: bs=1 *
    direction: faster
    why: removes ~1-2 us of launch gap + prologue between w2 Marlin and the post-norm, ~2-3% of a 49 us shape
  - shapes: bs=8 *
    direction: faster
    why: same absolute saving, ~1% of 108 us
  - shapes: bs=16 *
    direction: faster
    why: same, ~0.7-1% of 148 us
  - shapes: bs=32 *
    direction: flat
    why: untouched
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T183956Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T003236Z
base_sha: 1d31a26689bb109c514db2e2c51c7167dbbb76f0
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/post-norm-pdl-launch-small-m
worktree: .worktrees/muse_moe_w4a16/post-norm-pdl-launch-small-m
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T003435Z
impl_commit: a53a567af1e053d7c253efc4558d40021eef7390
submission_sha256: b36882b3db228eecd9823634c8ebb8fd331be66cbfc8b74ce0cda067d2a10c94
norm_sha256: c4785bcb2d2ed9891903ba93a91cf3e246ef2d7a38356d7083a55ad42e2169db
submission_blob: 062a12799f5d0eb53f65704727c665e6976fc6db
lint_warnings: 11
---

# post-norm-pdl-launch-small-m

## Change

num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel body is otherwise unchanged and every other launch is master's. Launch the post-norm with programmatic dependent launch so its launch latency and prologue overlap the tail of the w2 Marlin grid inside the CUDA graph. (1) In _post_norm_combine_kernel add a constexpr USE_PDL; after the static loads that do not depend on Marlin (offs, mask, nw = norm weight + 1) and before the first load of ids/topk_weights/y, do 'if USE_PDL: tl.extra.cuda.gdc_wait()'. (topk_weights and local_ids are written before Marlin, but keep every data-dependent load after the wait for simplicity.) No gdc_launch_dependents call (nothing after it in the graph needs it). (2) Launch with launch_pdl=True (Triton launcher kwarg) when USE_PDL. (3) prepare(): set state['pdl_ok'] by a self-test: build a tiny y/ids/tw case, capture in a torch.cuda.CUDAGraph a plain elementwise kernel writing y followed by the PDL post-norm launch, replay 3 times, and compare with the USE_PDL=False launch (torch.equal). Any exception (kwarg unknown to this Triton, gdc_wait missing, capture error) or mismatch -> pdl_ok False. Also warm up/compile both variants for every num_tokens < 32 there. kernel_fn passes USE_PDL=state['pdl_ok']. Print one line '[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)' in prepare so the run log says which path was measured.

## Expected

- `bs=1 *` faster: removes ~1-2 us of launch gap + prologue between w2 Marlin and the post-norm, ~2-3% of a 49 us shape
- `bs=8 *` faster: same absolute saving, ~1% of 108 us
- `bs=16 *` faster: same, ~0.7-1% of 148 us
- `bs=32 *` flat: untouched
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Mechanism parity with the running cuda-dispatch-combine-pdl-verify (PDL on its CUDA combine at >32), here on the Triton combine at <32: shape-disjoint. Marlin never triggers, so the implicit trigger at its block exits applies; the win is the launch/prologue overlap of the last w2 wave only. Must read the PATH CHOSEN line before judging: pdl off means the run measured master. Risk medium: Triton version on the judge (torch 2.14.1) should have launch_pdl and tl.extra.cuda.gdc_wait, but if not the fallback keeps it a master-equivalent run, not a failure. Kernel time in the profile will not move; judge on the shape median vs master's bs=1/8/16 rows. Overlaps post-norm-combine-hoisted-slot-loads (same kernel, different mechanism); keep them as separate runs for attribution, then restack. Replaced at bs=1 by the running bs1 fused rewrite if it lands.

## Adversarial review

Reviewed 2026-10-24T183956Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, rules-clean and shape-disjoint, with a safe fallback, so run it once with no companions. Expect at most a sub-microsecond gain at bs=1/8/16, because Marlin never triggers early and PDL here can only hide the dependent kernel's launch processing, not the tail of the w2 grid; judge it on the per-shape rows and the PATH CHOSEN line, not on the headline.

### Justification
Mechanism: launch_pdl plus gdc_wait (griddepcontrol.wait) on a Triton kernel captured into a CUDA graph becomes a programmatic edge, which is valid on H100 with CUDA 12.3+. In master's kernel_fn, the w2 moe_wna16_marlin_gemm is the node immediately before _post_norm_combine_kernel at num_tokens < 32: y.contiguous() is a no-op on the already contiguous view of cache3, and torch.empty for out launches nothing. So the edge exists. Size of the win: FINDINGS puts _post_norm_combine_kernel at 3.4/3.5/4.2 us at bs=1/8/16, against shape totals of 48.4/107.8/147.9 us (floors 11.4/37.3/52.5). Marlin is 87-93% of every shape, so the post-norm launch boundary is the only thing this touches. The hypothesis states the 1-2 us saving as fact, but it needs one of two things to be true: early triggering by the primary, or the inter-kernel gap in a graph replay being that large. WEAKEST LINK: 'overlap the tail of the w2 Marlin grid' is not what PDL does without a trigger. Marlin never calls griddepcontrol.launch_dependents, so its trigger is the implicit one at each block's exit. The dependent grid can be scheduled only after every Marlin block, last wave included, has exited. No post-norm block runs alongside the Marlin tail, and the pre-wait prologue (loading the 8 KB norm weight) hides almost nothing. What remains is the scheduling/launch latency between grid completion and the next node, typically about 0.3-1 us inside a graph. That is roughly 1-2% at bs=1, 0.5-1% at bs=8 and under 0.7% at bs=16, near or under per-shape noise. Headline: three of six loads gain at most about 1%, so the geomean moves about 0.3-0.5%, well inside the at-least-plus-or-minus-1.5% CI. A 'within-ci' headline is the most likely outcome and is uninformative by itself. The expected directions (faster at bs=1/8/16, flat at 32/128/256) agree with the code paths; only the magnitude is optimistic.

### Confounders on the current kernel
- post-expert-norm-triton-small-m: this change only exists on top of it (it adds the Triton kernel that gets the PDL attribute). Reverting it removes the subject rather than isolating an interaction, so it does not confound. Note that its kernel's prologue is tiny, so the 'prologue overlap' part of the claim is at most a few hundred ns.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify / marlin-block-m-two-smaller-on-postnorm: these change the align and the block_size_m pick. At bs<32 the restack's block_size_m pick decides how many w2 Marlin blocks run and how long the last wave is, which sets when the implicit trigger fires. That changes Marlin's own time, not the boundary saving, so it does not confound the A/B against master. The >=128 pieces are shape-disjoint.
- sglang-marlin-path-parity: sets up the Marlin -> post-norm sequence itself. No interaction beyond being the base.
- No landed hypothesis uses PDL. In-flight cuda-dispatch-combine-pdl-verify (PDL on the >32 combine) is shape-disjoint. post-norm-combine-hoisted-slot-loads and post-norm-triton-hoisted-decode32 edit the same kernel and must be restacked afterwards. cuda-fused-moe-bs1-persistent-w13-silu-w2 would remove the bs=1 boundary entirely if it lands.

### Interference and A/B needs
No companion is warranted. (i) ab-revert: there is no landed hypothesis whose removal would change how this result reads; the only one in the same area is this change's own substrate. (ii) ab-tune: the only knob that would change the mechanism is an early trigger in the primary (Marlin is a compiled sgl_kernel op that cannot be edited here) or a different pre-wait split. A 'gdc_wait at the very top' control would separate prologue overlap from launch-latency hiding, but both effects are sub-microsecond and a second run could not resolve the difference against noise. (iii) An A/A control of master to calibrate per-shape noise is impossible because identical code returns its cached result. Interpretation without companions: if the log says 'pdl off', the run measured master and says nothing about PDL. If 'pdl on', read the bs=1/8/16 per-shape paired ratios against master's rows (job 3cabc739f1af493e / BEST 45eacf4ca2e849d2) and treat bs=32/128/256 as the in-run control; if those drift as much as bs=1/8/16, the result is noise. A gain or loss inside the CI on the headline is expected and should not be read either way.

### Open questions
- Do the judge's Triton (with torch 2.14.1) and CUDA driver have both the launch_pdl kwarg and tl.extra.cuda.gdc_wait, and does torch's stream capture turn the PDL attribute into a programmatic graph edge? The PATH CHOSEN line answers the first. The second is only observable as a timing delta, or as a self-test mismatch if the edge silently becomes a full dependency (correct but no gain).
- Does moe_wna16_marlin_gemm launch anything after the Marlin kernel (for example a reduce or copy for use_fp32_reduce=True)? If it does, the PDL edge attaches to that tail kernel rather than to Marlin. That is still correct, but the 'w2 Marlin tail' framing does not apply. Confirm in the profile's kernel order at bs=1.
- Is the inter-kernel gap at the Marlin -> post-norm boundary in the bs=1 graph replay actually 1-2 us? An nsys trace of master's bs=1 graph would settle whether the expected magnitude is reachable before spending a check.


## Implementation notes

Only muse_moe_w4a16/submission.py changed (+68/-5).
- `_post_norm_combine_kernel` gains constexpr `USE_PDL` (default False); `tl.extra.cuda.gdc_wait()` sits after the norm-weight load and before the first ids/topk_weights/y load. No gdc_launch_dependents.
- New `_launch_post_norm(...)` wrapper passes `launch_pdl=True` only when use_pdl. kernel_fn (num_tokens < 32 only) calls it with `state["pdl_ok"]`; every other launch is master's.
- New `_pdl_self_test(device, post_norm_w)`, in-process, no child process: warms the PDL variant, captures a CUDAGraph of an elementwise `y.copy_` plus the PDL launch, replays 3 times, torch.equal vs the plain launch. Whole body in try/except Exception, returns (False, reason) on any error or mismatch, never raises.
- prepare() stores `pdl_ok`, compiles both variants for tokens 1..31 (only if the self-test passed; warmup also guarded), prints `[post-norm-pdl] PATH CHOSEN: pdl on|off (<reason>)`.
- Deviation: none. The self-test uses seeded synthetic data from torch.Generator, not any input.

## Rules check

- [x] CUDA-graph capturable: kernel_fn has no sync/.item()/prints; the PDL flag is a prepare-time bool, the branch is on shape only (checked in diff).
- [x] Nothing derived from `data` is cached across calls: state adds only pdl_ok (self-test on synthetic data) and the contiguous post_norm_w weight.
- [x] Weights: n/a: no weight repack change; post_norm_w is the same BF16 weight made contiguous in prepare().
- [x] Activations BF16 where SGLang keeps BF16; unchanged (n/a for KV/queries).
- [x] Attention: n/a: MoE problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: self-test is in-process, no subprocess/os/sys use.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: "OK with warnings" (only host-sync warnings in prepare()/self-check code, outside kernel_fn's call graph). py_compile passed; __pycache__ removed.
