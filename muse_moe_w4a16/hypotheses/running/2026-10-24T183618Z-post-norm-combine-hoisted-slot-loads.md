---
problem: muse_moe_w4a16
slug: post-norm-combine-hoisted-slot-loads
created: 2026-10-24T183618Z
base: master
area: post-expert norm+combine
change: >
  num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other
  kernel stay master byte for byte. Rewrite master's _post_norm_combine_kernel body so all
  local-slot row loads are issued before any reduction. Today each of the 8 slots sits under 'if
  eid >= 0:', so the compiler emits 8 serial (load ids -> branch -> load 8 KB row ->
  sum-of-squares) chains and the program pays ~8 DRAM round trips. New body: load the 8 ids at
  once (ids = tl.load(ids_ptr + t*TOP_K_C + tl.arange(0, TOP_K_C))) and the 8 weights g likewise;
  then in a tl.static_range(TOP_K_C) loop, extract eid_s and g_s with
  tl.sum(tl.where(tl.arange(0,TOP_K_C)==s, ids, 0)) (same for g), and load each row as v_s =
  tl.load(y_ptr + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no branch, so the 8
  loads are independent; then acc += tl.where(eid_s >= 0, g_s, 0.0) * v_s * nw *
  tl.rsqrt(tl.sum((v_s*nw)^2)/H + eps). Remote slots are still never read (masked load), so the
  uninitialised y rows of remote slots are never touched and a fully masked row contributes
  exactly 0 (g forced to 0, so rsqrt(eps) of a zero row times 0 is 0, no NaN). fp32 math, eps, nw
  = w + 1 in fp32, output bf16 and the launch (grid (num_tokens,), BLOCK_H = 4096) are unchanged
  except num_warps 8 -> 16 so each thread holds 8 elements per row (64 live fp32 values for 8
  rows). Keep the function name so the profile row stays comparable.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "_post_norm_combine_kernel 3.1 us -> ~1.6-2 us: one round trip for the <=8 local rows instead of 8 serial ones; ~2-3% of a 49 us shape"
  - shapes: bs=8 *
    direction: faster
    why: 3.8 us -> ~2-2.5 us, ~1-1.5% of 108 us
  - shapes: bs=16 *
    direction: faster
    why: 4.3 us -> ~2.5 us, ~1% of 148 us
  - shapes: bs=32 *
    direction: flat
    why: untouched (SGLang fused_post_expert_norm_weighted_sum)
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T183956Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T183956Z
base_sha: 91975540841cb14ebba858121c83ff760c528591
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/post-norm-combine-hoisted-slot-loads
worktree: .worktrees/muse_moe_w4a16/post-norm-combine-hoisted-slot-loads
mode: submit
implemented_at: 2026-10-24T184142Z
impl_commit: fcb99a3ef557c7d755cc8a578b65ed87e65ac699
submission_sha256: 0bc685de0533009d8117e84ffef50ad46fa5b252cecf0f891123087c21014466
norm_sha256: 756963f3e07935d7788af1319e6fc4aef0bb8c1acebc76e3f55059808ad09cc6
submission_blob: 1c0ea2b5746442d53faef544107710c2d27b689b
lint_warnings: 8
attempt_file: 2026-10-24T184142Z-submit-post-norm-combine-hoisted-slot-loads.md
prepared_at: 2026-10-24T184142Z
job: d7501477ce2748d5
dispatched_at: 2026-10-24T184157Z
---

# post-norm-combine-hoisted-slot-loads

## Change

num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other kernel stay master byte for byte. Rewrite master's _post_norm_combine_kernel body so all local-slot row loads are issued before any reduction. Today each of the 8 slots sits under 'if eid >= 0:', so the compiler emits 8 serial (load ids -> branch -> load 8 KB row -> sum-of-squares) chains and the program pays ~8 DRAM round trips. New body: load the 8 ids at once (ids = tl.load(ids_ptr + t*TOP_K_C + tl.arange(0, TOP_K_C))) and the 8 weights g likewise; then in a tl.static_range(TOP_K_C) loop, extract eid_s and g_s with tl.sum(tl.where(tl.arange(0,TOP_K_C)==s, ids, 0)) (same for g), and load each row as v_s = tl.load(y_ptr + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no branch, so the 8 loads are independent; then acc += tl.where(eid_s >= 0, g_s, 0.0) * v_s * nw * tl.rsqrt(tl.sum((v_s*nw)^2)/H + eps). Remote slots are still never read (masked load), so the uninitialised y rows of remote slots are never touched and a fully masked row contributes exactly 0 (g forced to 0, so rsqrt(eps) of a zero row times 0 is 0, no NaN). fp32 math, eps, nw = w + 1 in fp32, output bf16 and the launch (grid (num_tokens,), BLOCK_H = 4096) are unchanged except num_warps 8 -> 16 so each thread holds 8 elements per row (64 live fp32 values for 8 rows). Keep the function name so the profile row stays comparable.

## Expected

- `bs=1 *` faster: _post_norm_combine_kernel 3.1 us -> ~1.6-2 us: one round trip for the <=8 local rows instead of 8 serial ones; ~2-3% of a 49 us shape
- `bs=8 *` faster: 3.8 us -> ~2-2.5 us, ~1-1.5% of 108 us
- `bs=16 *` faster: 4.3 us -> ~2.5 us, ~1% of 148 us
- `bs=32 *` flat: untouched (SGLang fused_post_expert_norm_weighted_sum)
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Judge on the _post_norm_combine_kernel row in top_kernels (Triton times reproduce within ~1%), not the headline: +0.8-1.0% geomean is under one run's resolution. Overlap: cuda-fused-moe-bs1-persistent-w13-silu-w2 (running) replaces this kernel at bs=1 and the queued cuda-fused-moe-midm-single-launch-routed replaces it at 2..32; cuda-nvfp4-swapab-mma-persistent-small-m and marlin-two-stream keep it, so a gain stacks on them. Same mechanism as the running cuda-dispatch-combine-pdl-verify combine (issue all slot loads first) but on disjoint shapes (<32 vs >32). Risk: register pressure at 16 warps is low (64 fp32 per thread); a correctness slip would show as a failed test, not a slow run.

## Adversarial review

Reviewed 2026-10-24T183956Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible latency fix for a serially-chained, one-CTA-per-token kernel (3.4-4.2 us, 3-7% of the small-M shapes), but it only works if the 8 row loads are textually issued before the first tl.sum barrier; judge on the _post_norm_combine_kernel profile row, not the sub-CI headline.

### Justification
Mechanism is plausible: master's _post_norm_combine_kernel (submission.py:97-118) runs one program per token with 8 static_range iterations, each 'scalar id load -> data-dependent branch -> 8 KB row load -> block-wide tl.sum (shared-memory reduction with bar.sync)'. With grid=(num_tokens,) at bs=1/8/16, at most 16 SMs are busy, so the kernel is pure latency; FINDINGS shows 3.4 us (6.6% of bs=1), 3.5 us (3.1% of bs=8), 4.2 us (3.3% of bs=16), far above the ~0.5 us that moving <=1 MB of y would cost. Predicted rows (1.6-2.5 us) are consistent with ~8 serial L2/DRAM round trips plus 8 barriered reductions collapsing to one round trip; expected directions match TASK shapes (bs<32 faster, bs>=32 untouched; shapes branch on num_tokens only). Note y was just written by the w2 Marlin so the rows are mostly L2-resident: the per-slot latency is L2 (~0.3-0.5 us incl. reduction) not DRAM, so the saving is roughly as predicted but the floor of a tiny Triton kernel in a graph (~1.3-1.8 us) caps it; bs=1 expectation of 1.6 us is optimistic, ~2 us more likely. Headline effect ~+0.5-1% geomean, below the +-1.5% CI, as the hypothesis itself admits. WEAKEST LINK: the described body still puts the load and its tl.sum in the same static_range iteration. tl.sum over a 4096-wide block lowers to a cross-warp shared-memory reduction with barriers, and LLVM/ptxas will not move global loads across bar.sync, so 'load s; reduce s; load s+1; ...' stays serial even without the branch (removing the branch only removes the id->branch dependency, not the barrier ordering). The implementer must issue all 8 masked row loads in a first phase (or, cleaner, one 2D load of a [TOP_K_C, BLOCK_H] tile with mask (ids>=0)[:,None] & mask[None,:], then ss = tl.sum(v*v, axis=1), r = rsqrt(ss/H + eps), acc = tl.sum((g_masked*r)[:,None] * v, axis=0)), and only then reduce. The per-slot scalar extraction via tl.sum(tl.where(arange==s, ids, 0)) also adds 16 small reductions; a 2D tile avoids them entirely. Numerics: the zero-row argument (g forced to 0, rsqrt(1e-8)=1e4 times 0) is correct; fp32 math and bf16 output unchanged.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (192f225d8f): this is the kernel being rewritten, so it is the base, not a confounder; the change is a strict in-kernel edit with the same launch site, inputs and SMALL_M_POST_NORM gate.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: _align_small_kernel produces local_ids that this kernel reads, but neither its layout nor timing changes; the align kernel precedes w13/w2 Marlin so there is no overlap interaction. Not a confounder.
- marlin-block-m-two-smaller-on-postnorm and align-parallel-per-expert-verify: act only at num_tokens >= 128 / > 32, disjoint from the bs<32 shapes touched here. Not confounders.
- sglang-marlin-path-parity: defines y's layout ([M, 8, H] with uninitialised remote rows), which the masked-load design respects. Not a confounder.

### Interference and A/B needs
No ab-revert is meaningful: the only overlapping landed item is the one that created this kernel, and reverting it removes the code being changed. The hypothesis bundles two sub-changes (load hoisting and num_warps 8 -> 16). A gain is attributable enough to land as a package and is read directly off the kernel row. A loss would be ambiguous between 'hoisting not realised (loads still barrier-ordered)' and '16 warps make each cross-warp reduction more expensive', but this can be diagnosed from the same run by inspecting the compiled kernel (TTGIR/PTX order of ld.global vs bar.sync, which the implementer should dump locally before checking) and does not justify spending a check on a control for a ~1% geomean item. No companions.

### Open questions
- Will the implementer structure the body as all-loads-then-reductions (preferably a single 2D [8, 4096] masked tile load with axis=1 sum-of-squares) rather than load+reduce per iteration? If the latter, expect little or no change.
- Is num_warps=16 better than 8 for the 2D-tile form? 8 warps gives 128 fp32 live values per thread for the tile (still under the register limit) with fewer warps in each barriered reduction; worth a local microbench before checking.
- Overlaps the same kernel and shapes as queued post-norm-pdl-launch-small-m; whichever lands second should be restacked on the first rather than both run against master (their gains should add: one hides launch/prologue, the other the body's latency).
- If cuda-fused-moe-bs1-persistent-w13-silu-w2 or cuda-fused-moe-midm-single-launch-routed land, this kernel disappears at bs=1 or bs<=32, making this gain moot on those shapes.


## Implementation notes

Edited only `_post_norm_combine_kernel` body and its launch `num_warps` 8 -> 16 in the worktree submission.py. Deviation from the md's per-slot static_range form, following the adversarial review: a single [TOP_K_C, BLOCK_H] masked 2D tile load (mask = (ids>=0)[:,None] & mask[None,:]) issues all 8 row loads textually before any reduction; then ss = tl.sum(v*v, axis=1), r = rsqrt(ss/H+eps), acc = tl.sum((where(valid,g,0)*r)[:,None]*v, axis=0). This also removes the 16 scalar-extraction reductions. Remote rows stay masked (never read); fully masked row gives coef 0 so no NaN. fp32 math, nw = w+1, eps, bf16 output, grid and BLOCK_H unchanged. Not run locally (no GPU).

## Rules check

- [x] CUDA-graph capturable: diff touches only a Triton kernel body and a num_warps constant; no host syncs or input-dependent control flow.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: untouched (prepare() not in diff).
- [x] Activations BF16 where SGLang keeps BF16: unchanged; fp32 internal math, bf16 output as before.
- [x] Attention: n/a, not touched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune added; first-line marker kept.

Lint (okc_lint.py): `OK with warnings` — 8 host-sync warnings, all in prepare() outside kernel_fn's call graph, pre-existing, none in the diff. py_compile clean, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only; no GPU). **Verdict: dispatch** (no fixes).

- Spec fidelity: the single masked [TOP_K_C=8, BLOCK_H=4096] tile load replaces the per-slot static_range form; it realises the `change` paragraph's goal (all local-slot loads issued before any reduction, remote rows never read, zero contribution for masked rows) and is the form the adversarial review recommended. num_warps 8 -> 16 is named in the spec. Diff touches only this kernel body and its launch; bs>=32 path untouched.
- Correctness: H = 4096 on every TASK shape, so BLOCK_H = H and the H-mask is all-true; masked slots load 0, r = rsqrt(eps) is finite, coef = 0, no NaN. ids int32 / topk_weights fp32 contiguous [M, 8]; row offsets int64. Only fp32 summation order changes vs master.
- Registers at 16 warps: default blocked layout is sizePerThread [1,8] (16 B bf16 vector), threadsPerWarp [1,32], warpsPerCTA [1,16], so each thread holds all 8 rows x 8 cols = 64 fp32 of v (live across the axis=1 reduction), 8 of nw, ~16 for 8 row addresses: ~90-110 regs against the 128/thread cap a 512-thread CTA imposes. No spill expected. The axis=0 combine is intra-thread (no shuffle); the axis=1 sum-of-squares is one cross-warp smem reduction for all 8 rows. 8 warps would also fit (128 live, cap 255) but is not needed; grid <= 16 CTAs, occupancy irrelevant.
- Rules: no gates, no host syncs, no allocation or input-dependent host control flow added; graph-capturable; weights/activations unchanged.

Residual risk: if ptxas lands just above 128 regs at 16 warps a small spill could eat the ~1 us gain; read the _post_norm_combine_kernel row, not the headline.
