---
problem: muse_bf16_proj
slug: router-splitk32-small-m-all-loads-in-flight
created: 2026-10-24T075025Z
base: master
area: router GEMM
change: >
  Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the
  num_stages they launch with; the kernel body, the torch.zeros output + relaxed fp32 atomic_add
  epilogue, BLOCK_K=128, num_warps=4, and the m > 32 buckets (64,64,16) / (64,64,8) with
  num_stages=3 stay exactly as master. New small-m buckets: m <= 16: BLOCK_M=16, BLOCK_N=32,
  SPLIT_K=32; 17..32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=32. Grid becomes (1, 8, 32) = 256 CTAs (was
  128), K-slice 256 = 2 K-steps of BLOCK_K=128 (was 4), and these buckets launch with num_stages=2
  so both K-steps' hi, lo and x tiles are issued before the first dot (per CTA ~40 KB smem, so ~2
  CTAs per SM are co-resident and all 256 run in one wave). Implement by having _router_config
  return (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES) and passing num_stages from it;
  _router_shape_ok's k % (128 * SPLIT_K) check still holds (8192 % 4096 == 0). Keep ROUTER_WARM_MS
  so both new specializations compile in prepare().
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: "_router_hilo_kernel is 3.3 us for 8.4 MB (2.5 TB/s) vs a 1.8 us floor: with 4 K-steps and 3 stages each CTA pays a serial prologue + drain; 256 CTAs with all loads issued up front should cut ~0.6-1 us of the 5.8 us row"
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: m=8, same bucket, 5.7 us
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: same bucket, kernel 3.5 us, 6.2 us row
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: kernel 4.1 us, 6.5 us row; BLOCK_M=32 bucket
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: m=128 bucket unchanged; in-run control
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: m=256 bucket unchanged; in-run control
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T103959Z
review_companions: [router-splitk32-small-m-all-loads-in-flight-control-stages3]
stage: implemented
claimed_at: 2026-10-24T103959Z
base_sha: 71681a43d83042174393c75e858c09263d2334aa
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight
worktree: .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight
mode: check
implemented_at: 2026-10-24T104146Z
impl_commit: 60622cdebc40953738b4429ef80902673cd5f50f
submission_sha256: 9834d41b193d508e1f4e3a13ef0a0474b06c1d99274e63e6bf9062e821f8a9d5
norm_sha256: 25f5c6ce4256e43fd99221f01dbd3d5cdabee1d115fef5b6698a9497e0255641
submission_blob: a0fcbb70e25fa568ef94f81935748440798a1410
lint_warnings: 0
attempt_file: 2026-10-24T104146Z-check-router-splitk32-small-m-all-loads-in-flight.md
prepared_at: 2026-10-24T104146Z
---

# router-splitk32-small-m-all-loads-in-flight

## Change

Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + relaxed fp32 atomic_add epilogue, BLOCK_K=128, num_warps=4, and the m > 32 buckets (64,64,16) / (64,64,8) with num_stages=3 stay exactly as master. New small-m buckets: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=32; 17..32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=32. Grid becomes (1, 8, 32) = 256 CTAs (was 128), K-slice 256 = 2 K-steps of BLOCK_K=128 (was 4), and these buckets launch with num_stages=2 so both K-steps' hi, lo and x tiles are issued before the first dot (per CTA ~40 KB smem, so ~2 CTAs per SM are co-resident and all 256 run in one wave). Implement by having _router_config return (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES) and passing num_stages from it; _router_shape_ok's k % (128 * SPLIT_K) check still holds (8192 % 4096 == 0). Keep ROUTER_WARM_MS so both new specializations compile in prepare().

## Expected

- `layer=router load=decode_1 *` faster: _router_hilo_kernel is 3.3 us for 8.4 MB (2.5 TB/s) vs a 1.8 us floor: with 4 K-steps and 3 stages each CTA pays a serial prologue + drain; 256 CTAs with all loads issued up front should cut ~0.6-1 us of the 5.8 us row
- `layer=router load=verify_1 *` faster: m=8, same bucket, 5.7 us
- `layer=router load=decode_16 *` faster: same bucket, kernel 3.5 us, 6.2 us row
- `layer=router load=decode_32 *` faster: kernel 4.1 us, 6.5 us row; BLOCK_M=32 bucket
- `layer=router load=verify_16 *` flat: m=128 bucket unchanged; in-run control
- `layer=router load=verify_32 *` flat: m=256 bucket unchanged; in-run control

## Risk

A pure one-axis retune (grid depth x pipeline depth) of the landed kernel on the four m <= 32 router rows; m=128/256 rows are in-run controls for noise. The idea is memory-level parallelism, not tile shape: master's 128 CTAs each hold ~3 x 16 KB of weight in flight for only 4 steps, so each CTA's life is ~2 DRAM round trips plus launch; doubling the CTA count and halving the steps puts the whole 8.4 MB in flight at once. Costs: atomic traffic doubles (m=32: 262K fp32 atomics, ~1 MB, into a 32 KB output that stays in L2), and x is re-read by 8 N-blocks (trivial at m <= 32). Per-shape router effect is ~10-15% on 4 shapes but router is ~11% of each decode load, so geomean ~+1%: read the per-shape rows and the kernel us, not the headline. If flat, the router's small-m floor is launch + memset, and router-splitk-semaphore-no-memset is the remaining lever. Stacking: independent of the memset hypothesis (different lines); if both gain, combine and re-run.

## Adversarial review

Reviewed 2026-10-24T103959Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** This is a plausible, low-risk retune of the landed 3-D split-K router at m<=32. Doubling the grid to 256 CTAs and halving the K-steps is a reasonable way to attack the 1.5 us gap above the 1.8 us floor. But it moves two knobs at once (SPLIT_K 16->32 and num_stages 3->2), and the stated reason for the second knob does not hold. Run it with one control: SPLIT_K=32 at master's num_stages=3.

### Justification
Mechanism: FINDINGS puts _router_hilo_kernel at 3.3/3.3/3.5/4.1 us for m=1/8/16/32. Each run reads 8.4 MB of hi+lo, against a ~1.75 us floor at about 4.8 TB/s. The memset (vectorized_elementwise_kernel, 0.7-0.8 us) is separate and does not change. So the most the kernel can gain is ~1.5 us per row, and the 0.6-1 us claimed is ~10-15% of a 5.7-6.5 us row. On the per-shape noise of ~1.5% (LESSONS: Triton rows reproduce within ~1%), that is easy to read. Lineage supports grid depth: going from 32 CTAs at SPLIT_K=8 (router-triton-hilo-fused, kernel 5.3 us at m=1) to 128 CTAs at SPLIT_K=16 (master, 3.3 us) was a large drop. 256 CTAs at ~2 per SM on 132 SMs is one wave, so this is a fair next step. Directions match TASK weights: the four small-m router rows should be faster, and m=128/256 should be flat as in-run controls. Weak points: (1) Bytes in flight are already high. Master's 128 CTAs with 3 stages issue 2 K-steps (~40 KB) per CTA in the prologue, about 5-6 MB in flight. Little's law at ~1 us DRAM latency needs only ~5 MB. So the remaining gap may be fixed cost (first-byte latency, launch ramp, tail, atomic epilogue) rather than memory-level parallelism, and the gain may be smaller than claimed. (2) The num_stages=2 reasoning is wrong in both directions. With only 2 K-iterations, Triton's num_stages=3 pipeliner already issues both K-steps in the prologue, so 3 stages already gives 'all loads in flight'. And 3 stages x ~20 KB = ~60 KB/CTA still fits 2-3 CTAs per SM in 228 KB of smem, so 2 stages is not needed for co-residency. Going to 2 stages also changes the pipeline structure: the second K-step is issued from inside the loop rather than in the prologue. That is a second change with an unclear sign, bundled into a one-axis claim. (3) decode_32 is the weakest expected row. Its kernel is already 4.1 us vs 3.3 us at m=1, which suggests x traffic and atomic cost matter there. Doubling SPLIT_K doubles fp32 atomics (262K at m=32) into a 32 KB L2-resident output, so decode_32 could plausibly be flat or slower. Weakest link: the claim that the kernel is limited by memory-level parallelism or per-CTA serialization, rather than by fixed latency plus the atomic epilogue. No profile on record separates these.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (dba4ffd): this change retunes that kernel's own config, so it does not confound in the A/B sense. Its measured 128-CTA rows (5.8/5.67/6.2/6.54 us; kernel 3.3/3.3/3.5/4.1) are the per-shape baseline. Compare kernel-µs from the profile, not only row-µs, because the 0.7-0.8 us memset is a fixed part of every row.
- sglang-gemm-dispatch-parity (965b046): inert here. Only the bf16 layers use it, and the router no longer touches cuBLAS. The bf16 rows (qkv_gate decode_1 swung -3.5% in the last run) are noise and should not count against or for this change.
- router-hilo-at-m8 (f9dc05d): inert. Its ROUTER_FP32_MAX_TOKENS path was replaced wholesale by the 3-D kernel, and m=8 takes the m<=16 Triton bucket either way.
- Queued router-splitk-semaphore-no-memset is NOT on 'different lines', despite what the Risk section says. Its spec hardcodes the m<=32 buckets at SPLIT_K=16: workspace (16, 32, 256), ticket == 15, a fixed-order sum over 0..15. If both gain, stacking them needs that kernel rewritten for SPLIT_K=32: workspace (32, 32, 256), ticket == 31, a 32-way reduce, and a doubled reload of partials by the last CTA. That reload cost may erase part of this change's gain. They are not independently stackable.

### Interference and A/B needs
No ab-revert or ab-tune of a landed item is useful: the only landed item in this area is the kernel being retuned, and the others are inert on this path. A companion is still warranted, because the change moves two knobs and neither result would be readable without one. If the router rows lose, the cause could be the doubled atomics or the 2-stage pipeline structure. If they gain, the follow-up (stacking with the semaphore-reduce hypothesis, which wants the fewest partials) depends on whether SPLIT_K=32 itself is the lever or the stage change is. One control isolates this: SPLIT_K=32 grid with num_stages=3 kept. At 2 K-steps that already puts every load in the prologue, which is the hypothesis's stated mechanism in its cleanest form. Primary vs control gives the num_stages effect. Control vs master gives the SPLIT_K/grid effect. A second companion is not needed.

### Companions queued
- `router-splitk32-small-m-all-loads-in-flight-control-stages3`: Separates the SPLIT_K 16->32 grid-depth effect (control vs master) from the num_stages 3->2 effect (parent vs control). Without it, a loss could come from either the atomics or the stage change, and a gain cannot tell the semaphore-reduce follow-up which SPLIT_K to build on.

### Open questions
- Read kernel-µs from the profile for _router_hilo_kernel separately from the memset. Row-µs includes a fixed 0.7-0.8 us that this change cannot touch.
- Check how Triton lowers num_stages=2 with a 2-iteration K loop. If the pipeliner skips pipelining for very short loops, the 'all loads issued before the first dot' claim fails for the parent but holds for the stages=3 control.
- If decode_32 is flat or slower while m<=16 gains, give the 17..32 bucket its own SPLIT_K (keep 16) rather than one shared setting.
- Fix the Risk section's claim that this is 'independent of the memset hypothesis'. router-splitk-semaphore-no-memset hardcodes SPLIT_K=16 (16-way workspace, ticket == 15). If both gain, the combined run needs a 32-way version of that reduce.
- Confirm occupancy in the profile (achieved CTAs/SM ~2 and a single wave). Sixteen more registers per thread or a larger smem than estimated would split 256 CTAs into two waves and reverse the sign.


## Implementation notes

- Worktree file: `.worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight/muse_bf16_proj/submission.py` (base blob da55301). Diff: 1 file, 12 insertions, 11 deletions.
- `_router_config(m)` now returns `(BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES)`:
  - m <= 16: `(16, 32, 32, 2)`; 17..32: `(32, 32, 32, 2)`. Both use the new constant `ROUTER_SMALL_M_NUM_STAGES = 2`. Grid is (1, 8, 32) = 256 CTAs, K_SLICE = 256, so 2 K-steps of BLOCK_K = 128.
  - m <= 128: `(64, 64, 16, 3)`; m > 128: `(64, 64, 8, 3)`. These are unchanged from master and use `ROUTER_NUM_STAGES = 3`.
  - Docstring updated to match.
- `_router_forward` unpacks `num_stages` and passes `num_stages=num_stages` to the launch, replacing the global `ROUTER_NUM_STAGES`.
- `_router_shape_ok` unpacks the 4-tuple and ignores the stages value. Its `k % (128 * SPLIT_K)` check holds: 8192 % 4096 == 0.
- Unchanged: the kernel body, the torch.zeros output with the relaxed fp32 atomic_add epilogue, BLOCK_K, num_warps=4, and the bf16 layers. `ROUTER_WARM_MS = (1, 8, 16, 32, 64, 128, 256)` is also unchanged, so prepare() compiles both new (K_SLICE=256, stages=2) specializations at m=1/8/16 and m=32.
- No deviation from the spec. The reviewer's open questions (whether Triton pipelines a 2-iteration loop at num_stages=2, occupancy and wave count, kernel-us vs memset) are about reading the results and cannot be checked on this node. Read them from the profile. The stages=3 control is queued separately as `router-splitk32-small-m-all-loads-in-flight-control-stages3`.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints. The new branch picks num_stages from `m` (a shape) only. No new allocation; the zeroed output is the same as master.
- [x] Nothing derived from `data` is cached across calls. The diff adds no state writes.
- [x] Weights: unchanged. The router is still the BF16 hi + lo split built in prepare(), with FP32 accumulation. The bf16 weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16. x is still bf16 into the kernel, and the router output is fp32 as before.
- [x] n/a: attention. This problem has no attention, KV cache or queries; it is dense projections only.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff touches only launch-config constants.
- [x] All work happens inside the timed call, with no output caching or replay. The memset and kernel run on every call, and prepare() only warms compiles on synthetic x.
- [x] First line is still `# okc: problem=muse_bf16_proj`. No triton.autotune.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left in the worktree was removed.
