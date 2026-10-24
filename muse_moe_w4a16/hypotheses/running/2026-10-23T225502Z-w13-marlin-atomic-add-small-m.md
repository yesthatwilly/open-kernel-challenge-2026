---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-small-m
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires
  alongside atomic add) to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps SGLang's
  use_atomic_add=False / use_fp32_reduce=True. Marlin's non-atomic k-parallel reduction serializes
  slice partials through the workspace locks and an fp32 scratch (c_tmp) write+read; with atomic
  add each k-slice accumulates straight into the bf16 output, which must be zeroed first (one
  cache1.zero_() or memset launch, M*8 x 2I bf16: 128 KB at bs=1). This is the mode SGLang's dense
  Marlin path picks for small M (should_use_atomic_add_reduce); the MoE runner leaves it off.
  Shapes above 32 tokens untouched.
expected:
  - shapes: bs=1 *
    direction: faster
    why: at M=1 the w13 GEMM is one m-block with k-parallel slices; the lock-serialized fp32 reduce is a larger fraction of a ~30 us kernel than the memset it replaces
  - shapes: bs=8 *
    direction: faster
    why: same at bs=8
  - shapes: bs=16 *
    direction: faster
    why: same mechanism at bs=16; smaller share
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched (M=128 > 32)"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T040704Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T040704Z
base_sha: b0739163836e46200abd3b7d9a76de8b3467d783
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/w13-marlin-atomic-add-small-m
worktree: .worktrees/muse_moe_w4a16/w13-marlin-atomic-add-small-m
mode: check
implemented_at: 2026-10-24T040816Z
impl_commit: a1e0555ef3d13b14a47f7ddf2ce118450f5c57f9
submission_sha256: 53bfd3d1fcddb9f1561ecf52ca0c17959faabb9ec5efa2606b91dd129b277e2a
norm_sha256: 35019a046f63b8a7bc46696b66d327d6d57984ed93665acf3d9a23ad8ac77cb1
submission_blob: c2c950b382881f1b57b3a0c814bf085efdbf5086
lint_warnings: 1
attempt_file: 2026-10-24T040816Z-check-w13-marlin-atomic-add-small-m.md
prepared_at: 2026-10-24T040816Z
---

# w13-marlin-atomic-add-small-m

## Change

For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires alongside atomic add) to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps SGLang's use_atomic_add=False / use_fp32_reduce=True. Marlin's non-atomic k-parallel reduction serializes slice partials through the workspace locks and an fp32 scratch (c_tmp) write+read; with atomic add each k-slice accumulates straight into the bf16 output, which must be zeroed first (one cache1.zero_() or memset launch, M*8 x 2I bf16: 128 KB at bs=1). This is the mode SGLang's dense Marlin path picks for small M (should_use_atomic_add_reduce); the MoE runner leaves it off. Shapes above 32 tokens untouched.

## Expected

- `bs=1 *` faster: at M=1 the w13 GEMM is one m-block with k-parallel slices; the lock-serialized fp32 reduce is a larger fraction of a ~30 us kernel than the memset it replaces
- `bs=8 *` faster: same at bs=8
- `bs=16 *` faster: same mechanism at bs=16; smaller share
- `bs=128 *` flat: control: untouched (M=128 > 32)

## Risk

Risks: bf16 atomic accumulation across k-slices rounds each partial to bf16 (SGLang itself accepts this for its dense small-M Marlin path; rel-L2 tolerance 1.5% leaves wide margin, per-row 5x bound is the one to watch on near-zero rows); the added memset may cost what the reduce saves at bs=1, in which case the result is flat and the knob is retired. If the sgl_kernel binding rejects the combination on MoE, record as rejected. Activations remain bf16 as SGLang keeps them.

## Adversarial review

Reviewed 2026-10-24T040704Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A single-knob flag flip on the w13 Marlin call whose result is attributable on its own; the mechanism (replacing Marlin's lock-serialized k-slice reduction chain with concurrent bf16 atomics) is plausible and should show mainly at bs=1, not at bs=8/16 where the k-slice count is already ~1-2. Two stated premises are wrong or unsupported (the dense-path heuristic is n<2048 and k>=2048, not small M, so it would never pick atomic add for n=8192; and 'Marlin requires use_fp32_reduce=False alongside atomic add' is, to this reviewer's recollection of the vLLM/sgl_kernel template, not a requirement), but neither invalidates the run.

### Justification
Mechanism: in Marlin's template, k-slices of one (m-block, n-tile) are reduced through the workspace lock in strict slice order (barrier_acquire(lock, slice_idx) -> read/add/write the fp32 c_tmp tile -> barrier_release), so the reduction cost grows linearly with slice_count; with use_atomic_add the later slices only wait for slice 0's plain store (wait_negative_and_add) and then atomicAdd bf16x2 into C concurrently, so the chain depth collapses to ~2. slice_count is set by how the flattened (m-blocks x n-tiles x k-tiles) iteration space divides over the resident CTAs (sms*max_blocks_per_sm = 528 with the landed workspace). At bs=1 under ep8 there are only 1-4 m-blocks of 8 rows (block_size_m=8 for bs<=16), i.e. ~1k-4k tile iterations over 528 CTAs -> 2-8 iterations per CTA -> each n-tile's 32 k-tiles split across ~4-16 CTAs, so the serialized chain is 4-16 links deep; a ~0.5-1 us link latency would put the chain at a large share of a ~25 us w13 launch (the profile's 41.1 us Marlin figure at bs=1 aggregates both launches; the hypothesis's '~30 us' for w13 alone is an overestimate). That is consistent with the profile: bs=1 is 80 us against an 11.4 us floor, and the touched experts' w13 bytes (~18 MB each at 4.8 TB/s = ~4 us per expert) do not explain the 25 us launch, so the kernel is latency-bound there. At bs=8 (8-16 m-blocks, ~20 iterations per CTA) slice_count is ~1-2 and at bs=16 ~1, so the 'faster' expectations for bs=8 and bs=16 are overreach; expect flat there (and at bs=32, which the threshold also covers but the expected list omits: block_size_m=16, ~20-30 m-blocks, slice_count 1, where the kernel skips the reduce entirely so the flag is a no-op). Score arithmetic: a 10 us win at bs=1 is a 1.14x shape ratio -> ~1.02x geomean, so a success is just outside the +/-1.5% CI; a bs=1-only win of 5 us would be inside it. Weakest link: no evidence for the chain cost itself (no per-launch profile, no measured slice_count); the alternative explanation of the bs=1 gap is the poorly pipelined 2-8-iteration main loop per CTA (4-stage prologue amortised over almost nothing), which atomic add does nothing for. Second: the authority claim is wrong; vLLM/SGLang's dense should_use_atomic_add_reduce returns False when n >= 2048 or k < 2048 and would not pick it for n=8192 (the knob was added for the small-n, large-k case). To this reviewer's recollection vLLM's fused_marlin_moe does set use_atomic_add on sm90 for every M together with use_fp32_reduce=True (the kernel allocates c_tmp only when use_fp32_reduce && !use_atomic_add), so the mode is known-safe in a MoE runner but the 'requires use_fp32_reduce=False' premise is likely false. Precision: the atomic path accumulates the k-slice partials in bf16 (one rounding per slice, ~0.2% each), so at slice_count ~16 the w13 output carries roughly 0.5-0.8% extra relative error before silu*up and w2; the overall 1.5% rel-L2 budget should still hold (the base spends ~0.3-0.4% on bf16 activations) but the margin is not 'wide' at bs=1, and the pre-timing tests bs=1 and bs=4 exercise exactly the deepest slicing. The zero_() is probably unnecessary (slice 0 does a plain store before anyone adds) but harmless: ~1 us or less per shape, below the CI at every shape.

### Confounders on the current kernel
- sglang-marlin-path-parity (the base): fixes the workspace at marlin_make_workspace(max_blocks_per_sm=4), which sets the resident CTA count (528) and therefore slice_count at every shape; the size of any atomic-add gain is a function of that choice. Measured on master this is a fixed, known condition, so it does not confound the single run, but the effect would be larger under a bigger workspace and smaller under max_blocks_per_sm=1 (SGLang's dense runner).
- sglang-marlin-path-parity: its block_size_m pick (8 for bs<=16, 16 at bs=32) decides the m-block count and thus slice_count; marlin-block-size-m-next-larger (regression, 0.918x) showed that moving this knob changes bs=1 by ~5 us on its own, so the atomic-add result must only ever be compared against the same block_size_m rule (it is, on master).

### Interference and A/B needs
No companion. The change is one flag on one call (the fp32_reduce flip is dead code once the atomic path is taken) plus a sub-microsecond memset, so a gain is attributable to the atomic reduction and a flat/loss result retires the knob; the memset cannot hide a gain of the size that would matter (>=5 us at bs=1), so neither outcome is ambiguous. ab-revert of the only landed hypothesis is meaningless (there is no Marlin call without it). An ab-tune on the workspace (max_blocks_per_sm=8 for the w13 call) would be informative because the two knobs are coupled (more resident CTAs -> deeper slice chain -> more to gain from atomics), but marlin-workspace-blocks-per-sm-8 is already in flight on the same base and the right move is a merged run after both results land, not a third run now. Note for the harvester: if blocks-per-sm-8 shows bs=1 slower (its md predicts flat), the deeper fp32 lock chain is the likely reason and this hypothesis is its fix; if triton-w13-grouped-gemv-small-m (in flight, p2) lands, the w13 Marlin launch no longer runs at bs=1/8 and this hypothesis's only remaining shapes are bs=16/32 where its mechanism is near zero, so its value is contingent on that run not being promoted.

### Open questions
- Does the sgl_kernel MoE Marlin binding in the judge's environment honour use_atomic_add at all, or force it off? If it is silently ignored, the run measures use_fp32_reduce=False alone (the bf16 lock-chain reduce, i.e. the sibling w13-marlin-bf16-reduce-verify's change at small M) and a gain would be mis-attributed. Recommendation to the implementer: flip only use_atomic_add=True and leave use_fp32_reduce=True (vLLM's MoE runner passes that combination); only if the binding rejects it fall back to use_fp32_reduce=False and record the error text in the attempt log.
- Is the output zeroing actually required? In the vLLM template slice 0 stores plainly and later slices wait on the lock going negative before atomicAdd, so no pre-zero is needed. Keeping cache1.zero_() is safe either way; if the first run is flat at bs=1 by ~1 us, dropping the memset is the first thing to try, not a new mechanism.
- What is slice_count at bs=1 in practice (depends on the thread_k/thread_n config Marlin's heuristic picks for m_block_size_8 and on how many local routes the skewed routing lands on this GPU)? A print of n_padded/block_size_m in a check run's log would bound it and tell whether the mechanism can account for a 5-15 us win.
- If this gains at bs=1, the natural follow-ups are (a) the same flag on the w2 call (N=4096, K=4096, top_k=1: slice_count similar at bs=1), (b) atomic add for every M where slice_count > 1 rather than a token threshold, and (c) the merged run with marlin-workspace-blocks-per-sm-8.


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/w13-marlin-atomic-add-small-m`, file `muse_moe_w4a16/submission.py` only (+10/-1).

- New module constant `W13_ATOMIC_ADD_MAX_TOKENS = 32` beside `POST_NORM_EPS`.
- In `kernel_fn`, just before the w13 `moe_wna16_marlin_gemm` call: if `num_tokens <= W13_ATOMIC_ADD_MAX_TOKENS`, run `cache1.zero_()` and pass `w13_common = dict(common, use_atomic_add=True)` to the w13 call. Otherwise the w13 call gets `common` unchanged. The branch is on `num_tokens` (the shape) only.
- The w2 call is untouched and still gets `common` (`use_atomic_add=False`, `use_fp32_reduce=True`), so shapes above 32 tokens run exactly the base code.
- `cache1` is a view of the shared `cache13` buffer. Zeroing it before w13 is safe because `cache3` (the w2 output, the same memory) is only written later by the non-atomic w2 call, which stores plainly.
- **Deviation from the spec (from the adversarial review):** `use_fp32_reduce` stays **True** next to `use_atomic_add=True`, where the spec said `use_fp32_reduce=False`. Why: the spec's premise that Marlin needs fp32 reduce off when atomic add is on is probably false. vLLM's fused_marlin_moe passes atomic add together with use_fp32_reduce=True, and the kernel allocates c_tmp only when `use_fp32_reduce && !use_atomic_add`. Leaving it True also keeps this run separate from the sibling `w13-marlin-bf16-reduce-verify`, which flips only `use_fp32_reduce`. If the binding rejects the combination, fall back to `use_fp32_reduce=False` and record the error text in the attempt log.
- The output zeroing (`cache1.zero_()`) is kept as the spec wrote it, even though the review thinks it may be unnecessary (slice 0 stores plainly). If bs=1 comes out flat by about 1 us, dropping it is the first follow-up.
- **Precision watch (review):** the atomic path accumulates the k-slice partials in bf16, rounding once per slice. At bs=1/4 (slice_count up to ~16) that is roughly 0.5-0.8% extra relative error on the w13 output before silu*up and w2. The 1.5% rel-L2 budget should hold, but the margin is not wide, and the pre-timing tests at bs=1 and bs=4 exercise the deepest slicing. The per-row 5x bound on near-zero rows is the one to watch.
- **Determinism:** with atomic adds the summation order of the k-slice partials is nondeterministic, so outputs at bs<=32 can differ in the last bf16 bits between replays and runs. That does not break any rule, but bit-exact comparisons across runs will not hold.
- Expectation per the review: any gain should show mainly at bs=1. At bs=8/16/32, slice_count is already about 1-2 (a no-op at slice_count 1), so expect flat there despite the spec's "faster".

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/w13-marlin-atomic-add-small-m diff`:

- [x] CUDA-graph capturable: the new branch is on `num_tokens` (from `x.shape`), not on input values. `cache1.zero_()` is a capturable memset with no allocation (`dict(...)` is host-side kwargs only). No `.item()`/`.cpu()`/`synchronize()`/prints were added.
- [x] Nothing derived from `data` is cached across calls: no new state is written.
- [x] Weights: untouched. `prepare()` is unchanged and NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: cache1 is still the bf16 output of w13. Only the reduction mode of k-slice partials changes, which is a Marlin kernel option SGLang/vLLM expose.
- [x] n/a: KV cache FP8 / queries FP8 (no attention in this problem).
- [x] n/a: attention sees every key (no attention in this problem). No expert or route skipping was added.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call, with no output caching or replay.
- [x] First line is still `# okc: problem=muse_moe_w4a16`. No triton.autotune.

okc_lint output:

```
warn  host-sync    L114: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/w13-marlin-atomic-add-small-m/muse_moe_w4a16/submission.py
```

The L114 warning is pre-existing (in `prepare()`), not part of this diff. `python3 -I -m py_compile` passed, and its `__pycache__` was removed.
