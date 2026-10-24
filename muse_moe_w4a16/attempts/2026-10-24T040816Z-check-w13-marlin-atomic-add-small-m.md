# muse_moe_w4a16 · 2026-10-24T040816Z · check · w13-marlin-atomic-add-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225502Z-w13-marlin-atomic-add-small-m.md` (branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-small-m`).

- **Job:** `c8ea8e21eea447f9` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `a1e0555`

## Goal
For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires alongside atomic add) to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps SGLang's use_atomic_add=False / use_fp32_reduce=True. Marlin's non-atomic k-parallel reduction serializes slice partials through the workspace locks and an fp32 scratch (c_tmp) write+read; with atomic add each k-slice accumulates straight into the bf16 output, which must be zeroed first (one cache1.zero_() or memset launch, M*8 x 2I bf16: 128 KB at bs=1). This is the mode SGLang's dense Marlin path picks for small M (should_use_atomic_add_reduce); the MoE runner leaves it off. Shapes above 32 tokens untouched.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Risks: bf16 atomic accumulation across k-slices rounds each partial to bf16 (SGLang itself accepts this for its dense small-M Marlin path; rel-L2 tolerance 1.5% leaves wide margin, per-row 5x bound is the one to watch on near-zero rows); the added memset may cost what the reduce saves at bs=1, in which case the result is flat and the knob is retired. If the sgl_kernel binding rejects the combination on MoE, record as rejected. Activations remain bf16 as SGLang keeps them.

## Result (after the run)
- job `c8ea8e21eea447f9` (check), status done, passed True, tests 5/5
- speedup 0.9869x, 95% CI [0.9721, 1.0017], score 13626.657120834008 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-small-m.patch`; branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-small-m`
