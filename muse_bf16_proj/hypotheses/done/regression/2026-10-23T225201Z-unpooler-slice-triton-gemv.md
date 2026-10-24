---
problem: muse_bf16_proj
slug: unpooler-slice-triton-gemv
created: 2026-10-23T225201Z
base: master
area: unpooler GEMM
change: >
  For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight
  read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call in kernel_fn with one Triton
  launch and no split-K reduce: grid = (1024/BLOCK_N,) with BLOCK_N=8, i.e. 128 programs; each
  program streams its 8 weight rows over the full K=4096 in BLOCK_K=512 steps (num_stages=4),
  loads the x tile [16, BLOCK_K] with rows >= m masked, accumulates in fp32 with acc +=
  tl.sum(x[:, None, :] * w[None, :, :], axis=2) (FMA broadcast; tl.dot needs BLOCK_N >= 16 and
  this shape is launch-bound, not math-bound), and stores the [m, 8] bf16 slab into a
  torch.empty((m, 1024), bf16). prepare() unchanged; W[:1024] of the row-major weight is already
  contiguous. Kernels moved: nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT (4.0-4.3 us) +
  cublasLt::splitKreduce_kernel (1.5-1.6 us) and the gap between them become one kernel; the
  reported 6.9-7.1 us is 4x the 1.7 us floor, and the goal is 2 launches to 1 on a sub-10 us
  shape. Unpooler at m >= 16 (n=8192) and every other layer keep the cuBLAS dispatch.
expected:
  - shapes: layer=unpooler load=decode_1 *
    direction: faster
    why: 6.9 us for 8.4 MB (floor 1.7); one launch instead of nvjet splitK + splitKreduce; target ~4 us
  - shapes: layer=unpooler load=verify_1 *
    direction: faster
    why: m=8 same sliced shape, 7.1 us
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: m=16 is the full 8192-row weight, keeps cuBLAS; control
  - shapes: "*layer=o*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-23T230427Z
review_companions: null
stage: harvested
claimed_at: 2026-10-23T230427Z
base_sha: e64f27ef09b3512d26573dde4d35b8ec3ade1f63
base_blob: 5a2064af06f442f01dff926cb4f1eca75f992a24
branch: hyp/muse_bf16_proj/unpooler-slice-triton-gemv
worktree: .worktrees/muse_bf16_proj/unpooler-slice-triton-gemv
mode: check
implemented_at: 2026-10-24T015218Z
impl_commit: 7c05884905deaf8903e034a47905fe43b780ebc4
submission_sha256: 65e133c4f7f9eb9b99ab1528e2bd5b61e9cb0d8ebed282785307cb19f07a68ba
norm_sha256: 5b9dc0afdff281cdbecda8a1abaa1a2a80002ca2d464897fa502a784f2ab671f
submission_blob: 8b4a1d5c203a5775cb41fad007ee95a18f9765a3
lint_warnings: 0
attempt_file: 2026-10-24T015218Z-check-unpooler-slice-triton-gemv.md
prepared_at: 2026-10-24T015218Z
job: 4c2b113f81434570
dispatched_at: 2026-10-24T015224Z
outcome: regression
verdict_detail: regression
speedup: 0.880091413038843
speedup_ci: [0.8668900418432604, 0.8932927842344255]
harvested_at: 2026-10-24T021807Z
best_at_harvest: 0.9020752041800139
promotion: null
---

# unpooler-slice-triton-gemv

## Change

For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call in kernel_fn with one Triton launch and no split-K reduce: grid = (1024/BLOCK_N,) with BLOCK_N=8, i.e. 128 programs; each program streams its 8 weight rows over the full K=4096 in BLOCK_K=512 steps (num_stages=4), loads the x tile [16, BLOCK_K] with rows >= m masked, accumulates in fp32 with acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2) (FMA broadcast; tl.dot needs BLOCK_N >= 16 and this shape is launch-bound, not math-bound), and stores the [m, 8] bf16 slab into a torch.empty((m, 1024), bf16). prepare() unchanged; W[:1024] of the row-major weight is already contiguous. Kernels moved: nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT (4.0-4.3 us) + cublasLt::splitKreduce_kernel (1.5-1.6 us) and the gap between them become one kernel; the reported 6.9-7.1 us is 4x the 1.7 us floor, and the goal is 2 launches to 1 on a sub-10 us shape. Unpooler at m >= 16 (n=8192) and every other layer keep the cuBLAS dispatch.

## Expected

- `layer=unpooler load=decode_1 *` faster: 6.9 us for 8.4 MB (floor 1.7); one launch instead of nvjet splitK + splitKreduce; target ~4 us
- `layer=unpooler load=verify_1 *` faster: m=8 same sliced shape, 7.1 us
- `layer=unpooler load=decode_16 *` flat: m=16 is the full 8192-row weight, keeps cuBLAS; control
- `*layer=o*` flat: untouched; control

## Risk

Risk: 128 programs each reading 64 KB may not pull 8.4 MB faster than cuBLAS's 4-way splitK; the shape is so small that launch latency (~2-3 us per graph node) dominates, so the win mostly comes from dropping the second node, which is why it should still be faster even at equal bandwidth. The FMA-broadcast form keeps register use at 16x8x512 fp32 = 64K values per program if written naively; tile it as BLOCK_K=128 inner chunks or use tl.dot with BLOCK_N=16 (64 programs) if Triton spills; either is a tune, not a new idea. Tests touched: m=5 unpooler and m=130/m=1000 unpooler (the latter two are n=8192, untouched path). Correctness: same bf16-in/fp32-acc/bf16-out as cuBLAS. CUDA graph: fixed-shape torch.empty only.

## Adversarial review

Reviewed 2026-10-23T230427Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Mechanism is sound and the single run is interpretable per shape without companions: the unpooler slice (n=1024, k=4096, 8.4 MB) runs the same nvjet splitK + splitKreduce pair in the starter, the parent and SGLang, so any per-shape move on unpooler decode_1/verify_1 is attributable to the Triton kernel alone. The weakest link is the kernel formulation as written (FMA-broadcast with a fixed BLOCK_M=16 tile), which is issue-bound/register-heavy at num_warps=4 and could land near cuBLAS's 4.1 us rather than under it; the fix is a tune inside the same hypothesis (BLOCK_M=next_power_of_2(m), or tl.dot with a zero-masked 16-row weight tile keeping 128 programs), not a new run.

### Justification
A. Plausible. Profile (parent result job 017804b34fed40b3, FINDINGS): unpooler decode_1 reported 6.48 us = nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.1 us + cublasLt::splitKreduce 1.6 us + inter-node gap, vs a 1.75 us floor; verify_1 6.84 us (4.0 + 1.6). cuBLAS pulls 8.4 MB in 4.1 us = ~2 TB/s, well under 4.8 TB/s, because its 64x8 tiling leaves most SMs idle on a 1024-row N; a one-wave grid of 128 programs each streaming a contiguous 64 KB (W[:1024] of the row-major [8192,4096] weight is contiguous, no repack) has more memory-level parallelism, and with num_stages=4 x 8 KB per stage per CTA ~4 MB is in flight across the GPU, which is roughly Little's-law sufficient (4.8 TB/s x ~0.8 us). Dropping the reduce node also removes ~1 us of graph-node gap. A single kernel at ~2.5-3 us, i.e. reported ~3-4 us, is a realistic target; the hypothesis's own '~4 us' is conservative. Expected shapes are consistent with TASK.md: the only shapes with n=1024 are m=1 (decode_1) and m=8 (verify_1), both weight 62; decode_16 (n=8192) and layer=o are proper flat controls and the globs ('load=decode_1 ' with trailing space) do not accidentally match decode_16. Magnitude: a 3 us saving on unpooler is ~5% of the decode_1 load (~58.6 us) and ~4% of verify_1 (~75.2 us); geomean over 6 loads is ~+1.3-1.6%, which is AT the judge's minimum +-1.5% CI, so the headline speedup may not clear noise even when the kernel works perfectly. The harvest must be judged on the per-shape attribution rows (per-shape err_ns ~20 ns on 6.5 us, so a 2-3 us move is unmistakable), not on the CI of the aggregate. Weakest link: the kernel as specified. acc += tl.sum(x[:,None,:]*w[None,:,:], axis=2) with x tile [16,512] and w tile [8,512] materialises a [16,8,512] fp32 product (64K values/CTA = 512 regs/thread at num_warps=4): it will spill or fail to compile without chunking, and even chunked, 16x8x4096 = 524K MACs per CTA is 4096 FMA + reduction shuffles per thread at num_warps=4, ~2+ us of pure issue, comparable to the memory time it is supposed to hide under. At m=1 15 of the 16 x rows are zero work. The fixed BLOCK_M=16 is only needed by tl.dot; for the FMA form BLOCK_M should be triton.next_power_of_2(m) (1 at decode_1, 8 at verify_1), which cuts issue 2-16x and removes the register problem. Alternatively use tl.dot with BLOCK_N=16 where weight rows 8..15 are zero-masked loads (no extra HBM traffic), which keeps 128 programs rather than the 64 the hypothesis mentions. Either is inside the hypothesis's stated tuning envelope.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (parent, only landed item): on the unpooler m<=8 shapes its change is a no-op: _bf16_gemm_dispatch_impl(x, W[:1024], None) resolves to the same cuBLAS nvjet splitK + splitKreduce pair the starter's x @ w.T picked (attribution table: unpooler decode_1 1.025, verify_1 0.999, both flat). Its router hi/lo and prepare() changes touch neither the unpooler weight nor these shapes. A gain or loss here cannot be an interaction with it; the change would behave identically if the parent were absent.

### Interference and A/B needs
No companion is warranted. ab-revert: the only landed hypothesis is inert on the shapes this one touches (same kernel pair before and after), so reverting it would change nothing measurable here. ab-tune: the parent has one tunable (ROUTER_FP32_MAX_TOKENS) that does not reach the unpooler path. control: the hypothesis has one sub-change (one Triton launch replacing the cuBLAS pair on n=1024 only), and the judge's per-shape rows already isolate it (unpooler decode_1/verify_1 vs the untouched unpooler decode_16/32, verify_16/32 and every other layer). A per-shape gain is unambiguous: fewer launches and one kernel faster than the pair. A per-shape loss is only weakly ambiguous (fundamental 'Triton cannot beat nvjet on this shape' vs 'the FMA-broadcast tile is issue-bound'); that is resolved by the implementer choosing a sane tile up front (see justification) and, if it still loses, the profile's single kernel duration says which it was (kernel >4 us = formulation; kernel ~3 us but reported >6.5 us = launch/graph overhead not recovered). Interaction with queued items: bf16-cublaslt-backend-toggle could change the cuBLAS kernel on these same two shapes if it lands first, which would move this hypothesis's base, not confound this run (both build on the parent). unpooler-full-triton-skinny-m32 is disjoint (n=8192, m in {16,32}) and explicitly leaves the m<=8 slice to this hypothesis.

### Open questions
- Which tile does the implementer actually build? Recommend, in order: (1) FMA form with BLOCK_M = triton.next_power_of_2(m) as a constexpr (1 for decode_1, 8 for verify_1), BLOCK_N=8, BLOCK_K=256-512, num_stages=4, num_warps=4-8; or (2) tl.dot with BLOCK_M=16, BLOCK_N=16 via zero-masked weight rows 8..15 so the grid stays 128 programs. The fixed BLOCK_M=16 FMA-broadcast tile as written is the one configuration likely to be issue-bound/spill.
- Would 256 programs (BLOCK_N=4, 2 CTAs per SM) beat 128 on memory-level parallelism for an 8.4 MB stream? Cheap to try as a tune if the first result is between cuBLAS and the floor.
- Harvest criterion: the aggregate CI will likely straddle 1.0 even on success (~+1.5% geomean ceiling for a 2-of-30-shape change). The gain/flat call should rest on the unpooler decode_1/verify_1 per-shape rows (expect 6.5 -> 3-4 us) and the profile's kernel list collapsing from two entries to one.
- Confirm the judge's evaluator does an eager warm-up before graph capture; if it captures on the first call, Triton's first-call module load happens under capture. Pre-warming in prepare() removes the question either way.


## Implementation notes

Changed only `muse_bf16_proj/submission.py` (+66 lines, nothing removed):

- **Fast path in `kernel_fn`.** When `layer == "unpooler"` and `x.shape[0] <= 8` and `cfg["n"] == 1024` and `cfg["k"] == 4096`, the call goes to `_unpool_slice_gemv(x, state["unpooler"])`. Every other case falls through to the unchanged `_bf16_gemm_dispatch_impl` (cuBLAS) or router path. That includes unpooler at m >= 16 (n = 8192) and the m = 130 / m = 1000 tests.
- **New kernel `_unpool_slice_gemv_kernel` (Triton).**
  - Grid is `(1024 / BLOCK_N,)` = **128 programs**, with BLOCK_N = 8. There is one launch and no split-K reduce.
  - Each program streams its 8 contiguous weight rows (64 KB) over the full K = 4096. It reads the full `[8192, 4096]` row-major weight through its first 1024 rows, so no slice copy or repack is made.
  - Per step: `acc[BLOCK_M, 8] += tl.sum(x[:, None, :] * w[None, :, :], axis=2)`. Loads are bf16 and are upcast to fp32. Accumulation is fp32 and the store is bf16.
  - x rows `>= m` are masked: loads return 0 and those rows are not stored.
  - The output is `torch.empty((m, 1024), bf16)`.
  - Launch settings: num_warps = 4, num_stages = 4. There is no autotune.
- **Tile tune (inside the stated envelope, following the reviewer):**
  - **BLOCK_M = `triton.next_power_of_2(m)`** (1, 2, 4 or 8) instead of the fixed 16. decode_1 therefore does no wasted rows, and the FMA form no longer materialises a `[16, 8, 512]` fp32 product.
  - **BLOCK_K = `max(128, 512 // BLOCK_M)`**, which gives 512 at m = 1, 256 at m = 2 and 128 at m = 4..8. The per-CTA product stays at 4K-8K fp32 values (32-64 registers per thread at 4 warps), which avoids the spill the review warned about.
  - The spec's BLOCK_K = 512 is therefore kept only at m = 1. At m = 8 the K loop runs as 32 x 128-wide chunks.
  - BLOCK_N, the grid, num_stages and the no-split-K design are as specified.
- **`prepare()`.** The weights are untouched. The only addition is a pre-warm: each m in 1..8 runs once on synthetic `torch.zeros((m, 4096), bf16)` against the real weight tensor, so that every BLOCK_M / BLOCK_K and `M == 1` specialization is compiled and loaded before any timed call or graph capture. The outputs are discarded and nothing input-derived is kept.
- **Deviations.**
  - BLOCK_M and BLOCK_K were tuned as described above.
  - `x.contiguous()` was added, which is a no-op on judge inputs. The kernel assumes a unit k-stride, and this keeps any non-contiguous caller correct.
  - Not done: the tl.dot alternative and BLOCK_N = 4 (256 programs). These are left as follow-up tunes if the result lands between cuBLAS and the floor.
- **Not runnable here** (no GPU or Triton on this node). I checked the syntax with `ast.parse` and checked the logic by reading it. The first `okc check` is the real compile test.

## Rules check

Verified against `git diff` in the worktree (hyp/muse_bf16_proj/unpooler-slice-triton-gemv):

- [x] CUDA-graph capturable: the new path has no `.item()`, `.cpu()`, sync or prints. Branching is only on `x.shape[0]`, `cfg["n"]`, `cfg["k"]` and `layer`. Tiles derive from m (a shape). The output is a fixed-shape `torch.empty`. The kernels are pre-compiled in `prepare()`, so nothing compiles under capture.
- [x] Nothing derived from `data` is cached across calls: there are no module-level containers. The warm-up in `prepare()` uses synthetic zeros, and its outputs are discarded.
- [x] Weights: the unpooler is read as stored (bf16, row-major, first 1024 rows, in place). There is no new repack or precision change, and the router hi/lo is unchanged.
- [x] Activations stay BF16 where SGLang keeps them: x is read as bf16 and upcast in registers, with fp32 accumulation and bf16 output, the same contract as cuBLAS.
- [x] Attention sees every key: N/A, since this problem has no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only new imports are `triton` and `triton.language`.
- [x] All work happens inside the timed call, with no output caching or replay: every call launches the kernel on the fresh x and writes a fresh output.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj` -> `CLEAN: submission.py`

## Result

- job `4c2b113f81434570` (check), status done, passed True, tests 9/9
- speedup 0.8801x, 95% CI [0.8669, 0.8933], score 5488.975268284764 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/unpooler-slice-triton-gemv.result.json`, `muse_bf16_proj/hypotheses/done/regression/unpooler-slice-triton-gemv.patch`; branch `hyp/muse_bf16_proj/unpooler-slice-triton-gemv`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 13.96 | 14.35 | 0.9717 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.52 | 6.89 | 0.9784 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 21.22 | 21.52 | 0.9996 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.48 | 11.53 | 0.5978 | lost |
| layer=router load=decode_1 m=1 | 62 | 10.36 | 10.44 | 0.9731 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.89 | 14.33 | 0.994 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.7 | 6.69 | 0.9776 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 21.45 | 21.49 | 1.0048 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.66 | 18.79 | 0.9946 | flat |
| layer=router load=decode_16 m=16 | 62 | 14.51 | 15.19 | 0.9976 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 13.97 | 1.019 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.62 | 6.84 | 1.021 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 21.75 | 21.51 | 1.0097 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.55 | 18.86 | 0.9963 | flat |
| layer=router load=decode_32 m=32 | 62 | 15.14 | 15.1 | 1.0043 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 14.04 | 14.39 | 1.0054 | flat |
| layer=o load=verify_1 m=8 | 62 | 7.01 | 6.96 | 1.0374 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 21.28 | 21.17 | 1.0151 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.84 | 84.79 | 0.0857 | lost |
| layer=router load=verify_1 m=8 | 62 | 26.14 | 26.43 | 1.0067 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.9 | 17.26 | 1.011 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.58 | 7.79 | 1.0125 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.13 | 25.41 | 0.9935 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.84 | 20.29 | 0.9859 | flat |
| layer=router load=verify_16 m=128 | 62 | 16.62 | 16.64 | 1.0038 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.58 | 19.82 | 0.9892 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.25 | 9.81 | 0.9951 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.17 | 30.63 | 1.0066 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.33 | 25.0 | 0.993 | flat |
| layer=router load=verify_32 m=256 | 62 | 17.76 | 17.71 | 0.9919 | flat |

gained on o verify_1; lost on unpooler decode_1, verify_1. Expected hit: 1/4. Noise ±1.5%.

## Follow-ups

- record which shapes lost (Attribution) and do not retry this area in this form.

