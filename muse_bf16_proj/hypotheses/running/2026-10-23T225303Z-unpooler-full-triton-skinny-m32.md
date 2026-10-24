---
problem: muse_bf16_proj
slug: unpooler-full-triton-skinny-m32
created: 2026-10-23T225303Z
base: master
area: unpooler GEMM
change: >
  Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16,
  32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route unpooler at 8 < m <= 32 to the
  same kernel with BLOCK_N=64 (8192/64 = 128 programs, one wave), BLOCK_K=256, num_stages=4,
  BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). Unpooler at m <=
  8 (the 1024-row slice) and at m >= 128 keep whatever they have (cuBLAS, or
  unpooler-slice-triton-gemv if it landed); pooler keeps the parent's path. Kernel moved:
  nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT, a single non-splitK kernel at 18.5-18.8 us for 67
  MB (3.6 TB/s) against a 14.1 us floor; unlike pooler there is no reduce node to drop, so this
  run isolates the pure streaming-bandwidth question: does the full-K Triton stream beat nvjet's
  2x1 cluster kernel on an N-heavy, K-short (4096) weight.
expected:
  - shapes: layer=unpooler load=decode_16 *
    direction: faster
    why: 18.5 us single nvjet kernel at 3.6 TB/s; 128 programs streaming 512 KB each, target <= 17 us
  - shapes: layer=unpooler load=decode_32 *
    direction: faster
    why: 18.9 us, same
  - shapes: layer=unpooler load=decode_1 *
    direction: flat
    why: sliced path untouched here; control
  - shapes: layer=unpooler load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
risk: medium
depends_on: pooler-triton-skinny-m32
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T032454Z
review_companions: [unpooler-full-triton-skinny-m32-control-blockm16-all-m]
stage: implemented
claimed_at: 2026-10-24T032454Z
base_sha: d2840d088aa35522a03b40c59e174287cf935f67
base_blob: 2752a016d386e9261ec77e25b426c05b76c071ac
branch: hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32
worktree: .worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32
mode: null
implemented_at: 2026-10-24T032616Z
impl_commit: fbfb795472c2ae79a1b093a08fb5e42d1867da77
submission_sha256: 83ce8c03f0998b92424c4bf16cc65532a525c831594e41c331b11ebb7dfd3886
norm_sha256: cc6fb688cf298485ad99f1cd118fb3517770742d80e0ee0508613285ce0462d0
submission_blob: 29884ed7c43020beb1ad7c469eafaa8b9f362470
lint_warnings: 0
---

# unpooler-full-triton-skinny-m32

## Change

Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route unpooler at 8 < m <= 32 to the same kernel with BLOCK_N=64 (8192/64 = 128 programs, one wave), BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). Unpooler at m <= 8 (the 1024-row slice) and at m >= 128 keep whatever they have (cuBLAS, or unpooler-slice-triton-gemv if it landed); pooler keeps the parent's path. Kernel moved: nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT, a single non-splitK kernel at 18.5-18.8 us for 67 MB (3.6 TB/s) against a 14.1 us floor; unlike pooler there is no reduce node to drop, so this run isolates the pure streaming-bandwidth question: does the full-K Triton stream beat nvjet's 2x1 cluster kernel on an N-heavy, K-short (4096) weight.

## Expected

- `layer=unpooler load=decode_16 *` faster: 18.5 us single nvjet kernel at 3.6 TB/s; 128 programs streaming 512 KB each, target <= 17 us
- `layer=unpooler load=decode_32 *` faster: 18.9 us, same
- `layer=unpooler load=decode_1 *` flat: sliced path untouched here; control
- `layer=unpooler load=verify_16 *` flat: m=128 keeps cuBLAS; control

## Risk

Because the cuBLAS kernel here is already a single launch, a gain can only come from bandwidth, which makes this the cleanest test of the streaming kernel itself: if pooler wins but this is flat, the pooler gain was the dropped reduce node, which should go into FINDINGS. K=4096 means only 16 BLOCK_K steps per program; raise num_stages to 6 if the parent's profile shows the pipeline draining. Tests touched: m=130 unpooler outliers (n=8192, m=130 stays on cuBLAS), m=5 unpooler (slice path, untouched). Same correctness and graph properties as the parent.

## Adversarial review

Reviewed 2026-10-24T032454Z by an adversarial Fable subagent (model fable).

**Verdict: proceed-with-companions.** The run is interpretable per shape (only unpooler decode_16/32 can move and the profile gives the Triton kernel's own us, hence its TB/s), but the parent's result already answers the stated question against it: the same plain Triton full-K stream on the same 67 MB took 20.2 us at BLOCK_M=16 (3.3 TB/s) and 25.2 us at BLOCK_M=32 (2.7 TB/s), against nvjet's 18.4/18.8 us single kernel here, so the likely outcome is flat-to-loss at m=16 and a large loss at m=32; one control that isolates the BLOCK_M=32 tile is needed to read a mixed result.

### Justification
Mechanism: unpooler at m=16/32 is a 67.1 MB bf16 stream (n=8192, k=4096), floor 14.1 us; master runs one non-splitK nvjet 2x1 kernel at 18.4 us (m=16) / 18.8 us (m=32) with reported medians 18.3-18.9 us (BEST job 9aa98e889be140eb profile), i.e. ~3.6 TB/s with no reduce node and no inter-node gap to remove. A gain therefore needs the Triton kernel itself to beat 18.4 us (>= 3.65 TB/s), and the 'target <= 17 us' in expected needs 3.95 TB/s. The only measurement of this kernel family in the workspace is the parent pooler-triton-skinny-m32 (done/neutral, 0.998x): _pooler_skinny_kernel on the same 67 MB (BLOCK_N=32, BLOCK_K=256, 4 stages, 32 K-steps) ran 20.2 us at m=1/8/16 and 25.2 us at m=32. Its 5-6% pooler gains at m<=16 came entirely from dropping the 1.6-1.9 us splitKreduce plus the graph gap (cuBLAS kernel sum 19.9-20.6 us vs reported 21.2-21.9); the Triton stream was slower than nvjet's own kernel time. Unpooler has nothing to drop, so the parent's data predict ~20 us vs 18.4 (loss ~8%) at m=16 and ~25 us vs 18.8 (loss ~25%) at m=32 unless BLOCK_N=64 / 16 K-steps changes the bandwidth picture, which the hypothesis asserts but does not argue (2x the weight bytes in flight per SM is the one real difference; fewer K-steps amortise the pipeline prologue worse, not better). The hypothesis was written before the parent's result and does not mention the m=32 anomaly at all. Shape/weight consistency with TASK.md is fine: both target shapes weigh 62, unpooler is ~31% of decode_16 and ~30% of decode_32, so a full success (18.6 -> 17 us) is ~+2.7% on two of six loads, ~+0.9% geomean, under the +-1.5% CI, so the per-shape rows and the kernel us are the readout, never the headline. Weakest link: the stated question ('does the full-K Triton stream beat nvjet's 2x1 kernel') has already been answered negatively for this kernel at BLOCK_N=32 by the parent, and the parent's own Risk section says the dependents should be retuned (BLOCK_N=16/256 programs or num_stages=6) before running if the Triton kernel is not faster than nvjet; this spec changes BLOCK_N in the other direction (64) and keeps BLOCK_M=32 at m=32, the tile that lost 16% on pooler. Second concern: smem at BLOCK_M=32 is 4 stages x (64x256x2 + 32x256x2) = 192 KB, near the 227 KB limit; Triton may fail to pipeline at 4 stages, and raising to 6 as the Risk section suggests is impossible at that tile.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (landed): inert on unpooler m=16/32; _bf16_gemm_dispatch_impl(x, W[:8192], None) resolves to the same nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT the starter picked (unpooler rows flat in its attribution); this change replaces that call on exactly these two shapes and would behave identically with the parent absent.
- router-hilo-at-m8 (landed): a different layer timed in its own CUDA graph, no shared kernel, memory or numerics; cannot confound the unpooler rows. It does affect the headline: because depends_on=pooler-triton-skinny-m32 is unpromoted, churn.py will build on the pooler branch's impl_commit d2840d0 (parity base e64f27e, ROUTER_FP32_MAX_TOKENS=8), so this run's aggregate lacks the +2.2% router gain and is not comparable to BEST 1.0307x; the per-shape attribution against the pooler result (same base blob 2752a01) is still correct.
- pooler-triton-skinny-m32 (parent branch, unpromoted): shares the @triton.jit kernel; a BLOCK_N=64 specialization is a separate binary and pooler shapes are separate graphs, so no runtime interaction, but the inherited pooler decode_32 loss (0.887x, 24.5 vs 21.9 us) drags the headline and must be attributed away by comparing against the parent's result rows, not SGLang.

### Interference and A/B needs
A clean gain (both unpooler rows faster, kernel us < 18.4/18.8) would be unambiguous: nothing else in those graphs changes. A clean loss would be interpretable only at the family level if the kernel us lands near the parent's 3.3 TB/s. The most likely outcome on the parent's evidence is mixed: m=16 near-flat or a small loss and m=32 a large loss, and that m=32 row is ambiguous between (a) the BLOCK_M=32 tl.dot tile / 192 KB smem pipeline being pathological, as the pooler m=32 row (25.2 vs 20.2 us for identical bytes) already suggests, and (b) the Triton stream's bandwidth ceiling. A 'control' that holds everything else and runs BLOCK_M=16 for every m (two 16-row x tiles and two tl.dot per K-step against the one weight tile at m=32, weight still read once) separates the tile effect from the bandwidth effect, and also tells the o- and qkv-gate-triton-skinny siblings whether their m=32 rows are salvageable. ab-revert is not applicable: neither landed hypothesis touches the unpooler path, and reverting router-hilo-at-m8 is moot since the build base lacks it anyway. ab-tune: the parent's pooler tile has no effect on unpooler rows. One companion, priority 3; a second (BLOCK_N=32 / 256 programs for memory-level parallelism) is a retune to queue only after the kernel us is known, not a disambiguator.

### Companions queued
- `unpooler-full-triton-skinny-m32-control-blockm16-all-m`: Separates the m=32 row's result into tile (BLOCK_M=32 wgmma tile / smem pipeline) versus streaming bandwidth: if the parent loses at m=32 and this control does not, the family's m=32 problem is the 32-row tile and o-/qkv-gate-triton-skinny should adopt BLOCK_M=16 everywhere; if both lose equally, the plain Triton stream's ceiling (~3.3 TB/s on the pooler) is confirmed on a second shape and the whole skinny-Triton family should be withdrawn.

### Open questions
- The hypothesis predates the parent's result and should be read against it: on the same 67 MB the Triton stream took 20.2 us (m<=16) and 25.2 us (m=32) where nvjet's unpooler kernel takes 18.4/18.8 us; the implementer should state in the attempt file what BLOCK_N=64 / 16 K-steps is expected to change about that before spending the run, or apply the parent's own Risk-section retune (256 programs or more stages) inside this hypothesis's envelope.
- Build base: churn.py will base this on the parent's impl_commit d2840d0 (no router-hilo-at-m8, ROUTER_FP32_MAX_TOKENS=8) because the parent is unpromoted; the headline cannot be compared to BEST and promotion of a gain would need a rebase onto master plus a fresh run. Consider whether the dispatcher should instead base on master and copy the kernel in (depends_on null), since the unpooler path shares no runtime state with the pooler path.
- Smem budget at BLOCK_M=32: 4 x (32 KB + 16 KB) = 192 KB; confirm from the run log that Triton kept 4 stages (and did not fail to compile); 'raise num_stages to 6' in the Risk section is not possible at this tile.
- Harvest criterion: the headline will straddle 1.0 even on success (~+0.9% geomean ceiling for 2 of 30 shapes) and is further depressed by the inherited pooler decode_32 loss; judge on the unpooler decode_16/32 rows versus the parent's result rows (18.71 / 18.63 us) and record 67.1 MB / kernel us in FINDINGS as the second datapoint on the plain-Triton streaming ceiling.
- If the kernel us is below nvjet's 18.4/18.8 but the reported median is not, the loss is Triton launch-path overhead inside the graph, not bandwidth; separate those two readings before withdrawing the o- and qkv-gate siblings.


## Implementation notes

- File: `muse_bf16_proj/submission.py` in the worktree (only file touched; 53 insertions, 2 deletions). Built on d2840d0 (the parent's impl), so the pooler path is byte-identical.
- Reuses the parent's `@triton.jit _pooler_skinny_kernel` unchanged (K, BLOCK_M, BLOCK_N, BLOCK_K are constexpr, so the unpooler call is a separate specialization; M stays unspecialized).
- Added `UNPOOLER_*` constants: N=8192, K=4096, BLOCK_N=64 (128 programs, one wave), BLOCK_K=256 (16 K-steps), num_warps=4, num_stages=4; routed m range 9..32.
- Added `_unpooler_skinny_applies(x, cfg)`: true only for 9 <= cfg m <= 32, cfg n == 8192, cfg k == 4096, plus x 2-D bf16, x.shape == (m, k), x.stride(1) == 1. Keys on cfg, never on x.shape[0] alone. Therefore m=1/8 (n=1024 slice), the m=5 test, and m=128/130/256/1000 stay on `_bf16_gemm_dispatch_impl` (cuBLAS).
- Added `_unpooler_skinny_gemm(x, w)`: BLOCK_M = 16 if m <= 16 else 32 (BLOCK_M=32 at m=32, as specified; the BLOCK_M=16-everywhere control is the queued companion and is NOT folded in), fp32 accumulate in the kernel, bf16 store into `torch.empty((m, 8192), bf16)`.
- `prepare()`: sets `state["unpooler_skinny"]` from the weight's shape (weights-derived, not data-derived) and, when true, `_unpooler_skinny_warmup` runs every m in 9..32 on synthetic zeros so both specializations compile before graph capture and any compile/launch failure surfaces in prepare().
- `kernel_fn`: one new branch after the pooler branch: `layer == "unpooler" and state["unpooler_skinny"] and _unpooler_skinny_applies(x, cfg)`.
- Smem risk: at BLOCK_M=32, 4 stages x (64x256 + 32x256) x 2 B = 192 KB per program (160 KB at BLOCK_M=16), near the H200's 227 KB per-block limit. If Triton's pipeliner needs more than that (e.g. extra layout-conversion buffers), the BLOCK_M=32 specialization raises OutOfResources during warmup in prepare() (the run fails visibly rather than at capture) — check the run log. num_stages=6 is not possible at this tile (288 KB). No fallback was added, to keep the run attributable.
- Open question from review (what BLOCK_N=64 / 16 K-steps changes vs the parent's 20.2/25.2 us): the one real difference is 2x the weight bytes in flight per SM (128 KB of weight per program across 4 stages vs 64 KB), which may raise per-SM memory-level parallelism; fewer K-steps make the pipeline prologue/epilogue a larger fraction. The parent's data predict flat-to-loss at m=16 and a loss at m=32; judge per row against the parent's rows (18.71 / 18.63 us) and record kernel us.
- Not run (no GPU/torch/Triton on this node); only py_compile and the lint below.

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the diff; the new branch depends only on cfg, x shape/stride/dtype, and a prepare()-time bool; the only allocation is the fixed-shape `torch.empty((m, 8192))` output, same as the parent's pooler path.
- [x] Nothing derived from `data` is cached across calls: state gains only `unpooler_skinny` (from the weight shape in prepare()); warmup uses synthetic zeros and drops outputs.
- [x] Weights: unpooler used as stored (bf16, contiguous, no repack); router hi+lo split unchanged from parent.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16 into tl.dot, fp32 accumulator, bf16 output. KV cache / FP8 query: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention in this problem; the GEMM reads the full K and every weight row.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only constants, two helpers, a warmup and one branch.
- [x] All work happens inside the timed call; no output caching or replay: the output is computed fresh each call by the Triton kernel.
- [x] No triton.autotune (fixed config), first-line marker `# okc: problem=muse_bf16_proj` kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` OK; worktree __pycache__ removed.
