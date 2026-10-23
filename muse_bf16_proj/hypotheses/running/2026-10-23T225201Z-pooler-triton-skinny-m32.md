---
problem: muse_bf16_proj
slug: pooler-triton-skinny-m32
created: 2026-10-23T225201Z
base: master
area: pooler GEMM
change: >
  For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn
  with one Triton kernel, a split-K-free skinny GEMM that streams the 67 MB weight once: grid =
  (N/BLOCK_N,) with BLOCK_N=32 so 4096/32 = 128 programs (one wave on 132 SMs); each program owns
  32 weight rows and loops the full K=8192 in BLOCK_K=256 steps with num_stages=4 (tl.load of a
  [BLOCK_N, BLOCK_K] bf16 weight tile from the row-major [N, K] weight as stored, and of the x
  tile [BLOCK_M, BLOCK_K] with BLOCK_M=16 or 32 from m, rows >= m masked to zero), acc +=
  tl.dot(x, w.T) in fp32, then a masked bf16 store of [m, 32] into a torch.empty((m, 4096), bf16)
  output. prepare() stays as in the parity base (no repack). All other layers and pooler at m >=
  128 keep the cuBLAS dispatch. Kernels moved: nvjet_sm90_tst_128x8/16/32_..._splitK_TNT
  (18.8-20.2 us) plus cublasLt::splitKreduce_kernel (1.6-1.9 us) become one launch; the reduce
  launch and its graph-node gap go away and the weight tile reads are fully sequential per
  program. The question the run answers: can a plain Triton full-K stream hit >= 4.0 TB/s (67 MB
  in <= 17 us) where nvjet splitK sits at 3.6 TB/s.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: 21.5 us (18.8 nvjet + 1.6 reduce + gap) vs 14.0 floor; target <= 18 us
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: same shape class, 21.5 us, BLOCK_M=16
  - shapes: layer=pooler load=decode_32 *
    direction: faster
    why: same, 22.1 us, BLOCK_M=32
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: m=8, same as m=1 path with 8 live rows
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
  - shapes: "*layer=unpooler*"
    direction: flat
    why: untouched layer; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-23T230544Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-23T230545Z
base_sha: e64f27ef09b3512d26573dde4d35b8ec3ade1f63
base_blob: 5a2064af06f442f01dff926cb4f1eca75f992a24
branch: hyp/muse_bf16_proj/pooler-triton-skinny-m32
worktree: .worktrees/muse_bf16_proj/pooler-triton-skinny-m32
mode: check
implemented_at: 2026-10-23T231157Z
impl_commit: d2840d088aa35522a03b40c59e174287cf935f67
submission_sha256: 4b6c05d0d794df548e24c23a8b2f1277ad050bc2af57165f099eb5325a52da22
norm_sha256: 0a52fd3c358414044b2d1a31157de5a14a194a54644b1ea69d81491f90da5b75
submission_blob: 2752a016d386e9261ec77e25b426c05b76c071ac
lint_warnings: 0
attempt_file: 2026-10-23T231158Z-check-pooler-triton-skinny-m32.md
prepared_at: 2026-10-23T231158Z
job: 9f0ad00bbc874d31
dispatched_at: 2026-10-23T231206Z
---

# pooler-triton-skinny-m32

## Change

For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams the 67 MB weight once: grid = (N/BLOCK_N,) with BLOCK_N=32 so 4096/32 = 128 programs (one wave on 132 SMs); each program owns 32 weight rows and loops the full K=8192 in BLOCK_K=256 steps with num_stages=4 (tl.load of a [BLOCK_N, BLOCK_K] bf16 weight tile from the row-major [N, K] weight as stored, and of the x tile [BLOCK_M, BLOCK_K] with BLOCK_M=16 or 32 from m, rows >= m masked to zero), acc += tl.dot(x, w.T) in fp32, then a masked bf16 store of [m, 32] into a torch.empty((m, 4096), bf16) output. prepare() stays as in the parity base (no repack). All other layers and pooler at m >= 128 keep the cuBLAS dispatch. Kernels moved: nvjet_sm90_tst_128x8/16/32_..._splitK_TNT (18.8-20.2 us) plus cublasLt::splitKreduce_kernel (1.6-1.9 us) become one launch; the reduce launch and its graph-node gap go away and the weight tile reads are fully sequential per program. The question the run answers: can a plain Triton full-K stream hit >= 4.0 TB/s (67 MB in <= 17 us) where nvjet splitK sits at 3.6 TB/s.

## Expected

- `layer=pooler load=decode_1 *` faster: 21.5 us (18.8 nvjet + 1.6 reduce + gap) vs 14.0 floor; target <= 18 us
- `layer=pooler load=decode_16 *` faster: same shape class, 21.5 us, BLOCK_M=16
- `layer=pooler load=decode_32 *` faster: same, 22.1 us, BLOCK_M=32
- `layer=pooler load=verify_1 *` faster: m=8, same as m=1 path with 8 live rows
- `layer=pooler load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=unpooler*` flat: untouched layer; control

## Risk

Risk: a full-K loop has only num_stages x BLOCK_N x BLOCK_K x 2 = 64 KB in flight per SM; if that is not enough memory-level parallelism to beat nvjet's splitK (which uses 4x more CTAs), the pooler shapes get slower, not faster, and the per-shape rows will show it directly (nothing else moves). Correctness: bf16 inputs, fp32 accumulate, one bf16 rounding at the store, same as cuBLAS; the m=7 pooler test exercises the masked-row path. CUDA graph: fixed-shape torch.empty only, no syncs. Compile: 3 BLOCK_M specializations, seconds. If faster, qkv-gate-triton-skinny-m32, unpooler-full-triton-skinny-m32, o-triton-skinny-m32 and pooler-skinny-tiled-weight-layout stack on it; if slower, they should be withdrawn or retuned (BLOCK_N=16 with 256 programs, or num_stages=6) before running.

## Adversarial review

Reviewed 2026-10-23T230544Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A single run is interpretable: only the four pooler m<=32 rows can move, nothing landed touches that path, and the profile's per-kernel us answers the stated question (achieved TB/s of a plain Triton full-K stream) whether the shapes gain or lose. The headroom is thinner than the hypothesis states: the ~1 us graph gap exists only at m=1; at m=16/32 the reduce kernel (1.8-1.9 us) is the only removable cost, so the Triton kernel must at least tie nvjet's 19.8-20.2 us there.

### Justification
Mechanism is plausible: pooler is a 67.1 MB bf16 weight stream (4096x8192), memory-bound at every m<=32, floor 14.0-14.1 us at 4.8 TB/s; the current path is nvjet splitK (18.8 us at m=1 = 3.57 TB/s; 19.8/20.2 us at m=16/32 = 3.3-3.4 TB/s) plus a cublasLt splitKreduce launch (1.6-1.9 us). Fusing to one kernel removes the reduce and its gap, and 128 programs x 4 stages x 16 KB = 8 MB in flight satisfies Little's law for 4.8 TB/s at ~1 us loaded latency, so bandwidth is not latency-bound on paper. Shape/direction consistency with TASK.md is fine: all shapes weigh 62, the four 'faster' globs are exactly the m<=32 pooler shapes, verify_16 (m=128) and unpooler are correct flat controls; pooler verify_32 (m=256, cuBLAS) is missing from the expected list and should be added as flat. Load-level arithmetic: pooler is 28-36% of four of the six loads (decode_1/16/32, verify_1); a full success (~21.4 -> ~19 us, -11%) is ~+3-4% per load and ~+2.3% geomean, right at the judge's +-1.5% CI edge, so the per-shape rows and the profile, not the headline speedup, are the readout. Weakest link: the hypothesis's own arithmetic '21.5 = 18.8 + 1.6 + gap' holds only at m=1 (profile kernel sum 20.3 vs 21.2-21.45 median); at m=16 and m=32 the kernel sum equals the median, so the removable cost is 1.8-1.9 us (8-9%), and a Triton pointer-load kernel with one 4-warp CTA per SM has no supporting measurement that it reaches even nvjet's 3.3-3.6 TB/s, let alone the 4.0 TB/s target. If the Triton stream lands at 3.2 TB/s (21 us) every pooler shape loses despite the fused launch. Second-order: each program re-streams the whole x tile from L2 every K step (for m=32: 128 x 512 KB = 64 MB of L2 traffic, comparable to the HBM weight bytes), which competes for the same issue slots; nvjet pays this too, but it is why 1 CTA/SM is a thin margin.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity (parent, landed list on master is empty): no confound. Its pooler path is kernel-for-kernel the starter's (same nvjet splitK + splitKreduce, every pooler row flat 0.99-1.02x in its attribution), and its router hi/lo change is a different layer timed in its own CUDA graph. This hypothesis keeps _bf16_gemm_dispatch_impl only for pooler m>=128, which is untouched by the parent too. The change would behave identically with the parent absent (the starter's x @ w.T picks the same nvjet kernels).

### Interference and A/B needs
No companion is warranted. A gain is unambiguous: the only code path that changes is pooler at m<=32, the judge reports per shape, and nothing else in the graph for those shapes exists besides the one Triton launch and the torch.empty. A loss is interpretable at the mechanism level from the profile: the Triton kernel's own us gives 67.1 MB / t directly, which is the question the hypothesis poses (>= 4.0 TB/s or not); the total minus the kernel us gives the launch/gap cost. ab-revert is inapplicable (nothing landed that touches this path; reverting the parent would only change the router). ab-tune of the parent has no tunable that affects the pooler. A tile-config control (BLOCK_N=16/256 programs, num_warps=8, num_stages=6, or TMA descriptors) is a follow-on retune to queue only if this loses, not a disambiguator; the four dependents (qkv-gate, unpooler-full, o, pooler-skinny-tiled-weight-layout) are already gated by depends_on and should stay held until the achieved TB/s is recorded in FINDINGS.

### Open questions
- Record the achieved bandwidth from the profile (67.1 MB / Triton kernel us) in FINDINGS as the plain-tl.load skinny-GEMM ceiling on H200; the four dependents and the whole skinny-Triton family stand or fall on that one number.
- Expected list omits pooler verify_32 (m=256, keeps cuBLAS); add it as a flat control so the attribution grid is complete.
- num_warps is unspecified (Triton default 4): with one CTA per SM that is the entire issue capacity for 16 KB of cp.async per stage; state it in the change (4 vs 8) so a loss can be retuned on one axis.
- '3 BLOCK_M specializations' is really 2 (m=1/7/8/16 -> 16, m=32 -> 32); consider BLOCK_M=16 for every m<=32 to cut compile and keep one kernel, since compute is irrelevant at these shapes.
- If the per-shape rows lose while the Triton kernel's own us is below 18.8 (m=1) / 19.8 (m=16) / 20.2 (m=32), the loss is launch/gap overhead of the Triton launch path, not bandwidth; separate those two readings before withdrawing the dependents.
- At m=16/32 nvjet is slower than at m=1 (19.8-20.2 vs 18.8 us), so the bar is lower there; a Triton kernel flat across m at ~18.8 us would gain 1.10-1.15x on decode_16/32 even with zero bandwidth improvement.

## Implementation notes

Worktree `.worktrees/muse_bf16_proj/pooler-triton-skinny-m32`, one file changed: `muse_bf16_proj/submission.py` (+94/-7; first line unchanged).

- `kernel_fn`: a new first branch, `layer == "pooler" and _pooler_skinny_applies(x, cfg)`, calls `_pooler_skinny_gemm(x, state["pooler"])`. The guard is config/shape only: `cfg m <= 32`, `cfg n == 4096`, `cfg k == 8192`, `x` 2-D of shape `(m, k)`, bf16, unit inner stride. Anything else (pooler at m > 32, any other n/k, qkv_gate, o, unpooler, router) falls through to the untouched parity code (`_bf16_gemm_dispatch_impl` / router hi+lo).
- `_pooler_skinny_kernel` (`@triton.jit(do_not_specialize=["M"])`): grid `(4096 // 32,) = (128,)`, one program per 32 output columns, so one wave on 132 SMs. Tiles `BLOCK_N=32`, `BLOCK_K=256`, `BLOCK_M=16` for m <= 16 and `BLOCK_M=32` for 17 <= m <= 32; `K=8192` is a constexpr so the loop is a static 32 steps. Each step `tl.load`s the `[32, 256]` bf16 weight tile straight from the row-major `[N, K]` weight as stored (unmasked: n and k divide exactly, enforced by the guard) and the `[BLOCK_M, 256]` x tile with rows >= M masked to zero (`other=0.0`), then `acc = tl.dot(x_tile, tl.trans(w_tile), acc)` in fp32. After the loop a single `acc.to(tl.bfloat16)` store masked to rows < M into a `torch.empty((m, 4096), bf16)`. **`num_warps=4`, `num_stages=4`** (per stage 16 KB weight + 8/16 KB x; 96/128 KB smem at 4 stages, 64 KB of weight in flight per SM as the hypothesis states).
- `M` is passed as a runtime int with `do_not_specialize`, so Triton does not mint an `M == 1` or divisible-by-16 variant: exactly two binaries exist (BLOCK_M 16 and 32). Strides are runtime ints (8192 / 8192 / 4096, all 16-divisible, identical between warm-up and the judge's contiguous inputs).
- `prepare()`: unchanged parity body (contiguous bf16 weights, router hi/lo split, no repack), plus `_pooler_skinny_warmup(state["pooler"])`, which launches the kernel on synthetic `torch.zeros((m, 8192), bf16)` for every m in 1..32, so both specializations (and any int specialization Triton might still apply) are compiled before the first timed call; outputs are dropped and nothing derived from inputs is retained.
- Deviations from the spec / review: (1) the reviewer's "consider BLOCK_M=16 for every m <= 32" was not taken: `tl.dot` needs M >= 16, so m = 32 at BLOCK_M=16 would need either a second M tile (a second pass over the 67 MB weight) or a second dot per step; two specializations is the smaller change and both are pre-warmed. (2) `pooler verify_32` (m = 256) is a flat control like verify_16; it stays on cuBLAS by the m <= 32 guard. No autotune; no TMA; no tf32/fp8 anywhere.
- Untested on hardware (no GPU/Triton on this node): syntax checked with `ast.parse`, lint CLEAN. The one Triton-specific thing to watch in the run log is the `tl.trans(w_tile)` operand feeding `tl.dot` inside the pipelined loop; it is the standard pattern (`tl.dot(q, tl.trans(k))` in the Triton attention tutorial) but has not been compiled here.

## Rules check

Verified against `git diff` in the worktree (`muse_bf16_proj/submission.py`, +94/-7):

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints anywhere in the diff; the only allocation is `torch.empty((m, 4096))` from `x.shape`; all branching (`_pooler_skinny_applies`, `block_m`) is on `cfg` fields and `x.shape`/`stride`/`dtype`, never on values.
- [x] Nothing derived from `data` is cached across calls: `kernel_fn` writes no module-level state; `out` is fresh per call and returned; the warm-up tensors in `prepare()` are synthetic zeros that are not kept.
- [x] Weights: `prepare()` still only does `.contiguous()` on the bf16 weights (no repack, no dtype change); the Triton kernel reads `state["pooler"]` as stored; the router hi+lo split is the pre-existing parity code.
- [x] Activations BF16: `x` is loaded as bf16, `tl.dot` is bf16 x bf16 with an fp32 accumulator, the output is rounded to bf16 once at the store, the same chain cuBLAS runs; no tf32/fp8 casts in the diff.
- [x] Attention sees every key: not applicable, this problem is a dense GEMM; the diff computes the full `x @ W.T` over all 8192 K and all 4096 N (no skipping).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: imports are `torch`, `torch.nn.functional`, `triton`, `triton.language`, and the same `sglang` dispatch as before; lint CLEAN.
- [x] All work happens inside the timed call; no output caching or replay: `prepare()` only compiles the kernel on synthetic inputs (allowed untimed setup) and discards the result; every timed call streams the weight and writes a new output.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj` -> `CLEAN: muse_bf16_proj/submission.py`
