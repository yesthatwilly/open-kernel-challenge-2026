---
problem: muse_bf16_proj
slug: unpooler-full-m16-swapab-route
created: 2026-10-25T003858Z
base: master
area: unpooler GEMM
change: >
  Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67
  MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (cuBLAS nvjet_sm90_tst_64x16, 19.3
  us kernel, row 18.5 at decode_16), while at m 17..32 it takes the swap-AB Triton kernel
  _bf16_swapab_kernel (SWAPAB_CFG unpooler: SPLIT_K 1, BLOCK_K 256, 4 stages, grid 128 CTAs; 18.7
  us kernel at m=32, row 18.1). bf16-m16-swapab-wgmma-extend covered pooler/qkv/o/slice at m<=16
  but never the full unpooler. Change exactly one gate: add a per-layer lower bound,
  SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9} (others keep SWAPAB_MIN_M = 17), used by
  _swapab_route_ok, so unpooler with cfg n == 8192 at 9..16 takes the same swap-AB kernel with the
  same constexprs (BLOCK_T 32, token columns >= m masked; M is not specialized so no new compile).
  In _swapab_prepare extend the unpooler warm-up range to 9..32. The slice (cfg n == 1024, m <= 8)
  is still caught earlier by _unpool_shape_ok and is unaffected; m=130 and m=1000 tests
  unaffected. No other layer, constant or kernel changes.
expected:
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: nvjet 64x16 19.3 us kernel (row 18.5) -> the swap-AB kernel that runs 18.7 us at m=32 and should be no slower at m=16 (same bytes, fewer token columns); floor 14.1
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: same path as master
  - shapes: layer=unpooler load=decode_1 m=1
    direction: flat
    why: slice path untouched
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: slice path untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T004113Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T004113Z
base_sha: e024827fbfe10f55e792e5540c00390bc0df3798
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/unpooler-full-m16-swapab-route
worktree: .worktrees/muse_bf16_proj/unpooler-full-m16-swapab-route
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T004327Z
impl_commit: 9e2225c62f8220226eaabc2502e317e51f0157b5
submission_sha256: 29fe6c94af0f444942b0674b9b8fa5946e611c9251162a858682a42a4e6e4739
norm_sha256: 4cb5554e9610443db7e595eb3b753beb986976d2fbbff54f40ef25a08d6896b7
submission_blob: fa481a079056997fb3dc6ae5746193edccdcb297
lint_warnings: 0
attempt_file: 2026-10-25T004327Z-check-unpooler-full-m16-swapab-route.md
prepared_at: 2026-10-25T004327Z
job: 05782267384343eb
dispatched_at: 2026-10-25T004351Z
outcome: regression
verdict_detail: regression
speedup: 1.1884519539056506
speedup_ci: [1.170625174597066, 1.2062787332142353]
harvested_at: 2026-10-25T034131Z
best_at_harvest: 1.2215006063849365
promotion: null
---

# unpooler-full-m16-swapab-route

## Change

Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (cuBLAS nvjet_sm90_tst_64x16, 19.3 us kernel, row 18.5 at decode_16), while at m 17..32 it takes the swap-AB Triton kernel _bf16_swapab_kernel (SWAPAB_CFG unpooler: SPLIT_K 1, BLOCK_K 256, 4 stages, grid 128 CTAs; 18.7 us kernel at m=32, row 18.1). bf16-m16-swapab-wgmma-extend covered pooler/qkv/o/slice at m<=16 but never the full unpooler. Change exactly one gate: add a per-layer lower bound, SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9} (others keep SWAPAB_MIN_M = 17), used by _swapab_route_ok, so unpooler with cfg n == 8192 at 9..16 takes the same swap-AB kernel with the same constexprs (BLOCK_T 32, token columns >= m masked; M is not specialized so no new compile). In _swapab_prepare extend the unpooler warm-up range to 9..32. The slice (cfg n == 1024, m <= 8) is still caught earlier by _unpool_shape_ok and is unaffected; m=130 and m=1000 tests unaffected. No other layer, constant or kernel changes.

## Expected

- `layer=unpooler load=decode_16 m=16` faster: nvjet 64x16 19.3 us kernel (row 18.5) -> the swap-AB kernel that runs 18.7 us at m=32 and should be no slower at m=16 (same bytes, fewer token columns); floor 14.1
- `layer=unpooler load=decode_32 m=32` flat: same path as master
- `layer=unpooler load=decode_1 m=1` flat: slice path untouched
- `layer=unpooler load=verify_1 m=8` flat: slice path untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Small, one-shape change (~0.5-1 us on one 62-weight shape, ~+1% on decode_16); attributable from the unpooler decode_16 row alone. Risk: swap-AB at m=16 wastes half the wgmma n32 columns, but the kernel is weight-stream-bound so that is free; if the row is not below 18.5 the gate goes back.

## Adversarial review

Reviewed 2026-10-25T004113Z by an adversarial subagent (model opus).

**Verdict: proceed.** One-gate change moving the full-width unpooler at m=9..16 from cuBLAS nvjet 64x16 onto master's existing swap-AB kernel. Plausible and attributable from the unpooler decode_16 row and its profile, but small (~0.2-0.6 us on one row, <0.2% headline). No overlap with the pending restack, which leaves the full unpooler at m=16 on cuBLAS; they touch different shapes and code and would stack cleanly.

### Justification
On master the unpooler n=8192 at m=16 fails _unpool_shape_ok (needs n==1024) and _swapab_route_ok (needs m>=17), so it falls to _bf16_gemm_dispatch_impl: nvjet_sm90_tst_64x16 at 19.3 us against a 14.1 us floor (pure 67 MB weight stream). The m=32 swap-AB config (128 CTAs, BLOCK_K 256, 4 stages, wgmma m64n32) with token columns masked should cost about its m=32 time; M is do_not_specialize, so no new compile. Evidence: swap-AB unpooler at m=32 18.48 us vs SGLang 18.66 (job 1602a4e), 18.08 in the restack job; the older mma.sync skinny kernel ran decode_16 at 17.97 vs 18.56. Weakest link: swap-AB beat cuBLAS by only 1-3% at m=32, and at m=16 cuBLAS already uses its tight 64x16 tile. Judge against master's own row, 18.61 us (job 1602a4e), not the restack's numbers.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (landed): the dependency, supplies the kernel and gate; not a confounder.
- bf16-shape-gated-triton-winners-combo (landed): _unpool_shape_ok needs n==1024, separate UNPOOL_MAX_M constant; not a confounder.
- router-triton-hilo-fused-control-splitk-all-m / router-hilo-at-m8 / sglang-gemm-dispatch-parity: other layers; not confounders.

### Interference and A/B needs
No companion. Read the unpooler decode_16 row (err ~50 ns, a 300-600 ns change is resolvable) and the profile's kernel name (nvjet -> _bf16_swapab_kernel). The headline will read unproven-vs-best (base is master ~1.198, BEST is the restack 1.2215); grade on the row against master job 1602a4e. Restack overlap: none in shapes or logic, text hunks disjoint; a 3-way merge should be clean and the gains add.

### Open questions
- Will BLOCK_T 32 at m=16 match the m=32 time? If it lands ~18.4 us, a follow-up on top of the restack is a SWAPAB16_CFG['unpooler'] full-width entry (8192, 4096, 1, 256, 4) with wgmma m64n16.
- If the restack is promoted first, rebase onto it and re-judge decode_16 against its job 60b20fd0 unpooler row (18.51 us).


## Implementation notes

Edited only muse_bf16_proj/submission.py in the worktree (+7/-3). Added SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9}; _swapab_route_ok uses SWAPAB_MIN_M_BY_LAYER.get(layer, SWAPAB_MIN_M) as its lower bound; _swapab_prepare's warm loop now uses the same per-layer minimum, so the unpooler is warmed at m 9..31 (m=32 by the first warm call), others stay 17..31. Per the review, the minimum drives the warm loop rather than a hardcoded 9..32. No deviation from the spec. The slice (_unpool_shape_ok, n=1024) is checked earlier in kernel_fn and is unaffected.

## Rules check

- [x] CUDA-graph capturable: diff only changes a constant lookup on cfg["m"]/layer; no sync, allocation or value-dependent control flow in kernel_fn.
- [x] Nothing derived from data is cached across calls: warm-up uses synthetic randn x in prepare only.
- [x] Weights: unchanged; BF16 stays BF16, no repack.
- [x] Activations BF16: unchanged (route requires x bf16).
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint output:
```
CLEAN: muse_bf16_proj/submission.py
```

## Result

- job `05782267384343eb` (check), status done, passed True, tests 9/9
- speedup 1.1885x, 95% CI [1.1706, 1.2063], score 4064.7835902195084 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/unpooler-full-m16-swapab-route.result.json`, `muse_bf16_proj/hypotheses/done/regression/unpooler-full-m16-swapab-route.patch`; branch `hyp/muse_bf16_proj/unpooler-full-m16-swapab-route`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.18 | 13.26 | 0.925 | lost |
| layer=o load=decode_1 m=1 | 62 | 5.89 | 6.41 | 0.9271 | lost |
| layer=pooler load=decode_1 m=1 | 62 | 18.84 | 19.42 | 0.9551 | lost |
| layer=unpooler load=decode_1 m=1 | 62 | 6.15 | 5.92 | 0.9796 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.38 | 5.89 | 0.9177 | lost |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.15 | 13.91 | 0.9562 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.23 | 6.52 | 0.9683 | lost |
| layer=pooler load=decode_16 m=16 | 62 | 19.47 | 19.86 | 0.9602 | lost |
| layer=unpooler load=decode_16 m=16 | 62 | 18.51 | 18.48 | 1.0129 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.47 | 6.14 | 0.9072 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 14.0 | 1.01 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.47 | 1.0046 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 19.73 | 20.05 | 0.9719 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.08 | 18.2 | 0.9934 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.2 | 6.54 | 0.9277 | lost |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.5 | 13.49 | 0.9343 | lost |
| layer=o load=verify_1 m=8 | 62 | 6.13 | 6.71 | 0.9268 | lost |
| layer=pooler load=verify_1 m=8 | 62 | 18.89 | 19.45 | 0.9515 | lost |
| layer=unpooler load=verify_1 m=8 | 62 | 6.0 | 6.16 | 0.9694 | lost |
| layer=router load=verify_1 m=8 | 62 | 5.47 | 6.05 | 0.9048 | lost |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.05 | 16.99 | 1.0127 | flat |
| layer=o load=verify_16 m=128 | 62 | 6.98 | 7.55 | 0.9805 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.0 | 25.23 | 1.006 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.96 | 20.06 | 1.0085 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.28 | 7.28 | 0.9801 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 19.81 | 1.0066 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.26 | 9.5 | 0.9754 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.19 | 29.84 | 0.9976 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.35 | 0.9919 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.6 | 9.14 | 1.038 | gained |

gained on router verify_32; lost on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; pooler decode_1, decode_16, verify_1; router decode_1, decode_16, decode_32, verify_1; unpooler verify_1. Expected hit: 5/8. Noise ±1.5%.

## Follow-ups

- record which shapes lost (Attribution) and do not retry this area in this form.

