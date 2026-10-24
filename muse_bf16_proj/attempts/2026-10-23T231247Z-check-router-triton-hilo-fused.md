# muse_bf16_proj · 2026-10-23T231247Z · check · router-triton-hilo-fused

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225128Z-router-triton-hilo-fused.md` (branch `hyp/muse_bf16_proj/router-triton-hilo-fused`).

- **Job:** `c2a8de5969374dff` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `a392f0b`

## Goal
Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch at every m. prepare() keeps the hi/lo split but stores router_hi and router_lo as [256, 8192] bf16 row-major (N x K, not transposed). The kernel: grid = (N/BLOCK_N, SPLIT_K); each program loads an x tile [BLOCK_M, BLOCK_K] of bf16 (BLOCK_M=16 for m<=16 with masked rows, else 32/64/128 chosen from m at launch; M is looped in BLOCK_M chunks for m=256), loads the hi and lo weight tiles [BLOCK_N, BLOCK_K] for its K-slice, and accumulates acc += tl.dot(x, hi.T) + tl.dot(x, lo.T) in fp32 over its K range (BLOCK_K=128, num_stages=3); with SPLIT_K>1 it tl.atomic_add's the fp32 partial into an output that kernel_fn zeroes with torch.zeros((m, 256), fp32) just before the launch (SPLIT_K=8 and BLOCK_N=64 for m<=32 gives 32 programs x 8 = 256 CTAs; SPLIT_K=1 with a plain store for m>=128). The output is the fp32 [m, 256] tensor, dtype and shape as before. Kernels moved: at m<=8 the dot_kernel + x.float() cast + reduce_1Block (m=1) or the two xmma f32 SIMT GEMMs + cast (m=8) become one launch (plus one memset); at m>8 the 2 nvjet + 2 splitKreduce + 1 add become one launch (plus one memset). The weight read stays 8.4 MB (hi + lo bf16) which at 4.8 TB/s is the 1.8 us floor.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/router-triton-hilo-fused`, file `muse_bf16_proj/submission.py` only (diff stat: 1 file, +143/-19). Base `e64f27e` (sglang-gemm-dispatch-parity).

What changed:
- `prepare()`: still builds `hi = router.to(bf16)`, `lo = (router - hi.float()).to(bf16)` from the weights alone, but stores them as `state["router_hi"]` / `state["router_lo"]`, each `[256, 8192]` bf16 row-major (N x K). The transposed `router_hi_t` / `router_lo_t` are gone. The fp32 `state["router"]` is kept (it is the original weight) only for a shape-fallback that this problem's router never takes. prepare() then pre-warms the Triton kernel on synthetic `torch.randn` bf16 x for `m in (1, 8, 16, 32, 64, 128, 256)`, one launch each (no synchronize; prepare is untimed). `M` is declared `do_not_specialize`, so one compile serves a whole m-bucket: 4 real compiles, the other warm launches are cache hits. Unlisted m (7, 9, 33, 130, ...) land in an already-compiled bucket.
- `kernel_fn()`: bf16 layers unchanged (`_bf16_gemm_dispatch_impl(x, state[layer][:n], None)`). The whole router branch (fp32 `F.linear` at m<=8 and the two `torch.mm` hi/lo GEMMs + add above) is replaced by `_router_forward(x, hi, lo)`: one Triton launch at every m, plus one memset when SPLIT_K > 1. Guard `_router_shape_ok` branches only on shapes/dtype/strides (k multiple of BLOCK_K*SPLIT_K, n multiple of BLOCK_N, x row-major bf16); if it ever fails (not for this problem's [*, 8192] -> 256 router) the exact fp32 `F.linear` path runs instead.
- `_router_forward()`: picks the static config from m, creates the output **inside the call, right before the launch** (`torch.zeros((m, 256), fp32)` for SPLIT_K > 1, `torch.empty` for SPLIT_K == 1; never in state), launches `grid = (256 // BLOCK_N, SPLIT_K)` exactly as this hypothesis's change paragraph specifies (2-D; the 3-D split-K grid is the queued control's job), returns the fp32 [m, 256] tensor.
- `_router_hilo_kernel` (`@triton.jit(do_not_specialize=["M"])`): program (pid_n, pid_k) owns one N-block and one K-slice of `K_SLICE = K // SPLIT_K` (constexpr) and loops `for m_start in range(0, M, BLOCK_M)` over all of M. Per BLOCK_K=128 step it loads the x tile `[BLOCK_M, BLOCK_K]` (rows >= M masked to 0), the hi and lo tiles read directly as `[BLOCK_K, BLOCK_N]` (K contiguous, i.e. hi.T / lo.T for the tile, so no explicit `tl.trans`), and does `acc = tl.dot(x, hi, acc); acc = tl.dot(x, lo, acc)` in fp32. Epilogue: `tl.atomic_add(out, acc, mask=rows<M, sem="relaxed")` when SPLIT_K > 1, else `tl.store`. `num_warps=4`, `num_stages=3`, `BLOCK_K=128` for every config.

Per-m config table (grid = (256 // BLOCK_N, SPLIT_K); actual CTA counts, not the 256 the change paragraph claims):

| m | BLOCK_M | BLOCK_N | SPLIT_K | K-slice | grid | CTAs | M chunks | epilogue | smem/stage x 3 |
|---|---|---|---|---|---|---|---|---|---|
| 1..16 (1, 7, 8, 9, 16) | 16 | 64 | 8 | 1024 | (4, 8) | 32 | 1 | atomic_add into zeros | 36 KB x 3 |
| 17..32 (32) | 32 | 64 | 8 | 1024 | (4, 8) | 32 | 1 | atomic_add into zeros | 40 KB x 3 |
| 33..127 (33, 64) | 64 | 64 | 8 | 1024 | (4, 8) | 32 | 1-2 | atomic_add into zeros | 48 KB x 3 |
| >=128 (128, 130, 256, 512) | 128 | 32 | 1 | 8192 | (8, 1) | 8 | cdiv(m, 128) = 1 / 2 / 4 | plain store into empty | 48 KB x 3 |

Deviations from the change paragraph, and why:
- CTA count: the paragraph says "32 programs x 8 = 256 CTAs" for m<=32; the specified grid (N/BLOCK_N, SPLIT_K) = (4, 8) is 32 CTAs, as the reviewer noted. Built as specified (32 CTAs); the 256-CTA claim is simply wrong and the control hypothesis carries the filled grid.
- 33 <= m <= 127 is not covered by the paragraph (it gives m<=32 and m>=128). Filled with BLOCK_M=64, SPLIT_K=8 atomics (32 CTAs) rather than the 8-CTA SPLIT_K=1 path, so m=33/64 nearby-shape reruns are not pathologically slow.
- m>=128 uses BLOCK_N=32 (8 CTAs) rather than 64 (4 CTAs): the paragraph fixes BLOCK_N=64 only for m<=32 and leaves it open at m>=128; 32 doubles the CTAs within the reviewer's "4-8" range and keeps the 3-stage pipeline at 144 KB shared memory instead of 192 KB. Still a deliberately under-filled grid; that is the A/B.
- No `tl.trans`: the weight tile is addressed as [BLOCK_K, BLOCK_N] from the [N, K] row-major tensor, which is the same memory read and the same dot.
- `sem="relaxed"` on the atomic: there is no ordering requirement (the memset precedes the kernel in stream order). Requires Triton >= 3.0, which the SGLang dev image has.
- Warm-up set is (1, 8, 16, 32, 64, 128, 256), covering the 4 buckets plus every scored and tested m; with `do_not_specialize=["M"]` any other m reuses a bucket's compile (strides are shape constants, 8192/8192/256, identical between warm-up and timed calls).

Not verified here: no GPU and no Triton on this node, so the kernel was not run. Checked: `python3 -c "import ast; ast.parse(...)"` ok; lint CLEAN; `_router_config` enumerated for m in {1,7,8,9,16,32,33,64,127,128,130,256,512}. Things to watch in the run: (a) atomic + memset cost at m<=32 (look for a memset/fill entry in the profile); (b) the m>=128 rows, expected slower than cuBLAS splitK by the reviewer's arithmetic; (c) shared-memory or pipelining complaints from Triton at BLOCK_M=128 x BLOCK_K=128 x 3 stages (144 KB) if the image's Triton reserves more than expected; (d) probe-row error at m=1/8 under hi/lo (expected ~1.5e-5, bound 1e-3).

## Rules check (each must be true before the run)
Verified against `git diff` in the worktree (`.worktrees/muse_bf16_proj/router-triton-hilo-fused`, `git diff --stat`: `muse_bf16_proj/submission.py | 162 ++++---`, 1 file, +143/-19).

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no allocation or control flow on input values (shapes and `config` are fine). Grep of the added lines for `.item(`, `.cpu(`, `synchronize`, `print(` is empty; the only branches in the timed path are on `cfg["layer"]`, `cfg["n"]`, `x.shape`, `x.dtype`, `x.stride`; `torch.zeros`/`torch.empty` is a fixed-shape allocation from `x.shape`, a memset node under capture.
- [x] Nothing derived from `data` is cached across calls. `state` is written only in `prepare()` (`router_hi`, `router_lo`, from `weights["router"]`); `kernel_fn` and `_router_forward` only read `state` and allocate a fresh output every call; no module-level containers are touched (lint `cache-data`: none).
- [x] Weights: repacked losslessly in `prepare()` only; NVFP4 stays NVFP4, BF16 stays BF16 (router FP32 -> BF16 hi+lo with FP32 accumulation is the one exception). bf16 layers stay `w.contiguous()`; the router is the hi + lo bf16 pair built in `prepare()` with `tl.dot` accumulating in fp32 (`acc` is `tl.float32`, output fp32), the rules' explicit exception at any m.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. x enters the Triton kernel as bf16 (no cast); the only non-bf16 activation is the fp32 router output, which is the required output dtype; no KV cache or queries in this problem.
- [x] Attention sees every key (full window / full context); no top-k, sparse, block-skip, or weight-based skipping. Not applicable (dense projections); every K element of every weight row is read (K-slices partition K exactly, `K_SLICE * SPLIT_K == k` enforced by `_router_shape_ok`).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. Imports are `torch`, `torch.nn.functional`, `triton`, `triton.language`, `sglang...unquant._bf16_gemm_dispatch_impl` (same as the base); lint `harness`/`process`/`network`: none.
- [x] All work happens inside the timed call; no output caching or replay. prepare() runs warm-up launches on `torch.randn` synthetic x and discards their outputs; every timed call computes its own output from its own `data["x"]` into a tensor allocated in that call.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj` -> `CLEAN: muse_bf16_proj/submission.py` (exit 0, no warnings).

## Risks / what could fail
medium
Risks: (1) probe rows at m=1 now use hi/lo (allowed by the rules at any m) where the residual ~1.5e-5 relative must stay under the 1e-3 probe bound and the 1e-4 L2 bound; it should by ~10x, but m=1 has no averaging. (2) tl.atomic_add makes the fp32 sum order nondeterministic across replays; fp32 rounding ~1e-7 is far inside 1e-4. (3) Triton compile of 4-6 (m-bucket) specializations must fit the 10-minute run limit; well under. (4) Graph capture: torch.zeros inside kernel_fn is a fixed-shape allocation plus a memset node, allowed (shapes and config are fixed); no value-dependent branching. (5) The memset + atomics could be replaced by a SPLIT_K=1 full-K loop if the atomic path is slow at m<=32; note it as the ab-tune if the result is only partly faster. (6) If tl.dot with BLOCK_M=16 masked rows is slow at m=1, an FMA fallback (x[:, None, :] * w broadcast + tl.sum) is the alternative; keep that out of this run.

## Result (after the run)
- job `c2a8de5969374dff` (check), status done, passed True, tests 9/9
- speedup 0.9402x, 95% CI [0.9261, 0.9543], score 5137.966168671639 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused.result.json`, `muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused.patch`; branch `hyp/muse_bf16_proj/router-triton-hilo-fused`
