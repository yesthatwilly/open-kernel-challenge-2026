# muse_bf16_proj · 2026-10-24T075600Z · check · pooler-triton-splitk2-blockn64-semaphore

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T074946Z-pooler-triton-splitk2-blockn64-semaphore.md` (branch `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore`).

- **Job:** `08ec203e5717406e` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `cee590d`

## Goal
Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer == 'pooler' and cfg m <= 32 only (n=4096, k=8192, 67.1 MB bf16), replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/16/32_..._splitK_TNT 18.3-20.2 us + cublasLt::splitKreduce_kernel 1.6-1.9 us, two graph nodes) with ONE Triton launch of a new _bf16_splitk_kernel that applies the router's 2-D tile x K-slice grid to a bf16 output: grid = (N // BLOCK_N, SPLIT_K) = (4096/64, 2) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=2 (K-slice 4096, 32 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0), num_warps=4, num_stages=4; weight tiles read as [BLOCK_K, BLOCK_N] straight from the row-major [N, K] weight with the router's direct transposed addressing (no repack), acc = tl.dot(x_tile, w_tile, acc) in fp32. Per-CTA work is therefore exactly the unpooler-full-triton-skinny-m32 program (64 weight rows x 4096 K), the one Triton config measured at 3.73 TB/s at m=16. Epilogue (no memset, no second kernel): each CTA stores its fp32 partial to a persistent workspace state['pooler_ws'] = torch.empty((SPLIT_K, 32, 4096), fp32) allocated in prepare() (scratch fully overwritten each call, nothing derived from inputs survives), then tl.debug_barrier(), then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['pooler_cnt'] = torch.zeros(64, int32) made in prepare(); the CTA that draws ticket == SPLIT_K-1 reloads both partials (cache_modifier='.cg'), sums them in fixed order 0..SPLIT_K-1 (deterministic), casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 4096), bf16), and resets its counter with tl.atomic_xchg(cnt_ptr + pid_n, 0) so the next graph replay starts at 0. Pre-compile both BLOCK_M specializations in prepare() on synthetic x (m in 1, 8, 16, 32) and re-zero pooler_cnt after the warm-up. Pooler at m >= 128 and every other layer keep their master path.

## What changed, and where
All changes are in the worktree's `muse_bf16_proj/submission.py`. The diff adds 113 lines and removes none. The router kernel, `_router_forward` and the router warm-up are untouched.

- **New `_bf16_splitk_kernel`** (`@triton.jit(do_not_specialize=["M"])`).
  - Grid is `(4096 // 64, 2)`, which is 128 CTAs. BLOCK_N=64, BLOCK_K=128 (as specified; the BLOCK_K=256 control is a separate companion), SPLIT_K=2, K_SLICE=4096 (32 K-steps), num_warps=4, num_stages=4.
  - BLOCK_M is 16 for m<=16 and 32 for 17..32. There is one M-block, and rows >= m are masked.
  - The weight is read as `[BLOCK_K, BLOCK_N]` directly from the row-major `[N, K]` weight, using the router's addressing. The product is `acc = tl.dot(x_tile, w_tile, acc)` in fp32.
- **Epilogue, in this order:**
  1. Masked (`offs_m < M`) fp32 partial store to `ws[pid_k]`.
  2. `tl.debug_barrier()`.
  3. One scalar `tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")` per CTA.
  4. `tl.debug_barrier()`, so the other threads' reloads are ordered after thread 0's acquire. This is the reviewer's open question, resolved by adding the barrier unconditionally.
  5. `if ticket == SPLIT_K - 1`: reload slices 0..SPLIT_K-1 with `cache_modifier=".cg"` and the same `offs_m < M` mask, sum them in fixed order, cast to bf16 once, and do a masked store of `[m, 64]` into `out = torch.empty((m, 4096), bf16)`. Then `tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")`.
  - The kernel has no early return. Every CTA stores its partial and takes exactly one ticket on every path.
- **Release side.** No separate fence instruction: Triton has no `tl` threadfence primitive, and `inline_asm_elementwise` with no tensor operand is untested here. The release comes from the acq_rel gpu-scope atomic, preceded by `bar.sync`. This matches the review's "release fence / acq_rel" alternative.
- **State and prepare().** New state keys `pooler_ws` (fp32 `[2, 32, 4096]`, `torch.empty`, pure scratch; every row < m is rewritten each call before it is read) and `pooler_cnt` (int32 `[64]`, zeros). prepare() warms every m in 1..32 on synthetic x, which compiles both BLOCK_M specializations. It then calls `pooler_cnt.zero_()` (stream-ordered).
- **Dispatch.** `kernel_fn` takes the new path only when `layer == "pooler"` and `_pooler_shape_ok` holds. That check requires cfg m <= 32, cfg n == 4096, cfg k == 8192, x 2-D bf16, x.stride(1) == 1, 1 <= x.shape[0] <= 32, x.shape[1] == 8192, and the weight to be `[4096, 8192]` with stride(1) == 1. Pooler at m >= 128 and every other bf16 layer fall through to the unchanged `_bf16_gemm_dispatch_impl`.
- **Deviations.** None from the spec's `change` paragraph. Two items follow the review rather than adding to the spec: the post-ticket barrier, and warming every m in 1..32 rather than only 1/8/16/32.
- **Uncertainties (cannot run on this node).**
  - Whether Triton accepts the scalar `atomic_add` with `sem`/`scope` and the `atomic_xchg` keyword args on the judge's Triton version. Both are documented in Triton 3.x.
  - Whether the scalar atomic is lowered to a single thread's predicated `atom`. Triton's usual lowering is a thread-0 predicate plus an smem broadcast.
  - The shared workspace assumes pooler calls are not run concurrently on two streams. The judge runs them sequentially.

## Rules check (each must be true before the run)
Each box was checked against `git -C .worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore diff`.

- [x] **CUDA-graph capturable.** The diff adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. It allocates nothing that depends on values: `out` and the grid depend only on x.shape. The branches are on shapes, `cfg` and dtype only. The counter is 0 at capture and self-resets on every replay.
- [x] **Nothing derived from `data` is cached across calls.** `pooler_ws` is scratch, fully rewritten for rows < m before it is read in the same launch, and rows >= m are never read. `pooler_cnt` holds only ticket counts and returns to 0.
- [x] **Weights.** The pooler bf16 weight is used as is: the same `state["pooler"]` tensor, with no repack and no precision change.
- [x] **Activations are BF16 where SGLang keeps BF16.** x is bf16 and the output is bf16. Accumulation is fp32 with one bf16 rounding, the same as cuBLAS splitK.
- [x] n/a: no KV cache or queries in this problem (dense projections).
- [x] n/a: no attention in this problem.
- [x] **No forbidden operations.** No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff imports nothing new.
- [x] **All work happens inside the timed call; no output caching or replay.** The prepare() warm-up uses synthetic x only, and its outputs are discarded.
- [x] **No `triton.autotune`.** The configs are static constants chosen from m.

okc_lint output:
```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore/muse_bf16_proj/submission.py
```
`python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
medium
Differs from the failed/neutral skinny family (pooler-triton-skinny-m32: full-K, BLOCK_N=32, 128 programs, 3.3 TB/s, neutral 0.998x) in exactly the two things the evidence says matter: BLOCK_N=64 tiles (the only Triton config that beat nvjet's stream) and a split-K grid so the 64-row tile still fills 128 SMs; and unlike the router it emits bf16 without a memset or cast kernel, via a self-resetting per-tile semaphore. Risks: (1) the semaphore pattern (barrier + scalar acq_rel atomic, cumulativity of the release over the CTA's other threads' stores) must be correct under Triton; a wrong ordering shows up as a numeric failure on the m=7 pooler test or a timed-call check, not silently. (2) If the counter is ever left nonzero (e.g. the warm-up), every later call is wrong: re-zero after warm-up. (3) decode_32 may carry the BLOCK_M=32 penalty seen on unpooler; if decode_1/16/verify_1 gain and decode_32 loses, the follow-up is BLOCK_M=16 with two accumulators at m=32, not dropping the design. Readout: the kernel's own us per row vs nvjet 18.3/18.9/19.3/20.2 us; gain on 4 shapes that are 28-36% of four loads, ~2-2.5 us each, ~+3% geomean. Tests touched: m=7 pooler (masked rows, BLOCK_M=16). Correctness: fp32 accumulate, one bf16 rounding, same as cuBLAS splitK.

## Result (after the run)
- job `08ec203e5717406e` (check), status done, passed True, tests 9/9
- speedup 1.1664x, 95% CI [1.1489, 1.1839], score 4141.769852269138 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore.patch`; branch `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore`
