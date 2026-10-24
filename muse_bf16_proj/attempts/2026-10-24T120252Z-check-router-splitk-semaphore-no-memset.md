# muse_bf16_proj · 2026-10-24T120252Z · check · router-splitk-semaphore-no-memset

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T075041Z-router-splitk-semaphore-no-memset.md` (branch `hyp/muse_bf16_proj/router-splitk-semaphore-no-memset`).

- **Job:** `cb76350d40ee4617` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `0584e34`

## Goal
Built on master (dba4ffd). For the router at m <= 32 only (buckets (16,32,16) and (32,32,16): grid (1, 8, 16) = 128 CTAs, unchanged tiles, BLOCK_K=128, num_stages=3, num_warps=4), remove the torch.zeros((m, 256), fp32) fill kernel (at::native::vectorized_elementwise_kernel, 0.7-0.8 us plus its own graph node) and the relaxed atomic_add epilogue, and replace them with a serial split-K reduce inside the same launch: out = torch.empty((m, 256), fp32); each CTA stores its fp32 [BLOCK_M, 32] K-partial to a persistent scratch state['router_ws'] = torch.empty((16, 32, 256), fp32) allocated in prepare() (fully overwritten each call, nothing input-derived kept), then tl.debug_barrier(), then one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['router_cnt'] = torch.zeros(8, int32) from prepare(); the CTA drawing ticket == 15 reloads the 16 partials (cache_modifier='.cg'), sums them in fixed order 0..15 in fp32, masked-stores to out, and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Make this a separate @triton.jit kernel (_router_hilo_sem_kernel) sharing the K loop so the m > 32 buckets keep master's kernel, zeros and atomics byte-for-byte. Re-zero router_cnt after the prepare() warm-up. (The simpler 'split 0 stores, the others atomic_add' is not used: nothing orders split 0's plain store before another split's add into torch.empty memory, so a late store can overwrite earlier adds.)

## What changed, and where
Built at SPLIT_K=16 as written (neither SPLIT_K=32 run gained), in the worktree's `muse_bf16_proj/submission.py` only:

- New `@triton.jit(do_not_specialize=["M"]) _router_hilo_sem_kernel` with the same tiles and K loop as `_router_hilo_kernel`. The loop is copied verbatim, not factored into a shared helper, so master's kernel source stays byte-for-byte unchanged for m > 32. `SPLIT_K`, `N`, `WS_M` and `K_SLICE` are constexprs, and the winner test is `ticket == SPLIT_K - 1`. Epilogue: masked fp32 partial store to `ws[pid_k, rows<M, tile]`, `tl.debug_barrier()`, one scalar `tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")`, then `tl.debug_barrier()`. The winner reloads, sums, masked-stores fp32 to `out` and runs `tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")`. This is the ticket pattern copied from `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore`, which passed correctness.
- The reload is one `[SPLIT_K, BLOCK_M, BLOCK_N]` `.cg` load using the same `m_mask` footprint as the store, so all partials are in flight before any add. It is followed by `tl.sum(parts, axis=0)`. **Deviation:** the order is fixed at compile time and deterministic, but it is Triton's reduction order, not a literal left-to-right 0..15 chain. A `static_range` load-then-add loop cannot be guaranteed to hoist the loads, because issue is in order and each add stalls before the next load issues. The reviewer named the single tile load as an acceptable option. Register cost of the tile is 64 fp32/thread at BLOCK_M=16 and 128 at BLOCK_M=32. The dot accumulator is dead by then, but watch for spills at m=32. `tl.arange(0, SPLIT_K)` requires SPLIT_K to be a power of 2, which the `_router_config` docstring notes.
- `_router_forward(x, hi, lo, ws, cnt)`: for m <= `ROUTER_SEM_MAX_M` (32) it uses `torch.empty` and launches the new kernel on the same grid `(1, 8, 16)`, BLOCK_K=128, stages 3, warps 4. For m > 32 it runs master's `torch.zeros` and atomic kernel unchanged.
- `_router_shape_ok`: for m <= 32 it also requires `m <= block_m` (one M-block, so pid_m == 0 and a per-N-tile counter suffices) and `n == ROUTER_N`. This always holds for the problem.
- `prepare()`: `state["router_ws"] = torch.empty((16, 32, 256), fp32)` and `state["router_cnt"] = torch.zeros(8, int32)`, both sized from `_router_config` over m = 1..32. They are allocated before the warm-up (ROUTER_WARM_MS 1, 8, 16, 32 compile both small buckets), and `router_cnt.zero_()` runs after the warm-up.
- Module and `_router_config` docstrings were updated. Nothing else was touched.
- Not checked (no GPU or Triton on this node): the reviewer's open question on whether the scalar acq_rel ticket adds an smem broadcast on the winner path (TTGIR/PTX), and whether the 3-D tile load spills.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-splitk-semaphore-no-memset diff`:

- [x] CUDA-graph capturable: there is no `.item()`, `.cpu()`, `synchronize()` or tensor print. `torch.empty` in `_router_forward` is a shape-only allocation, the same as master's `torch.zeros`. Branches depend on `m` (shape) only. The ticket branch is in-kernel.
- [x] Nothing derived from `data` is cached across calls. `router_ws` rows < m are fully rewritten before they are read on every call, and rows >= m are never read (masked). `router_cnt` returns to 0 at the end of every launch.
- [x] Weights: hi + lo split built in `prepare()` (unchanged from master). The router keeps the allowed FP32 → BF16 hi+lo with FP32 accumulation. BF16 layers are untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is still loaded as bf16, and accumulation, partials and output are fp32. n/a for KV cache and queries: there is no attention in this problem.
- [x] n/a: there is no attention in this problem (dense projections only), so no key skipping.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The diff adds no imports.
- [x] All work happens inside the timed call. The GEMM and reduce both run in `kernel_fn`'s launch, with no output caching or replay. The prepare() warm-up runs on synthetic inputs only.
- [x] No `triton.autotune`. Configs are static from `m`.
- [x] The first line `# okc: problem=muse_bf16_proj` is kept.

okc_lint output:

```
CLEAN: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` succeeded, and the `__pycache__` it left was removed.

## Risks / what could fail
medium
Answers whether the memset node is a real cost on the router's small-m rows (master's profile: memset 0.7-0.8 us of a 4.1 us kernel sum, and a ~1.7 us reported-minus-kernel gap). The fp32 output also makes this deterministic (fixed-order sum instead of atomic order), a free side benefit for the router's 0.01% bound. Risks: (1) memory-ordering correctness of the barrier + scalar acq_rel ticket pattern in Triton; a bug shows as wrong outputs on the m=8 and m=9 router tests, not silently; (2) a counter left nonzero after warm-up corrupts every call, hence the re-zero; (3) the serial tail: the reducing CTA's 16 dependent-latency loads could cost ~0.5 us, more at m=32, which is why decode_32 is called flat. Independent of router-splitk32-small-m-all-loads-in-flight (that one changes grid/stages, this one the epilogue); if both gain, combine on SPLIT_K=32 only after re-checking the doubled reduce tail.

## Result (after the run)
- job `cb76350d40ee4617` (check), status done, passed True, tests 9/9
- speedup 1.1516x, 95% CI [1.1343, 1.1688], score 4195.0143172141225 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-splitk-semaphore-no-memset.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-splitk-semaphore-no-memset.patch`; branch `hyp/muse_bf16_proj/router-splitk-semaphore-no-memset`
