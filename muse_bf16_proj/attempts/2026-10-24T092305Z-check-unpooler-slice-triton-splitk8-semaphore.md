# muse_bf16_proj · 2026-10-24T092305Z · check · unpooler-slice-triton-splitk8-semaphore

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T075111Z-unpooler-slice-triton-splitk8-semaphore.md` (branch `hyp/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore`).

- **Job:** `bb31bb36545a47bb` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `7268a14`

## Goal
Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only, replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0 us + cublasLt::splitKreduce_kernel 1.5 us; 7.0-7.1 us reported vs a 1.7-1.8 us floor, the worst ratio to floor of any bf16 shape) with ONE Triton launch shaped like the landed router kernel: grid = (1024 // BLOCK_N, SPLIT_K) = (16, 8) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K-slice 512, 4 K-steps), BLOCK_M=16 (rows >= m masked to 0), num_warps=4, num_stages=3, so each CTA reads 64 rows x 512 K x 2 B = 64 KB, the same bytes per CTA as the router's 32 rows x 512 K x (hi + lo); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [8192, 4096] weight's first 1024 rows (row stride 4096, no copy of the slice, no repack), fp32 acc = tl.dot(x_tile, w_tile, acc). Epilogue with a bf16 output and no memset: store the fp32 partial to persistent scratch state['unpool_ws'] = torch.empty((8, 16, 1024), fp32) from prepare(); tl.debug_barrier(); scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['unpool_cnt'] = torch.zeros(16, int32) from prepare(); the CTA drawing ticket == 7 reloads the 8 partials (cache_modifier='.cg'), sums in fixed order, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 1024), bf16), and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm the binary in prepare() on synthetic x (m in 1, 5, 8) and re-zero unpool_cnt afterwards. Unpooler with n == 8192 (m >= 16) and every other layer keep master's path.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore`, file `muse_bf16_proj/submission.py` only (+117, no deletions). Router, pooler, qkv_gate, o and the unpooler at n == 8192 are byte-for-byte master.

- Added `_bf16_splitk_kernel`, copied verbatim from branch `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore` (the semaphore pattern that passed correctness there). Only the docstring changed, to note that w may be the leading N rows of a taller weight. The pattern: masked (rows < M) fp32 partial store to ws, `tl.debug_barrier()`, a scalar `tl.atomic_add(cnt+pid_n, 1, sem="acq_rel", scope="gpu")`, then a second `tl.debug_barrier()` (the review's open question 1). The `ticket == SPLIT_K-1` CTA reloads all SPLIT_K partials with `.cg` in fixed order 0..7 (masked to rows < M), casts once to bf16, does a masked store, then runs `tl.atomic_xchg(cnt+pid_n, 0)`.
- Constants: `UNPOOL_N=1024, UNPOOL_K=4096, UNPOOL_MAX_M=16 (=BLOCK_M=WS_M), BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K_SLICE=512), num_warps=4, num_stages=3`. The grid is (16, 8) = 128 CTAs.
- `_unpool_forward` passes the FULL `state["unpooler"]` [8192, 4096] with `stride_wn = w.stride(0) = 4096`. There is no slice copy and no repack. The grid only covers rows < 1024. out = `torch.empty((m, 1024), bf16)`, with no memset.
- `_unpool_shape_ok` gates on cfg n == 1024, k == 4096, m <= 16, bf16 contiguous-K x and w. `kernel_fn` checks it first for `layer == "unpooler"`. Every other case falls through to master's path unchanged.
- prepare(): `state["unpool_ws"] = torch.empty((8, 16, 1024), fp32)` and `state["unpool_cnt"] = torch.zeros(16, int32)`. It warms on synthetic randn x at m in (1, 5, 8) against the real unpooler weight, so the stride-4096 specialization matches the timed calls (review open question 5; M is do_not_specialize, so this is one compile), then runs `unpool_cnt.zero_()`.

Deviations and choices:
- The reducer reloads all 8 partials, including its own. It does not use its register acc plus 7 reloads (review open question 2). The brief said to copy the proven pooler pattern exactly, and that pattern reloads all of them. The cost is one extra masked read of at most 4 KB from L2.
- Partials are stored and reloaded only for rows < m (review open question 3). This is also what the pooler pattern does.
- The gate is m <= 16 rather than m <= 8. This is defensive and has no practical effect, because the problem only uses n == 1024 at m <= 8, and BLOCK_M=16 covers it.
- The kernel is the same function as on the pooler branch, under the same name. If both land, the merge is a deduplication, not a conflict in semantics.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the diff (grep is clean). The only allocation is `torch.empty((m, 1024))` sized by shape. The branch is on cfg/shape only (`_unpool_shape_ok`). The counter self-resets in-kernel, so every replay starts from 0.
- [x] Nothing derived from `data` is cached across calls: `unpool_ws` is scratch that is fully rewritten (rows < m) before it is read in the same launch, and `unpool_cnt` returns to 0. Both are allocated in prepare() from shapes only.
- [x] Weights: no repack and no precision change. The bf16 unpooler weight is read in place (its first 1024 rows). The router hi+lo is untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16, the output is bf16 [m, 1024], and accumulation is fp32 (as in cuBLAS).
- [x] n/a: no attention or KV cache in this problem. KV FP8 and query precision do not apply.
- [x] n/a: no attention in this problem, so key skipping does not apply. The GEMM reads every weight element of the slice. There is no sparsity or skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds only triton/torch code.
- [x] All work happens inside the timed call. There is no output caching or replay: one Triton launch per call computes the full product.
- [x] No triton.autotune. The first-line marker `# okc: problem=muse_bf16_proj` is kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` OK. `__pycache__` removed.

## Risks / what could fail
medium
Not a retry of unpooler-slice-triton-gemv (0.880x): that one used an FMA broadcast GEMV without tl.dot and ran 70.8 us at m=8. This copies the only Triton design proven fast on an 8.4 MB read here (the landed router: tl.dot, BLOCK_K=128, K-slice 512, 128 CTAs, 3 stages, 3.3 us kernel) and changes only what the bf16 output needs (semaphore reduce instead of zeros + fp32 atomics + a cast kernel, which would be 3 nodes and lose). Per-shape ~30% on 2 shapes, each ~13% of its load: ~+4% on decode_1 and verify_1, ~+1.3% geomean. Risks: semaphore ordering correctness (the m=5 unpooler test and the m=7.. tests exercise it); the serial last-CTA reduce reads 8 x 16 x 64 x 4 B = 32 KB, ~0.3 us; the row stride is 4096 while the weight tensor is [8192, 4096], so the kernel must take stride_wn from the full weight, not assume a contiguous slice. If it lands, the same kernel is the shared base for pooler-triton-splitk2-blockn64-semaphore and qkv-gate-triton-splitk4-blockn64-semaphore.

## Result (after the run)
- job `bb31bb36545a47bb` (check), status done, passed True, tests 9/9
- speedup 1.1546x, 95% CI [1.1373, 1.1720], score 4183.829196755866 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/unpooler-slice-triton-splitk8-semaphore.result.json`, `muse_bf16_proj/hypotheses/done/neutral/unpooler-slice-triton-splitk8-semaphore.patch`; branch `hyp/muse_bf16_proj/unpooler-slice-triton-splitk8-semaphore`
