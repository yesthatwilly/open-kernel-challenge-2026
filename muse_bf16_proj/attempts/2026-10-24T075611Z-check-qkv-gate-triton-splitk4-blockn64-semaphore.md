# muse_bf16_proj · 2026-10-24T075611Z · check · qkv-gate-triton-splitk4-blockn64-semaphore

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T075006Z-qkv-gate-triton-splitk4-blockn64-semaphore.md` (branch `hyp/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore`).

- **Job:** `135ef6499875416a` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `e54c744`

## Goal
Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT 8.1-10.8 us + cublasLt::splitKreduce_kernel 1.7-2.0 us; reported 13.8-14.4 us vs a 7.9 us floor, the largest reported-minus-kernel gap of any layer) with ONE Triton launch of a split-K kernel with a bf16 semaphore epilogue: grid = (N // BLOCK_N, SPLIT_K) = (2304/64, 4) = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K-slice 2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0), num_warps=4, num_stages=4 (24-32 KB smem per stage, so 2 CTAs fit per SM and all 144 CTAs are co-resident in one wave); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [N, K] weight (the router's transposed addressing, no repack), fp32 acc via tl.dot(x_tile, w_tile, acc). Epilogue: store the fp32 partial to a persistent scratch state['qkv_ws'] = torch.empty((4, 32, 2304), fp32) from prepare(); tl.debug_barrier(); one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['qkv_cnt'] = torch.zeros(36, int32) from prepare(); the CTA drawing ticket == SPLIT_K-1 reloads the 4 partials with cache_modifier='.cg', sums them in fixed order 0..3, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 2304), bf16) and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm both BLOCK_M binaries in prepare() on synthetic x (m in 1, 8, 16, 32) then re-zero qkv_cnt. qkv_gate at m >= 128 (and the m=512 outlier test) and all other layers keep master's path. This is the router's 3-D split-K design (filled grid, short K-slices) applied to qkv_gate, NOT the regressed full-K skinny stream.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore`, file `muse_bf16_proj/submission.py` only (+117 lines, nothing removed). Router kernel, its config, state keys and warm-up are untouched.

- New `_qkv_splitk_sem_kernel` (`@triton.jit(do_not_specialize=["M"])`), grid `(2304 // 64, 4) = (36, 4)` = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K_SLICE=2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32, num_warps=4, num_stages=4. Weight tiles are read as `[BLOCK_K, BLOCK_N]` straight from the row-major `[N, K]` weight (the router's transposed addressing, no repack), and the accumulator is fp32 via `tl.dot(x_tile, w_tile, acc)`.
- Workspace hygiene: the fp32 partial is stored to `ws[pid_k, offs_m, offs_n]` under `mask=offs_m < M`. The reducer loads the same tile offsets under the same mask. Every value it reads was therefore written by one of this launch's 4 CTAs for that n-tile, and stale rows (>= m) of `qkv_ws` are never read. The output store is masked the same way.
- Ordering: partial store, then `tl.debug_barrier()`, then `ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")`, then `tl.debug_barrier()` (unconditional, before the branch). Inside `if ticket == SPLIT_K - 1:` the reducer reloads partials 0..3 with `cache_modifier=".cg"` and sums them in fixed order with a `tl.static_range(1, SPLIT_K)` loop starting from partial 0. It casts once to bf16, does a masked store and resets the counter with `tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")`. Relaxed is enough for the reset because the next reader is a later kernel on the same stream, and the kernel boundary orders it. No CTA waits on another, so it cannot deadlock.
- One scalar atomic per CTA: `cnt_ptr + pid_n` is a scalar pointer (pid_n is a scalar program id) and the value is the scalar `1`, with no tensor operand. Triton lowers a scalar `atomic_rmw` to a single atom predicated on one thread and broadcasts the result through shared memory, so each CTA issues exactly one ticket. The `xchg` reset is also scalar. The PTX was NOT dumped (there is no GPU or Triton on this node). The attempt file should confirm one `atom.acq_rel.gpu.global.add` per CTA in the compiled PTX and the kernel's smem size (for the 2-CTA/SM claim).
- State: these use separate keys, `state["qkv_ws"] = torch.empty((4, 32, 2304), fp32)` and `state["qkv_cnt"] = torch.zeros(36, int32)`, both allocated in prepare(). prepare() warms every m in 1..32 (both BLOCK_M binaries) on synthetic `torch.zeros` x, then calls `state["qkv_cnt"].zero_()`.
- Route guard in kernel_fn: `layer == "qkv_gate"`, the stored weight shape is (2304, 8192), cfg `1 <= m <= 32`, `n == 2304` and `k == 8192`, and x is 2-D, bf16, shape `(cfg m, 8192)` with `stride(1) == 1`. Everything else (qkv_gate m >= 128, the m=512 outlier test, o/pooler/unpooler, router) runs master's code unchanged. The output is `torch.empty((m, 2304), bf16)` with shape-only allocation. There is no autotune and no host sync, and the code branches only on cfg and shapes.
- Deviation: the spec warmed m in {1, 8, 16, 32}. Per the review instructions, the warm-up covers all m in 1..32. That is still only two compiled binaries, because M is not specialized. Also, the module docstring gained a 4-line paragraph describing the new path.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore diff`:

- [x] CUDA-graph capturable: there is no `.item()`, `.cpu()`, `synchronize()` or tensor print in the diff. Output allocation is shape-only (`torch.empty((m, 2304))`). Branching is on cfg/shape/dtype/stride only. The in-kernel `if ticket == ...` is device-side control flow, not a host branch on input values. The counters reset in-kernel, so replays are self-consistent.
- [x] Nothing derived from `data` is cached across calls. `qkv_ws` holds per-launch partials, and every value the reducer reads is rewritten in the same launch (same mask). `qkv_cnt` returns to 0 at the end of each launch.
- [x] Weights: the qkv_gate bf16 weight is used as is (`state["qkv_gate"]`, the contiguous original), with no repack and no precision change. The router hi+lo split is unchanged from master.
- [x] Activations: x stays bf16 into `tl.dot`, accumulation and partials are fp32, and the output is bf16, as SGLang does.
- [x] Attention sees every key: n/a (dense projection problem, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only imports are the existing torch/triton/sglang ones.
- [x] All work happens inside the timed call, with no output caching or replay. The prepare() warm-up runs on synthetic zeros only and its outputs are discarded.
- [x] No `triton.autotune` (static constexpr config chosen from m).
- [x] Only qkv_gate m<=32 is rerouted. The router, o, pooler, unpooler and qkv_gate m>=128 keep master's code paths, as the diff shows (additions only).

Lint (`okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: submission.py
```

`python3 -I -m py_compile` passed. No `__pycache__` was left in the worktree.

## Risks / what could fail
medium
Why this is not a retry of qkv-gate-triton-skinny-m32 (0.924x, 22-26 us, ~1.6 TB/s): that kernel was full-K with BLOCK_N=16 (each CTA streamed 16 rows x 8192 K with 32 KB in flight per SM). This one uses the two levers the landed router proved and the unpooler control measured: BLOCK_N=64 tiles (3.73 TB/s on unpooler) and a split-K grid with short slices that keeps 144 CTAs each with 4 x 16 KB weight stages in flight (~9 MB fleet-wide, above Little's law for 4.8 TB/s). Main risk: nvjet is unusually efficient on qkv_gate (8.1 us = 4.66 TB/s at m=16), so the bar is high; the gain must come mostly from removing the reduce node and the gap. 144 CTAs on 132 SMs means 12 SMs host two CTAs; if the profile shows a 2x tail, the one-axis retune is SPLIT_K=8/num_stages=3 (288 CTAs, ~2.2 per SM) or BLOCK_N=128/SPLIT_K=8 (144 CTAs, larger tiles). Semaphore correctness risk as in pooler-triton-splitk2-blockn64-semaphore; a broken ordering fails the m=1 qkv_gate test. If both land they should share one kernel; they are minted independently so each attributes on its own.

## Result (after the run)
- job `135ef6499875416a` (check), status done, passed True, tests 9/9
- speedup 1.1431x, 95% CI [1.1260, 1.1603], score 4226.016767877891 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/qkv-gate-triton-splitk4-blockn64-semaphore.result.json`, `muse_bf16_proj/hypotheses/done/neutral/qkv-gate-triton-splitk4-blockn64-semaphore.patch`; branch `hyp/muse_bf16_proj/qkv-gate-triton-splitk4-blockn64-semaphore`
