# muse_bf16_proj · 2026-10-24T100317Z · check · pooler-unpooler-m128-wgmma-splitk-semaphore

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T092553Z-pooler-unpooler-m128-wgmma-splitk-semaphore.md` (branch `hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore`).

- **Job:** `d34fd76f9c444728` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `2393474`

## Goal
Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). For layer in {pooler, unpooler} and cfg m == 128 only (verify_16; any 65 <= m <= 128 may take the same path, rows >= m masked), replace _bf16_gemm_dispatch_impl (pooler: nvjet_sm90_tst_64x64_64x13_2x1 at 25.2 us, a 64x64 tile with no split-K, 2 x 64 = 128 CTAs each re-reading half of x; unpooler: nvjet_sm90_tst_128x64_64x8_1x2 at 20.0 us) with ONE Triton launch of a new tensor-core split-K kernel _bf16_tc_splitk_kernel: tile BLOCK_M=128 (all of m), BLOCK_N=128, BLOCK_K=64, num_warps=8 (two warpgroups, so Triton emits wgmma m64n128 per warpgroup), num_stages=4 (32 KB per stage: 16 KB x + 16 KB weight, 128 KB smem). Grid (N // 128, SPLIT_K): pooler (n=4096, k=8192) SPLIT_K=4 -> 32 x 4 = 128 CTAs, K-slice 2048 (32 K-steps); unpooler (n=8192, k=4096) SPLIT_K=2 -> 64 x 2 = 128 CTAs, K-slice 2048. Every CTA therefore moves 512 KB of weight from HBM and 512 KB of x from L2 (cuBLAS pooler moves 512 KB weight + 1 MB x per CTA), and does 67 MFLOP. Weight tiles read as [BLOCK_K, BLOCK_N] straight from the row-major [N, K] weight with master router direct transposed addressing (offs_k[:, None] + offs_n[None, :] * K; no repack); acc = tl.dot(x_tile, w_tile, acc) fp32. Reduce: reuse verbatim the ticket-semaphore epilogue from branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (masked fp32 partial store to a prepare()-allocated workspace state[layer + "_ws"] = torch.empty((SPLIT_K, 128, n), fp32), tl.debug_barrier(), one gpu-scope acq_rel tl.atomic_add ticket per N-tile on state[layer + "_cnt"] = torch.zeros(n // 128, int32), the CTA drawing SPLIT_K-1 reloads all partials with cache_modifier=".cg", sums in fixed order 0..SPLIT_K-1, casts once to bf16, masked-stores into out = torch.empty((m, n), bf16), and resets its counter with tl.atomic_xchg; add tl.debug_barrier() after the ticket as the review of that parent suggested). Pre-compile both layer specializations in prepare() on synthetic x at m=128 and re-zero the counters after the warm-up. Every other layer/m keeps its master path (pooler/unpooler at m <= 32 and m = 256 stay cuBLAS: m=256 is compute-bound, 17.2 GFLOP, where cuBLAS already runs ~580 TFLOP/s).

## What changed, and where
All changes in the worktree's `muse_bf16_proj/submission.py` (+155 lines, nothing removed; router and qkv_gate/o paths untouched):

- New constants `TC_*` and `TC_LAYERS = {"pooler": (4096, 8192, 4), "unpooler": (8192, 4096, 2)}` (N, K, SPLIT_K). BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=4; grid (N // 128, SPLIT_K) = 128 CTAs for both layers, K-slice 2048.
- New `_bf16_tc_splitk_kernel` (`do_not_specialize=["M"]`). Its mainloop and the ticket-semaphore epilogue are copied verbatim from `_bf16_splitk_kernel` on `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore`, which already has the `tl.debug_barrier()` after the ticket. The direct transposed `[BLOCK_K, BLOCK_N]` weight addressing needs no repack. The one addition is a `REDUCE: tl.constexpr` guard around the winner's reload, sum and store. The counter reset (`atomic_xchg`) runs whatever REDUCE is. The timed path always passes REDUCE=True, which compiles to the parent's epilogue.
- `_tc_shape_ok` gates on layer in {pooler, unpooler}, 65 <= m <= 128 (cfg and x.shape), the full cfg n/k, and bf16 contiguous x and w. The m=130 unpooler test and m=256/1000 therefore keep master's cuBLAS path, and so do m <= 64 (1/5/7/8/16/32). `_tc_forward` allocates `out = torch.empty((m, n), bf16)` and launches once.
- `prepare()` allocates `state[layer+"_ws"] = torch.empty((SPLIT_K, 128, n), fp32)` (8 MB each) and `state[layer+"_cnt"] = torch.zeros(n // 128, int32)`. It warms m=65 and m=128 on synthetic x (one compile per layer, since M is not specialized), then runs the reviewer's diagnostics via `_tc_diagnostics`, and re-zeroes the counters afterwards.
- `kernel_fn` takes the new path first when `_tc_shape_ok` holds. Otherwise the master logic is unchanged.
- Diagnostics follow the review's suggestion and run in prepare() only, untimed, wrapped in try/except. They print `[tc_splitk <layer>] wgmma_in_ptx=... n_regs=... n_spills=...`, plus the warm-L2 event-timed us of the full kernel and the REDUCE=False variant, which gives the tail estimate. They cost one extra compile per layer (REDUCE=False). The timing uses `end.synchronize()` in prepare(), which the lint warns about; it is outside kernel_fn's call graph.
- Deviation: only the constexpr REDUCE guard, as above, so the diagnostics can measure the tail. No other change to the verbatim epilogue. The suggestion to keep the winner's acc in registers is NOT applied, because the review lists it as a follow-up.
- The sibling qkv-gate-o-m128-wgmma-splitk-semaphore must merge the `_bf16_tc_splitk_kernel` name, the `TC_*` names and the `_ws`/`_cnt` keys by hand if both land.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore diff`:

- [x] CUDA-graph capturable. The timed path (`kernel_fn` → `_tc_shape_ok`, `_tc_forward`) has no `.item()`/`.cpu()`/`synchronize()`/tensor prints. It branches only on layer, cfg m/n/k and x shape/dtype/stride. Its only allocation is `torch.empty` for out, sized by shape. The counter self-resets in-kernel, so graph replays start from 0. The only `synchronize()` is in prepare()'s diagnostics.
- [x] Nothing derived from `data` is cached across calls. ws is fully rewritten for rows < m before it is read, every call, and the reducer reads the same mask. cnt holds only tickets and returns to 0. Warm-up and diagnostics use synthetic x.
- [x] Weights stay BF16 as given and are read in place: no repack, no precision change. The router hi+lo is unchanged from master.
- [x] Activations: x is read as BF16, accumulation is fp32 (tl.dot), and the output is a single cast to BF16, as SGLang does. n/a for the KV cache / FP8 queries: no attention in this problem.
- [x] n/a: no attention in this problem (dense GEMMs only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only new imports used are the existing torch/triton. torch.cuda.Event is used only for prepare()-local timing and is not reassigned.
- [x] All work happens inside the timed call. prepare() runs only warm-up/diagnostic launches on synthetic x, and kernel_fn recomputes the full GEMM every call. No output caching or replay.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept. No triton.autotune.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
warn  host-sync    L285: end.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

`python3 -I -m py_compile` OK; __pycache__ removed.

## Risks / what could fail
medium
Medium risk. (1) Triton must lower BLOCK_M=128 x BLOCK_N=128 with K-contiguous [BLOCK_K, BLOCK_N] weight tiles to wgmma (MN-major B from smem is supported for bf16); if it falls back to mma.sync the kernel is compute-limited near 9-12 us of math at 128 CTAs and may only match cuBLAS. Check the PTX for wgmma.mma_async before the run if possible. (2) Workspace traffic: SPLIT_K x 64 KB fp32 partials per N-tile (8 MB per call), L2-resident; the last-CTA reduce tail is ~0.5-1 us. (3) The semaphore is the measured-correct pattern from the pooler/qkv-gate split-K runs; counter must be re-zeroed after warm-up. Attribution: only the two verify_16 rows change code; their kernel us against nvjet 25.2/20.0 us is the readout. Load verify_16 is one of six; ~8 us out of ~76 us is ~+1.8% on the headline, so read per-shape rows.

## Result (after the run)
- job `d34fd76f9c444728` (check), status done, passed True, tests 9/9
- speedup 1.1158x, 95% CI [1.0991, 1.1326], score 4329.264268201423 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-wgmma-splitk-semaphore.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-wgmma-splitk-semaphore.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore`
