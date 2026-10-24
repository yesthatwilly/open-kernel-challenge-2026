# muse_bf16_proj · 2026-10-24T181732Z · submit · pooler-unpooler-m128-tma-wspec-rewrite

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T140242Z-pooler-unpooler-m128-tma-wspec-rewrite.md` (branch `hyp/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite`).

- **Job:** `61da9b91dab44bf4` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `fd7d7ee`

## Goal
Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler} at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m masked; the m=130 unpooler test stays cuBLAS). It starts from the regression pooler-unpooler-m128-wgmma-splitk-semaphore (branch hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore, _bf16_tc_splitk_kernel) and fixes the two costs that run's own prepare() diagnostics measured (d34fd76f9c444728.log: wgmma emitted, 140-154 regs, no spills; warm-L2 mainloop 26.0 us for both layers = 2.6 TB/s, reduce tail 6.5 us pooler / 4.3 us unpooler). Mainloop: (1) the weight tile arrives by TMA: a TensorDescriptor built once in prepare() from the weight alone (triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128]); this image's Triton is 3.8.0 with the tensor_descriptor API, per pooler-splitk2-tma-weight-stream's log) and w = w_desc.load([n0, k]) [BLOCK_N=128, BLOCK_K=128], fed as tl.dot(x_tile, tl.trans(w), acc); x [128, 128] stays a masked tl.load (2 MB, L2-resident). (2) BLOCK_K 128 instead of 64 (256 B contiguous per weight row per stage instead of 128 B), num_stages 3 (64 KB per stage, 192 KB), num_warps 8 (two warpgroups, wgmma m64n128 each). (3) Warp specialization: the K loop is `for k in tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=True)` so a producer warp issues TMA while the consumer warpgroups run wgmma; if Triton 3.8 rejects warp_specialize on sm90 at compile time in prepare(), fall back to the same loop without it and print which variant compiled (a constant fixed in prepare, never input-derived). Grid as before: pooler (n 4096, k 8192) (32, SPLIT_K=4) = 128 CTAs, K-slice 2048; unpooler (8192, 4096) (64, SPLIT_K=2) = 128 CTAs. Reduce tail: master's ticket semaphore (masked fp32 partial store into prepare()-allocated state[layer+'_m128_ws'] [SPLIT_K, 128, n], debug_barrier, one gpu-scope acq_rel ticket per N-tile, debug_barrier, atomic_xchg reset) with one change: the CTA that draws the last ticket does NOT store or reload its own partial; it keeps acc in registers and adds the other SPLIT_K-1 partials (loaded .cg in index order, skipping its own pid_k), then stores bf16 once. That removes one 64 KB store and one 64 KB reload from the critical path (pooler reloads 3 partials instead of 4, unpooler 1 instead of 2). The sum order then depends on which CTA wins; still fp32, well inside the 1% bound. prepare(): build descriptors, allocate workspaces and zeroed int32 counters [n // 128], warm m in {65, 96, 128} per layer on synthetic x, re-zero counters, and print (try/except, diagnostics only) per layer: variant compiled, n_regs, n_spills, PTX counts of wgmma.mma_async and cp.async.bulk.tensor, and warm-L2 event times of the full kernel vs a constexpr REDUCE=False variant (as the parent did), so the log attributes any loss to mainloop or tail. kernel_fn gates on layer, cfg m in 65..128, cfg n/k and x bf16 contiguous; everything else keeps master.

## What changed, and where
Built on worktree master 1221521 (has bf16-m32-swapab-wgmma-triton; the swap-AB gate and every other path are unchanged). The only file touched is `muse_bf16_proj/submission.py`, with +244 lines and nothing deleted:

- New section (6) after swap-AB: `_bf16_m128_tma_ws_kernel`. The tile is BLOCK_M = N = K = 128 with num_warps 8. The weight arrives by TMA as `w_desc.load([n0, k])`. The descriptor is `triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128])` and is built once in prepare() from the weight alone. x is a masked `tl.load` (rows >= M are zero) and the dot is `tl.dot(x_tile, tl.trans(w_tile), acc)`. The K loop is `tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=WARP_SPEC)`, where WARP_SPEC is a constexpr fixed in prepare(). Grid is (N // 128, SPLIT_K): pooler (32, 4), unpooler (64, 2), K-slice 2048.
- Reduce, per the REQUIRED note. EVERY CTA, the winner included, stores its masked fp32 partial into `state[layer+"_m128_ws"]` [SPLIT_K, 128, n]. Then come debug_barrier, one gpu-scope acq_rel ticket per N-tile, and debug_barrier. The last-ticket CTA starts from its in-register `acc` and adds all SPLIT_K slices in index order. Its own slice is masked off with `m_mask & (pid_k != s)`, so it costs no memory traffic and is not reloaded. The CTA then stores bf16 once and resets with atomic_xchg. The spec text said the winner does "NOT store" its own partial. As the review required, I did not implement that: only the reload is skipped. The constexpr REDUCE=False variant is used only by the diagnostics.
- `_m128_prepare` runs per layer. It allocates the descriptor, workspace and int32 counters [n // 128] under its own keys (`<layer>_m128_desc/_ws/_cnt`). It then tries the variants (warp_specialize, num_stages) in the order (T,3), (T,2), (F,3), (F,2). `OutOfResources` steps down to 2 stages. Any other compile or run exception, such as warp_specialize being rejected on sm90, moves to the next variant. The first variant that compiles AND passes `_m128_selfcheck` is fixed in `state["m128_variants"][layer]`. If none passes, the value is None and that layer keeps cuBLAS. After that, prepare() warms m in {65, 96, 128}, prints the diagnostics and re-zeroes the counters. The selected variant is printed on the `[m128_ws <layer>] selected variant:` line.
- The self-check runs on synthetic x at m in {65, 96, 128}, and each m runs twice. The workspace is NaN-poisoned first, so a masked row leaking into out shows up as a failure. Row 0 is a one-hot 2.0 and must be within 1e-3 rel L2. Every row must be within 5e-3 rel L2 against an fp64 reference, the output must be finite, and the counters must read 0 after each launch.
- The diagnostics (try/except, prints only) report n_regs, n_spills, PTX counts of `wgmma.mma_async` and `cp.async.bulk.tensor`, and warm-L2 event times at m=128 for the full kernel vs REDUCE=False (tail = the difference). Per the review, read these as a mainloop-vs-tail ratio and judge M1-M3 on the profile top_kernels.
- kernel_fn gate: after the swap-AB check and before cuBLAS. It requires `state["m128_variants"][layer]` not None, layer in {pooler, unpooler}, 65 <= cfg m <= 128, cfg n/k equal to the layer's, and x a contiguous bf16 [m, k]. m = 130, 256, 512 and 1000 keep cuBLAS.
- Not adopted: the review's optional host TensorDescriptor for x. The spec says x stays a masked tl.load. If the printed variant shows warp_specialize=False, this run measures TMA plus BLOCK_K 128 only, and the x-descriptor is the natural stage-1b follow-up. On the gate (open question 3), I kept 65..128 as specified. The self-check covers 65, 96 and 128.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite diff`:

- [x] CUDA-graph capturable. kernel_fn's new path is one Triton launch plus `torch.empty`. There is no `.item()`, `.cpu()`, sync or print. Branches depend only on cfg, shapes, dtype, contiguity and a prepare()-fixed variant. All `.item()` and synchronize calls are in `_m128_selfcheck`, `_m128_diagnostics` and `_m128_prepare`, which run in prepare() only.
- [x] Nothing derived from `data` is cached. The workspace and counters are allocated in prepare(). Partials are rewritten for rows < m on every call, and the reducer reads only those rows. The descriptor is built from the weight only. No `triton.autotune` is used.
- [x] Weights: the bf16 weights are used as-is (contiguous) through a TMA descriptor. There is no repack and no precision change. The router path is untouched.
- [x] Activations stay bf16 (x as given); accumulation is fp32 and the output bf16, as in cuBLAS. KV-cache / query FP8: n/a, since this problem has no attention.
- [x] Attention sees every key: n/a, since this problem has no attention. No weight-based skipping: every weight tile is read.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The only new import is `triton.tools.tensor_descriptor`.
- [x] All work happens inside the timed call; no output caching or replay. The self-check and warm-ups use synthetic x only, and their outputs are dropped.
- [x] First line `# okc: problem=muse_bf16_proj` is kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
warn  host-sync    L794: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L799: if row_err.max().item() > 5e-3:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L800: return f"m={m} rep={rep}: max row rel L2 {row_err.max().item():.3e}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L801: if row_err[0].item() > 1e-3:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L802: return f"m={m} rep={rep}: one-hot row rel L2 {row_err[0].item():.3e}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L803: if int(cnt.abs().sum().item()) != 0:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L831: end.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L864: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L886: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```

`python3 -I -m py_compile` passed; the __pycache__ was removed.

## Risks / what could fail
high
Big-lever rewrite: the pooler m=128 row is the largest single gap on the board (11.4 us over floor), unpooler 5.7 us. The parent's diagnostics show the loss is the mainloop (26 us warm-L2, 2.6 TB/s with cp.async 128-byte row segments) plus a 4-6.5 us reduce tail, not a missing wgmma. Three mainloop changes are bundled on purpose (rewrite stage, judged on the replaced kernels' time against Milestones, not the headline); the printed variant and mainloop/tail times attribute them. Risks: Triton 3.8's warp_specialize may be Blackwell-only (fallback compiles the plain pipelined loop and says so); LESSONS says TMA added little over deep cp.async at m<=16, but there the stream was BLOCK_K 256 and BW-bound, here the shape is co-bound (128 FLOP per weight byte vs ~206 ridge) and the warps must be freed for wgmma. Overlaps bf16-cublaslt-algo-pick-in-prepare-m128-m256 on these two rows; per-shape rows decide which a later combo keeps. Stage 2 (if M1 met): stream-K over 132 SMs or BLOCK_N 256 with one warpgroup per 128 rows; also extend to qkv_gate/o at m=128 (the held qkv-gate-o-m128-wgmma-splitk-semaphore).

## Result (after the run)
- job `61da9b91dab44bf4` (submit), status done, passed True, tests 9/9
- speedup 1.1639x, 95% CI [1.1465, 1.1814], score 4150.389250283167 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-tma-wspec-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-tma-wspec-rewrite.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite`
