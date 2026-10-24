# muse_bf16_proj · 2026-10-24T112600Z · check · bf16-m128-nn-weight-layout-probe

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T112213Z-bf16-m128-nn-weight-layout-probe.md` (branch `hyp/muse_bf16_proj/bf16-m128-nn-weight-layout-probe`).

- **Job:** `2dc5dce078be4a24` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `9cc504e`

## Goal
Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 us vs 14.6, nvjet_sm90_tst_64x64_64x13_2x1 with no split-K at ~2.7 TB/s; qkv_gate verify_16 17.0 vs 8.4, nvjet 96x64 splitK tile that pads 128 rows to 192 + a 3.1 us splitKreduce; o verify_32 9.5 vs 4.5; unpooler verify_16 19.7 vs 14.6), and the Triton wgmma replacement lost there (pooler-unpooler-m128-wgmma-splitk-semaphore: wgmma emitted, 154 regs, no spills, but its main loop alone was 26 us warm-L2 plus a 4-6.5 us semaphore-reduce tail, 29.4/28.7 us vs nvjet 25.2/20.0). This is a cheap dispatch probe that hands cuBLAS a different heuristic entry instead of writing a kernel: in prepare(), for qkv_gate, o, pooler and unpooler build a lossless transposed copy state[layer + "_t"] = W.t().contiguous() ([K, N] bf16 row-major; the original stays for the m <= 32 paths). In kernel_fn, for those four layers when cfg m >= 128 only, replace _bf16_gemm_dispatch_impl(x, W[:n], None) with torch.mm(x, state[layer + "_t"][:, :n]) (n is the full width at m >= 128, so the slice is a no-op view; contiguous [m, n] bf16 result, fp32 accumulation inside cuBLAS as before). cuBLAS then sees an NN problem instead of TN, the weight becomes an N-major A operand, and nvjet picks from a different tile/split-K table. No Triton, no new kernel; m <= 32 and the router are byte-identical to master. Print once in prepare() the chosen kernel is not observable host-side, so read top_kernels in the profile: the attempt file must record, per m=128/256 shape, the nvjet kernel name and time under both layouts (master from FINDINGS). Gate only on cfg layer and m. Tests m=130 unpooler, m=512 qkv_gate, m=1000 unpooler take the NN path (same math).

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-m128-nn-weight-layout-probe`, file `muse_bf16_proj/submission.py` only (14 insertions, 1 deletion, uncommitted).

- New module constants `NN_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")` and `NN_MIN_M = 128`.
- `prepare()`: after the router hi/lo split, `state[name + "_t"] = state[name].t().contiguous()` for each of the four bf16 layers. This is a truly contiguous [K, N] bf16 row-major copy, not a `.t()` view, so cuBLAS sees ld=N and an NN problem. It is lossless and keeps the same precision. The original [N, K] weights stay in state for m < 128. Extra memory is about 189 MB.
- `kernel_fn()`: for non-router layers with `cfg["m"] >= 128`, return `torch.mm(x, state[layer + "_t"][:, : cfg["n"]])`. Below 128 the `_bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)` path is byte-identical to master. The router branch is unchanged. The gate reads only `cfg["layer"]` (non-router means one of the four) and `cfg["m"]`. Tests m=130 unpooler, m=512 qkv_gate and m=1000 unpooler take the NN path.
- Module docstring updated to describe the m >= 128 path.
- Deviations: none. The spec's "Print once in prepare()" sentence is garbled: it says the chosen kernel is not observable host-side and must be read from `top_kernels`. No print was added. The attempt file must record the nvjet names and times per m=128/256 shape under both layouts. If the names are unchanged from master's TN kernels, the run is void, not neutral.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-m128-nn-weight-layout-probe diff`:

- [x] CUDA-graph capturable: torch.mm on preallocated state tensors. No `.item()`, `.cpu()`, `synchronize()` or prints in kernel_fn. The branch is on `cfg["m"]` and `cfg["layer"]` (config) only.
- [x] Nothing derived from `data` is cached across calls. The `_t` copies are built from the weights alone in prepare().
- [x] Weights: a lossless repack in prepare() only. A transposed copy stays BF16, with no precision change. The router hi+lo split is unchanged.
- [x] Activations stay BF16: x goes into torch.mm as bf16, and the output is bf16 [m, n] with fp32 accumulation inside cuBLAS.
- [x] n/a: no attention in this problem. KV cache, queries and key windows do not apply.
- [x] No timing patches, harness frames, I/O or secrets, module edits, reference import, subprocesses or network. Only `torch.mm` was added.
- [x] All work happens inside the timed call, with no output caching or replay. The GEMM runs every call.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept. No triton.autotune was added.

okc_lint output:

```
CLEAN: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` OK. The `__pycache__` it left was removed from the worktree.

## Risks / what could fail
medium
A probe, like bf16-cublaslt-backend-toggle: direction is genuinely unknown per shape, and it may lose on some (cuBLAS tuned TN most for nn.Linear). It is attributable per shape because only m >= 128 bf16 rows change, and it costs no engineering. Its value is the per-shape table: any m=128/256 shape where NN beats TN by more than drift becomes one more component of a later shape-gated combo (like bf16-shape-gated-triton-winners-combo); shapes where it loses are simply not routed. Also informs the held qkv-gate-o-m128-wgmma-splitk-semaphore (if cuBLAS NN already closes the m=128 qkv gap, that rewrite is not worth it). Extra memory: ~189 MB of transposed bf16 copies, lossless, same precision (allowed repack). Risk: torch.mm may internally detect a transposed-contiguous operand and re-normalize to the same TN call; the implementer must ensure the [K, N] copy is truly contiguous (not a .t() view of W) and confirm from top_kernels that kernel names changed (NN suffix); if every name is unchanged the run is void, not neutral.

## Result (after the run)
- job `2dc5dce078be4a24` (check), status done, passed True, tests 9/9
- speedup 1.1580x, 95% CI [1.1406, 1.1754], score 4171.644064994987 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-m128-nn-weight-layout-probe.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-m128-nn-weight-layout-probe.patch`; branch `hyp/muse_bf16_proj/bf16-m128-nn-weight-layout-probe`
