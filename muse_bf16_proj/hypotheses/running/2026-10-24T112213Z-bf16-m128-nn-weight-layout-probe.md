---
problem: muse_bf16_proj
slug: bf16-m128-nn-weight-layout-probe
created: 2026-10-24T112213Z
base: master
area: bf16 GEMM dispatch (m>=128, weight layout)
change: >
  Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are
  where cuBLAS sits furthest from the floor (pooler verify_16 25.2 us vs 14.6,
  nvjet_sm90_tst_64x64_64x13_2x1 with no split-K at ~2.7 TB/s; qkv_gate verify_16 17.0 vs 8.4,
  nvjet 96x64 splitK tile that pads 128 rows to 192 + a 3.1 us splitKreduce; o verify_32 9.5 vs
  4.5; unpooler verify_16 19.7 vs 14.6), and the Triton wgmma replacement lost there
  (pooler-unpooler-m128-wgmma-splitk-semaphore: wgmma emitted, 154 regs, no spills, but its main
  loop alone was 26 us warm-L2 plus a 4-6.5 us semaphore-reduce tail, 29.4/28.7 us vs nvjet
  25.2/20.0). This is a cheap dispatch probe that hands cuBLAS a different heuristic entry instead
  of writing a kernel: in prepare(), for qkv_gate, o, pooler and unpooler build a lossless
  transposed copy state[layer + "_t"] = W.t().contiguous() ([K, N] bf16 row-major; the original
  stays for the m <= 32 paths). In kernel_fn, for those four layers when cfg m >= 128 only,
  replace _bf16_gemm_dispatch_impl(x, W[:n], None) with torch.mm(x, state[layer + "_t"][:, :n]) (n
  is the full width at m >= 128, so the slice is a no-op view; contiguous [m, n] bf16 result, fp32
  accumulation inside cuBLAS as before). cuBLAS then sees an NN problem instead of TN, the weight
  becomes an N-major A operand, and nvjet picks from a different tile/split-K table. No Triton, no
  new kernel; m <= 32 and the router are byte-identical to master. Print once in prepare() the
  chosen kernel is not observable host-side, so read top_kernels in the profile: the attempt file
  must record, per m=128/256 shape, the nvjet kernel name and time under both layouts (master from
  FINDINGS). Gate only on cfg layer and m. Tests m=130 unpooler, m=512 qkv_gate, m=1000 unpooler
  take the NN path (same math).
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "largest gap on the board (25.2 us vs floor 14.6): master nvjet 64x64 non-split tile runs 2.7 TB/s; the NN table at m=128 should pick a larger N tile or split-K (as TN does at m=32: 128x32 splitK, 3.3 TB/s). Target <= 22 us"
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "17.0 vs 8.4 us (2.0x, largest ratio on a heavy shape): the 96-row TN tile wastes a third of the MMA at m=128 and needs a 3.1 us splitKreduce; any 64/128-row NN tile avoids the pad"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: 19.7 vs 14.6 us, nvjet 128x64 at ~3.4 TB/s; probe
  - shapes: layer=o load=verify_16 m=128
    direction: flat
    why: 7.0 vs 4.0 us; k=1024 is short, either layout is prologue-bound; probe
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: near compute floor for pooler/qkv (17.4/9.8 us at 989 TF; nvjet ~580 TF); o 9.5 vs 4.5 is the one that could move; probe, may lose
  - shapes: layer=* load=decode_* m=*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T112447Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T112447Z
base_sha: d7e6aa9aa0faeb496ce10cb8bef5d8c21c203563
base_blob: da55301de1e911aeab09878b0873809f544f436c
branch: hyp/muse_bf16_proj/bf16-m128-nn-weight-layout-probe
worktree: .worktrees/muse_bf16_proj/bf16-m128-nn-weight-layout-probe
mode: null
implemented_at: 2026-10-24T112600Z
impl_commit: 9cc504ee118bf1676c8b8088112d9ce099409c97
submission_sha256: aa17fdce397cf559fb58be2761bd3104a3f9868a3c534f72b9acbd819c9bf77a
norm_sha256: 02569bb6114e44cb416765475308984d6c07f6fdb9a9090bd58bcacd7153e834
submission_blob: 453ddb06377027a864a7611601f4317d21222ec7
lint_warnings: 0
---

# bf16-m128-nn-weight-layout-probe

## Change

Built on master (dba4ffd; router untouched; m <= 32 paths untouched). The m=128/256 shapes are where cuBLAS sits furthest from the floor (pooler verify_16 25.2 us vs 14.6, nvjet_sm90_tst_64x64_64x13_2x1 with no split-K at ~2.7 TB/s; qkv_gate verify_16 17.0 vs 8.4, nvjet 96x64 splitK tile that pads 128 rows to 192 + a 3.1 us splitKreduce; o verify_32 9.5 vs 4.5; unpooler verify_16 19.7 vs 14.6), and the Triton wgmma replacement lost there (pooler-unpooler-m128-wgmma-splitk-semaphore: wgmma emitted, 154 regs, no spills, but its main loop alone was 26 us warm-L2 plus a 4-6.5 us semaphore-reduce tail, 29.4/28.7 us vs nvjet 25.2/20.0). This is a cheap dispatch probe that hands cuBLAS a different heuristic entry instead of writing a kernel: in prepare(), for qkv_gate, o, pooler and unpooler build a lossless transposed copy state[layer + "_t"] = W.t().contiguous() ([K, N] bf16 row-major; the original stays for the m <= 32 paths). In kernel_fn, for those four layers when cfg m >= 128 only, replace _bf16_gemm_dispatch_impl(x, W[:n], None) with torch.mm(x, state[layer + "_t"][:, :n]) (n is the full width at m >= 128, so the slice is a no-op view; contiguous [m, n] bf16 result, fp32 accumulation inside cuBLAS as before). cuBLAS then sees an NN problem instead of TN, the weight becomes an N-major A operand, and nvjet picks from a different tile/split-K table. No Triton, no new kernel; m <= 32 and the router are byte-identical to master. Print once in prepare() the chosen kernel is not observable host-side, so read top_kernels in the profile: the attempt file must record, per m=128/256 shape, the nvjet kernel name and time under both layouts (master from FINDINGS). Gate only on cfg layer and m. Tests m=130 unpooler, m=512 qkv_gate, m=1000 unpooler take the NN path (same math).

## Expected

- `layer=pooler load=verify_16 m=128` faster: largest gap on the board (25.2 us vs floor 14.6): master nvjet 64x64 non-split tile runs 2.7 TB/s; the NN table at m=128 should pick a larger N tile or split-K (as TN does at m=32: 128x32 splitK, 3.3 TB/s). Target <= 22 us
- `layer=qkv_gate load=verify_16 m=128` faster: 17.0 vs 8.4 us (2.0x, largest ratio on a heavy shape): the 96-row TN tile wastes a third of the MMA at m=128 and needs a 3.1 us splitKreduce; any 64/128-row NN tile avoids the pad
- `layer=unpooler load=verify_16 m=128` faster: 19.7 vs 14.6 us, nvjet 128x64 at ~3.4 TB/s; probe
- `layer=o load=verify_16 m=128` flat: 7.0 vs 4.0 us; k=1024 is short, either layout is prologue-bound; probe
- `layer=* load=verify_32 m=256` flat: near compute floor for pooler/qkv (17.4/9.8 us at 989 TF; nvjet ~580 TF); o 9.5 vs 4.5 is the one that could move; probe, may lose
- `layer=* load=decode_* m=*` flat: m <= 32 untouched
- `layer=* load=verify_1 m=8` flat: untouched
- `layer=router *` flat: untouched

## Risk

A probe, like bf16-cublaslt-backend-toggle: direction is genuinely unknown per shape, and it may lose on some (cuBLAS tuned TN most for nn.Linear). It is attributable per shape because only m >= 128 bf16 rows change, and it costs no engineering. Its value is the per-shape table: any m=128/256 shape where NN beats TN by more than drift becomes one more component of a later shape-gated combo (like bf16-shape-gated-triton-winners-combo); shapes where it loses are simply not routed. Also informs the held qkv-gate-o-m128-wgmma-splitk-semaphore (if cuBLAS NN already closes the m=128 qkv gap, that rewrite is not worth it). Extra memory: ~189 MB of transposed bf16 copies, lossless, same precision (allowed repack). Risk: torch.mm may internally detect a transposed-contiguous operand and re-normalize to the same TN call; the implementer must ensure the [K, N] copy is truly contiguous (not a .t() view of W) and confirm from top_kernels that kernel names changed (NN suffix); if every name is unchanged the run is void, not neutral.

## Adversarial review

Reviewed 2026-10-24T112447Z by an adversarial subagent (model opus).

**Verdict: proceed.** A cheap, rules-clean cuBLAS heuristic probe on the m=128/256 bf16 shapes. Those shapes are disjoint from everything landed and in flight, so a single run reads per shape from the kernel names and times. Do not judge it on the headline: the best realistic gain (about 3 us each on pooler and qkv_gate at verify_16) is about 1-1.5% on the geomean, inside drift.

### Justification
The mechanism is real. torch.mm(x, Wt) with a contiguous [K, N] Wt reaches cuBLAS as an NN GEMM: the weight is an MN-major operand with ld=N. Master's x @ W.T reaches it as TN. cuBLASLt keys its nvjet table on layout, so a different tile and split-K choice is plausible. On Hopper, bf16 wgmma takes MN-major shared-memory operands natively, so NN has no inherent MMA penalty. The profile supports the targets. pooler verify_16 moves 67.1 MB in 25.2 us (2.66 TB/s; floor 14.6) on a 64x64 non-split tile. qkv_gate verify_16 moves 37.7 MB in 17.0 us (2.2 TB/s; floor 8.4) on a 96x64 split-K tile plus a 3.1 us splitKreduce, so the 96-row pad and the reduce are real waste. unpooler verify_16 is 19.7 vs 14.6 at about 3.4 TB/s. The verify_32 rows are close to compute-bound: pooler at m=256 is 17.2 GFLOP, a 17.4 us floor, and nvjet's 29.5 us is about 580 TFLOP/s. Calling them flat or probe is consistent with TASK.md's floors. Expected directions match the weights: every shape has weight 62, and only m>=128 rows change, so decode, verify_1 and router must be flat. Any movement there is drift (LESSONS: 1.5-2.5% per shape) and evidence of nothing. Weakest link: nothing says the NN table is better. cuBLAS is tuned hardest for TN, because that is nn.Linear's layout, and the heuristic that picked a 64x64 non-split tile for pooler m=128 may pick the same tile or a worse one under NN. The 'should pick a larger N tile or split-K' claim is a hope, not an inference. Second-weakest: the run is void, not neutral, if the nvjet names keep their TN prefix (e.g. _TNT/_TNN to _NN?). The attempt file must record the names per shape, and the harvest must read top_kernels before classifying.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: on the m>=128 bf16 rows the probe replaces _bf16_gemm_dispatch_impl with torch.mm, so two things change at once: the entry point and the layout. On this image the dispatch lands on the same nvjet kernels as plain x @ W.T (the parity review and the starter-vs-parity qkv_gate rows), so the entry-point part should be inert. The kernel-name change is what attributes the result. No revert is needed.
- router-triton-hilo-fused-control-splitk-all-m and router-hilo-at-m8: router only. Each shape is captured and timed as its own graph with a cold L2, so they share no launch sequence, memory traffic or numerical path with the bf16 m>=128 rows. They do not confound.

### Interference and A/B needs
No companion is warranted. Each landed hypothesis touches only the router or the m<=32/dispatch entry, and the judge times shapes independently with a cold L2, so there is nothing to interact with. Both a per-shape gain and a per-shape loss are readable without a second run, as long as top_kernels shows a changed nvjet name for that shape. Unchanged names mean a void run, and no companion fixes that. The only conceivable control is torch.mm(x, W[:n].t()) to separate the torch.mm entry from the layout. It is redundant, because the kernel name already shows which heuristic entry ran, and spending one of bf16's 4 remaining checks on it is not justified. It also does not need to wait for the in-flight harvests. Router split-K32 and its stages3 control touch the router only. Pooler TMA and the pooler BLOCK_K=256 control touch pooler at m<=32 only. This probe is gated to m>=128 on the four bf16 layers, so the shape sets are disjoint and any winner composes with it by gate later. Dispatch it as a check. Read it per shape, and route only the shapes where NN beats TN by more than about 2% into a later shape-gated combo. Read the qkv_gate m=128 row before unholding qkv-gate-o-m128-wgmma-splitk-semaphore.

### Open questions
- Do the nvjet names at m=128/256 actually change prefix under NN? If cuBLASLt canonicalizes the problem (for example by computing the transposed product) and picks the same kernel, the run is void. Record the names and do not file it as neutral.
- If NN wins on qkv_gate m=128 by removing the 96-row pad and the splitKreduce, the held qkv-gate-o-m128-wgmma-splitk-semaphore loses most of its rationale. Decide that from this run's per-shape row.
- Final review reruns nearby shapes (m=130 unpooler, m=512, m=1000 are already in the tests). If NN wins at m=128 but loses badly at a nearby m, gate tighter than m>=128 in the later combo.


## Implementation notes

Worktree `.worktrees/muse_bf16_proj/bf16-m128-nn-weight-layout-probe`, file `muse_bf16_proj/submission.py` only (14 insertions, 1 deletion, uncommitted).

- New module constants `NN_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")` and `NN_MIN_M = 128`.
- `prepare()`: after the router hi/lo split, `state[name + "_t"] = state[name].t().contiguous()` for each of the four bf16 layers. This is a truly contiguous [K, N] bf16 row-major copy, not a `.t()` view, so cuBLAS sees ld=N and an NN problem. It is lossless and keeps the same precision. The original [N, K] weights stay in state for m < 128. Extra memory is about 189 MB.
- `kernel_fn()`: for non-router layers with `cfg["m"] >= 128`, return `torch.mm(x, state[layer + "_t"][:, : cfg["n"]])`. Below 128 the `_bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)` path is byte-identical to master. The router branch is unchanged. The gate reads only `cfg["layer"]` (non-router means one of the four) and `cfg["m"]`. Tests m=130 unpooler, m=512 qkv_gate and m=1000 unpooler take the NN path.
- Module docstring updated to describe the m >= 128 path.
- Deviations: none. The spec's "Print once in prepare()" sentence is garbled: it says the chosen kernel is not observable host-side and must be read from `top_kernels`. No print was added. The attempt file must record the nvjet names and times per m=128/256 shape under both layouts. If the names are unchanged from master's TN kernels, the run is void, not neutral.

## Rules check

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
