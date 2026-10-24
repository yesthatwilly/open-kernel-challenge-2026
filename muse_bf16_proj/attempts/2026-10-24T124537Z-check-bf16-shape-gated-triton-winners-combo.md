# muse_bf16_proj · 2026-10-24T124537Z · check · bf16-shape-gated-triton-winners-combo

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T112147Z-bf16-shape-gated-triton-winners-combo.md` (branch `hyp/muse_bf16_proj/bf16-shape-gated-triton-winners-combo`).

- **Job:** `875f3e0172ee48a1` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `5aff367`

## Goal
Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neutral Triton kernels ONLY on the shapes where each one's own kernel beat cuBLAS in its own run (paired ratio vs SGLang > 1 and kernel time in top_kernels below nvjet+splitKreduce); every other shape keeps _bf16_gemm_dispatch_impl. Code is copied verbatim from the harvested patches in muse_bf16_proj/hypotheses/done/neutral/: (1) pooler, cfg m <= 16 (decode_1, verify_1, decode_16; NOT m=32, which read 0.994x): _bf16_splitk_kernel + _pooler_forward from pooler-triton-splitk2-blockn64-semaphore.patch (grid (64, 2), BLOCK_N=64, BLOCK_K=128, SPLIT_K=2, 4 warps, 4 stages, BLOCK_M=16, ticket-semaphore last-CTA reduce, ws fp32 [2, 16, 4096], cnt int32 [64]); change POOLER_MAX_M to 16. (2) qkv_gate, cfg m <= 16 (NOT m=32, which read 0.858x): the same _bf16_splitk_kernel (it is the same design as _qkv_splitk_sem_kernel in qkv-gate-triton-splitk4-blockn64-semaphore.patch; either kernel body is acceptable, the constants must match that patch: N=2304, grid (36, 4), SPLIT_K=4, BLOCK_N=64, BLOCK_K=128, 4 warps, 4 stages, BLOCK_M=16), its own ws fp32 [4, 16, 2304] and cnt int32 [36]. (3) unpooler, cfg n == 1024 (the sliced case, m <= 8: decode_1, verify_1): _bf16_splitk_kernel launch from unpooler-slice-triton-splitk8-semaphore.patch (w = state[unpooler][:1024], stride_wn 4096, grid (16, 8), SPLIT_K=8, BLOCK_N=64, BLOCK_K=128, 4 warps, 3 stages, BLOCK_M=16), its own ws fp32 [8, 16, 1024] and cnt int32 [16]. Unpooler at n == 8192 stays cuBLAS (its Triton m=16 row was only 1.014-1.033x, m=32 lost). (4) o, cfg m <= 16 (decode_1, verify_1, decode_16; NOT m=32, which read 0.907x): _pooler_skinny_kernel from pooler-triton-skinny-m32.patch launched as in o-triton-skinny-m32.patch (full K=1024, no split, grid (8192 // 64,) = 128 programs, BLOCK_N=64, BLOCK_K=256, 4 warps, 4 stages, BLOCK_M=16). Keep separate ws/cnt tensors per layer (no sharing) so each path is byte-for-byte the measured one. prepare(): allocate all ws/cnt, warm every routed specialization on synthetic bf16 x for m in 1..16 (pooler, qkv_gate, o) and 1..8 (unpooler slice), then zero_() every cnt. Route by cfg layer/m/n and the shape preconditions only (no data-dependent branch). The 9 tests: m=1 qkv_gate, m=7 pooler, m=5 unpooler (slice) take Triton paths; m=33 o, m=130 unpooler, m=512 qkv_gate, m=1000 unpooler stay cuBLAS. If pooler-splitk2-tma-weight-stream or pooler-triton-splitk2-blockn64-semaphore-control-blockk256 is harvested as a gain before this is claimed, use that pooler variant in (1) and say so in the attempt file.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-shape-gated-triton-winners-combo`, `muse_bf16_proj/submission.py` only (+372 lines, no deletions). The router code, the `_bf16_gemm_dispatch_impl` fallback and the first-line marker are untouched.

**Pooler variant choice.** The review said to choose by the m<=16 pooler rows, not by "harvested as a gain". These are pooler m in {1, 8, 16} from the three done/neutral result.json files:

| variant | ratio vs SGLang d1 / v1 / d16 | median_ns d1 / v1 / d16 (sum) | kernel us d1 / v1 / d16 (sum) |
|---|---|---|---|
| pooler-splitk2-tma-weight-stream | 1.1147 / 1.1079 / 1.1021 | 19148 / 19352 / 19483 (57983) | 18.7 / 20.4 / 20.1 (59.2) |
| ...-control-blockk256 (BLOCK_K 256) | 1.1142 / 1.0972 / 1.0871 | 19229 / 19613 / 19649 (58491) | 18.8 / 19.5 / 19.6 (57.9) |
| ...-semaphore (BLOCK_K 128, named in the spec) | 1.0712 / 1.0678 / 1.0590 | 20194 / 19868 / 20669 (60731) | 19.5 / 20.0 / 19.9 (59.4) |

TMA leads BLOCK_K 256 by about 0.8% on ratio and 0.9% on median time, but its kernel time is higher (59.2 vs 57.9 us). That is within ~1%, so per the brief the non-TMA variant wins: **pooler slot (1) uses pooler-triton-splitk2-blockn64-semaphore-control-blockk256**. That means POOLER_BLOCK_K = 256, with the rest as the spec says: grid (64, 2), BLOCK_N 64, SPLIT_K 2, 4 warps, 4 stages, BLOCK_M 16. The blockk256 and TMA runs were on base cee590d; the BLOCK_K 128 run was on 3fe0dcc.

**What was added.** Each part is copied from branch `hyp/muse_bf16_proj/<slug>`. AST comparison shows each matches its source apart from docstrings. The one exception is `_pooler_forward` (see deviations).
- `_bf16_splitk_kernel`: one definition, used by both pooler and unpooler. The blockk256 and unpooler-slice branches have identical bodies for it.
- (1) pooler:
  - `POOLER_*` constants with `POOLER_MAX_M = 16`, plus `_pooler_block_m`, `_pooler_shape_ok` and `_pooler_forward`.
  - ws fp32 [2, 16, 4096], cnt int32 [64]; warmed for m 1..16.
- (2) qkv_gate:
  - `_qkv_splitk_sem_kernel`, `_qkv_forward` and `_qkv_route_ok`, verbatim from qkv-gate-triton-splitk4-blockn64-semaphore, with `QKV_MAX_M = 16`. Using its own kernel body keeps the path byte-for-byte the measured one.
  - ws fp32 [4, 16, 2304] (strides are launch args), cnt int32 [36].
  - grid (36, 4), SPLIT_K 4, BLOCK_N 64, BLOCK_K 128, 4 warps, 4 stages; warmed on zeros for m 1..16.
- (3) unpooler slice:
  - `UNPOOL_*`, `_unpool_shape_ok` and `_unpool_forward`, verbatim.
  - n == 1024 only. w is the full unpooler weight read in place (stride_wn 4096).
  - grid (16, 8), SPLIT_K 8, BLOCK_K 128, 3 stages, BLOCK_M 16.
  - ws fp32 [8, 16, 1024], cnt int32 [16]; warmed for m 1..8.
- (4) o:
  - `_pooler_skinny_kernel` (from pooler-triton-skinny-m32 via o-triton-skinny-m32), plus `_o_skinny_applies`, `_o_skinny_gemm` and `_o_skinny_warmup`, with `O_SKINNY_MAX_M = 16`.
  - grid (128,), BLOCK_N 64, BLOCK_K 256, full K 1024, 4 warps, 4 stages; warmed on zeros for m 1..16.
- `prepare()`: allocates the three ws/cnt pairs (separate per layer) and runs the four warm-ups after the router warm-up. It then calls `zero_()` on every cnt.
- `kernel_fn()`: one shape gate placed before the existing cuBLAS fallback, in the order pooler -> qkv_gate -> unpooler(n==1024) -> o.
  - Each branch is gated by cfg layer/m/n/k and by the x/weight shape, dtype and stride preconditions only.
  - The 9 tests route as the spec says. m=1 qkv_gate, m=7 pooler and m=5 unpooler (n=1024) go to Triton. m=33 o, m=130 and m=1000 unpooler (n=8192), m=512 qkv_gate and the router tests are unchanged.
- Module docstring: added one paragraph that lists the four routed paths.

**Deviations:**
- Pooler uses the BLOCK_K 256 control variant instead of the BLOCK_K 128 one the spec names. The spec allows the swap, the review asked for it, and the table above shows why.
- The blockk256 branch's compile diagnostics were dropped: `_pooler_compile_diag`, the `diag=` return of the launch handle, and the print at m 16/32. That code ran only in prepare(), and the launch arguments are unchanged.
- The unpooler warm set changed from (1, 5, 8) to 1..8, as the spec asks. M is not specialized, so the compiled binary is the same.
- qkv_gate uses its own `_qkv_splitk_sem_kernel` instead of sharing `_bf16_splitk_kernel`; the spec accepts either. Its ws is [4, 16, 2304] instead of the branch's [4, 32, 2304]. Only the `stride_ws_s` launch argument changes.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-shape-gated-triton-winners-combo diff`:

- [x] CUDA-graph capturable: no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The only allocations in kernel_fn are `torch.empty` outputs whose size comes from the shape, as in the existing paths. Every branch reads cfg, shape, dtype and stride only. The ticket counters reset themselves in-kernel, so each graph replay starts at 0. The diff adds no print.
- [x] Nothing derived from `data` is cached across calls. ws holds per-call fp32 partials. Each launch fully rewrites them for rows < m and reads them back under the same mask. cnt holds only tickets. Neither is read as data across calls.
- [x] Weights: no repack. All four paths read the original bf16 `state[layer]` tensors; the unpooler slice is read in place through its row stride. The router hi+lo split is unchanged from master.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16, accumulation is fp32, outputs are bf16. The KV-cache and FP8-query parts do not apply because this problem has no attention.
- [x] n/a (no attention in this problem). Every path still computes the full K reduction over every weight row.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only imports are the existing torch, triton and sglang ones.
- [x] All work happens inside the timed call, with no output caching or replay. The prepare() warm-ups use synthetic randn/zeros x and drop their outputs.
- [x] No `triton.autotune`. The first-line marker `# okc: problem=muse_bf16_proj` is kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-shape-gated-triton-winners-combo/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed, and the `__pycache__` it left in the worktree was removed.

## Risks / what could fail
medium
Per-load arithmetic from the measured rows: decode 1 saves ~3.5 us of 55 (pooler 1.2, unpooler 1.3, qkv 0.6, o 0.4), verify 1 ~2.5 of 54, decode 16 ~1.5 of 66, the other three loads unchanged: geomean ~1.02x over master, ~1.17-1.18x headline (pooler split-K2 alone already read 1.166x). Each component was within the CI alone; together the summed per-shape wins should clear it. Risks: (a) the qkv_gate and o-verify_1 wins are 2-4%, near the 1.5-2.5% per-shape drift, so they may read flat; (b) three ticket-semaphore kernels in one file: each has its own counters and self-resets, and a missed reset would hang or corrupt only its own layer (the tests catch it); (b2) o-triton-skinny-m32 was measured on an older base (d2840d, pre-router-split-K), the others on dba4ffd-era bases; the o kernel is independent of the router, so this should not matter; (c) per-shape rows are the evidence: if a component reads below 1.0 vs SGLang here, the attribution names it and the next run drops it. A reviewer may want an ab-revert dropping qkv_gate (the weakest component).

## Result (after the run)
- job `875f3e0172ee48a1` (check), status done, passed True, tests 9/9
- speedup 1.1760x, 95% CI [1.1584, 1.1937], score 4107.658858054889 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-shape-gated-triton-winners-combo.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-shape-gated-triton-winners-combo.patch`; branch `hyp/muse_bf16_proj/bf16-shape-gated-triton-winners-combo`
