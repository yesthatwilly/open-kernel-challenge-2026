---
problem: muse_bf16_proj
slug: bf16-m16-swapab-wgmma-extend
created: 2026-10-24T165012Z
base: master
area: bf16 GEMM dispatch (m<=16, swap-AB Triton)
change: >
  Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the
  tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate
  _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024,
  m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync;
  at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and
  +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's
  existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K,
  BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8
  run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's
  swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K
  256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36,
  4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K
  4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8,
  BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new
  prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n //
  64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE
  master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k,
  or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full
  weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in
  1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per
  layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX
  counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents
  are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3
  (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for
  these shapes.
expected:
  - shapes: layer=pooler load=decode_* m=1
    direction: faster
    why: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
  - shapes: layer=pooler m=16
    direction: faster
    why: 19.5 us kernel vs 14.1 floor; same mechanism
  - shapes: layer=pooler m=8
    direction: faster
    why: 19.3 us kernel vs 14.0 floor
  - shapes: layer=qkv_gate m=1
    direction: faster
    why: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
  - shapes: layer=qkv_gate m=8
    direction: faster
    why: 12.0 us kernel vs floor 7.9
  - shapes: layer=qkv_gate m=16
    direction: faster
    why: 12.3 us kernel vs floor 7.9
  - shapes: layer=o m=1
    direction: faster
    why: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
  - shapes: layer=o m=8
    direction: faster
    why: 5.2 us kernel vs floor 3.5
  - shapes: layer=o m=16
    direction: faster
    why: 4.8 us kernel vs floor 3.6
  - shapes: layer=unpooler m=1
    direction: faster
    why: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
  - shapes: layer=unpooler m=8
    direction: faster
    why: slice kernel 4.6 us vs floor 1.8
  - shapes: layer=unpooler m=16
    direction: flat
    why: full weight stays cuBLAS nvjet
  - shapes: "* m=32"
    direction: flat
    why: master swap-AB path untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T165449Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T165449Z
base_sha: e82b94f9414d0fe3381c39c7722efd47759f3360
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend
worktree: .worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend
mode: submit
implemented_at: 2026-10-24T165742Z
impl_commit: bdeadf44e3811fcedbb18e1d3f7816b41e677f3c
submission_sha256: 842aaea331cffb88f97ec5e3190afa3600a44f7ed9415d1916aa236ce16b852e
norm_sha256: 6c67d2184cdc3340f8b7929330d864969d666b4db1f733264cfe77476ce97d91
submission_blob: bb41afb76ea0a0dea82ed0246aa441d6b3922a19
lint_warnings: 0
attempt_file: 2026-10-24T165742Z-submit-bf16-m16-swapab-wgmma-extend.md
prepared_at: 2026-10-24T165742Z
job: 89fabef70bd745f7
dispatched_at: 2026-10-24T165751Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2115489260435024
speedup_ci: [1.1933756921528498, 1.2297221599341548]
harvested_at: 2026-10-24T173232Z
best_at_harvest: 1.1958517819004566
promotion: null
---

# bf16-m16-swapab-wgmma-extend

## Change

Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024, m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync; at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K, BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8 run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K 256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36, 4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K 4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8, BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n // 64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k, or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in 1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3 (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for these shapes.

## Expected

- `layer=pooler load=decode_* m=1` faster: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
- `layer=pooler m=16` faster: 19.5 us kernel vs 14.1 floor; same mechanism
- `layer=pooler m=8` faster: 19.3 us kernel vs 14.0 floor
- `layer=qkv_gate m=1` faster: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
- `layer=qkv_gate m=8` faster: 12.0 us kernel vs floor 7.9
- `layer=qkv_gate m=16` faster: 12.3 us kernel vs floor 7.9
- `layer=o m=1` faster: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
- `layer=o m=8` faster: 5.2 us kernel vs floor 3.5
- `layer=o m=16` faster: 4.8 us kernel vs floor 3.6
- `layer=unpooler m=1` faster: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
- `layer=unpooler m=8` faster: slice kernel 4.6 us vs floor 1.8
- `layer=unpooler m=16` flat: full weight stays cuBLAS nvjet
- `* m=32` flat: master swap-AB path untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched
- `layer=router *` flat: untouched

## Risk

At m<=16 these shapes are HBM-bound, not MMA-bound, so the m=32 gain (which removed 255-reg tokens-on-M spills) may not transfer: m=16 incumbents already run 96-ish regs. The plausible win is the async wgmma letting more loads stay in flight and fewer registers per CTA; the prepare() PTX/regs print attributes it either way. wgmma n16 wastes nothing at m=16 but computes 15 dead columns at m=1 (free under the BW bound). Per-shape judging: if only some layers gain, a combo keeps those (as with bf16-shape-gated-triton-winners-combo).

## Adversarial review

Reviewed 2026-10-24T165449Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean, attributable A/B. Each of the four m<=16 incumbents already uses the same grid, BLOCK_K, stage count and cp.async depth as the proposed swap-AB config, so tl.dot orientation (mma.sync+ldmatrix vs wgmma) is the only variable, and the 9 tests exercise the new path (m=1 qkv, m=7 pooler, m=5 unpooler slice). The archived evidence predicts flat for pooler and qkv_gate, with o at m=8 and the unpooler slice as the only real candidates, so judge it per shape on top_kernels, not on the headline.

### Justification
The mechanism is plausible in form: Triton gives tokens-on-M at BLOCK_M=16 mma.sync, and a 64-row weight tile on M with 16 tokens on N can lower to wgmma m64n16k16. Triton's MMAv3 rule (N%8==0, num_warps=4) allows that. But the evidence is weaker than the hypothesis claims, on three points. (1) The archived m<=16 incumbent diagnostics (pooler diag log) show n_regs=96, 0 spills, 120 KB smem, cp.async=90. Those are the same registers and bytes in flight as the swap-AB m=32 kernel (n_regs=96, cp.async=106, log 1602a4e947e64090). The 255-register pathology the m=32 win removed was the TMA variant only, and it does not exist at m<=16, as the Risk section itself concedes. (2) '+9.3% with the identical grid' was measured against SGLang's nvjet+splitKreduce at m=32 (21.97 -> 20.00 us shape time), not against the same-grid tokens-on-M Triton kernel at m<=16. (3) The cross-m kernel times from the same profiles predict flat. The swap-AB pooler ran 20.5 us of kernel at m=32 against the incumbent's 19.0/19.3/19.5 at m=1/8/16, and shape medians were 20.00 vs 19.85 at m=16. qkv_gate ran 12.1 us swap-AB at m=32 vs 11.8-12.3 incumbent. o ran 4.5 us vs 4.6/5.2/4.8. x bytes at m=32 are about 0.5 MB on a 67 MB stream, so halving tokens should change little. The pooler 'target <= 18 us' (3.7 TB/s) and the qkv/pooler 'faster' rows are therefore not supported; the most likely result is flat, within about 1 us of the incumbent. The only real candidates are o at m=8, whose 5.2 us is an outlier against 4.6 at m=1, and the unpooler slice. The slice is the furthest from its floor (4.5 us for 8.4 MB), but there the cost looks like fixed latency plus the 8-way split-K reduce rather than MMA issue, so wgmma may not help either. The expected directions agree with TASK.md: every changed shape is m in {1,8,16}, weight 62 in decode_1/decode_16/verify_1, and unpooler m=16 (full width, cuBLAS) and m>=17 are untouched. WEAKEST LINK: the assumption that the mma.sync+ldmatrix issue path, rather than HBM delivery at about 3.5 TB/s with 128 KB in flight per CTA, binds these streams at m<=16. The m=32 kernel times argue against it for the pooler and qkv_gate.

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (658d63b): this is the incumbent being replaced on exactly these shapes (pooler/qkv_gate/o at m<=16, unpooler slice). It is the control, not a confounder. Its per-layer constants (grid, BLOCK_K, stages, SPLIT_K) are copied into SWAPAB16_CFG, so a difference in top_kernels times comes from the orientation/MMA path. If that combo had used other tiles, the comparison would mix tile and orientation effects; as written it does not.
- bf16-m32-swapab-wgmma-triton (8b9ad43): this supplies the kernel and is the evidence base. Shared-state risk only: the new path must use its own ws/cnt keys and its own WS_M=16. Master's _swapab_forward hardcodes WS_M=SWAPAB_MAX_M (32) and BLOCK_T=SWAPAB_BLOCK_T (32), and _swapab_route_ok requires 17<=m and w.shape==(n,k). If the build reuses those helpers unchanged, the [SPLIT_K,16,n] workspace gets addressed with a 32-row stride, which is an out-of-bounds/overlapping write and a correctness failure, and the unpooler slice (full 8192x4096 weight, n=1024) fails the shape check. These are not measurement confounders, but they must be parametrized.
- sglang-gemm-dispatch-parity / router hypotheses: these cover a different layer or different shapes (m>=17 bf16 fallback, router), and the comparison is a paired ratio against SGLang in the same run. Not a confounder.

### Interference and A/B needs
No companion is warranted. (i) ab-revert of the shape-gated combo would compare swap-AB against cuBLAS, which the combo's own run already measured (5-17% Triton wins), so it adds nothing. (ii) ab-tune: the swap-AB config deliberately equals the incumbent's per layer, and tuning BLOCK_K or stages is a follow-up to a win, not something needed to read this result. (iii) control: the one sub-question, whether wgmma was actually emitted at n16 and at what register count, is answered by the prepare() PTX/regs print for both the new and the incumbent kernels in the same log. The shapes need no second run to read: Triton kernel times reproduce within about 1% (LESSONS), so per-shape top_kernels against the incumbent's archived times (pooler 19.0/19.3/19.5, qkv 11.8/12.0/12.3, o 4.6/5.2/4.8, unpooler slice 4.5/4.6 us) decide each layer independently. Faster rows go into a combo, slower rows revert to master's kernel. The headline will probably sit inside the ±1.5% CI and should not be the deciding signal. With bf16 checks spent and this going out as a scored submit, spending a second scored run on a companion is not justified.

### Open questions
- Does Triton emit wgmma for N=16 (m64n16k16) with the [BLOCK_K,16] K-major B tile, or fall back to mma.sync? The m=32 log only proves n32. If the PTX shows mma.sync, the run is a no-op that only says Triton will not reach wgmma at n16.
- Unpooler slice: SPLIT_K=8 with K_SLICE=512 and BLOCK_K=128 is only 4 K-steps per CTA at 3 stages, so pipeline fill and the transposed fp32 reduce dominate. Will the wgmma path's prologue and the epilogue layout conversion through smem make it slower than the 4.5 us incumbent?
- o at m=8 (5.2 us incumbent vs 4.6 at m=1) is the most likely real gain. Is that m=8 bump an incumbent artifact (for example BLOCK_M=16 masked load behaviour) that any non-M-specialized compile would remove? If so, attribute it to the compile, not to wgmma per se.
- With do_not_specialize=['M'], m=1..16 share one compile per layer, but Triton still specializes stride_xm/stride_om by divisibility. Confirm the warm-up at m=1 and at m=16 hits the same cache entry, so the timed call never compiles.
- If pooler and qkv_gate read flat or slower, as the m=32 kernel times predict, keep master's kernels for them in the follow-up combo and route only the layers that gained, with no rerun of identical code.

## Implementation notes

Worktree `.worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend`, `muse_bf16_proj/submission.py` only. The diff only adds lines (+203, 0 removed); every line of master's code is still there.

- New section `(6)` after `_swapab_prepare`:
  - `SWAPAB16_CFG` uses the keys `pooler` (4096, 8192, SK 2, BK 256, 4 st), `qkv_gate` (2304, 8192, SK 4, BK 128, 4 st), `o` (8192, 1024, SK 1, BK 256, 4 st) and `unpooler_slice` (1024, 4096, SK 8, BK 128, 3 st).
  - `SWAPAB16_BLOCK_T = SWAPAB16_WS_M = 16`.
- `_swapab16_forward` is a separate launcher for the unchanged `_bf16_swapab_kernel`. It passes `WS_M=16` and `BLOCK_T=16`, as the review REQUIRED. Master's `_swapab_forward`, which hardcodes 32/32, is not reused and not modified. The unpooler slice passes the full `[8192, 4096]` weight in place (`stride_wn = w.stride(0) = 4096`), with grid `(1024 // 64, 8)`, so only the first 1024 rows are read.
- `_swapab16_route_ok` is a separate gate. Master's `_swapab_route_ok` (m >= 17, full weight shape) is not used here.
  - It requires `1 <= cfg m <= 16`, the key's cfg n/k, `x.shape == (m, k)`, x bf16 and `stride(1) == 1`, and w bf16 with `stride(1) == 1`.
  - pooler, qkv_gate and o also need `w.shape == (n, k)`. The unpooler slice needs `w.shape[0] >= 1024` and `w.shape[1] == 4096`.
  - `_swapab16_key` maps the unpooler to `unpooler_slice` only when `cfg n == 1024`. So the unpooler's full width at m = 16 (cuBLAS) and m = 130 / 1000 keep master.
- Buffers have their own keys: `swapab16_ws_{pooler,qkv_gate,unpooler_slice}` fp32 `[SPLIT_K, 16, n]` and `swapab16_cnt_*` int32 `[n // 64]`, zeroed. Master's `swapab_*`, `pooler_*`, `qkv_*` and `unpool_*` buffers are untouched by the routed path.
- `_swapab16_prepare`:
  - It warms m = 1..16 per layer (1..8 for the slice) on synthetic randn x. The first launch is at the top m with `report=True`.
  - On OutOfResources it steps down to 3 stages (`SWAPAB_FALLBACK_STAGES`), and records `state["swapab16_stages"]`.
  - It then calls `_incumbent16_report`, which relaunches master's four incumbent m <= 16 kernels once each at m = 16 on zero x, with the same constants and their own buffers, and prints `n_regs / n_spills / wgmma / mma.sync / cp.async` as `[incumbent16] ...` lines. After that it re-zeroes every counter it touched.
  - The new path prints `[swapab16] <key> m=...: ...` lines.
- `kernel_fn` checks the new gate first, before master's m <= 16 branches. Master's m <= 16 kernels stay in the file but are no longer reached for these shapes.
- The module docstring has a paragraph describing the new path.
- Deviation 1: the slice gate accepts `1 <= m <= 16` rather than only m <= 8. `cfg n == 1024` only occurs at m <= 8. The 16-row workspace covers m <= 16, and M is not specialized (`do_not_specialize=["M"]`; `stride_xm = k` and `stride_om = n` are shape constants), so any m 9..16 would hit the same compile. The warm-up is 1..8 as specified.
- Deviation 2: the incumbent PTX report relaunches the kernels directly instead of changing master's forward helpers to return the compiled handle. This keeps master's functions byte-identical.
- Open questions from the review:
  - wgmma at n16: the `[swapab16]` lines answer it in the log.
  - Same cache entry for m = 1 and m = 16: yes by construction. M is in `do_not_specialize` and the strides are shape constants. The full 1..16 warm-up also guarantees no compile in a timed call even if Triton specialized further.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend diff`.

- [x] CUDA-graph capturable. There is no `.item()`, `.cpu()`, `synchronize()` or tensor print in `kernel_fn` or `_swapab16_forward`. Prints are only in prepare(), and they print compile metadata, not tensors. The only allocation in the timed path is `torch.empty` of the output, whose shape comes from `x.shape` and the static cfg, never from input values. Branches are on cfg layer, m, n, k and on shapes, dtypes and strides.
- [x] Nothing derived from `data` is cached across calls. The workspaces are overwritten every call, and the reducer reads only token rows < M written in the same launch. The counters self-reset. Warm-ups use synthetic x.
- [x] Weights: no repack and no precision change. bf16 weights are read in place, and the unpooler slice reads the full weight's first 1024 rows via its stride. The router is untouched.
- [x] Activations stay bf16. Accumulation is fp32 in the MMA. Split-K partials are fp32, summed in fixed order 0..SPLIT_K-1, and stored bf16 once, the same as the incumbent.
- [x] n/a: attention. This problem has no attention, KV cache or queries.
- [x] No timing patches, harness frames or I/O, secrets, module edits, reference import, subprocesses or network. The diff only adds Triton launches, torch allocations and prepare-time prints.
- [x] All work happens inside the timed call, with no output caching or replay. Each call launches the full GEMM on its fresh x.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept (line 1 unchanged).
- [x] No `triton.autotune`. The config is static per layer key.
- [x] Workspace sizing: the `[SPLIT_K, 16, n]` workspace is indexed with `WS_M=16` and the tokens are masked `< M <= 16`, so nothing is written out of bounds. This is the review's REQUIRED fix.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend/muse_bf16_proj/submission.py
lint rc=0
```

`python3 -I -m py_compile` passed, and the leftover `__pycache__` was removed from the worktree.

## Result

- job `89fabef70bd745f7` (submit), status done, passed True, tests 9/9
- speedup 1.2115x, 95% CI [1.1934, 1.2297], score 3987.29254440901 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-m16-swapab-wgmma-extend.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-m16-swapab-wgmma-extend.patch`; branch `hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.91 | 12.46 | 1.03 | gained |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 6.27 | 1.0661 | gained |
| layer=pooler load=decode_1 m=1 | 62 | 19.0 | 18.95 | 1.0318 | gained |
| layer=unpooler load=decode_1 m=1 | 62 | 5.94 | 5.46 | 1.0696 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.73 | 5.9 | 0.9837 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.39 | 13.28 | 1.0211 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.66 | 6.49 | 1.0728 | gained |
| layer=pooler load=decode_16 m=16 | 62 | 19.85 | 19.49 | 1.0243 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.61 | 18.39 | 0.9977 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.03 | 6.41 | 0.957 | lost |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.69 | 14.2 | 0.973 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.67 | 6.73 | 1.0076 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 20.0 | 19.66 | 1.0066 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.48 | 18.57 | 1.0069 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.55 | 6.81 | 0.9793 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.18 | 12.67 | 1.0239 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.32 | 6.08 | 1.0539 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 19.14 | 18.79 | 1.0272 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.13 | 5.84 | 1.0473 | gained |
| layer=router load=verify_1 m=8 | 62 | 5.91 | 5.8 | 1.0174 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.81 | 17.49 | 0.9773 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.09 | 7.2 | 0.9889 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.55 | 25.13 | 1.0095 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.79 | 19.8 | 1.0053 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.35 | 7.4 | 0.9867 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.62 | 19.59 | 0.9847 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.01 | 9.22 | 0.9961 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.53 | 30.83 | 0.9788 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.5 | 23.89 | 1.0113 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.3 | 9.18 | 1.0084 | flat |

gained on qkv_gate decode_1; o decode_1, decode_16, verify_1; pooler decode_1; unpooler decode_1, verify_1; lost on router decode_16. Expected hit: 5/16. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-m16-swapab-wgmma-extend`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

