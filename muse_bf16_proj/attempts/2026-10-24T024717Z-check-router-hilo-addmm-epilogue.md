# muse_bf16_proj · 2026-10-24T024717Z · check · router-hilo-addmm-epilogue

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225128Z-router-hilo-addmm-epilogue.md` (branch `hyp/muse_bf16_proj/router-hilo-addmm-epilogue`).

- **Job:** `743020b6b43d458c` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `5fec321`

## Goal
In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc += torch.mm(x, router_lo_t, out_dtype=torch.float32)' with 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc = torch.addmm(acc, x, router_lo_t, beta=1.0, alpha=1.0, out_dtype=torch.float32)' so the lo-GEMM's cuBLAS epilogue adds the hi accumulator (beta=1) instead of a separate at::native elementwise add kernel. Everything else stays: prepare(), the hi/lo split, the fp32 F.linear path at m<=8, and the bf16 layers. Kernel moved: the vectorized_elementwise_kernel (0.9-1.0 us GPU plus its ~1-2 us graph-node gap) on every router shape with m>8 disappears; the two nvjet splitK GEMMs and their splitKreduce stay (the reduce is where cuBLAS applies beta, so it should cost nothing extra).

## What changed, and where
- File: `.worktrees/muse_bf16_proj/router-hilo-addmm-epilogue/muse_bf16_proj/submission.py`, `kernel_fn`, router branch only (above `ROUTER_FP32_MAX_TOKENS = 1`, i.e. m >= 2).
- Replaced `acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)` with
  `acc = torch.addmm(acc, x, state["router_lo_t"], beta=1.0, alpha=1.0, out_dtype=torch.float32)`
  (out-of-place, exactly as the change paragraph specifies), plus a one-line comment.
- Unchanged: the hi `torch.mm(..., out_dtype=torch.float32)`, `prepare()` and the hi/lo split, the fp32 `F.linear` path at m <= 1, the bf16 layers via `_bf16_gemm_dispatch_impl`, the first-line marker.
- Deviation: none. Deliberately NOT `out=acc`; the in-place form is the queued companion `router-hilo-addmm-epilogue-control-inplace-out`, so this primary vs that control is the A/B.
- Expected-table note (from review): base has ROUTER_FP32_MAX_TOKENS = 1, so verify_1 (m=8) is on the hi/lo path and is touched by this change (expect faster, not flat); only decode_1 (m=1) is the flat control. Attribute against job aa3d033c208e4511 per-shape router medians.
- Not run locally (no GPU, no torch on this node). Unverified: that the image's torch accepts `addmm(..., out_dtype=)` with bf16 mat1/mat2 and fp32 self; a TypeError/dtype check would fail the first router test (m 8).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-hilo-addmm-epilogue diff` (1 file, +4/-1, router branch of `kernel_fn` only).

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; the only branch is on `x.shape[0]` and `cfg["layer"]` (shape/config); addmm's output allocation is shape-determined, graph-safe.
- [x] Nothing derived from `data` is cached across calls: `acc` is a per-call local; `state` is only read.
- [x] Weights: repacked only in `prepare()` (unchanged); bf16 layers stay BF16; router uses the allowed BF16 hi+lo pair with FP32 accumulation (addmm with fp32 self and out_dtype=float32).
- [x] Activations BF16 where SGLang keeps BF16: x enters both router GEMMs as BF16, as before; bf16 layers untouched.
- [x] n/a: attention — this problem is dense projections; no attention, no key skipping.
- [x] n/a: KV cache / FP8 queries — no KV cache or queries in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only a `torch.addmm` call and a comment.
- [x] All work happens inside the timed call; no output caching or replay: both router GEMMs run every call.
- [x] First line still `# okc: problem=muse_bf16_proj`; `python3 -m py_compile` passes.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: submission.py
```
(exit 0; no findings)

## Risks / what could fail
medium
Risk: torch.addmm's out_dtype kwarg (fp32 accumulate from bf16 operands with an fp32 self) exists in torch >= 2.8 alongside torch.mm's; if the image's torch rejects it (TypeError or a dtype-mismatch check requiring self.dtype == out_dtype, which it is here, fp32) the run fails at the first router test, costing one check and nothing else. Fallback if rejected: torch.mm(..., out=acc) is not an accumulate, so there is no PyTorch-only substitute; the fused Triton kernel is the other route. Gain is small (~2 us of 15-17) and may sit inside the 1.5% CI on the load, so read the per-shape router rows, not the headline. If router-triton-hilo-fused lands first, this becomes moot; keep for attribution of the add-kernel cost.

## Result (after the run)
- job `743020b6b43d458c` (check), status done, passed True, tests 9/9
- speedup 1.0256x, 95% CI [1.0102, 1.0410], score 4710.315823445268 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.0307103903232853, "ci": [1.015249734468436, 1.0461710461781344], "job": "9aa98e889be140eb"}; base result: muse_bf16_proj/hypotheses/done/gain/router-hilo-at-m8.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-hilo-addmm-epilogue.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-hilo-addmm-epilogue.patch`; branch `hyp/muse_bf16_proj/router-hilo-addmm-epilogue`
