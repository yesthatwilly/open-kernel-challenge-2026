---
problem: muse_bf16_proj
slug: router-hilo-addmm-epilogue
created: 2026-10-23T225128Z
base: master
area: router GEMM
change: >
  In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate
  'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc += torch.mm(x, router_lo_t,
  out_dtype=torch.float32)' with 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc =
  torch.addmm(acc, x, router_lo_t, beta=1.0, alpha=1.0, out_dtype=torch.float32)' so the lo-GEMM's
  cuBLAS epilogue adds the hi accumulator (beta=1) instead of a separate at::native elementwise
  add kernel. Everything else stays: prepare(), the hi/lo split, the fp32 F.linear path at m<=8,
  and the bf16 layers. Kernel moved: the vectorized_elementwise_kernel (0.9-1.0 us GPU plus its
  ~1-2 us graph-node gap) on every router shape with m>8 disappears; the two nvjet splitK GEMMs
  and their splitKreduce stay (the reduce is where cuBLAS applies beta, so it should cost nothing
  extra).
expected:
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: one fewer launch of five (14.8 us reported, 12.5 us of kernels)
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same, 5 launches to 4
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: same at m=128
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: same at m=256; add kernel there is 1.0 us + gap
  - shapes: layer=router load=decode_1 *
    direction: flat
    why: m=1 stays on fp32 F.linear; control
  - shapes: layer=router load=verify_1 *
    direction: flat
    why: m=8 stays on fp32 F.linear unless router-hilo-at-m8 has landed first
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T024434Z
review_companions: [router-hilo-addmm-epilogue-control-inplace-out]
stage: harvested
claimed_at: 2026-10-24T024439Z
base_sha: fe9a5a37d2a2208f820e8c3696faae713a1bb583
base_blob: 349be31cabc7d9b669b94032ae723c0b0a5e7f71
branch: hyp/muse_bf16_proj/router-hilo-addmm-epilogue
worktree: .worktrees/muse_bf16_proj/router-hilo-addmm-epilogue
mode: check
implemented_at: 2026-10-24T024717Z
impl_commit: 5fec3218be0e36938926e1686f5f594fad1ebb92
submission_sha256: 2e80c092b0fc20de5989016d605fa6fadc3a53ea59be5f7b4ca17240c2be51b4
norm_sha256: d89ca258fd504b13e42bab75a167dc746e892dd74f749585cebb8d401b70bd7e
submission_blob: b30f7112740b8be9d4d5d5e4f42ea46fa49771f1
lint_warnings: 0
attempt_file: 2026-10-24T024717Z-check-router-hilo-addmm-epilogue.md
prepared_at: 2026-10-24T024717Z
job: 743020b6b43d458c
dispatched_at: 2026-10-24T024725Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0255787894210895
speedup_ci: [1.010195107579773, 1.0409624712624057]
harvested_at: 2026-10-24T035909Z
best_at_harvest: 1.0307103903232853
promotion: null
---

# router-hilo-addmm-epilogue

## Change

In kernel_fn's router branch above ROUTER_FP32_MAX_TOKENS, replace the two-statement accumulate 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc += torch.mm(x, router_lo_t, out_dtype=torch.float32)' with 'acc = torch.mm(x, router_hi_t, out_dtype=torch.float32); acc = torch.addmm(acc, x, router_lo_t, beta=1.0, alpha=1.0, out_dtype=torch.float32)' so the lo-GEMM's cuBLAS epilogue adds the hi accumulator (beta=1) instead of a separate at::native elementwise add kernel. Everything else stays: prepare(), the hi/lo split, the fp32 F.linear path at m<=8, and the bf16 layers. Kernel moved: the vectorized_elementwise_kernel (0.9-1.0 us GPU plus its ~1-2 us graph-node gap) on every router shape with m>8 disappears; the two nvjet splitK GEMMs and their splitKreduce stay (the reduce is where cuBLAS applies beta, so it should cost nothing extra).

## Expected

- `layer=router load=decode_16 *` faster: one fewer launch of five (14.8 us reported, 12.5 us of kernels)
- `layer=router load=decode_32 *` faster: same, 5 launches to 4
- `layer=router load=verify_16 *` faster: same at m=128
- `layer=router load=verify_32 *` faster: same at m=256; add kernel there is 1.0 us + gap
- `layer=router load=decode_1 *` flat: m=1 stays on fp32 F.linear; control
- `layer=router load=verify_1 *` flat: m=8 stays on fp32 F.linear unless router-hilo-at-m8 has landed first

## Risk

Risk: torch.addmm's out_dtype kwarg (fp32 accumulate from bf16 operands with an fp32 self) exists in torch >= 2.8 alongside torch.mm's; if the image's torch rejects it (TypeError or a dtype-mismatch check requiring self.dtype == out_dtype, which it is here, fp32) the run fails at the first router test, costing one check and nothing else. Fallback if rejected: torch.mm(..., out=acc) is not an accumulate, so there is no PyTorch-only substitute; the fused Triton kernel is the other route. Gain is small (~2 us of 15-17) and may sit inside the 1.5% CI on the load, so read the per-shape router rows, not the headline. If router-triton-hilo-fused lands first, this becomes moot; keep for attribution of the add-kernel cost.

## Adversarial review

Reviewed 2026-10-24T024434Z by an adversarial Fable subagent (model fable).

**Verdict: proceed-with-companions.** The mechanism (fold the hi accumulator into the lo GEMM's beta=1 epilogue so the 0.9-1.1 us add node and its graph gap disappear) is real and the per-shape router rows can read it, but the change paragraph's out-of-place form 'acc = torch.addmm(acc, x, router_lo_t, ...)' makes PyTorch copy acc into a fresh result with its own elementwise kernel before the GEMM, so as written it swaps add for copy and will most likely read flat for a reason unrelated to the mechanism. One control with out=acc (in-place, no copy) makes a flat primary interpretable.

### Justification
On master (router-hilo-at-m8, job aa3d033c208e4511) every router row with m>1 is nvjet splitK x2 (7.9-10.5 us), splitKreduce x2 (3.3-3.8 us), vectorized_elementwise add 0.9-1.1 us; 14.15/14.51/15.03/16.21/17.66 us at m=8/16/32/128/256 against floors 1.8-2.7 us, ~0.4 us gap per node. Removing one node saves ~1.3-1.5 us, 8-10% of a router row, readable per shape (untouched rows scatter ~+/-3%); ~1.5% geomean, at the CI edge, so classify from router rows. verify_1 (m=8) is now on hi/lo and must be expected faster, not flat; decode_1 (m=1) is the only flat control. Weakest link: out-of-place addmm with a 2-D self and beta != 0 runs at::native::copy_(result, self) as its own vectorized_elementwise kernel, same node count as mm + add, and the profile would look identical to master. Second: addmm.dtype/dtype_out overloads must exist in the image's torch. Third: cuBLAS may pick a different algorithm for beta=1. Numerics: summation order only, ~1e-7 relative.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: created the torch.mm hi/lo path this change edits; substrate, not a confounder.
- router-hilo-at-m8 (ROUTER_FP32_MAX_TOKENS 8 -> 1): widens reach to the m=8 row; the expected 'verify_1 flat' entry is stale and must be faster. Not a causal confounder.
- Attribution base: compare per-shape medians against router-hilo-at-m8's job aa3d033c208e4511, not the stale BEST.json starter.

### Interference and A/B needs
ab-revert/ab-tune of landed items is pointless. What is needed is a control on the hypothesis's own sub-change: the primary is both 'beta=1 epilogue' and 'out-of-place addmm', and the latter re-introduces a copy_ launch. A flat primary would be ambiguous between 'beta epilogue not free' and 'addmm copied self'. A control with out=acc resolves it. If the dispatcher may amend the primary to out=acc, drop the companion. No second companion.

### Companions queued
- `router-hilo-addmm-epilogue-control-inplace-out`: Separates 'cuBLAS's beta=1 epilogue is not cheaper than the separate add' from 'PyTorch's out-of-place addmm re-introduced an elementwise copy kernel'; this is the form to promote if it gains.

### Open questions
- Amend the primary to out=acc if allowed; if amended, drop the companion.
- verify_1 (m=8) should be expected faster, not flat; only decode_1 is the flat control.
- Confirm from the profile that the lo GEMM still resolves to nvjet splitK + splitKreduce and the vectorized_elementwise node is absent.
- If the image's torch rejects out= with out_dtype=, record that as the learning rather than retrying.
- Attribute against job aa3d033c208e4511 per-shape medians.
- Classify from the router rows (~8-10%), not the headline (~1.5%).
- If router-triton-hilo-fused-control-splitk-all-m gains at every m, this and its control become moot except for attributing the add node's cost.


## Implementation notes

- File: `.worktrees/muse_bf16_proj/router-hilo-addmm-epilogue/muse_bf16_proj/submission.py`, `kernel_fn`, router branch only (above `ROUTER_FP32_MAX_TOKENS = 1`, i.e. m >= 2).
- Replaced `acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)` with
  `acc = torch.addmm(acc, x, state["router_lo_t"], beta=1.0, alpha=1.0, out_dtype=torch.float32)`
  (out-of-place, exactly as the change paragraph specifies), plus a one-line comment.
- Unchanged: the hi `torch.mm(..., out_dtype=torch.float32)`, `prepare()` and the hi/lo split, the fp32 `F.linear` path at m <= 1, the bf16 layers via `_bf16_gemm_dispatch_impl`, the first-line marker.
- Deviation: none. Deliberately NOT `out=acc`; the in-place form is the queued companion `router-hilo-addmm-epilogue-control-inplace-out`, so this primary vs that control is the A/B.
- Expected-table note (from review): base has ROUTER_FP32_MAX_TOKENS = 1, so verify_1 (m=8) is on the hi/lo path and is touched by this change (expect faster, not flat); only decode_1 (m=1) is the flat control. Attribute against job aa3d033c208e4511 per-shape router medians.
- Not run locally (no GPU, no torch on this node). Unverified: that the image's torch accepts `addmm(..., out_dtype=)` with bf16 mat1/mat2 and fp32 self; a TypeError/dtype check would fail the first router test (m 8).

## Rules check

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

## Result

- job `743020b6b43d458c` (check), status done, passed True, tests 9/9
- speedup 1.0256x, 95% CI [1.0102, 1.0410], score 4710.315823445268 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.0307103903232853, "ci": [1.015249734468436, 1.0461710461781344], "job": "9aa98e889be140eb"}; base result: muse_bf16_proj/hypotheses/done/gain/router-hilo-at-m8.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-hilo-addmm-epilogue.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-hilo-addmm-epilogue.patch`; branch `hyp/muse_bf16_proj/router-hilo-addmm-epilogue`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 14.19 | 14.14 | 1.0033 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.64 | 6.74 | 1.005 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 21.55 | 21.55 | 1.0012 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 7.01 | 7.19 | 1.0089 | flat |
| layer=router load=decode_1 m=1 | 62 | 10.51 | 10.34 | 1.013 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.03 | 14.13 | 0.9899 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.94 | 6.75 | 1.0186 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 21.62 | 21.44 | 1.0142 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.62 | 18.87 | 0.9861 | flat |
| layer=router load=decode_16 m=16 | 62 | 14.51 | 14.66 | 1.0032 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.42 | 14.32 | 1.007 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.96 | 6.72 | 0.9933 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 21.9 | 22.14 | 0.9963 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.57 | 18.73 | 0.9881 | flat |
| layer=router load=decode_32 m=32 | 62 | 15.03 | 14.78 | 0.9986 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 14.13 | 14.2 | 0.9827 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.43 | 6.67 | 0.9782 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 20.94 | 21.69 | 0.9888 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.71 | 7.16 | 0.9749 | flat |
| layer=router load=verify_1 m=8 | 62 | 14.15 | 14.08 | 1.0075 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.27 | 17.23 | 1.0275 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.43 | 7.45 | 1.042 | gained |
| layer=pooler load=verify_16 m=128 | 62 | 24.96 | 25.35 | 1.0174 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.72 | 19.99 | 1.0125 | flat |
| layer=router load=verify_16 m=128 | 62 | 16.21 | 16.48 | 0.999 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.79 | 19.88 | 1.006 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.33 | 9.3 | 1.025 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.11 | 30.74 | 1.0355 | gained |
| layer=unpooler load=verify_32 m=256 | 62 | 24.07 | 24.58 | 1.0053 | flat |
| layer=router load=verify_32 m=256 | 62 | 17.66 | 17.55 | 0.9987 | flat |

gained on o verify_16; pooler verify_32. Expected hit: 2/6. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-hilo-addmm-epilogue`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

