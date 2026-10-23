# muse_bf16_proj · 2026-10-23T221952Z · check · sglang-gemm-dispatch-parity

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T221829Z-sglang-gemm-dispatch-parity.md` (branch `hyp/muse_bf16_proj/sglang-gemm-dispatch-parity`).

- **Job:** `017804b34fed40b3` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `e64f27e`

## Goal
Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(), router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(..., out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and attributes the starter's 10% loss to the GEMM dispatch path alone.

## What changed, and where
Rewrote `kernel_fn` and `prepare` in `muse_bf16_proj/submission.py` to mirror SGLang's own
path, nothing else:

- `prepare()`: keeps every weight contiguous and adds `router_hi_t` / `router_lo_t`, the bf16
  hi + lo split of the fp32 router transposed for `torch.mm` (built once, untimed).
- `kernel_fn()`, bf16 layers (`qkv_gate`, `o`, `pooler`, `unpooler`): one call to
  `sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None)`, replacing
  the starter's `x @ w.T`.
- `kernel_fn()`, router: `F.linear(x.float(), router)` at m ≤ 8 (as the starter effectively
  did); above 8 tokens two `torch.mm(..., out_dtype=torch.float32)` GEMMs on the hi/lo pair,
  accumulated in fp32, replacing the fp32 matmul.
- No deviation from the spec. The file is SGLang's dispatch expressed in our own file, so the
  expected outcome is parity (~1.0x) and it becomes the base for later bf16_proj hypotheses.

## Rules check (each must be true before the run)
Checked against the diff of the worktree's `submission.py` vs master:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; the only
      branch is on `config["layer"]` and on `x.shape[0]` (a shape), never on values.
- [x] Nothing derived from `data` is cached: `state` is built in `prepare()` from the weights only.
- [x] Weights: the bf16 weights are stored unchanged; the router is kept in FP32 and additionally
      as a BF16 hi + lo pair with FP32 accumulation, which the Weights rule names as the one
      allowed exception.
- [x] Activations: bf16 in, bf16 GEMM, fp32 router output, exactly where SGLang keeps them.
- [x] Attention/every-key rule: not applicable (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import,
      subprocesses or network: the only import beyond torch is SGLang's dispatch function, which
      baseline.py imports too.
- [x] All work happens inside the timed call; no output caching or replay.

Lint (`okc_lint.py`): CLEAN, no findings.

## Risks / what could fail
low
Low risk: it is SGLang's own code path. Only risk is an import path difference inside the judge image; baseline.py imports the same symbol, so it exists there.

## Result (after the run)
- job `017804b34fed40b3` (check), status done, passed True, tests 9/9
- speedup 1.0022x, 95% CI [0.9872, 1.0173], score 4820.055403537152 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/BEST.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json`, `muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.patch`; branch `hyp/muse_bf16_proj/sglang-gemm-dispatch-parity`
