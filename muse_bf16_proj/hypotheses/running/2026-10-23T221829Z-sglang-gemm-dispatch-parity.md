---
problem: muse_bf16_proj
slug: sglang-gemm-dispatch-parity
created: 2026-10-23T221829Z
base: master
area: bf16 GEMM dispatch
change: >
  Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses:
  sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four
  bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(),
  router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(...,
  out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This
  establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and
  attributes the starter's 10% loss to the GEMM dispatch path alone.
expected:
  - shapes: layer=router load=verify_*
    direction: faster
    why: "fp32 F.linear on m=128/256 is the starter's worst shapes; the hi/lo bf16 GEMMs are what SGLang runs there"
  - shapes: layer=qkv_gate *
    direction: faster
    why: cuBLAS/nvjet dispatch picks splitK kernels the plain matmul does not
  - shapes: layer=router load=decode_*
    direction: flat
    why: same F.linear path at m<=8
risk: low
depends_on: null
priority: 1
status: running
stage: dispatched
claimed_at: 2026-10-23T221905Z
base_sha: 965a5a2c44b96f0f62ef2d7cd77e99c879e9d3df
base_blob: 6ce773a44036578450b1489df0030d65d2c67c0e
branch: hyp/muse_bf16_proj/sglang-gemm-dispatch-parity
worktree: .worktrees/muse_bf16_proj/sglang-gemm-dispatch-parity
mode: check
implemented_at: 2026-10-23T221952Z
impl_commit: e64f27ef09b3512d26573dde4d35b8ec3ade1f63
submission_sha256: 476d72e0390c7112365e054459edf5e390f0c974fcd10232abb0c8af489d74cf
submission_blob: 5a2064af06f442f01dff926cb4f1eca75f992a24
lint_warnings: 0
attempt_file: 2026-10-23T221952Z-check-sglang-gemm-dispatch-parity.md
prepared_at: 2026-10-23T221952Z
job: 017804b34fed40b3
dispatched_at: 2026-10-23T222029Z
recover_note: adopted from ~/.config/okc/jobs.json
---

# sglang-gemm-dispatch-parity

## Change

Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(), router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(..., out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and attributes the starter's 10% loss to the GEMM dispatch path alone.

## Expected

- `layer=router load=verify_*` faster: fp32 F.linear on m=128/256 is the starter's worst shapes; the hi/lo bf16 GEMMs are what SGLang runs there
- `layer=qkv_gate *` faster: cuBLAS/nvjet dispatch picks splitK kernels the plain matmul does not
- `layer=router load=decode_*` flat: same F.linear path at m<=8

## Risk

Low risk: it is SGLang's own code path. Only risk is an import path difference inside the judge image; baseline.py imports the same symbol, so it exists there.

## Implementation notes

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

## Rules check

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
