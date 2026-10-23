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
  attributes the starter's 10% loss (per the adversarial review: the fp32 router GEMM at m>8, not the bf16 dispatch) to the router path alone.
expected:
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: "m=16 is above the 8-token threshold: fp32 SIMT GEMM becomes two bf16 tensor-core GEMMs"
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: "m=32, same path change"
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: "m=128, same path change; starter was 0.560x here"
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: "m=256, same path change; starter was 0.374x here"
  - shapes: layer=router load=decode_1 *
    direction: flat
    why: "m=1 stays on fp32 F.linear"
  - shapes: layer=router load=verify_1 *
    direction: flat
    why: "m=8 stays on fp32 F.linear"
  - shapes: layer=qkv_gate *
    direction: flat
    why: "the starter already picks the same nvjet splitK kernels (adversarial review)"
risk: low
depends_on: null
priority: 1
status: done
stage: harvested
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
review: proceed
review_at: 2026-10-23T222740Z
review_companions: null
outcome: gain
verdict_detail: gain
speedup: 1.0022291437677175
speedup_ci: [0.9871957066112018, 1.0172625809242333]
harvested_at: 2026-10-23T225813Z
best_at_harvest: 0.9020752041800139
promotion: pending
---

# sglang-gemm-dispatch-parity

## Change

Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(), router) at m <= 8 but build the BF16 hi + lo split in prepare() and run two torch.mm(..., out_dtype=torch.float32) GEMMs above 8 tokens, as baseline.py does. No other change. This establishes parity (expected ~1.0x) as the base every later bf16_proj hypothesis stacks on, and attributes the starter's 10% loss (per the adversarial review: the fp32 router GEMM at m>8, not the bf16 dispatch) to the router path alone.

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

## Adversarial review

Reviewed 2026-10-23T222740Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** The implementation is a functionally verbatim copy of baseline.py (diff vs baseline is docstrings, type hints and one local rename only), so the single run will land at ~1.00x within the ±1.5% interval and is fully interpretable as parity. But the answer is already on disk: job 1ccc6ba13da94194 ran baseline.py through the judge at 1.0028x [0.988, 1.018]; this run spends a daily check slot to re-measure it under a different sha256 so it can seat as BEST on master's lineage. Its real value is procedural (a parity base for later hypotheses), not evidential, and its expected-direction globs are wrong against the per-shape data already in BEST.result.json.

### Justification
Mechanism trivially plausible (SGLang's own path), but the hypothesis mislabels where the starter's 10% goes. The profiles show the four bf16 layers already run kernel-for-kernel identically in the starter and in SGLang (same nvjet splitK_TNT + splitKreduce pairs, every bf16 shape 0.96x-1.01x): on H200 _bf16_gemm_dispatch_impl resolves to the same cuBLAS TN GEMM as x @ w.T, so the dispatch half of the change is empty. The entire loss is the fp32 router above 8 tokens (SIMT/xmma f32 kernels, no tensor cores: 32.4 µs at m=256) vs SGLang's two bf16 tensor-core GEMMs: decode_16 0.563x, decode_32 0.763x, verify_16 0.560x, verify_32 0.374x. Expected globs: qkv_gate should be flat (starter already picks splitK); router verify_1 (m=8) stays on the identical fp32 path (flat); router decode_16/decode_32 (m=16/32) are above the threshold and will be the largest gains. Weakest link: 'attributes the loss to the GEMM dispatch path alone' is false; it is the router's fp32-vs-bf16-hi/lo path at m>8, and the two calibration runs already prove it. Router shapes sit 5-10x above their 1.8-2.7 µs floors even in SGLang; pooler/unpooler at ~1.5x their 14 µs floors: that is where later hypotheses must go.

### Confounders on the current kernel
- Nothing has landed on master's submission.py, so there is no landed-change confound.
- The change bundles two sub-changes, (a) bf16 layers via _bf16_gemm_dispatch_impl and (b) the fp32 router via bf16 hi+lo above 8 tokens; they are separable from one run because the judge reports per shape: (a) is the non-router layers, (b) is layer=router at m in {16,32,128,256}.
- Inert third sub-change: prepare() .contiguous() on every weight and the fp32 router kept beside the hi/lo pair; both no-ops for timing.
- The result is predetermined by job 1ccc6ba13da94194 (same executable code, 1.0028x [0.988,1.018]); any deviation beyond that CI is judge noise or image drift, not an effect of the change.

### Interference and A/B needs
No additional run is warranted. ab-revert is inapplicable (nothing landed). ab-tune: the only tunable is ROUTER_FP32_MAX_TOKENS, and changing it is a different, real hypothesis (hi/lo at m<=8 is allowed at any batch size; SGLang's m=8 fp32 path is 26.2 µs vs 14.9 µs at m=16 on hi/lo), not a disambiguator of parity. A control isolating (a) or (b) is unnecessary because per-shape results already isolate them by layer. A result inside [0.988,1.018] confirms parity; outside it indicates harness variance. The job is already dispatched, so holding would not recover the slot.

### Open questions
- Fix the expected globs before the result is scored: qkv_gate flat; router faster = load=decode_16|decode_32|verify_16|verify_32 (m>8); router flat = decode_1|verify_1 (m<=8).
- Rewrite the attribution claim: the starter's 10% is the fp32 router GEMM at m>8 (no tensor cores), not the GEMM dispatch path; FINDINGS should carry that as the first learning.
- Dedup against baseline.py and attempts: okc's same-code dedup did not fire because sha256 differs on docstrings alone; a normalized-AST hash would have returned job 1ccc6ba13da94194 for free.
- First real hypothesis from this review: ROUTER_FP32_MAX_TOKENS=0 (hi/lo at every m). Verify the 0.1% exact-column bound at m=1 first, where hi/lo error has nowhere to average out.
- Second target: pooler and unpooler at m>=16 run one nvjet kernel at ~1.5x their 14 µs floor on both sides; splitK/stream-K or a cuBLASLt algo search in prepare().

## Result

- job `017804b34fed40b3` (check), status done, passed True, tests 9/9
- speedup 1.0022x, 95% CI [0.9872, 1.0173], score 4820.055403537152 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/BEST.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json`, `muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.patch`; branch `hyp/muse_bf16_proj/sglang-gemm-dispatch-parity`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 14.08 | 13.96 | 1.028 | flat |
| layer=o load=decode_1 m=1 | 62 | 7.05 | 6.52 | 1.0162 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 21.45 | 21.22 | 1.0009 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.87 | 6.48 | 1.025 | flat |
| layer=router load=decode_1 m=1 | 62 | 10.45 | 10.36 | 1.0197 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.31 | 13.89 | 1.016 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.7 | 6.7 | 1.0194 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 21.45 | 21.45 | 0.9924 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.53 | 18.66 | 1.0033 | flat |
| layer=router load=decode_16 m=16 | 62 | 26.19 | 14.51 | 1.7916 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.46 | 14.19 | 0.9975 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.59 | 6.62 | 1.0243 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 22.06 | 21.75 | 1.006 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.88 | 18.55 | 1.0212 | flat |
| layer=router load=decode_32 m=32 | 62 | 19.75 | 15.14 | 1.3129 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 14.31 | 14.04 | 1.0135 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.86 | 7.01 | 0.984 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 21.66 | 21.28 | 1.0011 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 7.11 | 6.84 | 0.9989 | flat |
| layer=router load=verify_1 m=8 | 62 | 26.59 | 26.14 | 1.0006 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.96 | 16.9 | 0.9793 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.77 | 7.58 | 1.01 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.36 | 25.13 | 1.0197 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 20.1 | 19.84 | 1.0107 | flat |
| layer=router load=verify_16 m=128 | 62 | 29.15 | 16.62 | 1.7773 | gained |
| layer=qkv_gate load=verify_32 m=256 | 62 | 20.37 | 19.58 | 1.0254 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.65 | 9.25 | 1.0365 | gained |
| layer=pooler load=verify_32 m=256 | 62 | 30.88 | 31.17 | 1.002 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.59 | 24.33 | 1.0108 | flat |
| layer=router load=verify_32 m=256 | 62 | 45.85 | 17.76 | 2.6972 | gained |

gained on router decode_16, decode_32, verify_16, verify_32; o verify_32. Expected hit: 7/7. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

