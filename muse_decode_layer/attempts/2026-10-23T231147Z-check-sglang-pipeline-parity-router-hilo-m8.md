# muse_decode_layer · 2026-10-23T231147Z · check · sglang-pipeline-parity-router-hilo-m8

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-23T225318Z-sglang-pipeline-parity-router-hilo-m8.md` (branch `hyp/muse_decode_layer/sglang-pipeline-parity-router-hilo-m8`).

- **Job:** `e260dfe413e448c7` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `54112d4`

## Goal
Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules, the cos|sin table, the router hi/lo split and the side stream; kernel_fn runs _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler, the FP8 k/v casts plus store_cache, flash_attn_with_kvcache (FA3, scheduler_metadata passed through on full decode), the two boundary JIT kernels, the router on the side stream, moe_align_block_size plus the two moe_wna16_marlin_gemm calls, silu_and_mul and fused_post_expert_norm_weighted_sum. ONE deliberate structural change on top of parity, in the router GEMM: lower ROUTER_FP32_MAX_TOKENS from 8 to 1, so the fp32 F.linear router (a SIMT sm80_xmma f32 GEMM at m=8, measured at 26.2 us in the muse_bf16_proj review versus ~15 us for the hi/lo bf16 pair) only serves m=1 (where it is a gemv) and every m>=8 shape runs the two torch.mm(..., out_dtype=float32) hi/lo GEMMs; consequently attn_boundary's router_dtype is None at m=8 (one fewer fp32 output written). The rules name the hi/lo pair with fp32 accumulation as allowed at any batch size. Everything else stays byte-for-byte SGLang's path. Attribution: load=verify_1 shapes (T=8) carry the router change; all other shapes are pure parity and must land at ~1.0x, which becomes the base every later muse_decode_layer hypothesis stacks on.

## What changed, and where
Ported baseline.py (SGLang commit a55ab7d5, MuseSpark1xDecoderLayer.forward) into
`muse_decode_layer/submission.py` on the branch, whole: the same imports (sgl_kernel FA3 and
scalar_types; sglang.kernels silu_and_mul, store_cache, moe_wna16_marlin_gemm, FusedMuseSparkRouting,
FusedRMSNormGate, FusedPostNormResidualPreNorm, fused_post_expert_norm_weighted_sum, FusedQKNorm;
moe_align_block_size; the three MuseSpark1x norm/residual classes; marlin_make_workspace,
prepare_moe_nvfp4_layer_for_marlin, _bf16_gemm_dispatch_impl), `prepare()` (Marlin bank via
`_marlin_bank`, the fused modules, the [2^21, 64] cos|sin table, the router hi/lo bf16 split
transposed for torch.mm, the side stream), `_experts()` and `kernel_fn(data, state)` (bf16 GEMM
dispatch for qkv_gate/o/pooler/unpooler, FusedQKNorm with RoPE on sliding layers, FP8 k/v casts +
`store_cache`, `flash_attn_with_kvcache` with `scheduler_metadata` passed through unchanged, the two
boundary JIT kernels, the router + FusedMuseSparkRouting on the side stream with
`main_stream.wait_stream(side_stream)` before `local_ids`, `moe_align_block_size`, the two
`moe_wna16_marlin_gemm` calls, `silu_and_mul`, `fused_post_expert_norm_weighted_sum`, sliced unpooler
+ zero pad at T <= 8). The starter's Weights/Config/Inputs TypedDicts are kept verbatim (their field
comments are a little fuller than baseline's; AST-identical). No reference.py import; the file is
self-contained.

The one change: `ROUTER_FP32_MAX_TOKENS = 1` (baseline: 8), at
`muse_decode_layer/submission.py:148`, with its comment updated. Through baseline's own
`split_router = tokens > ROUTER_FP32_MAX_TOKENS`, this means T=1 still runs
`F.linear(router_input, state["router"])` in fp32 on the fp32 router input, while every T >= 2
(the T=8 verify_1 shapes are the only scored ones that move; T=16..256 were already on this path) runs
`torch.mm(ffn_input, router_hi_t, out_dtype=float32)` + `torch.mm(ffn_input, router_lo_t,
out_dtype=float32)`. router_dtype consequence: `attn_boundary(..., router_dtype=None if split_router
else torch.float32)` is therefore called with `router_dtype=None` at T=8, so the
FusedPostNormResidualPreNorm JIT kernel no longer writes the fp32 [8, 8192] router input there
(one fewer output), and the router consumes the bf16 `ffn_input` instead, exactly as SGLang does at
T > 8.

Verification: the commit step's normalized-AST hash (`churn.py:norm_sha256`, which strips
docstrings, annotations and TypedDict classes) differs from baseline.py's, and the same file with
the constant put back to 8 hashes identical to baseline.py, so the constant is the sole AST
difference. Diff vs baseline.py's code lines: the module docstring, the starter's TypedDict
comments, the constant and its comment; nothing else.

Deviations: none in code. Not verified at runtime (no GPU/SGLang on this machine): that
FusedMuseSparkRouting returns int32 [T, 8] ids (the judge's `collect` demands exactly that dtype;
baseline.py returns the same tensor unchanged and is the judge's own timed kernel, so it must), and
that m=8 `torch.mm(..., out_dtype=float32)` picks a decent cuBLAS kernel (the open question in the
review). Syntax checked with ast.parse; lint OK.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the only `torch.cuda.synchronize()` in the diff is baseline's own, in
      `prepare()` (untimed); kernel_fn/_experts have no `.item()`/`.cpu()`/prints; the only
      branches are on `cfg[...]` and `tokens` (config), and `torch.cuda.is_current_stream_capturing()`
      exactly as baseline.
- [x] Nothing derived from `data` is cached across calls: `state` is built only from `weights` in
      `prepare()`; kernel_fn writes nothing into `state` or module globals (diff grepped for
      `global`, container writes; none).
- [x] Weights: the Marlin bank comes from SGLang's own `prepare_moe_nvfp4_layer_for_marlin`
      (lossless NVFP4 repack); bf16 weights are only made contiguous; the router fp32 -> bf16 hi+lo
      split with fp32 `out_dtype` accumulation is the named exception, now used at T >= 2 (allowed
      "at any batch size").
- [x] Activations: BF16 everywhere baseline keeps BF16 (same `branch_dtype=torch.bfloat16` calls,
      same bf16 GEMM inputs); k/v cast to FP8 e4m3 as baseline; q to FP8 e4m3 for FA3 (the one
      permitted rounding).
- [x] Attention sees every key: `flash_attn_with_kvcache` call is byte-identical to baseline
      (`causal=True`, `window_size=(window-1, 0)` on sliding layers, `(-1, -1)` on full,
      `scheduler_metadata` passed through); no top-k/sparse/skipping anywhere.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses
      or network: imports are exactly baseline's set plus nothing; diff grepped for
      `reference`, `subprocess`, `socket`, `sys.modules`, `inspect`; none.
- [x] All work happens inside the timed call; no output caching or replay: kernel_fn recomputes
      everything from `data` each call; prepare() only repacks weights and builds modules/tables.

Lint: `OK with warnings: muse_decode_layer/submission.py` (one warn: `host-sync L245:
torch.cuda.synchronize() (outside kernel_fn's call graph; fine in prepare())`), exit 0.

## Risks / what could fail
low
Risk: this is SGLang's own code path plus a one-constant dispatch change the rules explicitly permit, so correctness risk is near zero (routing ties are judged with 0.03-0.05 margins, far above the hi/lo split's ~2^-16 relative error). The expected gain on verify_1 is small (2-4%) and could be inside the +-1.5% CI if the router already hid under the pooler at m=8; a flat verify_1 result still leaves the parity base intact. The commit step's normalized-AST dedup must see a difference from baseline.py: the threshold constant and the resulting router_dtype branch are that difference; do not drop them. Implementation must keep the file self-contained (no import of reference.py), keep the side stream join (main_stream.wait_stream) before local_ids, and pass scheduler_metadata through unchanged on full decode. If the judge image's import paths differ from baseline.py's, the run fails at import, which is a rejection without timing cost.

## Result (after the run)
- job `e260dfe413e448c7` (check), status done, passed True, tests 13/13
- speedup 1.0216x, 95% CI [1.0063, 1.0370], score 22141.335696926693 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 0.0045119870605644765, "ci": [0.00444430725465601, 0.004579666866472943], "job": "3c3181b1fce24b9a"}; base result: muse_decode_layer/hypotheses/BEST.result.json
- **outcome: gain** (gain)
- files: `muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json`, `muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.patch`; branch `hyp/muse_decode_layer/sglang-pipeline-parity-router-hilo-m8`
