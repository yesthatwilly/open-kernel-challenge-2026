---
problem: muse_decode_layer
slug: sglang-pipeline-parity-router-hilo-m8
created: 2026-10-23T225318Z
base: master
area: router GEMM
change: >
  Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py
  composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the
  FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules,
  the cos|sin table, the router hi/lo split and the side stream; kernel_fn runs
  _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler, the FP8 k/v casts plus
  store_cache, flash_attn_with_kvcache (FA3, scheduler_metadata passed through on full decode),
  the two boundary JIT kernels, the router on the side stream, moe_align_block_size plus the two
  moe_wna16_marlin_gemm calls, silu_and_mul and fused_post_expert_norm_weighted_sum. ONE
  deliberate structural change on top of parity, in the router GEMM: lower ROUTER_FP32_MAX_TOKENS
  from 8 to 1, so the fp32 F.linear router (a SIMT sm80_xmma f32 GEMM at m=8, measured at 26.2 us
  in the muse_bf16_proj review versus ~15 us for the hi/lo bf16 pair) only serves m=1 (where it is
  a gemv) and every m>=8 shape runs the two torch.mm(..., out_dtype=float32) hi/lo GEMMs;
  consequently attn_boundary's router_dtype is None at m=8 (one fewer fp32 output written). The
  rules name the hi/lo pair with fp32 accumulation as allowed at any batch size. Everything else
  stays byte-for-byte SGLang's path. Attribution: load=verify_1 shapes (T=8) carry the router
  change; all other shapes are pure parity and must land at ~1.0x, which becomes the base every
  later muse_decode_layer hypothesis stacks on.
expected:
  - shapes: "*load=verify_1*"
    direction: faster
    why: "T=8: the fp32 SIMT router GEMM (~26 us on the side stream, longer than the ~15-20 us pooler GEMM it overlaps) becomes two bf16 tensor-core GEMMs (~15 us) that hide under the pooler; expect 2-4% on these two shapes"
  - shapes: "*load=decode_1*"
    direction: flat
    why: T=1 stays on the fp32 F.linear gemv exactly as SGLang; pure parity, control
  - shapes: "*load=decode_16*"
    direction: flat
    why: T=16 was already on the hi/lo path in SGLang; pure parity
  - shapes: "*load=decode_32*"
    direction: flat
    why: T=32 already hi/lo in SGLang; pure parity
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 already hi/lo; pure parity
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 already hi/lo; pure parity
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# sglang-pipeline-parity-router-hilo-m8

## Change

Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules, the cos|sin table, the router hi/lo split and the side stream; kernel_fn runs _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler, the FP8 k/v casts plus store_cache, flash_attn_with_kvcache (FA3, scheduler_metadata passed through on full decode), the two boundary JIT kernels, the router on the side stream, moe_align_block_size plus the two moe_wna16_marlin_gemm calls, silu_and_mul and fused_post_expert_norm_weighted_sum. ONE deliberate structural change on top of parity, in the router GEMM: lower ROUTER_FP32_MAX_TOKENS from 8 to 1, so the fp32 F.linear router (a SIMT sm80_xmma f32 GEMM at m=8, measured at 26.2 us in the muse_bf16_proj review versus ~15 us for the hi/lo bf16 pair) only serves m=1 (where it is a gemv) and every m>=8 shape runs the two torch.mm(..., out_dtype=float32) hi/lo GEMMs; consequently attn_boundary's router_dtype is None at m=8 (one fewer fp32 output written). The rules name the hi/lo pair with fp32 accumulation as allowed at any batch size. Everything else stays byte-for-byte SGLang's path. Attribution: load=verify_1 shapes (T=8) carry the router change; all other shapes are pure parity and must land at ~1.0x, which becomes the base every later muse_decode_layer hypothesis stacks on.

## Expected

- `*load=verify_1*` faster: T=8: the fp32 SIMT router GEMM (~26 us on the side stream, longer than the ~15-20 us pooler GEMM it overlaps) becomes two bf16 tensor-core GEMMs (~15 us) that hide under the pooler; expect 2-4% on these two shapes
- `*load=decode_1*` flat: T=1 stays on the fp32 F.linear gemv exactly as SGLang; pure parity, control
- `*load=decode_16*` flat: T=16 was already on the hi/lo path in SGLang; pure parity
- `*load=decode_32*` flat: T=32 already hi/lo in SGLang; pure parity
- `*load=verify_16*` flat: T=128 already hi/lo; pure parity
- `*load=verify_32*` flat: T=256 already hi/lo; pure parity

## Risk

Risk: this is SGLang's own code path plus a one-constant dispatch change the rules explicitly permit, so correctness risk is near zero (routing ties are judged with 0.03-0.05 margins, far above the hi/lo split's ~2^-16 relative error). The expected gain on verify_1 is small (2-4%) and could be inside the +-1.5% CI if the router already hid under the pooler at m=8; a flat verify_1 result still leaves the parity base intact. The commit step's normalized-AST dedup must see a difference from baseline.py: the threshold constant and the resulting router_dtype branch are that difference; do not drop them. Implementation must keep the file self-contained (no import of reference.py), keep the side stream join (main_stream.wait_stream) before local_ids, and pass scheduler_metadata through unchanged on full decode. If the judge image's import paths differ from baseline.py's, the run fails at import, which is a rejection without timing cost.
