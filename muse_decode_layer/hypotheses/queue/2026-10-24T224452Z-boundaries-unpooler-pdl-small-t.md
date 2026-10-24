---
problem: muse_decode_layer
slug: boundaries-unpooler-pdl-small-t
created: 2026-10-24T224452Z
base: master
area: post-MoE unpooler + residual boundaries (T<=32)
change: >
  Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate
  plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte for byte). Three additions to
  the existing _DL_CUDA TU (new module name, so it recompiles): (1) dl_boundary_kernel: one CTA
  per token, 256 threads, exact math of FusedPostNormResidualPreNorm as master constructs it (read
  sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py and muse_spark_layers
  MuseSpark1xRMSNorm / MuseSpark1xRMSNormWithInputScale /
  MuseSpark1xNormPreservingResidualConnection: post-norm eps POST_NORM_EPS with gain center one,
  the residual coefficients from precompute_coefficients read once in prepare into fp32 device
  scalars, pre-norm with the effective 1+w weight in fp32, outputs branch bf16 [T,8192] and new
  stream fp32 [T,8192]). It executes griddepcontrol.launch_dependents as its FIRST instruction,
  then griddepcontrol.wait, then reads its inputs, so the PDL-launched kernel after it (P after
  the attention boundary) issues its pre-wait weight ring while the boundary runs. Launched itself
  with the PDL attribute. It replaces state["attn_boundary"](...) and state["ffn_boundary"](...)
  on the gate (router_dtype is already None there). (2) dl_unpool_kernel<NF>: the P body over
  dense unpooler bf16 (K = 4096): at T <= 8 only rows 0..1023 (UNPOOL_SLICE; 16 tiles x SPLIT 8 =
  128 CTAs) and its CTAs also store bf16 zeros to columns 1024..8191 of the [T,8192] ffn_output
  (each CTA a disjoint 56-column-per-token slice, after griddepcontrol.wait), replacing
  _gemm(expert_output, unpooler[:1024]) and F.pad; at 9..32 tokens all 8192 rows (128 tiles x
  SPLIT 2 = 256 CTAs) replacing _gemm(expert_output, unpooler). Weight ring prefetch before the
  wait, PDL launch, fixed-order ticket split-K reduce as P. (3) the ffn boundary then runs
  dl_boundary_kernel (PDL, it waits on the unpooler). Prepare self-check on synthetic data at T in
  1, 4, 8, 9, 16, 32: both boundary outputs vs the SGLang modules (branch rel L2 <= 2e-3, stream
  rel L2 <= 1e-5 relative to |stream|), ffn_output vs master unpooler+pad (rel L2 <= 2e-3, zero
  columns exactly 0), a captured CUDA graph of boundary -> P -> R -> experts -> unpool -> boundary
  replayed vs eager; on failure dlb_ok False. Print [dl-boundary-unpool] ON|OFF and CUDA-event
  times of each new kernel. Attention side, P, R, Marlin, combine unchanged.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent 132.7/142.6 us: unpooler 5.3 + splitKreduce ~1.7 + F.pad ~2 -> ~3 us, two SGLang boundaries 3.6+3.7 -> ~2+2, plus P ring fill overlapped with the boundary ~2 us: ~8-10 us (+6-7%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "same at T=8: ~8-9 us of 186.5/195.9 (+4-5%)"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "unpooler 67 MB ~19.5 us nvjet -> ~18 at 3.7 TB/s, boundaries and overlap ~4 us: ~5-6 us (+2-2.5% sliding)"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~5-7 us of 247/535 (+1-2.5%); headline +2.5-3.5%
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "outside gate: control"
risk: high
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# boundaries-unpooler-pdl-small-t

## Change

Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte for byte). Three additions to the existing _DL_CUDA TU (new module name, so it recompiles): (1) dl_boundary_kernel: one CTA per token, 256 threads, exact math of FusedPostNormResidualPreNorm as master constructs it (read sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py and muse_spark_layers MuseSpark1xRMSNorm / MuseSpark1xRMSNormWithInputScale / MuseSpark1xNormPreservingResidualConnection: post-norm eps POST_NORM_EPS with gain center one, the residual coefficients from precompute_coefficients read once in prepare into fp32 device scalars, pre-norm with the effective 1+w weight in fp32, outputs branch bf16 [T,8192] and new stream fp32 [T,8192]). It executes griddepcontrol.launch_dependents as its FIRST instruction, then griddepcontrol.wait, then reads its inputs, so the PDL-launched kernel after it (P after the attention boundary) issues its pre-wait weight ring while the boundary runs. Launched itself with the PDL attribute. It replaces state["attn_boundary"](...) and state["ffn_boundary"](...) on the gate (router_dtype is already None there). (2) dl_unpool_kernel<NF>: the P body over dense unpooler bf16 (K = 4096): at T <= 8 only rows 0..1023 (UNPOOL_SLICE; 16 tiles x SPLIT 8 = 128 CTAs) and its CTAs also store bf16 zeros to columns 1024..8191 of the [T,8192] ffn_output (each CTA a disjoint 56-column-per-token slice, after griddepcontrol.wait), replacing _gemm(expert_output, unpooler[:1024]) and F.pad; at 9..32 tokens all 8192 rows (128 tiles x SPLIT 2 = 256 CTAs) replacing _gemm(expert_output, unpooler). Weight ring prefetch before the wait, PDL launch, fixed-order ticket split-K reduce as P. (3) the ffn boundary then runs dl_boundary_kernel (PDL, it waits on the unpooler). Prepare self-check on synthetic data at T in 1, 4, 8, 9, 16, 32: both boundary outputs vs the SGLang modules (branch rel L2 <= 2e-3, stream rel L2 <= 1e-5 relative to |stream|), ffn_output vs master unpooler+pad (rel L2 <= 2e-3, zero columns exactly 0), a captured CUDA graph of boundary -> P -> R -> experts -> unpool -> boundary replayed vs eager; on failure dlb_ok False. Print [dl-boundary-unpool] ON|OFF and CUDA-event times of each new kernel. Attention side, P, R, Marlin, combine unchanged.

## Expected

- `*load=decode_1 *` faster: incumbent 132.7/142.6 us: unpooler 5.3 + splitKreduce ~1.7 + F.pad ~2 -> ~3 us, two SGLang boundaries 3.6+3.7 -> ~2+2, plus P ring fill overlapped with the boundary ~2 us: ~8-10 us (+6-7%)
- `*load=verify_1 *` faster: same at T=8: ~8-9 us of 186.5/195.9 (+4-5%)
- `*load=decode_16 *` faster: unpooler 67 MB ~19.5 us nvjet -> ~18 at 3.7 TB/s, boundaries and overlap ~4 us: ~5-6 us (+2-2.5% sliding)
- `*load=decode_32 *` faster: ~5-7 us of 247/535 (+1-2.5%); headline +2.5-3.5%
- `*load=verify_16 *` flat: outside gate: control
- `*load=verify_32 *` flat: outside gate: control

## Risk

High: the boundary must reproduce SGLang residual-gate numerics exactly (stream output is checked against fp32 reference); the self-check gates it. An early launch_dependents is legal because the dependent still griddepcontrol.waits before reading anything this kernel writes; only constant weights are read before the wait. Disjoint from dense-chain-stage2-attn-side-pdl (different kernels), so the two can stack by merge. bf16-m32-swapab-wgmma-port (held) also targets the T=32 unpooler.
