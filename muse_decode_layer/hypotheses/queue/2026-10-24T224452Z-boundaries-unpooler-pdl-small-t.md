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
review: proceed
status: queued
review_at: 2026-10-25T010851Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T010851Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible; duplicates nothing landed, running or do-not-retry. Two spec errors to fix before the build: residual coefficients are per-channel [8192] vectors, not scalars, and post_ffn applies (1+w) as an input scale before the RMS. The T=9..32 full-width unpooler will probably be flat or slower than nvjet, so decode_16/32 'faster' rests almost entirely on the boundaries.

### Justification
At T<=8 the replaced chain (nvjet sliced unpooler 8.4 MB + splitKreduce + F.pad, ~9 us vs a ~3 us P-body GEMV) is real overhead; the P-body GEMV already runs fault-free at ~3.4-3.6 TB/s for NF=1. Fusing both boundaries with an immediate launch_dependents makes attn boundary -> P the first adjacent hand-written pair, so P's W_pr prefetch overlaps for real: decode_1/verify_1 ~5-8 us. Incumbents are stale (master decode_1 127.5/142.9). Weakest link: the T=9..32 full-width unpooler (NF=2/4) inherits P's x L2 re-read (2.85 TB/s at T=32, ~3.1 at T=16) vs master nvjet ~3.1-3.2 TB/s: parity at T=16, up to ~2 us slower at T=32. At T=1 the one-CTA boundary runs two dependent block reductions over ~100 KB, ~2-3 us vs SGLang 3.6. Spec errors: (1) alpha = sqrt(clamp(sigmoid(-w/0.3)*(1+sigmoid(w/0.3)), 1e-3)) and beta = sigmoid(w/0.3) per channel; attn/ffn_residual_gate are [8192] (reference.py:109,123,339-344), so alpha/beta must be fp32 [8192] device vectors from the module's precomputed buffers, not scalars, else dlb_ok goes False and the check measures master. (2) ffn boundary (reference.py:472-476): stream2 = alpha2*stream1 + beta2*rms(f*(1+post_ffn_norm), 1e-8); (1+w) is an INPUT scale before the RMS; only the attn post-norm has no weight; next attention input rms(stream2)*(1+next_pre_attn_norm) eps 1e-5. (3) Optional: at T<=8, f columns 1024..8191 are zero, so the ffn boundary can read 1024 columns. Headline ~+1.5-2.5%, inside CI; read profile rows.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (base): overlap exists only because P is PDL-launched; re-run _dl_self_check graph (attn_boundary -> P -> R -> experts) with the new boundary; part of any gain shows as a shorter P row.
- post-expert-norm-triton-small-t: the unpooler's predecessor (Triton at T<32, SGLang at T=32); self-check must cover both layouts through the real _experts.
- stack-align-par-sidestream-sliding1: no interaction at T<=32; stale incumbents only.
- marlin-block-m-cap-16, fa3-hdim128-head-pairing-batched, sglang-pipeline-parity-router-hilo-m8: no confound.

### Interference and A/B needs
No companion. Rows dl_boundary_kernel, dl_unpool_kernel<1> (T<=8) and dl_unpool_kernel<2/4> (T=9..32) are separable from master's rows; a shorter P row isolates the PDL overlap; verify_16/32 are the drift control. Build note: in prepare time dl_unpool_kernel vs _gemm per NF on synthetic data and keep nvjet for any NF where the custom kernel is slower; print the pick per NF. Overlaps: dense-chain-stage2 (kernel-disjoint, same TU: textual conflict, complementary); split-KV checks (no overlap); dl-route-warp-topk-pdl edits P and R in the same TU; dl-p-wide-tile-t16-t32 fixes the x re-read this inherits.

### Open questions
- dl_boundary_kernel CUDA-event time at T=1 with one CTA vs SGLang 3.6/3.7 us? If >= 3 us, several CTAs per token?
- Does SGLang FusedPostNormResidualPreNorm read alpha/beta as precomputed fp32 buffers or recompute from bf16 gates? Match its rounding.
- Is dl_unpool_kernel<2/4> faster than nvjet at T=9..32 in prepare timing? If not, gate it off.

