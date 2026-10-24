---
problem: muse_decode_layer
slug: paired-heads-pad-unpad-triton
created: 2026-10-24T140156Z
base: master
area: output norm-gate
change: >
  Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers
  there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master's paired path in
  muse_decode_layer kernel_fn (full layers, batch > 1). Master's pairing adds five glue launches
  around FA3, visible in its profile (fb07a3bd3aa54baf) as elementwise_kernel<128,4> x3 (6.4 us)
  plus x2 (4.7 us) at full decode_32 and x3 6.4-6.8 us at full decode_16/verify_16:
  torch.zeros(q_attn) and the two strided qp[...] = qv[...] copies (which also cast to FP8), then
  torch.empty(attn) and the two a[...] = o[...] un-pad copies, followed by SGLang's
  FusedRMSNormGate. (1) Un-pad in the epilogue: copy _paired_norm_gate_kernel verbatim from
  muse_attention_decode/submission.py on master (grid (tokens, Hq), one program per head: loads
  its 64 bf16 values from FA3's [tokens, Hq, 128] output at half (h // G) % 2, fp32 scaleless
  RMSNorm with eps 1e-5, times sigmoid of the gate logits read from qkv row t at (Hq + 2*Hkv)*64 +
  h*64 with qkv.stride(0), bf16 store into a fresh [tokens, Hq*64] output) and use it in place of
  the attn/a/o un-pad block and state['o_norm_gate'] when paired; that output goes to the o GEMM.
  (2) Pad in one launch: since decode_layer has no fused pre-attention kernel on master
  (fused-preattention-triton was neutral, not merged), add a small Triton kernel
  _pair_pad_q_kernel, grid (tokens, Hq), that reads head h's 64 bf16 q values from FusedQKNorm's
  output (pass its strides), casts to FP8 E4M3 and writes them at column half*64 of the [tokens,
  Hq, 128] FP8 buffer (allocated torch.empty) and 64 FP8 zeros in the other half, replacing
  torch.zeros plus the two slice copies. FA3's call (paired k/v views, scheduler_metadata None),
  the unpaired path (batch 1 and sliding layers), the cache store and everything after the o GEMM
  are unchanged.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "5 glue launches + FusedRMSNormGate (~11-14 us) become 2 launches (~3-4 us): ~8-10 us off 407 us, +2%"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "elementwise<128,4> 6.4 + 4.7 us and the norm-gate become 2 launches: ~8-10 us off 544 us, +1.5-2%"
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "same glue at T=128 (6.8 us x3 row): ~8-12 us off 587 us, +1.5-2%"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "same glue at T=256, larger copies: ~10-14 us off 907 us, ~+1-1.5%, near drift"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: batch 1 unpaired, untouched
  - shapes: "*layer=full load=verify_1 *"
    direction: flat
    why: batch 1 unpaired, untouched
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: low
depends_on: fa3-hdim128-head-pairing-batched
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# paired-heads-pad-unpad-triton

## Change

Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master's paired path in muse_decode_layer kernel_fn (full layers, batch > 1). Master's pairing adds five glue launches around FA3, visible in its profile (fb07a3bd3aa54baf) as elementwise_kernel<128,4> x3 (6.4 us) plus x2 (4.7 us) at full decode_32 and x3 6.4-6.8 us at full decode_16/verify_16: torch.zeros(q_attn) and the two strided qp[...] = qv[...] copies (which also cast to FP8), then torch.empty(attn) and the two a[...] = o[...] un-pad copies, followed by SGLang's FusedRMSNormGate. (1) Un-pad in the epilogue: copy _paired_norm_gate_kernel verbatim from muse_attention_decode/submission.py on master (grid (tokens, Hq), one program per head: loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at half (h // G) % 2, fp32 scaleless RMSNorm with eps 1e-5, times sigmoid of the gate logits read from qkv row t at (Hq + 2*Hkv)*64 + h*64 with qkv.stride(0), bf16 store into a fresh [tokens, Hq*64] output) and use it in place of the attn/a/o un-pad block and state['o_norm_gate'] when paired; that output goes to the o GEMM. (2) Pad in one launch: since decode_layer has no fused pre-attention kernel on master (fused-preattention-triton was neutral, not merged), add a small Triton kernel _pair_pad_q_kernel, grid (tokens, Hq), that reads head h's 64 bf16 q values from FusedQKNorm's output (pass its strides), casts to FP8 E4M3 and writes them at column half*64 of the [tokens, Hq, 128] FP8 buffer (allocated torch.empty) and 64 FP8 zeros in the other half, replacing torch.zeros plus the two slice copies. FA3's call (paired k/v views, scheduler_metadata None), the unpaired path (batch 1 and sliding layers), the cache store and everything after the o GEMM are unchanged.

## Expected

- `*layer=full load=decode_16*` faster: 5 glue launches + FusedRMSNormGate (~11-14 us) become 2 launches (~3-4 us): ~8-10 us off 407 us, +2%
- `*layer=full load=decode_32*` faster: elementwise<128,4> 6.4 + 4.7 us and the norm-gate become 2 launches: ~8-10 us off 544 us, +1.5-2%
- `*layer=full load=verify_16*` faster: same glue at T=128 (6.8 us x3 row): ~8-12 us off 587 us, +1.5-2%
- `*layer=full load=verify_32*` faster: same glue at T=256, larger copies: ~10-14 us off 907 us, ~+1-1.5%, near drift
- `*layer=full load=decode_1 *` flat: batch 1 unpaired, untouched
- `*layer=full load=verify_1 *` flat: batch 1 unpaired, untouched
- `*layer=sliding*` flat: untouched; control

## Risk

Smallest of the four (~+0.3-0.5% geomean, on the 16-weight full shapes only), picked over a re-run of the MoE map+align port because that idea already ran here (moe-align-single-kernel-small-t, every shape within +0.2..+1.9%, neutral) and FINDINGS give no reason to expect more on master. Risk is low: the epilogue kernel is verbatim and correct in the sibling; the pad kernel is new but trivial (cast+scatter). Check that FusedQKNorm's q layout ([tokens, Hq*64], possibly a strided view of qkv) is read through its real strides. Gains are per-shape 1.5-2%, so read the elementwise_kernel<128,4> rows disappearing in top_kernels, not the headline.
