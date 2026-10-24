---
problem: muse_attention_decode
slug: fused-full-combine-normgate
created: 2026-10-23T225858Z
base: master
area: output norm-gate
change: >
  Stacked on triton-flash-decode-full-both-heads: on full-attention decode steps fold
  rms_norm_gate into the landed kernel 2 (the split combine, grid (B, Hq)) so the full decode path
  is pre-attention + two launches instead of three. After the program for (request b, head h) has
  reduced the num_splits partial (m, l, acc[64]) into the fp32 attention output o[64] = sum_s
  exp(m_s - M) * acc_s / sum_s exp(m_s - M) * l_s, it computes r = rsqrt(mean(o^2) + 1e-5), loads
  the 64 gate logits from qkv row b at column (Hq + 2*Hkv)*64 + h*64 (bf16 to fp32), and stores
  bf16 (o * r * sigmoid(gate)) directly into the final [B, 1, Hq*64] output at [b, h*64 :
  (h+1)*64]. kernel_fn on full decode steps returns that tensor and no longer calls rms_norm_gate;
  full verify blocks and sliding layers keep SGLang's rms_norm_gate (or whatever the landed base
  does there). Semantics match reference: _rms over the 64-wide head of the fp32 attention output
  then times sigmoid(gate.float()), cast to bf16 once at the end; SGLang rounds the attention
  output to bf16 before its norm, which this skips (finer, allowed: bf16 is kept where SGLang
  keeps it, the returned tensor).
expected:
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: one launch (~2.3 us) fewer on a ~25-30 us shape once the Triton decode kernel is in
  - shapes: "*layer=full load=decode_16*"
    direction: flat
    why: one launch on a 150-200 us shape, inside the CI
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: same, under 1%
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched; control
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: triton-flash-decode-full-both-heads
priority: 5
kind: primary
controls_for: null
review: required
status: hold
hold_reason: "moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held); the paired epilogue already fuses the full-layer norm-gate"
---

# fused-full-combine-normgate

## Change

Stacked on triton-flash-decode-full-both-heads: on full-attention decode steps fold rms_norm_gate into the landed kernel 2 (the split combine, grid (B, Hq)) so the full decode path is pre-attention + two launches instead of three. After the program for (request b, head h) has reduced the num_splits partial (m, l, acc[64]) into the fp32 attention output o[64] = sum_s exp(m_s - M) * acc_s / sum_s exp(m_s - M) * l_s, it computes r = rsqrt(mean(o^2) + 1e-5), loads the 64 gate logits from qkv row b at column (Hq + 2*Hkv)*64 + h*64 (bf16 to fp32), and stores bf16 (o * r * sigmoid(gate)) directly into the final [B, 1, Hq*64] output at [b, h*64 : (h+1)*64]. kernel_fn on full decode steps returns that tensor and no longer calls rms_norm_gate; full verify blocks and sliding layers keep SGLang's rms_norm_gate (or whatever the landed base does there). Semantics match reference: _rms over the 64-wide head of the fp32 attention output then times sigmoid(gate.float()), cast to bf16 once at the end; SGLang rounds the attention output to bf16 before its norm, which this skips (finer, allowed: bf16 is kept where SGLang keeps it, the returned tensor).

## Expected

- `*layer=full load=decode_1 page*` faster: one launch (~2.3 us) fewer on a ~25-30 us shape once the Triton decode kernel is in
- `*layer=full load=decode_16*` flat: one launch on a 150-200 us shape, inside the CI
- `*layer=full load=decode_32*` flat: same, under 1%
- `*load=verify_*` flat: untouched; control
- `*layer=sliding*` flat: untouched; control

## Risk

Medium risk: a 64-wide epilogue per (request, head) program; the hazard is the near-zero-row rule (rows measured against a tenth of the average row), so eps placement must match reference._rms exactly (1e-5 inside the sqrt over the mean of 64 squares) and the gate must be read from the right column of the bf16 qkv row. The gain is small and confined to decode_1 full (weight 16 in a load dominated by the sliding shape's 46 x 21 us), so this is a cleanup to run after the parent and its tuning children; the mirror of the queued fused-sliding-normgate-epilogue on the full path.
