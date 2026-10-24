---
problem: muse_attention_decode
slug: paired-heads-launch-cleanup
created: 2026-10-23T225840Z
base: master
area: attention boundary
change: >
  Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must
  have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad
  at the source: the fused pre-attention Triton kernel's query programs (one per (token, head))
  write their 64 normalized FP8 values straight into the padded [tokens, Hq, 128] FP8 buffer at
  column offset half*64 with half = (h // G) % 2, and write 64 FP8 zeros into the other half, so
  torch.zeros and the two slice copies disappear and FA3 reads the buffer directly. (2) Un-pad in
  the epilogue: replace rms_norm_gate on full layers with a Triton norm-gate kernel, grid (tokens,
  Hq), one program per head: it loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at
  [t, h, half*64 : half*64+64] (the garbage half is never read), computes r = rsqrt(mean(x^2) +
  1e-5) in fp32, loads the 64 gate logits from qkv row t at (Hq + 2*Hkv)*64 + h*64, and stores
  bf16 (x * r * sigmoid(gate)) into the final [B, L, Hq*64] output; the two un-pad copies and
  SGLang's rms_norm_gate disappear. The metadata rebuild on decode steps remains the only extra
  launch over the fused-preattention path. Sliding layers keep whatever the landed base does
  (fused pre-attention writing the [tokens, Hq, 64] buffer, unpaired FA3, SGLang rms_norm_gate).
expected:
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: five launches (~10-12 us) removed from a ~40 us launch-bound shape
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: same five launches on a ~45 us shape
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: same launches removed but they are under 6% of 150-200 us; expect a small gain, possibly inside the CI
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: launches are under 3% of this shape
  - shapes: "*layer=full load=verify_*"
    direction: faster
    why: same launches, same magnitudes as the decode shapes
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: fa3-hdim128-head-pairing
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# paired-heads-launch-cleanup

## Change

Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad at the source: the fused pre-attention Triton kernel's query programs (one per (token, head)) write their 64 normalized FP8 values straight into the padded [tokens, Hq, 128] FP8 buffer at column offset half*64 with half = (h // G) % 2, and write 64 FP8 zeros into the other half, so torch.zeros and the two slice copies disappear and FA3 reads the buffer directly. (2) Un-pad in the epilogue: replace rms_norm_gate on full layers with a Triton norm-gate kernel, grid (tokens, Hq), one program per head: it loads its 64 bf16 values from FA3's [tokens, Hq, 128] output at [t, h, half*64 : half*64+64] (the garbage half is never read), computes r = rsqrt(mean(x^2) + 1e-5) in fp32, loads the 64 gate logits from qkv row t at (Hq + 2*Hkv)*64 + h*64, and stores bf16 (x * r * sigmoid(gate)) into the final [B, L, Hq*64] output; the two un-pad copies and SGLang's rms_norm_gate disappear. The metadata rebuild on decode steps remains the only extra launch over the fused-preattention path. Sliding layers keep whatever the landed base does (fused pre-attention writing the [tokens, Hq, 64] buffer, unpaired FA3, SGLang rms_norm_gate).

## Expected

- `*layer=full load=decode_1 page*` faster: five launches (~10-12 us) removed from a ~40 us launch-bound shape
- `*layer=full load=verify_1 page*` faster: same five launches on a ~45 us shape
- `*layer=full load=decode_16*` faster: same launches removed but they are under 6% of 150-200 us; expect a small gain, possibly inside the CI
- `*layer=full load=decode_32*` flat: launches are under 3% of this shape
- `*layer=full load=verify_*` faster: same launches, same magnitudes as the decode shapes
- `*layer=sliding*` flat: untouched; control

## Risk

Medium risk: two kernels touched (pre-attention and the epilogue) but both edits are mechanical placement changes with no new math; the attributed quantity is launch count on full layers. Correctness: the norm-gate epilogue must match reference._rms (eps 1e-5 inside the sqrt over the mean of 64 squares) and sigmoid(gate.float()); SGLang's rms_norm_gate took a bf16 attention output and this reads FA3's bf16 output too, so the rounding points are identical. The pad write must cover the full 128 columns every call (fresh zeros each step, not relying on a buffer from an earlier call: fresh inputs rule). Needs both fa3-hdim128-head-pairing and fused-preattention-triton landed; the single depends_on names the pairing because its buffer layout is what this edits. If fa3-hdim128-head-pairing is rejected this is moot.
