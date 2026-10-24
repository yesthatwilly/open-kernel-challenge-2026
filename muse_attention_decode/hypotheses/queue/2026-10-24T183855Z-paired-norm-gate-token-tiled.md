---
problem: muse_attention_decode
slug: paired-norm-gate-token-tiled
created: 2026-10-24T183855Z
base: master
area: output norm-gate
change: >
  Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at
  full verify_16/32) from one 1-warp program per (token, query head) to one program per token:
  grid (tokens,), num_warps 4. Program t loads the [16, 64] tile of head h's own half (h // G) % 2
  of FA3's [tokens, 16, 128] output (column offset h * attn_stride_h + half * D + j, rows h =
  arange(16)), computes the scaleless RMSNorm per row (sum over axis 1, fp32, eps inside the sqrt
  of the mean of 64 squares), times sigmoid of the [16, 64] gate tile read from qkv, and stores
  bf16 into out[t, h * 64 + j]. Arithmetic and rounding are identical; only the decomposition
  changes. kernel_fn launches it with grid (tokens,) instead of (tokens, hq).
expected:
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: _paired_norm_gate_kernel is 3.8 us at verify_32 for 4096 one-warp programs moving 1 MB; 256 four-warp programs should be ~1.5 us, ~-0.8% on the 300 us shape
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: same, smaller (2048 -> 128 programs)
  - shapes: "*layer=sliding* *load=decode_* *load=verify_1 *"
    direction: flat
    why: these shapes do not run the paired epilogue
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# paired-norm-gate-token-tiled

## Change

Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at full verify_16/32) from one 1-warp program per (token, query head) to one program per token: grid (tokens,), num_warps 4. Program t loads the [16, 64] tile of head h's own half (h // G) % 2 of FA3's [tokens, 16, 128] output (column offset h * attn_stride_h + half * D + j, rows h = arange(16)), computes the scaleless RMSNorm per row (sum over axis 1, fp32, eps inside the sqrt of the mean of 64 squares), times sigmoid of the [16, 64] gate tile read from qkv, and stores bf16 into out[t, h * 64 + j]. Arithmetic and rounding are identical; only the decomposition changes. kernel_fn launches it with grid (tokens,) instead of (tokens, hq).

## Expected

- `*layer=full load=verify_32 *` faster: _paired_norm_gate_kernel is 3.8 us at verify_32 for 4096 one-warp programs moving 1 MB; 256 four-warp programs should be ~1.5 us, ~-0.8% on the 300 us shape
- `*layer=full load=verify_16 *` faster: same, smaller (2048 -> 128 programs)
- `*layer=sliding* *load=decode_* *load=verify_1 *` flat: these shapes do not run the paired epilogue

## Risk

full-verify-cuda-streamk-splitkv (in the judge now) replaces the paired FA3 path and this epilogue at full verify_16/32; if it lands, this is moot and should be held. Small by construction: judge on the kernel row.
