---
problem: muse_attention_decode
slug: preattention-tiled-gate-large-tokens
created: 2026-10-24T183855Z
base: master
area: pre-attention norm+rope
change: >
  Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is
  muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch) behind a
  token-count gate. Add that patch's group-tiled kernel as a NEW Triton kernel
  _pre_attention_tiled_kernel beside master's _pre_attention_kernel (do not modify the old one):
  grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the 8 query heads of group g as one
  [8, 64] tile plus key head g, applies RoPE when ROPE, writes FP8 q in the PAIRED or unpaired
  layout exactly as master (padded half zeroed when PAIRED), and stores the FP8 key and value rows
  divided by k_scale / v_scale at cache_loc[t]. In kernel_fn, launch the tiled kernel when tokens
  >= 64 (verify_16 and verify_32, sliding and full) and master's _pre_attention_kernel otherwise
  (decode at every batch and verify_1, where the tiled version was 0.1 us slower). Same arguments,
  same outputs, same rounding points.
expected:
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "measured in the parent run: _pre_attention_kernel 4.1 -> 2.7 us, sliding verify_32 28.7 -> 27.4 us (-4.5%)"
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: "parent: 2.7 -> 2.0 us, about -3% on the shape"
  - shapes: "*layer=full load=verify_16 * *layer=full load=verify_32 *"
    direction: faster
    why: "same kernel saving, ~1 us out of 162 / 301 us: within noise"
  - shapes: "*load=decode_* *load=verify_1 *"
    direction: flat
    why: "below the 64-token gate, master's kernel runs"
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# preattention-tiled-gate-large-tokens

## Change

Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch) behind a token-count gate. Add that patch's group-tiled kernel as a NEW Triton kernel _pre_attention_tiled_kernel beside master's _pre_attention_kernel (do not modify the old one): grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the 8 query heads of group g as one [8, 64] tile plus key head g, applies RoPE when ROPE, writes FP8 q in the PAIRED or unpaired layout exactly as master (padded half zeroed when PAIRED), and stores the FP8 key and value rows divided by k_scale / v_scale at cache_loc[t]. In kernel_fn, launch the tiled kernel when tokens >= 64 (verify_16 and verify_32, sliding and full) and master's _pre_attention_kernel otherwise (decode at every batch and verify_1, where the tiled version was 0.1 us slower). Same arguments, same outputs, same rounding points.

## Expected

- `*layer=sliding load=verify_32 *` faster: measured in the parent run: _pre_attention_kernel 4.1 -> 2.7 us, sliding verify_32 28.7 -> 27.4 us (-4.5%)
- `*layer=sliding load=verify_16 *` faster: parent: 2.7 -> 2.0 us, about -3% on the shape
- `*layer=full load=verify_16 * *layer=full load=verify_32 *` faster: same kernel saving, ~1 us out of 162 / 301 us: within noise
- `*load=decode_* *load=verify_1 *` flat: below the 64-token gate, master's kernel runs

## Risk

The sliding verify gain is replaced if sliding-cuda-cluster-dsmem-onelaunch lands (it fuses pre-attention into its one launch). full-verify-cuda-streamk-splitkv keeps master's _pre_attention_kernel (PAIRED=False) in front of its A8 kernel, so the full verify_16/32 part survives that rewrite, and the tiled kernel stacks there cleanly. Judge on the _pre_attention_tiled_kernel row.
