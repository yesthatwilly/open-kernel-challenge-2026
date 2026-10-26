---
problem: muse_attention_decode
slug: combo-sk-full-line-c320-tail
created: 2026-10-26T222147Z
base: master
area: combination (full decode B>=2 gather mapping + full decode_1 combine tail)
change: >
  COMBINATION for a same-day submit candidate (the solo checks run in parallel, so no stacked
  round fits before 23:45Z). On the v3chain base (depends_on merge-perkvhead-v3chain-sv32-tiled;
  claim restacks its patch onto master) apply EXACTLY the changes specified in three queued
  hypotheses and nothing else: (1) sk-full-line-gather: in _SK_CUDA's splitkv_attn_sk_kernel
  issue_tile, c = tid & 7 and tt = (tid >> 3) + 32 * j for j in {0, 1}, ok / row / dst computed
  per tt, same cp_async16 calls (byte-identical stage image); (2) c320-combine-onepass:
  c320_combine_kernel body as one pass (each warp loads up to 16 of its partials fully unrolled
  before use, online LSE in registers, group loop when n > NW * 16, warp results merged by warp 0
  after one __syncthreads, unchanged epilogue); (3) c320-pdl-early-trigger: c320_attn_kernel's
  griddepcontrol.launch_dependents moved from its last statement to right after the prologue's
  __syncthreads (plus the NW = 8 PDL combine instantiation only if the printed attrs say 16-warp
  combine CTAs cannot co-reside with A). If any of the three is already implemented on its own
  branch (hyp/muse_attention_decode/<slug>), copy that diff rather than re-deriving it. The rows
  are disjoint: full decode_16/32 (and the PSEUDO rows sliding verify_1 / decode_32) read change
  1; full decode_1 reads changes 2-3, whose chain gate prints V3 (change 2 only) and V4 (changes 2
  + 3) separately. All existing self-checks and gates (_sk_validate, PSEUDO / sd gates, GATE A,
  the V0..V4 chain gate) re-run unchanged.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: change 1 (kernel A precedent -4.7%); sk kernel 108.3 us
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: change 1 (kernel A precedent -14%); sk kernel 226.3 us
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "changes 2-3: combine tail ~5 us -> ~2-3 us; V3/V4 printed separately"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: untouched rows
risk: medium
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 2
kind: primary
controls_for: null
prior: "merge-perkvhead-v3chain-sv32-tiled: same base (1.5360 check); this stacks three small unmeasured changes on disjoint rows (sk gather mapping on full decode B>=2, c320 combine and PDL trigger on full decode_1) for a same-day submit"
review: required
status: queued
---

# combo-sk-full-line-c320-tail

## Change

COMBINATION for a same-day submit candidate (the solo checks run in parallel, so no stacked round fits before 23:45Z). On the v3chain base (depends_on merge-perkvhead-v3chain-sv32-tiled; claim restacks its patch onto master) apply EXACTLY the changes specified in three queued hypotheses and nothing else: (1) sk-full-line-gather: in _SK_CUDA's splitkv_attn_sk_kernel issue_tile, c = tid & 7 and tt = (tid >> 3) + 32 * j for j in {0, 1}, ok / row / dst computed per tt, same cp_async16 calls (byte-identical stage image); (2) c320-combine-onepass: c320_combine_kernel body as one pass (each warp loads up to 16 of its partials fully unrolled before use, online LSE in registers, group loop when n > NW * 16, warp results merged by warp 0 after one __syncthreads, unchanged epilogue); (3) c320-pdl-early-trigger: c320_attn_kernel's griddepcontrol.launch_dependents moved from its last statement to right after the prologue's __syncthreads (plus the NW = 8 PDL combine instantiation only if the printed attrs say 16-warp combine CTAs cannot co-reside with A). If any of the three is already implemented on its own branch (hyp/muse_attention_decode/<slug>), copy that diff rather than re-deriving it. The rows are disjoint: full decode_16/32 (and the PSEUDO rows sliding verify_1 / decode_32) read change 1; full decode_1 reads changes 2-3, whose chain gate prints V3 (change 2 only) and V4 (changes 2 + 3) separately. All existing self-checks and gates (_sk_validate, PSEUDO / sd gates, GATE A, the V0..V4 chain gate) re-run unchanged.

## Expected

- `*layer=full load=decode_16 *` faster: change 1 (kernel A precedent -4.7%); sk kernel 108.3 us
- `*layer=full load=decode_32 *` faster: change 1 (kernel A precedent -14%); sk kernel 226.3 us
- `*layer=full load=decode_1 *` faster: changes 2-3: combine tail ~5 us -> ~2-3 us; V3/V4 printed separately
- `*load=verify_16 *` flat: untouched rows

## Risk

Predicted about +1.5-2.5% over v3chain's 1.5360 if the pieces hold, i.e. a scored-submit candidate above the 1.5347 board. Read row by row: a row that loses here while its solo hypothesis gains points to an implementation difference, and a solo regression voids that row's reading here. Risk medium only because three edits ride one check; each has its own gate or self-check and falls back on failure (sk -> stage-1 A+B, c320 -> V0).
