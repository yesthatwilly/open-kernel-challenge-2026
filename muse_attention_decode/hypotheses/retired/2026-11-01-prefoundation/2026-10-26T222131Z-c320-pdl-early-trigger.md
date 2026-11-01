---
problem: muse_attention_decode
slug: c320-pdl-early-trigger
created: 2026-10-26T222131Z
base: master
area: launch overhead (full decode_1 c320 kernel A -> combine PDL)
change: >
  Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master). Its chain
  gate routed V4 (kernel A FUSED + c320_combine_kernel<..., PDL = true> launched with programmatic
  stream serialization) at 19.36 us against V3 (same pair, no PDL) at 19.39 us: PDL bought nothing
  because c320_attn_kernel issues griddepcontrol.launch_dependents as its LAST statement (after
  the partial stores, before the final cp_wait<0>), so the dependent combine grid is released only
  when every A CTA has finished. ONE change in c320_attn_kernel (both instantiations; a no-op
  without a dependent, so V1 / V3 are unaffected): issue the griddepcontrol.launch_dependents
  right after the prologue (after the first cp_commit / cp_wait<0> / __syncthreads that lands item
  0's page indices, before the tile loop) and remove it from the end. Correctness does not depend
  on the trigger's position: everything the combine reads from A is below its griddepcontrol.wait,
  which blocks until kernel A has completed and its memory is visible; the reads above the wait
  (seqlens, v_scale, gate logits) are graph inputs A never writes. What moves is the dependent's
  launch, CTA placement and pre-wait reads, which now overlap A. A uses 250 of 264 CTA slots (2
  per SM), so the 16 combine CTAs (16 warps each) can become resident only where an A slot is
  free; print c320_attrs for the PDL combine and, if fewer than 16 can co-reside with A, also
  instantiate the PDL combine with NW = 8 (256 threads) and let the chain gate time both. GATE A
  and the V0..V4 gate (best / V0 <= 0.98) re-run unchanged; read V4 vs V3 in the 'PATH CHOSEN:
  full decode_1 [chain]' line.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "the combine's launch and pre-wait reads leave the critical path after A: expect V4 0.5-1.5 us under V3 (today equal); 0.34%/us"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: c320 routes batch 1 only
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: low
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 2
kind: primary
controls_for: null
prior: "combo-b1-chain-c320-fused-pre-pdl: its V4 already launches the combine with PDL but triggers at the very end of kernel A and measured equal to V3 (18.91 vs 18.91 us); this moves only the trigger to A's prologue so the dependent launch overlaps A"
review: required
status: hold
hold_reason: "stale 2026-10-27: already run and landed by the campaign (sk full-line gather, c320 onepass, PDL trigger on master since 10-27), or superseded by s2b (scored a76ee9b0 / sw17 4e386647)"
---

# c320-pdl-early-trigger

## Change

Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master). Its chain gate routed V4 (kernel A FUSED + c320_combine_kernel<..., PDL = true> launched with programmatic stream serialization) at 19.36 us against V3 (same pair, no PDL) at 19.39 us: PDL bought nothing because c320_attn_kernel issues griddepcontrol.launch_dependents as its LAST statement (after the partial stores, before the final cp_wait<0>), so the dependent combine grid is released only when every A CTA has finished. ONE change in c320_attn_kernel (both instantiations; a no-op without a dependent, so V1 / V3 are unaffected): issue the griddepcontrol.launch_dependents right after the prologue (after the first cp_commit / cp_wait<0> / __syncthreads that lands item 0's page indices, before the tile loop) and remove it from the end. Correctness does not depend on the trigger's position: everything the combine reads from A is below its griddepcontrol.wait, which blocks until kernel A has completed and its memory is visible; the reads above the wait (seqlens, v_scale, gate logits) are graph inputs A never writes. What moves is the dependent's launch, CTA placement and pre-wait reads, which now overlap A. A uses 250 of 264 CTA slots (2 per SM), so the 16 combine CTAs (16 warps each) can become resident only where an A slot is free; print c320_attrs for the PDL combine and, if fewer than 16 can co-reside with A, also instantiate the PDL combine with NW = 8 (256 threads) and let the chain gate time both. GATE A and the V0..V4 gate (best / V0 <= 0.98) re-run unchanged; read V4 vs V3 in the 'PATH CHOSEN: full decode_1 [chain]' line.

## Expected

- `*layer=full load=decode_1 *` faster: the combine's launch and pre-wait reads leave the critical path after A: expect V4 0.5-1.5 us under V3 (today equal); 0.34%/us
- `*layer=full load=decode_16 *` flat: c320 routes batch 1 only
- `*layer=sliding *` flat: untouched

## Risk

Small (+0.2-0.5%), below the margin alone: the readout is V4 vs V3 in the chain gate and the FD1 row; stacks with c320-combine-onepass (same row, separate mechanism: V3 isolates the combine change, V4 - V3 the trigger). Risk low: PDL semantics keep it correct; the only failure mode is no gain. Decode-portable (d1f c320 port; decode already chains its own kernels with PDL).
