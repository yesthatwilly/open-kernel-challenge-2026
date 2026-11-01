---
problem: muse_attention_decode
slug: sk-full-line-5stage-select
created: 2026-10-26T222305Z
base: master
area: full-layer attention (decode 1<B<=64, stream-K ring depth after the whole-line remap)
change: >
  Stacked on sk-full-line-gather (build on its code; claim restacks it). The stream-K module
  already instantiates <5 stages, ticket> (variant 2: 112672 B dynamic smem, 119 regs, 2 CTAs/SM)
  for diagnostics only; with the old half-line mapping s2's sk-diag measured it 0.971x (B=16) /
  1.001x (B=32) of the 4-stage kernel, i.e. not depth-limited then. With whole-line requests the
  ring may now be the limiter, so re-time it and route by a gate. CHANGE (host side only, no CUDA
  edit): replace the constant _SK_VARIANT = 0 by an import-time frozen per-band table
  _SK_BAND_VARIANT (band A = 2 <= B <= 24 timed at B=16, band B = 25 <= B <= 64 timed at B=32;
  MappingProxyType, default 0). After _sk_validate passes: run variant 2 through the same _sk_case
  correctness cases as variant 0, then time variants 0 and 2 interleaved (_graph_of /
  _interleaved_us, 256 MB flush, empty graph subtracted, two rounds of 21 reps) on
  _splitkv_synthetic inputs with 80000-agentic lengths at B=16 and B=32; route variant 2 for a
  band only if it wins >= 1.02x in BOTH rounds with IQR <= 1.5x variant 0's, and only after
  _sk_replay_check-style two-state alternating replays of variant 2 at that band pass.
  _full_decode_sk takes the band's variant (config-only lookup; variant 2 uses the same
  single-level ticket epilogue, no combine). Print 'PATH CHOSEN: full decode B band <A|B>: <4|5>
  stages (t4 vs t5 us, both rounds)'. B=1 (two-level / c320) unchanged.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: only if 5 stages win >= 1.02x after the whole-line remap (was 0.971x before it); 0.107%/us
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: same, band B (was 1.001x); 0.060%/us
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: low
depends_on: sk-full-line-gather
priority: 4
kind: primary
controls_for: null
prior: "full-decode-splitkv-s2-streamk-ticket-inproc: its sk-diag measured 5 stages at 0.971x (B=16) / 1.001x (B=32) with the half-line mapping; this re-times 5 stages only after the whole-line remap changes the request pattern, and routes by a two-round gate instead of a fixed choice"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%"
---

# sk-full-line-5stage-select

## Change

Stacked on sk-full-line-gather (build on its code; claim restacks it). The stream-K module already instantiates <5 stages, ticket> (variant 2: 112672 B dynamic smem, 119 regs, 2 CTAs/SM) for diagnostics only; with the old half-line mapping s2's sk-diag measured it 0.971x (B=16) / 1.001x (B=32) of the 4-stage kernel, i.e. not depth-limited then. With whole-line requests the ring may now be the limiter, so re-time it and route by a gate. CHANGE (host side only, no CUDA edit): replace the constant _SK_VARIANT = 0 by an import-time frozen per-band table _SK_BAND_VARIANT (band A = 2 <= B <= 24 timed at B=16, band B = 25 <= B <= 64 timed at B=32; MappingProxyType, default 0). After _sk_validate passes: run variant 2 through the same _sk_case correctness cases as variant 0, then time variants 0 and 2 interleaved (_graph_of / _interleaved_us, 256 MB flush, empty graph subtracted, two rounds of 21 reps) on _splitkv_synthetic inputs with 80000-agentic lengths at B=16 and B=32; route variant 2 for a band only if it wins >= 1.02x in BOTH rounds with IQR <= 1.5x variant 0's, and only after _sk_replay_check-style two-state alternating replays of variant 2 at that band pass. _full_decode_sk takes the band's variant (config-only lookup; variant 2 uses the same single-level ticket epilogue, no combine). Print 'PATH CHOSEN: full decode B band <A|B>: <4|5> stages (t4 vs t5 us, both rounds)'. B=1 (two-level / c320) unchanged.

## Expected

- `*layer=full load=decode_16 *` faster: only if 5 stages win >= 1.02x after the whole-line remap (was 0.971x before it); 0.107%/us
- `*layer=full load=decode_32 *` faster: same, band B (was 1.001x); 0.060%/us
- `*layer=sliding *` flat: untouched

## Risk

Dispatch only after sk-full-line-gather harvests as gain or neutral (its parent); if the remap regressed, this is moot. Most likely outcome: gate keeps 4 stages and the run equals its parent; the printed t4/t5 pair is the answer either way. Decode-portable (same kernel in decode_layer).
