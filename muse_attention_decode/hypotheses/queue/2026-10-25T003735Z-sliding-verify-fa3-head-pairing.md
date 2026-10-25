---
problem: muse_attention_decode
slug: sliding-verify-fa3-head-pairing
created: 2026-10-25T003735Z
base: master
area: FA3 call (head packing, sliding verify)
change: >
  Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. Today
  paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv, so sliding verify_16/32 run
  the unpaired FA3 path (two 64-byte KV rows gathered per token, then sglang rms_norm_gate). New
  predicate: paired = hkv % 2 == 0 and batch > 1 and not splitkv (drop window == 0). Sliding
  decode never reaches this code (it returns through _sliding_forward), so only sliding verify at
  batch 16/32 changes. On the paired branch everything is master code as it already runs for full
  verify_16/32: _pre_attention_kernel with PAIRED=True writes the zero-padded [tokens, 16, 128]
  query; flash_attn_with_kvcache on the [pages, 1, 1, 128] paired cache views with
  scheduler_metadata=None (sliding already passes None, the judge gives none on sliding),
  causal=True, the SAME window_size=(window - 1, 0) as now (the window is per query/key position,
  independent of the head layout), the same descale expanded to attn_heads = 1; then
  _paired_norm_gate_kernel instead of rms_norm_gate. Pairing is exact: each query sits in its own
  head half of the 128-wide query, so the zero half adds exactly 0 to each logit, and FA3 maps
  query head h to super-head h // 16, keeping the grouping. Full layers, sliding verify_1 and all
  decode shapes unchanged.
expected:
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: master unpaired FA3 main 10.7 + combine 4.1 + norm-gate of a 22.7 us shape; pairing gained 14-24% on full verify_16/32 FA3 main by gathering one 128 B row per token, and _paired_norm_gate_kernel replaces sglang rms_norm_gate; expect 19-21 us
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: master FA3 main 14.5 + combine 4.3 of 28.7 us; expect 24-27 us
  - shapes: "*layer=sliding load=verify_1 * *load=decode_* *layer=full*"
    direction: flat
    why: batch 1 stays unpaired (pairing lost 27-35% there); other shapes untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-verify-fa3-head-pairing

## Change

Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. Today paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv, so sliding verify_16/32 run the unpaired FA3 path (two 64-byte KV rows gathered per token, then sglang rms_norm_gate). New predicate: paired = hkv % 2 == 0 and batch > 1 and not splitkv (drop window == 0). Sliding decode never reaches this code (it returns through _sliding_forward), so only sliding verify at batch 16/32 changes. On the paired branch everything is master code as it already runs for full verify_16/32: _pre_attention_kernel with PAIRED=True writes the zero-padded [tokens, 16, 128] query; flash_attn_with_kvcache on the [pages, 1, 1, 128] paired cache views with scheduler_metadata=None (sliding already passes None, the judge gives none on sliding), causal=True, the SAME window_size=(window - 1, 0) as now (the window is per query/key position, independent of the head layout), the same descale expanded to attn_heads = 1; then _paired_norm_gate_kernel instead of rms_norm_gate. Pairing is exact: each query sits in its own head half of the 128-wide query, so the zero half adds exactly 0 to each logit, and FA3 maps query head h to super-head h // 16, keeping the grouping. Full layers, sliding verify_1 and all decode shapes unchanged.

## Expected

- `*layer=sliding load=verify_16*` faster: master unpaired FA3 main 10.7 + combine 4.1 + norm-gate of a 22.7 us shape; pairing gained 14-24% on full verify_16/32 FA3 main by gathering one 128 B row per token, and _paired_norm_gate_kernel replaces sglang rms_norm_gate; expect 19-21 us
- `*layer=sliding load=verify_32*` faster: master FA3 main 14.5 + combine 4.3 of 28.7 us; expect 24-27 us
- `*layer=sliding load=verify_1 * *load=decode_* *layer=full*` flat: batch 1 stays unpaired (pairing lost 27-35% there); other shapes untouched

## Risk

Risk: sliding verify covers only ~2055 keys per request, so FA3 there is more latency/launch bound than the 80K full shapes where pairing paid; the doubled QK math on the zero-padded query and a different FA3 split decision (one kv head instead of two) could cost as much as the gather saves. Cheap, config-only, attributable to one predicate.
