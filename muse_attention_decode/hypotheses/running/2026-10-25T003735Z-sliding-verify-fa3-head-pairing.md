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
review: proceed
status: running
review_at: 2026-10-25T003958Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T003958Z
base_sha: 475f0ab5e572555bb2f4a4398a88bef1075a71ec
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/sliding-verify-fa3-head-pairing
worktree: .worktrees/muse_attention_decode/sliding-verify-fa3-head-pairing
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T003958Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, rule-clean, one-predicate change; per-kernel profile rows (FA3 main, FlashAttnFwdCombine, prepare_varlen, _pre_attention_kernel, norm-gate) make the result readable without a companion. But the mechanism carries the full-layer bandwidth argument to a latency-bound shape; most likely outcome is flat or a small loss on sliding verify_16/32.

### Justification
Correctness holds: the zero half adds 0 to each logit, FA3 maps query head h to super-head h//16, window_size is per position, ROPE with PAIRED is already sound. Performance is the weak part: sliding verify_16 reads ~8.4 MB (~2.8 us at 3 TB/s) vs FA3 main 10.7 us; verify_32 ~17 MB vs 14.5 us, so both are latency/parallelism-bound and halving gathered rows has little to act on. The closest analogue, full verify_1, showed pairing slowing FA3 main 14.9->18.1 us and combine 7.3->12.4 us after the launch cleanup. Pairing halves (request, kv-head) tiles (32->16) at 2x rows and 2x head dim, so FA3 likely picks more splits and combine grows. The epilogue swap is probably a cost: _paired_norm_gate_kernel 2.5/3.9 us vs sglang norm_gate <1.8 us, ~+1/+2 us. Weight: sliding is ~29%/22% of the verify_16/32 loads, so +-10% on the shape moves the load ~+-2-3%, near the CI floor.

### Confounders on the current kernel
- paired-heads-launch-cleanup (+ab-tune-batch-gate): supplies paired q layout and _paired_norm_gate_kernel; its own profile row, attributable.
- fa3-hdim128-head-pairing: same mechanism, gain measured only in the bandwidth-bound regime; its batch-1 results are the relevant prior.
- fused-preattention-triton: PAIRED writes 2x q bytes (<33 KB), ~0.1-0.3 us, its own row.
- sliding decode / splitkv paths do not touch sliding verify; they are the flat control (sliding decode_1 drifts 2-3.5% run to run).

### Interference and A/B needs
No companion warranted. Harvest on sliding verify_16/32 per-shape ratios and FA3 main + combine us vs master (10.7+4.1 and 14.5+4.3), not on the headline. 'Main down, combine and epilogue up, net flat' is a valid outcome pointing to a follow-up that folds the epilogue differently, not a rerun.

### Open questions
- Does the image's FA3 contain the e4m3 hdim-128 paged pack_gqa Is_local=true instantiation? If not, the sliding q_len-8 tests fail at dispatch: harvest as failed, not regression.
- Which split count does FA3 pick for 16/32 paired tiles of ~2K keys? If combine grows >1.5 us, pairing on sliding is dead.
- Is _paired_norm_gate_kernel slower than sglang norm_gate at 128/256 tokens here?
- The kv_scale 1/64 sliding verify test exercises descale (B,1) under a local window; a failure there points at descale.
- LESSONS 103-105 attributes the batch-1 pairing loss to launches, but post-cleanup rows show FA3 main +21% and combine +70%; consider correcting that lesson.

