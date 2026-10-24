---
problem: muse_attention_decode
slug: paired-heads-launch-cleanup-ab-tune-batch-gate
created: 2026-10-24T104027Z
base: master
area: FA3 call (head packing) gate
change: >
  From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing
  predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'],
  and use this single predicate both for the pre-attention kernel's q layout (pass it as a
  constexpr, e.g. PAIRED, plus the q buffer shape [tokens, Hq, 128] vs [tokens, Hq, 64]) and for
  the FA3/epilogue branch. When paired is false on a full layer (B=1 decode_1/verify_1, and odd
  Hkv), run exactly master's pre-pairing path. The pre-attention kernel writes the unpadded
  [tokens, Hq, 64] FP8 q. FA3 gets the original k_cache/v_cache views, descale expanded to (B,
  Hkv), and scheduler_metadata=data['scheduler_metadata'] (the judge's). The output goes through
  SGLang's rms_norm_gate on out.reshape(tokens, Hq*64), as before fa3-hdim128-head-pairing.
  Sliding layers are unchanged. B>=16 full layers are byte-identical to the parent.
expected:
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: "back to the unpaired FA3 + judge metadata path, ~27.9 us (fused-preattention-triton result) against an estimated ~31-32 us for the ungated cleanup: no 128-wide combine (10.3 vs 7.1 us) and no internal prepare launch"
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: "~30.0 us unpaired against an estimated ~39-40 us ungated: the paired single-super-head verify FA3 (18.3 vs 14.9 us) and combine (12.5 vs 7.3 us) go away"
  - shapes: "*layer=full load=*_16*"
    direction: flat
    why: same code path as the parent; a mismatch measures run-to-run drift
  - shapes: "*layer=full load=*_32*"
    direction: flat
    why: same code path as the parent
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
risk: low
depends_on: paired-heads-launch-cleanup
priority: 2
kind: ab-tune
controls_for: paired-heads-launch-cleanup
review: required
status: queued
---

# paired-heads-launch-cleanup-ab-tune-batch-gate

## Change

From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'], and use this single predicate both for the pre-attention kernel's q layout (pass it as a constexpr, e.g. PAIRED, plus the q buffer shape [tokens, Hq, 128] vs [tokens, Hq, 64]) and for the FA3/epilogue branch. When paired is false on a full layer (B=1 decode_1/verify_1, and odd Hkv), run exactly master's pre-pairing path. The pre-attention kernel writes the unpadded [tokens, Hq, 64] FP8 q. FA3 gets the original k_cache/v_cache views, descale expanded to (B, Hkv), and scheduler_metadata=data['scheduler_metadata'] (the judge's). The output goes through SGLang's rms_norm_gate on out.reshape(tokens, Hq*64), as before fa3-hdim128-head-pairing. Sliding layers are unchanged. B>=16 full layers are byte-identical to the parent.

## Expected

- `*layer=full load=decode_1 page*` faster: back to the unpaired FA3 + judge metadata path, ~27.9 us (fused-preattention-triton result) against an estimated ~31-32 us for the ungated cleanup: no 128-wide combine (10.3 vs 7.1 us) and no internal prepare launch
- `*layer=full load=verify_1 page*` faster: ~30.0 us unpaired against an estimated ~39-40 us ungated: the paired single-super-head verify FA3 (18.3 vs 14.9 us) and combine (12.5 vs 7.3 us) go away
- `*layer=full load=*_16*` flat: same code path as the parent; a mismatch measures run-to-run drift
- `*layer=full load=*_32*` flat: same code path as the parent
- `*layer=sliding*` flat: untouched

## Risk

Decides whether pairing must be gated to batch > 1 or whether the launch cleanup alone makes batch 1 acceptable; also the likely promotion candidate (~+4-5% geomean over the ungated cleanup if the estimates hold).
