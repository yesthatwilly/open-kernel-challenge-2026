---
problem: muse_attention_decode
slug: full-verify1-cuda-streamk-batch1
created: 2026-10-24T165342Z
base: master
area: full-layer attention (batch-1 verify gate)
change: >
  Stacked on full-verify-cuda-streamk-splitkv (stage 2). Config-only widening of
  _splitkv_verify_gate from 1 < batch <= 64 to 1 <= batch <= 64, so FULL-layer verify at batch 1
  (q_len 8, tp8, page 1) runs master _pre_attention_kernel (PAIRED=False, as master batch-1
  already is) + A8 splitkv_verify_kernel + B8 splitkv_verify_combine instead of
  prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate.
  Stream-K needs no batch-specific code: one 80K-key request is ~1250 tiles over the grid. One
  extra config-only choice: at batch == 1 use grid = SM count (132, one CTA per SM, ~9.5 tiles
  each) instead of 2 x SM count, so the partial traffic (one 32 KB [8 tokens x 16 heads x 64] fp32
  partial per CTA, all of them merged by B8 for the single request) stays ~4 MB instead of ~8.6 MB
  against a 20 MB gather; batch > 1 keeps 2 x SM count. The grid is passed through the existing
  grid argument; the workspace is sized grid + B as in stage 2. Full decode_1 (q_len 1) and every
  other shape are unchanged.
expected:
  - shapes: "*layer=full load=verify_1*"
    direction: faster
    why: "incumbent 30.3 us shape (FA3 main 14.9 + combine 7.7 + prepare 2.8 + pre-attn 1.5 + norm-gate 1.1) against a 4.3 us floor (20.5 MB of KV); A8 at one CTA per SM streams ~155 KB per CTA with 4 tiles in flight (~7-9 us at 2.5-3 TB/s) and B8 merges 132 partials per (token, head) (~2-3 us): ~13-15 us shape, ~2x on the shape; full is 37% of the verify_1 load, ~1.2x on that load"
  - shapes: "*layer=full load=verify_16* *layer=full load=verify_32*"
    direction: flat
    why: batch > 1 keeps the stage 2 grid and code
  - shapes: "*load=decode_* *layer=sliding*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: full-verify-cuda-streamk-splitkv
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# full-verify1-cuda-streamk-batch1

## Change

Stacked on full-verify-cuda-streamk-splitkv (stage 2). Config-only widening of _splitkv_verify_gate from 1 < batch <= 64 to 1 <= batch <= 64, so FULL-layer verify at batch 1 (q_len 8, tp8, page 1) runs master _pre_attention_kernel (PAIRED=False, as master batch-1 already is) + A8 splitkv_verify_kernel + B8 splitkv_verify_combine instead of prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate. Stream-K needs no batch-specific code: one 80K-key request is ~1250 tiles over the grid. One extra config-only choice: at batch == 1 use grid = SM count (132, one CTA per SM, ~9.5 tiles each) instead of 2 x SM count, so the partial traffic (one 32 KB [8 tokens x 16 heads x 64] fp32 partial per CTA, all of them merged by B8 for the single request) stays ~4 MB instead of ~8.6 MB against a 20 MB gather; batch > 1 keeps 2 x SM count. The grid is passed through the existing grid argument; the workspace is sized grid + B as in stage 2. Full decode_1 (q_len 1) and every other shape are unchanged.

## Expected

- `*layer=full load=verify_1*` faster: incumbent 30.3 us shape (FA3 main 14.9 + combine 7.7 + prepare 2.8 + pre-attn 1.5 + norm-gate 1.1) against a 4.3 us floor (20.5 MB of KV); A8 at one CTA per SM streams ~155 KB per CTA with 4 tiles in flight (~7-9 us at 2.5-3 TB/s) and B8 merges 132 partials per (token, head) (~2-3 us): ~13-15 us shape, ~2x on the shape; full is 37% of the verify_1 load, ~1.2x on that load
- `*layer=full load=verify_16* *layer=full load=verify_32*` flat: batch > 1 keeps the stage 2 grid and code
- `*load=decode_* *layer=sliding*` flat: outside the gate

## Risk

Cheap follow-up once stage 2 passes correctness: one predicate and one grid choice. Batch-1 pairing lost 27-35% because of extra pad/unpad launches (LESSONS); this path has 3 launches (pre-attn, A8, B8) vs master 5, and no padding. Risk: B8 at batch 1 is 128 blocks each merging 132 partials (68 KB per block, ~8.6 MB total at 2 x SM; ~4.3 MB at 1 x SM), so check its own row in top_kernels; if B8 > 4 us, a two-level merge is the fix. Tests reaching the path: none at batch 1 page 1 full verify (batch 1 ctx 1048576 full q_len 8 is page 64), so the self-check must include a batch-1 q_len 8 synthetic case (80K-ish length and one short length) with the 1 x SM grid. If full-fused-triton-decode1-gate (running) lands, decode_1 is handled there; this hypothesis does not touch decode_1.
