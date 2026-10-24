---
problem: muse_decode_layer
slug: sliding-decode1-fused-restack-on-pdl
created: 2026-10-24T224525Z
base: master
area: FA3 call (sliding decode, batch 1)
change: >
  Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4
  us, +4.4%, every other shape flat; it also carried the +8.75% sliding decode_1 row of
  decode-restack-sliding-port-paired-pad-gemms-t1) onto current master (f1f2f35, dense-chain PDL).
  3-way merge branch hyp/muse_decode_layer/sliding-fused-triton-decode1-port (patch
  done/neutral/sliding-fused-triton-decode1-port.patch) unchanged: its Triton helpers and
  _sliding_fused_kernel, the config gate (window != 0 and q_len == 1 and batch == 1), its
  prepare() warm-up and fallback; on that branch the one fused launch replaces FusedQKNorm, the
  FP8 casts, store_cache, flash_attn_with_kvcache + combine and o_norm_gate, and its output feeds
  the unchanged o GEMM. Bring nothing from the restack parent (no T=1 Triton GEMMs, no paired
  pad). If sliding-fused-batched-decode-port (running, batch > 1) has landed by claim time, its
  copies of the same helpers are already on master: reuse them, only add batch == 1 to the gate
  with that parent decode_1 split count, never two copies. Everything from the o GEMM on is
  master.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "attention chain FA3 10.4 + combine 4.0 + qk_norm/casts/store/norm-gate nodes -> one fused launch: measured +4.4% on old base, ~5-6 us of 132.7"
  - shapes: "*layer=full *"
    direction: flat
    why: outside gate
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: outside gate
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: outside gate
risk: low
depends_on: sliding-fused-triton-decode1-port
priority: 3
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T231559Z
base_sha: 2569d7e0d6f208921d432fd4237dc483601b39ca
base_blob: e33a32aa5580fbd5b38764ea287a6364c865432e
branch: hyp/muse_decode_layer/sliding-decode1-fused-restack-on-pdl
worktree: .worktrees/muse_decode_layer/sliding-decode1-fused-restack-on-pdl
mode: null
---

# sliding-decode1-fused-restack-on-pdl

## Change

Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4 us, +4.4%, every other shape flat; it also carried the +8.75% sliding decode_1 row of decode-restack-sliding-port-paired-pad-gemms-t1) onto current master (f1f2f35, dense-chain PDL). 3-way merge branch hyp/muse_decode_layer/sliding-fused-triton-decode1-port (patch done/neutral/sliding-fused-triton-decode1-port.patch) unchanged: its Triton helpers and _sliding_fused_kernel, the config gate (window != 0 and q_len == 1 and batch == 1), its prepare() warm-up and fallback; on that branch the one fused launch replaces FusedQKNorm, the FP8 casts, store_cache, flash_attn_with_kvcache + combine and o_norm_gate, and its output feeds the unchanged o GEMM. Bring nothing from the restack parent (no T=1 Triton GEMMs, no paired pad). If sliding-fused-batched-decode-port (running, batch > 1) has landed by claim time, its copies of the same helpers are already on master: reuse them, only add batch == 1 to the gate with that parent decode_1 split count, never two copies. Everything from the o GEMM on is master.

## Expected

- `*layer=sliding load=decode_1 *` faster: attention chain FA3 10.4 + combine 4.0 + qk_norm/casts/store/norm-gate nodes -> one fused launch: measured +4.4% on old base, ~5-6 us of 132.7
- `*layer=full *` flat: outside gate
- `*layer=sliding load=decode_16 *` flat: outside gate
- `*layer=sliding load=verify_*` flat: outside gate

## Risk

Low: parent passed tests; one shape. Conflicts with dense-chain-stage2-attn-side-pdl on sliding decode_1 (both replace the attention-side chain there); whichever lands first, the other gates out batch-1 sliding decode.
