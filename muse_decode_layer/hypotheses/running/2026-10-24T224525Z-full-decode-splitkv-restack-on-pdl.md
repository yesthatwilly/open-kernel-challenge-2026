---
problem: muse_decode_layer
slug: full-decode-splitkv-restack-on-pdl
created: 2026-10-24T224525Z
base: master
area: FA3 call (full decode, batch > 1)
change: >
  Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated
  shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2
  (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl,
  f1f2f35), the way moe stacked shape-disjoint wins by 3-way merge. Take branch
  hyp/muse_decode_layer/full-decode-cuda-splitkv-port (patch
  done/neutral/full-decode-cuda-splitkv-port.patch) and 3-way merge its additions unchanged: its
  load_inline CUDA module (separate module name from the dense-chain TU, so both compile at
  import: ~41 s each, still far under the 15 min run), its gate (_splitkv_gate: full layer, q_len
  1, batch > 1, config only), the splitkv attention + combine/norm-gate kernels that replace the
  paired pad copies, FA3 flash_fwd + FlashAttnFwdCombine, unpad copies and FusedRMSNormGate on
  those shapes, and its prepare() self-check and fallback. Resolve every conflict in favour of
  master outside the splitkv branch of kernel_fn: the dense-chain P/R path after the o GEMM,
  attn_boundary router_dtype logic and _experts(align=...) stay exactly as master. Print both
  modules ON/OFF lines.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "measured +12.8% on the old base (attention-only change, disjoint from the FFN-side dense chain): ~35 us of 347.5"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "measured +10.3%: ~45 us of 535; headline ~+1.2%"
  - shapes: "*layer=sliding *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_*"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: batch 1 outside gate
risk: low
depends_on: full-decode-cuda-splitkv-port
priority: 2
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T231559Z
base_sha: f6cc1452343c4d93911ad7303c4a5d7878afd2fe
base_blob: b83f5ec1976c82d5b45dcd92b6c0818ce04eda9f
branch: hyp/muse_decode_layer/full-decode-splitkv-restack-on-pdl
worktree: .worktrees/muse_decode_layer/full-decode-splitkv-restack-on-pdl
mode: null
---

# full-decode-splitkv-restack-on-pdl

## Change

Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2 (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl, f1f2f35), the way moe stacked shape-disjoint wins by 3-way merge. Take branch hyp/muse_decode_layer/full-decode-cuda-splitkv-port (patch done/neutral/full-decode-cuda-splitkv-port.patch) and 3-way merge its additions unchanged: its load_inline CUDA module (separate module name from the dense-chain TU, so both compile at import: ~41 s each, still far under the 15 min run), its gate (_splitkv_gate: full layer, q_len 1, batch > 1, config only), the splitkv attention + combine/norm-gate kernels that replace the paired pad copies, FA3 flash_fwd + FlashAttnFwdCombine, unpad copies and FusedRMSNormGate on those shapes, and its prepare() self-check and fallback. Resolve every conflict in favour of master outside the splitkv branch of kernel_fn: the dense-chain P/R path after the o GEMM, attn_boundary router_dtype logic and _experts(align=...) stay exactly as master. Print both modules ON/OFF lines.

## Expected

- `*layer=full load=decode_16 *` faster: measured +12.8% on the old base (attention-only change, disjoint from the FFN-side dense chain): ~35 us of 347.5
- `*layer=full load=decode_32 *` faster: measured +10.3%: ~45 us of 535; headline ~+1.2%
- `*layer=sliding *` flat: outside gate: control
- `*load=verify_*` flat: outside gate: control
- `*load=decode_1 *` flat: batch 1 outside gate

## Risk

Low: both parents passed tests; shapes are disjoint. Watch combined import compile time (two load_inline TUs) and GPU memory of both workspaces. Lands the base that queued full-verify-cuda-splitkv-stage2 builds on.
