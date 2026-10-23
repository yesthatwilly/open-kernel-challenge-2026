---
problem: muse_moe_w4a16
slug: align-single-token-ep
created: 2026-10-23T225502Z
base: master
area: MoE align/dispatch
change: >
  At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from
  sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-mapped ids (-1 for remote
  experts) instead of moe_align_block_size. baseline.py takes this path only when expert_map is
  None (the 'full' sharding), never under EP; this hypothesis tests whether the single-token
  kernel tolerates -1 ids and is cheaper than the general align (which at 256 or 32 experts runs a
  histogram + cumsum + scatter over a padded sorted_ids buffer) for the 8 ids of one token. Its
  outputs (sorted_ids, block_experts, n_padded) feed both Marlin GEMMs unchanged; all other shapes
  keep moe_align_block_size.
expected:
  - shapes: bs=1 *
    direction: faster
    why: bs=1 is 77 us; the general align is 2-3 launches over a 32- or 256-expert histogram for 8 ids, the single-token kernel is one small launch
  - shapes: bs=8 *
    direction: flat
    why: "control: num_tokens=8 keeps moe_align_block_size"
  - shapes: bs=16 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# align-single-token-ep

## Change

At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-mapped ids (-1 for remote experts) instead of moe_align_block_size. baseline.py takes this path only when expert_map is None (the 'full' sharding), never under EP; this hypothesis tests whether the single-token kernel tolerates -1 ids and is cheaper than the general align (which at 256 or 32 experts runs a histogram + cumsum + scatter over a padded sorted_ids buffer) for the 8 ids of one token. Its outputs (sorted_ids, block_experts, n_padded) feed both Marlin GEMMs unchanged; all other shapes keep moe_align_block_size.

## Expected

- `bs=1 *` faster: bs=1 is 77 us; the general align is 2-3 launches over a 32- or 256-expert histogram for 8 ids, the single-token kernel is one small launch
- `bs=8 *` flat: control: num_tokens=8 keeps moe_align_block_size
- `bs=16 *` flat: control: untouched

## Risk

Risks: if moe_align_single_token does not skip -1 ids, the bs=1 pre-timing test fails outright (a cheap, informative failure: log it as a do-not-retry). Superseded on bs=1 if triton-w2-grouped-gemv-small-m lands, since that removes the align from the small-M path; still worth a run as a low-cost probe of SGLang's own fast path. No rule exposure.
