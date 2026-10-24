---
problem: muse_prefill_layer
slug: experts-triton-first-turn-unmasked-loads-l2-hints
created: 2026-10-24T224437Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)"
change: >
  Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles,
  configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKED A LOADS: today a_lo/a_hi
  are tl.load(a_ptrs, mask=token_mask[None, :], other=0.0) every K step, where padding slots carry
  the sentinel id num_valid_tokens (row num_valid // TOP_K is out of bounds). Before the loop,
  compute safe_token = tl.where(token_mask, offs_token, 0) and build a_ptrs from safe_token //
  TOP_K, then load a_lo/a_hi with no mask (padding columns read row 0, a real row; their products
  land only in output columns the masked store already drops). This removes per-element predicates
  from the hot loop so the activation slabs load as unpredicated 16-byte cp.async. The output
  store keeps mask=token_mask. (2) L2 HINTS: weight codes b and scales s are read once per
  (expert, W tile) per token-tile group: load them with eviction_policy='evict_first'; the
  activation slabs are re-read by all N // BLOCK_W weight tiles of the group: load them with
  eviction_policy='evict_last'. Applies to every config (the kernel is shared), but only the T >
  4096 path runs it in the timed call. _check_expert_gemms runs unchanged (it covers padding
  slots, the -1 block and the past-n_padded block) and still logs regs/spills/smem per variant;
  compare them with master's log line.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel x2 at 8149-8873 us: unpredicated activation loads and keeping the 2 MB activation slabs resident in L2 across 64 (w13) / 32 (w2) weight tiles, est. 0-4% on the row, 0-2.5% per first_turn shape"
  - shapes: "*load=turn_*"
    direction: neutral
    why: Marlin path
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-first-turn-unmasked-loads-l2-hints

## Change

Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKED A LOADS: today a_lo/a_hi are tl.load(a_ptrs, mask=token_mask[None, :], other=0.0) every K step, where padding slots carry the sentinel id num_valid_tokens (row num_valid // TOP_K is out of bounds). Before the loop, compute safe_token = tl.where(token_mask, offs_token, 0) and build a_ptrs from safe_token // TOP_K, then load a_lo/a_hi with no mask (padding columns read row 0, a real row; their products land only in output columns the masked store already drops). This removes per-element predicates from the hot loop so the activation slabs load as unpredicated 16-byte cp.async. The output store keeps mask=token_mask. (2) L2 HINTS: weight codes b and scales s are read once per (expert, W tile) per token-tile group: load them with eviction_policy='evict_first'; the activation slabs are re-read by all N // BLOCK_W weight tiles of the group: load them with eviction_policy='evict_last'. Applies to every config (the kernel is shared), but only the T > 4096 path runs it in the timed call. _check_expert_gemms runs unchanged (it covers padding slots, the -1 block and the past-n_padded block) and still logs regs/spills/smem per variant; compare them with master's log line.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel x2 at 8149-8873 us: unpredicated activation loads and keeping the 2 MB activation slabs resident in L2 across 64 (w13) / 32 (w2) weight tiles, est. 0-4% on the row, 0-2.5% per first_turn shape
- `*load=turn_*` neutral: Marlin path

## Risk

Speculative tuning of the Triton first_turn engine, which experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) would replace; run only if that rewrite does not land or as a cheap hedge. Orthogonal to experts-triton-persistent-grid-first-turn (grid) and the dispatch hypotheses (what the grid contains), so it can stack with them. Packed asm dequant was flat (the kernel is not dequant-ALU bound), so if this is also flat the remaining cost is tl.dot's synchronous lowering, the rewrite's premise.
