---
problem: muse_prefill_layer
slug: moe-align-side-stream
created: 2026-10-23T230250Z
base: master
area: MoE align/dispatch
change: >
  Keep the parity base and move three pieces of work that depend only on the routing outputs and
  local_start onto the existing side stream, inside the same torch.cuda.stream(side_stream) block
  right after topk_weights, topk_ids = state['routing'](...): the fused local-id remap kernel,
  topk_weights.to(torch.float32).contiguous(), and moe_align_block_size(local_ids, block_size_m,
  N_EXPERTS) (block_size_m is chosen from config tokens on the host, so the choice does not move).
  main_stream.wait_stream(side_stream) moves from before the local-id remap to just before the
  first moe_wna16_marlin_gemm call, after _gemm(ffn_input, pooler) and
  state['pre_expert_norm'](expert_input) have been issued on the main stream, and the
  non-capturing record_stream loop is extended to local_ids, topk_weights, sorted_ids,
  block_experts and n_padded. _experts takes the precomputed (local_ids, topk_weights, sorted_ids,
  block_experts, n_padded) instead of computing them; the cache13/cache2 allocation stays on the
  main stream. Nothing else changes. Under graph capture the side stream becomes a parallel
  branch, so the remap + align kernels (4-6 launches over T*8 ids) overlap the pooler GEMM and
  pre-expert norm instead of sitting serially between them and the Marlin GEMM.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: the remap+align chain (~15-30 us serial on the main stream) is hidden behind the pooler GEMM (~80-100 us at M=753); 1-3% of 1120-1465 us
  - shapes: "*load=turn_4*"
    direction: faster
    why: same chain over 24K ids hidden behind a 4x larger pooler GEMM; ~0.5-1%
  - shapes: "*load=first_turn*"
    direction: flat
    why: align over 131K ids is ~50 us of 17-26 ms; inside the interval
risk: low
depends_on: sglang-pipeline-parity
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# moe-align-side-stream

## Change

Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.cuda.stream(side_stream) block right after topk_weights, topk_ids = state['routing'](...): the fused local-id remap kernel, topk_weights.to(torch.float32).contiguous(), and moe_align_block_size(local_ids, block_size_m, N_EXPERTS) (block_size_m is chosen from config tokens on the host, so the choice does not move). main_stream.wait_stream(side_stream) moves from before the local-id remap to just before the first moe_wna16_marlin_gemm call, after _gemm(ffn_input, pooler) and state['pre_expert_norm'](expert_input) have been issued on the main stream, and the non-capturing record_stream loop is extended to local_ids, topk_weights, sorted_ids, block_experts and n_padded. _experts takes the precomputed (local_ids, topk_weights, sorted_ids, block_experts, n_padded) instead of computing them; the cache13/cache2 allocation stays on the main stream. Nothing else changes. Under graph capture the side stream becomes a parallel branch, so the remap + align kernels (4-6 launches over T*8 ids) overlap the pooler GEMM and pre-expert norm instead of sitting serially between them and the Marlin GEMM.

## Expected

- `*load=turn_1*` faster: the remap+align chain (~15-30 us serial on the main stream) is hidden behind the pooler GEMM (~80-100 us at M=753); 1-3% of 1120-1465 us
- `*load=turn_4*` faster: same chain over 24K ids hidden behind a 4x larger pooler GEMM; ~0.5-1%
- `*load=first_turn*` flat: align over 131K ids is ~50 us of 17-26 ms; inside the interval

## Risk

Low risk: pure stream reordering of kernels that already run; no math changes. Failure modes: a missing record_stream on a side-stream tensor when run eagerly (the tests run under the judge's graph capture too, but keep the loop); moe_align_block_size allocating inside the side stream is fine under the graph pool. If the baseline's FusedMuseSparkRouting output topk_weights is already fp32 the cast is a no-op and only the remap+align move. The gain is bounded by the align kernels' time, so a flat result on first_turn attributes nothing and is expected.
