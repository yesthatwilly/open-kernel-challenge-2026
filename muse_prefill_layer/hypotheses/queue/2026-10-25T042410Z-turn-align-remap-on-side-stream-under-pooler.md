---
problem: muse_prefill_layer
slug: turn-align-remap-on-side-stream-under-pooler
created: 2026-10-25T042410Z
base: master
area: "MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM"
change: >
  Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on
  the main stream after the router join: 16.9 us at turn_1 (1.7%) and 70.5 us at turn_4 (2.1%),
  while the side stream (router GEMM ~15-25 us + routing ~33 us at turn_4) finishes well inside
  the pooler GEMM on the main stream. In kernel_fn, inside the existing `with
  torch.cuda.stream(side_stream)` block, after state['routing'], also run local_ids =
  _local_expert_ids(topk_ids, data['local_start']) and, when tokens <= MARLIN_MAX_TOKENS and
  state['align_par_ok'], _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) with
  block_size_m from the same shape-only BLOCK_SIZE_M_CANDIDATES loop _experts_marlin uses (factor
  it into a helper both call). Pass the resulting (sorted_ids, block_experts, n_padded) into
  _experts_marlin through a new optional argument, which then skips its own align call. The
  existing main_stream.wait_stream(side_stream) stays where it is and the record_stream loop
  covers the new outputs. T > 4096 and the align_par_ok=False fallback are unchanged. Relation:
  moe-align-side-stream (done/neutral 1.0076 on the old parity base) moved SGLang's
  moe_align_block_size path; this re-tests the placement with master's current align kernel, which
  is the one now in the profile.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "_align_parallel (70.5 us) leaves the critical path under the pooler (~110-150 us at T=3012): turn_4 -40 to -70 us (-1.2 to -2%)"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "17 us align at T=753 partly hidden: -5 to -15 us (-0.5 to -1.5%), may be inside the router branch's slack"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T > 4096 path untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# turn-align-remap-on-side-stream-under-pooler

## Change

Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) and 70.5 us at turn_4 (2.1%), while the side stream (router GEMM ~15-25 us + routing ~33 us at turn_4) finishes well inside the pooler GEMM on the main stream. In kernel_fn, inside the existing `with torch.cuda.stream(side_stream)` block, after state['routing'], also run local_ids = _local_expert_ids(topk_ids, data['local_start']) and, when tokens <= MARLIN_MAX_TOKENS and state['align_par_ok'], _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) with block_size_m from the same shape-only BLOCK_SIZE_M_CANDIDATES loop _experts_marlin uses (factor it into a helper both call). Pass the resulting (sorted_ids, block_experts, n_padded) into _experts_marlin through a new optional argument, which then skips its own align call. The existing main_stream.wait_stream(side_stream) stays where it is and the record_stream loop covers the new outputs. T > 4096 and the align_par_ok=False fallback are unchanged. Relation: moe-align-side-stream (done/neutral 1.0076 on the old parity base) moved SGLang's moe_align_block_size path; this re-tests the placement with master's current align kernel, which is the one now in the profile.

## Expected

- `*load=turn_4*` faster: _align_parallel (70.5 us) leaves the critical path under the pooler (~110-150 us at T=3012): turn_4 -40 to -70 us (-1.2 to -2%)
- `*load=turn_1*` faster: 17 us align at T=753 partly hidden: -5 to -15 us (-0.5 to -1.5%), may be inside the router branch's slack
- `*load=first_turn*` unchanged: T > 4096 path untouched

## Risk

Kill: the profile shows _align_parallel overlapping the pooler (GPU busy below the kernel sum) and turn_4 does not move beyond 1% -> placement is not the cost. Risk: a longer side branch than the pooler at turn_1 just moves the wait; the judge sees no harm beyond noise.
