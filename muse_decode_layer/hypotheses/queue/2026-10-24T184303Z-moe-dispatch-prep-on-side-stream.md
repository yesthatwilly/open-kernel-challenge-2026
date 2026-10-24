---
problem: muse_decode_layer
slug: moe-dispatch-prep-on-side-stream
created: 2026-10-24T184303Z
base: master
area: MoE align/dispatch (stream placement)
change: >
  Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto
  state["side_stream"], after state["routing"](...), so it overlaps the pooler GEMM and the
  pre-expert RMSNorm on the main stream instead of running serially after the join. Split _experts
  into two functions with the same code. (a) _expert_dispatch(topk_ids, topk_weights, local_start,
  num_tokens, state) does exactly master: shifted = (topk_ids - local_start) % N_EXPERTS;
  local_ids = bank["expert_map"][shifted].contiguous(); tw =
  topk_weights.to(torch.float32).contiguous(); the block_size_m pick over BLOCK_SIZE_M_CANDIDATES
  with min(..., MARLIN_BLOCK_SIZE_M_CAP); sorted_ids, block_experts, n_padded =
  moe_align_block_size(local_ids, block_size_m, N_EXPERTS); returns them all. (b)
  _expert_compute(x, dispatch, state) holds the rest of master _experts unchanged (cache2/cache13
  allocation, w13 Marlin, silu_and_mul, w2 Marlin, the post-norm tail at both T ranges). In
  kernel_fn: inside the existing `with torch.cuda.stream(side_stream):` block, right after the
  routing call, call dispatch = _expert_dispatch(topk_ids, topk_weights, data["local_start"],
  tokens, state). On main, after expert_input = _gemm(ffn_input, dense["pooler"]) and BEFORE
  main_stream.wait_stream(side_stream), compute xn = state["pre_expert_norm"](expert_input). Then
  the existing join; when not capturing, call record_stream(main_stream) on every tensor created
  on the side stream (topk_weights, topk_ids, and every tensor in dispatch), as master already
  does for the first two. Then expert_output = _expert_compute(xn, dispatch, state). topk_ids
  returned unchanged. Kernel set, arguments and dtypes stay identical to master; only the stream
  and order change.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "the sub/remainder elementwise pair, the expert_map gather and SGLang align (~3.6 us _moe_align_small_numel + launches, ~7-8 us) leave the critical path: side branch (router gemv 5 + routing + these) ~16 us < pooler + pre-norm ~22 us; -6..-8 us of 145-157 us, +4-5%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "same kernels off the path: side ~22 us (router hi+lo 9.4 + add + routing + map + align) < pooler 20 + pre-norm 5.3; -6..-8 us, +2-3%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: same, +1.5-2.5%
  - shapes: "*load=verify_1 *"
    direction: faster
    why: same, +3%
  - shapes: "*load=verify_16 *"
    direction: faster
    why: SGLang align at T=128 ~10 us + map now overlaps pooler 24.7 + pre-norm 6; side ~33 us vs 31 so ~-8..-10 us, +1.5-2%
  - shapes: "*load=verify_32 *"
    direction: faster
    why: same, ~-8 us, +1%
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# moe-dispatch-prep-on-side-stream

## Change

Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto state["side_stream"], after state["routing"](...), so it overlaps the pooler GEMM and the pre-expert RMSNorm on the main stream instead of running serially after the join. Split _experts into two functions with the same code. (a) _expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state) does exactly master: shifted = (topk_ids - local_start) % N_EXPERTS; local_ids = bank["expert_map"][shifted].contiguous(); tw = topk_weights.to(torch.float32).contiguous(); the block_size_m pick over BLOCK_SIZE_M_CANDIDATES with min(..., MARLIN_BLOCK_SIZE_M_CAP); sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS); returns them all. (b) _expert_compute(x, dispatch, state) holds the rest of master _experts unchanged (cache2/cache13 allocation, w13 Marlin, silu_and_mul, w2 Marlin, the post-norm tail at both T ranges). In kernel_fn: inside the existing `with torch.cuda.stream(side_stream):` block, right after the routing call, call dispatch = _expert_dispatch(topk_ids, topk_weights, data["local_start"], tokens, state). On main, after expert_input = _gemm(ffn_input, dense["pooler"]) and BEFORE main_stream.wait_stream(side_stream), compute xn = state["pre_expert_norm"](expert_input). Then the existing join; when not capturing, call record_stream(main_stream) on every tensor created on the side stream (topk_weights, topk_ids, and every tensor in dispatch), as master already does for the first two. Then expert_output = _expert_compute(xn, dispatch, state). topk_ids returned unchanged. Kernel set, arguments and dtypes stay identical to master; only the stream and order change.

## Expected

- `*load=decode_1 *` faster: the sub/remainder elementwise pair, the expert_map gather and SGLang align (~3.6 us _moe_align_small_numel + launches, ~7-8 us) leave the critical path: side branch (router gemv 5 + routing + these) ~16 us < pooler + pre-norm ~22 us; -6..-8 us of 145-157 us, +4-5%
- `*load=decode_16 *` faster: same kernels off the path: side ~22 us (router hi+lo 9.4 + add + routing + map + align) < pooler 20 + pre-norm 5.3; -6..-8 us, +2-3%
- `*load=decode_32 *` faster: same, +1.5-2.5%
- `*load=verify_1 *` faster: same, +3%
- `*load=verify_16 *` faster: SGLang align at T=128 ~10 us + map now overlaps pooler 24.7 + pre-norm 6; side ~33 us vs 31 so ~-8..-10 us, +1.5-2%
- `*load=verify_32 *` faster: same, ~-8 us, +1%

## Risk

Read GPU busy and per-shape times; the kernel rows themselves should not change. Risk: if the side branch outgrows pooler + pre-norm (T=256) the saving shrinks, and side-stream contention with the pooler (the unverified suspect behind bf16-shape-gated-triton-gemms-small-t losing at T=8/16) could cost a little. OVERLAP: small-t-launch-minimal-dense-chain-pdl (in judge) removes the side stream and fuses map+align at T<=32 when its path is ON, and moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 when ON. If both land ON this is moot; if either falls back, this covers its range. Judge it against those runs PATH CHOSEN lines.
