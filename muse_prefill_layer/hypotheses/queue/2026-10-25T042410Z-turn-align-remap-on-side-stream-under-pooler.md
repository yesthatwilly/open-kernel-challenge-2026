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
review: proceed
status: queued
review_at: 2026-10-25T043310Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T043310Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, rules-clean placement change with real slack to hide into at turn_4 (side branch ~105 us + align 70 us under a ~260-300 us pooler); can never be slower than master barring SM contention. Small headline (~+0.5-0.8%) and only readable from the profile's GPU-busy/overlap lines, not from shape z.

### Justification
Profile (submit 502f2eab): _align_parallel_kernel 70.5/69.3 us at turn_4 and 16.9/16.8 us at turn_1, serial on the main stream after the join. Side branch today at turn_4: router GEMM (~39-43 us) + logits add (~30 us) + routing_kernel 32.7 us = ~105 us; the pooler is one of the big nvjet rows (~260-300 us at T=3012). GPU busy 4879 vs kernel-sum 4978 us shows the router branch already overlaps ~100 us, so concurrency works on this GPU. New critical path = max(pooler, branch+align) <= old max(pooler, branch) + align, so it is monotone except for SM/issue contention with the pooler. At turn_1 the pooler is only ~40-50 us and the branch ~35-50 us, so most of the 17 us will not hide - expect ~0 there. Weakest link: 'moe-align-side-stream' (old parity base) measured 1.0076 neutral for the same idea; the difference is that the align kernel moved now is 70 us, not SGLang's cheaper count_and_sort - plausible but unproven.

### Confounders on the current kernel
- router-hilo-one-gemm-n512 (in stack-tonight-five): sets the side branch length (one N=512 fp32-out GEMM + add); a longer branch would eat the slack. Unchanged here.
- moe-align-local-routes-parallel-port (in stack-tonight-five): is the 70 us kernel being moved; its 33-program latency-bound shape is what makes it overlap-friendly (few SMs, little smem).
- Queued experts-turn4-cuda-dequant-producer-ss-wgmma and experts-turn1 port also call _align_parallel inside _experts_marlin with their own bsm (128/32); merge conflict if several land - the helper should take bsm as a parameter.

### Interference and A/B needs
No companion: the effect is read directly from the profile (the _align_parallel row starts before the pooler ends; GPU busy drops vs kernel sum by ~+60 us at turn_4). A revert-A/B would not add information.

### Open questions
- Does the 33-CTA align co-reside with the coopA pooler kernel or wait for SM slots? The profile timeline answers it.
- Could the turn_1 branch exceed the pooler and just move the wait? Harmless, but if so restrict to T > 1024 in a follow-up.

