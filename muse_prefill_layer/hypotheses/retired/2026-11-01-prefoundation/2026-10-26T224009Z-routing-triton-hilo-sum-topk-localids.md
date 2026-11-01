---
problem: muse_prefill_layer
slug: routing-triton-hilo-sum-topk-localids
created: 2026-10-26T224009Z
base: master
area: routing top-k (side-stream logits sum + FusedMuseSparkRouting + local-id remap in one Triton kernel)
change: >
  Built on master (blob f4751fc0). After the router GEMM (one N = 512 GEMM giving the hi | lo
  partial logits, fp32 [T, 512]) the side stream runs two more launches, logits = both[:, :256] +
  both[:, 256:] (a generic strided elementwise kernel, 31-42 us of side-stream time at turn_4 and
  first_turn) and SGLang's FusedMuseSparkRouting kernel, and after the join the main stream runs
  _local_expert_ids on the critical path. CHANGE: one Triton kernel on the side stream replaces
  all three. Per token row (BLOCK_T rows per program, [BLOCK_T, 256] tiles): read both halves, s =
  sigmoid((hi + lo) * 0.1767766952966369) in fp32, choice = s + router_bias, the top 8 of choice
  by 8 rounds of a row max with lowest-index tie-break (written in that descending order),
  topk_weights = s[top8] / (sum s[top8] + 1e-15) (route_scale 1), topk_ids int32 [T, 8], and
  local_ids int32 [T, 8] = ((id - local_start) & 255) when < 32 else -1 (exactly
  _local_expert_ids). kernel_fn passes these local_ids to _experts and drops the _local_expert_ids
  launch. SELF-CHECK in prepare at T = 753, 3012 and 16384 on random logits, popularity-skewed
  logits and logits with planted near-ties, against master's add + routing + remap: per row the
  same id set except rows whose 8th and 9th choice scores differ by < 1e-6, weights within 1e-6
  relative, local_ids identical; else OFF. TIMING GATE: cold-L2 graph of master's three launches
  vs the fused one, 2 rounds x 11 alternations; ON at <= 0.9; state['routing_fused_ok'][T class].
  Print both medians and PATH CHOSEN.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "at T=753 the side chain is only partly hidden under the one-wave pooler (kernel sum minus GPU busy is 13 us against ~31 us of side work) and _local_expert_ids is on the main path: ~-3 to -5 us (~0.5% sliding)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: ~-3 to -6 us from the serialized side tail and the remap launch
  - shapes: "*load=first_turn*"
    direction: faster
    why: "the strided add moves ~50 MB at T=16384: ~-10 to -20 us where the side chain overruns the pooler (~0.1-0.2%)"
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "router-hilo-one-gemm-n512: that merged the two router GEMMs into the N=512 GEMM master now runs and left the add and the routing as separate launches; this keeps that GEMM and folds the add, the routing kernel and the main-stream remap into one launch"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%, stack-only"
---

# routing-triton-hilo-sum-topk-localids

## Change

Built on master (blob f4751fc0). After the router GEMM (one N = 512 GEMM giving the hi | lo partial logits, fp32 [T, 512]) the side stream runs two more launches, logits = both[:, :256] + both[:, 256:] (a generic strided elementwise kernel, 31-42 us of side-stream time at turn_4 and first_turn) and SGLang's FusedMuseSparkRouting kernel, and after the join the main stream runs _local_expert_ids on the critical path. CHANGE: one Triton kernel on the side stream replaces all three. Per token row (BLOCK_T rows per program, [BLOCK_T, 256] tiles): read both halves, s = sigmoid((hi + lo) * 0.1767766952966369) in fp32, choice = s + router_bias, the top 8 of choice by 8 rounds of a row max with lowest-index tie-break (written in that descending order), topk_weights = s[top8] / (sum s[top8] + 1e-15) (route_scale 1), topk_ids int32 [T, 8], and local_ids int32 [T, 8] = ((id - local_start) & 255) when < 32 else -1 (exactly _local_expert_ids). kernel_fn passes these local_ids to _experts and drops the _local_expert_ids launch. SELF-CHECK in prepare at T = 753, 3012 and 16384 on random logits, popularity-skewed logits and logits with planted near-ties, against master's add + routing + remap: per row the same id set except rows whose 8th and 9th choice scores differ by < 1e-6, weights within 1e-6 relative, local_ids identical; else OFF. TIMING GATE: cold-L2 graph of master's three launches vs the fused one, 2 rounds x 11 alternations; ON at <= 0.9; state['routing_fused_ok'][T class]. Print both medians and PATH CHOSEN.

## Expected

- `*load=turn_1*` faster: at T=753 the side chain is only partly hidden under the one-wave pooler (kernel sum minus GPU busy is 13 us against ~31 us of side work) and _local_expert_ids is on the main path: ~-3 to -5 us (~0.5% sliding)
- `*load=turn_4*` faster: ~-3 to -6 us from the serialized side tail and the remap launch
- `*load=first_turn*` faster: the strided add moves ~50 MB at T=16384: ~-10 to -20 us where the side chain overruns the pooler (~0.1-0.2%)

## Risk

The judge accepts near-tie routing differences (each pick within 0.05 of the reference's 8th best; required picks only those 0.03-0.04 clear of the 9th). The router GEMM itself is untouched, so DIAGNOSIS's router-GEMM Do-not-try does not apply; the overlap figures are from the best profile (kernel total vs GPU busy). Size ~+0.2% headline: lands only stacked (front-routing-pdl-launch-diet-stack).
