---
problem: muse_moe_w4a16
slug: stack-onewarp-hoisted-dispatchcombine-from17
created: 2026-10-25T000511Z
base: master
area: MoE align (bs<=8 one warp; 17..32 parallel; bs>=128 CUDA) + post-expert combine
change: >
  Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches,
  unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_TOKENS 32 -> 16, BSM=8 in the
  _align_parallel warm-up; scored 1.3250), so tokens 17..32 use the 33-program parallel align.
  That range sits under the CUDA dispatch rewrite's (> 32 tokens) and above the one-warp change
  (<= 8), so all four are run-time disjoint.
expected:
  - shapes: "*"
    direction: faster
    why: the three-way stack plus the bs=32 parallel align
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# stack-onewarp-hoisted-dispatchcombine-from17

## Change

Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches, unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_TOKENS 32 -> 16, BSM=8 in the _align_parallel warm-up; scored 1.3250), so tokens 17..32 use the 33-program parallel align. That range sits under the CUDA dispatch rewrite's (> 32 tokens) and above the one-warp change (<= 8), so all four are run-time disjoint.

## Expected

- `*` faster: the three-way stack plus the bs=32 parallel align

## Risk

Reset priority 4 variant (user-requested), claimed with --skip-review on that instruction.
