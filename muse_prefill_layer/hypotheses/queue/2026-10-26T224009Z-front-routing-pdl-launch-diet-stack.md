---
problem: muse_prefill_layer
slug: front-routing-pdl-launch-diet-stack
created: 2026-10-26T224009Z
base: master
area: launch overhead (front fusion + routing fusion + PDL chain regate, stacked)
change: >
  Stack on front-qknorm-rope-fp8-store-fusion-restack (depends_on; churn restacks its patch onto
  master). Add, as their own change texts specify (take the code from their branches
  hyp/muse_prefill_layer/<slug> or their done patches when they exist):
  routing-triton-hilo-sum-topk-localids (one side-stream Triton launch for the logits sum, the
  routing and the local-id remap) and experts-rega-pdl-chain-regate-iqr-slack
  (done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch with the IQR-slack gate rule). The
  three touch disjoint code: kernel_fn's attention front; kernel_fn's side-stream routing block
  and the local_ids passed to _experts; _experts_rega's two rega launches and prepare. Each keeps
  its own self-check and gate, so a part that fails falls back alone. Print each part's PATH
  CHOSEN at the end of prepare.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "front -6 to -10 us, routing -3 to -5 us, PDL -5 us: ~-14 to -20 us per turn_1 shape (~2% sliding)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: ~-20 to -30 us per turn_4 shape (~1%)
  - shapes: "*load=first_turn*"
    direction: faster
    why: front and routing ~-40 to -70 us (~0.4-0.6%)
risk: low
depends_on: front-qknorm-rope-fp8-store-fusion-restack
priority: 4
kind: primary
controls_for: null
prior: "stack-tonight-four-prefill: that stacked four flat align / router / zero-store changes on the old 1.07 base; this stacks three launch removals measured separately on the 1.468 base"
review: required
status: queued
---

# front-routing-pdl-launch-diet-stack

## Change

Stack on front-qknorm-rope-fp8-store-fusion-restack (depends_on; churn restacks its patch onto master). Add, as their own change texts specify (take the code from their branches hyp/muse_prefill_layer/<slug> or their done patches when they exist): routing-triton-hilo-sum-topk-localids (one side-stream Triton launch for the logits sum, the routing and the local-id remap) and experts-rega-pdl-chain-regate-iqr-slack (done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch with the IQR-slack gate rule). The three touch disjoint code: kernel_fn's attention front; kernel_fn's side-stream routing block and the local_ids passed to _experts; _experts_rega's two rega launches and prepare. Each keeps its own self-check and gate, so a part that fails falls back alone. Print each part's PATH CHOSEN at the end of prepare.

## Expected

- `*load=turn_1*` faster: front -6 to -10 us, routing -3 to -5 us, PDL -5 us: ~-14 to -20 us per turn_1 shape (~2% sliding)
- `*load=turn_4*` faster: ~-20 to -30 us per turn_4 shape (~1%)
- `*load=first_turn*` faster: front and routing ~-40 to -70 us (~0.4-0.6%)

## Risk

Size ~+1% headline: its parts are each below the judge margin, so this is where they can land together. Eligible once its parent is done.
