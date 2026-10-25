---
problem: muse_attention_decode
slug: full-verify-fa3-num-splits-band-gate
created: 2026-10-25T153027Z
base: master
area: FA3 call (full verify split count)
change: >
  Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9
  + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and B=16/32 paired (main 154.5/293.1
  + combine 5.9/7.7 + prepare 3.0/3.6, profile 252abb7f11ec48c3). At B=1 FA3 splits one request's
  80K keys over ~all SMs and the combine (7.3 us, half the main kernel) merges every split's [8 x
  16 x 64] fp32 partial; fewer, longer splits trade main-kernel parallelism for a cheaper combine.
  num_splits is a cap under dynamic split (LESSONS), so lowering it is effective. CHANGE: in
  kernel_fn's FA3 branch for full verify only (window == 0, q_len 8; config-only) pass
  num_splits=_FV_SPLITS.get(band, 0), band 1 for B=1, 16 for 2 <= B <= 24 (paired), 32 for 25 <= B
  <= 64 (paired); everything else keeps the default. _FV_SPLITS is an import-time frozen table
  chosen in prepare per band at batch x 80000 (agentic lengths for 16/32): time master's step
  after pre-attention (B=1: FA3 + combine + rms_norm_gate; B=16/32: paired FA3 + combine +
  _paired_norm_gate_kernel) for num_splits in {0 (auto), 16, 24, 32, 48, 64, 96} at B=1 and {0, 4,
  6, 8, 12, 16} at B=16/32, interleaved cold-L2 graph replays (256 MB flush, empty graph
  subtracted, median of 31); print every number; route the fastest only at >= 1.03x vs auto and
  output rel L2 < 1e-2 vs auto, else keep auto. Self-check the chosen settings on the full-verify
  tests' shapes at page 1 (B 2 ctx 100000, B 4 ctx 6000 edges, kv 0.25) against master's output.
  Disjoint from full-verify-wgmma-ws-s1-producer-split (that one replaces the B=16/32 FA3 only
  where its gate routes; this one is measured against the FA3 path itself). RULES (every path):
  CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in
  kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32
  ticket arrays, zeroed once and returned to 0 by every launch, are fine).
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: combine 7.3 us is ~half of FA3 main 14.9; fewer splits should cut 2-4 us; shape 30.5, floor 4.3
  - shapes: "*layer=full load=verify_16 *"
    direction: flat
    why: combine 5.9 us of 163; small possible gain, gate keeps auto otherwise
  - shapes: "*layer=full load=verify_32 *"
    direction: flat
    why: combine 7.7 us of 302; small possible gain
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# full-verify-fa3-num-splits-band-gate

## Change

Built on current master (no depends_on). Full verify still runs FA3: B=1 unpaired (FA3 main 14.9 + combine 7.3 + prepare 2.8 + norm-gate 1.1 us, shape 30.5) and B=16/32 paired (main 154.5/293.1 + combine 5.9/7.7 + prepare 3.0/3.6, profile 252abb7f11ec48c3). At B=1 FA3 splits one request's 80K keys over ~all SMs and the combine (7.3 us, half the main kernel) merges every split's [8 x 16 x 64] fp32 partial; fewer, longer splits trade main-kernel parallelism for a cheaper combine. num_splits is a cap under dynamic split (LESSONS), so lowering it is effective. CHANGE: in kernel_fn's FA3 branch for full verify only (window == 0, q_len 8; config-only) pass num_splits=_FV_SPLITS.get(band, 0), band 1 for B=1, 16 for 2 <= B <= 24 (paired), 32 for 25 <= B <= 64 (paired); everything else keeps the default. _FV_SPLITS is an import-time frozen table chosen in prepare per band at batch x 80000 (agentic lengths for 16/32): time master's step after pre-attention (B=1: FA3 + combine + rms_norm_gate; B=16/32: paired FA3 + combine + _paired_norm_gate_kernel) for num_splits in {0 (auto), 16, 24, 32, 48, 64, 96} at B=1 and {0, 4, 6, 8, 12, 16} at B=16/32, interleaved cold-L2 graph replays (256 MB flush, empty graph subtracted, median of 31); print every number; route the fastest only at >= 1.03x vs auto and output rel L2 < 1e-2 vs auto, else keep auto. Self-check the chosen settings on the full-verify tests' shapes at page 1 (B 2 ctx 100000, B 4 ctx 6000 edges, kv 0.25) against master's output. Disjoint from full-verify-wgmma-ws-s1-producer-split (that one replaces the B=16/32 FA3 only where its gate routes; this one is measured against the FA3 path itself). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=full load=verify_1 *` faster: combine 7.3 us is ~half of FA3 main 14.9; fewer splits should cut 2-4 us; shape 30.5, floor 4.3
- `*layer=full load=verify_16 *` flat: combine 5.9 us of 163; small possible gain, gate keeps auto otherwise
- `*layer=full load=verify_32 *` flat: combine 7.7 us of 302; small possible gain
- `*layer=sliding *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Full verify_1 is 45% of the verify_1 load (16 x 30.5 of 1095 us). Low cost: config + prepare timing only.
