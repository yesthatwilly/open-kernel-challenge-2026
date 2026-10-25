---
problem: muse_prefill_layer
slug: marlin-w2-atomic-add-turn
created: 2026-10-25T042410Z
base: master
area: "w2 Marlin call on the Marlin path (T <= 4096): use_atomic_add, picked per shape class in prepare"
change: >
  Built on master (a81d19c). Port of muse_moe_w4a16 w2-marlin-atomic-add-small-m (checked 1.3445
  there: bs=1 Marlin 48.8 -> 44.9 us and its slow mode 44/100 -> 10/100 replays; see
  muse_moe_w4a16/submission.py _w2_atomic_self_check and its patch in that problem's done/). Only
  the second moe_wna16_marlin_gemm call in _experts_marlin changes: use_atomic_add=True
  (use_fp32_reduce stays True) when state['w2_mode'][cls] says so, cls in {'turn1' (T <= 1024),
  'turn4' (1024 < T <= 4096)}. prepare() picks each class's mode among 'atomic' (no memset),
  'atomic_zero' (cache3.zero_() first; cache3 aliases cache13, which holds w13's output when w2
  runs, so a non-zeroing atomic path that needs a zeroed C must fail its check) and 'master', on
  synthetic soft routings at T=753 and T=3012: correctness vs master's own w2 output (rel L2 <=
  1e-3, no non-finite), then CUDA-graph cold-L2 interleaved timing of the whole _experts_marlin; a
  non-master mode is chosen only at <= 0.97x. Print PATH CHOSEN per class with both times. w13,
  align, silu and the weighted sum are untouched.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "Marlin w2 at T=753 splits K=4096 across CTAs; atomic add removes the serialized global reduce: Marlin pair 650-770 us -> -2 to -5% if chosen"
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: moe saw no change at larger M (bs 8-32); expected master pick
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: Triton path
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-w2-atomic-add-turn

## Change

Built on master (a81d19c). Port of muse_moe_w4a16 w2-marlin-atomic-add-small-m (checked 1.3445 there: bs=1 Marlin 48.8 -> 44.9 us and its slow mode 44/100 -> 10/100 replays; see muse_moe_w4a16/submission.py _w2_atomic_self_check and its patch in that problem's done/). Only the second moe_wna16_marlin_gemm call in _experts_marlin changes: use_atomic_add=True (use_fp32_reduce stays True) when state['w2_mode'][cls] says so, cls in {'turn1' (T <= 1024), 'turn4' (1024 < T <= 4096)}. prepare() picks each class's mode among 'atomic' (no memset), 'atomic_zero' (cache3.zero_() first; cache3 aliases cache13, which holds w13's output when w2 runs, so a non-zeroing atomic path that needs a zeroed C must fail its check) and 'master', on synthetic soft routings at T=753 and T=3012: correctness vs master's own w2 output (rel L2 <= 1e-3, no non-finite), then CUDA-graph cold-L2 interleaved timing of the whole _experts_marlin; a non-master mode is chosen only at <= 0.97x. Print PATH CHOSEN per class with both times. w13, align, silu and the weighted sum are untouched.

## Expected

- `*load=turn_1*` faster: Marlin w2 at T=753 splits K=4096 across CTAs; atomic add removes the serialized global reduce: Marlin pair 650-770 us -> -2 to -5% if chosen
- `*load=turn_4*` unchanged: moe saw no change at larger M (bs 8-32); expected master pick
- `*load=first_turn*` unchanged: Triton path

## Risk

Kill: master picked for both classes, or Marlin rows within +-3% (turn_1 drift is up to 10%, so judge only by PATH CHOSEN plus runs_ns mode counts, LESSONS 03:41Z). Cheap single-flag port.
