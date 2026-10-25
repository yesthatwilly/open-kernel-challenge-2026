---
problem: muse_moe_w4a16
slug: marlin-exec-config-autotune-mid-m
created: 2026-10-25T042151Z
base: master
area: Marlin launch config (w13 + w2)
change: >
  On master (9080a71), at 8 <= num_tokens <= 32 only, choose Marlin's tile/exec config per T
  instead of its built-in heuristic. In prepare(): inspect.signature(moe_wna16_marlin_gemm); if it
  accepts thread_k, thread_n and/or blocks_per_sm (the vLLM-lineage kwargs), time candidate
  configs for w13 and w2 separately at T = 8, 16, 32 on synthetic skewed routings (cold L2, CUDA
  graph replays, interleaved, >= 30 reps): (thread_k, thread_n) in {(-1,-1) default, (128,128),
  (64,256), (128,64)} x blocks_per_sm in {-1, 1, 2}; reject any config that raises or misses
  rel-L2 <= 5e-3 vs the default; store the winner per (kernel, T) as Python ints in state only if
  it beats the default p50 by >= 3% and assert the stored choice is the printed winner. kernel_fn
  passes the stored kwargs by T (a shape, not a value). If the binding lacks the kwargs, print
  PATH CHOSEN = default (the run is then a master control - reviewer may prefer to drop it).
  Everything else master byte for byte. Mechanism: at 7-13 touched experts Marlin's heuristic
  picks a single small-M config; wave quantization over 132 SMs (e.g. w13 ~7 blocks x 32 N-tiles)
  is a plausible part of its ~1/3-of-peak bandwidth at mid M.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair ~95-114 us; a better config -5 to -15%
  - shapes: bs=16 *
    direction: faster
    why: ~170 us pair; -5 to -10%
  - shapes: bs=32 *
    direction: faster
    why: ~156-207 us pair; -5 to -10%
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-exec-config-autotune-mid-m

## Change

On master (9080a71), at 8 <= num_tokens <= 32 only, choose Marlin's tile/exec config per T instead of its built-in heuristic. In prepare(): inspect.signature(moe_wna16_marlin_gemm); if it accepts thread_k, thread_n and/or blocks_per_sm (the vLLM-lineage kwargs), time candidate configs for w13 and w2 separately at T = 8, 16, 32 on synthetic skewed routings (cold L2, CUDA graph replays, interleaved, >= 30 reps): (thread_k, thread_n) in {(-1,-1) default, (128,128), (64,256), (128,64)} x blocks_per_sm in {-1, 1, 2}; reject any config that raises or misses rel-L2 <= 5e-3 vs the default; store the winner per (kernel, T) as Python ints in state only if it beats the default p50 by >= 3% and assert the stored choice is the printed winner. kernel_fn passes the stored kwargs by T (a shape, not a value). If the binding lacks the kwargs, print PATH CHOSEN = default (the run is then a master control - reviewer may prefer to drop it). Everything else master byte for byte. Mechanism: at 7-13 touched experts Marlin's heuristic picks a single small-M config; wave quantization over 132 SMs (e.g. w13 ~7 blocks x 32 N-tiles) is a plausible part of its ~1/3-of-peak bandwidth at mid M.

## Expected

- `bs=8 *` faster: Marlin pair ~95-114 us; a better config -5 to -15%
- `bs=16 *` faster: ~170 us pair; -5 to -10%
- `bs=32 *` faster: ~156-207 us pair; -5 to -10%
- `bs=1 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

Kill: binding lacks the kwargs (log says default) or no (kernel, T) wins >= 3% in prepare, or judged rows flat. Prepare timings undercount the judge by 20-25% (LESSONS); use them only to rank.
