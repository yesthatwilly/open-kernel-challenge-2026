---
problem: muse_prefill_layer
slug: experts-dequant-threshold-turn4
created: 2026-10-23T230250Z
base: master
area: w13 marlin + w2 marlin (Marlin vs dequant crossover)
change: >
  Lower DEQUANT_MIN_TOKENS from 4096 to 2048 so the turn-4 shapes (T=3012, ~753 rows per local
  expert on average) also take the per-call dequant + Triton BF16 MoE path of
  experts-dequant-triton-moe-16k, while turn-1 (T=753) stays on Marlin. No other change. It
  measures the crossover: at 753 rows per expert Marlin streams each expert's 4-bit weights ~12
  times (64-row m-blocks) while the BF16 path pays its fixed ~1-1.5 ms dequant round trip against
  a 3.4-4.9 ms shape and then runs 128-row tiles on 0.3 TFLOP of GEMM work.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: 12 Marlin passes over 768 MB of 4-bit weights (plus Marlin dequant in the MMA loop) versus one 4 GB dequant round trip plus a 128-row-tile BF16 GEMM; the margin is thin, a loss pins the crossover above 753 rows per expert
  - shapes: "*load=turn_1*"
    direction: flat
    why: still Marlin (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: already on the dequant path in the parent; identical code (control)
risk: medium
depends_on: experts-dequant-triton-moe-16k
priority: 4
kind: primary
controls_for: null
review: required
status: hold
hold_reason: "standing answer: the per-call BF16 dequant family stays held; pursue in-kernel dequant only (experts-cuda-wgmma-mixed-input-grouped-gemm)"
---

# experts-dequant-threshold-turn4

## Change

Lower DEQUANT_MIN_TOKENS from 4096 to 2048 so the turn-4 shapes (T=3012, ~753 rows per local expert on average) also take the per-call dequant + Triton BF16 MoE path of experts-dequant-triton-moe-16k, while turn-1 (T=753) stays on Marlin. No other change. It measures the crossover: at 753 rows per expert Marlin streams each expert's 4-bit weights ~12 times (64-row m-blocks) while the BF16 path pays its fixed ~1-1.5 ms dequant round trip against a 3.4-4.9 ms shape and then runs 128-row tiles on 0.3 TFLOP of GEMM work.

## Expected

- `*load=turn_4*` faster: 12 Marlin passes over 768 MB of 4-bit weights (plus Marlin dequant in the MMA loop) versus one 4 GB dequant round trip plus a 128-row-tile BF16 GEMM; the margin is thin, a loss pins the crossover above 753 rows per expert
- `*load=turn_1*` flat: still Marlin (control)
- `*load=first_turn*` flat: already on the dequant path in the parent; identical code (control)

## Risk

Medium risk of a loss, no correctness risk beyond the parent's (the batch 3, ctx 5000, q_len 200 test with dist outliers stays on Marlin; the q_len 4096 test already exercised the path). Rules unchanged. If experts-dequant-overlap-attention has landed first, the dequant is already overlapped and the crossover moves lower; note which base the run used in the attempt file.
