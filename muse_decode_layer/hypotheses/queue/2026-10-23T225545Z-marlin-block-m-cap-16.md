---
problem: muse_decode_layer
slug: marlin-block-m-cap-16
created: 2026-10-23T225545Z
base: master
area: w13 marlin
change: >
  Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks
  BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9,
  which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as
  chosen and clamps the T=128/256 picks to 16. Mechanism: routing is skewed (a few experts take
  most tokens), so with block 48/64 the many cold experts holding 1-4 tokens each are padded to a
  48/64-row MMA tile and the hot expert gets only 4-6 tiles to spread over 132 SMs; block 16 pads
  cold experts 3-4x less and gives the hot expert 3-4x more m-blocks for the same 25 MB weight
  stream. The same block_size_m goes to moe_align_block_size and both moe_wna16_marlin_gemm calls
  (moe_block_size), as it must. Nothing else changes. Attribution: only T=128 and T=256 shapes
  (load=verify_16 and verify_32) can move; every other shape keeps its current block size and is
  the control.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "T=256: block 64 -> 16; the hot expert gains m-block parallelism and cold experts stop paying 64-row padding"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "T=128: block 48 -> 16, same mechanism"
  - shapes: "*load=decode_1*"
    direction: flat
    why: keeps block 8; control
  - shapes: "*load=decode_32*"
    direction: flat
    why: keeps block 16; control
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-m-cap-16

## Change

Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9, which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as chosen and clamps the T=128/256 picks to 16. Mechanism: routing is skewed (a few experts take most tokens), so with block 48/64 the many cold experts holding 1-4 tokens each are padded to a 48/64-row MMA tile and the hot expert gets only 4-6 tiles to spread over 132 SMs; block 16 pads cold experts 3-4x less and gives the hot expert 3-4x more m-blocks for the same 25 MB weight stream. The same block_size_m goes to moe_align_block_size and both moe_wna16_marlin_gemm calls (moe_block_size), as it must. Nothing else changes. Attribution: only T=128 and T=256 shapes (load=verify_16 and verify_32) can move; every other shape keeps its current block size and is the control.

## Expected

- `*load=verify_32*` faster: T=256: block 64 -> 16; the hot expert gains m-block parallelism and cold experts stop paying 64-row padding
- `*load=verify_16*` faster: T=128: block 48 -> 16, same mechanism
- `*load=decode_1*` flat: keeps block 8; control
- `*load=decode_32*` flat: keeps block 16; control

## Risk

Risk: SGLang's 0.9 occupancy rule exists because at large per-expert M a bigger tile reuses each weight fragment across more rows from registers/smem; with 16-row tiles Marlin re-streams the hot expert's 25 MB weight once per m-block from L2 (an 80K-token-context H200 has 50 MB L2, so one expert's w13 fits, w2 fits), which could make verify_32 slower instead. Either sign is a clean attribution to the tile choice; if slower, queue the opposite (cap at 32) as an ab-tune. Correctness unaffected (same fp32 reduce, same math). If this lands, a follow-up should also try block 8 at T=32 (decode_32), which the 0.9 rule just misses.
