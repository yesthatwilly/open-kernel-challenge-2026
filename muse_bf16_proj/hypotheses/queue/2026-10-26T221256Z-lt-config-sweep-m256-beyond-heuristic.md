---
problem: muse_bf16_proj
slug: lt-config-sweep-m256-beyond-heuristic
created: 2026-10-26T221256Z
base: master
area: bf16 GEMM dispatch (m=256, cuBLASLt configs beyond the heuristic)
change: >
  Built on master blob 2bcfe7d4. verify_32 is the weakest load (1.091x, 93.55 us per step in
  3834ceec) and every row in it runs cuBLAS at SGLang parity: qkv_gate m=256 19.84 us (nvjet 96x64
  2x4 TNN, only 96 CTAs on 132 SMs), o 9.44 (128x128 2x1 TNT), pooler 29.95 (NN 128x64 1x2 on
  pooler_t), unpooler 24.82 (128x128 2x1 TNT). Every Lt plan tried so far came from
  cublasLtMatmulAlgoGetHeuristic's top 8-16 results (all algo 66). Add to master's _LT_CPP one
  function enumerate(n, k, m, transa, ws_cap, max_out): cublasLtMatmulAlgoGetIds for bf16 A/B/C/D
  with fp32 compute and scale; for each id, cublasLtMatmulAlgoInit, then read
  CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_ALGO_CAP_STAGES_IDS and
  CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT; for every (tile, stages, swizzle 0 or 1 when supported)
  set those config attributes with SPLITK_NUM 1 and REDUCTION_SCHEME NONE and keep the algo if
  cublasLtMatmulAlgoCheck succeeds with workspace <= ws_cap; return the algo rows plus an info
  table shaped like candidates(); add the transa flag of
  done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.patch (TN on W as
  stored, NN on a [k, n] copy); new LT_EXT_NAME (C++ only). Four m=256 cells with their own flags
  lt256_<layer>_ok, plan rows, transa and per-state workspaces: qkv_gate and o on TN; pooler on NN
  (pooler_t) and TN; unpooler on TN and NN (unpooler_t, already in state). Per cell: rank up to
  512 enumerated configs by cold-L2 graph replay (15 reps, judge-like read flush), _cell_check the
  best 3, gate the best passing one against today's route (kernel_fn with the flag off) with two
  interleaved _cell_gate rounds under the read flush (new keyword flush_mode='read':
  flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged), master/cand >= 1.03
  in both and cand IQR <= 1.5x master's, then assert exactly one launched kernel with no splitK /
  reduce in its name; kernel_fn branch per cell at cfg m == 256 before today's route. State B
  inherits flags and plan rows; AB_CASES gains one m=256 case per layer mapped to its flag. Print
  the enumerated count, the top 5 and one PATH CHOSEN line per cell compactly at the end of
  prepare (16 KB tail). m=128 and below untouched.
expected:
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: "a non-heuristic tile can fill more than 96 CTAs without a split-K reduce node: 19.84 -> ~18-19 us if one wins the gate"
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: "same FLOPs and bytes as the unpooler row (24.82 us) but 29.95 us: a better tile on NN or TN could close part of that 5 us gap"
  - shapes: layer=o load=verify_32 m=256
    direction: faster
    why: "9.44 us vs a 4.5 us floor on the heuristic's 128x128 tile"
  - shapes: layer=unpooler load=verify_32 m=256
    direction: faster
    why: "already ~700 TFLOP/s; small or no gain expected, gate keeps today's route otherwise"
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "bf16-cublaslt-algo-pick-in-prepare-m128-m256: it chose among the heuristic's own candidates with one non-interleaved timing and flipped in the judge; this enumerates the non-heuristic tile / stage / swizzle configs of every algo id, m=256 only, and routes a config only after two interleaved read-flush rounds at >= 1.03"
review: required
status: queued
---

# lt-config-sweep-m256-beyond-heuristic

## Change

Built on master blob 2bcfe7d4. verify_32 is the weakest load (1.091x, 93.55 us per step in 3834ceec) and every row in it runs cuBLAS at SGLang parity: qkv_gate m=256 19.84 us (nvjet 96x64 2x4 TNN, only 96 CTAs on 132 SMs), o 9.44 (128x128 2x1 TNT), pooler 29.95 (NN 128x64 1x2 on pooler_t), unpooler 24.82 (128x128 2x1 TNT). Every Lt plan tried so far came from cublasLtMatmulAlgoGetHeuristic's top 8-16 results (all algo 66). Add to master's _LT_CPP one function enumerate(n, k, m, transa, ws_cap, max_out): cublasLtMatmulAlgoGetIds for bf16 A/B/C/D with fp32 compute and scale; for each id, cublasLtMatmulAlgoInit, then read CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_ALGO_CAP_STAGES_IDS and CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT; for every (tile, stages, swizzle 0 or 1 when supported) set those config attributes with SPLITK_NUM 1 and REDUCTION_SCHEME NONE and keep the algo if cublasLtMatmulAlgoCheck succeeds with workspace <= ws_cap; return the algo rows plus an info table shaped like candidates(); add the transa flag of done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.patch (TN on W as stored, NN on a [k, n] copy); new LT_EXT_NAME (C++ only). Four m=256 cells with their own flags lt256_<layer>_ok, plan rows, transa and per-state workspaces: qkv_gate and o on TN; pooler on NN (pooler_t) and TN; unpooler on TN and NN (unpooler_t, already in state). Per cell: rank up to 512 enumerated configs by cold-L2 graph replay (15 reps, judge-like read flush), _cell_check the best 3, gate the best passing one against today's route (kernel_fn with the flag off) with two interleaved _cell_gate rounds under the read flush (new keyword flush_mode='read': flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged), master/cand >= 1.03 in both and cand IQR <= 1.5x master's, then assert exactly one launched kernel with no splitK / reduce in its name; kernel_fn branch per cell at cfg m == 256 before today's route. State B inherits flags and plan rows; AB_CASES gains one m=256 case per layer mapped to its flag. Print the enumerated count, the top 5 and one PATH CHOSEN line per cell compactly at the end of prepare (16 KB tail). m=128 and below untouched.

## Expected

- `layer=qkv_gate load=verify_32 m=256` faster: a non-heuristic tile can fill more than 96 CTAs without a split-K reduce node: 19.84 -> ~18-19 us if one wins the gate
- `layer=pooler load=verify_32 m=256` faster: same FLOPs and bytes as the unpooler row (24.82 us) but 29.95 us: a better tile on NN or TN could close part of that 5 us gap
- `layer=o load=verify_32 m=256` faster: 9.44 us vs a 4.5 us floor on the heuristic's 128x128 tile
- `layer=unpooler load=verify_32 m=256` faster: already ~700 TFLOP/s; small or no gain expected, gate keeps today's route otherwise

## Risk

Medium, and the largest of this batch (about 45 minutes of C++ against the cuBLASLt algo-config API): dispatch it last. C++-only Lt rebuild, no nvcc; every cell keeps today's route unless a config wins two judge-like rounds by >= 3%, so a VOID run (the enumeration finds nothing beyond the heuristic for algo 66) costs only the check and answers that question. About 4 x 512 configs x ~4 ms adds ~8 s to prepare. Not a retry of the heuristic pickers: bf16-cublaslt-algo-pick-in-prepare-m128-m256 used a single non-interleaved timing, and the per-cell selector only saw heuristic results. Sizing: verify_32 is 0.178 %/us, so -2 to -4 us across the four rows is +0.4-0.7%, below the 1.24% one-run margin alone. Decode-portable to decode's verify T=256 GEMMs.
