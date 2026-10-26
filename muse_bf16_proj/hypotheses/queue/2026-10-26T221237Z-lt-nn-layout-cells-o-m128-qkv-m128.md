---
problem: muse_bf16_proj
slug: lt-nn-layout-cells-o-m128-qkv-m128
created: 2026-10-26T221237Z
base: master
area: o GEMM + qkv GEMM (m=128, cuBLASLt plan on an NN weight copy)
change: >
  Built on master blob 2bcfe7d4. Port the transa support of
  done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.patch (MuseLt2Desc /
  lt2_candidates / lt2_run) into master's _LT_CPP: MuseLtDesc(n, k, m, transa) with transa 1 =
  today's TN on W [n, k] as stored and transa 0 = NN on a [k, n] row-major copy (A layout rows n,
  cols k, ld n, op N); candidates(..., transa) and run(..., transa); new LT_EXT_NAME (C++ only, no
  nvcc). Two cells at m=128 (verify_16), each with its own flag, plan row, transa, weight name and
  per-state workspace. (i) o m=128 (row 7.56 us, nvjet 128x64 TNT 5.3 us, 0.98x SGLang in
  3834ceec): new lossless copy state['o_t'] = o.t().contiguous() [1024, 8192]; candidates =
  split-K-1 heuristic plans (workspace caps 0 and 1 MB, as cell B) on TN and on NN; canary +
  _cell_check each, rank by cold-L2 graph replay, gate the best against master's
  _bf16_gemm_dispatch_impl; flag lt_o128_ok; kernel_fn branch for o at cfg m == 128 before the
  dispatch fallback. (ii) qkv m=128 (cell B, row 15.72 us on its TN plan nvjet 40x64 4x2): keep
  cell B's TN choice exactly as today, then rank NN split-K-1 plans on a new copy
  state['qkv_gate_t'] [8192, 2304] and A/B the best NN plan against the chosen TN plan through a
  flag lt_qkv128_nn read by cell B's kernel_fn branch (on: NN plan on qkv_gate_t; off: today's TN
  plan). Gates: two interleaved rounds of _cell_gate with a judge-like read flush (new keyword
  flush_mode='read': flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged
  for every other caller; both weights fit in L2), master/cand >= 1.03 in both rounds and cand IQR
  <= 1.5x master's, then one torch.profiler call asserting exactly one kernel with no splitK /
  reduce in its name. State B inherits the flags, plan rows and transa, and builds o_t /
  qkv_gate_t only if its cell is on. AB_CASES gains (o, 8192, 1024, 128) mapped to lt_o128_ok; qkv
  m=128 maps to lt_qkv128_nn when it is on (a miss falls back to today's TN plan), else to LT_FLAG
  as today. One PATH CHOSEN line per cell with both rounds. m=256 and every other cell untouched.
expected:
  - shapes: layer=o load=verify_16 m=128
    direction: faster
    why: "the NN 128x64 plan on o_t gated 1.048 / 1.054 and judged 7.68 -> 7.24 us in 37ed8722: 7.56 -> ~7.2 us"
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "NN plans join cell B (never tried for qkv; NN cut pooler m=128 by 10% and unpooler m=128 by 2-4%); routed only at >= 3% over today's TN plan, else unchanged"
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: flat
    why: m=256 untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole: its o m=128 NN plan gated 1.048/1.054 and judged 7.68 -> 7.24 us, but the run was neutral because its qkv m=256 split-K 4 plan lost in the judge; this keeps only o m=128, adds NN plans to cell B (qkv m=128, never tried), drops every split-K plan and gates under a judge-like read flush on the bulk12 master"
review: required
status: queued
---

# lt-nn-layout-cells-o-m128-qkv-m128

## Change

Built on master blob 2bcfe7d4. Port the transa support of done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.patch (MuseLt2Desc / lt2_candidates / lt2_run) into master's _LT_CPP: MuseLtDesc(n, k, m, transa) with transa 1 = today's TN on W [n, k] as stored and transa 0 = NN on a [k, n] row-major copy (A layout rows n, cols k, ld n, op N); candidates(..., transa) and run(..., transa); new LT_EXT_NAME (C++ only, no nvcc). Two cells at m=128 (verify_16), each with its own flag, plan row, transa, weight name and per-state workspace. (i) o m=128 (row 7.56 us, nvjet 128x64 TNT 5.3 us, 0.98x SGLang in 3834ceec): new lossless copy state['o_t'] = o.t().contiguous() [1024, 8192]; candidates = split-K-1 heuristic plans (workspace caps 0 and 1 MB, as cell B) on TN and on NN; canary + _cell_check each, rank by cold-L2 graph replay, gate the best against master's _bf16_gemm_dispatch_impl; flag lt_o128_ok; kernel_fn branch for o at cfg m == 128 before the dispatch fallback. (ii) qkv m=128 (cell B, row 15.72 us on its TN plan nvjet 40x64 4x2): keep cell B's TN choice exactly as today, then rank NN split-K-1 plans on a new copy state['qkv_gate_t'] [8192, 2304] and A/B the best NN plan against the chosen TN plan through a flag lt_qkv128_nn read by cell B's kernel_fn branch (on: NN plan on qkv_gate_t; off: today's TN plan). Gates: two interleaved rounds of _cell_gate with a judge-like read flush (new keyword flush_mode='read': flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged for every other caller; both weights fit in L2), master/cand >= 1.03 in both rounds and cand IQR <= 1.5x master's, then one torch.profiler call asserting exactly one kernel with no splitK / reduce in its name. State B inherits the flags, plan rows and transa, and builds o_t / qkv_gate_t only if its cell is on. AB_CASES gains (o, 8192, 1024, 128) mapped to lt_o128_ok; qkv m=128 maps to lt_qkv128_nn when it is on (a miss falls back to today's TN plan), else to LT_FLAG as today. One PATH CHOSEN line per cell with both rounds. m=256 and every other cell untouched.

## Expected

- `layer=o load=verify_16 m=128` faster: the NN 128x64 plan on o_t gated 1.048 / 1.054 and judged 7.68 -> 7.24 us in 37ed8722: 7.56 -> ~7.2 us
- `layer=qkv_gate load=verify_16 m=128` faster: NN plans join cell B (never tried for qkv; NN cut pooler m=128 by 10% and unpooler m=128 by 2-4%); routed only at >= 3% over today's TN plan, else unchanged
- `layer=qkv_gate load=verify_32 m=256` flat: m=256 untouched

## Risk

Medium: a C++-only Lt module rebuild (no nvcc) and +54 MB of lossless weight copies per state. The o cell repeats a row gain that was measured once but sits inside per-row noise (-0.44 us); qkv NN is untried. No per-call preferred_blas_library toggles (Do-not-try): plans live in state, descriptors per call as cell B does. The qkv m=256 split-K 4 plan that lost in the judge is not carried over. Sizing: -0.3 to -1.5 us on verify_16 at 0.228 %/us is +0.07-0.34%, below the 1.24% one-run margin, so it lands stacked with pooler-khalf-m128-fp32-partials-triton-castadd. Decode-portable to decode's verify T=128 o / qkv GEMMs.
