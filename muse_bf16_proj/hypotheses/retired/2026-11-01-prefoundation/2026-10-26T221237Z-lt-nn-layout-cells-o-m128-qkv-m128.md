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
review: hold
status: hold
review_at: 2026-10-28T145555Z
review_companions: null
hold_reason: "adversarial review: Hold: the o m=128 half is already disproved by kernel times, since the NN 128x64 plan ran 5.5 us in 37ed8722 against the TN plan's 5.4 us on today's board and t"
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

## Adversarial review

Reviewed 2026-10-28T145555Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold: the o m=128 half is already disproved by kernel times, since the NN 128x64 plan ran 5.5 us in 37ed8722 against the TN plan's 5.4 us on today's board and the board's o row is already 7.23 us, the same as the claimed 7.24 target. The qkv m=128 NN half is untried, but its best plausible gain is about 0.35% of headline. The hypothesis itself sizes the whole thing at +0.07-0.34%, far under the 1.24% bar, and its stacking partner (pooler-khalf-m128-fp32-partials-triton-castadd) is on hold.

### Justification
A. (i) o m=128: the evidence cited, 'judged 7.68 -> 7.24 us in 37ed8722', is run-to-run drift and not a layout effect. In that run SGLang's own o verify_16 row also moved from 7.68 to 7.36 us. At kernel level, the NN plan (nvjet_sm90_tst_128x64_64x8_1x2_h_bz_NNT) took 5.5 us. Master's current TN plan (..._TNT) takes 5.4 us on the 1.2794 board (3e3ffc5a), where the o row is 7.23 us at 1.013x SGLang. The NN copy picks the same 128x64 tile with the operand transposed, so nothing is left to win. The gap that remains (7.23 row vs 5.4 kernel vs 4.0 floor) is about 1.8 us of row-minus-kernel overhead, which no plan swap touches. (ii) qkv m=128: master already runs this cell on a cuBLASLt TN plan (nvjet 40x64 4x2 TNN, 15.7 us kernel, 16.07 us row, floor 8.4), and since 62da2d9e that plan is ranked and gated under a read flush. NN plans on a [8192, 2304] copy have not been tried here. The pooler NN analogy (-10% at m=128) is weak: the pooler went through torch.mm, where a layout change switches cuBLAS's heuristic family, whereas this cell already searches Lt heuristic plans directly. Even a 10% kernel cut is about 1.6 us on verify_16 (70.8 us summed row time). At about 0.235%/us that is about 0.37% of headline, and a 3% gate win is about 0.1%. Staleness: the change is written against blob 2bcfe7d4. Master is now a36f3392, which already has the flush_mode='read' keyword (CELL_FLUSH_MODES), the read-flush ranking and gate in _lt_prepare, the bulk12 cells and the khalf/router-PDL landings. So the change paragraph's 'new keyword flush_mode=read' already exists, and its 'keep cell B's TN choice exactly as today' now means the read-flush plan, not the plan the hypothesis assumed. The `prior` field is honest about bf16-verify-per-cell-cublaslt-plan-selector (in Do-not-try, neutral), but sub-change (i) repeats that prior's o NN mechanism unchanged, and the prior's own kernel data refutes it. Weakest link: the o 'gain' is entirely drift. With it removed, the hypothesis is a single sub-1% qkv probe that needs a C++ Lt module rebuild and 54 MB of extra weight copies per state.

### Confounders on the current kernel
- phasec-pc-lt-qkv128-readflush (landed on master 62da2d9e, not in churn's landed list): it already re-ranked and re-gated the qkv m=128 TN Lt plan under the read flush, so the 'today's TN plan' this A/B compares against is the newly picked one. Any qkv NN win or loss is measured against a plan picked by the same machinery and inside the same row noise.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (landed): it introduced cell B (_LT_CPP / MuseLtDesc, TN only). This change rebuilds that module under a new LT_EXT_NAME, so a build or plan regression in cell B would mix with the NN effect.
- Run-level drift on verify_16: SGLang's o m=128 baseline varies by 4% between runs (7.33-7.68 us), larger than the ~0.1-0.4 us effects predicted for either cell.

### Interference and A/B needs
No companion would rescue this. The effects predicted per cell (o ~0 us given kernel parity; qkv 0-1.6 us) sit inside the per-row and per-run noise. An ab-revert or control run would double the cost of a hypothesis whose best plausible headline is about 0.4%. If the qkv NN idea is ever worth checking, it should be a prepare-time-only measurement: rank NN against TN Lt plans under the read flush and log the result, with no routing change. If it shows a 10% or larger kernel cut, it can be folded into a larger verify_16 stack. It does not justify a judge check alone.

### Open questions
- Does cuBLASLt's heuristic return any qkv m=128 NN plan with a different tile family (not 40x64 4x2) on the H200? If NN only yields the same tiles transposed, sub-change (ii) has no mechanism either.
- Is the ~1.8 us row-minus-kernel overhead on o m=128 (7.23 row vs 5.4 kernel) addressable at all, for example by a PDL chain or launch hiding? That gap, not the GEMM plan, separates o from its 4.0 us floor.

