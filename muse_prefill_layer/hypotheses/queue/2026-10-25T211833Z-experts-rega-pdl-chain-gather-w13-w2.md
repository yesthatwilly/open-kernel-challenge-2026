---
problem: muse_prefill_layer
slug: experts-rega-pdl-chain-gather-w13-w2
created: 2026-10-25T211833Z
base: master
area: "launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)"
change: >
  DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no
  programmatic dependent launch (griddepcontrol appears 0 times in submission.py). Port
  muse_moe_w4a16's helpers (submission.py ~line 1029: pdl_wait() = griddepcontrol.wait,
  pdl_launch() = griddepcontrol.launch_dependents; host side cudaLaunchKernelEx with
  cudaLaunchAttributeProgrammaticStreamSerialization = 1, ~line 1524) into the rega module.
  rega_gemm gains an int pdl argument; when 1 it launches with the attribute (graph capture keeps
  the programmatic edge, as moe's PDL chain does on the judge). In rega_kernel: everything that
  reads only align outputs (sorted_ids, block_experts, n_padded, chunk tables in smem: produced
  two or more grids back, complete before the previous grid launched) and weights may run BEFORE
  pdl_wait: barrier init, tensormap prefetch, the table decode, and the producer's first STAGES
  weight bulk copies (bulk_g2s of wsrc for the first items; weight addresses come from decode_item
  over align tables). pdl_wait() is called by the producer before its FIRST activation tma_2d and
  by every consumer warp before its first epilogue store, so no read of xs/hs and no write of out
  happens before the previous grid's memory is visible. pdl_launch() is issued by each CTA's
  producer after it has issued its last item's last stage. Apply pdl=1 to w13 (primary:
  _norm_gather_kernel or ext.rega_gather, launched normally; implicit trigger at its exit) and to
  w2 (primary: w13, explicit trigger). Do not touch fused_post_expert_norm_weighted_sum (SGLang
  Triton, launched normally after w2). All three rega classes. GATE per class in _rega_setup:
  exact-compare pdl=1 vs pdl=0 (bit-identical expected), two-state replay check, cold-L2
  interleaved graph timing of the whole _experts_rega call; ON at <= 0.99x (a small, low-risk
  win); state['rega_pdl'][cls] plain int. Print per class both medians and PATH CHOSEN.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "Persistent rega grid with a +-1-item tail at ~16-19 items per CTA, plus two launch gaps and two prologues (barrier init, table decode, first weight stages) on 756.7 / 1150.3 us shapes: ~10-25 us per shape, ~2-3% of the turn_1 load."
  - shapes: "*load=turn_4*"
    direction: faster
    why: same ~10-25 us on 2197.8 / 3744.8 us shapes, < 1%.
  - shapes: "*load=first_turn*"
    direction: flat
    why: "~15 us on 11-18.6 ms shapes, ~0.1%. Headline ~+0.8-1.0%: below the 1.24% margin, so this must stack (with mc2 or the low-class full slab)."
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
prior: none
review: required
status: queued
---

# experts-rega-pdl-chain-gather-w13-w2

## Change

DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no programmatic dependent launch (griddepcontrol appears 0 times in submission.py). Port muse_moe_w4a16's helpers (submission.py ~line 1029: pdl_wait() = griddepcontrol.wait, pdl_launch() = griddepcontrol.launch_dependents; host side cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization = 1, ~line 1524) into the rega module. rega_gemm gains an int pdl argument; when 1 it launches with the attribute (graph capture keeps the programmatic edge, as moe's PDL chain does on the judge). In rega_kernel: everything that reads only align outputs (sorted_ids, block_experts, n_padded, chunk tables in smem: produced two or more grids back, complete before the previous grid launched) and weights may run BEFORE pdl_wait: barrier init, tensormap prefetch, the table decode, and the producer's first STAGES weight bulk copies (bulk_g2s of wsrc for the first items; weight addresses come from decode_item over align tables). pdl_wait() is called by the producer before its FIRST activation tma_2d and by every consumer warp before its first epilogue store, so no read of xs/hs and no write of out happens before the previous grid's memory is visible. pdl_launch() is issued by each CTA's producer after it has issued its last item's last stage. Apply pdl=1 to w13 (primary: _norm_gather_kernel or ext.rega_gather, launched normally; implicit trigger at its exit) and to w2 (primary: w13, explicit trigger). Do not touch fused_post_expert_norm_weighted_sum (SGLang Triton, launched normally after w2). All three rega classes. GATE per class in _rega_setup: exact-compare pdl=1 vs pdl=0 (bit-identical expected), two-state replay check, cold-L2 interleaved graph timing of the whole _experts_rega call; ON at <= 0.99x (a small, low-risk win); state['rega_pdl'][cls] plain int. Print per class both medians and PATH CHOSEN.

## Expected

- `*load=turn_1*` faster: Persistent rega grid with a +-1-item tail at ~16-19 items per CTA, plus two launch gaps and two prologues (barrier init, table decode, first weight stages) on 756.7 / 1150.3 us shapes: ~10-25 us per shape, ~2-3% of the turn_1 load.
- `*load=turn_4*` faster: same ~10-25 us on 2197.8 / 3744.8 us shapes, < 1%.
- `*load=first_turn*` flat: ~15 us on 11-18.6 ms shapes, ~0.1%. Headline ~+0.8-1.0%: below the 1.24% margin, so this must stack (with mc2 or the low-class full slab).

## Risk

Prior art outside prefill: muse_moe_w4a16 combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 ran a PDL w13->w2 chain on the judge and front-end/epilogue wins did not interfere (LESSONS 183); attention combo-b1-chain-c320-fused-pre-pdl showed launch-gap wins compound with kernel wins. A read of xs/hs or a write of out before griddepcontrol.wait is a silent race that random routings may not expose: the exact-compare must run under graph replay with pdl=1. First measurement is an okc check.
