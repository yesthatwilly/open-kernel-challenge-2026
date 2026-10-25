---
problem: muse_decode_layer
slug: stack-route-topk-pdl-unpool-nf1
created: 2026-10-25T035400Z
base: master
area: routing top-k (kernel R) + post-MoE unpooler (T<=8)
change: >
  Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim
  restacks dl-route-warp-topk-pdl onto master from its patch (depends_on). (A)
  dl-route-warp-topk-pdl as is: per-warp top-8 in dl_route_norm_align_kernel (no block barriers),
  PDL between P (dl_pooler_router_kernel launch_dependents) and R (cudaLaunchKernelEx, norm_w/bias
  loaded before griddepcontrol.wait), binary-search pad loop. Run b471a950: R 7.01 -> 5.79 us at
  T=1, sliding decode_1 -3.4 us and full decode_1 -1.8 vs BEST. (B) From
  boundaries-unpooler-pdl-small-t
  (muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.patch), take ONLY
  dl_unpool_kernel at NF=1 (T<=8: rows 0..1023 = UNPOOL_SLICE, 16 tiles x SPLIT 8 = 128 CTAs, zero
  columns 1024..8191 written by the CTAs after griddepcontrol.wait, PDL launch, weight ring before
  the wait), replacing _gemm(expert_output, unpooler[:1024]) + F.pad; it ran 6.5 vs 9.2 us. Do NOT
  take dl_boundary_kernel: it lost its own timing (3.91/4.09/4.20 vs SGLang
  FusedPostNormResidualPreNorm 2.79/2.92/3.07 us) and was shipped anyway; both boundaries stay on
  master SGLang modules, and T=9..32 (NF=2/4) keep master _gemm(expert_output, unpooler). Since
  the unpool kernel no longer feeds a PDL boundary, it launches with PDL after Marlin/combine and
  the following SGLang boundary launches normally. THE TWO EDIT THE SAME _DL_CUDA TU (dense-chain
  module) and both rename the module, so the restack patch will likely conflict or the unpooler
  half must be ported by hand: the implementer hand-merges (B) into the restacked (A), uses one
  new module name, and keeps a single state flag per piece (state["dl"] route as in A; a new
  state["dlu1_ok"] for the NF=1 unpool). Self-check: the existing _dl_self_check cases (routing,
  near-tie, no-local-slot, align, Marlin y, graph replay) plus the unpool cases at T=1,4,8
  (ffn_output vs master unpooler+pad, rel L2 <= 2e-3, zero columns exactly 0) and a captured graph
  replay of P -> R -> experts -> unpool vs eager; failure turns off only the failing piece. Print
  PATH CHOSEN for route and unpool per T, and CUDA-event times of R (pdl 0/1) and unpool NF=1 vs
  master. T>32 master byte for byte. First measurement is an okc check.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: R -1.1 us (prepare) + unpool NF=1 -2.7 us per T=1 layer; each alone measured -2.4 to -3.6 us vs control at decode_1; expect about -4 to -6 us per decode_1 shape, ~+0.008-0.010 headline
  - shapes: "*load=verify_1 *"
    direction: faster
    why: T=8 uses both R (-1.0 to -1.5 us) and unpool NF=1; err 2.8-3.3 us so may not resolve
  - shapes: "*load=decode_16 * *load=decode_32 * *load=verify_16 * *load=verify_32 *"
    direction: flat
    why: R gain ~1 us inside 0.5-8 us err; unpooler master at NF=2/4
risk: medium
depends_on: dl-route-warp-topk-pdl
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# stack-route-topk-pdl-unpool-nf1

## Change

Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim restacks dl-route-warp-topk-pdl onto master from its patch (depends_on). (A) dl-route-warp-topk-pdl as is: per-warp top-8 in dl_route_norm_align_kernel (no block barriers), PDL between P (dl_pooler_router_kernel launch_dependents) and R (cudaLaunchKernelEx, norm_w/bias loaded before griddepcontrol.wait), binary-search pad loop. Run b471a950: R 7.01 -> 5.79 us at T=1, sliding decode_1 -3.4 us and full decode_1 -1.8 vs BEST. (B) From boundaries-unpooler-pdl-small-t (muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.patch), take ONLY dl_unpool_kernel at NF=1 (T<=8: rows 0..1023 = UNPOOL_SLICE, 16 tiles x SPLIT 8 = 128 CTAs, zero columns 1024..8191 written by the CTAs after griddepcontrol.wait, PDL launch, weight ring before the wait), replacing _gemm(expert_output, unpooler[:1024]) + F.pad; it ran 6.5 vs 9.2 us. Do NOT take dl_boundary_kernel: it lost its own timing (3.91/4.09/4.20 vs SGLang FusedPostNormResidualPreNorm 2.79/2.92/3.07 us) and was shipped anyway; both boundaries stay on master SGLang modules, and T=9..32 (NF=2/4) keep master _gemm(expert_output, unpooler). Since the unpool kernel no longer feeds a PDL boundary, it launches with PDL after Marlin/combine and the following SGLang boundary launches normally. THE TWO EDIT THE SAME _DL_CUDA TU (dense-chain module) and both rename the module, so the restack patch will likely conflict or the unpooler half must be ported by hand: the implementer hand-merges (B) into the restacked (A), uses one new module name, and keeps a single state flag per piece (state["dl"] route as in A; a new state["dlu1_ok"] for the NF=1 unpool). Self-check: the existing _dl_self_check cases (routing, near-tie, no-local-slot, align, Marlin y, graph replay) plus the unpool cases at T=1,4,8 (ffn_output vs master unpooler+pad, rel L2 <= 2e-3, zero columns exactly 0) and a captured graph replay of P -> R -> experts -> unpool vs eager; failure turns off only the failing piece. Print PATH CHOSEN for route and unpool per T, and CUDA-event times of R (pdl 0/1) and unpool NF=1 vs master. T>32 master byte for byte. First measurement is an okc check.

## Expected

- `*load=decode_1 *` faster: R -1.1 us (prepare) + unpool NF=1 -2.7 us per T=1 layer; each alone measured -2.4 to -3.6 us vs control at decode_1; expect about -4 to -6 us per decode_1 shape, ~+0.008-0.010 headline
- `*load=verify_1 *` faster: T=8 uses both R (-1.0 to -1.5 us) and unpool NF=1; err 2.8-3.3 us so may not resolve
- `*load=decode_16 * *load=decode_32 * *load=verify_16 * *load=verify_32 *` flat: R gain ~1 us inside 0.5-8 us err; unpooler master at NF=2/4

## Risk

Small gain near one check resolution; read the decode_1 rows (err ~0.6 us) and the R / dl_unpool_kernel profile rows, not the headline. A submit-grade A/B would be needed to land it.
