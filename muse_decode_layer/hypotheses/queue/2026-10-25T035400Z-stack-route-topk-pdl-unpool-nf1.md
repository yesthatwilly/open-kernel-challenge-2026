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
review: proceed
status: queued
review_at: 2026-10-25T040253Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** The stack is justified: two measured T=1 wins on disjoint kernels (R and the unpooler slice), each -2.4 to -3.6 us against the control, adding up to about -4 to -6 us on decode_1 where err is ~0.6 us. The merge is easier than the brief fears: boundaries-unpooler-pdl-small-t is a separate TU (_DLB_CUDA / _DLB_EXT), and its patch applies cleanly on top of the route patch.

### Justification
Both source runs used the current master blob (base_blob 4da9467 = master:muse_decode_layer/submission.py), so their deltas are against today's BEST. Route: R 7.01 -> 5.79 us in prepare, decode_1 -3.4/-1.8 us vs BEST. Unpool NF=1: 6.5 vs 9.2 us. That gain is mostly intrinsic (it fuses the slice GEMM and the F.pad zero fill), and its predecessor (experts/combine) is unchanged. The only context change is the successor: the SGLang FFN boundary now launches normally instead of the PDL'd dl_boundary_kernel, so a ~0.5-1 us overlap may be lost. That is the weakest link, but it cannot flip the sign. Merge plan, verified in a scratch repo against the master blob: dl-route-warp-topk-pdl.patch applies clean, and boundaries-unpooler-pdl-small-t.patch then applies clean on top of it. Its CUDA lives in a second load_inline TU (_DLB_CPP/_DLB_CUDA, 'a second, separate TU'), so there is no _DL_CUDA conflict and no hand-port. The builder then removes the boundary half in kernel_fn. Expected shapes (decode_1 faster, verify_1 maybe, T>=9 flat) match the gates: NF=1 is T<=8, and the route change spans T<=32 but is only ~1 us there, inside err.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (master): both pieces live inside its PDL chain (P -> R -> experts). R's PDL wait depends on P issuing launch_dependents, which the route patch adds. If the chain were absent, neither gain would exist, but it is landed and unchanged.
- stack-align-par-sidestream-sliding1 (master BEST): its side-stream dispatch is T>32 only, so it does not overlap at T<=8.
- post-expert-norm-triton-small-t (master): the Triton combine is the unpool kernel's PDL predecessor. It never triggers launch_dependents, so the unpool's pre-wait weight-ring prefetch overlaps only the combine's tail. That was the same in the boundaries run, so no new confound.
- dense-chain-stage2-attn-side-pdl-t1-gate (queued sibling): it also touches T=1 and edits _DL_CUDA (its patch does NOT apply on top of the route patch; it conflicts at the dl_pooler_router hunk ~line 295). Run both against master independently; whichever lands second hand-merges the _DL_CUDA TU.

### Interference and A/B needs
No companion. The two pieces are in separate kernels whose profile rows (dl_route_norm_align_kernel, and dl_unpool_kernel vs nvjet slice + pad) attribute each part in the same check. The decode_16/32 and verify_16/32 rows (T>=9, master unpooler) act as the drift control. Per-piece state flags let the self-check turn either off. A submit-grade A/B is the landing step, as the brief already says, not a companion.

### Open questions
- BUILDER: do not hand-port. git apply dl-route-warp-topk-pdl.patch, then git apply boundaries-unpooler-pdl-small-t.patch (both clean on master blob 4da9467). Then in kernel_fn always use state['attn_boundary'] and the SGLang FFN boundary (drop the use_dlb early return and _dlb_attn/_dlb_ffn calls). Call _dlb_unpool only when tokens <= SLICED_UNPOOL_MAX_TOKENS and dlb['unpool'][1] is True; NF=2/4 stay on master _gemm. Drop the dl_boundary build/self-check/timing, or leave them dead. Rename both modules.
- Make the NF=1 unpool selector compare the printed timings (custom vs master slice+pad) and assert the winner (LESSONS 163), and do the same for the R pdl 0/1 choice.
- Read decode_1 rows against the earlier control run (p-wide-tile: 126.5 / 144.1) as well as BEST, because the 03:41Z batch drifted about 1 us at decode_1.

