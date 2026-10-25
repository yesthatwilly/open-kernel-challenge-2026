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
review: proceed
status: queued
review_at: 2026-10-25T212524Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T212524Z by an adversarial subagent (model opus).

**Verdict: proceed.** PDL on norm_gather -> rega w13 -> rega w2 is a sound, bit-identical, low-risk change with no prior in prefill. The stated mechanism overclaims, though: griddepcontrol.wait cannot hide the ±1-item tail. Only launch latency and the prologue overlap (barrier init, table decode, first weight stages), which realistically saves about 4-12 µs per shape, mostly at turn_1. Read the result from the per-class gate medians, not the headline or the per-kernel profile rows.

### Justification
Mechanism: plausible but smaller than claimed. rega_kernel is __launch_bounds__(THREADS,1) with near-full smem, so one CTA fits per SM. Early-launched w2 CTAs can only become resident on SMs where a w13 CTA has already exited, and they block at pdl_wait before any activation TMA or MMA. So w13's ±1-item tail is not recovered. What overlaps is (a) the graph kernel-to-kernel launch gap, about 1-3 µs per boundary, and (b) the prologue (mbar init, s_first/chunk table decode over nb blocks, tensormap prefetch, the first STAGES weight bulk copies, which are cold-L2 HBM latency because the judge flushes L2). That is about 2-6 µs per boundary times two boundaries, about 4-12 µs per shape. At turn_1 that is 0.5-1.5% of 756.7/1150.3 µs (the hypothesis says 10-25 µs, 2-3%). At turn_4 it is under 0.5%, and first_turn is flat. Expected directions are consistent with TASK weights; only the magnitude is optimistic. Correctness argument checks out against the code. sorted_ids, block_experts and n_padded come from align, which precedes the normally-launched gather, so they are complete before w13 can start. Weights and gs are static. The err magic and atomic writes before the wait are harmless. Every read of xs/hs and write of out/y sits behind pdl_wait, which also covers allocator-aliasing WAR hazards inside the captured graph. No deadlock: the rega grid is persistent and all CTAs are co-resident, so the explicit trigger fires only after every w13 CTA has issued or exited. Prior: 'none' is correct for prefill (grep finds no PDL in done/ or running/; DIAGNOSIS says griddepcontrol appears 0 times). The cited moe precedent (LESSONS 188, combo-toprow on PDL master) and attention combo-b1 (LESSONS 191) are real. It is not on any Do-not-try list. WEAKEST LINK: the gate. ON at <= 0.99x of the whole _experts_rega call (align + gather + w13 + w2 + post-norm) needs about 4.5-5 µs saved at turn_1, about 11 µs at turn_4 and about 55 µs at first_turn. turn_4 and first_turn will almost certainly gate OFF, so 'all three classes' is moot. turn_1 sits right at the threshold, which is exactly the single-round marginal-selector misroute the new LESSONS line warns about. Because the output is bit-identical and the change cannot plausibly cost much, a better gate is 'not slower' (ratio <= 1.00-1.01 over two interleaved cold-L2 rounds with an IQR bound), or an absolute-µs criterion. That keeps the small win on every class instead of a coin flip at 0.99.

### Confounders on the current kernel
- combo-a-master-normgather-naive-stack (landed): the w13 primary is the Triton _norm_gather_kernel with a persistent grid of SMs x resident x NG_MULTS (1 or 2), chosen per class. With an implicit trigger at its exit, w13 overlap with the gather's tail depends on that grid choice. In classes where the gather is OFF, the primary is ext.rega_gather (one CTA per slot, a very different tail). Per-class savings at the gather->w13 boundary differ by which gather won, so this is not a confound of the sign, only of the magnitude.
- w2-nonlocal-zero-store-off-first-turn (in stack-tonight-five-prefill, landed): when state['zero_nonlocal_w2'] is True, _experts_rega issues torch.zeros(y), a memset/fill node, between w13 and w2. w2's programmatic edge would then be against that node, not w13 (an invalid or ignored primary under capture). On the live rega path it is False (the ft class refuses otherwise), but the build must force pdl=0 for w2 whenever zero_nonlocal_w2 is True.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128 (landed engine): the prologue PDL hides is this kernel's table decode. It is cheap in the small class (s_be in smem, about 50-100 blocks at T=753) and heavier in the BIG class (block_experts read through __ldg over thousands of blocks), so the overlap is per class. That is a property of the engine, not an interaction that could flip the sign.
- combo-b-normgather-two-cta-low-regate (best's job; low variant gated OFF): if the low-class V64/V32 variant were ON, w13/w2 would run through state['rega_lo_ext'].rega_gemm_lo, which this change does not touch. Today variant=0, so the change does apply at turn_1. If a later restack turns V64 ON, PDL silently drops out of the low class.

### Interference and A/B needs
No companion is warranted. The change is bit-identical and self-gated. The prepare gate already times pdl=1 against pdl=0 in-process, interleaved under cold-L2 graph replay, and prints both medians per class, so that log line is the A/B. No landed hypothesis needs reverting to read it: the only real coupling (the zero_nonlocal_w2 memset node) is a correctness guard to build in, not something to measure. A control separating 'launch attribute only' from 'pre-wait prologue + weight prefetch' would split 4-12 µs into two pieces under okc noise and could not be read. Two reading rules make the single run interpretable. (1) Do NOT read the per-kernel profile rows: with PDL, rega_kernel<false,*> (and <true,*>, behind the gather) start earlier and spin in griddepcontrol.wait, so their reported durations GROW even when the shape gets faster. Read whole-shape µs and the gate medians. (2) The headline gain (about 0.3-0.8%) is below the 1.24% margin by the hypothesis's own admission, so score the run on the turn_1 gate ratio and the turn_1 shape rows, and carry it as a stack component (with mc2 / full-slab low) rather than judging it as a standalone submit.

### Open questions
- Gate threshold: should the ON criterion be 'not slower' (<= 1.00-1.01, two interleaved rounds, IQR bound), given the output is bit-identical and the expected saving (about 1% at turn_1, under 0.5% elsewhere) is at or below the proposed 0.99x? As written, turn_4 and ft will almost surely gate OFF, and turn_1 is a marginal single-threshold selector (the 2026-10-25 LESSONS misroute pattern).
- Producer restructure: issuing the first STAGES weight bulk_g2s before pdl_wait and that stage's activation tma_2d after it needs the expect_tx (WST + slab bytes) armed once per stage, before the weight copy, with the activation TMA completing the same transaction. Confirm the builder does not double-arm or under-count the full barrier.
- Does the Triton _norm_gather_kernel launch interact with the programmatic attribute on w13 under torch graph capture (the primary is a Triton launch, not cudaLaunchKernelEx)? The moe precedent chained two CUDA-extension kernels, not Triton -> CUDA. Check that capture records a programmatic edge and not a full edge (a full edge would make the gather->w13 half a silent no-op).
- Stacking with the queued mc2 / cluster line (experts-rega-mc2-*, rega_mc_kernel): the PDL attribute and the pdl_wait/pdl_launch placement have to be re-ported into the cluster kernel's cudaLaunchKernelEx (the cluster-dim attribute plus the programmatic attribute). Plan that port so a later mc2 win does not silently drop the PDL saving.

