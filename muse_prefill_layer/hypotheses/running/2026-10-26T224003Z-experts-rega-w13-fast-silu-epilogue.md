---
problem: muse_prefill_layer
slug: experts-rega-w13-fast-silu-epilogue
created: 2026-10-26T224003Z
base: master
area: silu-mul (rega w13 epilogue)
change: >
  Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32
  accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module
  is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a
  full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread
  evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an
  estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's
  source by one asserted edit (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns
  hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in
  pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is
  -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class
  (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings:
  finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference
  (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2
  graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99
  in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs
  difference, PATH CHOSEN.
expected:
  - shapes: "*"
    direction: faster
    why: "w13 is two thirds of rega in every class; cutting most of the silu epilogue is ~2-3% of w13: sliding turn_1 ~-6 us, sliding turn_4 ~-15 to -20 us, sliding first_turn ~-70 to -110 us, ~0.7-1% of each shape"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-triton-w13-fused-silu-epilogue: that added silu to the old Triton engine's w13 epilogue (more epilogue work; lost 1-3% at first_turn); rega already fuses silu, and this only swaps the accurate expf and IEEE division for __expf and __frcp_rn"
review: proceed
status: running
review_at: 2026-10-27T112554Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T112554Z
base_sha: c984c14ea32ac08522ee284ee7e6b46826ad0312
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue
worktree: .worktrees/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue
mode: null
restack_of: null
stale_base: null
---

# experts-rega-w13-fast-silu-epilogue

## Change

Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's source by one asserted edit (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings: finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2 graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99 in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs difference, PATH CHOSEN.

## Expected

- `*` faster: w13 is two thirds of rega in every class; cutting most of the silu epilogue is ~2-3% of w13: sliding turn_1 ~-6 us, sliding turn_4 ~-15 to -20 us, sliding first_turn ~-70 to -110 us, ~0.7-1% of each shape

## Risk

Not bit-exact (rare 1-ulp bf16 differences in hs), bounded by the gate against master and the fp64 reference. Size ~+0.5-0.8% headline: below the judge margin alone, so it lands through experts-rega-rotate-finebucket-fastsilu-combo or experts-rega-rotation-fastsilu-combo.

## Adversarial review

Reviewed 2026-10-27T112554Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with sharp notes. The mechanism is plausible and rule-clean, and the predecessor's review asked for this run first and alone. Its size is overstated ~3-10x: per-item arithmetic gives at most ~1% of w13 (~0.1-0.3% of a shape), so a pair gate at <= 0.99 will very likely keep every class OFF. Gate on w13 alone, use a genuinely approximate divide, and read the result as 'does the silu epilogue cost anything at all'.

### Justification
silu_mul runs in the serialized consumer epilogue of rega_kernel<true,*> (NB/4 silus per thread per item, no fast math). An item is ~130K clk at ft (64 stages); the accurate silu costs ~1.2-1.8K issue slots per item (~1-1.4%), and both forms have 2 MUFU ops (EX2, RCP), so the fast form saves only FFMA/branch overhead: ~0.3-0.8% of w13 at ft, less at turn_1. Expected scored deltas: ft ~-10 to -30 us, turn_4 ~-2 to -6 us, turn_1 ~-1 to -2 us. __frcp_rn is not a fast reciprocal (rcp.rn.f32); use __fdividef(hg, 1.f + __expf(-hg)) * hu or rcp.approx.ftz. Closer unlisted prior: experts-rega-producer-scale-expand-all-t (6be2a8d3), a consumer-instruction diet that gated OFF everywhere; per-shape rows cannot read a <=1% effect. Landing route via the rotation combos and fine-N-high-plus-fast-silu is gone (held/dead).

### Confounders on the current kernel
- combo-b low class: muse_prefill_rega_lo has its own silu_mul copy; fsilu must be OFF at low unless rega_low_variant == 0, the variant printed, and the gate must time the kernel that would be dispatched.
- combo-a normgather: gate and replay-check on the same prenorm dispatch as the judge path.
- experts-turn-rega-s4-*: silu_mul lives in _REGA_HEAD, hashed by REGA_V2_SHA256; edit only a derived copy; assert the target occurs exactly once in _rega_src().
- align-two-phase-all-t-rega: part of the ~143-167 s of prepare before this phase.

### Interference and A/B needs
No companion. Make the single result mean something: (a) gate on w13 ALONE (one-launch graphs): w13 <= 0.99 in both rounds and the pair <= 1.00; (b) per LESSONS 2026-10-27, time each class on both cold-L2 and warm-L2 back-to-back replays and on a judge-like routing whose master w13 time brackets the judged row (turn_1 ~255-293 us, turn_4 ~707-763 us, ft ~3400-3850 us), choosing the pop alpha per class or requiring both pop and random; ON only if every condition passes; (c) read the scored rega_kernel<true,*> rows and the w13/w2 ratio vs 73b40cd1 and d00770b6; a near-1.00 gate everywhere is the expected null and closes the silu line.

### Open questions
- Use __fdividef or rcp.approx.ftz, not __frcp_rn.
- If the build stays under ~45 s, add a print-only ceiling variant (silu_mul returns hg*hu, never dispatched) to bound any silu speedup.
- Canary the fast vs accurate form on a synthetic hg sweep {-100, -88.7, -87, -20, 0, 20, 100}.
- Landing route if it gates ON: its own result or a re-mint.
- Reconcile FINDINGS' queued rotation+fastsilu combos (held).

