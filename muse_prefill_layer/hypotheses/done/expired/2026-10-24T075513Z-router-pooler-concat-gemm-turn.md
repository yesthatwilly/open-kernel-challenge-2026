---
problem: muse_prefill_layer
slug: router-pooler-concat-gemm-turn
created: 2026-10-24T075513Z
base: master
area: router GEMM + pooler GEMM (one concatenated GEMM, no side stream)
change: >
  For tokens <= 4096 only, fold the router's hi/lo GEMMs into the pooler GEMM, since both read the
  same ffn_input [T, 8192]. prepare() builds w_cat = torch.cat([pooler, router_hi, router_lo]) as
  bf16 [4608, 8192] row-major (pooler rows unchanged BF16, hi/lo the same bf16 split as today);
  the separate pooler and router weights stay for the tokens > 4096 path. In kernel_fn on that
  gate, on the main stream with no side-stream fork: y = torch.mm(ffn_input, w_cat.t(),
  out_dtype=torch.float32) [T, 4608]; expert_input = y[:, :4096].to(torch.bfloat16) (the same
  round-to-nearest of the fp32 accumulator that the BF16 cuBLAS epilogue does, so the pooler
  output keeps BF16 as SGLang does); logits = y[:, 4096:4352] + y[:, 4352:4608] (fp32, hi + lo as
  today); topk_weights, topk_ids = state['routing'](None, logits, TOP_K, True); then the parity
  remap, pre-expert norm and experts. For tokens > 4096 (first_turn) the parity side-stream fork
  is kept unchanged. Mechanism: at M=753 the router pair runs on the side stream as split-K nvjet
  (33-40 us) + splitKreduce (~46 us) and is the critical path of the fork (moe-align-side-stream
  review: side ~80-90 us vs pooler ~40-50 us + norm 7 us); appended as 512 more output columns to
  the pooler it is ~12% more of a GEMM that already streams x, with N=4608 giving more tiles
  (better wave fill at M=753) and no split-K, no reduce, no fork/join. Extra cost: the fp32 output
  (T x 4608 x 4 B, 14 MB at T=753) and one cast/slice-add (~3-5 us at T=753, ~15-20 us at T=3012).
  This is an alternative engine to router-triton-hilo-splitk-turn for the same limiter; they do
  not stack.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "removes the ~80-90 us side branch for ~+6 us on the pooler GEMM plus ~4 us of cast/add: up to ~35-40 us of 1079/1435 us (2-3%)"
  - shapes: "*load=turn_4*"
    direction: flat
    why: router (132 us) ran in the pooler shadow; folding adds ~25 us of GEMM + ~20 us cast/add against the SM contention removed; near zero either way
  - shapes: "*load=first_turn*"
    direction: flat
    why: parity side-stream path kept above 4096 tokens (control)
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-24T124300Z
review_companions: null
hold_reason: "adversarial review: Rule-clean, but the size of the win rests on a router side branch of ~80-90 us that the current master profile does not show. At turn_1 sliding the router pair "
outcome: expired
verdict_detail: "hold expired after 33 h: adversarial review: Rule-clean, but the size of the win rests on a router side branch of ~80-90 us that the current mast"
expired_at: 2026-10-25T211329Z
---

# router-pooler-concat-gemm-turn

## Change

For tokens <= 4096 only, fold the router's hi/lo GEMMs into the pooler GEMM, since both read the same ffn_input [T, 8192]. prepare() builds w_cat = torch.cat([pooler, router_hi, router_lo]) as bf16 [4608, 8192] row-major (pooler rows unchanged BF16, hi/lo the same bf16 split as today); the separate pooler and router weights stay for the tokens > 4096 path. In kernel_fn on that gate, on the main stream with no side-stream fork: y = torch.mm(ffn_input, w_cat.t(), out_dtype=torch.float32) [T, 4608]; expert_input = y[:, :4096].to(torch.bfloat16) (the same round-to-nearest of the fp32 accumulator that the BF16 cuBLAS epilogue does, so the pooler output keeps BF16 as SGLang does); logits = y[:, 4096:4352] + y[:, 4352:4608] (fp32, hi + lo as today); topk_weights, topk_ids = state['routing'](None, logits, TOP_K, True); then the parity remap, pre-expert norm and experts. For tokens > 4096 (first_turn) the parity side-stream fork is kept unchanged. Mechanism: at M=753 the router pair runs on the side stream as split-K nvjet (33-40 us) + splitKreduce (~46 us) and is the critical path of the fork (moe-align-side-stream review: side ~80-90 us vs pooler ~40-50 us + norm 7 us); appended as 512 more output columns to the pooler it is ~12% more of a GEMM that already streams x, with N=4608 giving more tiles (better wave fill at M=753) and no split-K, no reduce, no fork/join. Extra cost: the fp32 output (T x 4608 x 4 B, 14 MB at T=753) and one cast/slice-add (~3-5 us at T=753, ~15-20 us at T=3012). This is an alternative engine to router-triton-hilo-splitk-turn for the same limiter; they do not stack.

## Expected

- `*load=turn_1*` faster: removes the ~80-90 us side branch for ~+6 us on the pooler GEMM plus ~4 us of cast/add: up to ~35-40 us of 1079/1435 us (2-3%)
- `*load=turn_4*` flat: router (132 us) ran in the pooler shadow; folding adds ~25 us of GEMM + ~20 us cast/add against the SM contention removed; near zero either way
- `*load=first_turn*` flat: parity side-stream path kept above 4096 tokens (control)

## Risk

Small expected effect at the edge of the interval; worth a run only as the A/B to router-triton-hilo-splitk-turn (cuBLAS-only route to the same limiter, no new kernel). Risk: cuBLAS may pick a worse algorithm for the fp32-output N=4608 GEMM than for the bf16-output N=4096 pooler, which would show as the pooler-sized nvjet growing by more than ~12%; at turn_4 the extra fp32 write (55 MB) can make it lose slightly. Routing is computed from the same fp32 hi+lo sums, so routing numerics match the parity path up to cuBLAS accumulation order.

## Adversarial review

Reviewed 2026-10-24T124300Z by an adversarial subagent (model opus).

**Verdict: hold.** Rule-clean, but the size of the win rests on a router side branch of ~80-90 us that the current master profile does not show. At turn_1 sliding the router pair is 32.0 us for 2 calls, under a ~52 us pooler plus a 7 us norm. The realistic effect is SM-contention relief of ~0.3-0.7% on the geomean, inside the ~1.5% run-to-run drift. With 1 prefill check left today, it should not use that check: the turn loads now trail SGLang (0.93-0.98x) because of the Triton expert GEMM, which is a bigger and resolvable lever.

### Justification
Mechanism: folding the hi/lo router (512 extra output rows) into the pooler GEMM over the same ffn_input is sound on H200. It removes two split-K nvjet_tss launches, a splitKreduce and the stream fork/join, and a 4608-wide cuBLAS tile GEMM should run the router's 6.3 GFLOP at turn_1 at ~700+ TFLOP/s (~8-9 us) instead of ~190 TFLOP/s (32 us). Profile on master (b33b87b5083b4d1c): turn_1 sliding shows nvjet_tss_64x96 splitK at 32.0 us/2 calls. The pooler is one of the three nvjet_128x192 calls (156.0 us total, ~52 us each), so the main branch is ~59 us and the side branch (32 us + reduce + FusedMuseSparkRouting, ~40 us) already fits under it. The 81.8 us splitK row on turn_1 full, for an identical M and identical router code, is wait time folded into an overlapping kernel's row (the profiler says rows include waits), not work. That is the same conclusion the router-triton-hilo-splitk-turn review reached. So the prize is not the claimed ~35-40 us removed from the critical path. It is the router SM-time no longer competing with the pooler: ~32 us of router SM-time replaced by ~6-8 us of extra pooler GEMM plus a ~3-5 us cast/slice-add, i.e. ~15-20 us at turn_1 (~1.5% of the 1141 us sliding shape). At turn_4, 131 us of router is replaced by ~25-30 us of GEMM plus ~15-20 us of cast/add, ~60-80 us out of 3435/4975 us (~2%). The spec expects turn_4 flat, but it is plausibly the better load. In load terms (TASK.md weights 46/16, geomean over 3 loads, first_turn flat by its gate), that is ~0.3-0.5% from turn_1 plus ~0.6% from turn_4: ~0.5-1% headline. One run cannot resolve that (LESSONS: ~1.5-2.5% per-shape drift). The concat GEMM's own row is readable in top_kernels, but the per-shape deltas will not be. Weakest link: the inherited '80-90 us side branch on the critical path' premise. Second risk: the extra fp32 output write (14 MB at T=753, 55 MB at T=3012) and cuBLAS choosing a different, possibly worse, algorithm for the fp32-out N=4608 GEMM than for the bf16-out N=4096 pooler.

### Confounders on the current kernel
- sglang-pipeline-parity: it is the base that provides the side-stream fork this change removes. It is not separable: reverting its patch removes the fork, the router path and the whole pipeline together, so there is no A/B against it. The change's effect depends only on how much the forked router overlaps the pooler, which this base fixes.
- experts-triton-w4a16-first-turn-blockt256 (with stage 2 under it): it does not touch the router, pooler or fork, and its first_turn-only tile change sits behind the same >4096 gate this hypothesis leaves on the parity path, so there is no interaction at first_turn. At turn shapes it replaced Marlin with the Triton expert GEMM, which now runs 847 us at turn_1 sliding vs Marlin's 730 us and leaves turn loads at 0.93-0.98x of SGLang. That changes nothing about the router/pooler overlap, but it lowers this hypothesis's priority. A later gate back to Marlin at the turn shapes would change the denominator (shape time), not the absolute saving.

### Interference and A/B needs
No companion is warranted, and the parent itself should wait. ab-revert has no meaningful target: reverting parity removes the fork entirely, and reverting the expert rewrite does not touch this path. ab-tune of the landed fork would be the sibling router-triton-hilo-splitk-turn (held), which is a different engine for the same limiter, not a tuning of a landed change. A control isolating a sub-change, for example the concat GEMM with the fork kept, would need its own check to resolve an effect that is already below one run's resolution. When it does run, it can be attributed without a companion: the tss splitK rows and the splitKreduce should vanish, the pooler-sized nvjet should grow by about 12% (more means cuBLAS picked a worse fp32-out algorithm), and first_turn is a built-in control. The headline cannot be attributed from one run either way, and with 1 check left today a companion is out of the question.

### Open questions
- Is the router branch on the turn critical path at all? On master, turn_1 sliding has the router at 32 us vs a ~52 us pooler plus a 7 us norm. Only a per-call trace (the profile json) would settle it, and if it is off the path the turn_1 'up to 35-40 us' is at most ~15-20 us of contention relief.
- Turn_4 may be the better load (131 us of router SM-time vs ~45 us added), contrary to the 'flat' expectation. Should the expected block be revised before any future run?
- Priority: master's turn shapes are now 0.93-0.98x of SGLang because the Triton expert GEMM is slower than Marlin at T=753/3012. Gating the turn shapes back to Marlin, or a turn-shape stage of the rewrite, is a ~5-7% per-shape lever. Should prefill's last check go there instead?
- When quota is plentiful, run this and router-triton-hilo-splitk-turn as a pair on the same day. They target the same branch and do not stack, so pairing them makes the router-row times comparable even though the headlines are unresolvable.

