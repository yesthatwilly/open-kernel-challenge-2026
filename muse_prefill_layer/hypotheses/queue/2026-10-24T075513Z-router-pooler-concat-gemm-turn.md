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
review: required
status: queued
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
