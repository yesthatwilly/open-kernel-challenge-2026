---
problem: muse_decode_layer
slug: router-triton-splitk-side-stream
created: 2026-10-24T075400Z
base: master
area: router GEMM
change: >
  Port muse_bf16_proj's landed router kernel (master muse_bf16_proj/submission.py, hypothesis
  router-triton-hilo-fused-control-splitk-all-m, 1.151x there: 10.5-17.7 us -> 5.7-9.3 us at every
  m) into this layer's side-stream router, at every T. Copy verbatim ROUTER_BLOCK_K=128,
  ROUTER_NUM_STAGES=3, ROUTER_NUM_WARPS=4, _router_config(m) (m<=16: 16/32/16, m<=32: 32/32/16,
  m<=128: 64/64/16, else 64/64/8; grid (cdiv(M,BLOCK_M), 256//BLOCK_N, SPLIT_K), ~128 CTAs), the
  @triton.jit(do_not_specialize=['M']) _router_hilo_kernel (fp32 acc of x@hi + x@lo, relaxed fp32
  atomic_add into an output zeroed by torch.zeros right before the launch) and _router_forward. In
  prepare(), store router_hi and router_lo as [256, 8192] bf16 ROW-MAJOR (the kernel wants N x K;
  drop router_hi_t / router_lo_t and the fp32 'router' copy) and warm every config bucket on
  synthetic bf16 x for m in (1, 8, 9, 16, 24, 32, 33, 40, 64, 128, 256) so the timed call never
  compiles. In kernel_fn: set ROUTER_FP32_MAX_TOKENS = 0 so split_router is always True, i.e.
  attn_boundary is always called with router_dtype=None (it no longer writes the fp32 router input
  at T=1), and inside the existing 'with torch.cuda.stream(side_stream):' block replace both
  branches (the F.linear fp32 gemv at T=1 and the two torch.mm(..., out_dtype=float32) hi/lo GEMMs
  plus the += at T>=2) with logits = _router_forward(ffn_input, router_hi, router_lo). The routing
  call, the side stream, its wait_stream / record_stream handling, the pooler and everything else
  stay exactly as on master. Mechanism: at T>=16 the side-stream path is two cuBLAS tss splitK
  GEMMs (~9.3 us each in profile 5755a412dda244a1) + an fp32 add + FusedMuseSparkRouting, which is
  as long as or longer than the main stream's pooler (nvjet 128x16 splitK ~20 us + its splitK
  reduce), so main_stream.wait_stream(side_stream) stalls the MoE block; one ~6-9 us kernel takes
  the router off the critical path and halves the bandwidth it steals from the concurrent pooler
  (x is read once, not twice). Rules: the router's FP32 weights may run as a BF16 hi + lo pair
  with FP32 accumulation at any batch size.
expected:
  - shapes: "*load=decode_16*"
    direction: faster
    why: router path (2 x 9.3 us hi/lo GEMMs + add + routing) is at least as long as the ~20 us pooler it overlaps; one ~6-9 us kernel takes it off the critical path (expect ~4-10 us on ~265/440 us)
  - shapes: "*load=decode_32*"
    direction: faster
    why: same mechanism at m=32 (BLOCK_M=32 bucket); largest relative effect on the 265 us sliding twin
  - shapes: "*load=verify_16*"
    direction: flat
    why: m=128 router also shrinks, but the shape is 520-714 us dominated by Marlin; a few us is under the 1.5% drift
  - shapes: "*load=verify_32*"
    direction: flat
    why: m=256 bucket (SPLIT_K=8); same reasoning, 713-1050 us shapes
  - shapes: "*load=decode_1*"
    direction: flat
    why: T=1 router (fp32 gemv ~5 us) is already hidden under the ~28 us pooler; the bf16 input changes numerics only
  - shapes: "*load=verify_1*"
    direction: flat
    why: T=8 hi/lo pair already hidden under the pooler; control for the T>=16 reading
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# router-triton-splitk-side-stream

## Change

Port muse_bf16_proj's landed router kernel (master muse_bf16_proj/submission.py, hypothesis router-triton-hilo-fused-control-splitk-all-m, 1.151x there: 10.5-17.7 us -> 5.7-9.3 us at every m) into this layer's side-stream router, at every T. Copy verbatim ROUTER_BLOCK_K=128, ROUTER_NUM_STAGES=3, ROUTER_NUM_WARPS=4, _router_config(m) (m<=16: 16/32/16, m<=32: 32/32/16, m<=128: 64/64/16, else 64/64/8; grid (cdiv(M,BLOCK_M), 256//BLOCK_N, SPLIT_K), ~128 CTAs), the @triton.jit(do_not_specialize=['M']) _router_hilo_kernel (fp32 acc of x@hi + x@lo, relaxed fp32 atomic_add into an output zeroed by torch.zeros right before the launch) and _router_forward. In prepare(), store router_hi and router_lo as [256, 8192] bf16 ROW-MAJOR (the kernel wants N x K; drop router_hi_t / router_lo_t and the fp32 'router' copy) and warm every config bucket on synthetic bf16 x for m in (1, 8, 9, 16, 24, 32, 33, 40, 64, 128, 256) so the timed call never compiles. In kernel_fn: set ROUTER_FP32_MAX_TOKENS = 0 so split_router is always True, i.e. attn_boundary is always called with router_dtype=None (it no longer writes the fp32 router input at T=1), and inside the existing 'with torch.cuda.stream(side_stream):' block replace both branches (the F.linear fp32 gemv at T=1 and the two torch.mm(..., out_dtype=float32) hi/lo GEMMs plus the += at T>=2) with logits = _router_forward(ffn_input, router_hi, router_lo). The routing call, the side stream, its wait_stream / record_stream handling, the pooler and everything else stay exactly as on master. Mechanism: at T>=16 the side-stream path is two cuBLAS tss splitK GEMMs (~9.3 us each in profile 5755a412dda244a1) + an fp32 add + FusedMuseSparkRouting, which is as long as or longer than the main stream's pooler (nvjet 128x16 splitK ~20 us + its splitK reduce), so main_stream.wait_stream(side_stream) stalls the MoE block; one ~6-9 us kernel takes the router off the critical path and halves the bandwidth it steals from the concurrent pooler (x is read once, not twice). Rules: the router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation at any batch size.

## Expected

- `*load=decode_16*` faster: router path (2 x 9.3 us hi/lo GEMMs + add + routing) is at least as long as the ~20 us pooler it overlaps; one ~6-9 us kernel takes it off the critical path (expect ~4-10 us on ~265/440 us)
- `*load=decode_32*` faster: same mechanism at m=32 (BLOCK_M=32 bucket); largest relative effect on the 265 us sliding twin
- `*load=verify_16*` flat: m=128 router also shrinks, but the shape is 520-714 us dominated by Marlin; a few us is under the 1.5% drift
- `*load=verify_32*` flat: m=256 bucket (SPLIT_K=8); same reasoning, 713-1050 us shapes
- `*load=decode_1*` flat: T=1 router (fp32 gemv ~5 us) is already hidden under the ~28 us pooler; the bf16 input changes numerics only
- `*load=verify_1*` flat: T=8 hi/lo pair already hidden under the pooler; control for the T>=16 reading

## Risk

Risk: the router's 128 CTAs now run concurrently with the pooler's splitK nvjet kernel on the main stream; SM contention could slow the pooler by about what the router saves, which would read as flat at T>=16. Routing numerics at T=1 change from the fp32 router input to the bf16 ffn_input with a hi/lo weight split (already what every T>=2 does on master and what passed 13/13 for hilo-m8); the routing tie rules (0.05 / 0.03 margins) absorb bf16-input logit error at 8192-dim. Atomic fp32 accumulation order is nondeterministic across 16 K-slices: logit ties can flip run to run, still within the tie rules. The test shapes T in {1, 8, 9, 16, 24, 32, 33, 40} must all be warmed in prepare. If this is flat, the follow-up is running the router on the main stream ahead of the pooler (no overlap) to separate contention from the critical-path effect.
