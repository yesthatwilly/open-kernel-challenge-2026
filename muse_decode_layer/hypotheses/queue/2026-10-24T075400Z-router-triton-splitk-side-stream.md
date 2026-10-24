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
review: proceed
status: queued
review_at: 2026-10-24T092231Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T092231Z by an adversarial subagent (model opus).

**Verdict: proceed.** The change is cheap, breaks no rule, duplicates nothing, and the per-shape rows attribute it cleanly, so it should run. Its premise that the router is on the critical path at T>=16 comes from misreading the profile, though, so the most likely result is flat, and its priority should drop from 1.

### Justification
The kernel itself is well proven: muse_bf16_proj measured the same [T,8192]x[256,8192] hi/lo router going from 10.5-17.7 us to 5.7-9.3 us with this exact 3-D split-K Triton kernel, and LESSONS.md lists it as the one large bf16 win. The mechanism claimed for THIS layer is the weak part. It says the side-stream router at T>=16 is 'two cuBLAS tss splitK GEMMs (~9.3 us each)'. In the profile the GPU-time column is a total over all calls, and the master profile 2b1f82408ab64116 shows nvjet_sm90_tss_64x16_64x16_4x1_v_bz_splitK_NNT at 9.2 us for 2 calls at decode_16 (both layers) and nvjet_sm90_tss_32x64 at 9.5 us for 2 calls at decode_32 sliding. That is about 4.6-4.8 us per GEMM. Add two splitKreduce launches, the fp32 add and FusedMuseSparkRouting, and the side branch is roughly 14-18 us. The hilo-m8 review's own isolated numbers agree: hi/lo router 14.5 us at m=16 and 15.1 at m=32, routing ~3 us, against a pooler of nvjet 128x16/128x32 splitK at 20.0-22.4 us plus its reduce (~22-25 us). So at decode_16/32 the router most likely already finishes before the pooler and main_stream.wait_stream does not stall. Saving 6-9 us of side-stream time then saves nothing on the critical path. The verify shapes are the same story with more slack: the pooler is 25-34 us at m=128/256 against a 16.6-17.8 us router. decode_1 (a 5 us dot_kernel gemv) is clearly hidden. The secondary claim, 'halves the bandwidth it steals from the pooler (x is read once, not twice)', is also wrong in size: the traffic is the 8 MB hi+lo weight read, which the Triton kernel still does in full; reading x once saves only 256-512 KB. The real second-order effects are fewer graph nodes on the side stream (two splitKreduce, one add) and different SM occupancy next to the pooler's splitK kernel (128 CTAs, can go either way). These are worth perhaps 0-3 us, below the ~1.5% per-shape drift on 254-602 us shapes. The 'expected' directions agree with TASK.md's weights and floors only if the router is critical, which the data does not support, so 'faster' on decode_16/32 should be read as 'flat-to-slightly-faster'. Weakest link: the per-call versus total misreading of the tss splitK row, on which the whole critical-path argument rests.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (router GEMM, same code). This hypothesis replaces that patch's T>=2 torch.mm hi/lo pair and also overrides its ROUTER_FP32_MAX_TOKENS (1 -> 0). The change supersedes it rather than interacting with it. Reverting it would only restore SGLang's fp32 path at T<=8, which the new ROUTER_FP32_MAX_TOKENS=0 bypasses anyway, so an ab-revert would measure the same code at every T except the dead constant. Not a confounder.
- marlin-block-m-cap-16 (w13 marlin). It only shortens the main-stream MoE after the side-stream join, at T=128/256. It does not change how long the pooler runs relative to the router, and the router is not on the critical path at those shapes in either case. It cannot create or hide a router-attributable change. Not a confounder.

### Interference and A/B needs
No companion is warranted. Neither landed hypothesis confounds the change (see confounders), so ab-revert and ab-tune have nothing to separate. The bundled sub-changes are already separated by shape: switching T=1 to bf16 input + hi/lo shows only on decode_1, and the Triton kernel replacing the cuBLAS pair shows on decode_16/32 and verify_*. The per-shape rows therefore attribute each one without a control. A gain at decode_16/32 is unambiguous because only the side-stream router kernel changed. A loss at T>=16 would be readable from the profile: if the pooler's nvjet splitK row grows against master's 20.0-22.4 us, that is SM contention. The one ambiguous outcome, flat, cannot be resolved by a companion run of this change. Its explanation is that the router was already hidden, which master's numbers already predict, and the kernel's own top_kernels time in this run confirms the kernel got faster. The hypothesis's suggested follow-up (router on the main stream ahead of the pooler) is a new hypothesis, not a control, and should only be minted if this run is flat and the router-critical question still matters for a later fusion.

### Open questions
- Is the profile's GPU-time column per kernel name summed over calls (as Marlin's 2 calls suggest)? If so, the hypothesis's '~9.3 us each' is 9.3 us for both, and the decode_16/32 'faster' prediction should be restated as flat. Priority 1 is not justified on that basis.
- In this run's profile, does the pooler's nvjet 128x16/128x32 splitK row stay at master's 20.0-22.4 us next to a 128-CTA Triton kernel? A rise means SM contention, which makes the main-stream-router follow-up worth minting.
- Does 'GPU busy' versus 'total' in the decode_16/32 profiles move relative to master (for example decode_32 full: 639.2 total vs 611.4 busy)? It is the one in-run signal of whether the side branch was overlapping or overrunning.

