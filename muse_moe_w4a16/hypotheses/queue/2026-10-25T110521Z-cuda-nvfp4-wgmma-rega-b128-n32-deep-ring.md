---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-b128-n32-deep-ring
created: 2026-10-25T110521Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050).
  Changes the bucket where wgmma LOSES (B128 = 128<=T<=255: verify_16 and the skewed-128 test) and
  nothing else: B256 (256<=T<=512) keeps s5's wg_kernel byte for byte and stays forced on, as an
  in-run control; T<128 and T>512 stay master. MECHANISM: s5's wg_kernel streams NVFP4 weight at
  ~2.2 TB/s whatever the route count (w13 238.4 us judged at bs=128 vs 271.0 at bs=256; prepare
  cold w13 303 vs 308 us at 241 vs 464 local routes), i.e. it is capped by bytes in flight, not by
  MMA or dequant (s3's double-buffered register-A variant B was no faster than variant A in every
  N bucket, so the consumer is not the limiter). Each stage is 9216 B of weight plus a token slab
  sized for N=128 (2 x SLAB_HALF = 2 x 16384 B), so STAGES=4 fits and only 4 x 9216 = 37 KB of
  weight per SM is in flight; LESSONS (bf16): 64 KB in flight per CTA reached 3.3 TB/s, 128 KB
  reached 3.73 TB/s. At B128 the judge-like routing has ~7.5 routes per local expert, so nearly
  every chunk is the N16 bucket and the 32 KB slab space is idle. CHANGE: add a second
  instantiation wg_kernel<G, NMAX=32> used only for B128: (1) on-device chunking cuts chunks every
  2 blocks of 16 rows (N <= 32) instead of every 8, so only the N16/N32 buckets and TMA maps are
  used (an expert with more than 32 routes is split into several chunks and re-streams its weight
  per chunk: correct, slower only for skewed tests); (2) SLAB_HALF becomes 32 x 128 = 4096 B and
  STAGE_BYTES = 9216 + 8192 = 17408 (1024-B aligned for SWIZZLE_128B, as are the weight record and
  each slab half); make_desc offsets use the template SLAB_HALF; (3) STAGES = 10 for both w13 and
  w2 (dyn smem w13 1024 + 174080 + 18432 = 193536 B, w2 1024 + 174080 + 34816 = 209920 B, each +
  <=8192 static under 227 KB), so 92 KB of weight is in flight per SM, 2.5x s5; the mbarrier
  arrays and phase arithmetic take STAGES as the template constant. Nothing else moves: same
  weight bank, gather, variant A (wgmma_wait<0>), epilogues, per-call cuTensorMapEncodeTiled from
  bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and 128<=T<=255 picks the NMAX=32 launch;
  256<=T<=512 the s5 launch). Bump the load_inline module name. prepare: build-time register/spill
  print for both new instantiations (must be <=232 regs, 0 B spill, else B128 falls back to the s5
  kernel and the log says why); the s5 per-bucket canary for N16/N32 on the new kernel; s5's
  three-state copy check (A, deepcopy B, tree-cloned C) and device launch marker at T=128 with a
  forced >32-routes-per-expert routing (all-one-expert, skewed-4hot) so multi-chunk re-streaming
  is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing gate (forced,
  as s5, so the judged profile names the new kernel rows); print the s5-style Zipf cold-L2 timing
  at T=128 for NMAX=32 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph
  capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints,
  no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4
  (dequant in registers), activations BF16; attend every route; no child processes; no
  module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state
  copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "s5 judged wgmma rows at bs=128: w13+silu 238.4 + w2 110.9 + gather 3.7 = 353.0 us vs master Marlin pair 305.9 + act_and_mul 4.0 = 309.9 us (shape 364.8 vs master 317.8). 2.5x weight bytes in flight should lift ~2.2 TB/s toward 3.3-3.7 TB/s: target w13 <=170, w2 <=80, pair <=0.9x Marlin (<=279 us), shape row <=300 us"
  - shapes: bs=256 *
    direction: flat
    why: "in-run control: s5 kernel unchanged at B256 (404.7 us shape, wg rows 271.0 + 126.1 + 4.8)"
  - shapes: bs=1 *
    direction: flat
    why: "control: master below 128 tokens"
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=255
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-b128-n32-deep-ring

## Change

Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050). Changes the bucket where wgmma LOSES (B128 = 128<=T<=255: verify_16 and the skewed-128 test) and nothing else: B256 (256<=T<=512) keeps s5's wg_kernel byte for byte and stays forced on, as an in-run control; T<128 and T>512 stay master. MECHANISM: s5's wg_kernel streams NVFP4 weight at ~2.2 TB/s whatever the route count (w13 238.4 us judged at bs=128 vs 271.0 at bs=256; prepare cold w13 303 vs 308 us at 241 vs 464 local routes), i.e. it is capped by bytes in flight, not by MMA or dequant (s3's double-buffered register-A variant B was no faster than variant A in every N bucket, so the consumer is not the limiter). Each stage is 9216 B of weight plus a token slab sized for N=128 (2 x SLAB_HALF = 2 x 16384 B), so STAGES=4 fits and only 4 x 9216 = 37 KB of weight per SM is in flight; LESSONS (bf16): 64 KB in flight per CTA reached 3.3 TB/s, 128 KB reached 3.73 TB/s. At B128 the judge-like routing has ~7.5 routes per local expert, so nearly every chunk is the N16 bucket and the 32 KB slab space is idle. CHANGE: add a second instantiation wg_kernel<G, NMAX=32> used only for B128: (1) on-device chunking cuts chunks every 2 blocks of 16 rows (N <= 32) instead of every 8, so only the N16/N32 buckets and TMA maps are used (an expert with more than 32 routes is split into several chunks and re-streams its weight per chunk: correct, slower only for skewed tests); (2) SLAB_HALF becomes 32 x 128 = 4096 B and STAGE_BYTES = 9216 + 8192 = 17408 (1024-B aligned for SWIZZLE_128B, as are the weight record and each slab half); make_desc offsets use the template SLAB_HALF; (3) STAGES = 10 for both w13 and w2 (dyn smem w13 1024 + 174080 + 18432 = 193536 B, w2 1024 + 174080 + 34816 = 209920 B, each + <=8192 static under 227 KB), so 92 KB of weight is in flight per SM, 2.5x s5; the mbarrier arrays and phase arithmetic take STAGES as the template constant. Nothing else moves: same weight bank, gather, variant A (wgmma_wait<0>), epilogues, per-call cuTensorMapEncodeTiled from bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and 128<=T<=255 picks the NMAX=32 launch; 256<=T<=512 the s5 launch). Bump the load_inline module name. prepare: build-time register/spill print for both new instantiations (must be <=232 regs, 0 B spill, else B128 falls back to the s5 kernel and the log says why); the s5 per-bucket canary for N16/N32 on the new kernel; s5's three-state copy check (A, deepcopy B, tree-cloned C) and device launch marker at T=128 with a forced >32-routes-per-expert routing (all-one-expert, skewed-4hot) so multi-chunk re-streaming is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing gate (forced, as s5, so the judged profile names the new kernel rows); print the s5-style Zipf cold-L2 timing at T=128 for NMAX=32 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints, no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16; attend every route; no child processes; no module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.

## Expected

- `bs=128 *` faster: s5 judged wgmma rows at bs=128: w13+silu 238.4 + w2 110.9 + gather 3.7 = 353.0 us vs master Marlin pair 305.9 + act_and_mul 4.0 = 309.9 us (shape 364.8 vs master 317.8). 2.5x weight bytes in flight should lift ~2.2 TB/s toward 3.3-3.7 TB/s: target w13 <=170, w2 <=80, pair <=0.9x Marlin (<=279 us), shape row <=300 us
- `bs=256 *` flat: in-run control: s5 kernel unchanged at B256 (404.7 us shape, wg rows 271.0 + 126.1 + 4.8)
- `bs=1 *` flat: control: master below 128 tokens

## Risk

Risk: if bandwidth is limited by something other than bytes in flight (per-item epilogue bubbles, the single producer thread, L2 slab traffic), w13 stays near 238 us; the s3 variant-B null already rules out consumer dequant/MMA overlap. Second risk: smem/regs for STAGES=10 plus a second instantiation pair grows build time; s5 prepare was ~40 s with a 200 s setup budget. Read the judged bs=128 wg_kernel rows against s5's 238.4/110.9, not the headline (bs=128 Marlin rows drift 306-350 us across runs). Milestones: M1 new kernel rows appear in the bs=128 profile; M2 w13+w2+gather <=0.9x master's Marlin pair+act (<=279 us); M3 B128 enabled behind the LESSONS-173 selector in a later stage.

## Milestones

Rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=255`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
