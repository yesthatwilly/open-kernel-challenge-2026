---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-b256-n64-deep-ring
created: 2026-10-25T110544Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050).
  Widens the margin in the bucket where wgmma already WINS (B256 = 256<=T<=512: verify_32 and the
  bs=257 test) and changes nothing else: B128 (128<=T<=255) keeps s5's wg_kernel byte for byte and
  stays forced on, as an in-run control; T<128 and T>512 stay master. MECHANISM: at bs=256 s5's
  wgmma rows (w13+silu 271.0 + w2 126.1 + gather 4.8 = 401.9 us) already beat master's Marlin pair
  + act_and_mul (451.5 + 5.6 = 457.1 us), but the kernel is still ~3.7x the 108.9 us floor and its
  time barely grows with routes (prepare cold w13 303 -> 308 us from 241 to 464 local routes): it
  is capped by weight bytes in flight (STAGES=4 x 9216 B = 37 KB per SM, because each stage also
  reserves a 32 KB token slab sized for N=128), not by MMA or dequant (s3's double-buffered
  variant B was no faster in any N bucket). At ~14.5 routes per local expert nearly every chunk is
  N16/N32 and rarely N64. CHANGE: add a second instantiation wg_kernel<G, NMAX=64> used only for
  B256: (1) on-device chunking cuts chunks every 4 blocks of 16 rows (N <= 64; an expert with more
  than 64 routes splits into several chunks and re-streams its weight per chunk, correct and
  slower only for skewed tests), so only the N16/N32/N64 buckets and maps are used; (2) SLAB_HALF
  becomes 64 x 128 = 8192 B and STAGE_BYTES = 9216 + 16384 = 25600 (1024-B aligned for
  SWIZZLE_128B, as are the weight record and each slab half); make_desc offsets use the template
  SLAB_HALF; (3) STAGES = 7 for both w13 and w2 (dyn smem w13 1024 + 179200 + 18432 = 198656 B, w2
  1024 + 179200 + 34816 = 215040 B, each + <=8192 static under 227 KB), so 64.5 KB of weight is in
  flight per SM, 1.75x s5; mbarrier arrays and phase arithmetic use STAGES as the template
  constant. Nothing else moves: same weight bank, gather, variant A (wgmma_wait<0>), epilogues,
  per-call cuTensorMapEncodeTiled from bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and
  256<=T<=512 picks the NMAX=64 launch; 128<=T<=255 the s5 launch). Bump the load_inline module
  name. prepare: register/spill print for both new instantiations (<=232 regs, 0 B spill, else
  B256 falls back to the s5 kernel and the log says why); s5's per-bucket canary for N16/N32/N64
  on the new kernel; s5's three-state copy check and device launch marker at T=256 and T=257,
  including a routing with >64 routes on one expert (all-one-expert, skewed-4hot) so multi-chunk
  re-streaming is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing
  gate (forced, so the judged profile names the new rows); print the s5-style Zipf cold-L2 timing
  at T=256 for NMAX=64 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph
  capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints,
  no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4
  (dequant in registers), activations BF16; attend every route; no child processes; no
  module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state
  copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "s5 judged wgmma rows at bs=256: w13+silu 271.0 + w2 126.1 + gather 4.8 = 401.9 us (shape 404.7) vs master Marlin pair 451.5 + act 5.6 (shape 467.8). 1.75x weight bytes in flight (LESSONS bf16: 64 KB/CTA reached 3.3 TB/s vs ~2.2 TB/s here): target w13 <=210, w2 <=100, shape row <=350 us"
  - shapes: bs=128 *
    direction: flat
    why: "in-run control: s5 kernel unchanged at B128 (364.8 us shape, wg rows 238.4 + 110.9 + 3.7)"
  - shapes: bs=1 *
    direction: flat
    why: "control: master below 128 tokens"
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 256<=T<=512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-b256-n64-deep-ring

## Change

Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2764050). Widens the margin in the bucket where wgmma already WINS (B256 = 256<=T<=512: verify_32 and the bs=257 test) and changes nothing else: B128 (128<=T<=255) keeps s5's wg_kernel byte for byte and stays forced on, as an in-run control; T<128 and T>512 stay master. MECHANISM: at bs=256 s5's wgmma rows (w13+silu 271.0 + w2 126.1 + gather 4.8 = 401.9 us) already beat master's Marlin pair + act_and_mul (451.5 + 5.6 = 457.1 us), but the kernel is still ~3.7x the 108.9 us floor and its time barely grows with routes (prepare cold w13 303 -> 308 us from 241 to 464 local routes): it is capped by weight bytes in flight (STAGES=4 x 9216 B = 37 KB per SM, because each stage also reserves a 32 KB token slab sized for N=128), not by MMA or dequant (s3's double-buffered variant B was no faster in any N bucket). At ~14.5 routes per local expert nearly every chunk is N16/N32 and rarely N64. CHANGE: add a second instantiation wg_kernel<G, NMAX=64> used only for B256: (1) on-device chunking cuts chunks every 4 blocks of 16 rows (N <= 64; an expert with more than 64 routes splits into several chunks and re-streams its weight per chunk, correct and slower only for skewed tests), so only the N16/N32/N64 buckets and maps are used; (2) SLAB_HALF becomes 64 x 128 = 8192 B and STAGE_BYTES = 9216 + 16384 = 25600 (1024-B aligned for SWIZZLE_128B, as are the weight record and each slab half); make_desc offsets use the template SLAB_HALF; (3) STAGES = 7 for both w13 and w2 (dyn smem w13 1024 + 179200 + 18432 = 198656 B, w2 1024 + 179200 + 34816 = 215040 B, each + <=8192 static under 227 KB), so 64.5 KB of weight is in flight per SM, 1.75x s5; mbarrier arrays and phase arithmetic use STAGES as the template constant. Nothing else moves: same weight bank, gather, variant A (wgmma_wait<0>), epilogues, per-call cuTensorMapEncodeTiled from bsrc, shape+flag kernel_fn gate (state['wgmma_ok'] and 256<=T<=512 picks the NMAX=64 launch; 128<=T<=255 the s5 launch). Bump the load_inline module name. prepare: register/spill print for both new instantiations (<=232 regs, 0 B spill, else B256 falls back to the s5 kernel and the log says why); s5's per-bucket canary for N16/N32/N64 on the new kernel; s5's three-state copy check and device launch marker at T=256 and T=257, including a routing with >64 routes on one expert (all-one-expert, skewed-4hot) so multi-chunk re-streaming is exercised; error vs the fp32-dequant reference <=1.5x Marlin's own. No timing gate (forced, so the judged profile names the new rows); print the s5-style Zipf cold-L2 timing at T=256 for NMAX=64 vs s5 kernel vs Marlin, compact and last in the log. Hard rules: CUDA-graph capturable; kernel_fn does no host sync, no allocation or branching on input values, no prints, no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16; attend every route; no child processes; no module/static global holds device pointers, TMA maps or buffers (judge rotates TWO state copies); TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20 as s5.

## Expected

- `bs=256 *` faster: s5 judged wgmma rows at bs=256: w13+silu 271.0 + w2 126.1 + gather 4.8 = 401.9 us (shape 404.7) vs master Marlin pair 451.5 + act 5.6 (shape 467.8). 1.75x weight bytes in flight (LESSONS bf16: 64 KB/CTA reached 3.3 TB/s vs ~2.2 TB/s here): target w13 <=210, w2 <=100, shape row <=350 us
- `bs=128 *` flat: in-run control: s5 kernel unchanged at B128 (364.8 us shape, wg rows 238.4 + 110.9 + 3.7)
- `bs=1 *` flat: control: master below 128 tokens

## Risk

Twin of cuda-nvfp4-wgmma-rega-b128-n32-deep-ring on the other bucket, with each run's untouched wgmma bucket as the other's control; if both move, the bytes-in-flight mechanism is confirmed across N caps; if B128 (10 stages) moves and B256 (7 stages) does not, depth matters more than N cap. Risk: a limit other than bytes in flight (per-item epilogue bubbles, the single producer thread, L2 slab traffic) leaves w13 near 271 us. Read the judged bs=256 wg_kernel rows against s5's 271.0/126.1, not the headline (Marlin verify rows drift 10% across runs). Milestones: M1 new rows in the bs=256 profile; M2 wgmma pair+gather <=0.8x master's Marlin pair+act (<=366 us); M3 ship B256 behind the LESSONS-173 selector (with or after s5-copy-safe-bucket-gate).

## Milestones

Rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 256<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
