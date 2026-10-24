---
problem: muse_bf16_proj
slug: router-splitk32-small-m-all-loads-in-flight
created: 2026-10-24T075025Z
base: master
area: router GEMM
change: >
  Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the
  num_stages they launch with; the kernel body, the torch.zeros output + relaxed fp32 atomic_add
  epilogue, BLOCK_K=128, num_warps=4, and the m > 32 buckets (64,64,16) / (64,64,8) with
  num_stages=3 stay exactly as master. New small-m buckets: m <= 16: BLOCK_M=16, BLOCK_N=32,
  SPLIT_K=32; 17..32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=32. Grid becomes (1, 8, 32) = 256 CTAs (was
  128), K-slice 256 = 2 K-steps of BLOCK_K=128 (was 4), and these buckets launch with num_stages=2
  so both K-steps' hi, lo and x tiles are issued before the first dot (per CTA ~40 KB smem, so ~2
  CTAs per SM are co-resident and all 256 run in one wave). Implement by having _router_config
  return (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES) and passing num_stages from it;
  _router_shape_ok's k % (128 * SPLIT_K) check still holds (8192 % 4096 == 0). Keep ROUTER_WARM_MS
  so both new specializations compile in prepare().
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: "_router_hilo_kernel is 3.3 us for 8.4 MB (2.5 TB/s) vs a 1.8 us floor: with 4 K-steps and 3 stages each CTA pays a serial prologue + drain; 256 CTAs with all loads issued up front should cut ~0.6-1 us of the 5.8 us row"
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: m=8, same bucket, 5.7 us
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: same bucket, kernel 3.5 us, 6.2 us row
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: kernel 4.1 us, 6.5 us row; BLOCK_M=32 bucket
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: m=128 bucket unchanged; in-run control
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: m=256 bucket unchanged; in-run control
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# router-splitk32-small-m-all-loads-in-flight

## Change

Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + relaxed fp32 atomic_add epilogue, BLOCK_K=128, num_warps=4, and the m > 32 buckets (64,64,16) / (64,64,8) with num_stages=3 stay exactly as master. New small-m buckets: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=32; 17..32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=32. Grid becomes (1, 8, 32) = 256 CTAs (was 128), K-slice 256 = 2 K-steps of BLOCK_K=128 (was 4), and these buckets launch with num_stages=2 so both K-steps' hi, lo and x tiles are issued before the first dot (per CTA ~40 KB smem, so ~2 CTAs per SM are co-resident and all 256 run in one wave). Implement by having _router_config return (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES) and passing num_stages from it; _router_shape_ok's k % (128 * SPLIT_K) check still holds (8192 % 4096 == 0). Keep ROUTER_WARM_MS so both new specializations compile in prepare().

## Expected

- `layer=router load=decode_1 *` faster: _router_hilo_kernel is 3.3 us for 8.4 MB (2.5 TB/s) vs a 1.8 us floor: with 4 K-steps and 3 stages each CTA pays a serial prologue + drain; 256 CTAs with all loads issued up front should cut ~0.6-1 us of the 5.8 us row
- `layer=router load=verify_1 *` faster: m=8, same bucket, 5.7 us
- `layer=router load=decode_16 *` faster: same bucket, kernel 3.5 us, 6.2 us row
- `layer=router load=decode_32 *` faster: kernel 4.1 us, 6.5 us row; BLOCK_M=32 bucket
- `layer=router load=verify_16 *` flat: m=128 bucket unchanged; in-run control
- `layer=router load=verify_32 *` flat: m=256 bucket unchanged; in-run control

## Risk

A pure one-axis retune (grid depth x pipeline depth) of the landed kernel on the four m <= 32 router rows; m=128/256 rows are in-run controls for noise. The idea is memory-level parallelism, not tile shape: master's 128 CTAs each hold ~3 x 16 KB of weight in flight for only 4 steps, so each CTA's life is ~2 DRAM round trips plus launch; doubling the CTA count and halving the steps puts the whole 8.4 MB in flight at once. Costs: atomic traffic doubles (m=32: 262K fp32 atomics, ~1 MB, into a 32 KB output that stays in L2), and x is re-read by 8 N-blocks (trivial at m <= 32). Per-shape router effect is ~10-15% on 4 shapes but router is ~11% of each decode load, so geomean ~+1%: read the per-shape rows and the kernel us, not the headline. If flat, the router's small-m floor is launch + memset, and router-splitk-semaphore-no-memset is the remaining lever. Stacking: independent of the memset hypothesis (different lines); if both gain, combine and re-run.
