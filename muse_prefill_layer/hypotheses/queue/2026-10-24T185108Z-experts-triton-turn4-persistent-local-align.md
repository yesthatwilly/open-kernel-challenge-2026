---
problem: muse_prefill_layer
slug: experts-triton-turn4-persistent-local-align
created: 2026-10-24T185108Z
base: master
area: w13 marlin + w2 marlin at turn_4 (route 1024 < T <= 4096 to the Triton NVFP4 engine, with persistent grid and local-routes align)
change: >
  Built on master (635cfee). MOTIVATION: the Triton swap-AB kernel already measured 2100-2318 us
  for w13+w2 at turn_4 (experts-triton-w4a16-swapab-reglhs and -first-turn-blockt256 runs, BLOCK_T
  128 config), against Marlin's 2131-2300 us on master: flat. But those Triton numbers include
  dead work Marlin does not do: at T=3012 the static capacity of moe_align_block_size(local_ids,
  128, 256) is 24096 + 257*127 = ~56.7K slots (~444 token tiles) of which only ~40 hold local
  routes; ~165 are -1 tiles (w13 returns, w2 zero-stores ~170 MB) and ~240 lie past n_padded, i.e.
  ~26K dead w13 and ~13K dead w2 CTAs, est. 150-250 us. Without it the Triton engine should beat
  Marlin at turn_4 by ~5-10%. CHANGE, three parts, only for 1024 < T <= 4096 (turn_4 T=3012 and
  the T=4096 test; turn_1 and the smaller tests keep Marlin): (1) in _experts, route that range to
  _experts_triton (it already selects EXPERT_CONFIGS' (4096, BLOCK_T=128, BLOCK_W=128, 8 warps, 3
  stages) entry; _check_expert_gemms already compiles and checks it). (2) PERSISTENT GRID in
  _moe_nvfp4_gemm_kernel, exactly as hypothesis experts-triton-persistent-grid-first-turn
  specifies (grid = min(NUM_SMS * PROGS_PER_SM, static tiles) with PROGS_PER_SM = 2 for BLOCK_T
  128 if the compiled kernel's smem and regs allow two CTAs per SM, else 1; loop over tile = pid,
  pid + num_programs, ... < cdiv(n_padded, BLOCK_T) * (N // BLOCK_W); master's GROUP_SIZE_T
  mapping on tile; if/else instead of return). This changes the kernel for first_turn too, so
  first_turn will also move. (3) LOCAL-ROUTES ALIGN on the _experts_triton path, exactly as
  hypothesis moe-align-local-routes-parallel-port specifies (muse_moe_w4a16's
  _align_parallel_kernel, local_start loaded from data['local_start'] on device, local_count 32,
  block = cfg['BLOCK_T']), replacing _local_expert_ids + moe_align_block_size on that path;
  kernel_fn passes topk_ids and data['local_start'] down; the Marlin path keeps master's remap and
  align. silu_and_mul and fused_post_expert_norm_weighted_sum are unchanged (they receive the
  align's local_ids). prepare() gates, one log line each, any failure keeps master's routing
  (Marlin at T <= 4096) for the run and logs '[turn4-triton] OFF: <reason>', else 'ON': align
  self-check vs moe_align_block_size per-expert route sets at T=3012 and 16384; NaN-poison check
  that fused_post_expert_norm_weighted_sum ignores remote slots; _check_expert_gemms; and
  _check_experts_e2e run at T=3012 as well as 8192 (Triton vs Marlin, rel L2 <= 2e-2, worst row <=
  5x).
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "experts 2131-2300 us (Marlin) -> est. 1.9-2.05 ms (the measured 2.1-2.3 ms Triton turn_4 time minus ~150-250 us of dead CTAs and remote zero-stores, minus SGLang align ~25 us -> ~10 us): -4 to -9% on turn_4 shapes, ~+1.5-3% on the headline"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "the persistent grid and local align also apply at T > 4096 (see experts-triton-persistent-grid-first-turn): -3 to -5%"
  - shapes: "*load=turn_1*"
    direction: neutral
    why: T=753 keeps Marlin
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-turn4-persistent-local-align

## Change

Built on master (635cfee). MOTIVATION: the Triton swap-AB kernel already measured 2100-2318 us for w13+w2 at turn_4 (experts-triton-w4a16-swapab-reglhs and -first-turn-blockt256 runs, BLOCK_T 128 config), against Marlin's 2131-2300 us on master: flat. But those Triton numbers include dead work Marlin does not do: at T=3012 the static capacity of moe_align_block_size(local_ids, 128, 256) is 24096 + 257*127 = ~56.7K slots (~444 token tiles) of which only ~40 hold local routes; ~165 are -1 tiles (w13 returns, w2 zero-stores ~170 MB) and ~240 lie past n_padded, i.e. ~26K dead w13 and ~13K dead w2 CTAs, est. 150-250 us. Without it the Triton engine should beat Marlin at turn_4 by ~5-10%. CHANGE, three parts, only for 1024 < T <= 4096 (turn_4 T=3012 and the T=4096 test; turn_1 and the smaller tests keep Marlin): (1) in _experts, route that range to _experts_triton (it already selects EXPERT_CONFIGS' (4096, BLOCK_T=128, BLOCK_W=128, 8 warps, 3 stages) entry; _check_expert_gemms already compiles and checks it). (2) PERSISTENT GRID in _moe_nvfp4_gemm_kernel, exactly as hypothesis experts-triton-persistent-grid-first-turn specifies (grid = min(NUM_SMS * PROGS_PER_SM, static tiles) with PROGS_PER_SM = 2 for BLOCK_T 128 if the compiled kernel's smem and regs allow two CTAs per SM, else 1; loop over tile = pid, pid + num_programs, ... < cdiv(n_padded, BLOCK_T) * (N // BLOCK_W); master's GROUP_SIZE_T mapping on tile; if/else instead of return). This changes the kernel for first_turn too, so first_turn will also move. (3) LOCAL-ROUTES ALIGN on the _experts_triton path, exactly as hypothesis moe-align-local-routes-parallel-port specifies (muse_moe_w4a16's _align_parallel_kernel, local_start loaded from data['local_start'] on device, local_count 32, block = cfg['BLOCK_T']), replacing _local_expert_ids + moe_align_block_size on that path; kernel_fn passes topk_ids and data['local_start'] down; the Marlin path keeps master's remap and align. silu_and_mul and fused_post_expert_norm_weighted_sum are unchanged (they receive the align's local_ids). prepare() gates, one log line each, any failure keeps master's routing (Marlin at T <= 4096) for the run and logs '[turn4-triton] OFF: <reason>', else 'ON': align self-check vs moe_align_block_size per-expert route sets at T=3012 and 16384; NaN-poison check that fused_post_expert_norm_weighted_sum ignores remote slots; _check_expert_gemms; and _check_experts_e2e run at T=3012 as well as 8192 (Triton vs Marlin, rel L2 <= 2e-2, worst row <= 5x).

## Expected

- `*load=turn_4*` faster: experts 2131-2300 us (Marlin) -> est. 1.9-2.05 ms (the measured 2.1-2.3 ms Triton turn_4 time minus ~150-250 us of dead CTAs and remote zero-stores, minus SGLang align ~25 us -> ~10 us): -4 to -9% on turn_4 shapes, ~+1.5-3% on the headline
- `*load=first_turn*` faster: the persistent grid and local align also apply at T > 4096 (see experts-triton-persistent-grid-first-turn): -3 to -5%
- `*load=turn_1*` neutral: T=753 keeps Marlin

## Risk

Overlap: experts-triton-packed-dequant-turn4 (queued stage 6) also moves turn_4 to the Triton engine, but with the packed asm dequant that measured flat; this one keeps master's dequant and removes the dead work instead. experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) plans turn_4 for its stage 2. Parts 2-3 duplicate experts-triton-persistent-grid-first-turn and moe-align-local-routes-parallel-port; read the turn_4 _moe_nvfp4_gemm_kernel row against Marlin's 2131-2300 us, and the first_turn rows against those two runs. Risk high: Triton turn_4 times varied 2100-2318 us between runs, so the margin over Marlin is the dead-work estimate itself; if the kernel row lands at Marlin's time this is a wash on turn_4.
