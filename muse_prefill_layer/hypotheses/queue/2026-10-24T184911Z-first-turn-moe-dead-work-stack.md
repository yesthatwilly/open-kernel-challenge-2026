---
problem: muse_prefill_layer
slug: first-turn-moe-dead-work-stack
created: 2026-10-24T184911Z
base: master
area: MoE dispatch + silu-mul + Triton NVFP4 grid at T > 4096 (first_turn dead-work removal, combined)
change: >
  Built on master (635cfee). Touches only the T > 4096 path (_experts_triton and the Triton
  kernel's launch); _experts_marlin (T <= 4096) is byte-identical. Three parts, all aimed at work
  the first_turn expert block does on rows that are not this GPU's: (1) PERSISTENT GRID, exactly
  as hypothesis experts-triton-persistent-grid-first-turn specifies (grid = min(NUM_SMS, static
  tiles); loop `for tile in range(pid, num_tiles, num_programs)` with num_tiles = cdiv(n_padded,
  BLOCK_T) * (N // BLOCK_W) read on device; master's GROUP_SIZE_T mapping on `tile`; if/else
  instead of return). (2) LOCAL-ROUTES ALIGN in _experts_triton only, exactly as hypothesis
  moe-align-local-routes-parallel-port specifies (copy muse_moe_w4a16's
  _align_parallel_kernel/_align_parallel, local_start read from data['local_start'] on device,
  local_count 32, block = cfg['BLOCK_T'] = 256): it replaces _local_expert_ids +
  moe_align_block_size on this path, so there are no -1 blocks below n_padded and w2 writes
  nothing for remote slots (ZERO_NONLOCAL_W2 then never fires; leave the flag as is). kernel_fn
  must pass topk_ids and data['local_start'] down so this path gets them; the Marlin path keeps
  master's remap and align. (3) SORTED SILU: replace silu_and_mul(cache1, cache2,
  expert_ids=local_ids.view(-1), expert_step=1) in _experts_triton with a new Triton kernel
  _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots, N=4096, BLOCK_R=16,
  BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load
  n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange; s = load(sorted_ids
  + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots (with part 2 every valid s
  is a local slot); cols = pid1*BLOCK_N + arange; g = c1[s, cols], u = c1[s, N + cols] (bf16), out
  = (g_f32 / (1 + exp(-g_f32))) * u_f32 rounded once to bf16, store c2[s, cols] under valid. It
  reads ~T local rows (16384 x 24 KB) instead of launching over all T*8 slots. prepare() gates,
  each one log line, any failure turns the whole stack off for the run (master's _experts_triton
  runs; log '[ft-stack] OFF: <reason>', else '[ft-stack] ON'): the align self-check vs
  moe_align_block_size per-expert route sets at T=8192 and 16384; the NaN-poison check
  (fused_post_expert_norm_weighted_sum with NaN in remote slots equals the zero-filled call); the
  silu kernel vs silu_and_mul on the local rows of a synthetic cache1 (max diff <= 1 bf16 ulp);
  _check_expert_gemms (now also exercising the persistent loop) and _check_experts_e2e adapted to
  the new align (Triton stack vs Marlin, rel L2 <= 2e-2).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Removes, per first_turn shape: ~66K dead Triton CTAs (est. 0.3-0.6 ms), the w2 zero-store of ~940 MB (~0.2 ms), act_and_mul 284-290 us -> ~90 us, count_and_sort + moe_align ~137 us -> ~60 us: ~0.8-1.1 ms of 14.3-22.3 ms, i.e. -5 to -8% on the first_turn shapes, ~+2-2.5% on the headline geomean"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T <= 4096 path is byte-identical
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# first-turn-moe-dead-work-stack

## Change

Built on master (635cfee). Touches only the T > 4096 path (_experts_triton and the Triton kernel's launch); _experts_marlin (T <= 4096) is byte-identical. Three parts, all aimed at work the first_turn expert block does on rows that are not this GPU's: (1) PERSISTENT GRID, exactly as hypothesis experts-triton-persistent-grid-first-turn specifies (grid = min(NUM_SMS, static tiles); loop `for tile in range(pid, num_tiles, num_programs)` with num_tiles = cdiv(n_padded, BLOCK_T) * (N // BLOCK_W) read on device; master's GROUP_SIZE_T mapping on `tile`; if/else instead of return). (2) LOCAL-ROUTES ALIGN in _experts_triton only, exactly as hypothesis moe-align-local-routes-parallel-port specifies (copy muse_moe_w4a16's _align_parallel_kernel/_align_parallel, local_start read from data['local_start'] on device, local_count 32, block = cfg['BLOCK_T'] = 256): it replaces _local_expert_ids + moe_align_block_size on this path, so there are no -1 blocks below n_padded and w2 writes nothing for remote slots (ZERO_NONLOCAL_W2 then never fires; leave the flag as is). kernel_fn must pass topk_ids and data['local_start'] down so this path gets them; the Marlin path keeps master's remap and align. (3) SORTED SILU: replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) in _experts_triton with a new Triton kernel _silu_mul_sorted_kernel(c1, c2, sorted_ids, n_padded, num_slots, N=4096, BLOCK_R=16, BLOCK_N=512), grid (cdiv(sorted_ids.shape[0], BLOCK_R), N // BLOCK_N), num_warps 4: load n_padded; if pid0*BLOCK_R >= n_padded return; rows = pid0*BLOCK_R + arange; s = load(sorted_ids + rows, mask=rows < n_padded, other=num_slots); valid = s < num_slots (with part 2 every valid s is a local slot); cols = pid1*BLOCK_N + arange; g = c1[s, cols], u = c1[s, N + cols] (bf16), out = (g_f32 / (1 + exp(-g_f32))) * u_f32 rounded once to bf16, store c2[s, cols] under valid. It reads ~T local rows (16384 x 24 KB) instead of launching over all T*8 slots. prepare() gates, each one log line, any failure turns the whole stack off for the run (master's _experts_triton runs; log '[ft-stack] OFF: <reason>', else '[ft-stack] ON'): the align self-check vs moe_align_block_size per-expert route sets at T=8192 and 16384; the NaN-poison check (fused_post_expert_norm_weighted_sum with NaN in remote slots equals the zero-filled call); the silu kernel vs silu_and_mul on the local rows of a synthetic cache1 (max diff <= 1 bf16 ulp); _check_expert_gemms (now also exercising the persistent loop) and _check_experts_e2e adapted to the new align (Triton stack vs Marlin, rel L2 <= 2e-2).

## Expected

- `*load=first_turn*` faster: Removes, per first_turn shape: ~66K dead Triton CTAs (est. 0.3-0.6 ms), the w2 zero-store of ~940 MB (~0.2 ms), act_and_mul 284-290 us -> ~90 us, count_and_sort + moe_align ~137 us -> ~60 us: ~0.8-1.1 ms of 14.3-22.3 ms, i.e. -5 to -8% on the first_turn shapes, ~+2-2.5% on the headline geomean
- `*load=turn_*` neutral: T <= 4096 path is byte-identical

## Risk

Bold combined run of experts-triton-persistent-grid-first-turn, moe-align-local-routes-parallel-port (its first_turn half) and a sorted-row silu at first_turn; each part's effect is readable on its own kernel row (_moe_nvfp4_gemm_kernel, the align rows, act_and_mul vs _silu_mul_sorted_kernel). The silu part is the held silu-mul-triton-local-rows-marlin-turn mechanism folded into a larger change, as its hold asked, but at T > 4096 only (where act_and_mul is 285 us, not 14-53). Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) replaces the Triton GEMM at first_turn and plans to fuse silu in its stage 4; parts 2-3 compose with it, part 1 is moot if it lands. Risk high only because three parts must all pass their gates; each falls back to master.
