---
problem: muse_prefill_layer
slug: experts-triton-persistent-grid-first-turn
created: 2026-10-24T184832Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)"
change: >
  Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change;
  the per-tile math (pointer setup, K loop, in-register dequant, tl.dot, epilogue),
  EXPERT_CONFIGS, moe_align_block_size, silu_and_mul, the weighted sum, ZERO_NONLOCAL_W2 and the
  Marlin path at T <= 4096 all stay as they are. MECHANISM: today the grid is cdiv(EM, BLOCK_T) *
  (N // BLOCK_W) with EM = sorted_ids.shape[0], the static capacity of
  moe_align_block_size(local_ids, 256, 256) (about 131072 + 257*255 slots at T=16384, ~768 token
  tiles). Only ~80 token tiles hold local routes; ~448 are -1 tiles (the remote routes, which
  SGLang's align buckets into blocks of their own: w13 returns, w2 zero-stores a 256x128 tile) and
  ~240 lie past n_padded (return). A BLOCK_T 256 CTA holds ~145 KB of smem and 256 threads at ~255
  regs, so one CTA fits per SM, and ~44K dead w13 CTAs and ~22K w2 CTAs run one after another on
  each SM at ~1 us each (two dependent global loads, then exit): an estimated 0.4-0.6 ms of the
  8.1-8.9 ms expert row. CHANGE: make the kernel persistent. Launcher: grid = (min(NUM_SMS,
  cdiv(EM, BLOCK_T) * (N // BLOCK_W)),) with NUM_SMS =
  torch.cuda.get_device_properties(device).multi_processor_count (a host int, read once in
  prepare() or at import; it is a hardware constant, not input-derived). Kernel: load n_padded
  once at the top; num_pid_t = tl.cdiv(n_padded, BLOCK_T) (runtime: only tiles below n_padded,
  which includes the -1 tiles); num_tiles = num_pid_t * num_pid_w; then `for tile in
  range(tl.program_id(0), num_tiles, tl.num_programs(0)):` and inside it compute group_id,
  first_pid_t, group_size_t, pid_t, pid_w from `tile` with exactly master's GROUP_SIZE_T formulas
  (using the runtime num_pid_t); load off_expert; `if off_expert == -1:` master's ZERO_NONLOCAL
  store (unchanged), `else:` master's body unchanged. Triton has no `return`/`continue` inside a
  loop, so use the if/else; the inner K loop stays the innermost loop so Triton still pipelines it
  with the config's num_stages. Keep do_not_specialize on num_valid_tokens and EM.
  _check_expert_gemms runs unchanged (its -1 block and past-n_padded block assertions cover the
  new loop) and its log line adds `persistent grid=<g>`; if any variant now spills (n_spills > 0
  where master had 0), print it so the run can be read.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel x2 (8149-8873 us on the four first_turn shapes, 37-58% of each) loses its ~66K dead CTAs: target 7.6-8.0 ms, i.e. -3 to -5% on first_turn shapes (sliding ctx32768, weight 138, is the heaviest row of the problem); ~+1-1.5% on the headline geomean"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T <= 4096 runs Marlin; the Triton kernel only runs in prepare checks there
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-persistent-grid-first-turn

## Change

Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (pointer setup, K loop, in-register dequant, tl.dot, epilogue), EXPERT_CONFIGS, moe_align_block_size, silu_and_mul, the weighted sum, ZERO_NONLOCAL_W2 and the Marlin path at T <= 4096 all stay as they are. MECHANISM: today the grid is cdiv(EM, BLOCK_T) * (N // BLOCK_W) with EM = sorted_ids.shape[0], the static capacity of moe_align_block_size(local_ids, 256, 256) (about 131072 + 257*255 slots at T=16384, ~768 token tiles). Only ~80 token tiles hold local routes; ~448 are -1 tiles (the remote routes, which SGLang's align buckets into blocks of their own: w13 returns, w2 zero-stores a 256x128 tile) and ~240 lie past n_padded (return). A BLOCK_T 256 CTA holds ~145 KB of smem and 256 threads at ~255 regs, so one CTA fits per SM, and ~44K dead w13 CTAs and ~22K w2 CTAs run one after another on each SM at ~1 us each (two dependent global loads, then exit): an estimated 0.4-0.6 ms of the 8.1-8.9 ms expert row. CHANGE: make the kernel persistent. Launcher: grid = (min(NUM_SMS, cdiv(EM, BLOCK_T) * (N // BLOCK_W)),) with NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count (a host int, read once in prepare() or at import; it is a hardware constant, not input-derived). Kernel: load n_padded once at the top; num_pid_t = tl.cdiv(n_padded, BLOCK_T) (runtime: only tiles below n_padded, which includes the -1 tiles); num_tiles = num_pid_t * num_pid_w; then `for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):` and inside it compute group_id, first_pid_t, group_size_t, pid_t, pid_w from `tile` with exactly master's GROUP_SIZE_T formulas (using the runtime num_pid_t); load off_expert; `if off_expert == -1:` master's ZERO_NONLOCAL store (unchanged), `else:` master's body unchanged. Triton has no `return`/`continue` inside a loop, so use the if/else; the inner K loop stays the innermost loop so Triton still pipelines it with the config's num_stages. Keep do_not_specialize on num_valid_tokens and EM. _check_expert_gemms runs unchanged (its -1 block and past-n_padded block assertions cover the new loop) and its log line adds `persistent grid=<g>`; if any variant now spills (n_spills > 0 where master had 0), print it so the run can be read.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel x2 (8149-8873 us on the four first_turn shapes, 37-58% of each) loses its ~66K dead CTAs: target 7.6-8.0 ms, i.e. -3 to -5% on first_turn shapes (sliding ctx32768, weight 138, is the heaviest row of the problem); ~+1-1.5% on the headline geomean
- `*load=turn_*` neutral: T <= 4096 runs Marlin; the Triton kernel only runs in prepare checks there

## Risk

Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) replaces this kernel at first_turn with a persistent CUDA kernel that skips the same dead items; this is the cheap Triton-side hedge and it measures the dead-CTA cost that design assumes away. If the CUDA kernel lands and wins, this is moot. Risks: dead CTAs may retire faster than ~1 us (then ~1-2%, judge on the kernel row); the persistent loop may raise register pressure or change the pipeliner's schedule (read n_regs/n_spills in the log); tail imbalance with 132 programs over ~5120 w13 tiles is under one tile per program.
