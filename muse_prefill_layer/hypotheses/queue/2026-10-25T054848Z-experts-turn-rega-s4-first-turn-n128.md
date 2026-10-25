---
problem: muse_prefill_layer
slug: experts-turn-rega-s4-first-turn-n128
created: 2026-10-25T054848Z
base: master
area: "w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM"
change: >
  Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries
  the ported _wg_* module and its per-state scratch). First_turn is the biggest absolute gap: the
  Triton engine runs 8.08-9.17 ms per layer (37-57% of every first_turn shape; 57% of the
  weight-138 sliding ctx32768 shape) against a 1670 us math floor (~16384 local routes x 2 x 50.3
  MFLOP), ~5x, and every Triton knob is closed (packed asm, unmasked loads, persistent grid,
  BLOCK_T 256: REWRITE_READ). The s4 engine dequantizes into the register A fragment and pipelines
  it against async wgmma, which Triton's lowering does not. At T=16384 each local expert gets ~512
  routes, so: (1) in _experts_triton, when state['rega_ft_ok'], align with
  moe_align_block_size(local_ids, WG_BSM, LOCAL_EXPERTS), and give every expert ceil(routes/128)
  consecutive N=128 tiles, ordered expert-major so an expert's weight tile is re-read from L2 by
  its neighbours (print the tile count and the max tiles per expert); (2) size xs/hs scratch per
  state for 16384 x 8 local routes worst case only if memory allows (2 x ~1.07 GB per state; print
  max_memory_allocated), else for the scored all-soft case with a prepare-asserted capacity and
  Triton fallback for any T whose worst case does not fit (shape-based, never value-based); (3)
  the w13 silu*up epilogue removes act_and_mul (289 us) and cache1. If the single-CTA-per-tile
  kernel leaves SMs idle at the 128-tile granularity, try a 2-CTA-per-SM config only if regs <=
  128 (print occupancy). Gate shape T=16384 soft routing: new (align + gather + w13/silu + w2) vs
  master's (moe_align_block_size + count_and_sort + Triton w13 + act_and_mul + Triton w2). T <=
  4096 keeps whatever stage 1 chose. RULES (all stages): kernel_fn stays CUDA-graph capturable
  with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no
  data-dependent host control flow, no prints, no try/except and no per-call mutable Python state;
  dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision
  per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the
  kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16
  activations, fp32 accumulation; every route of every token is computed. No child processes
  (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup
  copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA
  maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either
  encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or
  stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed
  call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at
  bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)).
  prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in
  place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force
  every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound
  calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero
  device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set
  os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after);
  extra_cuda_cflags exactly
  ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg
  asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in
  the C++ bindings never call a templated Tensor::data_ptr<T>(); use
  reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol
  at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes /
  smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag;
  LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved
  alternations against master's exact incumbent block, empty-graph baseline subtracted, at the
  scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win,
  not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST
  lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a
  scored submit.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Triton pair + act_and_mul 8.4-9.5 ms -> ~4-5.5 ms if the engine holds s4's ~300 TFLOP/s at N=128 (2048 routes: 206 GFLOP in 700 us); sliding ctx32768 14265 -> ~10500 us; floor 1670 us"
  - shapes: "*load=turn_*"
    direction: unchanged
    why: "T <= 4096 keeps stage 1's choice"
risk: high
depends_on: experts-turn-rega-s4-port-t4096
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) and act_and_mul at T > 4096
review: required
status: queued
---

# experts-turn-rega-s4-first-turn-n128

## Change

Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is the biggest absolute gap: the Triton engine runs 8.08-9.17 ms per layer (37-57% of every first_turn shape; 57% of the weight-138 sliding ctx32768 shape) against a 1670 us math floor (~16384 local routes x 2 x 50.3 MFLOP), ~5x, and every Triton knob is closed (packed asm, unmasked loads, persistent grid, BLOCK_T 256: REWRITE_READ). The s4 engine dequantizes into the register A fragment and pipelines it against async wgmma, which Triton's lowering does not. At T=16384 each local expert gets ~512 routes, so: (1) in _experts_triton, when state['rega_ft_ok'], align with moe_align_block_size(local_ids, WG_BSM, LOCAL_EXPERTS), and give every expert ceil(routes/128) consecutive N=128 tiles, ordered expert-major so an expert's weight tile is re-read from L2 by its neighbours (print the tile count and the max tiles per expert); (2) size xs/hs scratch per state for 16384 x 8 local routes worst case only if memory allows (2 x ~1.07 GB per state; print max_memory_allocated), else for the scored all-soft case with a prepare-asserted capacity and Triton fallback for any T whose worst case does not fit (shape-based, never value-based); (3) the w13 silu*up epilogue removes act_and_mul (289 us) and cache1. If the single-CTA-per-tile kernel leaves SMs idle at the 128-tile granularity, try a 2-CTA-per-SM config only if regs <= 128 (print occupancy). Gate shape T=16384 soft routing: new (align + gather + w13/silu + w2) vs master's (moe_align_block_size + count_and_sort + Triton w13 + act_and_mul + Triton w2). T <= 4096 keeps whatever stage 1 chose. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=first_turn*` faster: Triton pair + act_and_mul 8.4-9.5 ms -> ~4-5.5 ms if the engine holds s4's ~300 TFLOP/s at N=128 (2048 routes: 206 GFLOP in 700 us); sliding ctx32768 14265 -> ~10500 us; floor 1670 us
- `*load=turn_*` unchanged: T <= 4096 keeps stage 1's choice

## Risk

Milestones: M1 correct and replay-safe at T=16384; M2 <= 0.95x Triton (gate ON); M3 <= 5 ms for the replaced block.

## Milestones

Rewrite stage 2 of `_moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) and act_and_mul at T > 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
