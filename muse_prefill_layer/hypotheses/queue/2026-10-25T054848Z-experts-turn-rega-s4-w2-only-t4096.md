---
problem: muse_prefill_layer
slug: experts-turn-rega-s4-w2-only-t4096
created: 2026-10-25T054848Z
base: master
area: "w2 expert GEMM only at T <= 4096 (turn_1, turn_4): s4 register-A swap-AB wgmma w2 fed by master's Marlin w13 + _silu_mul_sorted output"
change: >
  Built on master (a81d19c). The low-risk half of experts-turn-rega-s4-port-t4096, kept separate
  so a w13/silu problem in the full port cannot hide a w2 win. muse_moe_w4a16 s4
  (cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch) verified its w2 kernel FED BY MARLIN'S
  cache2 (rel-L2 3.47e-3 vs Marlin) and timed it cold at 185.4 us vs Marlin w2 310.0 us (1024
  all-local routes) and 230.2 vs 425.9 us (2048 routes): 0.54-0.60x. Port only the w2 path: in
  _experts_marlin, when 200 < T <= 4096 and state['rega_w2_ok'][T class] (<=1024 and 1025-4096,
  gated separately), keep master's _align_parallel + Marlin w13 + _silu_mul_sorted unchanged, then
  wg_gather cache2's live rows (in sorted_ids order at master's block_size_m, which the w2
  kernel's tiles must follow, or re-align at WG_BSM=16 from local_ids on the side stream under
  Marlin w13 if s4's w2 needs its own 16-row blocks) into the state's hs scratch and run s4's
  wg_gemm w2 writing master's slot-ordered y [T, 8, 4096]; fused_post_expert_norm_weighted_sum
  unchanged. Repack w2 once in prepare() (s4's lossless permutation). Gate shapes T=753 and
  T=3012: new (gather + w2) vs master's Marlin w2. RULES (all stages): kernel_fn stays CUDA-graph
  capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no
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
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin w2 is ~1/3 of the 736-769 us pair (~250 us) -> ~150 us; turn_1 shapes -80 to -100 us (7-9% sliding)
  - shapes: "*load=turn_4*"
    direction: faster
    why: Marlin w2 ~700 us -> ~400 us at ~3012 routes (s4 0.54x at 2048); turn_4 shapes -250 to -300 us (8% sliding)
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps Triton
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm (Marlin w2 only) at 200 < T <= 4096
review: required
status: queued
---

# experts-turn-rega-s4-w2-only-t4096

## Change

Built on master (a81d19c). The low-risk half of experts-turn-rega-s4-port-t4096, kept separate so a w13/silu problem in the full port cannot hide a w2 win. muse_moe_w4a16 s4 (cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch) verified its w2 kernel FED BY MARLIN'S cache2 (rel-L2 3.47e-3 vs Marlin) and timed it cold at 185.4 us vs Marlin w2 310.0 us (1024 all-local routes) and 230.2 vs 425.9 us (2048 routes): 0.54-0.60x. Port only the w2 path: in _experts_marlin, when 200 < T <= 4096 and state['rega_w2_ok'][T class] (<=1024 and 1025-4096, gated separately), keep master's _align_parallel + Marlin w13 + _silu_mul_sorted unchanged, then wg_gather cache2's live rows (in sorted_ids order at master's block_size_m, which the w2 kernel's tiles must follow, or re-align at WG_BSM=16 from local_ids on the side stream under Marlin w13 if s4's w2 needs its own 16-row blocks) into the state's hs scratch and run s4's wg_gemm w2 writing master's slot-ordered y [T, 8, 4096]; fused_post_expert_norm_weighted_sum unchanged. Repack w2 once in prepare() (s4's lossless permutation). Gate shapes T=753 and T=3012: new (gather + w2) vs master's Marlin w2. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=turn_1*` faster: Marlin w2 is ~1/3 of the 736-769 us pair (~250 us) -> ~150 us; turn_1 shapes -80 to -100 us (7-9% sliding)
- `*load=turn_4*` faster: Marlin w2 ~700 us -> ~400 us at ~3012 routes (s4 0.54x at 2048); turn_4 shapes -250 to -300 us (8% sliding)
- `*load=first_turn*` unchanged: T=16384 keeps Triton

## Risk

If experts-turn-rega-s4-port-t4096 lands with both kernels ON, retire this one; if only its w2 gate passes, this is the cleaner landing.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm (Marlin w2 only) at 200 < T <= 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
