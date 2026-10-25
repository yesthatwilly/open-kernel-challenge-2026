---
problem: muse_prefill_layer
slug: experts-first-turn-ssdq-s2-large-m
created: 2026-10-25T054848Z
base: master
area: "w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ssdq dequant-producer + SS wgmma grouped GEMM to large M, replacing the Triton NVFP4 GEMM"
change: >
  Stage 2 of the ssdq line, built on the branch of experts-turn4-ssdq-s1b-dataptr-symbol-fix
  whatever its score. ssdq is the classic Hopper large-M decomposition (tokens on M, dequantized
  weights as the smem B operand, a producer warpgroup feeding two consumer warpgroups), which fits
  first_turn far better than turn_4: at T=16384 each local expert has ~512 routes, so each
  dequantized [256 x 64] weight stage feeds 4 token tiles of 128 instead of ~0.75. The Triton
  engine is 8.08-9.17 ms vs a 1670 us floor (~200 TFLOP/s) and its tuning line is closed. Change:
  (1) in _experts_triton, when state['ssdq_ft_ok'], align with _align_parallel or
  moe_align_block_size(local_ids, 128, LOCAL_EXPERTS) (whichever stage 1 uses; master uses
  SGLang's 32-expert align here because _align_parallel takes 394 us at T=16384 vs 139 us, so keep
  SGLang's), (2) make the CTA own an (expert, 256-column N tile) and loop over ALL of that
  expert's 128-route M tiles while the dequantized B stage is reused, i.e. dequantize each weight
  stage once per expert, not once per token tile (B-stationary; the producer ring then advances K
  only after all M tiles consumed it, or keep the whole [256 x K] dequantized slice when it fits:
  256 x 4096 x 2 B = 2 MB does not, so iterate K outer / M inner with 2-4 accumulator sets per
  consumer as registers allow; print the choice), (3) w13's grid is 32 experts x 32 N tiles = 1024
  CTAs, w2's 32 x 16 = 512, (4) keep act_and_mul between the GEMMs and master's zero_nonlocal_w2
  setting. Gate shape T=16384 soft routing vs master's Triton pair. RULES (all stages): kernel_fn
  stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by
  input values, no data-dependent host control flow, no prints, no try/except and no per-call
  mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached
  from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are
  dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that
  family stays held), BF16 activations, fp32 accumulation; every route of every token is computed.
  No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two
  states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device
  pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are
  either encoded per launch on the host from the tensors being passed (as __grid_constant__
  params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in
  the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only
  Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and
  _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A
  with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert,
  mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant
  reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss,
  non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path
  runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline
  (restore after); extra_cuda_cflags exactly
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
    why: Triton pair 8.08-9.17 ms -> target <= 4 ms (1650 GFLOP at >= 400 TFLOP/s on SS wgmma with dequant amortized over ~4 token tiles); sliding ctx32768 14265 -> ~10000 us; floor 1670 us
  - shapes: "*load=turn_*"
    direction: unchanged
    why: "T <= 4096 keeps stage 1's choice"
risk: high
depends_on: experts-turn4-ssdq-s1b-dataptr-symbol-fix
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) at T > 4096
review: required
status: queued
---

# experts-first-turn-ssdq-s2-large-m

## Change

Stage 2 of the ssdq line, built on the branch of experts-turn4-ssdq-s1b-dataptr-symbol-fix whatever its score. ssdq is the classic Hopper large-M decomposition (tokens on M, dequantized weights as the smem B operand, a producer warpgroup feeding two consumer warpgroups), which fits first_turn far better than turn_4: at T=16384 each local expert has ~512 routes, so each dequantized [256 x 64] weight stage feeds 4 token tiles of 128 instead of ~0.75. The Triton engine is 8.08-9.17 ms vs a 1670 us floor (~200 TFLOP/s) and its tuning line is closed. Change: (1) in _experts_triton, when state['ssdq_ft_ok'], align with _align_parallel or moe_align_block_size(local_ids, 128, LOCAL_EXPERTS) (whichever stage 1 uses; master uses SGLang's 32-expert align here because _align_parallel takes 394 us at T=16384 vs 139 us, so keep SGLang's), (2) make the CTA own an (expert, 256-column N tile) and loop over ALL of that expert's 128-route M tiles while the dequantized B stage is reused, i.e. dequantize each weight stage once per expert, not once per token tile (B-stationary; the producer ring then advances K only after all M tiles consumed it, or keep the whole [256 x K] dequantized slice when it fits: 256 x 4096 x 2 B = 2 MB does not, so iterate K outer / M inner with 2-4 accumulator sets per consumer as registers allow; print the choice), (3) w13's grid is 32 experts x 32 N tiles = 1024 CTAs, w2's 32 x 16 = 512, (4) keep act_and_mul between the GEMMs and master's zero_nonlocal_w2 setting. Gate shape T=16384 soft routing vs master's Triton pair. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=first_turn*` faster: Triton pair 8.08-9.17 ms -> target <= 4 ms (1650 GFLOP at >= 400 TFLOP/s on SS wgmma with dequant amortized over ~4 token tiles); sliding ctx32768 14265 -> ~10000 us; floor 1670 us
- `*load=turn_*` unchanged: T <= 4096 keeps stage 1's choice

## Risk

Milestones: M1 builds, correct, replay-safe at T=16384; M2 <= 0.95x Triton; M3 <= 4 ms. Competes with experts-turn-rega-s4-first-turn-n128 on the same rows; whichever lands first becomes the incumbent the other must beat.

## Milestones

Rewrite stage 2 of `_moe_nvfp4_gemm_kernel x2 (Triton w13 + w2) at T > 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
