---
problem: muse_prefill_layer
slug: experts-turn-rega-s4-port-t4096
created: 2026-10-25T054847Z
base: master
area: "w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets"
change: >
  Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at
  turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_4 (64%/43%), against
  byte/math floors of 189 us and 306 us, so the two turn loads sit at 1.01x SGLang while
  first_turn is 1.19x. Every Triton attempt lost to Marlin below first-turn sizes (LESSONS
  03:41Z). The kernel that beats Marlin at these sizes now exists and is replay-safe:
  muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch (job
  4e7f8be895a145ba). It has the SAME expert dimensions as this problem (H=4096 expert input,
  I=4096, w13 N=8192, w2 N=4096, K=4096), 168 regs, 0 spill, variant A (wgmma_wait<0> before the
  in-register dequant) proven finite in all four N buckets (16/32/64/128), and its prepare timing
  on all-local routings (the prefill regime) was, cold L2: 1024 routes wgmma 580.8 us vs
  Marlin+silu 921.6 us (0.63x), 2048 routes 699.7 vs 1278.0 us (0.55x). Port its _wg_* module
  (wg_gather into the state's xs scratch, wg_gemm w13 with the fused silu*up epilogue into the
  state's hs scratch, wg_gemm w2 into master's slot-ordered y [T, 8, 4096]) into _experts_marlin:
  when 200 < T <= 4096 and state['rega_ok'][T class] (T classes: <=1024 and 1025-4096, gated
  separately), align with moe_align_block_size(local_ids, WG_BSM=16, LOCAL_EXPERTS) exactly as s4
  does (local ids are already 0-31 / -1 here, so drop s4's expert_map gather and its
  per-(start,count) bank dict: there is one bank, w13[0..31]), run the gather and the two GEMMs,
  then master's fused_post_expert_norm_weighted_sum unchanged (slot_expert_ids=local_ids skips
  remote slots). Bucket choice per expert is s4's (by that expert's route count). The weight
  repack is s4's lossless permutation of the stored E2M1 codes and e4m3 scales, done once in
  prepare() from weights['w13'/'w2'] (not from the nibble-reordered Triton bank). Size the xs/hs
  scratch per state for T=4096 x 8 local routes worst case (2 x 268 MB per state). Expect up to
  ~3012 routes at turn_4, so N=128 tiles dominate there; at turn_1 (~753 routes, ~24 per expert)
  the 16/32 buckets dominate. Gate shapes: T=753 and T=3012, new (align + gather + w13/silu + w2)
  vs master's (_align_parallel + Marlin w13 + _silu_mul_sorted + Marlin w2). T > 4096 (first_turn)
  and every test outside 200 < T <= 4096 run master unchanged. RULES (all stages): kernel_fn stays
  CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input
  values, no data-dependent host control flow, no prints, no try/except and no per-call mutable
  Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from
  inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are
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
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 736-769 us -> ~450-500 us (s4 measured 0.63x Marlin at 1024 all-local routes, cold L2); turn_1 sliding ~1101 -> ~830 us, full ~1530 -> ~1260 us; floor 189 us
  - shapes: "*load=turn_4*"
    direction: faster
    why: Marlin pair 2115-2148 us -> ~1100-1300 us (s4 0.55x at 2048 routes; N=128 bucket); turn_4 sliding ~3319 -> ~2400 us, full ~4849 -> ~3950 us; floor 306 us
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps the Triton NVFP4 engine
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and _silu_mul_sorted at 200 < T <= 4096
review: required
status: queued
---

# experts-turn-rega-s4-port-t4096

## Change

Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_4 (64%/43%), against byte/math floors of 189 us and 306 us, so the two turn loads sit at 1.01x SGLang while first_turn is 1.19x. Every Triton attempt lost to Marlin below first-turn sizes (LESSONS 03:41Z). The kernel that beats Marlin at these sizes now exists and is replay-safe: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch (job 4e7f8be895a145ba). It has the SAME expert dimensions as this problem (H=4096 expert input, I=4096, w13 N=8192, w2 N=4096, K=4096), 168 regs, 0 spill, variant A (wgmma_wait<0> before the in-register dequant) proven finite in all four N buckets (16/32/64/128), and its prepare timing on all-local routings (the prefill regime) was, cold L2: 1024 routes wgmma 580.8 us vs Marlin+silu 921.6 us (0.63x), 2048 routes 699.7 vs 1278.0 us (0.55x). Port its _wg_* module (wg_gather into the state's xs scratch, wg_gemm w13 with the fused silu*up epilogue into the state's hs scratch, wg_gemm w2 into master's slot-ordered y [T, 8, 4096]) into _experts_marlin: when 200 < T <= 4096 and state['rega_ok'][T class] (T classes: <=1024 and 1025-4096, gated separately), align with moe_align_block_size(local_ids, WG_BSM=16, LOCAL_EXPERTS) exactly as s4 does (local ids are already 0-31 / -1 here, so drop s4's expert_map gather and its per-(start,count) bank dict: there is one bank, w13[0..31]), run the gather and the two GEMMs, then master's fused_post_expert_norm_weighted_sum unchanged (slot_expert_ids=local_ids skips remote slots). Bucket choice per expert is s4's (by that expert's route count). The weight repack is s4's lossless permutation of the stored E2M1 codes and e4m3 scales, done once in prepare() from weights['w13'/'w2'] (not from the nibble-reordered Triton bank). Size the xs/hs scratch per state for T=4096 x 8 local routes worst case (2 x 268 MB per state). Expect up to ~3012 routes at turn_4, so N=128 tiles dominate there; at turn_1 (~753 routes, ~24 per expert) the 16/32 buckets dominate. Gate shapes: T=753 and T=3012, new (align + gather + w13/silu + w2) vs master's (_align_parallel + Marlin w13 + _silu_mul_sorted + Marlin w2). T > 4096 (first_turn) and every test outside 200 < T <= 4096 run master unchanged. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=turn_1*` faster: Marlin pair 736-769 us -> ~450-500 us (s4 measured 0.63x Marlin at 1024 all-local routes, cold L2); turn_1 sliding ~1101 -> ~830 us, full ~1530 -> ~1260 us; floor 189 us
- `*load=turn_4*` faster: Marlin pair 2115-2148 us -> ~1100-1300 us (s4 0.55x at 2048 routes; N=128 bucket); turn_4 sliding ~3319 -> ~2400 us, full ~4849 -> ~3950 us; floor 306 us
- `*load=first_turn*` unchanged: T=16384 keeps the Triton NVFP4 engine

## Risk

Supersedes the held experts-turn1-cuda-wgmma-rega-port-bucket32 (its hold waited on the moe s3 harvest; s3 failed on the two-state bug and s4 fixed it with variant A in every bucket). Check the profile for the wg kernel names before reading any turn row: the moe s4 judged run never ran its kernel. Turn_1 rows drift -7% to -10% on unchanged code, so read the Marlin vs wg kernel rows.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and _silu_mul_sorted at 200 < T <= 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
