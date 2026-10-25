---
problem: muse_prefill_layer
slug: experts-turn1-ssdq-s2-m64-tile
created: 2026-10-25T054848Z
base: master
area: "w13 + w2 expert GEMMs at T <= 1024 (turn_1 T=753): ssdq with a 64-route M tile, one consumer warpgroup per tile, dequant-once per weight stage"
change: >
  Stage 2 of the ssdq line, built on the branch of experts-turn4-ssdq-s1b-dataptr-symbol-fix
  whatever its score. At turn_1 the Marlin pair is 736-769 us for ~753 local routes (~24 per
  expert) against a 189 us byte floor (~905 MB of NVFP4 weights and scales): Marlin streams at
  ~1.2 TB/s. At ~24 routes per expert the shape is weight-bandwidth bound, so the only
  requirements are: read each packed weight byte once, dequantize it once, and keep enough bytes
  in flight. ssdq's producer already dequantizes each (expert, N tile, K step) once into smem; a
  128-route M tile would be 81% padding, so: (1) add a BLOCK_M=64 instantiation (one consumer
  warpgroup, m64n256k16, 128 fp32 accumulator regs) with an align at block 64
  (_align_parallel(local_ids, 0, LOCAL_EXPERTS, 64)); (2) raise the raw-weight ring to as many
  stages as smem allows (print bytes in flight per SM; >= 64 KB/SM is needed to approach 4 TB/s)
  since this regime is producer/HBM bound; (3) grid = 32 experts x N tiles (w13 32, w2 16 per
  expert at 256 columns), print the per-CTA producer-wait vs consumer-wait counters to show it is
  load-bound, not dequant-bound. In _experts_marlin route 200 < T <= 1024 here when
  state['ssdq_t1_ok']. Gate shape T=753 soft routing vs master's _align_parallel + Marlin w13 +
  _silu_mul_sorted + Marlin w2. RULES (all stages): kernel_fn stays CUDA-graph capturable with no
  host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host
  control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on
  shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md:
  NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no
  per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations,
  fp32 accumulation; every route of every token is computed. No child processes (in-process torch
  load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates
  them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch
  buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host
  from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port
  host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed
  PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn
  also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture
  one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays
  (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket
  shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's
  own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints
  one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set
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
    why: Marlin pair 736-769 us -> target <= 400 us (905 MB at >= 2.5 TB/s); turn_1 sliding 1101 -> ~750 us, full 1530 -> ~1180 us; floor 189 us
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: "T=3012 keeps stage 1's choice"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps Triton
risk: high
depends_on: experts-turn4-ssdq-s1b-dataptr-symbol-fix
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 200 < T <= 1024 (turn_1)
review: required
status: hold
hold_reason: "superseded: experts-turn-rega-s4-port-t4096 (gain 1.3275) beat ssdq 805 vs 1250 us at T=3012 and Marlin at T=753; rega owns 200<T<=4096 (first_turn rega extension in flight)"
---

# experts-turn1-ssdq-s2-m64-tile

## Change

Stage 2 of the ssdq line, built on the branch of experts-turn4-ssdq-s1b-dataptr-symbol-fix whatever its score. At turn_1 the Marlin pair is 736-769 us for ~753 local routes (~24 per expert) against a 189 us byte floor (~905 MB of NVFP4 weights and scales): Marlin streams at ~1.2 TB/s. At ~24 routes per expert the shape is weight-bandwidth bound, so the only requirements are: read each packed weight byte once, dequantize it once, and keep enough bytes in flight. ssdq's producer already dequantizes each (expert, N tile, K step) once into smem; a 128-route M tile would be 81% padding, so: (1) add a BLOCK_M=64 instantiation (one consumer warpgroup, m64n256k16, 128 fp32 accumulator regs) with an align at block 64 (_align_parallel(local_ids, 0, LOCAL_EXPERTS, 64)); (2) raise the raw-weight ring to as many stages as smem allows (print bytes in flight per SM; >= 64 KB/SM is needed to approach 4 TB/s) since this regime is producer/HBM bound; (3) grid = 32 experts x N tiles (w13 32, w2 16 per expert at 256 columns), print the per-CTA producer-wait vs consumer-wait counters to show it is load-bound, not dequant-bound. In _experts_marlin route 200 < T <= 1024 here when state['ssdq_t1_ok']. Gate shape T=753 soft routing vs master's _align_parallel + Marlin w13 + _silu_mul_sorted + Marlin w2. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=turn_1*` faster: Marlin pair 736-769 us -> target <= 400 us (905 MB at >= 2.5 TB/s); turn_1 sliding 1101 -> ~750 us, full 1530 -> ~1180 us; floor 189 us
- `*load=turn_4*` unchanged: T=3012 keeps stage 1's choice
- `*load=first_turn*` unchanged: T=16384 keeps Triton

## Risk

Competes with experts-turn-rega-s4-port-t4096 on turn_1; the rega engine is the favourite there (tokens on wgmma N fits ~24 routes better), this is the tokens-on-M alternative if rega's bucket-16/32 path loses its gate.

## Milestones

Rewrite stage 2 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 200 < T <= 1024 (turn_1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
