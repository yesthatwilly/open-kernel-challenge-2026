---
problem: muse_prefill_layer
slug: experts-turn4-ssdq-s1b-dataptr-symbol-fix
created: 2026-10-25T054848Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a dequant-producer warpgroup + SS wgmma grouped GEMM, stage 1b: rebuild with the undefined-symbol fix"
change: >
  Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714),
  which NEVER RAN: the log reads '[ssdq] load_inline failed after 30.8s: ...
  muse_prefill_ssdq_v1.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIxEEPT_v' then 'OFF:
  build failed (PATH CHOSEN = marlin at every T)', and its profile shows Marlin 2141.9/2066.2 us
  at turn_4, i.e. master re-measured. The sm_90a build flags themselves worked
  ('-gencode=arch=compute_90a,code=sm_90a'). Cause: the binding calls prof.data_ptr<long long>()
  (patch line 530); libtorch instantiates data_ptr<int64_t> (= long on Linux), not long long, so
  the .so links but fails to import. Fix: start from that hypothesis's implementation commit
  (44ca9c2, patch
  muse_prefill_layer/hypotheses/done/neutral/experts-turn4-cuda-dequant-producer-ss-wgmma.patch)
  rebased onto current master, and replace EVERY templated data_ptr<T>() in the C++ source with
  reinterpret_cast<T*>(t.data_ptr()) (the long long one, and the int/float ones too, for
  uniformity). Before the 30 s build, run a cheap source audit in prepare (regex for 'data_ptr<'
  in the C++ string; OFF with the offending line if any remain). Keep the design unchanged: tokens
  on the wgmma M side, CTA tile 128 routes x 256 columns, BLOCK_K 64, one expert per tile via
  _align_parallel(local_ids, 0, LOCAL_EXPERTS, 128), a producer warpgroup that TMA/cp.async loads
  packed E2M1 + e4m3 scales into a 4-stage raw ring and dequantizes once per (expert, N tile, K
  step) into a 128B-swizzled bf16 smem stage, two consumer warpgroups on SS wgmma m64n256k16 with
  setmaxnreg 232/160, _silu_mul_sorted between the GEMMs, the per-CTA producer-wait vs
  consumer-wait cycle counters printed once, and the existing two-state replay check
  (_ssdq_replay_check) and timing gate. The TMA map is encoded per launch over the tensor actually
  passed (already per call, not global): keep it that way. Gate shape T=3012. RULES (all stages):
  kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation
  sized by input values, no data-dependent host control flow, no prints, no try/except and no
  per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set;
  nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global
  memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert
  weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token
  is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge
  prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may
  hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA
  descriptors are either encoded per launch on the host from the tensors being passed (as
  __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can
  silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its
  judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in
  state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state,
  then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed
  4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each
  output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and
  worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>'
  line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just
  before load_inline (restore after); extra_cuda_cflags exactly
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
  - shapes: "*load=turn_4*"
    direction: faster
    why: Marlin pair 2115-2148 us vs a 306 us math floor; dequant once per weight tile (~1.3x per weight at ~94 routes/expert) on SS wgmma should reach <= 1200 us (M1 build+correct, M2 <= 0.95x Marlin, M3 <= 1000 us); turn_4 sliding 3319 -> ~2350 us
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: T=753 <= 1024 keeps Marlin
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps Triton
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)
review: required
status: queued
---

# experts-turn4-ssdq-s1b-dataptr-symbol-fix

## Change

Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714), which NEVER RAN: the log reads '[ssdq] load_inline failed after 30.8s: ... muse_prefill_ssdq_v1.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIxEEPT_v' then 'OFF: build failed (PATH CHOSEN = marlin at every T)', and its profile shows Marlin 2141.9/2066.2 us at turn_4, i.e. master re-measured. The sm_90a build flags themselves worked ('-gencode=arch=compute_90a,code=sm_90a'). Cause: the binding calls prof.data_ptr<long long>() (patch line 530); libtorch instantiates data_ptr<int64_t> (= long on Linux), not long long, so the .so links but fails to import. Fix: start from that hypothesis's implementation commit (44ca9c2, patch muse_prefill_layer/hypotheses/done/neutral/experts-turn4-cuda-dequant-producer-ss-wgmma.patch) rebased onto current master, and replace EVERY templated data_ptr<T>() in the C++ source with reinterpret_cast<T*>(t.data_ptr()) (the long long one, and the int/float ones too, for uniformity). Before the 30 s build, run a cheap source audit in prepare (regex for 'data_ptr<' in the C++ string; OFF with the offending line if any remain). Keep the design unchanged: tokens on the wgmma M side, CTA tile 128 routes x 256 columns, BLOCK_K 64, one expert per tile via _align_parallel(local_ids, 0, LOCAL_EXPERTS, 128), a producer warpgroup that TMA/cp.async loads packed E2M1 + e4m3 scales into a 4-stage raw ring and dequantizes once per (expert, N tile, K step) into a 128B-swizzled bf16 smem stage, two consumer warpgroups on SS wgmma m64n256k16 with setmaxnreg 232/160, _silu_mul_sorted between the GEMMs, the per-CTA producer-wait vs consumer-wait cycle counters printed once, and the existing two-state replay check (_ssdq_replay_check) and timing gate. The TMA map is encoded per launch over the tensor actually passed (already per call, not global): keep it that way. Gate shape T=3012. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=turn_4*` faster: Marlin pair 2115-2148 us vs a 306 us math floor; dequant once per weight tile (~1.3x per weight at ~94 routes/expert) on SS wgmma should reach <= 1200 us (M1 build+correct, M2 <= 0.95x Marlin, M3 <= 1000 us); turn_4 sliding 3319 -> ~2350 us
- `*load=turn_1*` unchanged: T=753 <= 1024 keeps Marlin
- `*load=first_turn*` unchanged: T=16384 keeps Triton

## Risk

The harvest must grep the log for '[ssdq] build' and the PATH CHOSEN line and the profile for the ssdq kernel row before reading turn_4.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
