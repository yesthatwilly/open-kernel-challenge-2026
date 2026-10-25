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
review: proceed
status: running
review_at: 2026-10-25T085139Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T085139Z
base_sha: de61d5aeb2cabe210254a89cbdd05b3f99ff7b1c
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-turn4-ssdq-s1b-dataptr-symbol-fix
worktree: .worktrees/muse_prefill_layer/experts-turn4-ssdq-s1b-dataptr-symbol-fix
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T085139Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a legitimate re-run: stage 1 never executed because of an import-time undefined symbol (data_ptr<long long>), and the fix is mechanical and correct. The design is copy-safe for the judge's two-state clone (TMA map encoded per launch from the passed tensor, dispatch gated only on the state boolean ssdq_ok), so run it as one okc check with no companions. Two spec-vs-code contradictions need fixing at build time: setmaxnreg 232/160 cannot fit in the register file, and the gate ratio is 0.92 in code but 0.95 in the spec.

### Justification
A: The mechanism is plausible. At turn_4 the Marlin pair costs 2148/2115 us (64.5% / 42.8% of the shape) against a 306 us math floor. At T=3012 there are about 94 local routes per expert, so one 128-route M tile per expert dequantizes each weight tile about once. That moves the dequant from per-use in registers (Marlin) to once per (expert, N tile, K step), and SS wgmma m64n256k16 then runs at near tensor rate. The <=1200 us target (about 4x the floor) is reachable on paper. The expected directions are consistent with the code: the path serves only 1024 < T <= 4096, so turn_1 (T=753) stays on Marlin and first_turn (T=16384) stays on Triton. The diagnosis is right. libtorch exports data_ptr<int64_t> (long on LP64), not long long, so the .so linked but failed at dlopen (patch line 530). data_ptr<int> and data_ptr<float> are exported, so replacing them is uniformity only, which is harmless. Weakest link: the kernel has never executed. This run is the first real test of the mbarrier/TMA ring, the producer dequant into a 128B-swizzled stage, and the wgmma descriptors. The most likely outcome is an OFF at the canary or self-check, which re-measures master (neutral about 1.07). That is still readable, because the log prints the error word, the first non-finite tensor and the per-case rel-L2. Second weak point: the occupancy and parallelism shape. __launch_bounds__(THREADS,1) gives 1 CTA/SM, with 32 local experts x (N/256) tiles per GEMM, about 27% M-padding (94 -> 128) and a single producer WG doing all dequant. If the counters show a large consumer wait, the bottleneck is the producer, and fixing it belongs in stage 2.

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn: defines MARLIN_MAX_TOKENS=4096 and the shape gate that the ssdq branch nests inside. If that threshold changes, the T range served changes. It does not confound turn_4 kernel time, which is the stage's milestone.
- stack-tonight-five-prefill: provides _align_parallel and the align_par_ok flag (ssdq setup goes OFF if parallel align is off), plus the zero_nonlocal_w2 flag that _experts_ssdq honours. The ssdq path reuses the parallel align, so the gate's Marlin-vs-ssdq comparison shares that cost and attribution stays clean. If align_par_ok is False in the judge run, the result is a silent master re-measure; the harvest must check the OFF line.
- sglang-pipeline-parity and experts-triton-w4a16-first-turn-blockt256: they touch different shapes or launch paths and do not confound the turn_4 expert kernel time.

### Interference and A/B needs
No companions. This is a kind: rewrite stage judged on the replaced kernels' own time in the profile (Marlin 2148/2115 us at turn_4 vs the ssdq kernel rows), not on the headline. The path is self-gated, with an in-prepare cold-L2 A/B against master's exact Marlin block. Every failure mode prints a distinct OFF line, so both a gain and a loss can be read from the log plus the profile without a second run. With 8 prefill checks left, a companion would buy nothing that the gate medians and the producer/consumer counters do not already print.

### Open questions
- After the fix, grep the built C++ string for 'data_ptr<' (the planned prepare audit). Also confirm that any at::Tensor accessor with a template (e.g. size<>, item<>) is absent, since those could hit the same long/long long trap.
- If the canary passes but the self-check fails, is the bound max(2x Marlin rel-L2, 8e-3) tight enough to separate a swizzle or descriptor bug from fp32-accumulation noise? The per-case print gives both numbers; read them before minting stage 2.
- Harvest: confirm the profile shows the ssdq kernel row at turn_4 and that Marlin is absent there. A PATH CHOSEN=ssdq line alone is not proof (moe s4 lesson).
- SSDQ_BUDGET_S=170 s plus about 31 s of compile: does prepare() on current master (with the stack-tonight-five checks) leave room, or will the run go OFF for budget before it builds?

