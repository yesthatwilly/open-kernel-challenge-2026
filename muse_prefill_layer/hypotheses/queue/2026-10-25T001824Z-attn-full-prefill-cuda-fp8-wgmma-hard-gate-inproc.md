---
problem: muse_prefill_layer
slug: attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc
created: 2026-10-25T001824Z
base: master
area: "FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 1b: fix of attn-full-prefill-cuda-fp8-exp-offload with build fix, hard timing gate, no exp offload)"
change: >
  BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its
  attention code is unchanged from the old base): start from master's submission.py and hand-port
  the attention path from done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch (branch
  hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload: _ATTN_CUDA, attn_fwd_kernel<SW>,
  attn_combine_kernel, _attn_splits, _full_attn_gate, _full_attn, _attn_prepare), WGMMA BACKEND
  ONLY. That run (job f5c30b15, 0.881) never built wgmma (same ptxas '.target sm_90' bug as the
  expert rewrite), and routed its mma.sync fallback, which was 3.07-3.21x slower than FA3 on all
  five full shapes, because its timing was print-only (REWRITE_READ_2026-10-25 #3). Changes vs
  that patch, and only these: (1) BUILD: os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately
  before load_inline (restore after), only -gencode arch=compute_90a,code=sm_90a; wrap all
  wgmma/fence/commit/wait asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that writes
  an error flag and returns; log the nvcc arch flags and compile seconds. (2) DELETE THE mma.sync
  BACKEND: build only the USE_WGMMA=1 module; there is no second CUDA kernel to fall back to; the
  only fallback is FA3 exactly as master calls it. (3) NO EXP OFFLOAD: compile attn_fwd_kernel<0>
  only (EXP_SW_NB = 0, every exponential on MUFU ex2); keep exp2_sw and its unit test out of this
  stage; sweep the fraction only in a later stage after wgmma parity is shown (the old run
  measured offload 2/8 as 3% slower than 0/8). (4) IN-PROCESS GUARDED FIRST LAUNCH (never in a
  child process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), confirm the sm_90a image
  actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags
  showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for
  attn_fwd_kernel<0> and attn_combine_kernel; anything else turns the path OFF); read back the
  error flag from (1) after the canary (non-zero -> OFF); then a tiny synthetic canary launch of
  attn_fwd_kernel<0> (both desc_swap orders) and attn_combine_kernel on synthetic FP8 caches (b1,
  tiny ctx), each followed by torch.cuda.synchronize() and a print so a fault names its kernel;
  any failure prints '[cuda full attn] OFF: <reason>' and FA3 runs. Then run the self-check cases
  in-process (b1 ctx3000 L753, b3 ctx5000 L200 outliers, b4 {0,777,5000,80000} L753, b1 ctx0
  L16384, b2 {37,1000} L211) with the existing pass rule. (5) HARD TIMING GATE in prepare(),
  replacing the print-only timing: for each full-layer shape class the judge scores (turn_1: b1
  L753 ctx~80000; turn_4: b4 L753 ctx~80000; first_turn: b1 L16384 at ctx 0, 32768, 65536),
  CUDA-graph capture both the FA3 call (+ its combine) and the new kernel (+ attn_combine_kernel
  when split) on synthetic inputs of that exact config, flush L2 between replays (write a >= 2x
  L2-size buffer), take the median of >= 20 replays, and route the new kernel for that shape class
  only if new <= 0.92 x FA3. The decision is a dict keyed by config-only fields (batch, q_len,
  page_table width bucket) stored in state and read by _full_attn_gate in kernel_fn; any shape not
  timed, and every sliding / page-64 / kv-scaled case, calls FA3 exactly as master. Log one line
  per class: '[cuda full attn] GATE <class>: FA3 X us, new Y us, ratio R -> CUDA|FA3', plus a
  final 'PATH CHOSEN' summary. Everything else in the layer (q cast, o_norm_gate, experts, router)
  is master's. The first measurement of this hypothesis must be an `okc check`, never a scored
  submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: "*layer=full*"
    direction: faster
    why: "Replaced: FA3 415.2 (turn_1 full, +7.0 combine), 1634.0 (turn_4 full), 892.2 / 4471.1 / 8006.7 us (first_turn ctx0/32768/65536); exp floors 250/1000/555/2770/4990 us (FA3 ~1.6x its floor). The gate routes a class only at <= 0.92x FA3, so each routed shape gains >= 8% of its FA3 time (~2-3% on turn_4 full and ctx65536 full) and an un-routed shape is flat by construction. Milestones: M0 build line OK + sm_90a image + canary clean + self-check ON; M1 at least one class passes the gate; M2 0.75x FA3 on every full class. A run where every class gates to FA3 is a measured result (the log shows the ratios), not a regression."
  - shapes: "*layer=sliding*"
    direction: neutral
    why: FA3 kept on sliding layers (25.7 / 61.4 / 304.4 us).
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1"
review: required
status: queued
---

# attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc

## Change

BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's submission.py and hand-port the attention path from done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch (branch hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload: _ATTN_CUDA, attn_fwd_kernel<SW>, attn_combine_kernel, _attn_splits, _full_attn_gate, _full_attn, _attn_prepare), WGMMA BACKEND ONLY. That run (job f5c30b15, 0.881) never built wgmma (same ptxas '.target sm_90' bug as the expert rewrite), and routed its mma.sync fallback, which was 3.07-3.21x slower than FA3 on all five full shapes, because its timing was print-only (REWRITE_READ_2026-10-25 #3). Changes vs that patch, and only these: (1) BUILD: os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before load_inline (restore after), only -gencode arch=compute_90a,code=sm_90a; wrap all wgmma/fence/commit/wait asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that writes an error flag and returns; log the nvcc arch flags and compile seconds. (2) DELETE THE mma.sync BACKEND: build only the USE_WGMMA=1 module; there is no second CUDA kernel to fall back to; the only fallback is FA3 exactly as master calls it. (3) NO EXP OFFLOAD: compile attn_fwd_kernel<0> only (EXP_SW_NB = 0, every exponential on MUFU ex2); keep exp2_sw and its unit test out of this stage; sweep the fraction only in a later stage after wgmma parity is shown (the old run measured offload 2/8 as 3% slower than 0/8). (4) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for attn_fwd_kernel<0> and attn_combine_kernel; anything else turns the path OFF); read back the error flag from (1) after the canary (non-zero -> OFF); then a tiny synthetic canary launch of attn_fwd_kernel<0> (both desc_swap orders) and attn_combine_kernel on synthetic FP8 caches (b1, tiny ctx), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; any failure prints '[cuda full attn] OFF: <reason>' and FA3 runs. Then run the self-check cases in-process (b1 ctx3000 L753, b3 ctx5000 L200 outliers, b4 {0,777,5000,80000} L753, b1 ctx0 L16384, b2 {37,1000} L211) with the existing pass rule. (5) HARD TIMING GATE in prepare(), replacing the print-only timing: for each full-layer shape class the judge scores (turn_1: b1 L753 ctx~80000; turn_4: b4 L753 ctx~80000; first_turn: b1 L16384 at ctx 0, 32768, 65536), CUDA-graph capture both the FA3 call (+ its combine) and the new kernel (+ attn_combine_kernel when split) on synthetic inputs of that exact config, flush L2 between replays (write a >= 2x L2-size buffer), take the median of >= 20 replays, and route the new kernel for that shape class only if new <= 0.92 x FA3. The decision is a dict keyed by config-only fields (batch, q_len, page_table width bucket) stored in state and read by _full_attn_gate in kernel_fn; any shape not timed, and every sliding / page-64 / kv-scaled case, calls FA3 exactly as master. Log one line per class: '[cuda full attn] GATE <class>: FA3 X us, new Y us, ratio R -> CUDA|FA3', plus a final 'PATH CHOSEN' summary. Everything else in the layer (q cast, o_norm_gate, experts, router) is master's.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=full*` faster: Replaced: FA3 415.2 (turn_1 full, +7.0 combine), 1634.0 (turn_4 full), 892.2 / 4471.1 / 8006.7 us (first_turn ctx0/32768/65536); exp floors 250/1000/555/2770/4990 us (FA3 ~1.6x its floor). The gate routes a class only at <= 0.92x FA3, so each routed shape gains >= 8% of its FA3 time (~2-3% on turn_4 full and ctx65536 full) and an un-routed shape is flat by construction. Milestones: M0 build line OK + sm_90a image + canary clean + self-check ON; M1 at least one class passes the gate; M2 0.75x FA3 on every full class. A run where every class gates to FA3 is a measured result (the log shows the ratios), not a regression.
- `*layer=sliding*` neutral: FA3 kept on sliding layers (25.7 / 61.4 / 304.4 us).

## Risk

supersedes attn-full-prefill-cuda-fp8-wgmma-hard-gate, held for a child-process self-check

## Milestones

Rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
