---
problem: muse_prefill_layer
slug: attn-full-prefill-cuda-fp8-wgmma-hard-gate
created: 2026-10-25T001301Z
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
  measured offload 2/8 as 3% slower than 0/8). (4) FIRST LAUNCH ISOLATED (LESSONS 2026-10-25): run
  the first launch and the correctness self-check of the wgmma kernel (both desc_swap orders) in a
  child process (multiprocessing spawn, synthetic FP8 caches, bounded timeout); a crash, hang or
  illegal instruction prints '[cuda full attn] OFF: <reason>' and FA3 runs. Then repeat the
  self-check cases in-process (b1 ctx3000 L753, b3 ctx5000 L200 outliers, b4 {0,777,5000,80000}
  L753, b1 ctx0 L16384, b2 {37,1000} L211) with the existing pass rule. (5) HARD TIMING GATE in
  prepare(), replacing the print-only timing: for each full-layer shape class the judge scores
  (turn_1: b1 L753 ctx~80000; turn_4: b4 L753 ctx~80000; first_turn: b1 L16384 at ctx 0, 32768,
  65536), CUDA-graph capture both the FA3 call (+ its combine) and the new kernel (+
  attn_combine_kernel when split) on synthetic inputs of that exact config, flush L2 between
  replays (write a >= 2x L2-size buffer), take the median of >= 20 replays, and route the new
  kernel for that shape class only if new <= 0.92 x FA3. The decision is a dict keyed by
  config-only fields (batch, q_len, page_table width bucket) stored in state and read by
  _full_attn_gate in kernel_fn; any shape not timed, and every sliding / page-64 / kv-scaled case,
  calls FA3 exactly as master. Log one line per class: '[cuda full attn] GATE <class>: FA3 X us,
  new Y us, ratio R -> CUDA|FA3', plus a final 'PATH CHOSEN' summary. Everything else in the layer
  (q cast, o_norm_gate, experts, router) is master's.
expected:
  - shapes: "*layer=full*"
    direction: faster
    why: "Replaced: FA3 415.2 (turn_1 full, +7.0 combine), 1634.0 (turn_4 full), 892.2 / 4471.1 / 8006.7 us (first_turn ctx0/32768/65536); exp floors 250/1000/555/2770/4990 us (FA3 ~1.6x its floor). The gate routes a class only at <= 0.92x FA3, so each routed shape gains >= 8% of its FA3 time (~2-3% on turn_4 full and ctx65536 full) and an un-routed shape is flat by construction. Milestones: M0 build line OK + child clean + self-check ON; M1 at least one class passes the gate; M2 0.75x FA3 on every full class. A run where every class gates to FA3 is a measured result (the log shows the ratios), not a regression."
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
review: hold
status: hold
review_at: 2026-10-25T001525Z
review_companions: null
hold_reason: "adversarial review: The rewrite's target and gate are sound, but sub-change (4) runs the first launch in a multiprocessing-spawn child process, and TASK.md says a submission that '"
---

# attn-full-prefill-cuda-fp8-wgmma-hard-gate

## Change

BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's submission.py and hand-port the attention path from done/regression/attn-full-prefill-cuda-fp8-exp-offload.patch (branch hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload: _ATTN_CUDA, attn_fwd_kernel<SW>, attn_combine_kernel, _attn_splits, _full_attn_gate, _full_attn, _attn_prepare), WGMMA BACKEND ONLY. That run (job f5c30b15, 0.881) never built wgmma (same ptxas '.target sm_90' bug as the expert rewrite), and routed its mma.sync fallback, which was 3.07-3.21x slower than FA3 on all five full shapes, because its timing was print-only (REWRITE_READ_2026-10-25 #3). Changes vs that patch, and only these: (1) BUILD: os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before load_inline (restore after), only -gencode arch=compute_90a,code=sm_90a; wrap all wgmma/fence/commit/wait asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that writes an error flag and returns; log the nvcc arch flags and compile seconds. (2) DELETE THE mma.sync BACKEND: build only the USE_WGMMA=1 module; there is no second CUDA kernel to fall back to; the only fallback is FA3 exactly as master calls it. (3) NO EXP OFFLOAD: compile attn_fwd_kernel<0> only (EXP_SW_NB = 0, every exponential on MUFU ex2); keep exp2_sw and its unit test out of this stage; sweep the fraction only in a later stage after wgmma parity is shown (the old run measured offload 2/8 as 3% slower than 0/8). (4) FIRST LAUNCH ISOLATED (LESSONS 2026-10-25): run the first launch and the correctness self-check of the wgmma kernel (both desc_swap orders) in a child process (multiprocessing spawn, synthetic FP8 caches, bounded timeout); a crash, hang or illegal instruction prints '[cuda full attn] OFF: <reason>' and FA3 runs. Then repeat the self-check cases in-process (b1 ctx3000 L753, b3 ctx5000 L200 outliers, b4 {0,777,5000,80000} L753, b1 ctx0 L16384, b2 {37,1000} L211) with the existing pass rule. (5) HARD TIMING GATE in prepare(), replacing the print-only timing: for each full-layer shape class the judge scores (turn_1: b1 L753 ctx~80000; turn_4: b4 L753 ctx~80000; first_turn: b1 L16384 at ctx 0, 32768, 65536), CUDA-graph capture both the FA3 call (+ its combine) and the new kernel (+ attn_combine_kernel when split) on synthetic inputs of that exact config, flush L2 between replays (write a >= 2x L2-size buffer), take the median of >= 20 replays, and route the new kernel for that shape class only if new <= 0.92 x FA3. The decision is a dict keyed by config-only fields (batch, q_len, page_table width bucket) stored in state and read by _full_attn_gate in kernel_fn; any shape not timed, and every sliding / page-64 / kv-scaled case, calls FA3 exactly as master. Log one line per class: '[cuda full attn] GATE <class>: FA3 X us, new Y us, ratio R -> CUDA|FA3', plus a final 'PATH CHOSEN' summary. Everything else in the layer (q cast, o_norm_gate, experts, router) is master's.

## Expected

- `*layer=full*` faster: Replaced: FA3 415.2 (turn_1 full, +7.0 combine), 1634.0 (turn_4 full), 892.2 / 4471.1 / 8006.7 us (first_turn ctx0/32768/65536); exp floors 250/1000/555/2770/4990 us (FA3 ~1.6x its floor). The gate routes a class only at <= 0.92x FA3, so each routed shape gains >= 8% of its FA3 time (~2-3% on turn_4 full and ctx65536 full) and an un-routed shape is flat by construction. Milestones: M0 build line OK + child clean + self-check ON; M1 at least one class passes the gate; M2 0.75x FA3 on every full class. A run where every class gates to FA3 is a measured result (the log shows the ratios), not a regression.
- `*layer=sliding*` neutral: FA3 kept on sliding layers (25.7 / 61.4 / 304.4 us).

## Risk

The previous 0.881 was the ungated mma.sync fallback, not the wgmma design. Remaining risk: the blind-built non-persistent, non-warp-specialized wgmma core may still be slower than FA3, in which case the gate keeps FA3 everywhere and the run is headline-neutral but tells the next stage the per-class ratio. Gate timing costs prepare() time: keep it under ~60 s total.

## Milestones

Rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001525Z by an adversarial subagent (model opus).

**Verdict: hold.** The rewrite's target and gate are sound, but sub-change (4) runs the first launch in a multiprocessing-spawn child process, and TASK.md says a submission that 'starts processes' is rejected (DECISIONS 2026-10-24 already struck cuobjdump for this reason). Re-mint with (4) replaced by an in-process guarded first launch, and keep (1)-(3) and (5) as they are.

### Justification
A: The target is real. In FINDINGS, FA3's cutlass device_kernel is 415.4 us (26.8%) on turn_1 full, 1620.2 us (32.8%) on turn_4 full, and 4467/8017 us (22.9%/36.6%) on first_turn full ctx32768/65536. REWRITE_READ #3 puts FA3 at about 1.6x its exp floor (250/1000/555/2770/4990 us). So a wgmma FP8 core has physical room: M2 (0.75x FA3) is still above the floor. The stage-1 fixes match what REWRITE_READ #3 asked for: the sm_90a build fix plus the FEAT_SM90_ALL guard, mma.sync deleted, exp offload at 0/8, and a hard cold-L2 CUDA-graph gate at <=0.92x. The gate is keyed only on config/shape fields (batch, q_len, page_table width; reference.py sizes page_table as longest+q_len, so ctx0/32768/65536 do differ in width). It is decided in prepare() on synthetic inputs, which is legal. Expected directions agree with TASK.md: full classes faster or flat by construction, sliding flat. The headline stake is small. FA3 on full layers is about 8.8% of the turn_1 load (16x415 of ~75 ms), about 11% of turn_4, and about 7.5% of first_turn. A routed class at the gate threshold therefore moves its load by only ~0.6-0.9%, inside the +-1.5% CI, so this stage must be read from the GATE log lines and the profile's top_kernels, not the headline (correct for kind: rewrite). Weakest link: per the hypothesis's own risk note, the wgmma core is non-persistent and non-warp-specialized, and was written without ever being compiled or run on the judge. FA3 is a warp-specialized ping-pong kernel with intra-warpgroup softmax/MMA overlap. A non-WS core that serializes QK wgmma, softmax and PV wgmma will probably land at 0.95-1.4x FA3, so the most likely outcome is M0 plus every class gated to FA3. That is still a useful measurement, because the per-class ratios set stage 2's design (WS/ping-pong). Secondary risks: prepare() time (about 40 s build, then the self-checks, then 5 classes x 2 kernels x >=20 cold-L2 graph replays, with ctx65536 first_turn ~8 ms each) has to stay well inside 300 s and the 15-minute run, given master's existing prepare cost. The synthetic gate inputs should use dist=soft per-batch length spread at turn_4 so split/causal balance matches the judge.

### Confounders on the current kernel
- none material: all four landed hypotheses (stack-tonight-five-prefill, sglang-pipeline-parity, experts-triton-w4a16-first-turn-blockt256, experts-hybrid-marlin-turn-triton-first-turn) touch router/align/experts/silu-mul, which run after attention. Master's side stream starts after attn_boundary, so nothing overlaps the FA3 call. REWRITE_READ confirms master's attention code is unchanged.
- sglang-pipeline-parity defines the incumbent FA3 call (paged page_size 1, scheduler_metadata None on prefill, FlashAttnFwdCombine). The new kernel and the gate must compare against exactly that call, including its combine, which the change already specifies.

### Interference and A/B needs
No companion is warranted. The change touches only the full-layer attention call, and the code it replaces is identical on master and on the old base. The hard gate makes each class's decision self-attributing: the log prints FA3 us, new us and the ratio per class, and the profile shows which kernel ran. The only internal sub-change worth isolating, exp offload, is already removed (0/8). No landed hypothesis shares this kernel's launch sequence, memory traffic or numerics, so an ab-revert or ab-tune would measure nothing new. For a rewrite, the next stage is the follow-up.

### Open questions
- Does the user want to confirm with the organizers whether a multiprocessing child used only inside prepare() counts as 'starts processes'? Until confirmed, treat it as forbidden. Record it in PENDING_USER.md rather than blocking.
- What is master's current prepare() wall time? The gate's 5 classes x 2 kernels x >=20 cold-L2 replays plus a ~40 s nvcc build must fit inside 300 s, alongside the queued CUDA expert rewrite if the two are ever stacked.
- Does the ported wgmma core use a 128-row Q tile per warpgroup with K/V double-buffered via cp.async or TMA? If K/V loads are synchronous with the wgmma, the ratio will be ~1.3x+ and stage 2 must go warp-specialized. Read the GATE ratios with that in mind.

