---
problem: muse_attention_decode
slug: full-verify-wgmma-s5-qfrag-reload
created: 2026-10-25T054448Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, multi-tile Q-fragment fix)
change: >
  Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job
  51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1-2t with no request
  crossing) is wrong with all 128 rows' running max bad; 1-tile segments pass at any ring stage;
  the fully synchronised variant fails identically, ruling out ring/mbarrier races. The only
  per-segment state not reloaded before the S wgmma of tile lt>=1 is the register-sourced Q
  A-fragment qa[2][4], loaded only at n == 0 || lt == 0 and passed as the {a regs} operand across
  iterations. Fix (a): reload qa from smem/L2 unconditionally before EVERY tile's S wgmma (2 KB),
  with wg_fence/fence_acc ordering around it as for the first tile; keep the rest of s4. Add the
  DIAG from the read (c): for B1-2t, dump S (pre-max) of tile 1 and compare with the fp32
  reference S, printing max abs err, so 'S wrong' vs 'carry wrong' separates in this one check. If
  (a) fixes S but rows still fail, print per-tile mA/lA and oacc error to localise the carry
  (ex2(mA - uA) with Max_offset folded). Drop the SYNC variant. Keep FP8 P with FA3's Max_offset
  (P = exp2(s - (max - 8))) and test on COHERENT long-context data, not random keys (LESSONS).
  Gate: after the full s4 bisect table passes (every multi-tile case, kv 0.25, grid 3, device
  grid, lens_a/lens_b, coherent), time [wgmma ticket], [wgmma b8] vs master paired FA3 path
  (prepare + main + combine + paired norm-gate) at B=16 and B=32 by interleaved graph replay;
  route a band only at >= 1.05x. Print regs/local per variant (refuse a spill, LESSONS prefill
  wgmma). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: incumbent 162.8 us shape (FA3 main 151.6 + combine 6.0 + prepare 3.0); floor 63.4 us; M1 = self-check passes and the kernel is timed, M2 <= 150 us
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: incumbent 302.2 us (FA3 288.0 + combine 7.5); floor 126.8 us; M2 <= 280 us
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: high
depends_on: full-verify-wgmma-s4-multitile-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)"
review: required
status: queued
---

# full-verify-wgmma-s5-qfrag-reload

## Change

Stage 5 on branch hyp/muse_attention_decode/full-verify-wgmma-s4-multitile-fix (job 51b0d5c110f3459c). s4's bisect: every multi-tile segment (>=2 tiles, even B1-2t with no request crossing) is wrong with all 128 rows' running max bad; 1-tile segments pass at any ring stage; the fully synchronised variant fails identically, ruling out ring/mbarrier races. The only per-segment state not reloaded before the S wgmma of tile lt>=1 is the register-sourced Q A-fragment qa[2][4], loaded only at n == 0 || lt == 0 and passed as the {a regs} operand across iterations. Fix (a): reload qa from smem/L2 unconditionally before EVERY tile's S wgmma (2 KB), with wg_fence/fence_acc ordering around it as for the first tile; keep the rest of s4. Add the DIAG from the read (c): for B1-2t, dump S (pre-max) of tile 1 and compare with the fp32 reference S, printing max abs err, so 'S wrong' vs 'carry wrong' separates in this one check. If (a) fixes S but rows still fail, print per-tile mA/lA and oacc error to localise the carry (ex2(mA - uA) with Max_offset folded). Drop the SYNC variant. Keep FP8 P with FA3's Max_offset (P = exp2(s - (max - 8))) and test on COHERENT long-context data, not random keys (LESSONS). Gate: after the full s4 bisect table passes (every multi-tile case, kv 0.25, grid 3, device grid, lens_a/lens_b, coherent), time [wgmma ticket], [wgmma b8] vs master paired FA3 path (prepare + main + combine + paired norm-gate) at B=16 and B=32 by interleaved graph replay; route a band only at >= 1.05x. Print regs/local per variant (refuse a spill, LESSONS prefill wgmma). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=verify_16 *` faster: incumbent 162.8 us shape (FA3 main 151.6 + combine 6.0 + prepare 3.0); floor 63.4 us; M1 = self-check passes and the kernel is timed, M2 <= 150 us
- `*layer=full load=verify_32 *` faster: incumbent 302.2 us (FA3 288.0 + combine 7.5); floor 126.8 us; M2 <= 280 us
- `*layer=sliding *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Even a fallback run is informative: the DIAG decides between the Q-fragment and the carry. First timed measurement of this kernel; okc check only.

## Milestones

Rewrite stage 5 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
