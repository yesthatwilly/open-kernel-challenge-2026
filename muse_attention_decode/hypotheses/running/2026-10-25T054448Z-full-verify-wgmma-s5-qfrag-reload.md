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
review: proceed
status: running
review_at: 2026-10-25T054737Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054738Z
base_sha: 3ee2b893476879190a690392a3295a35afed11bf
base_blob: 9bcf522b0dd23a0a8173046d2150473827122206
branch: hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload
worktree: .worktrees/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload
mode: null
restack_of: null
stale_base: true
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

## Adversarial review

Reviewed 2026-10-25T054737Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-5 correctness bisect step for the wgmma verify rewrite. Read against the s4 kernel, the Q A-fragment is the only per-segment input to tile lt>=1's S wgmma that differs from tile 0, and the S-vs-carry DIAG makes even a fallback run informative. Proceed with no companions, but fix how the reload is placed, and the lineage, before building.

### Justification
Mechanism (A). In s4's splitkv_verify_wgmma_kernel<TICKET,SYNC>, qa[2][4] is loaded only under `if (n == 0 || lt == 0)`, and the S wgmmas take it as the register A operand. The S accumulator is reset by scale_d=0 every tile. 1-tile segments pass at any ring stage, so the K descriptor/stage path and the V^T double buffer (n&1, which multi-request 1-tile runs also exercise) are already cleared, and SYNC rules out ring races. Between tile lt==0 and lt>=1 that leaves three differences: the qa reuse, the carried mA/lA/oacc, and the lt-dependent edge mask (lim). The rows go bad on all 128 rows, including unmasked ones, which points at S (qa) or the carry rather than the mask. The DIAG (tile-1 S vs the fp32 reference) separates those two in one check, so the run is interpretable whatever happens. Floors: verify_16 incumbent 162.8 us (FA3 main 151.6) against a 63.4 us floor, and verify_32 302.2 us (288.0) against 126.8 us. M2 (<=150 / <=280 us) only needs about 8% off the FA3 main kernel, a modest and physically reachable target for a single-pass FP8 wgmma kernel at ~2.4x above floor. Sliding and decode shapes are untouched, so flat is right. Weakest link (performance, not correctness): an unconditional per-tile reload 'from smem/L2' done as __ldg just before the S wgmma puts one L2 round trip (~0.5 us) on the critical path of EVERY tile. At ctx 80K, B=16 that is ~20K tiles over a 2*SM grid, ~75 tiles per CTA, i.e. tens of us exposed. That could erase the M2 margin and make the >=1.05x gate refuse a correct kernel. Stage Q in shared memory once per segment (ld.shared per tile), or issue the qa reload before wait_bar(full_bar(st)) so it overlaps the barrier wait. Root-causing also matters: a C++ register array held across iterations should not go stale. Dump SASS (cuobjdump -sass) and the ptxas wgmma-serialization warnings for qa's registers, so the fix is understood and not just worked around.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (master c3c1b01): the s4 branch lacks it (git merge-base confirms; s4's base table shows full decode_1 at 27.8 us vs master's 22.6 us). If s5 is built on s4 as written, the headline and full decode_1 will look worse than BEST from lineage alone, not from the verify change. The verify path is disjoint from the decode stream-K kernels, so rebase s5 onto master (frontmatter already says base: master) or judge strictly on the verify shapes' top_kernels.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup (+ ab-tune-batch-gate): these define the incumbent being replaced at full verify_16/32 (paired FA3 + prepare + combine + paired norm-gate). The interleaved graph-replay gate must time exactly master's paired path, including _paired_norm_gate_kernel, or the >=1.05x routing decision is mis-anchored. That is not a confounder of the measured kernel's own time.
- fused-preattention-triton: it writes the FP8 q (paired layout on full batched layers) that the new kernel reads via the qra/qrb addressing. If s5 sits on master, the q layout the kernel assumes (head-major, per-token HQ*D, unpadded vs paired-padded) must match what master's pre-attention kernel emits for the routed band. The self-check catches a mismatch, but it would show up as a fallback and be misread as the qfrag fix failing.

### Interference and A/B needs
No companions. This is a rewrite stage whose deliverable is the DIAG plus the self-check table; the next stage is the follow-up. A gain or loss on the headline is not the readout: verify_16/32 top_kernels against 151.6/288.0 us and the PATH CHOSEN lines are. The single-run ambiguities (S wrong vs carry wrong; correct-but-slower vs fallback) are already resolved inside the run by the DIAG and the printed paired timings. Attention has 8 checks left for 8 queued hypotheses, so an ab-revert or control would cost a check without adding information. The one lineage issue (s4 lacks c3c1b01) is fixed by rebasing, not by a companion run.

### Open questions
- Did s4's bisect include a 1-tile segment that STARTS mid-request (n == 0, lt >= 1, the stream-K tail case)? If it passed, the lt-based edge mask is cleared and qa is implicated even before s5's DIAG. If it was never tested, add it to the table.
- Why would qa go stale at all? Does the SASS show ptxas reusing or rematerialising qa's registers across the loop (e.g. under register pressure with sacc/oacc/pa at 256 threads), and are there 'wgmma serialized' ptxas warnings? A real register-clobber root cause also threatens pa and the oacc carry.
- After the fix, what are regs/thread and local bytes per variant? Reloading through smem keeps register pressure flat; an L2 __ldg reload before every tile adds latency on the critical path.
- Will s5 be rebased onto master (c3c1b01) so that a promotable result carries master's full decode_1 stream-K kernel?

