---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc
created: 2026-10-25T001825Z
base: master
area: whole-layer fusion
change: >
  Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART B of the held
  cuda-fused-moe-bs1-persistent-nospin-bandwidth (split per its review), on top of stage 2
  cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc (Part A, slow mode removed): raise
  bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is
  1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight
  layout, routing and numerics unchanged. Keep stage 2's one-wave sizing and tight poll, plain
  cp.async.cg and -lineinfo; first launches in-process (never in a child process: the rules
  statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS
  2026-10-25, corrected)) with the sm_90a image check and a synthetic canary followed by
  torch.cuda.synchronize() and a print. Run only after stage 2's log shows which phase stalled and
  the replay distribution is unimodal. The first measurement of this hypothesis must be an `okc
  check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "stage 2 fast mode ~40-47 us at ~1.5 TB/s, floor 11.4 us (~55 MB). M2: below stage 2's bs=1 row; M3 <= 18 us (~3 TB/s) via deeper cp.async ring + phase-1 K-split at R 1-2"
  - shapes: bs=8 *
    direction: flat
    why: "control: master"
  - shapes: bs=16 *
    direction: flat
    why: "control: master"
  - shapes: bs=32 *
    direction: flat
    why: "control: master"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: _align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1
review: proceed
status: running
review_at: 2026-10-25T002107Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T013639Z
base_sha: ea3464f424a209eb0f46541178cd3dd11062426f
base_blob: c80e3a831390ffb38cbcb7560231b59a64a2c6a0
branch: hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc
worktree: .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc
mode: null
restack_of: null
stale_base: true
---

# cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc

## Change

Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART B of the held cuda-fused-moe-bs1-persistent-nospin-bandwidth (split per its review), on top of stage 2 cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc (Part A, slow mode removed): raise bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is 1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight layout, routing and numerics unchanged. Keep stage 2's one-wave sizing and tight poll, plain cp.async.cg and -lineinfo; first launches in-process (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)) with the sm_90a image check and a synthetic canary followed by torch.cuda.synchronize() and a print. Run only after stage 2's log shows which phase stalled and the replay distribution is unimodal. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: stage 2 fast mode ~40-47 us at ~1.5 TB/s, floor 11.4 us (~55 MB). M2: below stage 2's bs=1 row; M3 <= 18 us (~3 TB/s) via deeper cp.async ring + phase-1 K-split at R 1-2
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

split from cuda-fused-moe-bs1-persistent-nospin-bandwidth (Part B), held for a child-process self-check; stage 3 after cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Milestones

Rewrite stage 3 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002107Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, but only after stage 2 has run: the bandwidth target (<= 18 us, about 3 TB/s for about 56 MB at R=2) is physically reachable, and no landed hypothesis confounds the bs=1 path. The one real design flaw is that 'phase-1 K-split when R is 1-2' as written means a cross-CTA pre-silu reduction, which contradicts 'keep stage 2's one-wave sizing' and brings back the inter-CTA stall that stage 2 removes. Do the K-split inside each CTA across warps instead, and treat dequant issue rate, not ring depth, as the likely limiter.

### Justification
The mechanism is plausible. At bs=1, R is about 2 local routes. Each expert is w13 (2*4096*4096) + w2 (4096*4096) NVFP4, about 25 MB of nibbles plus about 3 MB of e4m3 scales, so about 56 MB per call. That gives the 11.4 us floor at 4.8 TB/s. Stage 1's fused row of 37.7 us is about 1.5 TB/s, and the incumbent is Marlin 44.7 + post-norm 3.1 + act_and_mul 1.6 + align, about 49-51 us (FINDINGS: decode_1 48.8 us vs SGLang 77.1 us). Hitting M3 (<= 18 us) would take decode_1 from 1.58x to about 3.9x vs SGLang, about +16% on the headline geomean. Stage 2's ~42 us alone is only about +2-3%, so stage 3 is where the payoff is. The expected directions match TASK.md: only bs=1 moves, and every other M runs master byte for byte. WEAKEST LINK: the hypothesis blames bandwidth on ring depth, but bytes in flight are probably already adequate. 132 SMs x 27-54 KB is 3.5-7 MB in flight, and Little's law at about 1 us DRAM latency already gives >= 3.5 TB/s. The likelier limiter is dequant and MMA issue rate at 1 CTA x 8 warps per SM, which is 2 warps per SMSP. At about 27 ops per lane per m16n8k16, over 112M weights at bs=1, the dequant alone is about 380M lane-ops, roughly 12-14 us of ALU across 132 SMs. M3 therefore needs near-perfect load/compute overlap, and 2 warps per scheduler cannot hide it. A deeper ring alone may land around 25-30 us (M2, not M3). Second weak point: a cross-CTA K-split of w13 at R=1-2 needs gate/up partials reduced before silu, which is nonlinear. That means a new global ticket/reduction level and a new spin dependency, exactly the stall class stage 2 removes. It also conflicts with 'keep stage 2's one-wave sizing' (it changes the item count) and with 'keep numerics unchanged' (fp32 summation order changes; harmless for the 1.5% bound, but not 'unchanged'). An intra-CTA split-K (each warp a K slice of the same 64-row tile, reduced in smem before the silu epilogue) gets the same parallelism with no inter-CTA dependency. Spending the freed smem (the ring is ~100 KB of 227 KB) on more warps per CTA (384-512 threads) would also hide the dequant better. The profile row for moe_bs1_kernel is readable against stage 2's row, so the milestones are measurable.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (landed gain): its _post_norm_combine_kernel is among the replaced bs=1 kernels and is bypassed when num_tokens == 1. It defines the incumbent (3.1 us) and does not interact at runtime.
- triton-map-align-on-block-m-restack (landed gain): supplies _align_small for bs=1, which the fused kernel bypasses. It is incumbent-defining only.
- align-parallel-per-expert-verify (master, neutral): touches only num_tokens > 32, so the bs=1 path is identical. No confound.
- marlin-block-m-two-smaller-on-postnorm (landed gain): block_size_m change at num_tokens >= 128 only, disjoint from bs=1.
- sglang-marlin-path-parity (landed): defines the Marlin incumbent at bs=1, which the fused kernel replaces wholesale. No runtime interaction.
- Not landed, but the real confounder: stage 2 (cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc). If its replay distribution is still bimodal, a stage-3 median cannot separate bandwidth from slow-mode effects. The spec already gates on stage 2's log; enforce that gate. Also, in-flight stack-onewarp-hoisted-dispatchcombine(-from17) may lower the bs=1 incumbent by 1-2 us, so read stage 3 against master at dispatch.

### Interference and A/B needs
No companion is warranted. Every landed hypothesis that touches bs=1 is bypassed when num_tokens == 1, so neither an ab-revert nor an ab-tune would change the reading. This is a rewrite stage judged on the fused kernel's own profile row against stage 2's row and the 11.4 us floor. The internal attribution problem is that the ring deepening and the K-split are bundled. It is not worth a separate check: stage 2's per-phase globaltimer instrumentation, kept in the self-check, attributes phase-1 vs phase-2 time directly. Print per-phase min/median/max for stage 3 as well, and the two sub-changes can be read from the log. If the result is ambiguous, the next stage is the follow-up.

### Open questions
- After stage 2, is the fused kernel load-bound or dequant/issue-bound? Check with Nsight-style evidence from -lineinfo, or a self-check variant that skips the dequant and does raw MMA on the loaded bytes. If it is issue-bound, ring depth will not reach M3.
- Can the K-split be intra-CTA (warps split K, smem reduction before silu) rather than inter-CTA, so no pre-silu global reduction or extra spin is introduced?
- Would 1 CTA x 384-512 threads, or 2 CTAs per SM with co-residency still asserted, hide dequant better than a deeper ring at 8 warps?
- What R values do the judge's bs=1 test and decode_1 seed actually produce? K-split only 'when R is 1-2' needs a branch on R that is data-dependent in-kernel (fine on device, but both code paths must be self-checked, including R=0 and R=8).
- Does stage 2 actually produce a unimodal replay distribution? If not, this stage should not dispatch.

