---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
created: 2026-10-25T001824Z
base: master
area: whole-layer fusion
change: >
  Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the
  bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only,
  record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2
  start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per
  phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with
  a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back);
  (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave
  (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu
  reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1
  chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer
  written between replays (cold L2, as the judge does) and asserts the replay distribution is
  unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child
  process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on
  the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() ==
  (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny
  synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a
  print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with
  -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be
  an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work"
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
depends_on: cuda-fused-moe-bs1-persistent-w13-silu-w2
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1
review: proceed
status: running
review_at: 2026-10-25T002132Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T002132Z
base_sha: 0efab7bdfa79cfea6a430d51c5b7c038a7bd2af3
base_blob: 62b00bb77cbc9e2fce8ed39dd25d87fb78503a5d
branch: hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
worktree: .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
mode: null
restack_of: null
stale_base: true
---

# cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Change

Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only, record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2 start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back); (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer written between replays (cold L2, as the judge does) and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

supersedes cuda-fused-moe-bs1-persistent-nospin-bandwidth, held for a child-process self-check; Part A only (stage 2), Part B split to stage 3 per review

## Milestones

Rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002132Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a diagnostic rewrite stage. The re-mint fixes the blocking child-process issue and splits off Part B as asked. However, two of its three slow-mode fixes have little leverage on the code stage 1 actually shipped, so the per-CTA timestamp log is the real deliverable here, and the bounded-spin and unimodality-assert handling must not turn a slow mode into wrong outputs or a silent fallback.

### Justification
The target is sound. In stage 1 (job 1105a02aa3fa4e56), moe_bs1_kernel profiled at 37.7 us against 51.2 us for the incumbent's replaced kernels; FINDINGS now shows decode_1 at 48.8 us, 87.9% of it Marlin. The scored per-pair speedup was 1.568 against a speedup of medians of 1.289, with err 1.29 us. That fits a two-mode distribution on our side, so removing the slow mode should put decode_1 near 1.8-1.9x. That is about +3% on the headline, not the +16% claimed for stage 1. The floor arithmetic (~55 MB, 11.4 us) is right. The M2 target of ~42 us scored is plausible if the slow mode goes away. WEAKEST LINK: the proposed fixes assume a cause that was never measured, and the shipped stage-1 code weakens them further. (a) __nanosleep(32) cannot account for a ~24 us second mode, because the sleep granularity is tens of ns. Replacing it with a tight poll will change almost nothing. (b) The spec's 'second wave' premise comes from the stage-1 spec (64-row tiles, 128R items), but the patch shipped ROWS=128 tiles (64 gate + 64 up), so T1=64 and there are 64R phase-1 items. At R=2 that is 128 items on 132 CTAs, which is already one wave: 128 CTAs get one item and 4 get none. Item (3) is a no-op at decode_1 unless R>2. (c) Each phase-2 item waits on a 1024-column chunk fed by 16 phase-1 tiles held by 16 different CTAs, and items are ordered (r, s, n). Phase 2 is therefore close to a global barrier on phase 1, and reordering phase-2 items by finish time has little room to help. The most likely remaining causes are a late-starting CTA (an SM still occupied at launch, or a clock or power state) or L2/HBM contention patterns. Only the per-CTA entry globaltimer and %smid, recorded under judge-like cold-L2 replays, can tell these apart. So this stage's value is the diagnostic log plus M1. The milestone should be read as 'the cause is identified', not 'M2 reached'. Implementers must start from the archived patch (ROWS=128, STAGES=8, SPLIT2=4, SPIN_MAX=1<<22, sticky C_ERR), not from the stage-1 spec text.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (gain): _post_norm_combine_kernel is part of the bs=1 incumbent the fused kernel replaces, but it is bypassed at num_tokens == 1, so it only defines the incumbent's time and causes no runtime interaction.
- triton-map-align-on-block-m-restack (gain): supplies _align_small at bs=1, which is likewise bypassed by the fused path. It defines the incumbent and does not confound.
- align-parallel-per-expert-verify (master, neutral): only num_tokens > 32, so it is disjoint from bs=1. The rebase leaves the bs=1 base identical to stage 1's.
- marlin-block-m-two-smaller-on-postnorm (gain): only num_tokens >= 128, so it is disjoint.
- sglang-marlin-path-parity (gain): supplies the Marlin bank that the fallback path and the self-check's master reference use. It does not confound the fused kernel's timing.
- Not landed but running: stack-onewarp-hoisted-dispatchcombine(-from17) changes the bs<=8 align and the bs<32 combine. If it lands first, the bs=1 incumbent drops by ~1-2 us, so read this stage against master at dispatch.

### Interference and A/B needs
No companion is needed. Every landed hypothesis that touches bs=1 is bypassed when num_tokens == 1, so neither an ab-revert nor an ab-tune would change the reading. The internal attribution problem (whether the spin change or the reordering killed the slow mode) is better answered by the in-kernel per-phase timestamps and the spin-iteration counts the self-check already logs than by a control run. If the slow mode persists, the next stage follows from the log.

### Open questions
- What is R at the judge's decode_1 seed? At R=2 the shipped kernel already runs phase 1 in one wave (64R = 128 items <= 132 CTAs), so step (3) changes nothing. Should the one-wave sizing target R=3..8 instead, where it does matter?
- Does the self-check under cold-L2 graph replays reproduce the 41/59 bimodality at all? If not, the per-CTA log cannot explain the judge's slow mode, and the next stage would need the diagnostic to run in the timed path under a flag (which costs a check).
- Is the slow mode a late CTA entry (an occupied SM or the launch tail of the judge's L2-flush/read kernel) rather than a phase-2 wait? If entry skew is ~24 us, neither the spin nor the ordering fix helps, and the remedy is a grid < SM count (e.g. 128 CTAs) or a work-stealing atomic item counter instead of the static c, c+G assignment.
- Would a dynamic item queue (atomicAdd ticket over items) remove the dependency on a straggler CTA entirely? It is a better candidate for stage 3 than reordering.

