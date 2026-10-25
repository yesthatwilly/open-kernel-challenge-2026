---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-nospin-bandwidth
created: 2026-10-25T001254Z
base: master
area: whole-layer fusion
change: >
  Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A, kill the slow mode
  first: (1) in the self-check only, record per-CTA globaltimer at phase-1 start/end and phase-2
  start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per
  phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with
  a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back);
  (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave
  (128R items split in K to a multiple of 132 CTAs) and order phase-2 (route, k-split) items so
  each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph
  replays and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's
  first launches in a spawned child process (LESSONS 2026-10-25), keep plain cp.async.cg (no
  createpolicy), build with -lineinfo. PART B, only after A: raise bandwidth from ~1.5 TB/s toward
  3 TB/s with a deeper cp.async ring and phase-1 K-split when R is 1-2, overlapping the ~27-op/k16
  dequant with the next stage's loads. Keep dequant, weight layout, routing and numerics
  unchanged.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (+15% on decode_1, ~40-47 us fast mode); M3 <= 18 us (~3 TB/s) after Part B"
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
review: hold
status: hold
review_at: 2026-10-25T001616Z
review_companions: null
hold_reason: "adversarial review: The design is sound and stage 1's data supports a stage 2, but the change as written runs the self-check in a spawned child process, and muse_moe_w4a16/TASK.md "
---

# cuda-fused-moe-bs1-persistent-nospin-bandwidth

## Change

Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A, kill the slow mode first: (1) in the self-check only, record per-CTA globaltimer at phase-1 start/end and phase-2 start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back); (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave (128R items split in K to a multiple of 132 CTAs) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph replays and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's first launches in a spawned child process (LESSONS 2026-10-25), keep plain cp.async.cg (no createpolicy), build with -lineinfo. PART B, only after A: raise bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is 1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight layout, routing and numerics unchanged.

## Expected

- `bs=1 *` faster: incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (+15% on decode_1, ~40-47 us fast mode); M3 <= 18 us (~3 TB/s) after Part B
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

Part A is the attributable step; if the slow mode persists the debug timestamps name the stalled phase. Part B confounds attribution if A is also new, so judge A on the replay distribution and B on the profile row. Fallback: master at bs=1.

## Milestones

Rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001616Z by an adversarial subagent (model opus).

**Verdict: hold.** The design is sound and stage 1's data supports a stage 2, but the change as written runs the self-check in a spawned child process, and muse_moe_w4a16/TASK.md says a submission that starts processes is rejected. Re-mint without the child process (stage 1 already ran this exact kernel in-process on the judge without a fault), and split Part B into stage 3 so this run reads only the slow-mode fix.

### Justification
The mechanism holds up. Stage 1 (job 1105a02aa3fa4e56) profiled moe_bs1_kernel at 37.7 us against the incumbent's 51.2 us of replaced kernels. The scored paired-ratio speedup at decode_1 (1.568) sits well above the speedup of medians (1.289), and the error grew from 0.47 to 1.29 us. That is what an uncorrelated two-mode distribution on our side looks like: when 41% of replays are fast and 59% slow, the median pair is a mixed fast+slow pair, giving about 85/56.5 = 1.5. So killing the slow mode is worth roughly 77/43 = 1.8-1.95x on decode_1 (vs master's 1.578). That is about +20% on that load and about +3.5% on the headline as a geomean over 6 loads, not the +16% stage 1 claimed. The ~42 us scored M2 target is plausible. The floor arithmetic is right: R is about 2 routes x ~27 MB per expert, about 55 MB, so 11.4 us at 4.8 TB/s, and 37.7 us is about 1.46 TB/s. M3 (18 us, ~3 TB/s) is physically reachable for a weight-stream GEMV with a deep cp.async ring on 132 SMs. WEAKEST LINK: the cause of the slow mode is assumed, not measured. The ~24 us extra is about one whole phase, which fits a chain stall, but it also fits a co-residency hazard. If any SM is briefly busy at launch (another stream's work, or the harness's L2-flush tail), one CTA starts late. Every phase-2 CTA that waits on its phase-1 tiles then stalls, and neither a tight poll nor item reordering fixes that. The planned timestamps (phase-1 start/end, phase-2 start) miss it. Record each CTA's entry globaltimer and %smid as well. The prepare-time self-check also replays with a warm L2 and nothing else on the GPU, so it may never reproduce the judge's slow mode. Write a buffer larger than L2 between the 100 self-check replays, the way the judge does. Second weak point: '128R items split in K to a multiple of 132' is not a free change. A K-split w13 tile needs its gate/up partials reduced before silu (silu is nonlinear), which adds a phase-1 ticket/reduction level and another place to stall. A simpler path to one wave is to size items so each CTA owns ceil(128R/132) whole tiles, and to have phase-2 CTAs take h chunks in the order they finish.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (gain): its _post_norm_combine_kernel is one of the replaced bs=1 kernels, so it is part of the incumbent that the fused kernel's row is compared against (3.1 us of 48.8). No interaction at runtime because num_tokens == 1 bypasses it entirely; nothing to disambiguate.
- triton-map-align-on-block-m-restack (gain): supplies the _align_small path the fused kernel replaces at bs=1; bypassed at num_tokens == 1, so it does not confound the measurement, only defines the incumbent.
- align-parallel-per-expert-verify (neutral, master): touches only num_tokens > 32; the bs=1 path is identical to stage 1's base, so the rebase changes nothing at decode_1.
- marlin-block-m-two-smaller-on-postnorm (gain): changes block_size_m only at num_tokens >= 128; disjoint.
- In flight, not landed: stack-onewarp-hoisted-dispatchcombine(-from17) rewrites bs<=8 align and the bs<32 combine. If it lands first, the bs=1 incumbent drops by ~1-2 us, so read stage 2 against whatever master is at dispatch, not against the 51.2 us figure.

### Interference and A/B needs
No companion. This is a rewrite stage judged on the fused kernel's own profile row and the bs=1 replay distribution. Every landed hypothesis that touches bs=1 is bypassed when num_tokens == 1, so no ab-revert or ab-tune would change the reading. The one real attribution problem is internal: Part A (spin, one-wave and ordering) and Part B (deeper ring, K-split) in the same run would mix 'the slow mode is gone' with 'bandwidth went up'. The fix is to stage them, not to add a companion. Stage 2 = Part A plus the extended diagnostics. Stage 3 = Part B, minted only after stage 2's log shows which phase stalled and the replay distribution is unimodal.

### Open questions
- Is the slow mode a phase-2 spin/second-wave stall, or a late-starting CTA (an occupied SM at launch)? Per-CTA entry time + %smid decides it, and only if the self-check reproduces cold-L2, judge-like replays.
- What is R at the judge's decode_1 seed? If R == 2 (256 phase-1 tiles over 132 CTAs), the second wave is nearly full and ordering helps; if R varies across the 5 tests, the one-wave sizing must cover R in 0..8.
- Can the one-wave goal be met without a K-split of w13 tiles (which needs a pre-silu reduction), for example 2 whole tiles per CTA with phase-2 order matched to chunk completion?
- Once a pre-flight check confirms the judge's static checker rejects multiprocessing, the LESSONS 2026-10-25 child-process advice should be amended for all problems.

