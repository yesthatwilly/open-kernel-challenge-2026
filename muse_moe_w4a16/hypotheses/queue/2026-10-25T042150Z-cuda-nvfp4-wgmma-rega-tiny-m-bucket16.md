---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-tiny-m-bucket16
created: 2026-10-25T042150Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Tiny-M branch of the wgmma register-A grouped rewrite: bs=1 (decode_1) and the bs=4 test, which
  master runs at 44.9 us (Marlin pair ~42 us) vs a floor of 11.4 us - the largest ratio of any
  shape (3.9x). Not the dead bs1 fused persistent kernel: no grid-wide flags, no spins, no phase
  dependency - two ordinary launches (wg_kernel<w13+silu> then wg_kernel<w2>) using only the
  proven N16 bucket (1-7 tokens pad to 16 rows; MMA waste is irrelevant, the shape is
  weight-bandwidth bound). Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current
  master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched
  shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as
  stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo,
  wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: own bound constant WG_TINY_MAX_TOKENS = 7; every item is N16 (cap 16
  rows per item on device); align is master's _align_small; combine stays master's
  _post_norm_combine_kernel. For bandwidth at 1-2 experts, the w13 grid has 64 N-tiles x experts
  items, which under-fills 132 CTAs at one expert: add a K-split of 2 for w13 and 4 for w2 when
  the item count < 132 (fp32 partials in a T-sized workspace reduced in the consumer epilogue by
  the last-arriving CTA via a per-item ticket that the last CTA resets to 0, so replays start
  clean; no spin waits). Hard gate in prepare(): after the correctness self-check passes, time the
  new path against master's path on the same synthetic routings (skewed and random, cold L2 via a
  >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T
  bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice
  equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T
  bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert,
  all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant
  reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not
  against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is
  unmeasured. First measurement is an okc check, never a submit. T buckets: 1 and 4. Because the
  judge's timed loop has shown bimodal bs=1 behaviour unseen in prepare (LESSONS 2026-10-25), also
  report the judge's runs: read the decode_1 median vs raw median in the result.
expected:
  - shapes: bs=1 *
    direction: faster
    why: master 44.9 us (Marlin pair ~42.1 + combine 3.2 + act 1.6 + align 1.6), floor 11.4; M2 wgmma pair < 42 us; M3 <= 25 us
  - shapes: bs=8 *
    direction: flat
    why: "control: above the 7-token gate"
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 1 <= num_tokens <= 7
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-tiny-m-bucket16

## Change

Tiny-M branch of the wgmma register-A grouped rewrite: bs=1 (decode_1) and the bs=4 test, which master runs at 44.9 us (Marlin pair ~42 us) vs a floor of 11.4 us - the largest ratio of any shape (3.9x). Not the dead bs1 fused persistent kernel: no grid-wide flags, no spins, no phase dependency - two ordinary launches (wg_kernel<w13+silu> then wg_kernel<w2>) using only the proven N16 bucket (1-7 tokens pad to 16 rows; MMA waste is irrelevant, the shape is weight-bandwidth bound). Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: own bound constant WG_TINY_MAX_TOKENS = 7; every item is N16 (cap 16 rows per item on device); align is master's _align_small; combine stays master's _post_norm_combine_kernel. For bandwidth at 1-2 experts, the w13 grid has 64 N-tiles x experts items, which under-fills 132 CTAs at one expert: add a K-split of 2 for w13 and 4 for w2 when the item count < 132 (fp32 partials in a T-sized workspace reduced in the consumer epilogue by the last-arriving CTA via a per-item ticket that the last CTA resets to 0, so replays start clean; no spin waits). Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets: 1 and 4. Because the judge's timed loop has shown bimodal bs=1 behaviour unseen in prepare (LESSONS 2026-10-25), also report the judge's runs: read the decode_1 median vs raw median in the result.

## Expected

- `bs=1 *` faster: master 44.9 us (Marlin pair ~42.1 + combine 3.2 + act 1.6 + align 1.6), floor 11.4; M2 wgmma pair < 42 us; M3 <= 25 us
- `bs=8 *` flat: control: above the 7-token gate
- `bs=16 *` flat: control
- `bs=32 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

Kill: prepare gate keeps Marlin at T=1, or the judged decode_1 is not below 44.9 us by > 1.5%, or decode_1 shows a slow mode in more than 10/100 replays. The ticket reduction is the only new device logic; if it is judged risky, drop it (K-split 1) and accept an under-filled grid for stage 3.

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 1 <= num_tokens <= 7`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
