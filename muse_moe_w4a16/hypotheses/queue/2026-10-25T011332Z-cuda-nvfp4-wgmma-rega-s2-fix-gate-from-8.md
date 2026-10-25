---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8
created: 2026-10-25T011332Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 2 of the wgmma register-A grouped NVFP4 rewrite with the gate widened to the small
  verify/decode shapes. Implement exactly the correctness work of
  cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix (read its spec: per-launch localization in the
  canary, zero smem ring/slab at CTA start and zero-filled padding rows, offline numpy check of
  the wgmma m64nNk16 register-A mapping, B smem descriptor LBO/SBO/swizzle, mbarrier expect_tx
  bytes and scale application), and additionally lower the wgmma path's own bound constant from
  num_tokens >= 128 to num_tokens >= 8 (bs=8 verify_1, bs=16, bs=32, and the bs=33 test), using
  the kernel's existing smallest N buckets (16 and 32 tokens per expert tile; under EP8 with
  skewed routing each local expert gets about 1-8 routes at bs 8-32). bs=1 and the bs=4 test stay
  on master Marlin. Extend the canary and the full self-check to M in {8, 16, 32, 33} as well as
  128/256/257, each with its own rel-L2 line, and enable the small-M range only if every small-M
  case passes (separate flag, separate PATH CHOSEN line for 8..127 and for 128..512). No tiling,
  pipeline or dequant change otherwise. Motivation: Marlin is furthest from the floor in the
  middle shapes (bs=8 114.0 us vs floor 37.3, bs=16 176.1 vs 52.5, bs=32 207.5 vs 70.8 in the best
  profile), and the persistent 132-CTA grid streams each touched expert's weight once regardless
  of rows, which is the property Marlin's m-block scheme lacks. First measurement is an okc check.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair 114.0 + 2.0 us (best profile) vs floor 37.3; M2 below Marlin, M3 <= 56
  - shapes: bs=16 *
    direction: faster
    why: 176.1 + 2.2 us vs floor 52.5; M3 <= 79
  - shapes: bs=32 *
    direction: faster
    why: 207.5 + 2.3 us vs floor 70.8; M3 <= 106. decode_32 is master weakest load (1.03x vs SGLang)
  - shapes: bs=128 *
    direction: faster
    why: "as in the finite-fix sibling: below 292.5 + 3.9 us"
  - shapes: bs=256 *
    direction: faster
    why: below 391.4 + 6.4 us
  - shapes: bs=1 *
    direction: flat
    why: "control: stays Marlin"
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 512
review: hold
status: hold
review_at: 2026-10-25T012041Z
review_companions: null
hold_reason: "adversarial review: Hold, and set depends_on to cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix. It re-implements the same unfixed correctness work and also widens the gate into shapes"
---

# cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8

## Change

Stage 2 of the wgmma register-A grouped NVFP4 rewrite with the gate widened to the small verify/decode shapes. Implement exactly the correctness work of cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix (read its spec: per-launch localization in the canary, zero smem ring/slab at CTA start and zero-filled padding rows, offline numpy check of the wgmma m64nNk16 register-A mapping, B smem descriptor LBO/SBO/swizzle, mbarrier expect_tx bytes and scale application), and additionally lower the wgmma path's own bound constant from num_tokens >= 128 to num_tokens >= 8 (bs=8 verify_1, bs=16, bs=32, and the bs=33 test), using the kernel's existing smallest N buckets (16 and 32 tokens per expert tile; under EP8 with skewed routing each local expert gets about 1-8 routes at bs 8-32). bs=1 and the bs=4 test stay on master Marlin. Extend the canary and the full self-check to M in {8, 16, 32, 33} as well as 128/256/257, each with its own rel-L2 line, and enable the small-M range only if every small-M case passes (separate flag, separate PATH CHOSEN line for 8..127 and for 128..512). No tiling, pipeline or dequant change otherwise. Motivation: Marlin is furthest from the floor in the middle shapes (bs=8 114.0 us vs floor 37.3, bs=16 176.1 vs 52.5, bs=32 207.5 vs 70.8 in the best profile), and the persistent 132-CTA grid streams each touched expert's weight once regardless of rows, which is the property Marlin's m-block scheme lacks. First measurement is an okc check.

## Expected

- `bs=8 *` faster: Marlin pair 114.0 + 2.0 us (best profile) vs floor 37.3; M2 below Marlin, M3 <= 56
- `bs=16 *` faster: 176.1 + 2.2 us vs floor 52.5; M3 <= 79
- `bs=32 *` faster: 207.5 + 2.3 us vs floor 70.8; M3 <= 106. decode_32 is master weakest load (1.03x vs SGLang)
- `bs=128 *` faster: as in the finite-fix sibling: below 292.5 + 3.9 us
- `bs=256 *` faster: below 391.4 + 6.4 us
- `bs=1 *` flat: control: stays Marlin

## Risk

Shares the sibling's bug risk: if the finite fix is wrong both fail. Read per-shape rows: small-M and large-M ranges have separate flags, so one can be on while the other is off.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T012041Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold, and set depends_on to cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix. It re-implements the same unfixed correctness work and also widens the gate into shapes where the kernel has never run a single correct tile. Widening is worth a check only after finite-fix reaches M1 at >=128 and shows the N=16/32 buckets correct and competitive.

### Justification
The motivation is real: Marlin at bs=8/16/32 is 114.0/176.1/207.5 us against floors of 37.3/52.5/70.8, and decode_32 is master's weakest load. The mechanism claimed is weak, though. 'The persistent grid streams each touched expert's weight once, which Marlin's m-block scheme lacks' is not true at small M: with 1-8 routes per local expert, Marlin's single m-block per expert also streams each touched weight once. Marlin's gap at small M is more plausibly occupancy and latency (few experts x N-tiles over 132 SMs, short K loops). This kernel's 1-producer/2-consumer, 4-stage, BK-128 pipeline with a 64-row M per warpgroup does not obviously fix that: at bs=8 under EP8 roughly 6-8 experts x (I/64) w13 items is a few hundred items over 132 CTAs, with pipeline fill costs on each. The direct evidence is negative: the persistent swap-AB mma.sync sibling, with the same weights-on-M idea and the same 132-CTA grid, ran correctly at T<=32 and was 1.7-2x SLOWER than Marlin (LESSONS 2026-10-25, REWRITE_READ 01:04Z), and the prefill wgmma grouped GEMM was 1.8x slower than Triton. WEAKEST LINK: the bs 8-32 'faster' expectations rest on an untested premise, and the kernel has never produced a correct tile in any bucket. Bundling the gate change with the finite fix means a correctness failure in the small-M buckets (N=16/32, never exercised: the canary used only N=128) is confounded with the large-M fix. Separate flags reduce that but do not remove it, and the 8 extra self-check sizes add prepare time and log volume to a log that is only a 15 KB tail.

### Confounders on the current kernel
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: at 8 <= T <= 32, master uses the fused _align_small Triton align. The wgmma path replaces it with expert_map gather + moe_align_block_size + gather (3 launches), which at small M is a measurable fraction (several us out of 114-207). The replaced-kernel sum must include it, and it biases against the rewrite.
- post-expert-norm-triton-small-m: at T < 32 master uses the Triton _post_norm_combine_kernel, but _moe_wgmma always calls SGLang's fused norm. Widening the gate to 8-31 silently reverts that landed gain on bs=8/16 unless the wgmma path keeps master's small-M norm branch. This is a definite regression source that would be misread as the GEMM losing.
- marlin-block-m-two-smaller-on-postnorm: no effect below 128 (SGLang's block pick is kept there).

### Interference and A/B needs
No ab companion now. The right sequencing is a dependency, not a companion: run finite-fix first. With the merged design from the finite-fix review, its self-check covers buckets 16/32/64 at M=256 skewed and prints prepare-time per-kernel timing. Mint the widened gate only if that run shows (i) N=16/32-bucket items correct and (ii) wgmma per-kernel prepare time at a small-M-like routing that beats Marlin. A cheap way to get (ii) without widening the gate is to time the wgmma kernels in finite-fix's prepare at synthetic M=8/16/32 routings as well (print only, no enable). That turns this hypothesis's premise into data at zero extra check cost. When re-minted, keep master's small-M post-norm and align branches on the wgmma path for T < 32.

### Open questions
- Why is Marlin far from the floor at bs=8-32: occupancy/latency, or bytes? If occupancy, a split-K or more-CTAs-per-expert scheme is needed, which this kernel does not have.
- Does the bs=33 test and the verify_1 bs=8 shape weight justify the extra self-check time and the larger rules surface before M1 exists at >=128?

