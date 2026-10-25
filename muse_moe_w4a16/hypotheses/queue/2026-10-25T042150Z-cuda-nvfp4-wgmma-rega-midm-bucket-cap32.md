---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-midm-bucket-cap32
created: 2026-10-25T042150Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Mid-M branch of the wgmma register-A grouped NVFP4 rewrite, using ONLY the token buckets stage 2
  proved correct. Stage 2 (job 583a73418fbf4d01) showed the N16 and N32 bucket instantiations
  correct for both kernels (w13+silu rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3) and only N64/N128
  broken; s3 (running) fixes those for T >= 128. This hypothesis does not touch N64/N128 and does
  not change T >= 128. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current
  master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched
  shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as
  stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo,
  wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: (1) add a second gate with its own bound constants WG_MIDM_MIN_TOKENS =
  8 and WG_MIDM_MAX_TOKENS = 32 (verify_1 bs=8, decode_16, decode_32; bs=1, the bs=4 test and the
  bs=33 test stay master Marlin); (2) in the device-side work-item builder (wg_gather / bucket
  split), cap every item at 32 rows: an expert with cnt routes becomes ceil(cnt/32) items of <= 32
  rows, each assigned the N16 bucket if its rows <= 16 else N32 - computed on device from the
  align counts, no host branching on values, buffers sized from T (capacity T*8 + 32*(32-1) rows)
  so it is graph-capturable; an expert split into several items re-reads its weights (from L2)
  once per item; (3) the w13+silu and w2 kernels, tiling (BK 128, 4 stages, grid 132 x 384, 1
  producer + 2 consumer warpgroups), dequant and epilogues are unchanged; the post-expert
  norm+combine stays master's _post_norm_combine_kernel (T < 32) /
  fused_post_expert_norm_weighted_sum (T = 32) fed with y in master's layout. Hard gate in
  prepare(): after the correctness self-check passes, time the new path against master's path on
  the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph
  replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if
  its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS
  2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The
  self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random
  routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error
  printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without
  PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc
  check, never a submit. T buckets for the gate: 8, 16, 32. Motivation: in the best profile
  Marlin's pair is 94.6-114 us at bs=8 (floor 37.3), 170-176 at bs=16 (floor 52.5), 156-207 at
  bs=32 (floor 70.8) - 2.5-3x off the floor, and stage-2 prepare timing put the wgmma pair at
  0.57x Marlin at M=128/256; the persistent grid streams each touched expert once regardless of
  route count, which Marlin's m-block scheme lacks.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair ~95-114 + act 2.0 us vs floor 37.3; M1 PATH CHOSEN = WGMMA at T=8 with tests 5/5; M2 wgmma pair below Marlin pair; M3 <= 56 us
  - shapes: bs=16 *
    direction: faster
    why: Marlin pair ~170-176 + 2.2 us vs floor 52.5; M3 <= 79 us
  - shapes: bs=32 *
    direction: faster
    why: "Marlin pair ~156-207 + 2.3 us vs floor 70.8; decode_32 is master's weakest load (~1.03x); M3 <= 106 us"
  - shapes: bs=1 *
    direction: flat
    why: "control: below the 8-token gate, master Marlin with w2 atomic"
  - shapes: bs=128 *
    direction: flat
    why: "control: T >= 128 is master (s3 owns it)"
  - shapes: bs=256 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 32
review: proceed
status: hold
review_at: 2026-10-25T042602Z
review_companions: null
hold_reason: "Reviewer: proceed but wait for s3 wide-bucket-fix (3bdec0e58d684c17) harvest; rebuild on s3 commit + master 9080a71 with the variant s3 picks at N16/N32; cap32 is an assert at T<=32; use master _align_small block 16; then unhold."
---

# cuda-nvfp4-wgmma-rega-midm-bucket-cap32

## Change

Mid-M branch of the wgmma register-A grouped NVFP4 rewrite, using ONLY the token buckets stage 2 proved correct. Stage 2 (job 583a73418fbf4d01) showed the N16 and N32 bucket instantiations correct for both kernels (w13+silu rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3) and only N64/N128 broken; s3 (running) fixes those for T >= 128. This hypothesis does not touch N64/N128 and does not change T >= 128. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: (1) add a second gate with its own bound constants WG_MIDM_MIN_TOKENS = 8 and WG_MIDM_MAX_TOKENS = 32 (verify_1 bs=8, decode_16, decode_32; bs=1, the bs=4 test and the bs=33 test stay master Marlin); (2) in the device-side work-item builder (wg_gather / bucket split), cap every item at 32 rows: an expert with cnt routes becomes ceil(cnt/32) items of <= 32 rows, each assigned the N16 bucket if its rows <= 16 else N32 - computed on device from the align counts, no host branching on values, buffers sized from T (capacity T*8 + 32*(32-1) rows) so it is graph-capturable; an expert split into several items re-reads its weights (from L2) once per item; (3) the w13+silu and w2 kernels, tiling (BK 128, 4 stages, grid 132 x 384, 1 producer + 2 consumer warpgroups), dequant and epilogues are unchanged; the post-expert norm+combine stays master's _post_norm_combine_kernel (T < 32) / fused_post_expert_norm_weighted_sum (T = 32) fed with y in master's layout. Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets for the gate: 8, 16, 32. Motivation: in the best profile Marlin's pair is 94.6-114 us at bs=8 (floor 37.3), 170-176 at bs=16 (floor 52.5), 156-207 at bs=32 (floor 70.8) - 2.5-3x off the floor, and stage-2 prepare timing put the wgmma pair at 0.57x Marlin at M=128/256; the persistent grid streams each touched expert once regardless of route count, which Marlin's m-block scheme lacks.

## Expected

- `bs=8 *` faster: Marlin pair ~95-114 + act 2.0 us vs floor 37.3; M1 PATH CHOSEN = WGMMA at T=8 with tests 5/5; M2 wgmma pair below Marlin pair; M3 <= 56 us
- `bs=16 *` faster: Marlin pair ~170-176 + 2.2 us vs floor 52.5; M3 <= 79 us
- `bs=32 *` faster: Marlin pair ~156-207 + 2.3 us vs floor 70.8; decode_32 is master's weakest load (~1.03x); M3 <= 106 us
- `bs=1 *` flat: control: below the 8-token gate, master Marlin with w2 atomic
- `bs=128 *` flat: control: T >= 128 is master (s3 owns it)
- `bs=256 *` flat: control

## Risk

Kill: if every T bucket's prepare gate keeps Marlin, or the judged bs=8/16/32 rows are not below master's Marlin pair by more than the ~20% Marlin profile noise, stop the mid-M wgmma line. Risk: at 7-13 touched experts the 132-CTA persistent grid may be under-filled (bs=8 w2: ~7 experts x 32 N-tiles = 224 items, 1.7 waves); the gate protects the headline. Distinct from the held cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8 (widened into broken buckets) and from running s3 (T >= 128, fixes N64/N128).

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042602Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, but do not dispatch until s3 (job 3bdec0e58d684c17) is harvested. Then build on s3's commit, not stage 2's, with s3's per-bucket variant (A, wait<0>; or B, double-buffered A) chosen for N16/N32. The s3 review found a register-A WAR hazard under wait<1> that is undefined behaviour in every instantiation, so stage 2's 'N16/N32 correct' and its 0.57x timing were measured on racy code. Clause (2), the 32-row cap, is a no-op for 8 <= T <= 32, which simplifies the change and lowers its risk.

### Justification
Design (kind: rewrite, stage 3 of the mid-M branch). The target is real. Marlin's pair is 95-114 / 170-176 / 156-207 us at bs=8/16/32, against floors of 37.3 / 52.5 / 70.8 us. These three shapes carry weight 180 of 360. The plan for evidence is good: a prepare gate per T bucket against master's exact path with cold L2, graph replays, interleaved A/B, p50 with a 3% margin, and an fp32-dequant reference with master's error printed beside it. Problems, in order of weight. (1) WEAKEST LINK: both pieces of evidence for N16/N32 come from stage-2 code that still has `wgmma_wait<1>()` in consume_item (s2 submission.py line ~724). The s3 review showed the next k-step's dequant rewrites a[] while the previous group may still read it. That is undefined behaviour in all <NB,G> instantiations, not only N64/N128, so stage 2 passing at N16/N32 may come from register allocation luck. Its prepare timing (0.57x Marlin) was also taken with the hazard present. If variant A turns out to be the fix, the wgmma pipeline loses its one-group overlap and the speed claim must be measured again. s3's per-bucket canary already times both variants at every bucket, N16/N32 included, with a cold-L2 pair time. That is the M2 evidence this hypothesis needs, and it arrives for free. (2) The 0.57x figure was measured at M=128/256 with all 32 experts hit and warm L2. At mid-M, about 7-14 local experts are touched with 1-4 rows each, and a 132-CTA persistent grid over 64 N-tiles x experts items gives roughly 1.7-3.5 waves, with tail imbalance and almost no reuse. The kernel is weight-bandwidth bound there, which is a different regime. (3) The mechanism claim 'Marlin's m-block scheme re-streams experts' does not hold at T <= 16. Every expert has at most 16 rows, so with block_size_m 16 Marlin already reads each expert once. Marlin's gap to the floor there is parallelism and dequant, not re-reads. (4) Counter-evidence: cuda-nvfp4-swapab-mma-persistent-small-m-inproc (0.925x) LOST on verify_1, decode_16 and decode_32, the same three shapes, with a persistent custom NVFP4 MMA. The per-bucket timing gate is what makes this attempt safe, so keep it strict. (5) Clause (2) is dead code in range. decode_item sets the bucket from nbk (16-row blocks): nbk <= 1 gives N16 and nbk <= 2 gives N32. A local expert can receive at most T <= 32 routes (top-8 experts are distinct per token), so nbk <= 2 always, and chunks are already cut every 8 blocks. ceil(cnt/32) is always 1. Implement it as a device assert or err-word bit (nbk > 2 means fall back), not as a new split path with extra capacity (T*8 + 32*31). That removes a new indexing surface. In practice this hypothesis is s2-fix-gate-from-8 restricted to 8..32 plus the timing gate, and that restriction is the correct way to make it distinct. (6) Milestone M3 needs <= 56/79/106 us. At ~1.5x the floor, that requires the persistent grid near HBM bandwidth with 7-14 experts, which is plausible only if the w13 item count fills 132 CTAs. Ask for the per-bucket achieved GB/s to be printed.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, master tip): it defines the incumbent at T <= 32 (w2 with use_atomic_add=True, use_fp32_reduce=False). Stage 2's _wg_time_vs_marlin and its Marlin reference calls predate it. After the rebase, the prepare gate must time master's own _moe_marlin, with the atomic kwargs, not stage 2's copy. Otherwise the 3% margin is measured against a w2 that is slower than the one the judge runs.
- triton-map-align-on-block-m-restack (8440ae5): master's mid-M align is one _align_small_kernel launch (2.7-4.2 us). Stage 2's wgmma path calls SGLang's moe_align_block_size (plus the expert_map gather) at block 16. If that is kept at mid-M, the wgmma arm quietly drops a landed win, and the headline delta mixes kernel time with an align regression of several us. Feed the wgmma path from master's _align_small at block_size_m 16 (master self-checks it against moe_align_block_size), or count the align delta in the replaced-kernel sum.
- post-expert-norm-triton-small-m (192f225): at T < 32 the combine is master's Triton _post_norm_combine_kernel, which skips remote rows by slot id -1. The wgmma w2 epilogue scatters into cache3 by route. Unwritten local rows (cache3 is torch.empty) would be read. Stage 2's sentinel coverage check covered that only at M=128/256, so repeat it at T = 8/16/32.
- marlin-block-m-two-smaller-on-postnorm / align-parallel-per-expert-verify: they act only at T >= 128 / T > 32. Disjoint from the gate, so no confound.

### Interference and A/B needs
No ab-revert or ab-tune companion. This is a rewrite gated per T bucket, so PATH CHOSEN = MARLIN reproduces master byte for byte and is the in-run control. The real attribution problem is the two siblings and s3. (a) Sequencing: wait for s3. s3 edits the same consume_item and wg_kernel source and decides per bucket between A and B. If cap32 is built now on stage 2, it runs the racy kernel and will then conflict with s3 on promotion. If s3 shows neither variant fixes N64/N128, the hazard is not the root cause, but s3's N16/N32 rows still give cap32's correctness and speed baseline. Either way, cap32 gets its base from s3's commit, and its per-bucket variant from s3's log. (b) The three siblings stay attributable this way. tiny-m-bucket16 (T <= 7, scored bs=1 only) is shape-disjoint from this run and w2-only-midm, so it can run in the same batch. midm-cap32 and w2-only-midm cover the same bs=8/16/32 shapes. Run them as two separate checks from the same s3-based base, never stacked on each other or on tiny-m. Each gets its own load_inline module name and a distinct PATH CHOSEN tag (e.g. 'WGMMA-PAIR' vs 'WGMMA-W2'). w2-only-midm is in effect this run's control: it isolates the w2 kernel against Marlin w2, and the remainder (cap32 minus w2-only, per shape) is the w13+silu contribution. That is the sub-change split a 'control' companion would give, so none is minted. Require cap32 to also print separate prepare timings for w13+silu and w2 (the judge profile merges Marlin's two launches and shows wg_kernel<...> rows by template). No promotion of any of the three before s3 is harvested and read.

### Open questions
- Which variant did s3 pick at N16 and N32, and what were its cold-L2 pair times there? If A (wait<0>) is the only passing variant, is the N16/N32 pair still under Marlin at M=128 single-expert? If not, the mid-M line has no M2 path, and cap32 should not be dispatched.
- Does the build log (ptxas -Werror probe) report C7510 'wgmma serialized' for the N16/N32 instantiations under the real 168-register cap (__launch_bounds__(384,1))? Serialization would cap mid-M bandwidth.
- At bs=8 / verify_1, how many local experts does the judge's routing touch (7 or fewer?), and does the w13 item count (64 N-tiles x experts) fill 132 CTAs with a reasonable tail? Print item counts and achieved GB/s per T bucket in prepare.
- Should the synthetic routings for the prepare gate match the judge's load distributions (decode_16, decode_32, verify_1 skew)? A gate won on uniform-random routing may not transfer to the judged load.
- Confirm that the rebase keeps master's _align_small (block 16) as the wgmma path's align at T <= 32, rather than reintroducing SGLang's moe_align_block_size.

