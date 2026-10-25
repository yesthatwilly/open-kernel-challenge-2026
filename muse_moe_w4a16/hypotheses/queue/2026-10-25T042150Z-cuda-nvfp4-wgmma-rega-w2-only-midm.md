---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-w2-only-midm
created: 2026-10-25T042150Z
base: master
area: w2 marlin
change: >
  Per-kernel arm of the mid-M wgmma line: replace ONLY the w2 Marlin call at 8 <= T <= 32 with
  stage 2's wg_kernel<w2>, fed by master's Marlin w13 output after silu_and_mul (cache2) through
  wg_gather. Stage 2's diagnostic already ran exactly this pairing ('w2 fed Marlin's h') and
  passed at N16/N32/N64 (rel 3.4-3.5e-3), so this is the arm most likely to run correctly. Base:
  the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which
  includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for
  byte; rename the load_inline module so it recompiles. Build exactly as stage 2:
  TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma
  bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: own bound constants WG_W2_MIDM_MIN = 8, WG_W2_MIDM_MAX = 32; w13
  Marlin, silu_and_mul, align and the post-norm combine stay master byte for byte (including
  master's w2 use_atomic_add path for T <= 32, which becomes the fallback); work items capped at
  32 rows on device (ceil(cnt/32) items per expert, N16 if <= 16 rows else N32), y written in
  master's [T, 8, H] layout with remote route rows zeroed so the combine is unchanged. Hard gate
  in prepare(): after the correctness self-check passes, time the new path against master's path
  on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph
  replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if
  its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS
  2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The
  self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random
  routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error
  printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without
  PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc
  check, never a submit. T buckets: 8, 16, 32. Compare against w2's own Marlin row (profile the
  two Marlin launches separately once in prepare and print each, since the judge profile merges
  them).
expected:
  - shapes: bs=8 *
    direction: faster
    why: "w2 is ~1/3 of the Marlin pair bytes (8.4 of 25.2 MB per expert): roughly 30-38 us of the ~95-114 us pair; target w2 <= 20 us"
  - shapes: bs=16 *
    direction: faster
    why: w2 share ~55 us of ~170; target <= 30 us
  - shapes: bs=32 *
    direction: faster
    why: w2 share ~55-70 us of ~156-207; target <= 40 us
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w2 (second moe_wna16_marlin_gemm) at 8 <= num_tokens <= 32
review: proceed
status: queued
review_at: 2026-10-25T042602Z
review_companions: null
---

# cuda-nvfp4-wgmma-rega-w2-only-midm

## Change

Per-kernel arm of the mid-M wgmma line: replace ONLY the w2 Marlin call at 8 <= T <= 32 with stage 2's wg_kernel<w2>, fed by master's Marlin w13 output after silu_and_mul (cache2) through wg_gather. Stage 2's diagnostic already ran exactly this pairing ('w2 fed Marlin's h') and passed at N16/N32/N64 (rel 3.4-3.5e-3), so this is the arm most likely to run correctly. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: own bound constants WG_W2_MIDM_MIN = 8, WG_W2_MIDM_MAX = 32; w13 Marlin, silu_and_mul, align and the post-norm combine stay master byte for byte (including master's w2 use_atomic_add path for T <= 32, which becomes the fallback); work items capped at 32 rows on device (ceil(cnt/32) items per expert, N16 if <= 16 rows else N32), y written in master's [T, 8, H] layout with remote route rows zeroed so the combine is unchanged. Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets: 8, 16, 32. Compare against w2's own Marlin row (profile the two Marlin launches separately once in prepare and print each, since the judge profile merges them).

## Expected

- `bs=8 *` faster: w2 is ~1/3 of the Marlin pair bytes (8.4 of 25.2 MB per expert): roughly 30-38 us of the ~95-114 us pair; target w2 <= 20 us
- `bs=16 *` faster: w2 share ~55 us of ~170; target <= 30 us
- `bs=32 *` faster: w2 share ~55-70 us of ~156-207; target <= 40 us
- `bs=1 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

A/B arm for cuda-nvfp4-wgmma-rega-midm-bucket-cap32: separates the w2 kernel's contribution from w13's. Kill: prepare gate keeps Marlin at all three T, or judged rows flat within noise. If the reviewer judges one check enough, run this one first (higher chance of PATH CHOSEN = WGMMA).

## Milestones

Rewrite stage 3 of `Marlin w2 (second moe_wna16_marlin_gemm) at 8 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042602Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, but dispatch only after s3 (job 3bdec0e58d684c17) harvests, rebased onto s3's hazard-free consume_item plus master 9080a71. This is the right first mid-M arm: it uses the w2 instantiations stage 2 proved correct, and its in-prepare interleaved A/B gate makes a Marlin fallback an in-run control. The case for a gain at bs=8 is weak, because the 0.51x w2 advantage stage 2 measured at M=128 comes mostly from Marlin re-reading weights per m-block, and that does not happen at 1-2 routes per expert.

### Justification
Design (kind: rewrite, stage 3). The scored shapes are bs=8/16/32, where Marlin is 2.5-3x off the floor (pair 107/148/199 us against floors 37.3/52.5/70.8). w2 is 8.4 of 25.2 MB per expert, so w2's own floor is about 12.4/17.5/23.6 us. The stated targets of <=20/30/40 us are about 1.6-1.7x that floor and physically reachable. Stage 2's log (archive/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix/583a73418fbf4d01.log lines 14, 79, 82) confirms that w2 fed Marlin's cache2 is correct at N16/N32/N64 (rel 3.4-3.5e-3, 0 unwritten, 0 non-finite). It ran in mixed, multi-chunk launches, so correctness is the strongest part of this arm. WEAKEST LINK, the speed premise: the only wgmma timing evidence is stage 2's warm eager prepare figure (w2 162.6 vs Marlin w2 315.8 us at M=128, 216.4 vs 424.5 at M=256), with three problems. (a) It comes from all-local uniform routings: 32 or 64 routes per expert, where Marlin at block_size_m 16/32 re-streams each expert's weights once per m-block and the persistent wgmma grid streams them once. At bs=8 (~64 routes, ~7-13 touched local experts, 1-2 routes each, block_size_m 8) Marlin already reads each expert once, so most of that 2x is gone. (b) It mostly ran the N128 bucket, which was producing NaN, possibly through an early-release race that would also flatter its speed. (c) In absolute terms it is only ~1.6 TB/s, about what Marlin's pair already reaches at bs=8 (179 MB in ~110 us). So the expected direction is most plausible at bs=32: skewed busiest-GPU routing gives some hot experts >16 routes and BSM 16, so Marlin still re-reads them. It is plausible at bs=16 and weakest at bs=8, where the w2 grid has ~7 experts x 32 N-tiles = ~224 items (1.7 waves on 132 CTAs) and the extra wg_gather + device item builder (stage 2's gather alone was 7 us at M=128) eats a third or more of a ~15 us best-case saving. Include those launches in the replaced-kernel sum. Second link: stage 2's N16/N32 code still contains the register-A WAR hazard that s3's review names (wgmma_wait<1> while the next dequant rewrites a[]). That is undefined behaviour whose effect depends on register allocation, and w2 passing at N64 while w13 failed at N64 is exactly allocation-dependent. Rebuilding under a new module with a new item builder can change allocation. Because the judge also checks every timed call, an intermittent miss on bs=8/16/32 fails the whole run with no fallback. The prepare gate cannot catch a race that only shows up under judge timing. The expected directions (bs=8/16/32 faster, bs=1/128/256 flat controls) are consistent with TASK.md weights (all 60) and floors. The 3% p50 gate keeps a slow kernel off the headline.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, the hand-landed incumbent at T<=32): the arm replaces exactly this call at 8<=T<=32, so the gate's 'master path' must be master's state['w2_mode'] path (atomic or atomic_zero), not plain Marlin. Its own run measured bs=8/16/32 flat (1.0014/1.0044/1.0009), so the incumbent's time at these shapes equals the pre-atomic w2 and no ab-revert is needed. Two traps: keep w2_mode's self-check (it still governs bs=1 and the fallback), and do not write wgmma y into cache3, which aliases cache13. Use a separate y buffer, or zero remote rows in-kernel, since master's atomic path relies on cache13's finite w13 output in those rows.
- post-expert-norm-triton-small-m (SMALL_M_POST_NORM=32): T=8/16 use _post_norm_combine_kernel, which masks remote slots. T=32 uses SGLang's fused_post_expert_norm_weighted_sum, which stage 2 measured as NOT skipping remote rows ('fused norm skips remote rows: False'). Remote rows of the new y must therefore be finite (zeroed) at T=32 at least, or decode_32 goes NaN. A zero memset of 2 MB at T=32 costs ~1-2 us, which counts against the w2 row. The spec already says 'remote route rows zeroed'. Make it an in-kernel or epilogue write, not a separate memset, if possible.
- triton-map-align-on-block-m-restack: Marlin w13 at bs<=16 uses block_size_m 8 and at bs=32 uses 16 from _align_small. The wgmma w2 needs its own sorted/bucketed item list (wg_gather from cache2, which is indexed by route). That adds an align/gather launch master does not have. It is part of the replaced cost and must appear in the per-kernel prepare timing and the profile sum, or the gate compares the wrong things.
- sglang-marlin-path-parity: provides the Marlin bank, the fallback and the reference. The wgmma path keeps a second, losslessly retiled NVFP4 copy of w2, which is rules-OK. No timing confound.

### Interference and A/B needs
No companion. (i) No ab-revert: the only landed change on this call (w2 atomic add) measured flat at 8-32, so reverting it would not change the incumbent. (ii) No ab-tune: block_size_m and the norm split are fixed at these T. (iii) No control: the hypothesis already carries one. Its prepare gate times the new w2 path against master's path per T bucket, interleaved with cold L2 under graph replay. The judged bs=1/128/256 rows stay master byte for byte. A Marlin fallback for a bucket reproduces master there. That makes a gain attributable to w2 alone and a loss attributable to the gate, readable from the PATH CHOSEN lines and the printed per-kernel rows. The natural companion is the sibling cuda-nvfp4-wgmma-rega-midm-bucket-cap32 (w13+w2): w13's share = cap32 minus w2-only on the same base. SEQUENCING: wait for s3 (job 3bdec0e58d684c17, already in flight, no extra check spent). (1) If s3 shows variant A (wait<0>) or B (double-buffered a[]) fixes N64/N128, the hazard is real and the N16/N32 code is passing by luck. Rebase this arm and both siblings onto s3's commit and use the hazard-free variant at N16/N32 too (register cap 168 is not binding at N16/N32: accumulators are 8/16 regs, and B adds 32). (2) s3's per-bucket single-expert canary prints cold-L2 N16/N32 pair times against Marlin. That is the first real timing evidence for these buckets. If w2 at N16/N32 does not beat Marlin w2 there, demote cap32 and treat this run as the kill test. (3) If s3 fails both variants, still apply variant A (wait<0>) to N16/N32 in this arm. It is correct by the PTX ISA, and the cost is a little speed. ATTRIBUTABILITY across siblings: rebuild all three from one base_sha (s3 commit rebased on 9080a71), each with its own load_inline module name and its own bound constants. tiny-m-bucket16 (T<=7) is shape-disjoint from the mid-M arms (8<=T<=32) and from s3 (T>=128), so it can run in the same tick as either mid-M arm. w2-only and cap32 overlap on bs=8/16/32 and are alternatives, not a stack. Run them in the same tick from the identical base, with w2-only first if only one slot is free. Never stack any of the three onto master until each is harvested with its own PATH CHOSEN per bucket.

### Open questions
- Does s3 (3bdec0e58d684c17) confirm the register-A WAR hazard? If yes, rebase all three siblings onto s3's consume_item (variant A or B) before dispatch. If both variants fail, use wait<0> at N16/N32 anyway.
- What are s3's canary cold-L2 single-expert N16/N32 w2 times against Marlin w2? This is the only timing evidence at mid-M bucket sizes. Stage 2's 0.51x was at 32-64 routes per expert, where Marlin re-reads weights.
- What fraction of the judge's bs=8/16/32 Marlin row is w2? Print both Marlin launches separately in prepare, as the spec says. If w2 is under ~30% of the pair at bs=8, the ceiling for this arm at bs=8 is a few us and inside noise.
- Item builder + wg_gather cost at T=8: if it exceeds ~4 us, the bs=8 bucket cannot clear the 3% gate. Consider fusing the gather into the w2 producer (TMA from cache2 rows by sorted route id) in the next stage.
- At T=32 the combine is SGLang's fused norm, which reads remote rows. Confirm the new y buffer's remote rows are written as zero by the kernel/epilogue, not left uninitialised. Print a remote-row finiteness check in the self-check.
- Is the w2 kernel's 8 B spill (stage 2, 168 regs) in the k-loop? At small T the kernel is latency-bound, and a spill in the hot loop costs more than at M=128.

