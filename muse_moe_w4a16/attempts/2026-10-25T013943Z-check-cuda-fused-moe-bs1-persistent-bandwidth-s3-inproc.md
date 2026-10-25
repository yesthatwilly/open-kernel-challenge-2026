# muse_moe_w4a16 · 2026-10-25T013943Z · check · cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T001825Z-cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc.md` (branch `hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc`).

- **Job:** `cd0f6b08a6dd4165` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `e571580`

## Goal
Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART B of the held cuda-fused-moe-bs1-persistent-nospin-bandwidth (split per its review), on top of stage 2 cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc (Part A, slow mode removed): raise bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is 1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight layout, routing and numerics unchanged. Keep stage 2's one-wave sizing and tight poll, plain cp.async.cg and -lineinfo; first launches in-process (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)) with the sm_90a image check and a synthetic canary followed by torch.cuda.synchronize() and a print. Run only after stage 2's log shows which phase stalled and the replay distribution is unimodal. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
What stage 2's log (archive/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc/7a6919420b724ee7.log) showed: the path RAN (arch gate OK, canaries and R in {0,1,2,3,8} passed, PATH CHOSEN = fused, no spin-bound hits). Replays are unimodal (R=2: p10 42.2, p50 42.7, p90 43.6 us; master 51.8). Scores 1.3257 vs best 1.3278, bs=1 row 47.7 us (1.61x). Per-CTA at R=2 (median replay 44.3 us): setup 1.8-1.9 us, phase 1 ends 20.6/21.1/22.6, first phase-2 wait 22.2-23.8 (median wait 1.25 us, spins med 1 / max 38), exit 27.5/33.4/38.8, span 38.9, time - span 5.4. Phase 1 is one 16-stage item per CTA (128 of 132 CTAs): ~19 us is 1.2 us per 18 KB stage, about 2 TB/s with ~7 stages already in flight (16 MB, well above Little's law), so it is dequant/issue-bound, as the review expected, not ring-depth-bound. Work is already balanced (3072 stages / 132 = 23.3; actual max 24). The spin is no longer a stall. Other sizes: R=1 33.9 us, R=3 66.2, R=8 141.6.

Changes (muse_moe_w4a16/submission.py, CUDA source only; kept: arch gate, two explicit template launches, bounded spin, tight poll, one-wave sizing, plain cp.async.cg, -lineinfo, dequant, layout, routing, in-process self-check unchanged):
1. Setup: thread 0 loads all 8 ids and weights up front (independent loads) before compacting, instead of a load-per-iteration loop (setup was 1.9 us; the log shows nothing else in it).
2. Two MMA accumulators (even/odd k16 step; c and cb, summed in the item epilogue) to halve the dependent mma chain at 2 warps per scheduler. fp32 summation order changes (review noted this is within the 1.5% bound; weights and scale products are unchanged).
3. STAGES 8 -> 10 (SMEM 158208 -> 195072 B, still 1 CTA/SM, under the 227 KB limit): the ring now holds a CTA's whole phase-2 share (8 stages at R=2), so after the phase-1 wait phase 2 is compute-only.

Deviations from the spec / review: no cross-CTA K-split (review: contradicts one-wave sizing, reintroduces the stall; the log shows 128 items already fill the SMs at R=2, so it would not help). No intra-CTA K-split or 384-512 threads: not buildable blind (ROWS tile = 16 x WARPS fixes layout) and issue count per weight would not drop; left as the open question, to be answered by this run's phase times. Biggest uncertainty: dequant instruction count is unchanged, so the gain may be small (est. 1-4 us); the self-check prints per-phase times as in stage 2 for reading it.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff touches only the kernel body (device code); no host calls added in kernel_fn.
- [x] No host syncs in kernel_fn: none added (lint warnings are prepare()-only).
- [x] Nothing derived from `data` is cached across calls: routing still read on device every launch.
- [x] Weights: repacked losslessly in prepare() only; NVFP4 stays NVFP4: layout and dequant unchanged.
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] No child process/subprocess/multiprocessing: none added; self-check stays in-process.
- [x] No timing patches, harness frames/IO/secrets, reference import: none in diff.
- [x] All work inside the timed call; no output caching or replay: unchanged.
- n/a: attention items (KV cache FP8, full window): MoE problem.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all "outside kernel_fn's call graph; fine in prepare()"). py_compile rc=0, __pycache__ removed.

## Risks / what could fail
high
split from cuda-fused-moe-bs1-persistent-nospin-bandwidth (Part B), held for a child-process self-check; stage 3 after cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Result (after the run)
- job `cd0f6b08a6dd4165` (check), status done, passed True, tests 5/5
- speedup 1.3295x, 95% CI [1.3095, 1.3494], score 10115.528742282053 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc.patch`; branch `hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc`
