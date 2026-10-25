# muse_moe_w4a16 · 2026-10-25T041252Z · check · cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T035329Z-cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix`).

- **Job:** `3bdec0e58d684c17` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `8261c04`

## Goal
Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrite on the parent commit); rename the module so it recompiles. Stage 2 (job 583a73418fbf4d01) diagnosed the bug: build sm_90a clean, w13+silu 168 regs/0 spill, w2 168 regs/8 B spill, cuTensorMapEncodeTiled x8 rc 0, device error word 0, gather clean (0/644 bad rows, pads zero, 0 remote writes); the 16- and 32-row token buckets are correct (w13 rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3), but w13 is all NaN/inf at the 64- and 128-row buckets and w2 at 128 only; whole rows are non-finite in every N-tile, i.e. a per-bucket (M-tile instantiation) error. Change ONLY what differs between the 16/32 and the 64/128 instantiations. First, in the prepare canary, per kernel and per bucket (one-expert routings that force M=64 and M=128 into a single bucket, w13 and w2 separately, w2 fed Marlin h), after its own sync + flushed print: first non-finite (row, col), whether the whole row is NaN vs scattered inf, and for the failing bucket print the template parameters actually instantiated (wgmma n-shape, accumulator register count per thread, B descriptor LBO/SBO/swizzle, expect_tx bytes per stage, smem offsets of every A/B/slab carve-out vs total dyn smem). Then fix, checking each item against the passing 16/32 path and an offline numpy emulation before dispatch: (1) the wgmma instruction shape at wider N: m64n64k16/m64n128k16 need 32/64 fp32 accumulator registers per thread; the fragment-to-(row,col) mapping and the epilogue loop must use the bucket N, not a hard-coded 16/32 extent; (2) the B shared-memory matrix descriptor at wider N: leading/stride byte offsets and swizzle mode must match what TMA wrote for the wider box (128B swizzle with SBO = 8 rows x 128 B; LBO between core-matrix columns), and the start-address advance per k16 step; (3) the TMA box / expect_tx bytes per stage must equal the real bytes landed for 64/128 rows (a short expect_tx releases the barrier before the data lands; a long one hangs, but the error word was 0, so look for short); (4) the smem carve-out beyond 32 rows: the slab/A region for 64/128 rows must not overlap a B ring stage (w13 8192+187392 B, w2 8192+203776 B; print offsets and assert no overlap), and the second consumer warpgroup row base (rows 64..127) must point into its own slab rows; (5) the w2 8 B spill: confirm it is not the 128-row accumulator (spilled accumulators are fine numerically, but check the index). Keep the both-kernels-pass gate: WGMMA is chosen only if every bucket passes for w13 and w2 and the end-to-end canary is finite at rel L2 within the existing bound; else PATH CHOSEN = MARLIN with the failing (kernel, bucket, cause) printed. When chosen, print cold-L2 CUDA-event times vs Marlin at M=128 and 256 (stage 2 warm eager figures were 515.9 vs 911.3 us at M=128 and 728.0 vs 1258.5 at M=256; the judge reads 20-25% higher than prepare). Gate unchanged (num_tokens >= 128); Marlin below. Build as stage 2 (sm_90a only, -lineinfo, in-process, never a child process). First measurement is an okc check, never a scored submit.

## What changed, and where
Edited only `muse_moe_w4a16/submission.py` in the worktree (+289/-65). Extension renamed `muse_moe_w4a16_wgmma_rega_v3` so it recompiles; sm_90a-only build, `-lineinfo`, in-process, no child process (unchanged from s2).

**The review wins over the spec.** Suspects (3) expect_tx and (4) slab carve-out are demoted to one print per bucket (instantiated NA, tx bytes, TMA box, stage/slab offsets, dyn smem) in the canary. (2) the descriptor is the same for every N and (1) the asm is generated per NB, so neither changed. The fix targets the review's register-A WAR hazard (`consume_item` ran `wgmma_wait<1>`, then the next k-step's dequant rewrote `a[]` while the previous group might still read it).

What changed:
- `consume_item<NB, G, V>`: dequant split into `dq_stage`; one `step` lambda. V=1 (variant A): `wgmma_wait<0>` right after the commit, so `a[]` is never rewritten with a group in flight, and the stage is released at once (the wait sits at the end of the iteration, which is the same ordering as "before the next dequant"). V=2 (variant B): `a0`/`a1` = a[2][8][4], k loop unrolled by 2 (odd KS tail uses a0), `wgmma_wait<1>` kept, so a buffer is rewritten only after the group that read it retired.
- `wg_kernel` / `wg_gemm` take `vmask` (2 bits per N bucket: 1 = A, 2 = B); dispatch picks the variant per `m.bucket` (16 instantiations per kernel pair instead of 8: expect a longer build, ptxas time unmeasured).
- `_wg_bucket_canary` (new, runs first in `_wg_setup`, before `_wg_diag`): for variants A and B and each bucket, one expert gets 16/32/64/128 routes (M=128, others remote) so the launch is a single bucket; judges w13+silu, w2 fed Marlin cache2 and the production out against Marlin, printing max-abs, rel-L2, first non-finite (route, token, col, tile, bucket) and whole-row vs scattered, ref non-finite counts, and the cold-L2 pair time. Per bucket it picks the fastest passing variant, written to `state["wg_vmask"]` and used by `_moe_wgmma`.
- `_wg_cmp` records `max_abs`.
- `_wg_time_vs_marlin` adds cold-L2 per-launch medians (192 MB flush) beside the warm ones and returns whether the wgmma pair beats the Marlin pair at M=128 and 256 (warm). The path is turned off if not (LESSONS 163).

Deviations:
- "Else Marlin for that bucket" is whole-path: kernel_fn cannot branch on a data-dependent bucket without a host sync, so any bucket with no passing variant sets PATH CHOSEN = MARLIN and prints each failing (kernel, bucket, variant, cause). Later diagnostics still run with variant B for such a bucket.
- No offline numpy emulation (no GPU or torch on this node); not run, nothing here has executed.
- Not added: the profile-kernel-name assertion from the review's last open question, and a grep for ptxas 'serialized'/C75xx beyond the existing -Werror probe.
- The reference-finite question is answered at runtime: `ref non-finite h/y2` is printed per canary case and a non-finite reference fails the case with its own cause.

Biggest risk: build time and registers for 16 instantiations (variant B holds 64 extra a[] registers beside up to 64 accumulators under a 232-register cap, w2 already spilled 8 B), and the budget (`WG_SETUP_BUDGET_S` 200 s) now also covers 8 canary cases. If the hazard is not the cause, both variants fail at N64/N128 and the log names the cause per bucket.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn / `_moe_wgmma` only add a constant `state["wg_vmask"]` int argument; every sync/.item() is in prepare-only code (lint: all warnings are "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached: vmask is chosen in prepare from synthetic seeded inputs.
- [x] Weights: unchanged lossless retiling; NVFP4 stays NVFP4, dequant is exact.
- [x] Activations BF16 as before; no precision change (diff touches only register-A hazard handling and prepare diagnostics).
- [x] Attention: n/a, MoE problem.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (diff grep for subprocess/fork/multiprocessing/autotune: none).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a-only PTX stays inside `__CUDA_ARCH_FEAT_SM90_ALL` guards with the error-word-2 `#else`; `TORCH_CUDA_ARCH_LIST=9.0a` set and restored around load_inline (unchanged `_wg_build`); `-lineinfo` kept; mbarrier waits remain bounded.
- [x] Any build error, canary miss or unsupported shape lands on Marlin without raising; PATH CHOSEN line printed last with the per-bucket variants.

okc_lint output: `OK with warnings: muse_moe_w4a16/submission.py` (0 errors; ~50 warnings, all host-sync in prepare-only code or uint8 scratch dtypes, same classes as the parent). `python3 -I -m py_compile` passed; `__pycache__` removed. Not committed, okc not run.

## Risks / what could fail
high
The cause is inferred from bucket behaviour, not a named line; several suspects may need fixing. Prepare timings are warm and optimistic (the log warns a race releasing early would also look fast), so a pass with an implausibly fast time should be read with the finiteness/rel-L2 evidence.

## Result (after the run)
- job `3bdec0e58d684c17` (check), status done, passed False, tests 5/5
- best at harvest: {"speedup": 1.344525539017997, "ci": [1.324357655932727, 1.364693422103267], "job": "a2744bc496b84790"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix.result.json
- **outcome: failed** (test:failed)
- files: `muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix.result.json`, `muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix`
