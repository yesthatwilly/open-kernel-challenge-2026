---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
created: 2026-10-25T035329Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit
  (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrite on the parent
  commit); rename the module so it recompiles. Stage 2 (job 583a73418fbf4d01) diagnosed the bug:
  build sm_90a clean, w13+silu 168 regs/0 spill, w2 168 regs/8 B spill, cuTensorMapEncodeTiled x8
  rc 0, device error word 0, gather clean (0/644 bad rows, pads zero, 0 remote writes); the 16-
  and 32-row token buckets are correct (w13 rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3), but w13 is all
  NaN/inf at the 64- and 128-row buckets and w2 at 128 only; whole rows are non-finite in every
  N-tile, i.e. a per-bucket (M-tile instantiation) error. Change ONLY what differs between the
  16/32 and the 64/128 instantiations. First, in the prepare canary, per kernel and per bucket
  (one-expert routings that force M=64 and M=128 into a single bucket, w13 and w2 separately, w2
  fed Marlin h), after its own sync + flushed print: first non-finite (row, col), whether the
  whole row is NaN vs scattered inf, and for the failing bucket print the template parameters
  actually instantiated (wgmma n-shape, accumulator register count per thread, B descriptor
  LBO/SBO/swizzle, expect_tx bytes per stage, smem offsets of every A/B/slab carve-out vs total
  dyn smem). Then fix, checking each item against the passing 16/32 path and an offline numpy
  emulation before dispatch: (1) the wgmma instruction shape at wider N: m64n64k16/m64n128k16 need
  32/64 fp32 accumulator registers per thread; the fragment-to-(row,col) mapping and the epilogue
  loop must use the bucket N, not a hard-coded 16/32 extent; (2) the B shared-memory matrix
  descriptor at wider N: leading/stride byte offsets and swizzle mode must match what TMA wrote
  for the wider box (128B swizzle with SBO = 8 rows x 128 B; LBO between core-matrix columns), and
  the start-address advance per k16 step; (3) the TMA box / expect_tx bytes per stage must equal
  the real bytes landed for 64/128 rows (a short expect_tx releases the barrier before the data
  lands; a long one hangs, but the error word was 0, so look for short); (4) the smem carve-out
  beyond 32 rows: the slab/A region for 64/128 rows must not overlap a B ring stage (w13
  8192+187392 B, w2 8192+203776 B; print offsets and assert no overlap), and the second consumer
  warpgroup row base (rows 64..127) must point into its own slab rows; (5) the w2 8 B spill:
  confirm it is not the 128-row accumulator (spilled accumulators are fine numerically, but check
  the index). Keep the both-kernels-pass gate: WGMMA is chosen only if every bucket passes for w13
  and w2 and the end-to-end canary is finite at rel L2 within the existing bound; else PATH CHOSEN
  = MARLIN with the failing (kernel, bucket, cause) printed. When chosen, print cold-L2 CUDA-event
  times vs Marlin at M=128 and 256 (stage 2 warm eager figures were 515.9 vs 911.3 us at M=128 and
  728.0 vs 1258.5 at M=256; the judge reads 20-25% higher than prepare). Gate unchanged
  (num_tokens >= 128); Marlin below. Build as stage 2 (sm_90a only, -lineinfo, in-process, never a
  child process). First measurement is an okc check, never a scored submit.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "M1: every bucket finite and within bound, PATH CHOSEN = WGMMA, tests pass; M2: wgmma pair below Marlin pair + act_and_mul (prepare 515.9 vs 911.3 us warm); floor 101.3 us, M3 <= 170 us"
  - shapes: bs=256 *
    direction: faster
    why: "M2: below Marlin (prepare 728.0 vs 1258.5 us warm); floor 108.9 us, M3 <= 220 us"
  - shapes: bs=1 * bs=8 * bs=16 * bs=32 *
    direction: flat
    why: below the num_tokens >= 128 gate; Marlin unchanged
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: proceed
status: running
review_at: 2026-10-25T040253Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T040259Z
base_sha: 570f5e4ac5465931183addaf7a472e01668ced15
base_blob: 1ae7e9cac6dc078a0e71c7a6b0b77ec48b0ed01f
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T041251Z
impl_commit: 8261c0480931bf1fdcb44c0d04d1dbf675b0c06a
submission_sha256: e17235b17240bad1393d762e3408383cdb40d2ac6ce1cdb4d24ff5e2c096fc69
norm_sha256: c43518a43fe432fcc81bf0ea64a290d0c5308b6493625d7ba191ad075b77351b
submission_blob: bf65a3afdf6ebe0d9b73a817ffd2be1240f91870
lint_warnings: 50
attempt_file: 2026-10-25T041252Z-check-cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix.md
prepared_at: 2026-10-25T041252Z
---

# cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix

## Change

Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrite on the parent commit); rename the module so it recompiles. Stage 2 (job 583a73418fbf4d01) diagnosed the bug: build sm_90a clean, w13+silu 168 regs/0 spill, w2 168 regs/8 B spill, cuTensorMapEncodeTiled x8 rc 0, device error word 0, gather clean (0/644 bad rows, pads zero, 0 remote writes); the 16- and 32-row token buckets are correct (w13 rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3), but w13 is all NaN/inf at the 64- and 128-row buckets and w2 at 128 only; whole rows are non-finite in every N-tile, i.e. a per-bucket (M-tile instantiation) error. Change ONLY what differs between the 16/32 and the 64/128 instantiations. First, in the prepare canary, per kernel and per bucket (one-expert routings that force M=64 and M=128 into a single bucket, w13 and w2 separately, w2 fed Marlin h), after its own sync + flushed print: first non-finite (row, col), whether the whole row is NaN vs scattered inf, and for the failing bucket print the template parameters actually instantiated (wgmma n-shape, accumulator register count per thread, B descriptor LBO/SBO/swizzle, expect_tx bytes per stage, smem offsets of every A/B/slab carve-out vs total dyn smem). Then fix, checking each item against the passing 16/32 path and an offline numpy emulation before dispatch: (1) the wgmma instruction shape at wider N: m64n64k16/m64n128k16 need 32/64 fp32 accumulator registers per thread; the fragment-to-(row,col) mapping and the epilogue loop must use the bucket N, not a hard-coded 16/32 extent; (2) the B shared-memory matrix descriptor at wider N: leading/stride byte offsets and swizzle mode must match what TMA wrote for the wider box (128B swizzle with SBO = 8 rows x 128 B; LBO between core-matrix columns), and the start-address advance per k16 step; (3) the TMA box / expect_tx bytes per stage must equal the real bytes landed for 64/128 rows (a short expect_tx releases the barrier before the data lands; a long one hangs, but the error word was 0, so look for short); (4) the smem carve-out beyond 32 rows: the slab/A region for 64/128 rows must not overlap a B ring stage (w13 8192+187392 B, w2 8192+203776 B; print offsets and assert no overlap), and the second consumer warpgroup row base (rows 64..127) must point into its own slab rows; (5) the w2 8 B spill: confirm it is not the 128-row accumulator (spilled accumulators are fine numerically, but check the index). Keep the both-kernels-pass gate: WGMMA is chosen only if every bucket passes for w13 and w2 and the end-to-end canary is finite at rel L2 within the existing bound; else PATH CHOSEN = MARLIN with the failing (kernel, bucket, cause) printed. When chosen, print cold-L2 CUDA-event times vs Marlin at M=128 and 256 (stage 2 warm eager figures were 515.9 vs 911.3 us at M=128 and 728.0 vs 1258.5 at M=256; the judge reads 20-25% higher than prepare). Gate unchanged (num_tokens >= 128); Marlin below. Build as stage 2 (sm_90a only, -lineinfo, in-process, never a child process). First measurement is an okc check, never a scored submit.

## Expected

- `bs=128 *` faster: M1: every bucket finite and within bound, PATH CHOSEN = WGMMA, tests pass; M2: wgmma pair below Marlin pair + act_and_mul (prepare 515.9 vs 911.3 us warm); floor 101.3 us, M3 <= 170 us
- `bs=256 *` faster: M2: below Marlin (prepare 728.0 vs 1258.5 us warm); floor 108.9 us, M3 <= 220 us
- `bs=1 * bs=8 * bs=16 * bs=32 *` flat: below the num_tokens >= 128 gate; Marlin unchanged

## Risk

The cause is inferred from bucket behaviour, not a named line; several suspects may need fixing. Prepare timings are warm and optimistic (the log warns a race releasing early would also look fast), so a pass with an implausibly fast time should be read with the finiteness/rel-L2 evidence.

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** Diagnosis is right at the bucket level and the upside is large (prepare 0.57x of Marlin at M=128/256). Reading the s2 kernel rules out two of the five listed suspects (expect_tx and smem carve-out are bucket-independent). It also adds a stronger one: register-A operands are overwritten by the next k-step's dequant while the previous wgmma group may still be reading them (wait_group 1), which is undefined behaviour that shows up per <NB,G> instantiation.

### Justification
Stage 2's per-bucket table localises the bug. N16/N32 are correct, w13 fails at N64 and N128, w2 only at N128, and whole rows go non-finite in every N-tile. That is a per-instantiation fault, and the hypothesis targets exactly the NB-dependent code. From the s2 source (consume_item<NB,G>, producer loop, wg_set_maps): suspect (3) is ruled out, because tx = WST + 2*(16<<bucket)*128 equals the two TMA boxes (64 bf16 x rows) plus the weight record exactly. Suspect (4) is ruled out too: the slab is a fixed 2 x SLAB_HALF (128 rows x 128 B) per stage for every bucket, there is no per-bucket carve-out, and the second consumer warpgroup's base is a weight-row offset (cw*512) independent of the bucket. Suspect (2) is unlikely, because make_desc is identical for every N (K-major SW128, SBO 1024) and is already proven at N16/N32. Suspect (1) reads correct: the asm is generated per NB with na = NB/2 operands, and the epilogue loops use NB/8. WEAKEST LINK, and a missing suspect: in consume_item the loop dequantizes the next stage into a[8][4], issues 8 RS wgmmas, commits, then wgmma_wait<1>. That leaves the just-issued group in flight while the next iteration's dequant8 rewrites a[] (the same physical registers, since the kb loop is not unrolled). The PTX ISA makes touching A-fragment registers of an in-flight wgmma undefined. Whether ptxas reuses those registers depends on allocation, which differs per <NB, G>; that would explain w13 failing at 64 while w2 passes at 64. Wider N also makes each wgmma longer, which widens the race window. A transient register holding bf16x2_bcast(scale * 2^119) read as A would give inf, which silu turns into whole-row NaN. The milestones are plausible if fixed: the floor is 101/109 us, and Marlin's pair is 911/1259 us warm.

### Confounders on the current kernel
- align-parallel-per-expert-verify (master, BEST): it produces the sorted_ids/expert blocks that wg_gather and the bucket split consume at T > 32. A gain at bs=128/256 includes its align, which stays master in both arms. No confound for the stage's own milestones (kernel rows vs the Marlin pair).
- marlin-block-m-two-smaller-on-postnorm (master): it changes Marlin's block_size_m at T >= 128. The incumbent to beat is that tuned Marlin pair plus silu_and_mul, so compare to the profile Marlin rows of this batch, not to stage 2's prepare figures.
- w2-marlin-atomic-add-small-m (land candidate): bs=1 only. Disjoint from the num_tokens >= 128 gate.

### Interference and A/B needs
No companion. This is a rewrite correctness stage, and the both-kernels-pass gate makes the fallback a free in-run control (PATH CHOSEN = MARLIN reproduces master). The per-bucket canary already forces every bucket via one-expert routings. The extra discriminator below belongs inside the canary, not in a second run.

### Open questions
- BUILDER (first fix to try, cheap and decisive): eliminate the register-A WAR hazard. Either wgmma_wait<0>() before writing a[] for the next k-step (a quick test), or double-buffer a as a[2][8][4] with the kb loop unrolled by 2 so the in-flight group's registers are never rewritten. Also grep the ptxas output for 'serialized' / C75xx. Re-run the per-bucket canary for both variants inside the same prepare so the log shows whether N64/N128 turn finite.
- BUILDER: skip suspects (3) expect_tx and (4) slab overlap, or demote them to one print. The s2 code shows tx and the slab layout are identical across buckets. Print the instantiated NA and tx once per bucket as evidence.
- The s2 log says 'the fp32-dequant reference also reads inf on those'. Confirm the reference itself is finite on the skewed/all-one-expert routings (Marlin passes the judge). If the reference is inf, the comparison metric is broken, not just the kernel.
- When chosen, verify the cold-L2 time is plausible against bytes: a race that releases early would also look fast. The gate should also assert the profile kernel name in one probe call (LESSONS 164).


## Implementation notes

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

## Rules check

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

## Pre-flight audit

Audited 2026-10-25 by reading the code (model opus). No GPU here, so nothing was run; a small Python pipeline simulation checked the ordering.

**Verdict: dispatch.** No code changes. okc_lint reports `OK with warnings` (all warnings are prepare-only host syncs), `py_compile` passes, and no `__pycache__` was left behind.

- **Variant A (wgmma_wait<0>).** Correct. The commit is followed by wait<0>, then `fence_acc`, then the stage-s release, all before the next `dq_stage` writes `a0`. `prev` stays -1, so the tail `mbar_arrive` never double-releases. A also removes the other per-instantiation suspect: accumulator chaining across an in-flight group (s2's wait<1>). So if A passes and B fails at N64/N128, the culprit is the accumulators or something else, not the A registers.
- **Variant B (a[2][8][4], unrolled by 2, wait<1>).** Correct at the PTX level. The simulation covered KS = 1, 2, 3, 12, 16, 31 and 32, starting at n = 5. A buffer is never written while the group reading it is in flight. The odd tail `step(a0)` comes after `a1`'s commit and wait<1>, which retires `a0`'s group. Every stage gets exactly one arrive per warp, and only after its group has retired. `wgmma_fence` comes after each `dq_stage` and before every batch of 8 register-A wgmmas. Items are separated by the trailing wait<0>, so mixing variants per bucket inside one launch is safe. The double buffer holds because `a0` and `a1` are separate SSA values, so they get separate PTX virtual registers. Keeping physical registers live for an in-flight wgmma is ptxas's job: it either does it or serializes the wgmma and warns.
- **Register cap.** The real cap is 168, not 232. `__launch_bounds__(384, 1)` makes ptxas compile the whole kernel for 168 registers, which matches s2's numRegs; setmaxnreg.inc 232 does not change that. B adds 32 registers over A, not 64 (64 is B's A-fragment total). At N128, B holds 64 accumulators and 64 A-fragment registers, plus overhead, which is roughly 150-170 of 168. That leaves two outcomes: spills, or the function-wide C7510-class "wgmma serialized" warning.
- **Build and B's effect on A.** Neither outcome fails the build: spills are not ptxas errors. A warning does fail the `-Xptxas -Werror` probe, and the unflagged second build then runs, still within the budget. A serialization warning would also slow A and the 16/32 buckets, because it applies to the whole kernel. That costs speed, not correctness, and the log names it. So B cannot take A down, but B at N128 could make the M2 timing unreadable. If the log shows `serialized: yes`, the next stage should drop B at N128 (instantiate `consume_item<128, G, 1>` only) or build B as its own module.
- **Budget.** s2 built in 37.6 s with 8 instantiations, and setup took 38.9 s in total. With 16 instantiations, expect about 50-75 s per build, or about 150 s for probe plus rebuild. The rebuild gate is 140 s at its start, the post-canary check is at 200 s and timing is at 255 s. The 8 canary cases (Marlin reference, 4 launches, 12 cold launches each) add under about 10 s. Prepare should stay well under 300 s.
- **Canary routing.** Routing c = 16/32/64/128 routes from distinct tokens to local expert 3 at M = 128, with all other routes remote, gives 1/2/4/8 blocks of 16. That is a single chunk at bucket 0/1/2/3, and `buckets_seen == {b}` is asserted per case. w13 and w2 (fed Marlin's cache2) are judged separately, so the s2 failures (w13 at N64/N128, w2 at N128) are reproduced in isolation. Mixed-bucket and multi-chunk launches are still covered by `_wg_diag` and the self-check, which remain in the gate.
- **Gates and fallback.** Nothing always fails. Every key that `_wg_cause` reads exists in `_wg_eval_case`, and `err` is zeroed per case. `t` is NaN only on failing cases, which are never min-picked. A case that raises is caught. Any build, canary, diagnostic, self-check or timing miss sets `ok = False`, and the run prints `PATH CHOSEN = MARLIN` along with the failing (kernel, bucket, variant, cause).
- **Rules.** OK. kernel_fn adds only the prepare-time constant `state["wg_vmask"]` (a Python int, graph-safe). The canary uses seeded synthetic inputs. Weights stay NVFP4 and activations stay BF16.

Residual risk: the device lambda `step` (3 call sites in B) must be inlined. If NVVM does not inline it, ptxas serializes the wgmma for crossing a call, which is slow but correct. ptxas may also serialize the whole kernel because of B at N128. Either outcome shows as a failed `-Werror` probe in the log. And if the real fault is neither the A-register nor the accumulator hazard, both variants fail at N64/N128, and the log names the cause per bucket.
