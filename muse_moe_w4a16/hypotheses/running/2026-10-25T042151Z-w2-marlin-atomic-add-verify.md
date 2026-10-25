---
problem: muse_moe_w4a16
slug: w2-marlin-atomic-add-verify
created: 2026-10-25T042151Z
base: master
area: w2 marlin
change: >
  On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by
  giving the verify range its own bound constant W2_ATOMIC_VERIFY = True applying at num_tokens >
  32 (bs=128, 256 and the bs=33/257/skewed-128 tests); T <= 32 keeps master's landed
  W2_ATOMIC_MAX_TOKENS path unchanged. Re-run master's _w2_atomic_self_check at T in (33, 128,
  256, 257) with skewed and random routings, its own 'atomic'/'atomic_zero'/'master' decision and
  PATH CHOSEN line; use_fp32_reduce stays True. Note on bs=8/16: master already applies w2 atomic
  at T <= 32, so there is no smaller-M extension left to test. Mechanism: the verify Marlin rows
  swing 391-451 us at bs=256 across identical-code runs today, the same symptom (a lock-serialized
  reduction slow mode) that w2 atomic removed at bs=1. At bs=128/256, block_size_m is 8/16
  (two-smaller pick) so w2 has many slices per output tile and the fp32 reduce takes the lock
  path.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "Marlin pair 264-292 us; if w2's reduce slow mode exists, -5 to -15 us"
  - shapes: bs=256 *
    direction: faster
    why: Marlin pair 391-451 us run to run; target the low end consistently, -10 to -30 us
  - shapes: bs=1 *
    direction: flat
    why: "control: master's landed path"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042602Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T042602Z
base_sha: 9d3c58e166fc04c1df6f9c7b4eccd2672e82be7c
base_blob: 60dfe35d5417d9c0f89067ff313efdc2c41d03a2
branch: hyp/muse_moe_w4a16/w2-marlin-atomic-add-verify
worktree: .worktrees/muse_moe_w4a16/w2-marlin-atomic-add-verify
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043034Z
impl_commit: dfcd2a53161a12eca630e34c48ea12a4d29ac1cc
submission_sha256: 368af90c389cc11ddb7744c2542f344707fa26ef7ab0a31c5a70d5b6132e52e5
norm_sha256: 50eb74f865262beaeaa444ab71c7530d3a3b78445dc12cb366f2a25a4e2af7d9
submission_blob: 28fdab5be6f60d2b4e91078e80a5bf4e2f8d098e
lint_warnings: 18
attempt_file: 2026-10-25T043034Z-check-w2-marlin-atomic-add-verify.md
prepared_at: 2026-10-25T043034Z
job: ccbcda65da584be0
dispatched_at: 2026-10-25T043040Z
---

# w2-marlin-atomic-add-verify

## Change

On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by giving the verify range its own bound constant W2_ATOMIC_VERIFY = True applying at num_tokens > 32 (bs=128, 256 and the bs=33/257/skewed-128 tests); T <= 32 keeps master's landed W2_ATOMIC_MAX_TOKENS path unchanged. Re-run master's _w2_atomic_self_check at T in (33, 128, 256, 257) with skewed and random routings, its own 'atomic'/'atomic_zero'/'master' decision and PATH CHOSEN line; use_fp32_reduce stays True. Note on bs=8/16: master already applies w2 atomic at T <= 32, so there is no smaller-M extension left to test. Mechanism: the verify Marlin rows swing 391-451 us at bs=256 across identical-code runs today, the same symptom (a lock-serialized reduction slow mode) that w2 atomic removed at bs=1. At bs=128/256, block_size_m is 8/16 (two-smaller pick) so w2 has many slices per output tile and the fp32 reduce takes the lock path.

## Expected

- `bs=128 *` faster: Marlin pair 264-292 us; if w2's reduce slow mode exists, -5 to -15 us
- `bs=256 *` faster: Marlin pair 391-451 us run to run; target the low end consistently, -10 to -30 us
- `bs=1 *` flat: control: master's landed path
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Kill: bs=128 and bs=256 within noise or slower (atomic contention on the 32-expert outputs at 1024-2048 routes can lose). Risk: atomic adds into bf16 at large M may lose precision vs fp32 reduce; the self-check bound 5e-3 vs master catches it.

## Adversarial review

Reviewed 2026-10-25T042602Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. This is a cheap, untested extension of the landed w2 atomic flag to the verify shapes. The mechanism is weak, though, and the expected effect sits at the noise floor at bs=256, so make the run self-interpreting: time w2 master vs atomic in prepare() with CUDA events and print it. Also keep the verify mode decision fully separate from the T<=32 one, and make the run report whether the gate picked master, which would make the run a silent no-op.

### Justification
Plausibility: low to moderate. The hypothesis explains the bs=256 swing by a lock-serialized fp32 reduce. Its evidence is the Marlin pair row moving 391.4 us (FINDINGS best profile, job 1947e3ec) to 451.5 us (a2744bc496b84790 profile) on identical verify code, with timed bs=256 at 454.28 vs 467.80 us. That shows variance, not its cause. Nothing in either profile separates the w2 launch from w13 (one row, calls=2), and nothing measures slice_count. The closest prior evidence points against a reduce bottleneck at verify. w13-marlin-bf16-reduce-verify (use_fp32_reduce=False on w13 at T>=128) was within-ci flat. w13-marlin-atomic-add-small-m was flat. The landed w2 atomic run moved only bs=1 (-3.9 us), and bs=8/16/32 were flat even though they are the shapes with deeper K-slicing than verify. At verify, w2 has more m-blocks (bs=256: about 2048 global routes skewed onto the 32 local experts, block_size_m=32), so with 528 resident CTAs (workspace max_blocks_per_sm=4) and 16-32 n-tiles each, slice_count should be about 1. Atomic add then removes almost no reduce and just turns stores into atomics. A factual error in the change paragraph: block_size_m at bs=128/256 is 16/32, not 8/16. _block_size_m gives 48/64, and the two-smaller pick turns that into 16/32 (bs=257 also 32). Size versus floors: verify_16 is 317.8 us (floor 101.3) and verify_32 is 467.8 us (floor 108.9). Marlin is about 96% of both, so there is headroom, but the claimed -5..-15 us at bs=128 is 1.5-4.7%, and -10..-30 us at bs=256 is 2-6.4%. Identical-code noise at bs=256 is already about 3% (454 vs 468). Only the upper end of the range would be readable per shape, and one load moving 3% moves the geomean about 0.5%, inside the ±1.5% CI. The directions match the shape weights (all 60), and bs<=32 flat is right, because those shapes run master's landed path byte for byte. WEAKEST LINK: the 'swing = lock slow mode' inference. It has no profile or slice_count evidence, and the bs=8-32 flat result plus the w13 verify reduce flat result both suggest the reduce is not where verify time goes. Second weak link: numerics headroom. The landed T<=32 self-check chose atomic with worst rel-L2 vs master 0.00479 against a 5e-3 gate (log a2744bc496b84790 L148). That margin is thin, and at T in {33,128,256,257} with skewed routing the gate may pick 'master', which would make the run flat by construction.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71): same flag, same call, same self-check machinery. Risk 1: if the implementer reuses state['w2_mode'] or makes T<=32 depend on the verify self-check passing, a verify-range failure would turn bs=1 back to master and show up as a bs=1 'regression' that has nothing to do with verify. Keep a separate key (e.g. state['w2_mode_verify']), and leave the T<=32 branch and its self-check untouched. Risk 2: if the verify check picks atomic_zero, cache3.zero_() at bs=256 clears num_tokens*8*4096*2 B = 16.8 MB (8.4 MB at bs=128). That is a ~4-7 us memset, which by itself cancels the low end of the expected gain. Read the memset row.
- marlin-block-m-two-smaller-on-postnorm (cefc282): sets block_size_m 16/32 at verify, which sets the m-block count and therefore slice_count, the only lever atomic add acts on. Under SGLang's 48/64 there would be fewer m-blocks and possibly deeper K-slicing, so a flat result here does not retire the knob for other block_m picks, and a gain does not transfer if block_m is retuned. Fixed on master, so it does not confound this run's sign.
- sglang-marlin-path-parity (a646cb4): the shared workspace (max_blocks_per_sm=4 -> 528 CTAs) sets the grid and therefore slice_count for both calls. Fixed on master, so not a confound for this run, but the result only holds for this workspace.
- align-parallel-per-expert-verify (1a5da3d): produces sorted_ids/block_experts/n_padded at T>32. The padding rows and the -1 sentinel for remote routes decide which C rows Marlin touches. Under atomic add, remote and padded rows must stay untouched exactly as under plain store, or the SGLang combine (slot_expert_ids masking) reads garbage. The T=33/257/skewed self-check covers this as long as it compares the final combined output, not just y.
- post-expert-norm-triton-small-m (192f225): no interaction at T>32, which uses SGLang's fused_post_expert_norm_weighted_sum. It matters only that cache3 still aliases cache13 (non-zero stale cache1 data), so that the verify self-check is a true test of whether Marlin's atomic path pre-zeroes C.

### Interference and A/B needs
No companion is warranted. The ambiguity in a single run is noise at bs=256 plus an unattributed mechanism, and none of ab-revert, ab-tune or control fixes either. Reverting block-m-two-smaller or the small-m atomic patch would only measure a different configuration. A 'use_fp32_reduce=False at w2 verify' control would test a different knob, which is the w2 twin of an already-flat w13 run. The cheaper and stronger fix is in-run: in prepare() (untimed, synthetic inputs only), after the self-check, time the w2 moe_wna16_marlin_gemm call alone with CUDA events at T=128 and T=256. Use the skewed routing, flush L2 with a >50 MB buffer write before each rep, and take the median of ~50 reps each for 'master' and the chosen atomic mode. Print '[w2-atomic-verify] T=128 master X us atomic Y us; T=256 ...'. This gives a paired, same-process A/B of exactly the changed launch. A per-shape gain then counts only if the prepare A/B agrees in sign, and a flat or loss result retires w2 atomic at verify regardless of headline noise. Also print the chosen verify mode and the worst rel-L2 per T. If the verify gate chose master, harvest the run as a no-op, not as 'flat'.

### Open questions
- Spec fix: block_size_m at bs=128/256/257 is 16/32/32 (SGLang 48/64/64, two candidates down), not 8/16 as the change paragraph says.
- What is w2's slice_count at T=128/256 under block_m 16/32 and 528 CTAs? If it is 1, atomic add saves no reduce, and the expected-faster rows have no mechanism. The prepare-time event timing answers this indirectly.
- Will the verify gate pass? T<=32 passed at 0.00479 vs 5e-3. Log the worst rel-L2 at each of T=33/128/256/257 with skewed and random routing. If any T fails, decide up front whether to gate per T (allowed: branching on shape) or treat the whole verify range as master.
- Does the Marlin MoE launcher zero C itself when use_atomic_add=True (the landed check chose 'atomic' with no memset on a stale aliased buffer, which implies it does or that slice 0 plain-stores)? If it zeroes internally, a ~16.8 MB internal memset at bs=256 may hide inside the Marlin row and eat any gain.
- The 391 vs 451 us Marlin-row swing at bs=256 appears in profile rows on identical code. Is it clock or thermal variance? Comparing the SGLang column at bs=256 across the two runs (616.0 vs 629.5 us, also +2%) suggests at least part of it is machine-wide, not a Marlin-specific slow mode.


## Implementation notes

Edited only `muse_moe_w4a16/submission.py` in the worktree (+160/-11). Review overrides applied (the review wins over the spec):
- block_size_m at bs=128/256/257 is 16/32/32 (master's per-T value, unchanged); the spec's 8/16 was wrong. No block_size_m code touched.
- Verify mode is decided apart from the landed one: `state["w2_mode"]` (T<=32, `_w2_atomic_self_check`, untouched) vs new `state["w2_mode_verify"]` = {class: mode} from `_w2_verify_self_check`, wrapped in try/except so an error leaves verify on master and never touches bs<=32. `kernel_fn` picks by `hidden_states.shape[0]` (host shape only): T<=32 -> w2_mode, else verify class (33..127, 128..255, >=256).
- `_forward` takes the mode as an argument; the w2 launch is now a local closure `w2_call(mode)`; the old `num_tokens <= W2_ATOMIC_MAX_TOKENS` test moved into kernel_fn's pick. Same launch/args for master and T<=32.
- Verify self-check: T in (33,128,256,257), random and skewed routing, compares the final combined output vs master. atomic_zero is measured but never chosen (16.8 MB / 8.4 MB memset cancels the gain); if only atomic_zero passes a class stays master.
- Tolerance: rel-L2 vs master <= 5e-3 on the whole output and <= 2.5e-2 on any row (row denominator max(row norm, 0.1*mean row norm)). Justified from TASK: output vs FP32 ref <= 1.5%, any row <= 5x; master's own error spends first, so atomic may add at most a third of each bound (the T<=32 path sat at 4.79e-3 against the same 5e-3).
- In prepare(), w2 alone timed at T=128 and T=256 (skewed) for master/atomic/atomic_zero: CUDA events around the call (memset included for atomic_zero), 256 MB `fill_` before each rep (cold L2), 50 interleaved reps, median. A class takes atomic only if it passes the gate and atomic <= master in this timing (class 33 is not timed; gate only). Extra gate beyond the review, to avoid a loss: a slower atomic stays master.
- Prints: per-class `PATH CHOSEN`, per-T verify mode with worst rel-L2/row for atomic and atomic_zero, the w2-alone timings, and `NO-OP` if every class chose master (harvest as no-op, not flat).
- No CUDA/PTX added, no processes, no mbarriers; the only new kernel work is the existing Marlin launch.

## Rules check

- [x] CUDA-graph capturable: no sync/.item() in kernel_fn's call graph; the mode pick is a host dict lookup on the shape; timing/sync only in prepare().
- [x] Nothing derived from `data` is cached: modes come from synthetic inputs in prepare().
- [x] Weights: unchanged, repacked in prepare() as on master.
- [x] Activations BF16 as master; use_fp32_reduce stays True; only use_atomic_add toggles on w2 (same as landed).
- [x] Attention: n/a, MoE problem.
- [x] No timing patches, harness frames, subprocesses or network (verified against the diff).
- [x] All work inside the timed call; no output caching or replay.
- [x] Any self-check miss or error lands on master's path and never raises.

okc_lint: `OK with warnings` (0 errors; host-sync warnings only in prepare()-side code, one dtype warning for the uint8 L2-flush buffer in prepare()). py_compile OK; __pycache__ deleted.

## Pre-flight audit

Audited 2026-10-24 (opus). **Verdict: dispatch-with-fixes.** Rules OK.

- kernel_fn: host pick on `hidden_states.shape[0]` only; dict lookups, no per-call mutable state, no prints. T<=32 passes `state["w2_mode"]` into the same launch, so it behaves byte for byte like master (master's `num_tokens <= 32` test is now the kernel_fn branch, and the landed self-check only uses T in 1..32). The verify self-check runs in its own try/except, so it never touches `w2_mode`.
- Gate (5e-3 rel-L2, 2.5e-2 row vs master): this is probably not a gate no-op. A numpy bf16 emulation of slice-0 store plus bf16 atomic adds, against an fp32 reduce with one final rounding, gives rel-L2 0 / 2.2e-3 / 3.0e-3 / 3.9e-3 / 5.1e-3 at slice_count 1 / 2 / 4 / 8 / 16. The 16-slice figure matches the landed T<=32 result (4.79e-3). At verify, about 20-50 m-blocks x 16 n-tiles against 528 CTAs means slice_count is about 1-2, so expect <= ~2.5e-3. A looser bound could be justified if needed. The reference uses the same NVFP4 weights in fp32, so master's own error is only bf16 intermediates (~2-3e-3 estimated). 1e-2 would leave sqrt(3e-3^2 + 1e-2^2) ~ 1.05e-2 < 1.5e-2. The constant was left at 5e-3.
- Zeroed C: no zeroed C is needed for "atomic" (no memset). Marlin's slice 0 plain-stores and later slices atomic-add. The landed T<=32 check passed on the stale aliased cache3, which empirically confirms this. The verify self-check also uses the aliased cache3, so it re-tests the same behaviour at T=33..257.
- FIX: timing gate `atomic <= master` changed to `atomic <= master * 1.02` (W2_VERIFY_SLOWER_TOL). At slice_count ~1 the two launches are near-identical, and a bare `<=` on two 50-rep medians would pick master about half the time. That makes a class a silent coin-flip no-op. Now only an atomic path that is clearly slower stays on master. The PATH CHOSEN lines now print both medians.
- lint OK with warnings (prepare-only syncs, uint8 L2-flush buffer); py_compile OK; __pycache__ removed.

Residual risk: at slice_count ~1 atomic add has no mechanism, so bs=128/256 will most likely read flat. The w2-alone A/B timings in the log are the attribution.
