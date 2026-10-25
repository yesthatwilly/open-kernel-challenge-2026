---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-on-master-small-m
created: 2026-10-25T042150Z
base: master
area: w13 marlin
change: >
  On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the
  FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 only, under its own bound
  constant W13_ATOMIC_MAX_TOKENS = 32; keep use_fp32_reduce=True (as the landed w2 change does -
  Marlin accepted it and 5/5 tests passed). Extend master's _w2_atomic_self_check pattern into a
  _w13_atomic_self_check: candidates 'atomic' (no memset) and 'atomic_zero' (cache13/cache1 zeroed
  before the call), each checked against master on the same synthetic T in (1, 4, 8, 16, 32)
  routings with rel-L2 <= 5e-3 and finiteness; print PATH CHOSEN and whether a memset is in the
  path. Nothing else changes. Mechanism: the landed w2 change removed most of bs=1's +23 us judge
  slow mode (44/100 -> 10/100 replays) by replacing the fp32-reduce lock serialization; w13 still
  uses the lock path, the likely source of the remaining 10/100. The earlier w13-only atomic test
  (0.987x) ran on the pre-post-norm base before the slow mode was measured, so this is a new
  measurement.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "remaining slow mode 10/100 -> ~0 and w13 slice reduction cheaper: 44.9 -> ~42-43 us"
  - shapes: bs=8 *
    direction: faster
    why: "small: w13 lock reduction at 8 tokens; 1-3 us at most"
  - shapes: bs=16 *
    direction: flat
    why: w2 atomic did not move it; expected noise
  - shapes: bs=32 *
    direction: flat
    why: noise
  - shapes: bs=128 *
    direction: flat
    why: control, T > 32
  - shapes: bs=256 *
    direction: flat
    why: control
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042601Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T042601Z
base_sha: 001ddce7af3678bd9f650680c4da3a693cea5170
base_blob: 60dfe35d5417d9c0f89067ff313efdc2c41d03a2
branch: hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m
worktree: .worktrees/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043232Z
impl_commit: c3a8442027474995381e941c6d3a09c4275823d2
submission_sha256: dba4831298c15e97cf3482be3f7ad594d858704769341f7ed01d10a2985f48af
norm_sha256: 1e8f2379b57178ccc0a805301a252b44b53a86d314f514cd8eecb00631742124
submission_blob: 4ee9fda80c96b0149edd61482cbab157c48d39d8
lint_warnings: 23
---

# w13-marlin-atomic-add-on-master-small-m

## Change

On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 only, under its own bound constant W13_ATOMIC_MAX_TOKENS = 32; keep use_fp32_reduce=True (as the landed w2 change does - Marlin accepted it and 5/5 tests passed). Extend master's _w2_atomic_self_check pattern into a _w13_atomic_self_check: candidates 'atomic' (no memset) and 'atomic_zero' (cache13/cache1 zeroed before the call), each checked against master on the same synthetic T in (1, 4, 8, 16, 32) routings with rel-L2 <= 5e-3 and finiteness; print PATH CHOSEN and whether a memset is in the path. Nothing else changes. Mechanism: the landed w2 change removed most of bs=1's +23 us judge slow mode (44/100 -> 10/100 replays) by replacing the fp32-reduce lock serialization; w13 still uses the lock path, the likely source of the remaining 10/100. The earlier w13-only atomic test (0.987x) ran on the pre-post-norm base before the slow mode was measured, so this is a new measurement.

## Expected

- `bs=1 *` faster: remaining slow mode 10/100 -> ~0 and w13 slice reduction cheaper: 44.9 -> ~42-43 us
- `bs=8 *` faster: small: w13 lock reduction at 8 tokens; 1-3 us at most
- `bs=16 *` flat: w2 atomic did not move it; expected noise
- `bs=32 *` flat: noise
- `bs=128 *` flat: control, T > 32
- `bs=256 *` flat: control

## Risk

Kill: bs=1 not below 44.9 us by > 1.5% and its slow-mode count not below 10/100; then the w13 lock path is not the source. Risk: if only 'atomic_zero' passes, the memset costs ~1-2 us and may cancel the gain.

## Adversarial review

Reviewed 2026-10-25T042601Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. use_atomic_add matters for w13 exactly as much as for w2: in both calls it only changes Marlin's in-GEMM split-K (k-slice) reduction, and that path is active at bs=1 (estimated slice_count >1) but probably degenerate from bs=8 up. The expected bs=1 gain is overstated. The scorer uses a median, so removing a 10/100 slow-mode tail barely moves it, and the fast-mode w13 effect has already measured at about 0. Read the raw bs=1 slow-mode count and p10/p50, not the headline.

### Justification
WHAT ATOMIC ADD DOES HERE: In Marlin, use_atomic_add only replaces the global reduction of K-slice partials for one (m-block, n-tile) output tile. Without it, partials go through the workspace lock in slice order (barrier_acquire -> fp32 c_tmp read/add/write -> release). With it, slice 0 stores and the other slices atomicAdd bf16x2 into C. It is NOT a reduction over experts or over top_k routes. Both moe_wna16_marlin_gemm calls write one row per (token, route): w13 with top_k=8 rows, w2 with top_k=1 rows. The sum over routes happens later, in _post_norm_combine / fused_post_expert_norm_weighted_sum. So it makes no difference that w13 feeds silu_and_mul rather than a sum: the knob is purely about split-K inside the w13 GEMM (N=2I=8192, K=H=4096). DOES THE PATH EXIST AT THESE SHAPES: At bs=1 under ep8 there is about 1 local route, so 1 m-block (block_size_m 8). That gives 1 x (8192/thread_n = 32-64 n-tiles) x (4096/thread_k = 32-64 k-tiles) spread over a resident grid of 132-528 CTAs (workspace max_blocks_per_sm=4). Each n-tile's K is therefore split across several CTAs, roughly slice_count 4-16. This is an estimate: Marlin's thread_n/thread_k pick and the moe grid size were not read from source, because sglang is not readable on this node. w13 has twice w2's n-tiles, so it slices K about half as deeply as w2 does, and its reduce chain is shorter. At bs=8 (~8 m-blocks x 32-64 n-tiles, about the CTA count), slice_count is about 1-2. At bs=16/32 it is about 1, and Marlin skips the global reduce entirely. So the path exists only at bs=1, plus the bs=4 test, and is marginal at bs=8. EVIDENCE: (1) The old w13-only atomic run (c8ea8e21, 0.987x) was flat at bs=1 (80.23 -> 79.64 us) and its Marlin row was flat. It used the same block_size_m=8, the same workspace and the same w13 call shape at bs=1 as master does now, so its fast-mode reading transfers directly. The w13 reduce chain is worth about 0 us in the fast mode. (2) The landed w2 atomic moved p10 as well as p50 (p10 50.0 -> 46.1, p50 51.4 -> 47.3), so most of its 3.9 us was a fast-mode gain in w2's deeper reduce chain. Only part of it came from the slow mode dropping from 44/100 to 10/100. The hypothesis credits slow-mode removal for w2's gain, which is only partly right. WEAKEST LINK: the expected 44.9 -> 42-43 us at bs=1. A shape's time is the median of paired ratios (TASK.md 'How a run is timed'). A slow mode in 10/100 replays sits entirely above the median, so eliminating it moves the median by only a few tenths of a us. That is below decode_1's ~0.65 us err. The fast-mode component was measured at about 0 in (1). The attribution of the remaining 10/100 to the w13 lock is also unproven. The same +23 us slow mode appeared in the fused bs=1 kernel (41-53/100), which replaced both Marlin calls and had no Marlin lock at all, so at least part of the slow mode is not Marlin's lock chain. Realistic expectation: bs=1 flat to -1 us; bs=8 flat, not 'faster 1-3 us', since w2 atomic moved bs=8 by only -1.6% at z -0.6; bs=16/32 flat (no-op at slice_count 1); bs=128/256 are clean controls. Even a full success at bs=1 is worth well under 1% geomean, inside the headline CI. The hypothesis is still worth one cheap check, because it is the only remaining test of whether the residual bs=1 slow mode lives in w13's lock chain, and the raw replay distribution can answer that.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71): this is the condition being tested, not a confounder to revert. It removed w2's lock chain, so any change in the bs=1 slow-mode count can now be attributed to w13. Reverting it would just repeat the already-measured w13-only run (0.987x, flat). Numerically it stacks: master's w2 atomic path already sits at rel-L2 4.79e-3 vs the non-atomic path, so a self-check gated at 5e-3 against a FULLY non-atomic reference would very likely reject the combination for rounding alone.
- sglang-marlin-path-parity: it fixes the workspace (max_blocks_per_sm=4) and with it the resident CTA count, which sets slice_count and the size of any atomic effect. That is unchanged here, so it is a fixed condition and does not confound the run, but the result does not transfer to another workspace setting (cf. marlin-workspace-blocks-per-sm-8, neutral).
- triton-map-align-on-block-m-restack / marlin-block-m-two-smaller-on-postnorm: block_size_m at T<=32 (8 at bs<=16, 16 at bs=32) sets the m-block count and therefore the slicing depth. These are unchanged at T<=32, so the comparison is clean. A later small-M block_size_m change (e.g. queued marlin-two-class-block-m-decode32) would change this flag's value at bs=32.
- post-expert-norm-triton-small-m: it reads y (cache3, which aliases cache13) at bs<32. A cache13 zero_() before w13 ('atomic_zero') does not reach y, because w2 writes cache3 afterwards, but it does change what w2's no-memset atomic path accumulates onto. If 'atomic_zero' is chosen, w2's self-check premise (C is non-zero when w2 runs) no longer holds, though that only makes w2 safer, not wrong.

### Interference and A/B needs
No companion. The change is one kwarg on one call at T<=32, plus at most one memset that shows up as its own profile row. The useful readouts are the bs=1 per-shape median, and especially the raw bs=1 timed distribution: slow-mode count out of 100 (master: 10/100), p10 and p50 (master: about 46.1 / 47.3). The prior REWRITE_READ harvest already extracts all of these. The readouts are unambiguous. If the slow-mode count drops to ~0, the w13 lock chain was the residual source, whatever the median says. If it stays ~10/100 and p10 does not move, the w13 knob is retired for good at small M. A median loss of more than 1 us with an unchanged slow-mode count is either the memset ('atomic_zero', visible as a memset row) or slower bf16 atomics; the PATH CHOSEN line says which. ab-revert of w2-marlin-atomic-add-small-m would just reproduce w13-marlin-atomic-add-small-m (already measured, flat). An ab-tune of the workspace would only be informative after a positive result. A control isolating the memset is unnecessary, because the memset has its own row and the w2 run showed the no-memset path passes.

### Open questions
- Self-check gate: compare against the CURRENT master path (state['w2_mode'] as chosen, w13 non-atomic), not against a fully non-atomic path, and keep rel-L2 <= 5e-3 for the w13 delta alone. Also log the rel-L2 of the combined path vs a fully non-atomic reference (and, if cheap, an fp32-dequant reference), so a near-threshold rejection is visible. If PATH CHOSEN = master, the run measured nothing and must be harvested as void, not as 'neutral'.
- Make the no-memset 'atomic' candidate a real test. cache1 comes from a fresh torch.empty in _forward, which can read as zero in prepare and hide a missing zero. Before the 'atomic' candidate, poison the buffer: e.g. add a self-check-only flag to _forward that fills cache13 with a large finite value or NaN before the w13 call. The w2 run already showed Marlin's moe atomic path needs no pre-zero (C was non-zero and 'atomic' passed), so 'atomic' should win. 'atomic_zero' is a fallback that costs about 1-2 us at bs=1 and would cancel any plausible gain.
- Keep use_fp32_reduce=True, as in the spec and as in landed w2. Note the spec wording 'Marlin requires the pair' is not true; it is not required.
- Print once from prepare the n-tiles and k-tiles and the resulting slice_count (or at least n_padded/block_size_m and the grid size) for w13 and w2 at T=1 and T=8. If w13 slice_count at bs=1 is <=2, the mechanism cannot work and a flat result is expected, not informative.
- Is the residual 10/100 slow mode tied to Marlin at all? The fused bs=1 kernel showed the same +23 us mode with no Marlin lock, so it may be judge-side (clock, cold-L2 flush interaction). If this run leaves it at ~10/100, stop attributing it to lock chains.
- Correct the expected rows before harvest: bs=8 should be 'flat', not 'faster', and the kill criterion should read the raw slow-mode count and p10, not only the median, because a median metric cannot see a 10% tail.


## Implementation notes

Edited only `muse_moe_w4a16/submission.py` in the worktree (+260/-3). Nothing could be run (no GPU).

- `_forward(data, state, w2_mode, w13_mode="master", poison=False)`: for `num_tokens <= W13_ATOMIC_MAX_TOKENS (32)` and `w13_mode != "master"` the w13 Marlin call gets `use_atomic_add=True` (`use_fp32_reduce=True` kept); `atomic_zero` does `cache1.zero_()` first. `poison=True` (self-check only) fills `cache13` with 3e4 before the w13 call. `kernel_fn` picks `w13_mode` from `state["w13_modes"][_w13_band(T)]` (shape-only host dict lookup; unlisted T -> master).
- `_w13_atomic_self_check` (prepare, never raises): per T in (1,4,8,16,32), reference = CURRENT master (`state["w2_mode"]` as chosen, w13 non-atomic), candidates `atomic` (poisoned buffer) then `atomic_zero`, gate rel-L2 <= 5e-3 and finite. Selector atomic -> atomic_zero -> master, PATH CHOSEN printed per T band with memset flag. Also prints rel-L2 of current master vs a fully non-atomic path and of each candidate vs both. Prints "harvest as VOID" if every band is master. Each launch is announced with a flushed print before and after (synchronize).
- Slice-count print at T=1 and T=8 for w13 and w2: m_blocks from the real align, n/k tiles and split-K slices are ESTIMATES (128x128 tile, 4 CTAs/SM), labelled as such, since Marlin's real tile pick is not readable.
- `_marlin_exec_config_probe` (print-only, prepare only, try/except, 13 s budget): prints `inspect.signature`, module and `__doc__` of `moe_wna16_marlin_gemm`; for w13 and w2 at T=8/16/32, two skewed routings (10 and 13 local experts), master's block_size_m per T, w2 in master's own w2 mode, builds a CUDA graph per config (binding default, then (thread_k, thread_n) in {(-1,-1),(128,128),(64,256),(128,64)} x blocks_per_sm in {-1,1,2}), >=320 MB flush before each of 15 interleaved replays, prints the sorted median-us table with ratio to binding default and rel-L2 vs default, and each rejected config's first error line once. Affects no state.
- Deviations: fp32-dequant reference not added (not cheap). Review corrections applied (override the spec): gate vs current master not fully non-atomic; junk-filled buffer for `atomic`; per-T-band selection; slice print; probe. The expected rows should be corrected at harvest (bs=8 flat); read raw bs=1 slow-mode count and p10, and a PATH CHOSEN = master means VOID.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only a shape-keyed dict lookup and (in atomic_zero) a `zero_()` memset; all syncs/`.item()` are in prepare helpers (lint: warnings only, "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls: `state["w13_modes"]` is built from synthetic prepare-time inputs.
- [x] Weights repacked losslessly in prepare only (unchanged); NVFP4 stays NVFP4.
- [x] Activations BF16 as before; the probe's flush buffer is a uint8 scratch, not an activation (lint dtype warning noted).
- [x] n/a: attention checklist item, this is MoE.
- [x] No timing patches, harness frames/IO/secrets, module edits, reference import, subprocess, processes or network (only `inspect`, `time` added).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Only `submission.py` changed, first-line marker kept; no triton.autotune.

okc_lint: `OK with warnings: submission.py` (host-sync warnings in prepare helpers only, one dtype warning for the uint8 flush buffer). `py_compile` ok, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent. The audit was done by reading the code; this node has no GPU. **Verdict: dispatch-with-fixes.** Rules OK.

Fixes, in the worktree's `submission.py`:
- **The w13 self-check gate was at the noise floor (category 1).** The w13 gate used `rel <= 5e-3` against current master. The landed w2 atomic alone reads 4.79e-3 against non-atomic, and the worst T is probably T=1. The w13 delta moves w2's input by about one bf16 ulp. That decorrelates w2's own atomic rounding between the reference and the candidate, so a correct candidate can read about 6-8e-3 at T=1. That would put the target band on master and void the run. The fix adds `W13_ATOMIC_TOL = 1e-2`. A missing zero, i.e. the 3e4 poison left in place, or a wrong write still reads far above 1e-1. The candidate's error against FP32 also stays well inside the judge's 1.5%: the earlier w13-only atomic passed 5/5.
- **The probe now stops on the first device-side fault or open capture.** If a config's exception names a device-side fault (illegal or misaligned address, unspecified launch failure, device-side assert), or `torch.cuda.is_current_stream_capturing()` is still true, the probe prints ABORT and returns. It no longer goes on launching more configs.

Checked and fine:
- **Probe launch order.** Each config runs eagerly and synchronizes before it is captured. A capture runs only when the eager call succeeded, inside the same try. `torch.cuda.graph.__exit__` always ends the capture, and a failed graph is never stored.
- **Unknown kwargs.** `thread_k`, `thread_n` and `blocks_per_sm` are rejected at the binding or with a TORCH_CHECK before any launch.
- **Budget.** The 13 s budget is checked before each of the 12 (kernel, T, routing) units.
- **Flush buffer.** The 320 MB buffer is allocated once and reused. Each flush is a memset of about 0.1 ms, and there are about 2.3k of them.
- **Memory.** Up to 13 graphs are live per unit, then freed. Headroom on 80 GB is ample.
- **Poison.** It fills the `cache13`/`cache1` that `_forward` allocates, which is the same buffer the timed path writes w13 into.
- **`kernel_fn`.** It adds only a host-side dict lookup on the shape. There are no prints, no try/except and no state.

okc_lint after the fixes: `OK with warnings` (the same prepare-only warnings). `py_compile` OK, and `__pycache__` was removed.

Residual risk: a sticky CUDA fault cannot be ruled out without the GPU. That would need an explicit `thread_k`/`thread_n` config that Marlin accepts but mis-executes, in a prepare-time probe that prints only. The T=1 gate also still depends on my rounding estimate of 6-8e-3, which comes from reasoning about rounding, not from a measurement.
