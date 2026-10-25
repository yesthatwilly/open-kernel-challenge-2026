# muse_moe_w4a16 · 2026-10-25T043232Z · check · w13-marlin-atomic-add-on-master-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T042150Z-w13-marlin-atomic-add-on-master-small-m.md` (branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m`).

- **Job:** `171684977ec1470f` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `c3a8442`

## Goal
On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 only, under its own bound constant W13_ATOMIC_MAX_TOKENS = 32; keep use_fp32_reduce=True (as the landed w2 change does - Marlin accepted it and 5/5 tests passed). Extend master's _w2_atomic_self_check pattern into a _w13_atomic_self_check: candidates 'atomic' (no memset) and 'atomic_zero' (cache13/cache1 zeroed before the call), each checked against master on the same synthetic T in (1, 4, 8, 16, 32) routings with rel-L2 <= 5e-3 and finiteness; print PATH CHOSEN and whether a memset is in the path. Nothing else changes. Mechanism: the landed w2 change removed most of bs=1's +23 us judge slow mode (44/100 -> 10/100 replays) by replacing the fp32-reduce lock serialization; w13 still uses the lock path, the likely source of the remaining 10/100. The earlier w13-only atomic test (0.987x) ran on the pre-post-norm base before the slow mode was measured, so this is a new measurement.

## What changed, and where
Edited only `muse_moe_w4a16/submission.py` in the worktree (+260/-3). Nothing could be run (no GPU).

- `_forward(data, state, w2_mode, w13_mode="master", poison=False)`: for `num_tokens <= W13_ATOMIC_MAX_TOKENS (32)` and `w13_mode != "master"` the w13 Marlin call gets `use_atomic_add=True` (`use_fp32_reduce=True` kept); `atomic_zero` does `cache1.zero_()` first. `poison=True` (self-check only) fills `cache13` with 3e4 before the w13 call. `kernel_fn` picks `w13_mode` from `state["w13_modes"][_w13_band(T)]` (shape-only host dict lookup; unlisted T -> master).
- `_w13_atomic_self_check` (prepare, never raises): per T in (1,4,8,16,32), reference = CURRENT master (`state["w2_mode"]` as chosen, w13 non-atomic), candidates `atomic` (poisoned buffer) then `atomic_zero`, gate rel-L2 <= 5e-3 and finite. Selector atomic -> atomic_zero -> master, PATH CHOSEN printed per T band with memset flag. Also prints rel-L2 of current master vs a fully non-atomic path and of each candidate vs both. Prints "harvest as VOID" if every band is master. Each launch is announced with a flushed print before and after (synchronize).
- Slice-count print at T=1 and T=8 for w13 and w2: m_blocks from the real align, n/k tiles and split-K slices are ESTIMATES (128x128 tile, 4 CTAs/SM), labelled as such, since Marlin's real tile pick is not readable.
- `_marlin_exec_config_probe` (print-only, prepare only, try/except, 13 s budget): prints `inspect.signature`, module and `__doc__` of `moe_wna16_marlin_gemm`; for w13 and w2 at T=8/16/32, two skewed routings (10 and 13 local experts), master's block_size_m per T, w2 in master's own w2 mode, builds a CUDA graph per config (binding default, then (thread_k, thread_n) in {(-1,-1),(128,128),(64,256),(128,64)} x blocks_per_sm in {-1,1,2}), >=320 MB flush before each of 15 interleaved replays, prints the sorted median-us table with ratio to binding default and rel-L2 vs default, and each rejected config's first error line once. Affects no state.
- Deviations: fp32-dequant reference not added (not cheap). Review corrections applied (override the spec): gate vs current master not fully non-atomic; junk-filled buffer for `atomic`; per-T-band selection; slice print; probe. The expected rows should be corrected at harvest (bs=8 flat); read raw bs=1 slow-mode count and p10, and a PATH CHOSEN = master means VOID.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only a shape-keyed dict lookup and (in atomic_zero) a `zero_()` memset; all syncs/`.item()` are in prepare helpers (lint: warnings only, "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls: `state["w13_modes"]` is built from synthetic prepare-time inputs.
- [x] Weights repacked losslessly in prepare only (unchanged); NVFP4 stays NVFP4.
- [x] Activations BF16 as before; the probe's flush buffer is a uint8 scratch, not an activation (lint dtype warning noted).
- [x] n/a: attention checklist item, this is MoE.
- [x] No timing patches, harness frames/IO/secrets, module edits, reference import, subprocess, processes or network (only `inspect`, `time` added).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Only `submission.py` changed, first-line marker kept; no triton.autotune.

okc_lint: `OK with warnings: submission.py` (host-sync warnings in prepare helpers only, one dtype warning for the uint8 flush buffer). `py_compile` ok, __pycache__ removed.

## Risks / what could fail
low
Kill: bs=1 not below 44.9 us by > 1.5% and its slow-mode count not below 10/100; then the w13 lock path is not the source. Risk: if only 'atomic_zero' passes, the memset costs ~1-2 us and may cancel the gain.

## Result (after the run)
- job `171684977ec1470f` (check), status done, passed True, tests 5/5
- speedup 1.3558x, 95% CI [1.3355, 1.3761], score 9919.2981689056 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-on-master-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-on-master-small-m.patch`; branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m`
