# muse_moe_w4a16 · 2026-10-25T043034Z · check · w2-marlin-atomic-add-verify

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T042151Z-w2-marlin-atomic-add-verify.md` (branch `hyp/muse_moe_w4a16/w2-marlin-atomic-add-verify`).

- **Job:** `ccbcda65da584be0` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `dfcd2a5`

## Goal
On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by giving the verify range its own bound constant W2_ATOMIC_VERIFY = True applying at num_tokens > 32 (bs=128, 256 and the bs=33/257/skewed-128 tests); T <= 32 keeps master's landed W2_ATOMIC_MAX_TOKENS path unchanged. Re-run master's _w2_atomic_self_check at T in (33, 128, 256, 257) with skewed and random routings, its own 'atomic'/'atomic_zero'/'master' decision and PATH CHOSEN line; use_fp32_reduce stays True. Note on bs=8/16: master already applies w2 atomic at T <= 32, so there is no smaller-M extension left to test. Mechanism: the verify Marlin rows swing 391-451 us at bs=256 across identical-code runs today, the same symptom (a lock-serialized reduction slow mode) that w2 atomic removed at bs=1. At bs=128/256, block_size_m is 8/16 (two-smaller pick) so w2 has many slices per output tile and the fp32 reduce takes the lock path.

## What changed, and where
Edited only `muse_moe_w4a16/submission.py` in the worktree (+160/-11). Review overrides applied (the review wins over the spec):
- block_size_m at bs=128/256/257 is 16/32/32 (master's per-T value, unchanged); the spec's 8/16 was wrong. No block_size_m code touched.
- Verify mode is decided apart from the landed one: `state["w2_mode"]` (T<=32, `_w2_atomic_self_check`, untouched) vs new `state["w2_mode_verify"]` = {class: mode} from `_w2_verify_self_check`, wrapped in try/except so an error leaves verify on master and never touches bs<=32. `kernel_fn` picks by `hidden_states.shape[0]` (host shape only): T<=32 -> w2_mode, else verify class (33..127, 128..255, >=256).
- `_forward` takes the mode as an argument; the w2 launch is now a local closure `w2_call(mode)`; the old `num_tokens <= W2_ATOMIC_MAX_TOKENS` test moved into kernel_fn's pick. Same launch/args for master and T<=32.
- Verify self-check: T in (33,128,256,257), random and skewed routing, compares the final combined output vs master. atomic_zero is measured but never chosen (16.8 MB / 8.4 MB memset cancels the gain); if only atomic_zero passes a class stays master.
- Tolerance: rel-L2 vs master <= 5e-3 on the whole output and <= 2.5e-2 on any row (row denominator max(row norm, 0.1*mean row norm)). Justified from TASK: output vs FP32 ref <= 1.5%, any row <= 5x; master's own error spends first, so atomic may add at most a third of each bound (the T<=32 path sat at 4.79e-3 against the same 5e-3).
- In prepare(), w2 alone timed at T=128 and T=256 (skewed) for master/atomic/atomic_zero: CUDA events around the call (memset included for atomic_zero), 256 MB `fill_` before each rep (cold L2), 50 interleaved reps, median. A class takes atomic only if it passes the gate and atomic <= master in this timing (class 33 is not timed; gate only). Extra gate beyond the review, to avoid a loss: a slower atomic stays master.
- Prints: per-class `PATH CHOSEN`, per-T verify mode with worst rel-L2/row for atomic and atomic_zero, the w2-alone timings, and `NO-OP` if every class chose master (harvest as no-op, not flat).
- No CUDA/PTX added, no processes, no mbarriers; the only new kernel work is the existing Marlin launch.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no sync/.item() in kernel_fn's call graph; the mode pick is a host dict lookup on the shape; timing/sync only in prepare().
- [x] Nothing derived from `data` is cached: modes come from synthetic inputs in prepare().
- [x] Weights: unchanged, repacked in prepare() as on master.
- [x] Activations BF16 as master; use_fp32_reduce stays True; only use_atomic_add toggles on w2 (same as landed).
- [x] Attention: n/a, MoE problem.
- [x] No timing patches, harness frames, subprocesses or network (verified against the diff).
- [x] All work inside the timed call; no output caching or replay.
- [x] Any self-check miss or error lands on master's path and never raises.

okc_lint: `OK with warnings` (0 errors; host-sync warnings only in prepare()-side code, one dtype warning for the uint8 L2-flush buffer in prepare()). py_compile OK; __pycache__ deleted.

## Risks / what could fail
medium
Kill: bs=128 and bs=256 within noise or slower (atomic contention on the 32-expert outputs at 1024-2048 routes can lose). Risk: atomic adds into bf16 at large M may lose precision vs fp32 reduce; the self-check bound 5e-3 vs master catches it.

## Result (after the run)
- job `ccbcda65da584be0` (check), status done, passed True, tests 5/5
- speedup 1.3407x, 95% CI [1.3206, 1.3608], score 10030.80507908941 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-verify.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-verify.patch`; branch `hyp/muse_moe_w4a16/w2-marlin-atomic-add-verify`
