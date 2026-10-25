# muse_prefill_layer · 2026-10-24T233023Z · check · align-local-parallel-first-turn

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T232955Z-align-local-parallel-first-turn.md` (branch `hyp/muse_prefill_layer/align-local-parallel-first-turn`).

- **Job:** `c2f3e9d8297f4a18` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `0b61b2b`

## Goal
Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']) (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384 at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and their w2 zero-stores.

## What changed, and where
Built in the main thread 2026-10-24 ~23:30Z with user-approved self-review, on moe-align-local-routes-parallel-port's implementation commit (claim based it there: parent still running). `_experts_triton`: `_align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg["BLOCK_T"])` when `state["align_par_ok"]`, else master's call. `_align_par_setup` now also tests T=16384 at the Triton path's BLOCK_T (256), which also compiles that variant before capture; a mismatch there turns the whole parallel align OFF (both paths), never just one.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: shape-only allocations, prepare()-set gate, no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Main-thread build before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `c2f3e9d8297f4a18` (check), status done, passed True, tests 5/5
- speedup 1.0626x, 95% CI [1.0466, 1.0785], score 364586.61664186686 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/neutral/moe-align-local-routes-parallel-port.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/align-local-parallel-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/align-local-parallel-first-turn.patch`; branch `hyp/muse_prefill_layer/align-local-parallel-first-turn`
