# muse_prefill_layer · 2026-10-25T011203Z · check · experts-triton-turn4-persistent-local-align-control-no-persi

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T010741Z-experts-triton-turn4-persistent-local-align-control-no-persi.md` (branch `hyp/muse_prefill_layer/experts-triton-turn4-persistent-local-align-control-no-persi`).

- **Job:** `374efc72c4d14856` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `8c5e690`

## Goal
Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 1024, with a comment); T <= 1024 keeps _experts_marlin. (3) In _experts_triton, when state['align_par_ok'] is set, replace moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']), which already exists on master for the Marlin path. Extend _align_par_setup's self-check to cover block 128 and 256 at T=3012 and 16384 (per-expert route sets vs moe_align_block_size). Because no -1 tiles remain, pass zero_nonlocal=False to the w2 call on this path. Add a prepare() NaN-poison check: fill cache3 with NaN, run the finalize, and assert a finite output. Run _check_experts_e2e at T=3012 as well as 8192 (rel L2 <= 2e-2, worst row <= 5x). Any failure sets MARLIN_MAX_TOKENS back to 4096 for the run and logs '[turn4-triton] OFF: <reason>'. Otherwise log 'ON'. _moe_nvfp4_gemm_kernel's grid stays as on master (no persistent loop). Everything else is unchanged.

## What changed, and where
All in muse_prefill_layer/submission.py (+97/-7). Review fixes applied:
- MARLIN_MAX_TOKENS = 1024 (commented); `_experts` reads `state.get('marlin_max_tokens', MARLIN_MAX_TOKENS)`; fallback sets `state['marlin_max_tokens'] = 4096`; global never mutated (b).
- New `_align_par_triton_setup`: own flag `state['align_par_triton_ok']`, `align_par_ok` untouched (a). Self-check of `_align_parallel` vs `moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS)` at bsm 128/256, T=3012 and 16384 (random/skewed/all_remote). Warms both BSMs before capture (c). Prints n_padded at T=3012 for both aligns (f).
- Instead of NaN poison, the flag is forced False when `state['zero_nonlocal_w2']` is True (d); `_experts_triton` then passes zero_nonlocal=False only when the local align is used.
- `_check_experts_e2e(state, device, tokens)`; new `_turn4_triton_setup` runs 8192 (as master; if it fails with local align, drops the flag and rechecks under master align) then 3012 (e) and logs `[turn4-triton] ON` / `OFF: <reason>` (g).
- Deviation: the local align also applies at T>4096 (first_turn) when the flag is on, per spec part (3); review expects flat there.
- Kernel grid unchanged. Nothing run (no GPU).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only prepare()-time syncs/.cpu(); kernel_fn path adds `_align_parallel` launch (no sync) and a shape/state-based branch.
- [x] Nothing derived from `data` is cached: state flags come from synthetic prepare() data only.
- [x] Weights: unchanged; NVFP4 stays NVFP4.
- [x] Activations: n/a: no dtype changes.
- [x] Attention: n/a: untouched.
- [x] No timing patches, harness access, subprocesses or network (checked diff).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all in prepare(), outside kernel_fn). py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Separates the turn_4 Triton-vs-Marlin question and the -1-tile removal from the persistent-grid change still being measured.

## Result (after the run)
- job `374efc72c4d14856` (check), status done, passed True, tests 5/5
- speedup 1.0557x, 95% CI [1.0398, 1.0715], score 366974.9300711438 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-turn4-persistent-local-align-control-no-persi.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-turn4-persistent-local-align-control-no-persi.patch`; branch `hyp/muse_prefill_layer/experts-triton-turn4-persistent-local-align-control-no-persi`
