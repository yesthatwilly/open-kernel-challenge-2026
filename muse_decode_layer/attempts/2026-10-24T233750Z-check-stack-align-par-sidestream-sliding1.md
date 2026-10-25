# muse_decode_layer · 2026-10-24T233750Z · check · stack-align-par-sidestream-sliding1

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T233721Z-stack-align-par-sidestream-sliding1.md` (branch `hyp/muse_decode_layer/stack-align-par-sidestream-sliding1`).

- **Job:** `cc92097b64784c0f` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `511013c`

## Goal
Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side stream and passes the result to _experts via dispatch=, which would bypass the align port's branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm) when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The sliding batch-1 restack is disjoint.

## What changed, and where
Hand merge in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL); `git apply -3` of moe-align-parallel-large-t-port's and moe-dispatch-prep-on-side-stream's done/ patches, then the sliding batch-1 restack as `git diff c9bcd4b..b85563b`.
- One conflict in `_experts`: resolved to `if dispatch is not None` (side-stream result) / `elif` the align port's T >= 33 branch / `elif align is None` master's path.
- `_expert_dispatch` (side stream, T > 32): when `state["align_par_ok"]`, `_align_parallel(shifted, 0, LOCAL_EXPERTS, _block_size_m(T))` replaces the expert_map gather + moe_align_block_size (the same call and inputs prepare() self-checked); otherwise master's two calls. Without this the align port would be bypassed whenever dispatch= is passed.
- kernel_fn's existing record_stream loop already covers local_ids and the align tensors produced on the side stream.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: side-stream fork/join and allocations as in the parents; prepare()-set gate; no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parents.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Main-thread hand merge before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `cc92097b64784c0f` (check), status done, passed True, tests 13/13
- speedup 1.2495x, 95% CI [1.2308, 1.2683], score 18103.13551858608 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.patch`; branch `hyp/muse_decode_layer/stack-align-par-sidestream-sliding1`
