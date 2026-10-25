# muse_prefill_layer · 2026-10-25T043641Z · check · turn-align-remap-on-side-stream-under-pooler

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T042410Z-turn-align-remap-on-side-stream-under-pooler.md` (branch `hyp/muse_prefill_layer/turn-align-remap-on-side-stream-under-pooler`).

- **Job:** `40f21883155b47eb` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `7b424f9`

## Goal
Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) and 70.5 us at turn_4 (2.1%), while the side stream (router GEMM ~15-25 us + routing ~33 us at turn_4) finishes well inside the pooler GEMM on the main stream. In kernel_fn, inside the existing `with torch.cuda.stream(side_stream)` block, after state['routing'], also run local_ids = _local_expert_ids(topk_ids, data['local_start']) and, when tokens <= MARLIN_MAX_TOKENS and state['align_par_ok'], _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) with block_size_m from the same shape-only BLOCK_SIZE_M_CANDIDATES loop _experts_marlin uses (factor it into a helper both call). Pass the resulting (sorted_ids, block_experts, n_padded) into _experts_marlin through a new optional argument, which then skips its own align call. The existing main_stream.wait_stream(side_stream) stays where it is and the record_stream loop covers the new outputs. T > 4096 and the align_par_ok=False fallback are unchanged. Relation: moe-align-side-stream (done/neutral 1.0076 on the old parity base) moved SGLang's moe_align_block_size path; this re-tests the placement with master's current align kernel, which is the one now in the profile.

## What changed, and where
Edited only `muse_prefill_layer/submission.py` in the worktree (+33/-12). Changes:
- New helper `_marlin_block_size_m(num_tokens)` (shape-only BLOCK_SIZE_M_CANDIDATES loop); `_experts_marlin` uses it.
- `_experts` / `_experts_marlin` take an optional `align=(sorted_ids, block_experts, n_padded)`; when given, `_experts_marlin` skips its own align. With `align=None` (T > 4096, align_par_ok False) behaviour is unchanged. first_turn path unchanged.
- In `kernel_fn`, inside the existing `with torch.cuda.stream(side_stream)` block after `state["routing"]`: `local_ids = _local_expert_ids(topk_ids, data["local_start"])` (device tensor, no host read), then, if `ffn_input.shape[0] <= MARLIN_MAX_TOKENS` and `state["align_par_ok"]`, `_, align = _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm)`.
- Review overrides applied: the `data['local_start']` tensor is NOT passed to `_align_parallel`'s scalar argument (it gets the Python 0 on already-remapped ids, same as master's call); `record_stream(main_stream)` now covers topk_weights, topk_ids, local_ids and the three align outputs (skipped while capturing, as before); the existing `main_stream.wait_stream(side_stream)` join (an event edge under capture) sits after all side-stream work and before any consumer. The fork `side_stream.wait_stream(main_stream)` is unchanged. Graph-capturable.
- Deviation: the review's "pass bsm as helper parameter" is met by `_align_parallel`'s existing bsm argument; the block-size pick helper takes num_tokens.
- Expected effect: about +0.5-0.8% headline (turn_4 -40 to -70 us, turn_1 about 0). Read it from the profile's GPU-busy vs kernel-sum / overlap lines (the _align_parallel row should start before the pooler ends), not from shape z.
- Risk: the side branch is now longer; the 33-CTA align may not co-reside with the pooler kernel; at turn_1 the wait may just move.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/synchronize; the side stream forks and joins from the capturing stream via wait_stream; record_stream guarded by is_current_stream_capturing.
- [x] Nothing derived from `data` is cached across calls (new tensors are per-call locals).
- [x] Weights: n/a, untouched.
- [x] Activations/KV/queries precision: n/a, untouched.
- [x] Attention: n/a, untouched.
- [x] No timing patches, harness access, module edits, process spawning: diff is pure kernel_fn / helper plumbing.
- [x] All work inside the timed call; no output caching or replay.
- [x] No new CUDA/load_inline: build-brief arch/mbarrier/self-check items n/a: reason, no CUDA added.

okc_lint: "OK with warnings: submission.py" (only pre-existing host-sync warnings in prepare()-side code, outside kernel_fn). py_compile OK; __pycache__ removed.

## Risks / what could fail
low
Kill: the profile shows _align_parallel overlapping the pooler (GPU busy below the kernel sum) and turn_4 does not move beyond 1% -> placement is not the cost. Risk: a longer side branch than the pooler at turn_1 just moves the wait; the judge sees no harm beyond noise.

## Result (after the run)
- job `40f21883155b47eb` (check), status done, passed True, tests 5/5
- speedup 1.0683x, 95% CI [1.0523, 1.0844], score 362625.49272810924 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/turn-align-remap-on-side-stream-under-pooler.result.json`, `muse_prefill_layer/hypotheses/done/neutral/turn-align-remap-on-side-stream-under-pooler.patch`; branch `hyp/muse_prefill_layer/turn-align-remap-on-side-stream-under-pooler`
