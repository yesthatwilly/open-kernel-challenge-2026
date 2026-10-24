# muse_prefill_layer · 2026-10-24T044914Z · check · moe-align-side-stream

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-23T230250Z-moe-align-side-stream.md` (branch `hyp/muse_prefill_layer/moe-align-side-stream`).

- **Job:** `f70d1f2eaef842d2` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `0199b99`

## Goal
Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.cuda.stream(side_stream) block right after topk_weights, topk_ids = state['routing'](...): the fused local-id remap kernel, topk_weights.to(torch.float32).contiguous(), and moe_align_block_size(local_ids, block_size_m, N_EXPERTS) (block_size_m is chosen from config tokens on the host, so the choice does not move). main_stream.wait_stream(side_stream) moves from before the local-id remap to just before the first moe_wna16_marlin_gemm call, after _gemm(ffn_input, pooler) and state['pre_expert_norm'](expert_input) have been issued on the main stream, and the non-capturing record_stream loop is extended to local_ids, topk_weights, sorted_ids, block_experts and n_padded. _experts takes the precomputed (local_ids, topk_weights, sorted_ids, block_experts, n_padded) instead of computing them; the cache13/cache2 allocation stays on the main stream. Nothing else changes. Under graph capture the side stream becomes a parallel branch, so the remap + align kernels (4-6 launches over T*8 ids) overlap the pooler GEMM and pre-expert norm instead of sitting serially between them and the Marlin GEMM.

## What changed, and where
Worktree `.worktrees/muse_prefill_layer/moe-align-side-stream`, file `muse_prefill_layer/submission.py` only (diff stat: 1 file, +24 / -12).

- `kernel_fn`: inside the existing `torch.cuda.stream(side_stream)` block, right after `state['routing'](...)`, now issue `_local_expert_ids(topk_ids, local_start)`, `topk_weights.to(torch.float32).contiguous()` and `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)`. On the main stream the order is pooler GEMM (issued before the side block, as before), then `state['pre_expert_norm'](expert_input)`, then `main_stream.wait_stream(side_stream)`, then `_experts`. The join was moved from before the remap to after the pre-expert norm, i.e. directly before the first `moe_wna16_marlin_gemm` (only host-side `torch.empty` allocations of cache2/cache13 and Python view slicing sit between the join and that GEMM; no GPU work).
- The non-capturing `record_stream(main_stream)` loop now covers every side-stream tensor consumed on the main stream: `topk_weights` (the fp32 one), `topk_ids` (returned), `local_ids`, `sorted_ids`, `block_experts`, `n_padded`.
- `block_size_m` selection is factored into `_block_size_m(num_tokens)` and called on the host in `kernel_fn` from `cfg['tokens']` (same loop, same candidates; `tokens == x.shape[0]` as before). `_experts` now takes `(x, topk_weights, local_ids, sorted_ids, block_experts, n_padded, block_size_m, state)` and no longer computes the cast, the block size or the align. cache13/cache2 are still allocated in `_experts` on the main stream.
- No new kernels, no precision change, no host syncs, nothing cached in state.
- Deviation: none in substance. The only nuance is that cache allocations happen after the join rather than strictly before it; torch.empty issues no GPU work, so the captured graph is the same as the spec's ordering.
- FusedMuseSparkRouting dtype: the SGLang source is not on this node, so whether its topk_weights are already contiguous fp32 could not be verified. The cast is kept on the side stream; if they are already fp32 contiguous, `.to()`/`.contiguous()` return the same tensor and the cast is a no-op (no launch), and only the remap + align move.
- Eager safety beyond the loop: `ffn_input` (main-stream allocation read on the side stream) stays referenced until after the join, as in the base; `logits`, the pre-cast topk_weights and moe_align's temporaries are allocated on the side stream and freed there, so no extra record_stream is needed.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/moe-align-side-stream diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/`synchronize()`/tensor prints; block_size_m is chosen from `cfg['tokens']` (config), no allocation or control flow on input values. The `is_current_stream_capturing()` guard on record_stream is existing code.
- [x] Nothing derived from `data` is cached across calls: no new state entries; all moved tensors are locals.
- [x] Weights: untouched; `prepare()` is unchanged.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: no dtype changes (the fp32 topk_weights cast is the same cast, moved).
- [x] Attention sees every key: attention code untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added; imports unchanged.
- [x] All work happens inside the timed call; no output caching or replay: all moved work is still inside `kernel_fn`.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept; no `triton.autotune`.

Lint (`okc_lint.py ... --problem muse_prefill_layer`):

```
warn  host-sync    L261: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_prefill_layer/moe-align-side-stream/muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed. Not committed, not staged, okc not run.

## Risks / what could fail
low
Low risk: pure stream reordering of kernels that already run; no math changes. Failure modes: a missing record_stream on a side-stream tensor when run eagerly (the tests run under the judge's graph capture too, but keep the loop); moe_align_block_size allocating inside the side stream is fine under the graph pool. If the baseline's FusedMuseSparkRouting output topk_weights is already fp32 the cast is a no-op and only the remap+align move. The gain is bounded by the align kernels' time, so a flat result on first_turn attributes nothing and is expected.

## Result (after the run)
- job `f70d1f2eaef842d2` (check), status done, passed True, tests 5/5
- speedup 1.0076x, 95% CI [0.9925, 1.0227], score 384474.7931363558 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-align-side-stream.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-align-side-stream.patch`; branch `hyp/muse_prefill_layer/moe-align-side-stream`
