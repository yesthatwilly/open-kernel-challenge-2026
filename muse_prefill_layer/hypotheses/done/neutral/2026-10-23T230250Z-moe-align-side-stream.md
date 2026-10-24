---
problem: muse_prefill_layer
slug: moe-align-side-stream
created: 2026-10-23T230250Z
base: master
area: MoE align/dispatch
change: >
  Keep the parity base and move three pieces of work that depend only on the routing outputs and
  local_start onto the existing side stream, inside the same torch.cuda.stream(side_stream) block
  right after topk_weights, topk_ids = state['routing'](...): the fused local-id remap kernel,
  topk_weights.to(torch.float32).contiguous(), and moe_align_block_size(local_ids, block_size_m,
  N_EXPERTS) (block_size_m is chosen from config tokens on the host, so the choice does not move).
  main_stream.wait_stream(side_stream) moves from before the local-id remap to just before the
  first moe_wna16_marlin_gemm call, after _gemm(ffn_input, pooler) and
  state['pre_expert_norm'](expert_input) have been issued on the main stream, and the
  non-capturing record_stream loop is extended to local_ids, topk_weights, sorted_ids,
  block_experts and n_padded. _experts takes the precomputed (local_ids, topk_weights, sorted_ids,
  block_experts, n_padded) instead of computing them; the cache13/cache2 allocation stays on the
  main stream. Nothing else changes. Under graph capture the side stream becomes a parallel
  branch, so the remap + align kernels (4-6 launches over T*8 ids) overlap the pooler GEMM and
  pre-expert norm instead of sitting serially between them and the Marlin GEMM.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: the remap+align chain (~15-30 us serial on the main stream) is hidden behind the pooler GEMM (~80-100 us at M=753); 1-3% of 1120-1465 us
  - shapes: "*load=turn_4*"
    direction: faster
    why: same chain over 24K ids hidden behind a 4x larger pooler GEMM; ~0.5-1%
  - shapes: "*load=first_turn*"
    direction: flat
    why: align over 131K ids is ~50 us of 17-26 ms; inside the interval
risk: low
depends_on: sglang-pipeline-parity
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T044735Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T044735Z
base_sha: a8970404e91613553aa3868dbafdca06fbc4b549
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/moe-align-side-stream
worktree: .worktrees/muse_prefill_layer/moe-align-side-stream
mode: check
implemented_at: 2026-10-24T044913Z
impl_commit: 0199b9927fac74d4189a4107b74801542abda5a8
submission_sha256: c68c2d40fd6702c142ecf827abd26e28a3c075b645d779565f705c57d8ca3e79
norm_sha256: 4a606e5a258f85dfd83542336b87fe4fd7a94dd1188165e2189221c0de3f21ba
submission_blob: 4be1dd8c787a25f3d8b23461adea032d3588c6d8
lint_warnings: 1
attempt_file: 2026-10-24T044914Z-check-moe-align-side-stream.md
prepared_at: 2026-10-24T044914Z
job: f70d1f2eaef842d2
dispatched_at: 2026-10-24T044925Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0076184626819096
speedup_ci: [0.9925041857416809, 1.0227327396221382]
harvested_at: 2026-10-24T063613Z
best_at_harvest: 1.0038664293875537
promotion: null
---

# moe-align-side-stream

## Change

Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.cuda.stream(side_stream) block right after topk_weights, topk_ids = state['routing'](...): the fused local-id remap kernel, topk_weights.to(torch.float32).contiguous(), and moe_align_block_size(local_ids, block_size_m, N_EXPERTS) (block_size_m is chosen from config tokens on the host, so the choice does not move). main_stream.wait_stream(side_stream) moves from before the local-id remap to just before the first moe_wna16_marlin_gemm call, after _gemm(ffn_input, pooler) and state['pre_expert_norm'](expert_input) have been issued on the main stream, and the non-capturing record_stream loop is extended to local_ids, topk_weights, sorted_ids, block_experts and n_padded. _experts takes the precomputed (local_ids, topk_weights, sorted_ids, block_experts, n_padded) instead of computing them; the cache13/cache2 allocation stays on the main stream. Nothing else changes. Under graph capture the side stream becomes a parallel branch, so the remap + align kernels (4-6 launches over T*8 ids) overlap the pooler GEMM and pre-expert norm instead of sitting serially between them and the Marlin GEMM.

## Expected

- `*load=turn_1*` faster: the remap+align chain (~15-30 us serial on the main stream) is hidden behind the pooler GEMM (~80-100 us at M=753); 1-3% of 1120-1465 us
- `*load=turn_4*` faster: same chain over 24K ids hidden behind a 4x larger pooler GEMM; ~0.5-1%
- `*load=first_turn*` flat: align over 131K ids is ~50 us of 17-26 ms; inside the interval

## Risk

Low risk: pure stream reordering of kernels that already run; no math changes. Failure modes: a missing record_stream on a side-stream tensor when run eagerly (the tests run under the judge's graph capture too, but keep the loop); moe_align_block_size allocating inside the side stream is fine under the graph pool. If the baseline's FusedMuseSparkRouting output topk_weights is already fp32 the cast is a no-op and only the remap+align move. The gain is bounded by the align kernels' time, so a flat result on first_turn attributes nothing and is expected.

## Adversarial review

Reviewed 2026-10-24T044735Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A pure stream reorder of kernels that already run, rule-clean and low risk, but the parity profile says the side stream is already as long as or longer than the pooler GEMM at turn_1, so the realistic saving is ~7-20 us per turn_1 shape and ~20-40 us per turn_4 shape: 0.5-1.5%, inside the +/-1.5% interval. Run it, expect 'flat' verdicts, and read the gain as bounded rather than confirmed.

### Justification
Mechanism: under capture, a fork/join on torch streams becomes two independent graph branches, so remap + fp32 cast + moe_align_block_size on the side stream can run concurrently with the pooler GEMM and pre-expert norm on the main stream. That is sound CUDA-graph behaviour and the base already forks the router this way. The numbers are where it fails. In the parity profile (job 62025c368c3c4e8e) the align chain at turn_1 is small: count_and_sort_expert_tokens_kernel 5.1 us, moe_align_block_size_kernel below the 5.1 us cut-off (it is 52 us only at 16K tokens), the Triton remap and any cast also below it, so the chain the hypothesis calls '15-30 us' is more like 10-20 us serial. The window to hide it in is also smaller than claimed: the pooler at M=753 is a ~40-52 us nvjet, not 80-100 us, and the pre-expert norm (vectorized_layer_ 6.6 us) is 7 us. Worst, the side stream is not idle during that window: the router hi/lo pair runs as split-K GEMMs (33-40 us, 2 calls) plus cublasLt splitKreduce (45.6 us for 2 calls on turn_1 full, in the top 12), plus FusedMuseSparkRouting, i.e. roughly 40-85 us already on the side branch versus a ~45-50 us main branch. Critical path old = max(P,S)+R+N+A+M, new = max(P+N, S+R+A)+M; when S >= P the saving collapses to the norm (~7 us); when S < P it is at most P+N-S. On turn_1 that is 7-20 us of 1079-1435 us (0.5-1.5%), not the 1-3% stated. At turn_4 the router is 132 us (no split-K) against a ~152-256 us pooler, so S < P by a margin and the full chain (count_sort 17 us + align ~15 us + remap ~5 us + norm 18 us) could be hidden: ~35-50 us of 3367-4950 us, 0.7-1.1%, consistent with the hypothesis's 0.5-1% and still inside noise. first_turn flat is right (align ~140 us of 17-25 ms, and the side branch is far longer than any window there anyway). Shape weights: turn_1 sliding (w46) and turn_4 sliding (w46) carry the loads, so direction is right but magnitude lands inside the interval on every load. Weakest link: the argument assumes the side stream is idle while the pooler runs; the profile says the router branch already fills or exceeds that window at turn_1, so the saving is bounded by the pre-expert norm plus whatever slack remains, and concurrent kernels also share SMs so even that is not free. The single run will almost certainly report flat on all eight shapes and will not falsify the mechanism either way.

### Confounders on the current kernel
- sglang-pipeline-parity: it is the base, not a confounder in the A/B sense; its fused Triton remap is one of the three kernels being moved, and its router-on-side-stream fork is the branch being extended. Both are integral to the base and cannot be reverted independently (the patch is the whole file against a plain-PyTorch starter). A gain or loss here is attributable to the reorder alone.

### Interference and A/B needs
Only one hypothesis has landed and it is the whole SGLang-path base, so ab-revert has no target: reversing muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.patch returns to the plain-PyTorch starter, where neither side stream nor Marlin exists. ab-tune has no knob: the parity base copies baseline.py's stream placement and block_size_m selection one for one, and a tune (for example moving the router back to the main stream) would change the comparison point rather than disambiguate. A control isolating one sub-change is not warranted because the sub-changes are not separately resolvable below the 1.5% interval: moving only the remap, only the cast, or only the align would each be 0-1% and every result would be flat. The interpretation of the single run is already unambiguous as to cause: a gain is the reorder, a loss outside noise would be side-branch contention delaying the join or the align kernels stealing SMs from the pooler, and a flat result is the expected bounded outcome. No companions.

### Open questions
- Which nvjet is the pooler at M=753: the 39.3 us 112x128 or one of the three 128x192 calls (~52 us each)? Either way the window is 40-52 us, not 80-100 us; the next FINDINGS entry should record the per-kernel stream assignment so the side-branch length is known.
- Does the router's split-K + splitKreduce branch (33-85 us) exceed the pooler at turn_1? If so the only recoverable serial time is the pre-expert norm (~7 us) and this hypothesis should be followed by one that shortens the router branch (e.g. a non-split-K cuBLAS selection or fusing hi/lo into one GEMM on a concatenated [8192, 512] weight with an fp32 add), which is the real limiter of this overlap.
- Does moe_align_block_size in the judge's SGLang build take num_experts >= 224 through a path with a device cumsum buffer only (no host read)? The base already runs it under capture so a host sync would have failed there; confirm no new code path is hit when called from a non-default stream.
- Does the queued experts-dequant-overlap-attention also put work on state['side_stream']? Two consumers of one side stream serialize; if both land, the side branch may become the critical path and this hypothesis's small gain would be undone.
- The parity base's turn_1 full shape shows GPU busy 1420 us vs kernel total 1498 us (78 us overlap) while sliding shows only 18 us; worth checking whether cublas chose a different router algorithm per shape, which would make the side-branch length vary run to run and widen the noise this hypothesis has to beat.


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/moe-align-side-stream`, file `muse_prefill_layer/submission.py` only (diff stat: 1 file, +24 / -12).

- `kernel_fn`: inside the existing `torch.cuda.stream(side_stream)` block, right after `state['routing'](...)`, now issue `_local_expert_ids(topk_ids, local_start)`, `topk_weights.to(torch.float32).contiguous()` and `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)`. On the main stream the order is pooler GEMM (issued before the side block, as before), then `state['pre_expert_norm'](expert_input)`, then `main_stream.wait_stream(side_stream)`, then `_experts`. The join was moved from before the remap to after the pre-expert norm, i.e. directly before the first `moe_wna16_marlin_gemm` (only host-side `torch.empty` allocations of cache2/cache13 and Python view slicing sit between the join and that GEMM; no GPU work).
- The non-capturing `record_stream(main_stream)` loop now covers every side-stream tensor consumed on the main stream: `topk_weights` (the fp32 one), `topk_ids` (returned), `local_ids`, `sorted_ids`, `block_experts`, `n_padded`.
- `block_size_m` selection is factored into `_block_size_m(num_tokens)` and called on the host in `kernel_fn` from `cfg['tokens']` (same loop, same candidates; `tokens == x.shape[0]` as before). `_experts` now takes `(x, topk_weights, local_ids, sorted_ids, block_experts, n_padded, block_size_m, state)` and no longer computes the cast, the block size or the align. cache13/cache2 are still allocated in `_experts` on the main stream.
- No new kernels, no precision change, no host syncs, nothing cached in state.
- Deviation: none in substance. The only nuance is that cache allocations happen after the join rather than strictly before it; torch.empty issues no GPU work, so the captured graph is the same as the spec's ordering.
- FusedMuseSparkRouting dtype: the SGLang source is not on this node, so whether its topk_weights are already contiguous fp32 could not be verified. The cast is kept on the side stream; if they are already fp32 contiguous, `.to()`/`.contiguous()` return the same tensor and the cast is a no-op (no launch), and only the remap + align move.
- Eager safety beyond the loop: `ffn_input` (main-stream allocation read on the side stream) stays referenced until after the join, as in the base; `logits`, the pre-cast topk_weights and moe_align's temporaries are allocated on the side stream and freed there, so no extra record_stream is needed.

## Rules check

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

## Result

- job `f70d1f2eaef842d2` (check), status done, passed True, tests 5/5
- speedup 1.0076x, 95% CI [0.9925, 1.0227], score 384474.7931363558 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-align-side-stream.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-align-side-stream.patch`; branch `hyp/muse_prefill_layer/moe-align-side-stream`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1078.66 | 1049.83 | 1.0058 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1434.85 | 1458.98 | 1.0014 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3367.03 | 3209.04 | 1.0054 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4949.82 | 4894.58 | 1.0022 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18279.1 | 17632.24 | 1.0012 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 16995.2 | 16784.84 | 0.9997 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21535.66 | 21441.63 | 1.0216 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 25286.61 | 24539.1 | 1.0043 | flat |

no shape moved outside noise. Expected hit: 1/3. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-side-stream`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

