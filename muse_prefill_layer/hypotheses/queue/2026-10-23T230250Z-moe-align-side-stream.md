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
status: queued
review_at: 2026-10-24T044735Z
review_companions: null
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

