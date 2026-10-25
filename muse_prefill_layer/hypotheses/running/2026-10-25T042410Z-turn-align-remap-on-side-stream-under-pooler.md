---
problem: muse_prefill_layer
slug: turn-align-remap-on-side-stream-under-pooler
created: 2026-10-25T042410Z
base: master
area: "MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM"
change: >
  Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on
  the main stream after the router join: 16.9 us at turn_1 (1.7%) and 70.5 us at turn_4 (2.1%),
  while the side stream (router GEMM ~15-25 us + routing ~33 us at turn_4) finishes well inside
  the pooler GEMM on the main stream. In kernel_fn, inside the existing `with
  torch.cuda.stream(side_stream)` block, after state['routing'], also run local_ids =
  _local_expert_ids(topk_ids, data['local_start']) and, when tokens <= MARLIN_MAX_TOKENS and
  state['align_par_ok'], _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) with
  block_size_m from the same shape-only BLOCK_SIZE_M_CANDIDATES loop _experts_marlin uses (factor
  it into a helper both call). Pass the resulting (sorted_ids, block_experts, n_padded) into
  _experts_marlin through a new optional argument, which then skips its own align call. The
  existing main_stream.wait_stream(side_stream) stays where it is and the record_stream loop
  covers the new outputs. T > 4096 and the align_par_ok=False fallback are unchanged. Relation:
  moe-align-side-stream (done/neutral 1.0076 on the old parity base) moved SGLang's
  moe_align_block_size path; this re-tests the placement with master's current align kernel, which
  is the one now in the profile.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "_align_parallel (70.5 us) leaves the critical path under the pooler (~110-150 us at T=3012): turn_4 -40 to -70 us (-1.2 to -2%)"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "17 us align at T=753 partly hidden: -5 to -15 us (-0.5 to -1.5%), may be inside the router branch's slack"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T > 4096 path untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T043310Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T043310Z
base_sha: 1a306457948093d586d8985f430235643dff1d1c
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/turn-align-remap-on-side-stream-under-pooler
worktree: .worktrees/muse_prefill_layer/turn-align-remap-on-side-stream-under-pooler
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043641Z
impl_commit: 7b424f9049e560c81fe66a434c0ffc4811736f1a
submission_sha256: 02617200060af1e25c246f4c985cf9a423a95941450d9dbf8c8f1ea34dc022a4
norm_sha256: ccaca75a2e779cb7590e98d1dbf59485ea5c17682b3715e8c1520ac76bd34715
submission_blob: 116f43fe0bea9fd64170c2f38973e38d7d3be8b4
lint_warnings: 15
attempt_file: 2026-10-25T043641Z-check-turn-align-remap-on-side-stream-under-pooler.md
prepared_at: 2026-10-25T043641Z
---

# turn-align-remap-on-side-stream-under-pooler

## Change

Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) and 70.5 us at turn_4 (2.1%), while the side stream (router GEMM ~15-25 us + routing ~33 us at turn_4) finishes well inside the pooler GEMM on the main stream. In kernel_fn, inside the existing `with torch.cuda.stream(side_stream)` block, after state['routing'], also run local_ids = _local_expert_ids(topk_ids, data['local_start']) and, when tokens <= MARLIN_MAX_TOKENS and state['align_par_ok'], _align_parallel(local_ids, 0, LOCAL_EXPERTS, block_size_m) with block_size_m from the same shape-only BLOCK_SIZE_M_CANDIDATES loop _experts_marlin uses (factor it into a helper both call). Pass the resulting (sorted_ids, block_experts, n_padded) into _experts_marlin through a new optional argument, which then skips its own align call. The existing main_stream.wait_stream(side_stream) stays where it is and the record_stream loop covers the new outputs. T > 4096 and the align_par_ok=False fallback are unchanged. Relation: moe-align-side-stream (done/neutral 1.0076 on the old parity base) moved SGLang's moe_align_block_size path; this re-tests the placement with master's current align kernel, which is the one now in the profile.

## Expected

- `*load=turn_4*` faster: _align_parallel (70.5 us) leaves the critical path under the pooler (~110-150 us at T=3012): turn_4 -40 to -70 us (-1.2 to -2%)
- `*load=turn_1*` faster: 17 us align at T=753 partly hidden: -5 to -15 us (-0.5 to -1.5%), may be inside the router branch's slack
- `*load=first_turn*` unchanged: T > 4096 path untouched

## Risk

Kill: the profile shows _align_parallel overlapping the pooler (GPU busy below the kernel sum) and turn_4 does not move beyond 1% -> placement is not the cost. Risk: a longer side branch than the pooler at turn_1 just moves the wait; the judge sees no harm beyond noise.

## Adversarial review

Reviewed 2026-10-25T043310Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, rules-clean placement change with real slack to hide into at turn_4 (side branch ~105 us + align 70 us under a ~260-300 us pooler); can never be slower than master barring SM contention. Small headline (~+0.5-0.8%) and only readable from the profile's GPU-busy/overlap lines, not from shape z.

### Justification
Profile (submit 502f2eab): _align_parallel_kernel 70.5/69.3 us at turn_4 and 16.9/16.8 us at turn_1, serial on the main stream after the join. Side branch today at turn_4: router GEMM (~39-43 us) + logits add (~30 us) + routing_kernel 32.7 us = ~105 us; the pooler is one of the big nvjet rows (~260-300 us at T=3012). GPU busy 4879 vs kernel-sum 4978 us shows the router branch already overlaps ~100 us, so concurrency works on this GPU. New critical path = max(pooler, branch+align) <= old max(pooler, branch) + align, so it is monotone except for SM/issue contention with the pooler. At turn_1 the pooler is only ~40-50 us and the branch ~35-50 us, so most of the 17 us will not hide - expect ~0 there. Weakest link: 'moe-align-side-stream' (old parity base) measured 1.0076 neutral for the same idea; the difference is that the align kernel moved now is 70 us, not SGLang's cheaper count_and_sort - plausible but unproven.

### Confounders on the current kernel
- router-hilo-one-gemm-n512 (in stack-tonight-five): sets the side branch length (one N=512 fp32-out GEMM + add); a longer branch would eat the slack. Unchanged here.
- moe-align-local-routes-parallel-port (in stack-tonight-five): is the 70 us kernel being moved; its 33-program latency-bound shape is what makes it overlap-friendly (few SMs, little smem).
- Queued experts-turn4-cuda-dequant-producer-ss-wgmma and experts-turn1 port also call _align_parallel inside _experts_marlin with their own bsm (128/32); merge conflict if several land - the helper should take bsm as a parameter.

### Interference and A/B needs
No companion: the effect is read directly from the profile (the _align_parallel row starts before the pooler ends; GPU busy drops vs kernel sum by ~+60 us at turn_4). A revert-A/B would not add information.

### Open questions
- Does the 33-CTA align co-reside with the coopA pooler kernel or wait for SM slots? The profile timeline answers it.
- Could the turn_1 branch exceed the pooler and just move the wait? Harmless, but if so restrict to T > 1024 in a follow-up.


## Implementation notes

Edited only `muse_prefill_layer/submission.py` in the worktree (+33/-12). Changes:
- New helper `_marlin_block_size_m(num_tokens)` (shape-only BLOCK_SIZE_M_CANDIDATES loop); `_experts_marlin` uses it.
- `_experts` / `_experts_marlin` take an optional `align=(sorted_ids, block_experts, n_padded)`; when given, `_experts_marlin` skips its own align. With `align=None` (T > 4096, align_par_ok False) behaviour is unchanged. first_turn path unchanged.
- In `kernel_fn`, inside the existing `with torch.cuda.stream(side_stream)` block after `state["routing"]`: `local_ids = _local_expert_ids(topk_ids, data["local_start"])` (device tensor, no host read), then, if `ffn_input.shape[0] <= MARLIN_MAX_TOKENS` and `state["align_par_ok"]`, `_, align = _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm)`.
- Review overrides applied: the `data['local_start']` tensor is NOT passed to `_align_parallel`'s scalar argument (it gets the Python 0 on already-remapped ids, same as master's call); `record_stream(main_stream)` now covers topk_weights, topk_ids, local_ids and the three align outputs (skipped while capturing, as before); the existing `main_stream.wait_stream(side_stream)` join (an event edge under capture) sits after all side-stream work and before any consumer. The fork `side_stream.wait_stream(main_stream)` is unchanged. Graph-capturable.
- Deviation: the review's "pass bsm as helper parameter" is met by `_align_parallel`'s existing bsm argument; the block-size pick helper takes num_tokens.
- Expected effect: about +0.5-0.8% headline (turn_4 -40 to -70 us, turn_1 about 0). Read it from the profile's GPU-busy vs kernel-sum / overlap lines (the _align_parallel row should start before the pooler ends), not from shape z.
- Risk: the side branch is now longer; the 33-CTA align may not co-reside with the pooler kernel; at turn_1 the wait may just move.

## Rules check

- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/synchronize; the side stream forks and joins from the capturing stream via wait_stream; record_stream guarded by is_current_stream_capturing.
- [x] Nothing derived from `data` is cached across calls (new tensors are per-call locals).
- [x] Weights: n/a, untouched.
- [x] Activations/KV/queries precision: n/a, untouched.
- [x] Attention: n/a, untouched.
- [x] No timing patches, harness access, module edits, process spawning: diff is pure kernel_fn / helper plumbing.
- [x] All work inside the timed call; no output caching or replay.
- [x] No new CUDA/load_inline: build-brief arch/mbarrier/self-check items n/a: reason, no CUDA added.

okc_lint: "OK with warnings: submission.py" (only pre-existing host-sync warnings in prepare()-side code, outside kernel_fn). py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only; no GPU). **Verdict: dispatch** (no fixes).

- Fork/join: `side_stream.wait_stream(main_stream)` precedes the `with torch.cuda.stream(side_stream)` block (fork); `main_stream.wait_stream(side_stream)` follows all side work, including `_local_expert_ids` and `_align_parallel`, and precedes `pre_expert_norm`/`_experts` (join). Under capture both are event edges, so the side branch stays inside the graph.
- record_stream: `cross` = topk_weights, topk_ids, local_ids, sorted_ids, block_experts, n_padded (eager only, as on master). The discarded `local_ids` output of `_align_parallel` lives and dies on the side stream.
- No host sync or value-dependent host control: `data["local_start"]` stays a device tensor read inside `_local_expert_ids_kernel`; `_align_parallel` gets the Python 0 on already-remapped ids, the same call master made in `_experts_marlin`. The remap is master's kernel on the same inputs, only moved to the side stream, so its results are identical. cap, n_blocks and bsm depend only on the shape, so replaying the captured graph with new routing is safe (nothing routing-dependent is computed on the host).
- Gate: `num_tokens <= MARLIN_MAX_TOKENS and align_par_ok`. align_par_ok is set in prepare() by `_align_par_setup`. The gate matches the Marlin gate in `_experts` (x.shape[0] == ffn_input.shape[0]), and `_marlin_block_size_m(num_tokens)` gives the same bsm on both sides. first_turn (T > 4096) gets align=None and runs the Triton path unchanged; only its local_ids launch moves to the side stream.
- okc_lint: OK with warnings (the warnings are pre-existing and in prepare()-side code only); py_compile OK.

Residual risk: performance only. The longer side branch may not hide under the pooler at turn_1, or may contend with it for SMs at turn_4.
