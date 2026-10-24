---
problem: muse_decode_layer
slug: moe-dispatch-prep-on-side-stream
created: 2026-10-24T184303Z
base: master
area: MoE align/dispatch (stream placement)
change: >
  Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto
  state["side_stream"], after state["routing"](...), so it overlaps the pooler GEMM and the
  pre-expert RMSNorm on the main stream instead of running serially after the join. Split _experts
  into two functions with the same code. (a) _expert_dispatch(topk_ids, topk_weights, local_start,
  num_tokens, state) does exactly master: shifted = (topk_ids - local_start) % N_EXPERTS;
  local_ids = bank["expert_map"][shifted].contiguous(); tw =
  topk_weights.to(torch.float32).contiguous(); the block_size_m pick over BLOCK_SIZE_M_CANDIDATES
  with min(..., MARLIN_BLOCK_SIZE_M_CAP); sorted_ids, block_experts, n_padded =
  moe_align_block_size(local_ids, block_size_m, N_EXPERTS); returns them all. (b)
  _expert_compute(x, dispatch, state) holds the rest of master _experts unchanged (cache2/cache13
  allocation, w13 Marlin, silu_and_mul, w2 Marlin, the post-norm tail at both T ranges). In
  kernel_fn: inside the existing `with torch.cuda.stream(side_stream):` block, right after the
  routing call, call dispatch = _expert_dispatch(topk_ids, topk_weights, data["local_start"],
  tokens, state). On main, after expert_input = _gemm(ffn_input, dense["pooler"]) and BEFORE
  main_stream.wait_stream(side_stream), compute xn = state["pre_expert_norm"](expert_input). Then
  the existing join; when not capturing, call record_stream(main_stream) on every tensor created
  on the side stream (topk_weights, topk_ids, and every tensor in dispatch), as master already
  does for the first two. Then expert_output = _expert_compute(xn, dispatch, state). topk_ids
  returned unchanged. Kernel set, arguments and dtypes stay identical to master; only the stream
  and order change.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "the sub/remainder elementwise pair, the expert_map gather and SGLang align (~3.6 us _moe_align_small_numel + launches, ~7-8 us) leave the critical path: side branch (router gemv 5 + routing + these) ~16 us < pooler + pre-norm ~22 us; -6..-8 us of 145-157 us, +4-5%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "same kernels off the path: side ~22 us (router hi+lo 9.4 + add + routing + map + align) < pooler 20 + pre-norm 5.3; -6..-8 us, +2-3%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: same, +1.5-2.5%
  - shapes: "*load=verify_1 *"
    direction: faster
    why: same, +3%
  - shapes: "*load=verify_16 *"
    direction: faster
    why: SGLang align at T=128 ~10 us + map now overlaps pooler 24.7 + pre-norm 6; side ~33 us vs 31 so ~-8..-10 us, +1.5-2%
  - shapes: "*load=verify_32 *"
    direction: faster
    why: same, ~-8 us, +1%
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T185010Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T185010Z
base_sha: 6f8cac5640783bda29e10ba56c80ba50fc396141
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/moe-dispatch-prep-on-side-stream
worktree: .worktrees/muse_decode_layer/moe-dispatch-prep-on-side-stream
mode: check
implemented_at: 2026-10-24T231258Z
impl_commit: 6c8d697b4b4037e842d132d72243352b85abe10a
submission_sha256: 8c1a130276e46913ceebef067b4777e81d6f7f966b13e607170313cb6c917082
norm_sha256: 6bfee5a728e9525e163cbdc21c63b6ff96a5892f79bfdb7283a3473969ccd303
submission_blob: f3200629fd5e0cab6475535321218a16b010c752
lint_warnings: 18
attempt_file: 2026-10-24T231258Z-check-moe-dispatch-prep-on-side-stream.md
prepared_at: 2026-10-24T231258Z
job: fe3422db9ea540d8
dispatched_at: 2026-10-24T231312Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2294349224543215
speedup_ci: [1.2109933986175068, 1.2478764462911363]
harvested_at: 2026-10-24T232003Z
best_at_harvest: 1.228328123558382
promotion: null
---

# moe-dispatch-prep-on-side-stream

## Change

Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto state["side_stream"], after state["routing"](...), so it overlaps the pooler GEMM and the pre-expert RMSNorm on the main stream instead of running serially after the join. Split _experts into two functions with the same code. (a) _expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state) does exactly master: shifted = (topk_ids - local_start) % N_EXPERTS; local_ids = bank["expert_map"][shifted].contiguous(); tw = topk_weights.to(torch.float32).contiguous(); the block_size_m pick over BLOCK_SIZE_M_CANDIDATES with min(..., MARLIN_BLOCK_SIZE_M_CAP); sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS); returns them all. (b) _expert_compute(x, dispatch, state) holds the rest of master _experts unchanged (cache2/cache13 allocation, w13 Marlin, silu_and_mul, w2 Marlin, the post-norm tail at both T ranges). In kernel_fn: inside the existing `with torch.cuda.stream(side_stream):` block, right after the routing call, call dispatch = _expert_dispatch(topk_ids, topk_weights, data["local_start"], tokens, state). On main, after expert_input = _gemm(ffn_input, dense["pooler"]) and BEFORE main_stream.wait_stream(side_stream), compute xn = state["pre_expert_norm"](expert_input). Then the existing join; when not capturing, call record_stream(main_stream) on every tensor created on the side stream (topk_weights, topk_ids, and every tensor in dispatch), as master already does for the first two. Then expert_output = _expert_compute(xn, dispatch, state). topk_ids returned unchanged. Kernel set, arguments and dtypes stay identical to master; only the stream and order change.

## Expected

- `*load=decode_1 *` faster: the sub/remainder elementwise pair, the expert_map gather and SGLang align (~3.6 us _moe_align_small_numel + launches, ~7-8 us) leave the critical path: side branch (router gemv 5 + routing + these) ~16 us < pooler + pre-norm ~22 us; -6..-8 us of 145-157 us, +4-5%
- `*load=decode_16 *` faster: same kernels off the path: side ~22 us (router hi+lo 9.4 + add + routing + map + align) < pooler 20 + pre-norm 5.3; -6..-8 us, +2-3%
- `*load=decode_32 *` faster: same, +1.5-2.5%
- `*load=verify_1 *` faster: same, +3%
- `*load=verify_16 *` faster: SGLang align at T=128 ~10 us + map now overlaps pooler 24.7 + pre-norm 6; side ~33 us vs 31 so ~-8..-10 us, +1.5-2%
- `*load=verify_32 *` faster: same, ~-8 us, +1%

## Risk

Read GPU busy and per-shape times; the kernel rows themselves should not change. Risk: if the side branch outgrows pooler + pre-norm (T=256) the saving shrinks, and side-stream contention with the pooler (the unverified suspect behind bf16-shape-gated-triton-gemms-small-t losing at T=8/16) could cost a little. OVERLAP: small-t-launch-minimal-dense-chain-pdl (in judge) removes the side stream and fuses map+align at T<=32 when its path is ON, and moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 when ON. If both land ON this is moot; if either falls back, this covers its range. Judge it against those runs PATH CHOSEN lines.

## Adversarial review

Reviewed 2026-10-24T185010Z by an adversarial subagent (model opus).

**Verdict: proceed.** Since small-t-launch-minimal-dense-chain-pdl landed, master runs no side stream at T<=32, so this change only affects verify_16 (T=128) and verify_32 (T=256). There it is a plausible, low-risk overlap worth about 1-2% per shape, but only about +0.4% on the headline, so judge it on those four shapes' times and not on the headline.

### Justification
The mechanism holds at T>32 only. On master (f1f2f35c), kernel_fn takes the dense-chain path (_dl_forward, no side stream) whenever tokens <= DL_MAX_TOKENS=32, and the BEST profile shows dl_pooler_router_kernel at decode_1/16/32 and verify_1, so that path was ON. That makes all four T<=32 'faster' expectations (decode_1/16/32, verify_1) wrong: those shapes run code this change does not touch and should read flat. The hypothesis was written while the dense chain was still being judged and admits the overlap. At T=128/256 the master non-dl branch runs, in order: pooler GEMM (~25-28 us, nvjet 64x64 / 128x128) on main in parallel with the router hi+lo mm, the add and routing on the side stream; then the join; then pre_expert_norm (~6 us), the sub/mod, the expert_map gather, the fp32 cast and moe_align_block_size (~8-10 us at T=128 by the hypothesis's own estimate), all serial. Moving the dispatch before the join and the pre-norm before the join changes the critical path from ~max(pooler, router) + norm + dispatch to max(pooler + norm, router + routing + dispatch). The saving is roughly min(norm, dispatch) plus slack, about 6-10 us of 448/599 us (verify_16) and 627/911 us (verify_32): +1-2% on sliding and less on full. The per-shape median err is 4-5 us, so a ~8 us move is only marginally resolvable per shape, a bit better across the two loads (46+16 weights). It is invisible in the headline geomean: about +0.4% against a CI of about +/-1.5%. Weakest link: the 'side ~33 us vs main ~31 us' budget at T=128, and worse at T=256. The router hi+lo mm, routing and align all grow with T, while pooler + norm is weight-bound and nearly flat. At verify_32 the side branch is likely the critical path after the change, so the saving drops to whatever dispatch time still hides behind the pooler. On top of that, the small side kernels compete with a full-SM pooler GEMM for SMs, so 'overlap' can turn into queueing (the same unverified contention suspect named for bf16-shape-gated-triton-gemms-small-t). The profile top-4 does not list align/gather/routing at T=128/256, so the ~10 us align figure is unverified.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (base): at T<=32 the change is a no-op because there is no side stream. Decode_1/16/32 and verify_1 deltas are pure noise and must not be credited. If the run's DL log line says 'PATH CHOSEN = MASTER at every T' (self-check or compile fallback), the change also covers T<=32 and the decode expectations apply. The judge must read that line before attributing anything.
- sglang-pipeline-parity-router-hilo-m8: it sets the side-branch length at T>1 (two bf16 mm, hi and lo, plus the add, ROUTER_FP32_MAX_TOKENS=1). The saving is capped by how much the side branch exceeds pooler + norm, so a loss or a flat result at verify_32 could reflect the hi+lo router being long rather than the dispatch move itself.
- marlin-block-m-cap-16: it fixes block_size_m=16 at T=128/256, which sets moe_align_block_size's padded output size and its cost. With the old 48/64 the align would be cheaper and the overlap saving smaller. This is a minor magnitude effect only, not a sign flip.
- post-expert-norm-triton-small-t and fa3-hdim128-head-pairing-batched: neither confounds. The first is T<32 only and the second only touches attention before the fork.

### Interference and A/B needs
No companion is warranted. The change keeps the same kernel set, arguments and dtypes and only moves stream placement and order, so a verify_16/32 gain can only come from the overlap, and a loss can only come from side-stream SM contention or the side branch becoming the critical path. A control that splits the two sub-moves (pre-norm before the join vs dispatch on the side stream) would chase a ~3-5 us difference below the 4-5 us per-shape err, so it could not be read. An ab-revert of the hilo router would change the numerics path and is not meaningful. For attribution: read verify_16/verify_32 per-shape medians against BEST job 81ee6714b7614a99 (sliding 447.6/626.8 us, full 599.4/910.9 us), check that GPU busy rose or stayed flat while the kernel rows stayed unchanged, and confirm the DL line says PATH CHOSEN = DENSE-CHAIN at T <= 32. Treat decode/verify_1 as controls that should be flat. moe-chain-cuda-swapab-expert-once-large-t FAILED, so master's align at T>32 stays and this hypothesis stands on its own: it does not depend on that parent winning.

### Open questions
- Does moe_align_block_size at T=128/256 with block 16 actually take ~8-10 us? It is not in the profile top-4. Its real cost bounds the gain.
- At T=256, does router hi+lo + routing + dispatch exceed pooler + pre-norm? If so, the verify_32 gain collapses to near zero.
- Does the nvjet pooler GEMM occupy all 132 SMs for its whole duration, so that side-stream kernels serialize anyway? A GPU-busy and overlap read from the profile would settle it.


## Implementation notes

Only muse_decode_layer/submission.py in the worktree. Added `_expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state)` (shift, expert_map gather, fp32 cast, moe_align_block_size with `_block_size_m`) and an optional `dispatch=` argument on `_experts` (when given it replaces the in-function gather/cast/align). Deviation: instead of splitting `_experts` into two functions, I kept one function with an optional argument so the T<=32 dense-chain path and the prepare() self-check call it unchanged (byte-identical). In the T>32 branch of kernel_fn the dispatch runs on the side stream right after routing; pre_expert_norm runs on main before `main_stream.wait_stream(side_stream)`; record_stream(main_stream) (not when capturing) now covers topk_weights, topk_ids, local_ids, tw and the three align tensors. Side stream is still the one created in prepare(); events/wait_stream only, no host syncs. topk_ids returned unchanged.

## Rules check

- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/synchronize(); stream fork/join uses wait_stream only; no input-value control flow.
- [x] Nothing derived from `data` cached across calls: dispatch tensors are per-call locals.
- [x] Weights: n/a: no weight changes.
- [x] Activations dtype: n/a: kernels, args and dtypes identical to master.
- [x] Attention: n/a: untouched.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.

Lint: `OK with warnings: muse_decode_layer/submission.py` (only host-sync warnings in prepare()-only code, none in kernel_fn). py_compile passed; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading (no GPU on this node). **Verdict: dispatch** (no fixes).

- Graph capture: the fork (side.wait_stream(main) before the pooler) and join (main.wait_stream(side) before _experts) bracket all side-stream work; no syncs, no value-dependent control flow; record_stream is skipped while capturing and otherwise covers every side-produced tensor (topk_weights, topk_ids, local_ids, tw and the three moe_align_block_size tensors, all tensors). The shifted intermediate lives and dies on the side stream.
- Consume-after-join: local_ids, tw and align are read only inside _experts, after the join; topk_ids is returned after the join. xn is produced and consumed on main.
- Numerics: same kernels, args and dtypes as master; _block_size_m(tokens) on the side equals _experts own pick (num_tokens = x.shape[0] = tokens). With dl ON (T <= 32) the dense-chain branch is untouched, so T <= 32 is byte-identical; with dl OFF the T <= 32 fallback runs the same kernels in a new order and is also bitwise identical. The prepare() self-check calls _experts with the old signature unchanged.
- Spec deviation (one function with an optional dispatch= argument rather than a split) is behaviour-neutral. Lint OK, with prepare()-only host-sync warnings.

Residual risk: the gain shows only at verify_16 and verify_32 (T > 32), and it can vanish if the side branch outruns pooler + pre-norm at T=256 or the side kernels queue behind a full-SM pooler.

## Result

- job `fe3422db9ea540d8` (check), status done, passed True, tests 13/13
- speedup 1.2294x, 95% CI [1.2110, 1.2479], score 18398.940510688506 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/moe-dispatch-prep-on-side-stream.result.json`, `muse_decode_layer/hypotheses/done/neutral/moe-dispatch-prep-on-side-stream.patch`; branch `hyp/muse_decode_layer/moe-dispatch-prep-on-side-stream`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 132.46 | 0.9948 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 142.67 | 0.9975 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 207.51 | 1.0041 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 351.48 | 0.9931 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 243.71 | 1.0027 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 523.25 | 1.0051 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 184.66 | 1.0085 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 201.84 | 0.9976 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 447.55 | 436.99 | 0.9921 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 599.37 | 594.57 | 1.0032 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 626.79 | 607.78 | 1.0058 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 910.91 | 889.82 | 1.0039 | flat |

no shape moved outside noise. Expected hit: 0/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-dispatch-prep-on-side-stream`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

