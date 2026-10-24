# muse_decode_layer · 2026-10-24T231258Z · check · moe-dispatch-prep-on-side-stream

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T184303Z-moe-dispatch-prep-on-side-stream.md` (branch `hyp/muse_decode_layer/moe-dispatch-prep-on-side-stream`).

- **Job:** `fe3422db9ea540d8` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `6c8d697`

## Goal
Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto state["side_stream"], after state["routing"](...), so it overlaps the pooler GEMM and the pre-expert RMSNorm on the main stream instead of running serially after the join. Split _experts into two functions with the same code. (a) _expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state) does exactly master: shifted = (topk_ids - local_start) % N_EXPERTS; local_ids = bank["expert_map"][shifted].contiguous(); tw = topk_weights.to(torch.float32).contiguous(); the block_size_m pick over BLOCK_SIZE_M_CANDIDATES with min(..., MARLIN_BLOCK_SIZE_M_CAP); sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS); returns them all. (b) _expert_compute(x, dispatch, state) holds the rest of master _experts unchanged (cache2/cache13 allocation, w13 Marlin, silu_and_mul, w2 Marlin, the post-norm tail at both T ranges). In kernel_fn: inside the existing `with torch.cuda.stream(side_stream):` block, right after the routing call, call dispatch = _expert_dispatch(topk_ids, topk_weights, data["local_start"], tokens, state). On main, after expert_input = _gemm(ffn_input, dense["pooler"]) and BEFORE main_stream.wait_stream(side_stream), compute xn = state["pre_expert_norm"](expert_input). Then the existing join; when not capturing, call record_stream(main_stream) on every tensor created on the side stream (topk_weights, topk_ids, and every tensor in dispatch), as master already does for the first two. Then expert_output = _expert_compute(xn, dispatch, state). topk_ids returned unchanged. Kernel set, arguments and dtypes stay identical to master; only the stream and order change.

## What changed, and where
Only muse_decode_layer/submission.py in the worktree. Added `_expert_dispatch(topk_ids, topk_weights, local_start, num_tokens, state)` (shift, expert_map gather, fp32 cast, moe_align_block_size with `_block_size_m`) and an optional `dispatch=` argument on `_experts` (when given it replaces the in-function gather/cast/align). Deviation: instead of splitting `_experts` into two functions, I kept one function with an optional argument so the T<=32 dense-chain path and the prepare() self-check call it unchanged (byte-identical). In the T>32 branch of kernel_fn the dispatch runs on the side stream right after routing; pre_expert_norm runs on main before `main_stream.wait_stream(side_stream)`; record_stream(main_stream) (not when capturing) now covers topk_weights, topk_ids, local_ids, tw and the three align tensors. Side stream is still the one created in prepare(); events/wait_stream only, no host syncs. topk_ids returned unchanged.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/synchronize(); stream fork/join uses wait_stream only; no input-value control flow.
- [x] Nothing derived from `data` cached across calls: dispatch tensors are per-call locals.
- [x] Weights: n/a: no weight changes.
- [x] Activations dtype: n/a: kernels, args and dtypes identical to master.
- [x] Attention: n/a: untouched.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.

Lint: `OK with warnings: muse_decode_layer/submission.py` (only host-sync warnings in prepare()-only code, none in kernel_fn). py_compile passed; __pycache__ removed.

## Risks / what could fail
low
Read GPU busy and per-shape times; the kernel rows themselves should not change. Risk: if the side branch outgrows pooler + pre-norm (T=256) the saving shrinks, and side-stream contention with the pooler (the unverified suspect behind bf16-shape-gated-triton-gemms-small-t losing at T=8/16) could cost a little. OVERLAP: small-t-launch-minimal-dense-chain-pdl (in judge) removes the side stream and fuses map+align at T<=32 when its path is ON, and moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 when ON. If both land ON this is moot; if either falls back, this covers its range. Judge it against those runs PATH CHOSEN lines.

## Result (after the run)
- job `fe3422db9ea540d8` (check), status done, passed True, tests 13/13
- speedup 1.2294x, 95% CI [1.2110, 1.2479], score 18398.940510688506 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/moe-dispatch-prep-on-side-stream.result.json`, `muse_decode_layer/hypotheses/done/neutral/moe-dispatch-prep-on-side-stream.patch`; branch `hyp/muse_decode_layer/moe-dispatch-prep-on-side-stream`
