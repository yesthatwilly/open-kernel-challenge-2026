# muse_decode_layer · 2026-10-24T231258Z · check · sliding-fused-batched-decode-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T184247Z-sliding-fused-batched-decode-port.md` (branch `hyp/muse_decode_layer/sliding-fused-batched-decode-port`).

- **Job:** `b01fd2df41a040b3` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `3ddcc4f`

## Goal
Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-on-cuda-splitkv, job 698d4cd3838c4a63: sliding decode_16 20.7 -> 15.3 us, decode_32 24.9 -> 24.2 us there, with 16 splits at every decode batch) to decode_layer kernel_fn for sliding-window decode at batch > 1. Copy verbatim from muse_attention_decode/submission.py on master: LOG2E, _INV_FREQ and _inv_freq, _COUNTERS and _counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel. Add _sliding_forward_batched(data, qkv, cfg), adapted from attention _sliding_forward: the qkv operand is master qkv = _gemm(data["attn_input"], dense["qkv_gate"]) ([T, 2304] bf16, pass qkv.stride(0)) instead of data["qkv"]; splits = 16 (q_len 1, attention rule); same ws_acc / ws_ml / _counters buffers, same launch arguments (positions, cache_loc, seq_lens, page_table, inv_freq, k_scale, v_scale, k_cache/v_cache viewed as uint8, HQ, HKV, D, L=1, M, LN, WINDOW=cfg window, PAGE=cfg page_size, S=16, BLOCK_N=64, ROPE=bool(rope_theta), EPS=RMS_EPS, num_warps=4, num_stages=2); it returns [B, 1, 1024] bf16 that is already QK-normed, RoPEd, cached (FP8 k/v stored at cache_loc), attended over the 2048-token window with FP8-rounded queries, split-combined and per-head RMSNorm x sigmoid(gate) gated. Gate in kernel_fn on config only: fused = window != 0 and q_len == 1 and batch > 1 and hq == 16 and hkv == 2 and d == 64. On that branch skip FusedQKNorm, the k/v/q FP8 casts, store_cache, flash_attn_with_kvcache and state["o_norm_gate"], and feed o = out.reshape(tokens, hq * d) to the unchanged o GEMM; everything after the o GEMM is master. Every other shape keeps master byte for byte, including sliding decode at batch 1 (that shape is the in-judge restack decode_1 port). If decode-restack-sliding-port-paired-pad-gemms-t1 has landed on master by claim time, its copies of these helpers are already in the file: reuse them and only widen its gate to batch > 1 with splits 16 at q_len 1, never add a second copy.

## What changed, and where
- Worktree muse_decode_layer/submission.py only. Inserted just before kernel_fn, copied verbatim from muse_attention_decode/submission.py (master): _INV_FREQ, _inv_freq, LOG2E, _COUNTERS, _counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel. (_num_splits is copied but unused: splits fixed at 16 for q_len 1.)
- Added _sliding_forward_batched(data, qkv, cfg): attention's _sliding_forward with qkv = the qkv GEMM output (stride(0) passed), splits = 16, same workspaces/counters/launch args.
- kernel_fn: `fused = window != 0 and q_len == 1 and batch > 1 and hq == 16 and hkv == 2 and d == 64` (config only). On it: o = out.reshape(tokens, hq*d) feeds the unchanged o GEMM. Else branch is the master attention code verbatim, only re-indented (diff shows it as changed lines). Everything after is untouched.
- Deviation: none. Per review, no merge from the restack branch.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: new path has torch.empty + a Triton launch, no sync; _counters zeroed once on first eager call, branches on config only.
- [x] Nothing derived from `data` cached: _counters/_inv_freq key on (batch, hkv, device) / (theta, d, device); inv_freq is a function of config, counters hold no data (reset to 0 by the kernel).
- [x] Weights: n/a: no weight change; prepare() untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (k/v stored E4M3 at cache_loc); queries bf16 (finer than FP8 cap).
- [x] Attention sees every key: whole 2048-token window plus the new key; no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (checked in diff).
- [x] All work inside the timed call; no output caching or replay.
- [x] Gate shape-disjoint: full, verify (q_len 8) and batch 1 take the master branch.

okc_lint: "OK with warnings" (only host-sync warnings outside kernel_fn's call graph, in prepare/self-check helpers that predate this change; no errors). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Proven kernel (attention 1.309x -> 1.395x lineage); the batch-1 port of the same kernel gave +3.4-4.3% on sliding decode_1 here, so the decode_layer glue it removes is real. Overlap: decode-restack-sliding-port-paired-pad-gemms-t1 (in judge) adds the same helpers for batch 1 only; shapes are disjoint, code overlaps (merge, do not duplicate). The fused kernel zeroes _counters once on first eager call and leaves them at 0 (no data carried). Risk: kernel never ran inside the whole layer at batch 16/32; the 13 tests include sliding decode at batch 8 (page 64) and batch 9 (page 1).

## Result (after the run)
- job `b01fd2df41a040b3` (check), status done, passed True, tests 13/13
- speedup 1.2248x, 95% CI [1.2064, 1.2431], score 18469.002971446916 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-fused-batched-decode-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-fused-batched-decode-port.patch`; branch `hyp/muse_decode_layer/sliding-fused-batched-decode-port`
