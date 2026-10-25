# muse_decode_layer · 2026-10-25T042712Z · check · sliding-batched-triton-b16-band

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T042121Z-sliding-batched-triton-b16-band.md` (branch `hyp/muse_decode_layer/sliding-batched-triton-b16-band`).

- **Job:** `02bf417257e14a3d` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `1cabea7`

## Goal
Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Re-apply decode's harvested sliding-fused-batched-decode-port (done/neutral patch: the Triton _sliding_fused_kernel batched path with 16 splits, ported from attention's landed sliding-deep-splits-on-cuda-splitkv), but narrow its config gate from 'sliding, q_len 1, batch > 1' to sliding, q_len 1, page 1, 2 <= batch <= 16 (sliding decode_16 and the batch-8/9 tests). batch 17+ (sliding decode_32) stays on master FA3 so that band is left for sliding-cluster-c4-decode32-port and the two can later stack without overlap. No other change; apply the patch hunks onto master and resolve any conflict in kernel_fn by keeping both master's full-decode split-KV route and this sliding route.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (+67/-2). Hand-merged the patch's kernel branch onto master; the patch's helper hunk was NOT applied (master's `_inv_freq`, `_counters`, `_sliding_fused_kernel` are used as they are).
- New `_sliding_b16_gate(cfg)`: window != 0, q_len 1, page_size 1, 2 <= batch <= 16, hq 16 / hkv 2 / d 64. Config only. New `_sliding_forward_b16` (same launch as master's `_sliding_forward`, `splits = 16` hard-coded) and `_sliding_b16_or_none` (try/except, per-config on/off dict, prints `PATH CHOSEN = FUSED TRITON S=16` once, or `PATH CHOSEN = MASTER (FA3)` after a failure; never raises).
- `kernel_fn`: `o = None`; `if batch == 1 sliding: _sliding_forward` / `elif _sliding_b16_gate(cfg): o = _sliding_b16_or_none(...)` / `if o is None:` master's existing body unchanged (indentation kept; a batch 17..32 elif can sit directly under the b16 elif). Master's splitkv/paired full-decode route is untouched.
- Review fixes applied (the review wins over the spec text): S=16 hard-coded rather than `_num_splits` (4 at B=16); no duplicated helpers; gate has the page==1 term, so only the batch-9 test exercises the path (the page-64 batch-8 test stays on FA3), as the review accepted. Batch 1 stays on master's `_sliding_forward`.
- Not done (profile read is post-run): record the `_sliding_fused_kernel` row vs FA3 main + combine + glue from the profile.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/synchronize in the new code; the branch is config-only; workspace allocations are by shape, not values.
- [x] Nothing derived from `data` is cached across calls: only a config-keyed on/off flag and master's zero-reset ticket counters (hold no data).
- [x] Weights untouched (n/a: no weight change).
- [x] Activations BF16, KV cache FP8, query bf16 in the fused kernel (finer than E4M3), same kernel as master's batch-1 path.
- [x] Attention sees every key of the window (kernel unchanged; all splits cover [lo, seq-1] plus the new keys).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess or child process.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune; Triton compile errors fall back to master, never raise.

okc_lint: `OK with warnings` (only pre-existing host-sync warns in prepare(), outside kernel_fn's call graph). py_compile OK, __pycache__ removed.

## Risks / what could fail
low
Its earlier run's verify_32 'loss' was on untouched paths (drift).

## Result (after the run)
- job `02bf417257e14a3d` (check), status done, passed True, tests 13/13
- speedup 1.2743x, 95% CI [1.2552, 1.2934], score 17751.053533663424 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/sliding-batched-triton-b16-band.result.json`, `muse_decode_layer/hypotheses/done/neutral/sliding-batched-triton-b16-band.patch`; branch `hyp/muse_decode_layer/sliding-batched-triton-b16-band`
