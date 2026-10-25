# muse_attention_decode · 2026-10-25T034225Z · check · full-decode-splitkv-s3-fused-preattn

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T003621Z-full-decode-splitkv-s3-fused-preattn.md` (branch `hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn`).

- **Job:** `825baa03798f431a` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `dbb8456`

## Goal
Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token tiles + per-request ticket combine/norm-gate in splitkv_attn_sk_kernel, module muse_full_decode_splitkv_v2), whatever its score. Gate unchanged (_splitkv_gate: full, q_len 1, 1 < batch <= 64, tp8, head_dim 64, page 1). One change: remove the separate _pre_attention_kernel launch on the gated path so full decode_16/32 run ONE kernel (REWRITE_READ_2026-10-25 sections 1 and 5 list fused pre-attention as the idea worth keeping on this kernel). New kernel splitkv_attn_sk_fused_kernel in a NEW module muse_full_decode_splitkv_v3 (v2 stays compiled as the fallback): takes the raw qkv pointer/stride, positions, cache_loc, inv_freq, k_scale, v_scale instead of the precomputed FP8 q. (a) Q: when a CTA starts a request segment it builds that request q block in smem from qkv: the exact math of master _pre_attention_kernel for the 16 q heads (_norm_rope: scaleless RMSNorm over 64, eps 1e-5, RoPE with the same inv_freq table at positions[b], then the same FP8 cast and scale master writes), one warp per 2 heads, fp32 math; the redundancy (each request is touched by ~2-17 CTAs) is 2 KB of qkv per segment. (b) New K/V row: the CTA whose range contains request b last tile (key seq_b - 1, which is the token being decoded, stored at cache_loc[b]) computes the 2 kv heads new k (norm+rope+fp8) and v (fp8) exactly as master, stores them to k_cache/v_cache at cache_loc[b] (the cache update stays an output), then __syncthreads() before issuing that tile cp.async so the ring reads the stored row (same CTA, L2-coherent .cg loads). No other CTA reads key seq_b - 1, so no cross-CTA ordering is needed; the self-check asserts that ownership invariant. Launches on full decode_16/32 go from two (pre-attn, A) to one. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check with torch.cuda.synchronize() and a print after each kernel; never a child process). The self-check additionally compares the stored cache rows of v3 against master _pre_attention_kernel by rel L2 (not byte equality: LESSONS 2026-10-25, Triton sin/cos approximations) and q-dependent outputs by rel L2 < 5e-3. Fallback on any failure: v2 path (pre-attn + v2 kernel), printing PATH CHOSEN: full decode streamk v2 (fallback: <reason>); success prints PATH CHOSEN: full decode streamk fused-preattn v3. Every other shape byte for byte. First measurement is an okc check, never a scored submit.

## What changed, and where
Edited only `muse_attention_decode/submission.py` in the worktree (+643/-10 lines, nothing committed). Built on stage 2 as-is (relaxed arch gate `binaryVersion == 90`, effective grid `gx = min(gridDim.x, T)`, tickets re-zeroed in the `finally` after diagnostics; all kept).

- New module `muse_full_decode_splitkv_v3` (`_SK3_CPP`/`_SK3_CUDA`, built by `_build_sk(name, cpp, cuda, funcs)`; TORCH_CUDA_ARCH_LIST=9.0a around load_inline and restored; `-O3 -lineinfo`). Its source is stage 2's helper prefix (sliced from `_SK_CUDA` at import, so v2 is byte-identical) plus `splitkv_attn_sk_fused_kernel`: 4 stages, ticket protocol, same tile math and finalize as stage 2. v2 stays compiled as the fallback.
- Q build (review: overlap with the ring): the segment's first tile cp.asyncs the request's 2 KB bf16 q slice of qkv into a per-stage slot with the K/V rows. At consumption (tile landed, next 3 tiles in flight) each thread normalizes its own row (scaleless RMSNorm, eps 1e-5, 2-step shuffle across the 4 lanes of a quad), rounds to e4m3 (`__nv_cvt_float_to_fp8` SATFINITE, RN) and builds the MMA A fragments. Nothing runs ahead of the ring prologue. Unpaired q layout.
- New k/v row: the CTA whose stream-K range holds a request's last tile (key seq_b - 1 = cache_loc[b]) stores k = norm(k)/k_scale and v = v/v_scale as e4m3 while block 0's page indices are in flight (warps 0-3), `__threadfence()`, and the barrier after the index wait orders it before every ring cp.async. Tile ranges are disjoint, so no other CTA reads that row.
- Review correction applied (review wins over spec): NO RoPE. Verified: reference.py `rope_theta = ROPE_THETA if layer == "sliding" else 0.0`, TASK.md "no RoPE" on full layers, master's `_pre_attention_kernel` is launched with `ROPE=bool(theta)` = False. So positions/inv_freq are dropped from the v3 signature (answers the review's open question), and kernel_fn takes v3 only when `not theta` (else v2/master path). The spec's sin/cos tolerance caveat is moot; new-row check is still rel L2 < 5e-3.
- Dispatch in kernel_fn: `splitkv and _EXT3 is not None and not theta` -> `_full_decode_sk3` (one launch, no q tensor, no `_pre_attention_kernel`). Everything else unchanged byte for byte.
- Self-check (`_sk3_validate`, in-process, only after stage 2 validated): image check (cap 9.0, binaryVersion 90, CTAs/SM), canary B=2, 8 ragged cases (B=2/16/32/64, kv_scale 1 and 0.25, grids 3/7/device). Per case: host ownership check (cache_loc == page_table[b, seq-1]; exactly one CTA owns each last tile), master `_pre_attention_kernel` + v2 on cloned caches as reference, v3 on the originals; asserts v3 out vs FP32 ref < 2e-2 and vs v2 out < 5e-3, new k/v rows vs master rel L2 < 5e-3 and finite, all other cache bytes equal. Each launch printed before, synced and printed after. Then graph capture + 2 NaN-poisoned replays through kernel_fn, and a ticket-sum == 0 check.
- PATH CHOSEN lines: success `full decode streamk fused-preattn v3`; any v3 failure `full decode streamk v2 (fallback: <reason>)` (v2 keeps running); v2 failures keep the stage-2 messages.
- Added `_sk3_diagnostics` (eager, cold L2, B=16/32 x 80000): pre-attn + v2 vs fused v3 and v3 - v2 kernel time, as the review asked.
- Not verified: no nvcc/GPU on this node; the CUDA was only read-checked and py_compiled.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn v3 path only allocates workspaces from shapes and launches one kernel; no sync/.item()/value-dependent control flow (syncs are in prepare-time self-check only; replay check captures kernel_fn).
- [x] Nothing derived from `data` is cached across calls: tickets are device-allocated, zeroed once, and every launch resets them; the cache rows are outputs, not cached inputs.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] KV cache FP8 e4m3 (k/v rows stored e4m3), queries rounded to FP8 E4M3 (not lower), activations bf16.
- [x] Attention sees every key: stream-K covers all tiles of every request including the new token; no skipping.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network (grep of diff: no subprocess/fork/multiprocessing; load_inline is in-process).
- [x] All work inside the timed call; no output caching or replay (self-check runs at import on synthetic data only).
- [x] No triton.autotune; first-line marker `# okc: problem=muse_attention_decode` kept.

okc_lint output (`python3 -I okc_lint.py submission.py --problem muse_attention_decode`): `OK with warnings: submission.py`; 16 `warn host-sync` lines, all `torch.cuda.synchronize()` / `.item()` in prepare-time self-check code, "outside kernel_fn's call graph; fine in prepare()". No errors. `py_compile` passed; `__pycache__` removed.

## Risks / what could fail
medium
Small absolute win (~2-3 us on 120-250 us shapes) but it completes the one-launch full-decode path; risk is the new-key ownership and Q recomputation correctness, covered by the self-check. If stage 2 fails its gate, this branch still builds on its code and the v1 fallback.

## Result (after the run)
- job `825baa03798f431a` (check), status done, passed True, tests 16/16
- speedup 1.4283x, 95% CI [1.4069, 1.4497], score 2636.9450599469105 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.patch`; branch `hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn`
