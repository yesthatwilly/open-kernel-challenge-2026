# muse_decode_layer · 2026-10-25T043231Z · check · full-decode1-two-level-ticket-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T042121Z-full-decode1-two-level-ticket-port.md` (branch `hyp/muse_decode_layer/full-decode1-two-level-ticket-port`).

- **Job:** `5e44c7aedecd4cc1` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `6474188`

## Goal
Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's full-decode1-streamk-s3-two-level-ticket (branch hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket; it contains the stage-2 stream-K kernel) to decode_layer, gated to full layer, q_len 1, page 1, batch == 1 ONLY (config). batch > 1 keeps master's split-KV kernel A/B unchanged so this run attributes to full decode_1 alone. At batch 1: grid = SM count, 64-token tiles, two-level ticket (groups of G = ceil(sqrt(nseg)) CTAs merge into a group slot via gticket[g], the last group merges the group slots, applies v_scale + RMSNorm x sigmoid(gate) and stores bf16; both ticket levels reset to 0 for replay). It replaces, at full decode_1, the FA3 flash_attn_with_kvcache call + FlashAttnFwdCombine + the 3 vectorized elementwise pad/unpad copies + FusedRMSNormGate; q comes unpaired from the same pre-attention path master's batch>1 split-KV route consumes (extend that route's q preparation to batch 1). Self-check at batch 1 with 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count, two consecutive replays. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/full-decode1-two-level-ticket-port`, `muse_decode_layer/submission.py` only (+872/-1). Nothing could be run on this node (no GPU/torch).

What changed:
- New section before `prepare()`: the CUDA source of `splitkv_attn_sk_kernel` taken verbatim from muse_attention_decode master c3c1b01, module `muse_decode_sk_ticket` (name shared with sibling full-decode-streamk-ticket-port), built with `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored, nvcc extras `-O3 -lineinfo` (same as the landed attention build; no `-gencode` override, so no `-std=c++20` needed), arch flags printed. Built at import, like `_EXT`.
- Review fixes applied (review wins over spec): ONLY the `<4, true, true>` instantiation is built (`sk_pick()` returns it; `sk_attn`/`sk_attrs` TORCH_CHECK variant == 3; the diagnostic combine kernel, `sk_combine` and variants 0-2 are removed); the B=16 diagnostic and the [sk3-diag] sweep are dropped (compile/prepare budget); the gate is a separate `_sk_gate` (full, q_len 1, page 1, batch == 1, tp8 heads) ahead of the paired/FA3 else-branch, `_splitkv_gate` is untouched (decode_16/32 unchanged); the q/k/v FP8 casts and store_cache stay; the "elementwise 3 us" is not replaced (it is the casts).
- `kernel_fn`: `if _sk_gate(cfg): q_fp8 = q.view(...).to(FP8).contiguous(); o = _full_decode_sk(...)`, then `elif splitkv:` (master's, unchanged), else FA3. One launch replaces FA3 + FlashAttnFwdCombine + FusedRMSNormGate at full decode_1. Grid = SM count, workspaces from shapes/device only, no sync.
- `prepare()` -> `_sk_prepare(state, device)`, all in-process: attrs print (binaryVersion, regs, local, smem, CTAs/SM), cubin == 90 and arch-flag checks; tiny canary (63 keys); self-check at B=1 with 80000, 4097, 63, 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count (rel L2 < 2e-2 vs FP32 reference, finite); two graph replays (NaN-poisoned output) for 80000/kv1 and 4097/kv0.25; both ticket levels must sum to 0; every launch has flushed print before and synchronize + print after. Then a CUDA-graph, cold-L2 timing selector at B=1 x 80000: new vs incumbent (FA3 + combine + FusedRMSNormGate); the new path is kept only if faster; margin printed (flagged if < 2 us). Tickets are created in prepare (outside capture) and zeroed after the self-check regardless of outcome. Any failure/raise sets `_SK=None` and prints `PATH CHOSEN: full decode_1 FA3 + combine + norm-gate (fallback: <reason>)`; success prints `PATH CHOSEN: full decode_1 streamk two-level ticket`.
- No sm_90a-only PTX in this kernel (mma.sync, cp.async, cvt e4m3x2), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard was added; the 9.0a build is kept per the spec.

Deviations: the self-check uses this file's `_splitkv_synthetic` (FP8 q fed directly, real lengths 4097/63/1, no pre-attention kernel); the selector times graphs rather than eager.

Landing note: dense-chain-stage2-attn-side-pdl-t1-gate also takes full decode_1. Whichever lands second must be restacked by hand as Q(dl_qkv) -> two-level kernel with norm-gate epilogue -> dl_o prologue=False, then a fresh check. Expectation: 142.7 -> ~135-138 us, kill at 140.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path has no `.item()`/`.cpu()`/sync/prints; workspaces from shapes only; tickets pre-created in prepare; the self-check captures and replays it (checked in diff).
- [x] Nothing derived from `data` is cached across calls (only device-keyed ticket buffers, zeroed, input-free).
- [x] Weights: n/a: no weight change.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; query FP8 E4M3 (unchanged casts).
- [x] Attention sees every key: the kernel tiles all cache_seqlens keys, no skipping (self-check vs FP32 reference).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (lint PROCESS clean; os used only for TORCH_CUDA_ARCH_LIST).
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warnings in prepare()-side code, none in kernel_fn); `py_compile` OK, __pycache__ removed.

## Risks / what could fail
medium
Shares full decode_1 with running dense-chain-stage2-attn-side-pdl-t1-gate (that one replaces the Q/O GEMMs, this one the attention kernels): read the attention kernel rows, not the headline.

## Result (after the run)
- job `5e44c7aedecd4cc1` (check), status done, passed True, tests 13/13
- speedup 1.2746x, 95% CI [1.2555, 1.2937], score 17746.968294420974 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2855726646854595, "ci": [1.2662890747151776, 1.3048562546557412], "job": "7a9f5de5aecc4aba"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode1-two-level-ticket-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode1-two-level-ticket-port.patch`; branch `hyp/muse_decode_layer/full-decode1-two-level-ticket-port`
