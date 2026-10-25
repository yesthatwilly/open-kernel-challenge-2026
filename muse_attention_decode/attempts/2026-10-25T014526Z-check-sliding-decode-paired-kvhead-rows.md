# muse_attention_decode · 2026-10-25T014526Z · check · sliding-decode-paired-kvhead-rows

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T003735Z-sliding-decode-paired-kvhead-rows.md` (branch `hyp/muse_attention_decode/sliding-decode-paired-kvhead-rows`).

- **Job:** `48ffbe5690cc4ab4` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `990710b`

## Goal
One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding decode at batch > 1 only (q_len 1; config-only predicate pair = length == 1 and batch > 1 and HKV == 2; batch 1 and the S == 1 path keep today code). Today program (b, g, s) gathers kv head g alone: a 64-byte K row and a 64-byte V row per token (kvoff = (row * HKV + g) * D + d), and its M = 16 query tile holds only GROUP = 8 real rows (8 padding rows). LESSONS: page-size-1 row gather is the limit of these kernels, and FA3 head pairing (one 128-byte row per token for both kv heads) gained 14-24% on full batch-16/32 shapes. New constexpr PAIR: grid (B, 1, S) with S = 16 unchanged; the program serves BOTH kv heads: rows r = 0..15 are query heads 0..15 (head = r, kv head r // 8, so the M = 16 tile has no padding rows at L = 1); the query is laid out as [16, 2*D] with each row values in its own kv head half and zeros in the other half (same trick as master paired FA3 query, exact: the zero half adds 0 to every logit); each old-key tile loads one contiguous 128-byte K row and one 128-byte V row per token: offset row * HKV * D + arange(2*D); qk = dot(q[16,128], K^T[128,BLOCK_N]); PV gives acc[16,128] and each row keeps only its own half (select by r // 8 when writing the partial and in the epilogue, so ws_acc stays [16, D] per partial). The last split builds both kv heads new k/v (norm + rope + E4M3, stored at cache_loc for g = 0 and 1 exactly as today) and attends them from registers in the same 128-wide form. Ticket counters indexed by b (one per request instead of b*HKV + g; _counters(batch, 1, device)). Workspace sized (B * S, 16, D) and (B * S, 2, 16). _gate_epilogue unchanged per row (head = r). BLOCK_N 64, num_warps 4, num_stages 2 unchanged. Everything else (full layers, sliding verify, sliding decode_1) byte for byte.

## What changed, and where
- Worktree `muse_attention_decode/submission.py`, +172 lines, nothing removed. `_sliding_fused_kernel` and all other paths are untouched (byte-identical).
- Deviation: a new kernel `_sliding_paired_kernel` instead of a PAIR constexpr inside `_sliding_fused_kernel`, so the unpaired kernel's code cannot change. Gate in `_sliding_forward` (config only): `length == 1 and batch > 1 and hkv == 2 and hq == 16 and d == 64`. Grid is (B, 16) with S = 16, M = 16, BLOCK_N 64, warps 4, stages 2.
- Rows r = 0..15 are query heads (kv head r // 8, no padding). The query is loaded as its 64 values twice into 128 columns, normed and roped, then zeroed outside its own half.
- Review item 1: `_norm_rope_halves` takes the RMSNorm per 64-wide half, for the query and for the new k. RoPE freq indexes `(d % 64) // 2`. The new k/v is stored contiguously at `loc * 128 + d2`, which is `(loc * 2 + g) * 64 + d` for g = 0, 1.
- Review item 2: `_own_half` reshapes acc to [16, 2, 64] and takes a masked sum over axis 1. The partial write and the epilogue both use the result, so ws_acc stays [B*S, 16, 64]. Whether Triton lowers this without a large smem shuffle is unchecked (no GPU here).
- Review item 3: counters are `_counters(batch, 1, device)`, one per request. Each of the 16 programs of request b draws one ticket, the last one combines and resets it to 0, so the zero-reset invariant holds across graph replay. The (B, 1) buffer is a different dict key from the old (B, 2) buffer, so they cannot alias.
- Review item 4: the submission has no `prepare()`. The kernel's n_regs and n_spills are printed once, on the first (eager) launch, from the launch handle, with no sync. Check the log_tail for `[paired] ...`.
- Weakest point: untested on GPU. The query norm over duplicated columns is mathematically equal to the 64-wide norm, but the sum order may differ in the last ulp.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the only host work is the one-time print of n_regs and n_spills (host metadata); the branch is on config only; workspaces are allocated from shapes.
- [x] Nothing derived from `data` is cached: counters are zero-by-invariant and carry no data; `_inv_freq` is unchanged.
- [x] n/a: weights untouched, no repacking.
- [x] KV cache stays FP8 E4M3; the query is bf16 as in master.
- [x] Every key is attended: old window keys in all 16 splits, plus the new key from registers.
- [x] No timing patches, subprocesses or network: the diff only adds a Triton kernel and a gated launch.
- [x] All work is inside the timed call; no output caching or replay.

okc_lint: `CLEAN: muse_attention_decode/submission.py`. `python3 -I -m py_compile` passed and `__pycache__` removed.

## Risks / what could fail
medium
Risk: per-program work doubles while program count halves (B=16: 256 programs of 4 warps on 132 SMs, under one wave per 2 CTAs/SM); if occupancy hurts decode_16, an ab-tune with S=32 at batch 16 is the companion. The 128-wide dot doubles MMA work, but these shapes are gather/latency bound (the fp8-qk mma companion on the CUDA path was flat).

## Result (after the run)
- job `48ffbe5690cc4ab4` (check), status done, passed True, tests 16/16
- speedup 1.2865x, 95% CI [1.2672, 1.3058], score 2927.5755768145236 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-decode-paired-kvhead-rows.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-decode-paired-kvhead-rows.patch`; branch `hyp/muse_attention_decode/sliding-decode-paired-kvhead-rows`
