# muse_decode_layer · 2026-10-25T004030Z · check · full-verify-cuda-splitkv-stage2

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T165224Z-full-verify-cuda-splitkv-stage2.md` (branch `hyp/muse_decode_layer/full-verify-cuda-splitkv-stage2`).

- **Job:** `e9d5eeffddef4f70` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `bed733b`

## Goal
Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3 (153.3 us at verify_16, 296.3 us at verify_32, the same ~328/650 MB KV gather as decode plus 8x the query rows). Build on stage 1 code. Widen _splitkv_gate to q_len in (1, 8) (config only; batch 1, sliding and page 64 unchanged). Add a new kernel splitkv_attn_verify_kernel (keep stage 1 decode kernel untouched so decode rows are the control) with the same persistent 2xSM grid, 1024-token (b, chunk) work items, smem page-index prefetch and 4-stage cp.async ring of 128-byte K and V rows. Query: master FP8 q [B*8, 16, 64] (unpaired cast, as stage 1), loaded once per item into registers as f16 A fragments. Warp layout for 64 query rows per kv head (8 tokens x 8 heads): 8 warps = 2 kv heads x 4 row groups, warp w owns kv head g = w % 2 and query rows (w / 2) * 16 .. +16 (2 query tokens x 8 heads) over the WHOLE 64-token tile, so warps own disjoint rows and need no intra-CTA merge; per tile S[16x64] = Q K^T with mma.sync m16n8k16 (K converted fp8 e4m3 -> f16; convert each stage once into an f16 smem tile shared by the 4 warps of a head if smem allows, else per warp in registers), causal mask inside the block: query token i (0..7) of request b sees key t only if t < cache_seqlens[b] - 7 + i (and t < cache_seqlens[b]; masked tokens never dereferenced, zero-filled via cp.async src-size 0), online softmax in exp2 with -inf-safe m, P @ V with mma.sync, fp32 acc 16x64 per warp. Item end: fp32 partial acc [8 tokens, 16 heads, 64] and m, l [8,16] to a workspace sized B * max_chunks * 8 (from shapes). Combine: splitkv_combine_verify_kernel, grid (B*8, 16), one warp per (token, head): LSE merge over that request chunks, x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024] for the o GEMM. Extend the prepare() self-check with q_len 8 cases (same ragged lengths, kv scale 1 and 0.25, grid device and 3) against an FP32 causal reference. Nothing else changes.

## What changed, and where
Edited only `.worktrees/muse_decode_layer/full-verify-cuda-splitkv-stage2/muse_decode_layer/submission.py` (+498/-30). **Base is stale** (claim reported `stale_base: true`): the worktree sits on stage 1's commit f6cc145, which predates master's current submission.py (stack-align-par-sidestream-sliding1, 1.2521). As instructed, stage 2 is built on stage 1 and NOT rebased; a merge to master will need a restack.

What changed:
- `splitkv_attn_verify_kernel` (new, stage 1 decode kernel byte-identical): same persistent 2xSM grid, 1024-token (b, chunk) items, smem page-index prefetch, 4-stage cp.async.cg ring (plain cp.async, no cache hints). Q for the item (8 tokens x 16 x 64 e4m3) is cp.async'd into a double-buffered smem tile with the indices and read into f16 A fragments at item start. Warp w = kv head w%2, query rows (w/2)*16..+16 (2 tokens x 8 heads) over the whole 64-token tile, so no intra-CTA merge. Per 64-token tile the warp runs two 32-token halves (S is 16 fp32 regs, to ease register pressure vs the 128-reg cap): S = QK^T with mma.sync m16n8k16 (K e4m3 -> f16 per warp in registers), causal mask `pos < cache_seqlens - 7 + i`, online exp2 softmax with -inf-safe m per row, P@V with mma.sync, fp32 acc 16x64 per warp. Masked/out-of-range tokens are zero-filled via cp.async src-size 0 and never dereferenced. Item end: each warp stores its two rows' fp32 acc as float4 and m, l to the workspace `[B*max_chunks, 8, 16, 64]` / `[B*max_chunks, 2, 8, 16]`.
- `splitkv_combine_verify_kernel`: grid (B*8, 16), one warp per (token, head): LSE merge (-inf partials weigh 0), x v_scale, scaleless RMSNorm eps 1e-5, x sigmoid(gate) from qkv row b*8+i, bf16 into [T, 1024].
- `_splitkv_gate` admits q_len 8 once `_VERIFY_OK`; decode keeps q_len 1. `kernel_fn` dispatches q_len 1 to stage 1 and q_len 8 to `_full_verify_splitkv` (workspace from `page_table.shape[1]`, grid from SM count, no input values). Batch 1, sliding, page 64 unchanged.
- Self-check: the decode check is unchanged but now traced; a new verify check (lens 9000/1025/70/4097/1023, kv scale 1 and 0.25, device grid and 3 CTAs, plus a batch-33 case) against an FP32 causal reference (`_splitkv_reference` generalised over q_len). A verify-check failure disables only the verify path (`_VERIFY_OK=False`, master paired FA3 on verify, decode keeps stage 1); a decode-check failure still disables everything. Loud `PATH CHOSEN` lines per shape class.
- Build-brief items: `-lineinfo` added, module renamed `..._v2`, nvcc flags and TORCH_CUDA_ARCH_LIST printed (no wgmma/setmaxnreg/sm_90a-only PTX is used, so the arch list is left alone); every self-check launch prints before it and synchronizes plus prints after it (via a hook defined outside kernel_fn's call graph, so okc_lint sees no sync there); no spin waits exist (cp.async only), loops are bounded by B/n; no child process.
- Review fixes applied: the review is "proceed" with no required fix. Applied its guidance: print `cudaFuncGetAttributes` regs/localBytes/smem for the verify kernel to the log (open question 1). Not applied (stage 3 levers per the review): e4m3 m16n8k32 QK^T, larger verify chunk, f16 smem K tile. The review's sequencing advice (dispatch only after stage 1 harvests, on a check not a scored submit) is for the dispatcher; stage 1 has not been confirmed here.

Deviations: none from the spec except the 32-token sub-tile split inside the 64-token tile (same ring, same math) and Q fetched through smem alongside the indices rather than straight from global.

Uncertainties: register count / spills of the verify kernel under `__launch_bounds__(256,2)` (unmeasured, no GPU); the workspace is B*max_chunks*32 KB (about 100 MB at verify_32 with a wide agentic page table) allocated inside graph capture; the first launch cannot be exercised here.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/synchronize/tensor prints in the kernel_fn path (the only synchronize calls are in prepare() and the self-check hook); workspace and grid come from shapes and device properties; the gate reads config only (checked in `git diff`).
- [x] Nothing derived from `data` is cached across calls: workspaces are fresh per call, `_EXT`/`_VERIFY_OK` are set only by the synthetic-data self-check.
- [x] Weights: n/a: no weight is touched or stored by this change.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; query FP8 E4M3 at most (unpaired cast as master; K/V widened to f16 exactly, fp32 accumulate, bf16 output).
- [x] Attention sees every key: all keys of every request in 1024-token chunks, causal inside the verify block; no skipping by weight (fully masked tokens are exactly the causal mask).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (grep for subprocess/multiprocessing/fork/spawn: none; load_inline is the same build as stage 1).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune, no host syncs in kernel_fn; first line `# okc: problem=muse_decode_layer` kept.

okc_lint output:

```
warn  host-sync    L1102: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1348: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed; `__pycache__` removed from the worktree.

## Risks / what could fail
high
High: new causal-in-block masking and a 4x larger query tile; mma.sync compute (~41/82 GFLOP at verify_16/32) is no longer negligible against the gather, so if A exceeds ~150 us at verify_16 the fix is fp8 conversion once per stage in smem, not more splits. Run only after stage 1 lands or is judged on its replaced kernels. Not attempted in muse_attention_decode yet; a win here is portable back there.

## Result (after the run)
- job `e9d5eeffddef4f70` (check), status done, passed True, tests 13/13
- speedup 1.1559x, 95% CI [1.1386, 1.1732], score 19569.43768728252 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-cuda-splitkv-port.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/full-verify-cuda-splitkv-stage2.result.json`, `muse_decode_layer/hypotheses/done/regression/full-verify-cuda-splitkv-stage2.patch`; branch `hyp/muse_decode_layer/full-verify-cuda-splitkv-stage2`
