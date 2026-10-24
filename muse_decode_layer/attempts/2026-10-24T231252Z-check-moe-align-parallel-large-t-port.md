# muse_decode_layer · 2026-10-24T231252Z · check · moe-align-parallel-large-t-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T184511Z-moe-align-parallel-large-t-port.md` (branch `hyp/muse_decode_layer/moe-align-parallel-large-t-port`).

- **Job:** `b42e33406f2d4978` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `18c7ee4`

## Goal
Port muse_moe_w4a16 align-parallel-per-expert-verify (patch muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check 45eacf4ca2e849d2: SGLang align 10.6 / 11.4 us -> _align_parallel_kernel 4.0 / 6.9 us at bs=128 / 256, verify_16 / verify_32 +3-4% there) to decode _experts for num_tokens > 32 only. Copy verbatim from the patch: ALIGN_PAR_CHUNK, ALIGN_PAR_WARPS, ALIGN_PAR_BLK, _align_parallel_kernel and _align_parallel. In _experts, when num_tokens > 32: replace the two master statements `local_ids = bank["expert_map"][topk_ids].contiguous()` and `sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` with `local_ids, (sorted_ids, block_experts, n_padded) = _align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)` (decode hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are 0..31: local_start 0, count 32; the kernel writes local_ids itself, -1 for remote slots, exactly the expert_map semantics). block_size_m stays master pick with the cap (16 at T=128 and T=256). Everything downstream (cache allocation, both Marlin calls, silu_and_mul with expert_ids=local_ids, fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids) is unchanged. prepare(): compile one variant per block_size_m it can see above 32 tokens (8, 16) on dummy ids, then a self-check on synthetic skewed routings at T in 33, 40, 57, 128, 256, 257: per local expert, the set of routes in sorted_ids and its block count must equal moe_align_block_size(expert_map[ids], bsm, N_EXPERTS) and local_ids must be equal; ignore capacity and padding-count differences (LESSONS: never gate on a check that counts the capacity line, it always differs). On a real mismatch set state["align_par_ok"] = False and keep master; print "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". T <= 32 untouched.

## What changed, and where
Worktree muse_decode_layer/submission.py only (+188/-1). Added ALIGN_PAR_* constants, `_align_parallel_kernel` and `_align_parallel` copied verbatim from muse_moe_w4a16 master (docstring of the wrapper reworded only). `_experts`: when `align is None`, num_tokens >= 33 and `state["align_par_ok"]`, the expert_map gather and moe_align_block_size are replaced by `_align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)`; otherwise master statements unchanged. T<=32 never reaches the new branch (dense chain path and its master fallback byte-identical). `_align_par_setup` (called in prepare() after `_dl_setup`) warms every block_size_m reachable above 32 tokens (computed from `_block_size_m` over T 33..1024, plus 8 and 16; in practice 16) for int32 and int64 id dtypes, then self-checks T in 33, 40, 57, 128, 256, 257 on random / skewed / remote-heavy routings: local_ids equality, per-expert route set and block count vs moe_align_block_size(expert_map[ids], bsm, 256); capacity/padding ignored. Mismatch or exception -> align_par_ok False (master kept). Prints "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". Deviation: none in the kernel; self-check is structural only (no Marlin y comparison, per review open question left as optional).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: new kernel_fn path only launches a Triton kernel and torch.empty allocs on shapes; host syncs/.cpu() are only in prepare()-time setup (checked in diff).
- [x] Nothing derived from `data` is cached across calls (state gets only the prepare-time bool align_par_ok).
- [x] Weights: n/a: no weight changes in the diff.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8: n/a: integer routing work only, no dtype changes.
- [x] Attention sees every key: n/a: attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (diff checked).
- [x] All work happens inside the timed call; no output caching or replay (align computed per call from topk_ids).

okc_lint: "OK with warnings: muse_decode_layer/submission.py" (only host-sync warnings, all in prepare()-time code, noted as fine). py_compile clean, __pycache__ removed.

## Risks / what could fail
low
A proven per-shape win in moe (measured on the align kernel row), ported unchanged; the headline effect (~0.3%) is under resolution, so judge on the align rows at verify_16/32. OVERLAP: moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 with its own route-list kernel when its path is ON; if it lands ON this is moot, if it falls back to Marlin this is the cheap T>32 fallback. Run it only after that harvest.

## Result (after the run)
- job `b42e33406f2d4978` (check), status done, passed True, tests 13/13
- speedup 1.2433x, 95% CI [1.2247, 1.2620], score 18193.699172659657 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/moe-align-parallel-large-t-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/moe-align-parallel-large-t-port.patch`; branch `hyp/muse_decode_layer/moe-align-parallel-large-t-port`
