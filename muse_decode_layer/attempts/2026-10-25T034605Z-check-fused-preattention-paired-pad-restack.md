# muse_decode_layer · 2026-10-25T034605Z · check · fused-preattention-paired-pad-restack

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T165619Z-fused-preattention-paired-pad-restack.md` (branch `hyp/muse_decode_layer/fused-preattention-paired-pad-restack`).

- **Job:** `5e9b5464d1444708` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `963bcf0`

## Goal
Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch, _pre_attention_kernel reading RoPE from state['cos_sin']) onto master once paired-heads-pad-unpad-triton has landed (build on master if it landed, else on its branch), and finish the port of muse_attention_decode's landed chain fused-preattention-triton + paired-heads-launch-cleanup part (1) 'pad at the source', which decode_layer never got. 3-way merge the patch's kernel, its prepare() warm-up and its kernel_fn call onto master, then add attention master's PAIRED and G constexprs to the kernel verbatim (muse_attention_decode/submission.py on master, _pre_attention_kernel lines 117-124: when PAIRED, query program h writes its 64 FP8 values into half (h // G) % 2 of a [tokens, Hq, 128] FP8 row and 64 FP8 zeros into the other half). In kernel_fn: one launch, grid (tokens, Hq + Hkv), num_warps 1, replaces state['qk_norm'](...), the k/v divide by the scales, k.to(FP8), v.to(FP8), store_cache(...) and either q.view(T,16,64).to(FP8) (unpaired) or the _pair_pad_q_kernel launch that paired-heads-pad-unpad-triton added (paired: window == 0 and batch > 1 and hkv % 2 == 0, unchanged predicate, passed as PAIRED). q buffer is torch.empty((tokens, hq, 2*d if paired else d), FP8). gate and v stay slices of qkv; FA3, _paired_norm_gate_kernel / o_norm_gate and everything after the o GEMM are byte-identical to the base. If sliding-fused-triton-decode1-port has landed, its sliding batch-1 decode branch keeps running before this code and is untouched; if full-decode-cuda-splitkv-port has landed, pass PAIRED=False on its gated shapes exactly as attention master does (paired = ... and not splitkv). Delete _pair_pad_q_kernel only if nothing else calls it.

## What changed, and where
Worktree base was clean master (8ac7a2f), because the restack's 3-way apply conflicted. Only `muse_decode_layer/submission.py` changed (+129/-36).

Merge resolution:
1. Hand-merged the parent paired-heads-pad-unpad-triton.patch onto current master. Master already carried the torch.zeros/slice-copy paired path (the parent's target text had moved), so I swapped those blocks for the patch's `_pair_pad_q_kernel` launch and its `_paired_norm_gate_kernel` un-pad+norm+gate launch. The un-paired branch keeps `state["o_norm_gate"]`. Both kernels are placed just before `_sliding_forward`, since master's `prepare()` now sits after the sliding and dense-chain code. The `prepare()` hunk had only context changes, so nothing was needed there.
2. Applied this hypothesis: `_pre_attention_kernel` (from fused-preattention-triton.patch) with attention master's `G` and `PAIRED` constexprs, taken verbatim. It replaces `_pair_pad_q_kernel`, which was deleted because nothing else calls it. In `kernel_fn`: one launch on grid (tokens, hq+hkv), num_warps 1. It replaces `qk_norm`, the k/v scale divides, both FP8 casts, `store_cache` and the pair pad. The q buffer is `torch.empty((tokens, hq, 2*d if paired else d), FP8)`. `paired = window == 0 and batch > 1 and hkv % 2 == 0` is unchanged and is passed as PAIRED. The FA3 call, the descale and `_paired_norm_gate_kernel` are untouched.
3. Removed the `qk_norm` entry from `prepare()`, plus the now-unused `store_cache` and `FusedQKNorm` imports. Updated the module docstring line.
4. Kept intact: the sliding batch-1 `_sliding_forward` branch (it runs before this code), the small-t dense chain with PDL, and the align-par/side-stream code. full-decode-cuda-splitkv-port has not landed on master, so there is no splitkv gate and PAIRED is not forced to False anywhere.

Deviations: the parent fused patch has no prepare() warm-up (only the qk_norm removal), so none was merged. `v` is no longer a slice in `kernel_fn`, since the kernel reads it from qkv; `gate` is still a slice of qkv. Nothing could be run on this node: no GPU, torch or Triton.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: `git diff` adds only `torch.empty` and Triton launches in `kernel_fn`, with no `.item()`, `.cpu()` or sync, and no control flow on values. The `paired` and `scaled` branches read config only.
- [x] Nothing derived from `data` is cached: the q buffer is allocated per call, and cos_sin is a prepare() table indexed by the positions input.
- [x] Weights: n/a, no weight change; only `qk_norm` was removed from state, and it held no weights.
- [x] Activations and KV: the cache stays FP8, divided by the scales before the cast as before. The query is FP8 E4M3 only, as on master (master also cast q to FP8).
- [x] Attention sees every key: FA3 call and window are unchanged, and the pad half is zero, as on master.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: the diff adds none.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune; the first-line marker is kept.

okc_lint: `OK with warnings` (0 errors). The warnings are pre-existing host-sync lines in prepare()/self-check code, outside kernel_fn.
py_compile: ok; `__pycache__` removed.

## Risks / what could fail
low
Low expected value (~+1% headline, near the 1.5% drift), so judge it per shape: the paired rows against paired-heads-pad-unpad-triton's rows, the unpaired batch-1 rows against the neutral parent's. It is the only attention-side landed piece not yet ported or queued for decode_layer (port audit 2026-10-24). Confounders: full-decode-cuda-splitkv-port replaces the paired path on full decode_16/32 if it lands first (then only verify_16/32 keep the paired pad), and sliding-fused-triton-decode1-port removes sliding decode_1 from this code path. Correctness: the k/v cache store moves into Triton, so the cache contents are checked by the judge's later reads; keep the parent's kv_scale handling (divide before the cast) and cache_loc int64 indexing.

## Result (after the run)
- job `5e9b5464d1444708` (check), status done, passed True, tests 13/13
- speedup 1.2745x, 95% CI [1.2553, 1.2936], score 17749.02653999309 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/fused-preattention-paired-pad-restack.result.json`, `muse_decode_layer/hypotheses/done/neutral/fused-preattention-paired-pad-restack.patch`; branch `hyp/muse_decode_layer/fused-preattention-paired-pad-restack`
