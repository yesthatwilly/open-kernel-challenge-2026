# muse_decode_layer · 2026-10-25T041933Z · submit · full-decode-splitkv-restack-on-pdl

Hypothesis: `muse_decode_layer/hypotheses/done/neutral/2026-10-24T224525Z-full-decode-splitkv-restack-on-pdl.md` (branch `hyp/muse_decode_layer/full-decode-splitkv-restack-on-pdl`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `f6d2dbd`

## Goal
Promote check-proven gain full-decode-splitkv-restack-on-pdl (1.2680x [1.2490,1.2871], check job c243a29f26dc4725) to the board.

Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2 (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl, f1f2f35), the way moe stacked shape-disjoint wins by 3-way merge. Take branch hyp/muse_decode_layer/full-decode-cuda-splitkv-port (patch done/neutral/full-decode-cuda-splitkv-port.patch) and 3-way merge its additions unchanged: its load_inline CUDA module (separate module name from the dense-chain TU, so both compile at import: ~41 s each, still far under the 15 min run), its gate (_splitkv_gate: full layer, q_len 1, batch > 1, config only), the splitkv attention + combine/norm-gate kernels that replace the paired pad copies, FA3 flash_fwd + FlashAttnFwdCombine, unpad copies and FusedRMSNormGate on those shapes, and its prepare() self-check and fallback. Resolve every conflict in favour of master outside the splitkv branch of kernel_fn: the dense-chain P/R path after the o GEMM, attn_boundary router_dtype logic and _experts(align=...) stay exactly as master. Print both modules ON/OFF lines.

## What changed, and where
Hand merge (2026-10-24) of done/neutral/full-decode-cuda-splitkv-port.patch onto decode master 6ee651f, worktree muse_decode_layer/submission.py only. `git apply -3` had given 7 interleaved conflicts because the split-KV and dense-chain CUDA modules share scaffolding lines (#includes, namespace, build-function shape). They were not merged as text. The parent's split-KV block went in whole, as one contiguous section:
- **Placement:** the parent's whole split-KV section (its comment, `_SPLITKV_CPP`/`_SPLITKV_CUDA` source unchanged, `_build_splitkv` with load_inline name `muse_full_decode_splitkv_v1`, `_EXT`, `_sm_count`, `_splitkv_gate`, the launch helpers, the synthetic data, the reference, the self-check and `_splitkv_prepare`) is inserted as one block after `_sliding_forward`, before `prepare()`. It is a separate translation unit from master's `muse_decode_dense_chain_v1` (`_DL_CPP`/`_DL_CUDA`, untouched), and the two modules share no Python names.
- **LOG2E:** the parent's `LOG2E` definition was dropped. Master already defines the same constant in the sliding section, and it is resolved at call time.
- **prepare():** calls `_splitkv_prepare(device)` after master's `_dl_setup` and `_align_par_setup`, before the final synchronize.
- **kernel_fn:** inside master's non-sliding `else`, after FusedQKNorm and store_cache: `splitkv = _splitkv_gate(cfg)` (`_EXT` built and self-checked, window 0, q_len 1, 1 < batch <= 64, tp8 16/2 heads, d 64, page 1; config only). It runs kernels A and B, whose output feeds the unchanged o GEMM. The `else` branch is master's paired FA3, unpad and FusedRMSNormGate code, indented one level, with `and not splitkv` added to `paired` as the parent had it. The sliding fused path, the dense-chain P/R path, attn_boundary/router_dtype, `_experts(align=...)`, the side stream and the parallel align are byte-identical to master (`git diff -w` shows only the docstring bullet and the `paired` line as changed master lines).
- **Deviations, per LESSONS (none change kernel code):**
  - `_full_decode_splitkv` is split into `_splitkv_attn` (kernel A) and `_splitkv_combine` (kernel B). The self-check launches them separately, with `torch.cuda.synchronize()` and a print after each, so a fault is attributed to its kernel. kernel_fn's path has no sync; okc_lint rejected an in-function `check` flag.
  - The build adds `-lineinfo`.
  - Every [splitkv] print is flushed.
  - Explicit ON/OFF lines: `[splitkv] ON: self-check passed`, `[splitkv] OFF: SELF-CHECK FAILED`, `[splitkv] OFF: module not built`.
  - There are no child processes; the self-check runs in-process in prepare() on synthetic data only.
- **Harvest:** read the `[splitkv]` and `[small-t-dense-chain]` lines. Two load_inline compiles run at import, about 41 s each.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn's split-KV path has no .item()/.cpu()/synchronize()/prints. Its workspaces are torch.empty sized from page_table.shape and config, and the grid comes from the SM count (a device property). The syncs are only in `_splitkv_self_check`/`_splitkv_prepare`, called from prepare() (lint warns L2202/L2206/L2284, outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: `_SMS` caches the device SM count only, and `_SPLITKV_GRID` is set only during the self-check and reset in its finally block.
- [x] Weights: n/a: no weight is touched by this change. Master's prepare-time repacks are unchanged.
- [x] Activations: q is cast to FP8 E4M3 (as master's FA3 path does), the KV cache stays FP8, and the output is bf16 into the o GEMM. Partials are fp32.
- [x] Attention sees every key: kernel A walks every 1024-token chunk up to cache_seqlens[b], masking only positions >= the length. There is no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. load_inline compiles in-process at import (the same mechanism as master's dense-chain module). A grep for subprocess/Popen/os.system finds nothing.
- [x] All work happens inside the timed call: the attention and combine run per kernel_fn call, with no output caching or replay.
- [x] The first line keeps the marker `# okc: problem=muse_decode_layer`. There is no triton.autotune.

okc_lint output:
```
OK with warnings: .worktrees/muse_decode_layer/full-decode-splitkv-restack-on-pdl/muse_decode_layer/submission.py
30 x warn host-sync, all "(outside kernel_fn's call graph; fine in prepare())":
L968 L969 L970 L973 L977 L995 L1003 L1006 L1010 L1024 L1080 L1082 L1119 L1124 L1125 L1130 L1131 L1158
L1204 L1231 L1236 L1254 L1255 L1262 L1263 L1291 L1297 (master's align/dense-chain self-checks)
L2202 L2206 L2284 (split-KV self-check per-kernel syncs, _splitkv_prepare)
```
py_compile: OK. The __pycache__ was removed. Diff: 1 file, +692 / -54 (the 54 are the re-indented FA3 block plus the docstring line).

## Risks / what could fail
low

## Result (after the run)
(pending)
