# muse_decode_layer · 2026-10-31T071105Z · check · swapab-root

- **Job:** `476d3ff11ba34347` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `c4fcd162` (branch fix/swapab-root; master's scored blob b5f84e27 + the
  fix), 997,735 B, sha256 e028ce5c9351

## Goal
Judge confirmation of the root-cause fix for the dense swap-AB copy-check illegal address (scored cb182920). Swap-AB ON,
copy check ON. If it passes and beats the board 1.6989, scored submit and promote. Either way, a new scored entry
replaces the hazardous one.

## What changed, and where
- `_rt_cap(g)`: before each prepare-time graph capture, a one-kernel dummy graph is captured and destroyed. Its
  CUDAGraph::reset clears the capture stream's cached cuBLAS workspace, so each real capture allocates its own workspace
  in its own private pool. Used at the 4 capture sites: _rt_graph, _sk_graph_of, _dl_self_check, _dl_pdl_probe.
- Root cause (shown on the H100, campaigns/2026-10-31-swapab-root/RESEARCH.md): torch 2.14.1 shares one cuBLAS
  workspace per stream across captures, and destroying any graph frees it. The next empty_cache unmaps it, and a later
  replay of a live graph writes there (Warp MMU Fault in cuBLAS dot_kernel). Master faulted in 8 of 10 stressed
  prepares; the fix faulted in 0 of 10, over 586 checks.
- Diagnostics only: a per-T progress line in _dsa_copy_check, and the head and tail of the error in the DSA except
  handler.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn is byte-identical to master; the change is prepare-only (AST walk: _rt_cap is
      unreachable from kernel_fn).
- [x] Nothing derived from `data` is cached across calls. The only new persistent object is a constant 1-element zeros
      tensor.
- [x] Weights: unchanged; BF16 stays BF16, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16; KV FP8; queries at most FP8 E4M3. Unchanged.
- [x] Attention sees every key; no top-k, sparse, block-skip, or weight-based skipping. Unchanged.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. Public
      torch API only.
- [x] All work happens inside the timed call; no output caching or replay.
- File 997,735 B ≤ 999,000. Prepare on the H100: 122.3 s cold (master 127-134 s), within 300 s. Lint: OK.

## Risks / what could fail
- The H200 recurrence rate is about 1 in 11 per prepare, so one passing check is weak confirmation; the H100 A/B is the
  real evidence.
- The harness's own captures could hit the same class (outside our control).

## Result (after the run)
**PASSED, 13/13. 1.714x (95%: 1.688-1.739x), 13.20 ms.** That beats the board, 1.6989.
- No illegal access in the log.
- The judge log is head-truncated (110 lines), so the [dense-swapab] prepare lines are cut. The timed profile shows
  `_bf16_swapab_kernel` in 7 kernel rows, so swap-AB was chosen.
- Next: a scored submit of the same bytes (c4fcd162).
