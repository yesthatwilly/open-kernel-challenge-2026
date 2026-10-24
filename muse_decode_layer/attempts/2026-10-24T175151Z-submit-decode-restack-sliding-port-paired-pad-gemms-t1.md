# muse_decode_layer · 2026-10-24T175151Z · submit · decode-restack-sliding-port-paired-pad-gemms-t1

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T174421Z-decode-restack-sliding-port-paired-pad-gemms-t1.md` (branch `hyp/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1`).

- **Job:** `4739dd4590524487` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `62dae0d`

## Goal
Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand (the code paths are disjoint). (1) sliding-fused-triton-decode1-port as is: sliding decode_1 +3.4-4.3%, _sliding_fused_kernel 13.7 us replaces FA3 main + combine + glue. (2) paired-heads-pad-unpad-triton as is: full decode_16/32 and verify_16 +1-3%, elementwise glue launches gone. (3) bf16-shape-gated-triton-gemms-small-t, but gate its Triton split-K GEMMs to tokens == 1 ONLY. At T=8/16 they lost 2-4.5% despite faster kernel sums, so T>1 keeps master's GEMM path exactly. Known conflicts: (1) and (2) both insert helpers at the same spot and edit the kernel_fn attention block around qk_norm; (3) also touches the module top, the helper block and the qkv = _gemm(...) line where (1) hooks in. Keep each piece's own self-check. Nothing else changes.

## What changed, and where
Worktree file: `.worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1/muse_decode_layer/submission.py` (base blob cc466fce; uncommitted). Diff stat: 1 file, +756 / -72.

How it was merged: `git merge-file` of port + paired over cc466fce conflicted in two hunks (the helper insertion point and the whole re-indented attention block), as the review predicted. Resolved by hand: the file is the port branch blob (e33a32aa) verbatim, then the paired branch's two Triton kernels (`_pair_pad_q_kernel`, `_paired_norm_gate_kernel`) inserted after `_sliding_forward`, then the GEMM branch's helper block (`_bf16_splitk_kernel` through `_prepare_small_t_gemms`) inserted before `prepare`, then the kernel_fn edits of the paired and GEMM branches re-applied.

- Port (1): unchanged from its branch (docstring bullet, `_INV_FREQ`/`_counters`/`_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue`, `_sliding_fused_kernel`, `_sliding_forward`, the `if cfg["window"] and batch == 1 and length == 1: o = _sliding_forward(...) else:` re-indent).
- Paired (2): both kernels byte-identical to the branch. Its two kernel_fn edits (the `_pair_pad_q_kernel` launch with its `assert qh.stride(2) == 1` replacing torch.zeros + slice copies, and the `if paired: _paired_norm_gate_kernel ... else: o = o_norm_gate(...)` epilogue) sit inside the port's `else:` branch, one indent deeper. `diff -w` of the merged kernel_fn against the paired branch's kernel_fn shows only the port and GEMM hunks.
- GEMMs (3): kernels, shape guards, `_fallback` and its once-per-shape log kept unchanged. Changes from the branch: `SMALL_T_TRITON_GEMM = 1` and `small_t = tokens == SMALL_T_TRITON_GEMM` (was `1 <= tokens <= 16`); the unpooler-slice route is now gated `small_t and _unpool_shape_ok(...)` too (in the branch it ran at every T <= 8, i.e. also at verify_1 T=8, so leaving it would have broken "T>1 takes master's path"). The T>1 unpooler call is `_gemm(expert_output, unpool_w)` with `unpool_w = dense["unpooler"][:UNPOOL_SLICE]`, the same call as master's. Warmup trimmed to m=1 (`for m in (SMALL_T_TRITON_GEMM,)`), as the review allowed; the workspaces keep their 16-row size, since the kernels are verbatim and use WS_M=16 / BLOCK_M=16. Comments and docstring bullet updated to T == 1. `_prepare_small_t_gemms(state)` is called in prepare after `del freqs`, as in the branch.
- `o` is defined on all three routes: sliding batch-1 decode (`_sliding_forward`), paired (`torch.empty` + `_paired_norm_gate_kernel`), plain (`o_norm_gate`). The o / pooler / unpooler GEMM selection follows after the attention block at the outer indent, so it sees `o` from any route.
- T>1: `small_t` is False, so every GEMM is `_gemm(...)` with master's arguments. The only T>1 differences from master are the paired-path kernels (full layer, batch > 1), as intended.
- Self-checks kept: paired `assert qh.stride(2) == 1`, GEMM `_*_shape_ok` guards with `_fallback` logging, port's config-only gate and `_num_splits`.

Deviations: none beyond the specified T == 1 gate, the unpooler gate that it requires, and the warmup trim.

Uncertainties: none tested (no GPU on this node). The port's `_counters` allocates its zeroed ticket buffer lazily on the first eager call (unchanged from the port branch, which passed); it is keyed on (batch, hkv, device), never on data values.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1 diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's call graph (the three synchronize calls are in prepare / `_prepare_small_t_gemms`; `_fallback` prints shapes only, never values); routing on `tokens`, `batch`, `q_len`, `window`, `rope_theta` and tensor shapes/strides only.
- [x] Nothing derived from `data` is cached across calls: `_INV_FREQ` is keyed on (theta, d, device) from config, `_counters` on (batch, hkv, device) and holds only self-reset tickets, the GEMM split-K workspaces are fully rewritten for rows < M every launch, and `_FALLBACK_SEEN` stores shapes only.
- [x] Weights: no new repack; the Triton GEMMs read the original BF16 `dense` weights in place (the unpooler slice by stride); NVFP4 experts untouched.
- [x] Activations BF16 where SGLang keeps BF16 (GEMMs bf16 in / fp32 accumulate / bf16 out; paired epilogue stores bf16); KV cache FP8; queries rounded to FP8 E4M3 (paired pad kernel, `tl.float8e4nv`), and the port's kernel as in its branch.
- [x] Attention sees every key: FA3 full context / whole window unchanged; the port's sliding kernel attends the full 2048-token window in 16 splits (unchanged from its branch).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: imports unchanged from master.
- [x] All work happens inside the timed call; no output caching or replay: prepare only allocates workspaces/counters and warms kernels on synthetic random inputs.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept; no `triton.autotune`.

okc_lint output:

```
warn  host-sync    L881: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L885: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L928: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1/muse_decode_layer/submission.py
```

`python3 -I -m py_compile`: OK; `__pycache__` removed.

## Risks / what could fail
low
Restack of measured pieces; expected headline about 1.18 vs 1.165 (harvest analysis). Run as a scored experiment.

## Result (after the run)
- job `4739dd4590524487` (submit), status done, passed True, tests 13/13
- speedup 1.1937x, 95% CI [1.1758, 1.2116], score 18950.355764161493 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_decode_layer/hypotheses/done/neutral/decode-restack-sliding-port-paired-pad-gemms-t1.result.json`, `muse_decode_layer/hypotheses/done/neutral/decode-restack-sliding-port-paired-pad-gemms-t1.patch`; branch `hyp/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1`
