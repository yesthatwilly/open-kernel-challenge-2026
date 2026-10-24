# muse_attention_decode · 2026-10-24T184520Z · check · paired-norm-gate-token-tiled

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-paired-norm-gate-token-tiled.md` (branch `hyp/muse_attention_decode/paired-norm-gate-token-tiled`).

- **Job:** `5e01430a8dfe4977` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `0a45063`

## Goal
Regrid master _paired_norm_gate_kernel (the paired full-layer epilogue, which now runs only at full verify_16/32) from one 1-warp program per (token, query head) to one program per token: grid (tokens,), num_warps 4. Program t loads the [16, 64] tile of head h's own half (h // G) % 2 of FA3's [tokens, 16, 128] output (column offset h * attn_stride_h + half * D + j, rows h = arange(16)), computes the scaleless RMSNorm per row (sum over axis 1, fp32, eps inside the sqrt of the mean of 64 squares), times sigmoid of the [16, 64] gate tile read from qkv, and stores bf16 into out[t, h * 64 + j]. Arithmetic and rounding are identical; only the decomposition changes. kernel_fn launches it with grid (tokens,) instead of (tokens, hq).

## What changed, and where
Only `_paired_norm_gate_kernel` and its launch in the worktree's muse_attention_decode/submission.py changed. The grid is now (tokens,) with num_warps=4. Each program loads a [HQ=16, 64] tile with a 2-D pointer block (row offset h * attn_stride_h + ((h // G) % 2) * D, per the review's correctness note). The RMSNorm is tl.sum over axis 1 in fp32, and gate and store are 2-D. Arithmetic and rounding are unchanged. No deviation. The kernel now requires HQ to be a power of two (16 on this path).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the diff adds no host syncs, `.item()` or prints.
- [x] Nothing derived from `data` is cached across calls: the diff adds no state.
- [x] Weights: n/a, not touched.
- [x] Activations stay BF16/fp32 as before; the output is bf16, KV cache and queries are unchanged.
- [x] Attention sees every key: n/a, only the epilogue changed.
- [x] No timing patches, harness access, module edits or reference import in the diff.
- [x] All work stays inside the timed kernel_fn; no output caching or replay.
- [x] No triton.autotune; the `# okc: problem=muse_attention_decode` first line is untouched.

Lint output: `CLEAN: muse_attention_decode/submission.py`. py_compile passed and __pycache__ was removed.

## Risks / what could fail
low
full-verify-cuda-streamk-splitkv (in the judge now) replaces the paired FA3 path and this epilogue at full verify_16/32; if it lands, this is moot and should be held. Small by construction: judge on the kernel row.

## Result (after the run)
- job `5e01430a8dfe4977` (check), status done, passed True, tests 16/16
- speedup 1.3911x, 95% CI [1.3702, 1.4149], score 2707.4133087460073 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/paired-norm-gate-token-tiled.result.json`, `muse_attention_decode/hypotheses/done/neutral/paired-norm-gate-token-tiled.patch`; branch `hyp/muse_attention_decode/paired-norm-gate-token-tiled`
