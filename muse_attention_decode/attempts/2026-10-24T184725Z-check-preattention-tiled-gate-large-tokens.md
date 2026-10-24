# muse_attention_decode · 2026-10-24T184725Z · check · preattention-tiled-gate-large-tokens

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-preattention-tiled-gate-large-tokens.md` (branch `hyp/muse_attention_decode/preattention-tiled-gate-large-tokens`).

- **Job:** `c2ebf6ed05824692` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `880b01b`

## Goal
Restack preattention-group-tiled-programs (neutral at 1.261x on an older master; its patch is muse_attention_decode/hypotheses/done/neutral/preattention-group-tiled-programs.patch) behind a token-count gate. Add that patch's group-tiled kernel as a NEW Triton kernel _pre_attention_tiled_kernel beside master's _pre_attention_kernel (do not modify the old one): grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the 8 query heads of group g as one [8, 64] tile plus key head g, applies RoPE when ROPE, writes FP8 q in the PAIRED or unpaired layout exactly as master (padded half zeroed when PAIRED), and stores the FP8 key and value rows divided by k_scale / v_scale at cache_loc[t]. In kernel_fn, launch the tiled kernel when tokens >= 64 (verify_16 and verify_32, sliding and full) and master's _pre_attention_kernel otherwise (decode at every batch and verify_1, where the tiled version was 0.1 us slower). Same arguments, same outputs, same rounding points.

## What changed, and where
In the worktree's muse_attention_decode/submission.py: added `_pre_attention_tiled_kernel` (the group-tiled kernel from the done/neutral patch, copied; same arithmetic and rounding points) after `_pre_attention_kernel`, which is untouched. kernel_fn now builds the shared argument tuple and launches the tiled kernel on grid (tokens, hkv), num_warps=4, when `tokens >= 64`, else the old kernel on (tokens, hq+hkv), num_warps=1. The gate is config-only (batch * q_len). No deviation from the spec; the reviewer's open questions (dropping the gate, a masked PAIRED store) were left alone as new ideas.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only a Triton kernel and a shape-based launch branch; no .item()/.cpu()/sync/allocation on input values.
- [x] Nothing derived from `data` is cached across calls: no new caches.
- [x] Weights: n/a: not touched.
- [x] Activations BF16 / KV cache FP8 / queries at most FP8 E4M3: q, k, v are stored as float8e4nv exactly as the old kernel.
- [x] Attention sees every key: n/a: attention untouched; only the pre-attention launch changed.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none in the diff.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune; no host syncs in kernel_fn.

okc_lint output: `CLEAN: muse_attention_decode/submission.py`. py_compile passed, __pycache__ removed.

## Risks / what could fail
low
The sliding verify gain is replaced if sliding-cuda-cluster-dsmem-onelaunch lands (it fuses pre-attention into its one launch). full-verify-cuda-streamk-splitkv keeps master's _pre_attention_kernel (PAIRED=False) in front of its A8 kernel, so the full verify_16/32 part survives that rewrite, and the tiled kernel stacks there cleanly. Judge on the _pre_attention_tiled_kernel row.

## Result (after the run)
- job `c2ebf6ed05824692` (check), status done, passed True, tests 16/16
- speedup 1.3863x, 95% CI [1.3655, 1.4071], score 2716.8174960997503 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/preattention-tiled-gate-large-tokens.result.json`, `muse_attention_decode/hypotheses/done/neutral/preattention-tiled-gate-large-tokens.patch`; branch `hyp/muse_attention_decode/preattention-tiled-gate-large-tokens`
