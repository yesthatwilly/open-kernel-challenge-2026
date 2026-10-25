# muse_attention_decode · 2026-10-25T004113Z · check · sliding-verify-fa3-head-pairing

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T003735Z-sliding-verify-fa3-head-pairing.md` (branch `hyp/muse_attention_decode/sliding-verify-fa3-head-pairing`).

- **Job:** `031731b5db294a92` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `caf807d`

## Goal
Config-only change in kernel_fn: extend master head-pairing predicate to sliding verify. Today paired = window == 0 and hkv % 2 == 0 and batch > 1 and not splitkv, so sliding verify_16/32 run the unpaired FA3 path (two 64-byte KV rows gathered per token, then sglang rms_norm_gate). New predicate: paired = hkv % 2 == 0 and batch > 1 and not splitkv (drop window == 0). Sliding decode never reaches this code (it returns through _sliding_forward), so only sliding verify at batch 16/32 changes. On the paired branch everything is master code as it already runs for full verify_16/32: _pre_attention_kernel with PAIRED=True writes the zero-padded [tokens, 16, 128] query; flash_attn_with_kvcache on the [pages, 1, 1, 128] paired cache views with scheduler_metadata=None (sliding already passes None, the judge gives none on sliding), causal=True, the SAME window_size=(window - 1, 0) as now (the window is per query/key position, independent of the head layout), the same descale expanded to attn_heads = 1; then _paired_norm_gate_kernel instead of rms_norm_gate. Pairing is exact: each query sits in its own head half of the 128-wide query, so the zero half adds exactly 0 to each logit, and FA3 maps query head h to super-head h // 16, keeping the grouping. Full layers, sliding verify_1 and all decode shapes unchanged.

## What changed, and where
In the worktree's muse_attention_decode/submission.py, `kernel_fn` now sets `paired = hkv % 2 == 0 and batch > 1 and not splitkv`. The `window == 0` term is gone, and a 2-line comment was added. That is the whole diff (+3/-1). There is no deviation from the spec. Sliding decode still returns through `_sliding_forward` before reaching this line. `_splitkv_gate` covers full decode only. The paired branch is master code. The sliding `window_size=(window - 1, 0)` and `scheduler_metadata=None` handling were not touched.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the diff changes one shape/config predicate only. No syncs, no input-value control flow.
- [x] Nothing derived from `data` is cached across calls: the diff adds no state.
- [x] Weights: n/a, no weight change.
- [x] KV cache stays FP8, queries stay FP8 (E4M3): the pre-attention kernel and `FP8` dtype are unchanged.
- [x] Attention sees every key: the window is the same per-position `(window - 1, 0)`, and the zero query half adds exactly 0 to each logit.
- [x] No timing patches, harness access, subprocesses or network: none added.
- [x] All work inside the timed call, no output caching or replay.
- [x] No child process, no triton.autotune.

okc_lint output: `CLEAN: muse_attention_decode/submission.py`. py_compile passed and __pycache__ was removed.

## Risks / what could fail
medium
Risk: sliding verify covers only ~2055 keys per request, so FA3 there is more latency/launch bound than the 80K full shapes where pairing paid; the doubled QK math on the zero-padded query and a different FA3 split decision (one kv head instead of two) could cost as much as the gather saves. Cheap, config-only, attributable to one predicate.

## Result (after the run)
- job `031731b5db294a92` (check), status done, passed True, tests 16/16
- speedup 1.3475x, 95% CI [1.3273, 1.3677], score 2795.0957297962145 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-verify-fa3-head-pairing.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-verify-fa3-head-pairing.patch`; branch `hyp/muse_attention_decode/sliding-verify-fa3-head-pairing`
