# muse_attention_decode · 2026-10-24T184519Z · check · sliding-decode-blockn128-single-tile

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-sliding-decode-blockn128-single-tile.md` (branch `hyp/muse_attention_decode/sliding-decode-blockn128-single-tile`).

- **Job:** `88c62334775e48e4` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `e5892c6`

## Goal
Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split covers ceil(2047 / 16) = 128 old keys rounded up to BLOCK_N, so each program now runs its old-key loop once, issuing the page-index load and the whole chunk's K and V loads in one step instead of two dependent iterations. num_warps stays 4, num_stages 2, the new-key block, the masks, the ticket combine and verify (length 8, which keeps BLOCK_N 64) are unchanged.

## What changed, and where
In `_sliding_forward` of muse_attention_decode/submission.py, the kernel launch now passes `BLOCK_N=128 if length == 1 else 64` (was 64) and the adjacent comment was updated. Nothing else changed: num_warps 4, num_stages 2, S=16, kernel body untouched. The file has no `prepare()` and no warm-up, so there is nothing to extend; the new BLOCK_N variant compiles on first call as the old one did (the same constexpr-specialised launch, no extra variant for verify since length 8 keeps 64). No deviation from the spec.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff only changes a constexpr from a config-derived value (`length`); no syncs, allocation or value-dependent control flow.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a: no weight handling touched.
- [x] Activations/KV dtypes: n/a: unchanged by the diff.
- [x] Attention sees every key: 16 splits x 128 keys cover all 2047 old keys (last split masked); no skipping.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune, no host syncs in kernel_fn.

okc_lint output: `CLEAN: submission.py`. py_compile passed; __pycache__ removed.

## Risks / what could fail
low
Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Sliding decode_1 carries weight 46 and is 58% of the decode_1 load, so ~1 us there is ~+4% on that load.

## Result (after the run)
- job `88c62334775e48e4` (check), status done, passed True, tests 16/16
- speedup 1.3834x, 95% CI [1.3627, 1.4079], score 2722.4398420162893 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode-blockn128-single-tile.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode-blockn128-single-tile.patch`; branch `hyp/muse_attention_decode/sliding-decode-blockn128-single-tile`
