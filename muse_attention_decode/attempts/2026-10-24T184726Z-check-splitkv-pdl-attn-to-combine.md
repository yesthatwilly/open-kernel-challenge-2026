# muse_attention_decode · 2026-10-24T184726Z · check · splitkv-pdl-attn-to-combine

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-splitkv-pdl-attn-to-combine.md` (branch `hyp/muse_attention_decode/splitkv-pdl-attn-to-combine`).

- **Job:** `113cbf3093dd40b9` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `9b85387`

## Goal
Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv_combine (host), launch kernel B with cudaLaunchKernelEx: cudaLaunchConfig_t with gridDim (B, 16), blockDim 128, stream = current stream, and one attribute cudaLaunchAttributeProgrammaticStreamSerialization with programmaticStreamSerializationAllowed = 1. (2) First statement of splitkv_combine_kernel: asm volatile("griddepcontrol.wait;" ::: "memory"); this blocks until kernel A has completed and its writes are visible, so the combine's reads are as safe as today. (3) In splitkv_attn_kernel, after the main tile loop and before the final cp_wait<0>(), every thread executes asm volatile("griddepcontrol.launch_dependents;" :::); a CTA that exits early (bx >= total) triggers implicitly by exiting. Nothing else changes; CUDA graph capture records the programmatic edge (CUDA >= 12.3; the judge image is cu130). Rename the module muse_full_decode_splitkv_v2. If cudaLaunchKernelEx returns an error at the self-check, fall back to the plain <<<>>> launch and print '[splitkv] PDL off'; print '[splitkv] PDL on' otherwise.

## What changed, and where
All in the worktree's muse_attention_decode/submission.py (+114/-8), not run (no GPU here).
- Kernel A: `griddepcontrol.launch_dependents` after the main tile loop, before the final cp_wait<0>().
- Kernel B: `griddepcontrol.wait` (with memory clobber) as the first statement.
- Host splitkv_combine: when a static `g_pdl` is set, launches via cudaLaunchKernelEx with the programmatic-stream-serialization attribute; on any error it clears the error, sets g_pdl=false and does the plain <<<>>> launch. New `splitkv_set_pdl` / `splitkv_get_pdl` exported. Module renamed muse_full_decode_splitkv_v2.
- Review note applied: `_pdl_ab` times the graph-captured attn+combine sequence (synthetic batch 16 and 32, 80K lengths) with and without PDL, 20 rounds x 20 replays, alternating order, CUDA events, median per sequence; prints `[splitkv] pdl A/B ...`. Runs at import only.
- Deviation: this file has no prepare(); its once-per-process setup is the import-time block, so the new `_prepare_splitkv()` (build, PDL on, self-check, A/B) is called from there. Self-check runs with PDL on; if it fails, it is retried with PDL off. The A/B does not choose the path; PDL stays on whenever the launch is supported. Final line: `[splitkv] PATH CHOSEN: split-KV CUDA, PDL on|off`.
- Trigger stays after the loop (review's earlier-trigger and pre-wait-load ideas not taken: new ideas).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path has no syncs/allocation on values; PDL launch is capturable; syncs only in import-time _pdl_ab/self-check.
- [x] Nothing derived from `data` is cached across calls (A/B uses synthetic tensors, freed).
- [x] Weights: n/a: no weight handling touched.
- [x] Activations/KV/query dtypes unchanged (diff touches only launch and PDL asm).
- [x] Attention sees every key: n/a, no change to what A computes; B's wait guarantees all partials.
- [x] No timing patches, harness access, reference import, subprocesses or network: CUDA events are used only on own synthetic work at import.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune; module-level marker line kept.

Lint output:
```
warn  host-sync    L1126: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1136: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1145: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_attention_decode/submission.py
```
py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). Below headline noise by design; judge on the full decode_16/32 medians against the paired SGLang ratio.

## Result (after the run)
- job `113cbf3093dd40b9` (check), status done, passed True, tests 16/16
- speedup 1.3929x, 95% CI [1.3720, 1.4138], score 2703.9856229611055 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/splitkv-pdl-attn-to-combine.result.json`, `muse_attention_decode/hypotheses/done/neutral/splitkv-pdl-attn-to-combine.patch`; branch `hyp/muse_attention_decode/splitkv-pdl-attn-to-combine`
