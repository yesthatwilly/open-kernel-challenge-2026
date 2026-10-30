# muse_moe_w4a16 · 2026-10-29T235620Z · check · eae-cold-iso-keepalive-fix

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-29T235440Z-eae-cold-iso-keepalive-fix.md` (branch `hyp/muse_moe_w4a16/eae-cold-iso-keepalive-fix`).

- **Job:** `582bb33467dc4663` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `92e1dcd3`

## Goal
Correctness fix with no timed-path change: in _eae_class (muse_moe_w4a16/submission.py ~18447-18477), the iso[('c', a)] timing graphs are captured over h3/y3/l3 tensors allocated outside the capture and held only by the lambda's default args; the loop frees the previous set while its graph still writes to it (use-after-free during _rt_time; sticky illegal address on gpunode06 5/5 with the SGLang baseline in process, the judge's layout; written into freed memory without faulting on the judge so far). Keep a list of every (h3, y3, l3) until timing ends (keep.append after allocation; del keep with the graphs). Routes, picks and kernel_fn are unchanged; the moe board's final reruns (top entries rerun several times on held-out data) must not carry the fault.

## What changed, and where
Applied the 3-line hunk from branch hyp/muse_moe_w4a16/eae-t1-guard (b743b4e4) to `_eae_class` only (main thread, no agent): `keep = []` beside graphs/outs/iso; `keep.append((h3, y3, l3))` after the cold-iso buffers are allocated; `keep` deleted with the graphs after `_rt_time`. Resulting file sha256 986a2bad, 948,330 B: identical to the bytes validated 4/4 on gpunode06 with the SGLang baseline in process (research/h100-eae-crash-1029.md). No other change: kernel_fn, routes and picks (from `rat` on the kernel_fn graphs) are unchanged; `crat` is print-only.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no change to kernel_fn or its call graph; the edit is in prepare-only timing code.
- [x] Nothing derived from `data` is cached across calls: `keep` lives only inside `_eae_class` and is deleted before it returns.
- [x] Weights: unchanged (no repack change).
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention sees every key: n/a (MoE expert block).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added; the fix only extends three prepare-time tensors' lifetime.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] Size: `wc -c` = 948,330 B < 1,000,000.
- [x] First line kept; `python3 -I -m py_compile` OK; __pycache__ removed; okc_lint: OK with warnings (prepare-only, same as master).

## Risks / what could fail
low
Root cause and node validation: research/h100-eae-crash-1029.md (branch hyp/muse_moe_w4a16/eae-t1-guard b743b4e4): with the fix, natural/forced v1/forced v6pe2 T=1 routes all pass with the baseline in process; every raced arm exact. Decode's stack is unaffected (graphs allocate inside the capture).

## Result (after the run)
- job `582bb33467dc4663` (check), status done, passed True, tests 5/5
- speedup 2.0469x, 95% CI [2.0161, 2.0799], score 6570.333081756526 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 2.020119462295186, "ci": [1.9898176703607582, 2.0504212542296134], "job": "24f183ff300949f2"}; base result: muse_moe_w4a16/hypotheses/done/gain/engine-a-a4b-a5b-encodings-race.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/eae-cold-iso-keepalive-fix.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/eae-cold-iso-keepalive-fix.patch`; branch `hyp/muse_moe_w4a16/eae-cold-iso-keepalive-fix`
