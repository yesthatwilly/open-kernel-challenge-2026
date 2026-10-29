# muse_attention_decode · 2026-10-29T223301Z · check · full-verify-paged-gather-gate-kernel-r2

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-29T222928Z-full-verify-paged-gather-gate-kernel-r2.md` (branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2`).

- **Job:** `184991095dff4d50` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `37649103`

## Goal
Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the judge's PyTorch headers require C++20 (#error 'C++20 or later compatible compiler is required to use PyTorch.' in main.o), so the background build failed and master was kept (check 06c05172). Set -std=c++20 in _FVG_FLAGS, matching master's other extensions (c320, sliding window use '-std=c++20'), and also pass extra_cflags=['-O3', '-std=c++20'] to the fvg load_inline so the host main.cpp builds with C++20. Nothing else changes: same route gates, race (<= 0.95 per row), cooperative launch, layouts and [fvg] prints.

## What changed, and where
Restacked the parent (full-verify-paged-gather-gate-kernel, patch applied cleanly on master by churn claim) and changed four lines, all in the [fvg] section, as the r2 review asked:
- `_FVG_FLAGS`: `-std=c++17` -> `-std=c++20` (the parent judge run 06c05172 failed on PyTorch's `#error C++20 or later`; the hypothesis's extra_cuda_cflags overrode PyTorch's default C++20). extra_cflags not added (the review found main.cpp already builds as C++20; master's sk/skF/slide pass none).
- build-error capture `str(e)[-600:]` -> `[-3000:]` (the 600-char tail cut off the root compiler line).
- `wg_pv`: `setp.ne.b32 p, 1, 0` -> `setp.ne.b32 p, %37, 0` with a new `"r"(1)` input, the same register form as `wg_qk` (removes a literal first operand; same semantics: scale-d = 1).
No other change. Done by the main thread (no agent); py_compile OK, __pycache__ removed.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: `_fvg_forward` makes Triton + one ext launch on the current stream; no `.item()/.cpu()/synchronize()` in kernel_fn's call graph (lint: none); branches only on config (`_fvg_gate`) and shapes (`_fvg_shape_ok`); lens/page table read on device; capture verified in prepare by `_alt_check` and the race graphs.
- [x] Nothing derived from `data` is cached across calls: `_FVG_BUF` holds partials/counters sized from G and batch only; `_FVG_ROUTE` frozen at import from synthetic races.
- [x] Weights: n/a: attention problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (read as e4m3, untouched format); queries rounded to FP8 E4M3 (as FA3); P f16 and fp32 O/softmax (finer than FA3); output bf16.
- [x] Attention sees every key: stream-K covers every 64-key tile of every request; only causal / length masking on the last tiles.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import (own fp64 reference), subprocesses (a `threading.Thread` around load_inline, as master's builds) or network.
- [x] All work happens inside the timed call: pre-attention, attention, combine and epilogue all run per call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
warn  host-sync  ... (outside kernel_fn's call graph; fine in prepare())   [x ~105, master's prepare + [fvg] prepare lines]
warn  dtype      L7646: q = torch.full(..., dtype=torch.uint8 ...).view(FP8)   [master, pre-existing]
OK with warnings: .../muse_attention_decode/submission.py
```
`python3 -I -m py_compile`: OK (__pycache__ removed).

## Risks / what could fail
high
06c05172 never ran the kernel: the extension failed to compile (C++17 flag vs PyTorch's C++20 requirement); its 1.6154 is master's level. The review of the parent (proceed; H200 re-sized to ~+2-3.5%) applies unchanged.

## Result (after the run)
- job `184991095dff4d50` (check), status done, passed True, tests 16/16
- speedup 1.6422x, 95% CI [1.6176, 1.6668], score 2293.454117006171 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2`
