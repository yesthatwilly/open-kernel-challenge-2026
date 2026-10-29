---
problem: muse_attention_decode
slug: full-verify-paged-gather-gate-kernel-r2
created: 2026-10-29T222928Z
base: master
mint_blob: "2eb03f76ceefa9bf81e28616ced3b33be3a70443"
best_at_mint: 1.634
area: full-layer verify attention (FA3 call at verify_16 / verify_32)
change: >
  Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one
  build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the judge's PyTorch headers
  require C++20 (#error 'C++20 or later compatible compiler is required to use PyTorch.' in
  main.o), so the background build failed and master was kept (check 06c05172). Set -std=c++20 in
  _FVG_FLAGS, matching master's other extensions (c320, sliding window use '-std=c++20'), and also
  pass extra_cflags=['-O3', '-std=c++20'] to the fvg load_inline so the host main.cpp builds with
  C++20. Nothing else changes: same route gates, race (<= 0.95 per row), cooperative launch,
  layouts and [fvg] prints.
expected:
  - shapes: layer=full load=verify_16*
    direction: faster
    why: node e2e 0.83-0.84; H200 est 0.87-0.95
  - shapes: layer=full load=verify_32*
    direction: faster
    why: node e2e 0.82; H200 est 0.86-0.92
  - shapes: layer=sliding*
    direction: same
    why: not routed
risk: high
depends_on: full-verify-paged-gather-gate-kernel
priority: 1
kind: primary
controls_for: null
prior: "full-verify-paged-gather-gate-kernel: its kernel never ran (C++17 build flag failed against PyTorch's C++20 headers); this is the same change with -std=c++20"
target: check
value_gain: 3.0
value_p: 0.35
value_ev: 1.05
value_cost: 150k tokens, 1 judge run
evidence: >
  H100 fa3gate (research/h100-fa3gate-1029.md): e2e 272 vs 331 us at v32 (0.82), 147-151 vs
  176-181 us at v16 (0.83-0.84); judge board 4e386647 full verify_16 162.7 us, verify_32 302.1 us;
  judge 06c051723643462f fvg build failed on the C++17 flag, master kept
user_requested: true
review: proceed
status: done
review_at: 2026-10-29T223204Z
review_companions: null
review_best: 1.6339575246416982
review_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
stage: harvested
claimed_at: 2026-10-29T223204Z
base_sha: e3a95e5080f0844b6bb601639a5bb3051e58c707
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2
worktree: .worktrees/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2
mode: check
restack_of: full-verify-paged-gather-gate-kernel
stale_base: null
implemented_at: 2026-10-29T223301Z
impl_commit: 37649103a331b49c92b219f1c9d06caaf5d82fc9
submission_sha256: 65e27326262544aaf568f231004cc9590e6f74ef2c217956b5e3e7b47b1f539d
norm_sha256: 139856ac1aefff5e19ea405807d72d6e1149eed43376726291b8bea4e188fa1b
submission_blob: 0f7d6195263717e5dff426814b77ed679a083408
lint_warnings: 105
attempt_file: 2026-10-29T223301Z-check-full-verify-paged-gather-gate-kernel-r2.md
prepared_at: 2026-10-29T223301Z
job: 184991095dff4d50
dispatched_at: 2026-10-29T223312Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.6421954867431368
speedup_ci: [1.6175625544419896, 1.6668284190442837]
harvested_at: 2026-10-29T224331Z
best_at_harvest: 1.6339575246416982
promotion: null
mechanism: null
---

# full-verify-paged-gather-gate-kernel-r2

## Change

Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the judge's PyTorch headers require C++20 (#error 'C++20 or later compatible compiler is required to use PyTorch.' in main.o), so the background build failed and master was kept (check 06c05172). Set -std=c++20 in _FVG_FLAGS, matching master's other extensions (c320, sliding window use '-std=c++20'), and also pass extra_cflags=['-O3', '-std=c++20'] to the fvg load_inline so the host main.cpp builds with C++20. Nothing else changes: same route gates, race (<= 0.95 per row), cooperative launch, layouts and [fvg] prints.

## Expected

- `layer=full load=verify_16*` faster: node e2e 0.83-0.84; H200 est 0.87-0.95
- `layer=full load=verify_32*` faster: node e2e 0.82; H200 est 0.86-0.92
- `layer=sliding*` same: not routed

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 fa3gate (research/h100-fa3gate-1029.md): e2e 272 vs 331 us at v32 (0.82), 147-151 vs 176-181 us at v16 (0.83-0.84); judge board 4e386647 full verify_16 162.7 us, verify_32 302.1 us; judge 06c051723643462f fvg build failed on the C++17 flag, master kept
- **Gain if it works:** 3.0% headline; **p** 0.35; **EV** 1.05% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (EV 1.05% < 1.5%), minted because the user described it.
- **Cost:** 150k tokens, 1 judge run.

## Risk

06c05172 never ran the kernel: the extension failed to compile (C++17 flag vs PyTorch's C++20 requirement); its 1.6154 is master's level. The review of the parent (proceed; H200 re-sized to ~+2-3.5%) applies unchanged.

## Adversarial review

Reviewed 2026-10-29T223204Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: fresh (BEST 1.6340 / blob 2eb03f76 unchanged since mint; parent reviewed 'proceed'). The -std=c++20 fix is sufficient. The extra_cflags addition is redundant but harmless. A scan of the never-compiled fvg CUDA text found no obvious compile error. Widen the build-error tail so a second failure can be read.

### Justification
Fix sufficiency: the #error came from the nvcc compile of cuda.cu (gcc-style caret output from the host pass). load_inline appends extra_cuda_cflags after torch's own default -std=c++20, so the explicit -std=c++17 in _FVG_FLAGS won. The '[2/3] ccache c++ ... main.o' line after the error is only ninja's progress line; no failure output follows it. Master's sk, skF and slide modules pass no -std and no extra_cflags and build on the judge, so torch's default host flags are already C++20. Setting -std=c++20 matches c320 (L6812) and sw (L8824), and nvcc on the judge accepts it because sw built in the same log. extra_cflags=['-O3','-std=c++20'] is a no-op safety. Dropping -std from _FVG_FLAGS would work equally well. Scan of _FVG_CUDA (branch L9437-9988): the wgmma operand counts are consistent (qk %0-31 d, %32-35 a, %36 desc, %37 acc; pv likewise with tnspB). The scale/transpose immediates match the e4m3 (no transpose) and f16 register-A forms. The cudaLaunchKernelEx template call matches the KFn signature arg for arg. The cond ? attr : nullptr array decay is legal. Includes cover C10_CUDA_CHECK, at::kInt and uintptr_t. The pin() asm '::' clobber form is the CUTLASS idiom. Device lambdas need no --extended-lambda. Dynamic smem is about 87 KB x MINB 2, under the 227 KB limit. Weakest link: this exact stripped text has never been compiled anywhere (the parent md says the node has no nvcc), so a second build or ptxas error is possible. A minor candidate is wg_pv's 'setp.ne.b32 p, 1, 0' with an immediate first operand. ptxas normally accepts it, but copying wg_qk's register form (or folding the constant) removes the doubt at no cost.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
No companions are needed. This is a one-flag restack of a reviewed hypothesis, and the parent review's race, gates and confounder reasoning carry over unchanged.

### Open questions
- Raise the build error capture from str(e)[-600:] to about 3000 chars, as [sw] does (L8840). The parent's 600-char tail cut off the root compiler line. A second failure in this never-compiled text should be diagnosable from one judge log.
- TORCH_CUDA_ARCH_LIST is not pinned for fvg. torch adds its default sm_90 gencode next to the explicit compute_90a, which compiles the #else stub (error bit 4). Master's sk/skF builders also set and restore os.environ['TORCH_CUDA_ARCH_LIST'] while the fvg thread runs, so the arch list depends on timing. The runtime should load the sm_90a cubin on H100/H200. If the stub ran, the import-time counter/error-word check (L10205) keeps master, so this is safe but could waste the check. Optional: drop the duplicate gencode, or note in the [fvg] print which cubin ran (fvg_attr numRegs > ~40 implies the real kernel).


## Implementation notes

Restacked the parent (full-verify-paged-gather-gate-kernel, patch applied cleanly on master by churn claim) and changed four lines, all in the [fvg] section, as the r2 review asked:
- `_FVG_FLAGS`: `-std=c++17` -> `-std=c++20` (the parent judge run 06c05172 failed on PyTorch's `#error C++20 or later`; the hypothesis's extra_cuda_cflags overrode PyTorch's default C++20). extra_cflags not added (the review found main.cpp already builds as C++20; master's sk/skF/slide pass none).
- build-error capture `str(e)[-600:]` -> `[-3000:]` (the 600-char tail cut off the root compiler line).
- `wg_pv`: `setp.ne.b32 p, 1, 0` -> `setp.ne.b32 p, %37, 0` with a new `"r"(1)` input, the same register form as `wg_qk` (removes a literal first operand; same semantics: scale-d = 1).
No other change. Done by the main thread (no agent); py_compile OK, __pycache__ removed.
## Rules check

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

## Result

- job `184991095dff4d50` (check), status done, passed True, tests 16/16
- speedup 1.6422x, 95% CI [1.6176, 1.6668], score 2293.454117006171 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 10.82 | 11.26 | 0.9441 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 15.62 | 15.74 | 0.9941 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.38 | 15.44 | 1.0075 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 104.77 | 103.45 | 1.0099 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 16.81 | 16.94 | 1.0217 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 177.91 | 189.43 | 0.9352 | lost |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.9 | 13.46 | 1.0207 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.09 | 30.35 | 0.9855 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 19.51 | 19.21 | 1.008 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.74 | 151.44 | 1.0767 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 27.1 | 27.83 | 0.9777 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.09 | 282.33 | 1.0714 | gained |

gained on full verify_16, verify_32; lost on sliding decode_1; full decode_32. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

