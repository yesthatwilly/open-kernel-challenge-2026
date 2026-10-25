---
problem: muse_decode_layer
slug: full-decode-splitkv-restack-on-pdl
created: 2026-10-24T224525Z
base: master
area: FA3 call (full decode, batch > 1)
change: >
  Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated
  shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2
  (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl,
  f1f2f35), the way moe stacked shape-disjoint wins by 3-way merge. Take branch
  hyp/muse_decode_layer/full-decode-cuda-splitkv-port (patch
  done/neutral/full-decode-cuda-splitkv-port.patch) and 3-way merge its additions unchanged: its
  load_inline CUDA module (separate module name from the dense-chain TU, so both compile at
  import: ~41 s each, still far under the 15 min run), its gate (_splitkv_gate: full layer, q_len
  1, batch > 1, config only), the splitkv attention + combine/norm-gate kernels that replace the
  paired pad copies, FA3 flash_fwd + FlashAttnFwdCombine, unpad copies and FusedRMSNormGate on
  those shapes, and its prepare() self-check and fallback. Resolve every conflict in favour of
  master outside the splitkv branch of kernel_fn: the dense-chain P/R path after the o GEMM,
  attn_boundary router_dtype logic and _experts(align=...) stay exactly as master. Print both
  modules ON/OFF lines.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "measured +12.8% on the old base (attention-only change, disjoint from the FFN-side dense chain): ~35 us of 347.5"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "measured +10.3%: ~45 us of 535; headline ~+1.2%"
  - shapes: "*layer=sliding *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_*"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: batch 1 outside gate
risk: low
depends_on: full-decode-cuda-splitkv-port
priority: 2
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T231559Z
base_sha: 6ee651fff7bb5c5df237c4dac9b6b32e474908f2
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/full-decode-splitkv-restack-on-pdl
restack_of: full-decode-cuda-splitkv-port
restack_patch: muse_decode_layer/hypotheses/done/neutral/full-decode-cuda-splitkv-port.patch
restack: conflicts
worktree: .worktrees/muse_decode_layer/full-decode-splitkv-restack-on-pdl
mode: check
implemented_at: 2026-10-25T002142Z
impl_commit: b2f95e63a6e8ad3cff7947760e69e448399e295d
submission_sha256: a90477b0e505faf805e24b9aae2037f8a91f63ca2b11165bb7c64813b8b83532
norm_sha256: f4fbc23fe33d9ee561f4344ec0899c79abb7cd699158d5065e0c9f5e6401b871
submission_blob: c9067b8201f2480c64910c4a71098df7ce616f73
lint_warnings: 30
attempt_file: 2026-10-25T002143Z-check-full-decode-splitkv-restack-on-pdl.md
prepared_at: 2026-10-25T002143Z
job: c243a29f26dc4725
dispatched_at: 2026-10-25T002202Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2680313318618024
speedup_ci: [1.2490108618838753, 1.2870518018397292]
harvested_at: 2026-10-25T010415Z
best_at_harvest: 1.2521147631671776
promotion: merged
hand_landed: true
promotion_commit: b047510f64cf1919b8313cb88b2f6fe17c1c2305
promotion_attempt: 2026-10-25T043405Z-submit-full-decode-splitkv-restack-on-pdl.md
---

# full-decode-splitkv-restack-on-pdl

## Change

Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decode_32 535.8 -> 490.2 (+10.3%), every other shape flat) onto current master (small-t-launch-minimal-dense-chain-pdl, f1f2f35), the way moe stacked shape-disjoint wins by 3-way merge. Take branch hyp/muse_decode_layer/full-decode-cuda-splitkv-port (patch done/neutral/full-decode-cuda-splitkv-port.patch) and 3-way merge its additions unchanged: its load_inline CUDA module (separate module name from the dense-chain TU, so both compile at import: ~41 s each, still far under the 15 min run), its gate (_splitkv_gate: full layer, q_len 1, batch > 1, config only), the splitkv attention + combine/norm-gate kernels that replace the paired pad copies, FA3 flash_fwd + FlashAttnFwdCombine, unpad copies and FusedRMSNormGate on those shapes, and its prepare() self-check and fallback. Resolve every conflict in favour of master outside the splitkv branch of kernel_fn: the dense-chain P/R path after the o GEMM, attn_boundary router_dtype logic and _experts(align=...) stay exactly as master. Print both modules ON/OFF lines.

## Expected

- `*layer=full load=decode_16 *` faster: measured +12.8% on the old base (attention-only change, disjoint from the FFN-side dense chain): ~35 us of 347.5
- `*layer=full load=decode_32 *` faster: measured +10.3%: ~45 us of 535; headline ~+1.2%
- `*layer=sliding *` flat: outside gate: control
- `*load=verify_*` flat: outside gate: control
- `*load=decode_1 *` flat: batch 1 outside gate

## Risk

Low: both parents passed tests; shapes are disjoint. Watch combined import compile time (two load_inline TUs) and GPU memory of both workspaces. Lands the base that queued full-verify-cuda-splitkv-stage2 builds on.

## Implementation notes

Hand merge (2026-10-24) of done/neutral/full-decode-cuda-splitkv-port.patch onto decode master 6ee651f, worktree muse_decode_layer/submission.py only. `git apply -3` had given 7 interleaved conflicts because the split-KV and dense-chain CUDA modules share scaffolding lines (#includes, namespace, build-function shape). They were not merged as text. The parent's split-KV block went in whole, as one contiguous section:
- **Placement:** the parent's whole split-KV section (its comment, `_SPLITKV_CPP`/`_SPLITKV_CUDA` source unchanged, `_build_splitkv` with load_inline name `muse_full_decode_splitkv_v1`, `_EXT`, `_sm_count`, `_splitkv_gate`, the launch helpers, the synthetic data, the reference, the self-check and `_splitkv_prepare`) is inserted as one block after `_sliding_forward`, before `prepare()`. It is a separate translation unit from master's `muse_decode_dense_chain_v1` (`_DL_CPP`/`_DL_CUDA`, untouched), and the two modules share no Python names.
- **LOG2E:** the parent's `LOG2E` definition was dropped. Master already defines the same constant in the sliding section, and it is resolved at call time.
- **prepare():** calls `_splitkv_prepare(device)` after master's `_dl_setup` and `_align_par_setup`, before the final synchronize.
- **kernel_fn:** inside master's non-sliding `else`, after FusedQKNorm and store_cache: `splitkv = _splitkv_gate(cfg)` (`_EXT` built and self-checked, window 0, q_len 1, 1 < batch <= 64, tp8 16/2 heads, d 64, page 1; config only). It runs kernels A and B, whose output feeds the unchanged o GEMM. The `else` branch is master's paired FA3, unpad and FusedRMSNormGate code, indented one level, with `and not splitkv` added to `paired` as the parent had it. The sliding fused path, the dense-chain P/R path, attn_boundary/router_dtype, `_experts(align=...)`, the side stream and the parallel align are byte-identical to master (`git diff -w` shows only the docstring bullet and the `paired` line as changed master lines).
- **Deviations, per LESSONS (none change kernel code):**
  - `_full_decode_splitkv` is split into `_splitkv_attn` (kernel A) and `_splitkv_combine` (kernel B). The self-check launches them separately, with `torch.cuda.synchronize()` and a print after each, so a fault is attributed to its kernel. kernel_fn's path has no sync; okc_lint rejected an in-function `check` flag.
  - The build adds `-lineinfo`.
  - Every [splitkv] print is flushed.
  - Explicit ON/OFF lines: `[splitkv] ON: self-check passed`, `[splitkv] OFF: SELF-CHECK FAILED`, `[splitkv] OFF: module not built`.
  - There are no child processes; the self-check runs in-process in prepare() on synthetic data only.
- **Harvest:** read the `[splitkv]` and `[small-t-dense-chain]` lines. Two load_inline compiles run at import, about 41 s each.

## Rules check

- [x] CUDA-graph capturable: kernel_fn's split-KV path has no .item()/.cpu()/synchronize()/prints. Its workspaces are torch.empty sized from page_table.shape and config, and the grid comes from the SM count (a device property). The syncs are only in `_splitkv_self_check`/`_splitkv_prepare`, called from prepare() (lint warns L2202/L2206/L2284, outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: `_SMS` caches the device SM count only, and `_SPLITKV_GRID` is set only during the self-check and reset in its finally block.
- [x] Weights: n/a: no weight is touched by this change. Master's prepare-time repacks are unchanged.
- [x] Activations: q is cast to FP8 E4M3 (as master's FA3 path does), the KV cache stays FP8, and the output is bf16 into the o GEMM. Partials are fp32.
- [x] Attention sees every key: kernel A walks every 1024-token chunk up to cache_seqlens[b], masking only positions >= the length. There is no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. load_inline compiles in-process at import (the same mechanism as master's dense-chain module). A grep for subprocess/Popen/os.system finds nothing.
- [x] All work happens inside the timed call: the attention and combine run per kernel_fn call, with no output caching or replay.
- [x] The first line keeps the marker `# okc: problem=muse_decode_layer`. There is no triton.autotune.

okc_lint output:
```
OK with warnings: .worktrees/muse_decode_layer/full-decode-splitkv-restack-on-pdl/muse_decode_layer/submission.py
30 x warn host-sync, all "(outside kernel_fn's call graph; fine in prepare())":
L968 L969 L970 L973 L977 L995 L1003 L1006 L1010 L1024 L1080 L1082 L1119 L1124 L1125 L1130 L1131 L1158
L1204 L1231 L1236 L1254 L1255 L1262 L1263 L1291 L1297 (master's align/dense-chain self-checks)
L2202 L2206 L2284 (split-KV self-check per-kernel syncs, _splitkv_prepare)
```
py_compile: OK. The __pycache__ was removed. Diff: 1 file, +692 / -54 (the 54 are the re-indented FA3 block plus the docstring line).

## Pre-flight audit

Verdict: **dispatch** (no fixes).

- The two CUDA modules do not collide. They have distinct load_inline names (`muse_decode_dense_chain_v1` / `muse_full_decode_splitkv_v1`), so they get separate build dirs and PyInit symbols. Every kernel is in an anonymous namespace. The host functions (`dl_*` vs `splitkv_*`) are distinct. Python globals are distinct (`_DL_EXT` vs `_EXT`/`_SMS`/`_SPLITKV_*`). The `cudaFuncSetAttribute` calls are per-kernel: split-KV 93,200 B dynamic smem x 2 CTAs/SM fits in 227 KB.
- The `_SPLITKV_CUDA` source is byte-identical to the parent patch, which passed the judge. The Python differs only in the deviations listed in the implementation notes (A/B split with syncs only in the self-check, `-lineinfo`, flushes, ON/OFF lines).
- The gate is window 0, q_len 1, 1 < batch <= 64, tp8, page 1, config only. It cannot overlap the sliding batch-1 fused path (window != 0). At T <= 32 the output `o` is bf16 [T, 1024], contiguous, the same as master's `o_norm_gate` output, so the dense-chain P/R path, attn_boundary, `_experts(align=)` and the side stream (batch-33 test) are unchanged.
- The self-check is passable: it is the parent's synthetic check and 2e-2 tolerance, and the parent's gated shapes sped up, so it ran ON. Build failure, check miss and exceptions each set `_EXT=None` and print an OFF line. A device fault inside the self-check would poison the context and cannot fall back, the same as the parent.
- Compile: it uses mma.sync m16n8k16, cp.async and cvt.rn.f16x2.e4m3x2 (sm_89+), all valid under `-arch=sm_90a`, with no wgmma or sm_90a-only feature. There are no child processes. lint: OK with the 30 host-sync warnings, all outside kernel_fn. py_compile: OK.

Residual risk: two sequential ~41 s load_inline compiles at import (about 80 s of the 15 min budget). An illegal-address fault in the self-check would fail the run rather than fall back, though that kernel is unchanged from the passing parent.

## Result

- job `c243a29f26dc4725` (check), status done, passed True, tests 13/13
- speedup 1.2680x, 95% CI [1.2490, 1.2871], score 17838.912518657933 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.patch`; branch `hyp/muse_decode_layer/full-decode-splitkv-restack-on-pdl`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 127.48 | 126.15 | 1.0145 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.93 | 142.7 | 1.004 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 204.98 | 209.82 | 0.9972 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 349.47 | 306.72 | 1.1446 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 245.77 | 242.33 | 1.0012 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 529.71 | 479.6 | 1.105 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 190.8 | 187.97 | 0.9987 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 192.44 | 201.51 | 0.9789 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 429.61 | 420.42 | 0.9978 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 563.65 | 564.2 | 1.0007 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 592.52 | 600.82 | 0.9948 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 861.3 | 864.67 | 0.9986 | flat |

gained on full decode_16, decode_32. Expected hit: 5/5. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-splitkv-restack-on-pdl`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

