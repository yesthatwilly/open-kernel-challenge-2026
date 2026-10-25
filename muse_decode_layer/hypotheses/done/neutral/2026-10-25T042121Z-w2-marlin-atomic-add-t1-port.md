---
problem: muse_decode_layer
slug: w2-marlin-atomic-add-t1-port
created: 2026-10-25T042121Z
base: master
area: w2 marlin (T=1)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m
  (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100
  -> 10/100 replays). In _experts (the two moe_wna16_marlin_gemm calls), at num_tokens == 1 ONLY
  (config: every decode_1 shape), build a separate kwargs dict for the SECOND call (w2,
  size_n=size_k=hidden) with use_atomic_add=True and use_fp32_reduce=False; the w13 call and every
  num_tokens > 1 call keep `common` unchanged. Because w2 then accumulates into its C buffer, give
  that path a cache3 that is zeroed (a dedicated slice with .zero_() inside kernel_fn, graph-safe)
  unless sglang's wrapper already zeroes C with atomic add (read the wrapper; do not double-zero).
  Prepare self-check at T=1 vs master's path: rel L2 vs an fp32-dequant reference no worse than
  1.5x master's own, finite, else keep master and print PATH CHOSEN.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "Marlin (2 calls) 40.3-40.9 us at T=1, same w2 shape as moe bs=1 (w2 K-slice fp32 reduce replaced by atomics): expect -3..4 us per decode_1 shape (-2.5%), weights 46+16; also check runs_ns mode counts. Kill: both decode_1 shapes within +-1% (z<2)"
  - shapes: "*load=decode_16*"
    direction: flat
    why: T>1 keeps master kwargs (moe bs 8/16/32 flat)
  - shapes: "*load=decode_32*"
    direction: flat
    why: T>1 keeps master kwargs
  - shapes: "*load=verify_*"
    direction: flat
    why: T>1 keeps master kwargs
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T042335Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T042335Z
base_sha: 4086e385d4c82ab39da7076fd7a040127403956c
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/w2-marlin-atomic-add-t1-port
worktree: .worktrees/muse_decode_layer/w2-marlin-atomic-add-t1-port
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T042854Z
impl_commit: 1e70f0cb3fdfaae3151262adec6153b5a06d3e88
submission_sha256: 83a46d0d62743469dfe7967584070fac1d6422217328efeef980b774ce3c838f
norm_sha256: d27fda6c4a9e3625660f6deabea785f8b8cdc1eb05deb5ab7db0d66b7e5b6a83
submission_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
lint_warnings: 34
attempt_file: 2026-10-25T042854Z-check-w2-marlin-atomic-add-t1-port.md
prepared_at: 2026-10-25T042854Z
job: 7a9f5de5aecc4aba
dispatched_at: 2026-10-25T042908Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2855726646854595
speedup_ci: [1.2662890747151776, 1.3048562546557412]
harvested_at: 2026-10-25T043619Z
best_at_harvest: 1.2680313318618024
promotion: merged
hand_landed: true
promotion_commit: 00e36e3a74224d6c0cc4f10708dfd9e759d8ee84
promotion_attempt: 2026-10-25T045638Z-submit-w2-marlin-atomic-add-t1-port.md
---

# w2-marlin-atomic-add-t1-port

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100 -> 10/100 replays). In _experts (the two moe_wna16_marlin_gemm calls), at num_tokens == 1 ONLY (config: every decode_1 shape), build a separate kwargs dict for the SECOND call (w2, size_n=size_k=hidden) with use_atomic_add=True and use_fp32_reduce=False; the w13 call and every num_tokens > 1 call keep `common` unchanged. Because w2 then accumulates into its C buffer, give that path a cache3 that is zeroed (a dedicated slice with .zero_() inside kernel_fn, graph-safe) unless sglang's wrapper already zeroes C with atomic add (read the wrapper; do not double-zero). Prepare self-check at T=1 vs master's path: rel L2 vs an fp32-dequant reference no worse than 1.5x master's own, finite, else keep master and print PATH CHOSEN.

## Expected

- `*load=decode_1 *` faster: Marlin (2 calls) 40.3-40.9 us at T=1, same w2 shape as moe bs=1 (w2 K-slice fp32 reduce replaced by atomics): expect -3..4 us per decode_1 shape (-2.5%), weights 46+16; also check runs_ns mode counts. Kill: both decode_1 shapes within +-1% (z<2)
- `*load=decode_16*` flat: T>1 keeps master kwargs (moe bs 8/16/32 flat)
- `*load=decode_32*` flat: T>1 keeps master kwargs
- `*load=verify_*` flat: T>1 keeps master kwargs

## Risk

Shares decode_1 with running stack-route-topk-pdl-unpool-nf1 and dense-chain T1 gate: attribute by the Marlin row and runs_ns modes.

## Adversarial review

Reviewed 2026-10-25T042335Z by an adversarial subagent (model opus).

**Verdict: proceed.** Decode's T=1 MoE still runs SGLang Marlin w2. On master b047510 the dense chain (use_dl, tokens<=32) hands kernel R's align to _experts, which still makes both moe_wna16_marlin_gemm calls (Marlin row 40.9/46.2 us at sliding/full decode_1), so the moe win can carry over. Proceed with no companion, but fix the spec to match what actually landed on moe: keep use_fp32_reduce=True, keep cache3 aliasing cache13, and add no dedicated zeroed slice unless the self-check proves a zero is needed.

### Justification
Gate check (the point the caller asked about): neither the landed dense chain (small-t-launch-minimal-dense-chain-pdl) nor any other landed work replaced Marlin at small T. moe-chain-cuda-swapab and moe-small-t-cuda-swapab-port are failed, regressed or still queued and are not on master. kernel_fn line ~2465 calls _experts(xn, ..., align=align) when tokens<=DL_MAX_TOKENS, and _experts (line 2304) still issues w13 Marlin, then silu_and_mul, then w2 Marlin with common use_atomic_add=False and use_fp32_reduce=True. No hold is needed. Mechanism: the w2 shape at T=1 matches moe bs=1 exactly. Both have size_n=size_k=4096, top_k=1, 8 local routes, the same 32 local experts out of 256, block_size_m=8 (marlin-block-m-cap-16 leaves 8 at T=1), and the same marlin_make_workspace(max_blocks_per_sm=4), so slice_count and the K-slice reduce that atomic add removes should be the same. moe measured bs=1 48.82 -> 44.88 us (-3.9 us, about 6 err units) with bs 8/16/32 flat, so gating at T=1 only is the right scope here (verify_1 is T=8, which was flat in moe). Expected size: -3..4 us on 127.5 us (sliding) and 142.9 us (full) decode_1, about -2.5..3%, with a per-shape median err of about 0.5-1 us. That is readable per shape and invisible at the headline, consistent with the stated kill rule. Correctness has a large margin: moe's atomic path came to rel-L2 0.0048 vs master, while decode's bounds are 3-15% vs the fp32 reference, and determinism is not required. WEAKEST LINK: the change paragraph is not a faithful port of what landed. (1) It says use_fp32_reduce=False. The landed moe code (master 9080a71, submission.py ~L590) keeps use_fp32_reduce=True and adds only use_atomic_add=True. Its reviewer established that the pair is not required, because c_tmp is only allocated when fp32_reduce && !atomic. Setting False changes the reduce numerics and makes this a different, untested knob. (2) It proposes 'a dedicated slice with .zero_()'. moe's self-check chose mode 'atomic' with no memset ('zeroing memset in path: False', log a2744bc496b84790 L148-149), with cache3 still aliasing cache13, which holds w13's output and so is guaranteed non-zero. A dedicated or fresh buffer could read as zero and hide a missing zero, and an unneeded memset costs about 1 us of the 3-4 us gain. Port the moe pattern: try 'atomic' (no zero) first, then 'atomic_zero', then master, on the aliased buffer. A secondary weak link: in decode the Marlin row differs between sliding (40.9) and full (46.2) at the same weights, so cold-L2 and launch-overlap context differ from moe's isolated benchmark. The gain may come out unequal between the two decode_1 shapes. Judge each shape on its own median.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl: at T=1, sorted_ids/block_experts/n_padded come from kernel R (_dl_forward), not from moe's triton-map-align. Kernel R's padding sets the m-block count Marlin sees, and so the slice depth atomic add acts on. A different n_padded than moe's would change the gain's size, not its sign. The prepare self-check must run through this same align=... path, with remote (-1) routes present, not the align=None fallback.
- post-expert-norm-triton-small-t: at T<32, _post_norm_combine_kernel reads y.contiguous() and indexes it by local_ids. With atomic and no zero, rows of remote routes in cache3 hold stale w13 data. This is safe only because the combine masks remote slots. The moe equivalent passed with this layout, but decode's self-check must compare the final _experts output (after the combine), not y.
- marlin-block-m-cap-16: fixes block_size_m=8 at T=1, the same as moe's bs=1. A later block-m change at T=1 would shift the effect, but it does not confound this run.
- full-decode-splitkv-restack-on-pdl (this run's base): its gate is full decode with batch>1, so it never runs at decode_1. It is not a confounder, only the base commit.

### Interference and A/B needs
No companion is warranted. The change is one kwarg on one launch at one T, plus at most one memset that has its own profile row. A ~3-4 us gain at decode_1 is attributable from the per-shape medians against base b047510, and every other shape is byte-identical code (num_tokens>1 keeps common), so those shapes are a built-in control. A null or loss result is also unambiguous: with no memset it retires the knob for decode, and with a memset that row shows the cost. An ab-revert of the dense chain is pointless, because it is the only T=1 path on master and reverting it would measure a configuration nobody will ship. The running stack-route-topk-pdl-unpool-nf1 and dense-chain-stage2-attn-side-pdl-t1-gate also touch decode_1 but are not on master, so they do not confound this run's base comparison. Their interaction matters only when they are restacked later.

### Open questions
- Does kernel R's n_padded at T=1 equal moe's (8 routes -> up to 8 m-blocks of 8)? If kernel R pads to a fixed capacity, Marlin may launch extra empty m-blocks and slicing may differ from moe.
- Which mode does decode's self-check pick (atomic vs atomic_zero)? The PATH CHOSEN line must appear in the log, and if atomic_zero is chosen, read the memset row against the Marlin delta.
- Does profile.txt split w13 and w2 Marlin launches? If so, read w2's own row at decode_1 (expect -3..4 us). If not, rely on per-shape medians.
- Should the gate later widen to T=8 (verify_1)? moe saw bs=8 flat (1.0014), so no for now.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (+88/-3).
- `_experts(..., w2_mode=None)`: w2_mode defaults to `state["w2_mode"]`. When it is not "master" AND `num_tokens == 1` AND `align is not None` (kernel R / dense-chain path) AND `dispatch is None`, the w2 `moe_wna16_marlin_gemm` call gets `dict(common, use_atomic_add=True)`. The w13 call, every T>1 call and the align=None path keep `common` unchanged.
- Review corrections applied (review wins over the spec text): `use_fp32_reduce` stays True; `cache3` still aliases `cache13` (no dedicated zeroed slice); selector order "atomic" (no memset) -> "atomic_zero" (`cache3.zero_()` before w2) -> master, as moe.
- `_w2_atomic_self_check(state, device)` runs in prepare() after `_dl_setup`/`_align_par_setup`. It drives kernel R (`_dl_route`) at T=1 with crafted logits (1-4 forced local experts, the rest remote, 6 local_start values, 12 trials), then compares the post-combine `_experts` output with `w2_mode="atomic"/"atomic_zero"` against `w2_mode="master"` on identical inputs and the same align (rel L2 <= 5e-3, moe's tolerance, finite). Requires both local and remote slots to have been exercised, else master. If `state["dl"]` is None (dense chain off) the mode is master. Every launch is announced with a flushed print and followed by `torch.cuda.synchronize()` plus a flushed print. Loud `[w2-atomic] PATH CHOSEN = ...` line (T=1 chosen mode, T>1 MASTER).
- Deviation: the self-check compares to master's w2 numerics (as moe did), not to an fp32-dequant reference; the review sets the comparison target as master within moe's tolerance.
- No GPU here: nothing was run.

## Rules check

- [x] CUDA-graph capturable: diff adds only a kwarg choice (config-only: num_tokens, align presence, prepare-time mode) and an optional `cache3.zero_()`; no sync/.item in kernel_fn; self-check is prepare()-only.
- [x] Nothing derived from `data` is cached across calls: the mode is chosen from synthetic inputs in prepare().
- [x] Weights: unchanged (n/a: no repacking touched).
- [x] Activations BF16 / KV FP8: unchanged; w2 still bf16 output, fp32 reduce kept.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness, module edits, reference import, subprocess or child processes: none added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: OK with warnings (only pre-existing and new prepare()-only host-sync warnings, no errors). py_compile passed; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent. This was a read-only audit: there is no GPU here, so nothing was run. **Verdict: dispatch-with-fixes.**

- Path reach: kernel_fn's MoE block does not depend on the layer type. At T=1, both the sliding and the full decode_1 shapes take `use_dl` -> `_experts(..., align=align)` with dispatch=None. So the atomic gate (mode != master, T==1, align passed, no dispatch) is live on both shapes whenever the dense-chain self-check passes, which its own `[dl] PATH CHOSEN` line reports. If dl is OFF, both shapes fall to the dispatch path, the run is a no-op, and the `[w2-atomic]` line says so.
- Fix 1 (a gate that could fail on noise): `W2_ATOMIC_TOL` goes from 5e-3 to 2e-2. moe measured atomic vs non-atomic at 4.79e-3 worst. This check also takes the max over 12 trials, so at 5e-3 it could fall back to master on bf16 rounding alone. 2e-2 is still far below the judge's 3-15% bound, and far below the O(1) error a missing zero produces (see fix 2).
- Fix 2 (detecting stale C): the self-check now fills cache3 with 100.0 right before w2, through `w2_fill=W2_JUNK`. That is a prepare()-only kwarg of `_experts` with default None, so kernel_fn is unchanged. It applies to both the master reference and the candidates. Before this fix, C could hold w13's output, the previous call's y or zeros, depending on the caching allocator, so a non-zeroing atomic path could pass on zeroed memory. Now "atomic" (no memset) passes only if Marlin's first k-slice stores rather than adds, as on moe 9080a71. The master reference with junk also checks that the combine never reads remote rows.
- Fix 3 (hygiene): `dl["cnt_r"].zero_()` after the w2 self-check, as `_dl_setup` does after its own self-check. The w2 check launches kernel R 12 more times, and an exception partway through must not leave the counter dirty.
- Rules: OK. The only changes on kernel_fn's path are a kwarg choice made from config and prepare-time state, plus an optional `cache3.zero_()`. No sync, allocation or branch on input values was added. okc_lint is OK; its warnings are all prepare()-only syncs. py_compile is OK, and __pycache__ was removed.

Residual risk: the gain may be smaller in decode than in moe, because kernel R's align and the in-layer cold-L2 and launch context differ. If Marlin's atomic path does accumulate onto C, the self-check now picks atomic_zero, and its ~1 us memset eats part of the gain.

## Result

- job `7a9f5de5aecc4aba` (check), status done, passed True, tests 13/13
- speedup 1.2856x, 95% CI [1.2663, 1.3049], score 17595.504806050383 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/w2-marlin-atomic-add-t1-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/w2-marlin-atomic-add-t1-port.patch`; branch `hyp/muse_decode_layer/w2-marlin-atomic-add-t1-port`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 126.15 | 123.88 | 1.0215 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.7 | 139.18 | 1.0241 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 209.82 | 204.68 | 0.998 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 306.72 | 308.69 | 0.9975 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 242.33 | 240.67 | 1.0066 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 479.6 | 473.31 | 0.9983 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 187.97 | 191.42 | 0.9998 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 201.51 | 201.63 | 1.0044 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 420.42 | 414.6 | 1.0147 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 564.2 | 565.16 | 1.0122 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 600.82 | 600.78 | 1.0541 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 864.67 | 880.71 | 1.0286 | flat |

gained on sliding verify_32. Expected hit: 2/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: w2-marlin-atomic-add-t1-port`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

