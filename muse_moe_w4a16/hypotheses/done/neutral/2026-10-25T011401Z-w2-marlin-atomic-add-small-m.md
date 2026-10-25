---
problem: muse_moe_w4a16
slug: w2-marlin-atomic-add-small-m
created: 2026-10-25T011401Z
base: master
area: w2 marlin
change: >
  For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and
  use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2)
  only, by building a separate kwargs dict from master's common for that call; the w13 call and
  every num_tokens > 32 call keep master's common exactly. Reason: w13-marlin-atomic-add-small-m
  tried this on w13 only and was neutral, but w2 is the other shape: size_n = H = 4096 (half of
  w13's 8192) at the same K = I = 4096, so Marlin has half as many N tiles to spread over 132 SMs
  and slices K more, i.e. more of w2's time is the K-slice fp32 global reduce that atomic add
  replaces. y (cache3) is then accumulated in place, so make sure cache3 is zeroed where Marlin
  expects it (Marlin's moe path with use_atomic_add zeroes or requires a zeroed C; read sglang's
  moe_wna16_marlin_gemm wrapper and zero cache3 with one .zero_() only if the wrapper does not,
  inside kernel_fn, graph-safe). Self-check in prepare: at T in {1, 4, 8, 16, 32} compare the
  final output against master's path (rel-L2 <= 5e-3, finite), else keep master and print
  '[w2-atomic] PATH CHOSEN = atomic|master (<reason>)'.
expected:
  - shapes: bs=1 *
    direction: faster
    why: Marlin pair 44.7 us at bs=1 (best profile); w2 is about 1/3 of the bytes; a removed reduce pass is worth a few us if K-slicing is active
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair 114.0 us
  - shapes: bs=16 *
    direction: faster
    why: Marlin pair 176.1 us
  - shapes: bs=32 *
    direction: faster
    why: Marlin pair 207.5 us; decode_32 is the weakest load
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T011915Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T011915Z
base_sha: 0ad8ffa797c1d65f94a31d3ee5951c1a36987746
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m
worktree: .worktrees/muse_moe_w4a16/w2-marlin-atomic-add-small-m
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T012018Z
impl_commit: 6960c4c3e43ee1e20f6920f6e94949f1ebb09793
submission_sha256: 121fdb1d41669ab68551399843439d591ec20709eec731ca8288e779cf6fcf98
norm_sha256: cdac564d58950b4c0c15329c436088da4fcf19ae5c8ed13f972448ba0b7f5630
submission_blob: 60dfe35d5417d9c0f89067ff313efdc2c41d03a2
lint_warnings: 11
attempt_file: 2026-10-25T012018Z-check-w2-marlin-atomic-add-small-m.md
prepared_at: 2026-10-25T012018Z
job: a2744bc496b84790
dispatched_at: 2026-10-25T012041Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.344525539017997
speedup_ci: [1.324357655932727, 1.364693422103267]
harvested_at: 2026-10-25T034137Z
best_at_harvest: 1.3278240305297715
promotion: merged
hand_landed: true
promotion_commit: 9080a71e9183df77bc71f28ecfa0f635b7d0aff0
---

# w2-marlin-atomic-add-small-m

## Change

For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2) only, by building a separate kwargs dict from master's common for that call; the w13 call and every num_tokens > 32 call keep master's common exactly. Reason: w13-marlin-atomic-add-small-m tried this on w13 only and was neutral, but w2 is the other shape: size_n = H = 4096 (half of w13's 8192) at the same K = I = 4096, so Marlin has half as many N tiles to spread over 132 SMs and slices K more, i.e. more of w2's time is the K-slice fp32 global reduce that atomic add replaces. y (cache3) is then accumulated in place, so make sure cache3 is zeroed where Marlin expects it (Marlin's moe path with use_atomic_add zeroes or requires a zeroed C; read sglang's moe_wna16_marlin_gemm wrapper and zero cache3 with one .zero_() only if the wrapper does not, inside kernel_fn, graph-safe). Self-check in prepare: at T in {1, 4, 8, 16, 32} compare the final output against master's path (rel-L2 <= 5e-3, finite), else keep master and print '[w2-atomic] PATH CHOSEN = atomic|master (<reason>)'.

## Expected

- `bs=1 *` faster: Marlin pair 44.7 us at bs=1 (best profile); w2 is about 1/3 of the bytes; a removed reduce pass is worth a few us if K-slicing is active
- `bs=8 *` faster: Marlin pair 114.0 us
- `bs=16 *` faster: Marlin pair 176.1 us
- `bs=32 *` faster: Marlin pair 207.5 us; decode_32 is the weakest load
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

LESSONS lists use_atomic_add as a knob that did nothing, but that was measured on w13 only; this is the untested w2 half. If a zero-fill launch is needed it costs ~1-2 us and may cancel the gain; read the Marlin row and the memset row.

## Adversarial review

Reviewed 2026-10-25T011915Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. This is a single-flag change on an untested call (w2), and its main effect lands at bs=1, where the per-shape median is tight (err about 0.5 us on 48.8 us), so a win or a loss of 1-2 us can be read. Two premises need fixing: keep use_fp32_reduce=True, and keep the cache13 aliasing so the self-check actually tests the zeroing.

### Justification
Mechanism: w2 has size_n = H = 4096 and size_k = I = 4096 with top_k=1. That gives half as many N tiles as w13 (8192) over the same K. At bs=1 under ep8 only about 1 local route is active, so Marlin's flattened (m-blocks x n-tiles x k-tiles) space over 528 resident CTAs (workspace 4 per SM) splits each n-tile's K across more slices than w13 does. The lock-ordered fp32 reduce chain is therefore deeper, and that chain is what atomic add collapses. This is plausible, and it is genuinely the untested half: w13-marlin-atomic-add-small-m changed only w13, on the old parity base. WEAKEST LINK: the strongest prior is against it. The w13 run's Marlin row at bs=1 was 44.4 us against the base's ~41-45, a true zero at the shape with the deepest slicing. vLLM/SGLang's dense heuristic (should_use_atomic_add_reduce) enables atomic add only for n < 2048 and k >= 2048. w2's n = 4096 is still outside that range, so the upstream authors' own measurements say global reduce wins at this n. Expected directions: faster at bs=1 is the only one with real support. At bs=8/16/32, slice_count is about 1-2, and at slice_count 1 the reduce is skipped entirely, so expect flat there, not faster. If a cache3.zero_() is needed, it costs a ~1-1.5 us memset at every gated shape (2 MB at bs=32), which can turn bs=8-32 slightly slower. Readability: the Marlin profile row aggregates both launches and moves about ±20% between runs of identical code (bs=16 read 114-176 us on the same code), so the brief's 'read the Marlin row' cannot attribute a few us. Judge on per-shape medians instead: decode_1's err is about 0.5 us, so a 1-2 us change there is z 2-4. Read the memset row for the zeroing cost. At the headline, a 3 us gain at bs=1 alone is about 1% geomean, inside the CI, so the per-shape median is the read.

### Confounders on the current kernel
- sglang-marlin-path-parity: it sets the shared workspace (max_blocks_per_sm=4 -> 528 CTAs), which sets slice_count and therefore the size of any atomic-add effect. This is fixed on master, so it does not confound this run, but a result here does not transfer to a different workspace.
- post-expert-norm-triton-small-m: at bs<32 it consumes y (= cache3) via y.contiguous() and indexes by local_ids. Rows of remote routes are never written by Marlin in either mode, so atomic add changes nothing there. But if the implementer zeroes cache3, those rows become 0 instead of stale cache1 data. That is harmless only if the combine kernel masks remote slots, which it should already do, and the self-check confirms it.
- triton-map-align-on-block-m-restack: its sorted_ids/n_padded padding (BSM 8 at bs<=16, 16 at bs=32) sets the m-block count and thus the slicing depth. It is unchanged here, so the comparison is clean, but a later block_size_m change at small M would shift this flag's value.

### Interference and A/B needs
No companion. The change is one flag on one call, plus at most one memset that has its own profile row. A gain at bs=1 is attributable to the w2 atomic reduce, and a flat or loss result retires the knob for both Marlin calls. A loss is unambiguous because the memset row says how much of it is the zeroing. An 'atomic on w13 and w2' stack is pointless while w13 alone measured zero. An ab-tune of the workspace (blocks per SM) would be informative, since deeper slicing means more for atomics to save, but it belongs in a follow-up only if this run gains at bs=1.

### Open questions
- Spec correction: do NOT set use_fp32_reduce=False. The w13 implementation established that the pair is not required: vLLM's MoE runner passes atomic add with use_fp32_reduce=True, and c_tmp is allocated only when fp32_reduce && !atomic. Keeping True also keeps this the same knob that w13 tested. Flip it only if the binding rejects the combination, and log the error text.
- Is the zero_() needed? cache3 aliases cache13, which holds w13's output (cache1) at the time w2 runs, so the buffer is guaranteed non-zero. Keep that aliasing: it makes the prepare self-check a true test of whether Marlin's atomic path pre-zeroes (or plain-stores slice 0) or needs an explicit zero. Do not give cache3 a fresh buffer, because a freshly cudaMalloc'd buffer may read as zero and hide a missing zero_().
- What is the actual slice_count for w2 at bs=1 (it depends on Marlin's thread_n/thread_k pick for m_block 8)? Printing it once from the bank shapes would show whether a 2-5 us win is even possible.
- Does profile.txt list the two Marlin launches separately? If yes, read w2's own row. If not, rely on the per-shape median and treat the aggregated Marlin row as noise.


## Implementation notes

- Worktree submission.py only. kernel_fn is now a thin wrapper over `_forward(data, state, w2_mode)`; the body is unchanged except the w2 call, which uses `w2_common = dict(common, use_atomic_add=True)` when `w2_mode != "master"` and num_tokens <= 32 (`W2_ATOMIC_MAX_TOKENS`). w13 and every T > 32 call keep master's `common` exactly.
- Review corrections applied: use_fp32_reduce stays True; cache3 still aliases cache13 (holds w13's output, so non-zero); prepare() prints atomic enabled / memset in path.
- Deviation: sglang source is not readable on this node, so zeroing is decided empirically. `_w2_atomic_self_check` (prepare, synthetic seeded inputs only) tries "atomic" (no memset) at T in {1,4,8,16,32}, then "atomic_zero" (one `cache3.zero_()` before w2, graph-safe) if that fails, else "master". Gate: finite and rel-L2 vs master <= 5e-3. Binding exceptions are caught and logged. Chosen mode is stored in state["w2_mode"] (set in prepare, not derived from inputs).
- Prints: `[w2-atomic] PATH CHOSEN = atomic|master (<reason>)` and a line with atomic enabled, fp32_reduce=True, memset in path.
- Not run (no GPU). slice_count printing was skipped (needs Marlin internals).

## Rules check

- [x] CUDA-graph capturable: diff adds no sync/.item()/.cpu() in kernel_fn path; branch on num_tokens and state string only; `zero_()` is a graph-safe kernel. Syncs/.item() are only in prepare().
- [x] Nothing derived from `data` is cached: only w2_mode from prepare's synthetic inputs.
- [x] Weights: untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 as before; unchanged.
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network in diff.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all in prepare(), outside kernel_fn's call graph). py_compile OK; __pycache__ removed.

## Result

- job `a2744bc496b84790` (check), status done, passed True, tests 5/5
- speedup 1.3445x, 95% CI [1.3244, 1.3647], score 10002.413200587025 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.patch`; branch `hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.82 | 44.88 | 1.0903 | gained |
| bs=16 load=decode_16 sharding=ep8 | 60 | 148.37 | 147.71 | 1.0044 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.88 | 198.72 | 1.0009 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.74 | 106.03 | 1.0014 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 318.26 | 317.77 | 1.0026 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 454.28 | 467.8 | 0.9924 | flat |

gained on decode_1 decode_1. Expected hit: 3/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: w2-marlin-atomic-add-small-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

