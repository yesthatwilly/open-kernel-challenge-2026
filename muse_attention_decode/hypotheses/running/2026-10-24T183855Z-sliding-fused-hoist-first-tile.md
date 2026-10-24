---
problem: muse_attention_decode
slug: sliding-fused-hoist-first-tile
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, gather prologue)
change: >
  In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so
  the KV round trip overlaps the q load, RMSNorm and RoPE. Move the computation of lo, count,
  chunk, start, end (it needs only seq = seq_lens[b], already loaded from seq_ptr) to the top of
  the kernel, right after `d`/`freq`, then load tile 0: t0 = start + n, tmask0 = t0 < end, page0 =
  tl.load(pt_ptr + b * pt_stride + t0 // PAGE, mask=tmask0, other=0), and kb0 / vb0 with the same
  kvoff formula as the loop. Then the existing q load + _norm_rope + the `if s == S - 1` new-key
  block run as now (its stores go to cache_loc rows, which are positions >= seq and never in [lo,
  seq - 1], so the hoisted loads cannot read them). After it, process kb0/vb0 with the same qk,
  valid mask (t0 >= p_row - (WINDOW - 1)) and _online_update call as the loop body, then run the
  loop from start + BLOCK_N. The partial store, ticket and combine are unchanged; verify (q_len 8,
  not routed here today) is unaffected.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "every program today serializes q load -> norm/rope -> page-index load -> KV load; hoisting overlaps ~1-2 us of the 10.4 us kernel's dependency chain, and on split S-1 also the new-key store"
  - shapes: "*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *"
    direction: faster
    why: same chain in 512 / 1024 programs; smaller relative gain
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on the sliding decode path
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T184058Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T184059Z
base_sha: b0b3da52d52b15557f59e3c3631b16eb6fc5c2cd
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/sliding-fused-hoist-first-tile
worktree: .worktrees/muse_attention_decode/sliding-fused-hoist-first-tile
mode: check
implemented_at: 2026-10-24T184626Z
impl_commit: 06db73a29a01daaea8c8a656b9816a345e566fca
submission_sha256: 1a1712e56fdc8988cf9972cf01d60657386e64549a7fb68e61e6f9ab28a9a2bb
norm_sha256: beef4e52de0acece11918f87e1b845ed85cd2a4d326b046264f3e4590c5d996e
submission_blob: b7aab820247a0f90842e548756b6633dc3c7673d
lint_warnings: 0
attempt_file: 2026-10-24T184626Z-check-sliding-fused-hoist-first-tile.md
prepared_at: 2026-10-24T184626Z
---

# sliding-fused-hoist-first-tile

## Change

In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so the KV round trip overlaps the q load, RMSNorm and RoPE. Move the computation of lo, count, chunk, start, end (it needs only seq = seq_lens[b], already loaded from seq_ptr) to the top of the kernel, right after `d`/`freq`, then load tile 0: t0 = start + n, tmask0 = t0 < end, page0 = tl.load(pt_ptr + b * pt_stride + t0 // PAGE, mask=tmask0, other=0), and kb0 / vb0 with the same kvoff formula as the loop. Then the existing q load + _norm_rope + the `if s == S - 1` new-key block run as now (its stores go to cache_loc rows, which are positions >= seq and never in [lo, seq - 1], so the hoisted loads cannot read them). After it, process kb0/vb0 with the same qk, valid mask (t0 >= p_row - (WINDOW - 1)) and _online_update call as the loop body, then run the loop from start + BLOCK_N. The partial store, ticket and combine are unchanged; verify (q_len 8, not routed here today) is unaffected.

## Expected

- `*layer=sliding load=decode_1 *` faster: every program today serializes q load -> norm/rope -> page-index load -> KV load; hoisting overlaps ~1-2 us of the 10.4 us kernel's dependency chain, and on split S-1 also the new-key store
- `*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *` faster: same chain in 512 / 1024 programs; smaller relative gain
- `*layer=full* *load=verify_*` flat: not on the sliding decode path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Mechanism-overlapping with sliding-decode-blockn128-single-tile; if both run they combine trivially (at BLOCK_N 128 the hoisted tile is the whole chunk).

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a plausible, rules-clean latency-hiding change to the single kernel on the sliding decode shapes (S=16, BLOCK_N=64, so 2 tiles per program, and this hoists half of them); judge it on the _sliding_fused_kernel profile row rather than the headline, because the expected ~1.5% headline gain sits inside the CI.

### Justification
The mechanism holds up. Every sliding decode shape spends 100% of its time in _sliding_fused_kernel (10.4us at decode_1 with only 32 CTAs, 12.9us at decode_16, 22.1us at decode_32). At decode_1 the floor is 0.1us, so the kernel is pure latency: a serial chain of dependent DRAM round trips under a cold L2. That chain is q load -> norm/rope -> seq load -> page-index load -> KV load, x2 tiles, then partial store -> ticket -> .cg reloads -> gate load. With count = 2047 and S = 16, chunk = 128 = 2 BLOCK_N tiles. Hoisting tile 0's seq->page->KV loads ahead of the q work removes about one round trip plus the norm/rope from the critical path. On a ~10us kernel that is roughly 0.5-1.5us (5-12%) at decode_1, consistent with the 'faster' call. The gain shrinks at decode_16/32, where 512/1024 CTAs over ~132 SMs make occupancy and bandwidth hide more latency; 'faster but smaller' is right there, and flat is equally likely. 'Flat' for full and verify is correct: kernel_fn routes only window && q_len == 1 here. Headline arithmetic: on the decode_1 load, the sliding shape is 46*13.2 = 607us vs full 16*27.3 = 437us. A 1us kernel gain is about 4.4% of that load and about 0.7% of the 6-load geomean, plus small decode_16/32 contributions, so the headline lands inside the +/-1.5% CI and the reported 2-3.5% decode_1 batch drift. Weakest link: the argument assumes the compiled code really does serialize as the source does. ptxas can already float the independent seq load above the q math. Triton's num_stages=2 pipeliner already issues the loop's first-iteration loads in a prologue, which today sits after the q and new-key work. If ptxas already overlaps part of the chain, the gain is smaller than estimated. Also, holding kb0/vb0 (2 x 64x64 fp8, ~16 regs/thread at 4 warps) live across _norm_rope and the S-1 new-key dot/stores adds register pressure. Spills would eat the gain; check the compiled kernel's n_spills/n_regs.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: set S = 16 for every decode batch, which fixes tiles-per-program at 2. The hoisted tile's relative value depends directly on this (with fewer splits, more tiles per program, smaller relative gain), so the result is specific to S = 16 / BLOCK_N = 64. It is not a confound of attribution, though, since the split count is unchanged by this hypothesis.
- sliding-fused-triton-decode1-gate: introduced _sliding_fused_kernel and its fixed source ordering (q work before seq/page loads), the very thing being changed. No interaction beyond being the base.
- full-decode-cuda-gather-splitkv: does not touch the sliding path, but its run recorded a sliding decode_1 'loss' with no code change on that path. That is direct evidence of a 2-3.5% between-batch drift on exactly the shape this targets, so a headline-level sliding decode_1 move of that size is uninterpretable without the per-kernel row.

### Interference and A/B needs
No companion is warranted. The change is a single mechanism (instruction ordering) on one kernel, with no landed hypothesis whose revert or retune would change its sign. The split count only scales its size. The real ambiguity is noise, not interference, and the in-run SGLang pairing plus the profile's _sliding_fused_kernel row (10.4 / 12.9 / 22.1us today) resolve it better than a second run would. A num_stages or tile-count retune would test a different hypothesis (and sliding-decode-blockn128-single-tile already covers the tile-count axis). Read a >= 0.5us drop on the decode_1 kernel row as a gain and an unchanged or higher row as a null, regardless of the headline.

### Open questions
- The change text says seq is 'already loaded from seq_ptr' at the top, but in master `seq = tl.load(seq_ptr + b)` comes after the q load/norm/rope. The implementer must move the seq load up as well (and compute n there), or the hoist only partly overlaps.
- Does the compiled kernel spill once kb0/vb0 are live across _norm_rope and the S-1 new-key block? Compare n_regs/n_spills before and after; if it spills, reduce what stays live (for example, issue only the page and K loads early and V after the q work).
- Should the hoist cover both tiles of the 128-key chunk, not just tile 0? Today tile 1's loads still wait behind tile 0's dot/softmax unless the pipeliner prefetches them. This overlaps sliding-decode-blockn128-single-tile; if that one wins, this is subsumed.
- Scheduling note, not a defect: if sliding-cuda-cluster-dsmem-onelaunch (running) lands and replaces this kernel on every sliding shape at tp8/page 1, this becomes moot. Gate dispatch on that outcome.


## Implementation notes

In `_sliding_fused_kernel` (worktree muse_attention_decode/submission.py): moved `seq = tl.load(seq_ptr + b)` (per the review it was NOT at the top in master), lo/count/chunk/start/end and `n` to right after `freq`, then issued tile 0's page-index load and raw uint8 K/V gathers (kb0/vb0, same kvoff formula, mask `t0 < end`) before the q load. After the `s == S - 1` new-key block, tile 0 is processed under `if start < end:` with the loop body's exact qk, `valid = tmask0 & (t0 >= p_row - (WINDOW - 1))` and `_online_update`; the loop now runs `range(start + BLOCK_N, end, BLOCK_N)`. Deviation: the `start < end` guard (equivalent to the loop not running) avoids a spurious update on empty chunks. kb0/vb0 are kept as uint8 and bitcast to e4m3 at use. Spill risk from live kb0/vb0 is unmeasured (no GPU here); check n_regs/n_spills.

## Rules check

- [x] CUDA-graph capturable: diff adds only tl.load/dot code; no host syncs, allocations or value-dependent host control flow.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a, no weight or prepare() change.
- [x] Activations BF16, KV cache FP8, q bf16: unchanged (diff does not touch dtypes).
- [x] Attention sees every key: tile 0 uses the loop's window mask; loop covers the remaining tiles; no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network in the diff.
- [x] All work inside the timed call; no caching or replay.
- [x] No triton.autotune added.

Lint: `CLEAN: muse_attention_decode/submission.py`

## Pre-flight audit

Audited 2026-10-24 (opus). **Verdict: dispatch.** No fixes.

- Hoisted tile is the loop's first iteration verbatim (t0 = start + n, tmask0 = t0 < end, same page/row/kvoff, uint8 load then e4m3 bitcast, same qk, same `valid = tmask & (t >= p_row - (WINDOW-1))`, same _online_update); the loop resumes at start + BLOCK_N. Emulated S=16, BLOCK_N=64, WINDOW=2048 over seq 0..4200 (+2047/2048/2049/80000), L=1 and 8, every split: the set of scored keys per row is identical to master's, including partial windows and partial chunks.
- start >= end (short contexts, e.g. ctx 10 where splits 1..15 are empty): the hoisted loads are fully masked (no OOB address used), `if start < end` skips the update, and the loop range is empty, matching master's zero-iteration loop.
- Ordering: the hoisted reads hit page_table rows of positions [lo, seq-1]; the S-1 program's new-key stores go to cache_loc rows (positions >= seq), so moving the reads ahead of the store changes nothing.
- Triton scoping: kb/vb/qk/valid defined only inside the `if` are not used after it (the loop rebinds them), the same pattern master already uses for the s == S-1 block; m_i/l_i/acc keep their types across the branch.
- Fallback: no new gate. As in master, a Triton compile error in _sliding_fused_kernel would fail the run (no incumbent fallback exists for this kernel); the edit uses only constructs already present in the kernel.
- Rules: unchanged from master (no syncs, allocation, caching or precision change; every window key scored). okc_lint CLEAN, py_compile OK.

Residual risk: kb0/vb0 (2 x 64x64 uint8) live across the q norm/RoPE may raise registers/spills at num_warps=4 and eat the 1-2 us overlap; check n_spills and the _sliding_fused_kernel row.
