---
problem: muse_attention_decode
slug: sliding-fused-triton-decode1-splits-32
created: 2026-10-24T140241Z
base: master
area: sliding-layer attention (batch-1 decode, split depth)
change: >
  Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only
  shape master routes to _sliding_fused_kernel) use S = 32 splits instead of _num_splits(1, 2) =
  16, so each program owns ceil(2047/32) rounded up to 64 = 64 old keys, exactly one BLOCK_N=64
  tile, and the grid is (1, 2, 32) = 64 CTAs instead of 32 CTAs of two serial tiles. To keep the
  combine compact, the two tl.static_range(S) loops in the last-ticket combine may stay static (32
  iterations) or become a runtime range(S) when S > 16 behind a constexpr branch; the S <= 16
  instantiation must compile unchanged. Workspace and counters already scale with S from config
  (ws_acc 64 x 16 x 64 fp32 = 256 KB). Leave _num_splits for other batches, BLOCK_N, num_warps=4,
  num_stages=2 and everything else as master.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "incumbent 13.12 us shape / 10.5 us kernel against a 0.1 us floor (512 KB); the measured cost of a program is ~1.5-1.8 us per dependent page-table+gather tile (sliding-single-launch-fused-attention: 2/8/16 tiles -> 13.5/23.9/38.5 us), so one tile per program should save ~1.5 us, partly paid back by a 32-way instead of 16-way combine (+~0.3 us); sliding decode_1 is 58% of the decode_1 load at weight 46, so ~1.2 us is ~5% on that load"
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
  - shapes: "*load=decode_16* *load=decode_32* *load=verify_*"
    direction: flat
    why: outside the batch-1 decode gate
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164523Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T164753Z
base_sha: aef5220ede19f282df557fb2e7493a375aeb6107
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/sliding-fused-triton-decode1-splits-32
worktree: .worktrees/muse_attention_decode/sliding-fused-triton-decode1-splits-32
mode: null
implemented_at: 2026-10-24T164945Z
impl_commit: 5c273c610329d018bc24efcc0b7258b86916c1ab
submission_sha256: 94189d78786b4af9b506718e173eb13b577e86274a8f53604e9cfabb34da33ab
norm_sha256: d8d704215ab9869a714a18e9b4f9c5af8759dfba687cb4d89a5df00518dbf126
submission_blob: 12abfe018e63baaf449e4c4e21afb7b93bcdb6f2
lint_warnings: 0
---

# sliding-fused-triton-decode1-splits-32

## Change

Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits instead of _num_splits(1, 2) = 16, so each program owns ceil(2047/32) rounded up to 64 = 64 old keys, exactly one BLOCK_N=64 tile, and the grid is (1, 2, 32) = 64 CTAs instead of 32 CTAs of two serial tiles. To keep the combine compact, the two tl.static_range(S) loops in the last-ticket combine may stay static (32 iterations) or become a runtime range(S) when S > 16 behind a constexpr branch; the S <= 16 instantiation must compile unchanged. Workspace and counters already scale with S from config (ws_acc 64 x 16 x 64 fp32 = 256 KB). Leave _num_splits for other batches, BLOCK_N, num_warps=4, num_stages=2 and everything else as master.

## Expected

- `*layer=sliding load=decode_1*` faster: incumbent 13.12 us shape / 10.5 us kernel against a 0.1 us floor (512 KB); the measured cost of a program is ~1.5-1.8 us per dependent page-table+gather tile (sliding-single-launch-fused-attention: 2/8/16 tiles -> 13.5/23.9/38.5 us), so one tile per program should save ~1.5 us, partly paid back by a 32-way instead of 16-way combine (+~0.3 us); sliding decode_1 is 58% of the decode_1 load at weight 46, so ~1.2 us is ~5% on that load
- `*layer=full*` flat: untouched
- `*load=decode_16* *load=decode_32* *load=verify_*` flat: outside the batch-1 decode gate

## Risk

Cheap single-knob probe of split depth on the one shape the fused kernel owns today; it also calibrates how far deep splitting goes before the combine and the atomics cost more than a tile, which informs sliding-fused-triton-decode-batched-deep-splits (B>1, S=16; disjoint shapes, the two can be 3-way merged). Risk: 64 partials of fresh tiles may be dominated by the fixed seq/page-table/gather/ticket latency chain, so the gain may be under 1 us (within ~1.5% per-shape drift); read the kernel row in the profile.

## Adversarial review

Reviewed 2026-10-24T164523Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-knob, config-only probe (S 16 -> 32 at sliding batch-1 decode) on the one shape master routes to _sliding_fused_kernel. The arithmetic checks out: chunk = ceil(2047/32) rounded to 64 = 64, so there is one tile per program and 32*64 >= 2047. Correctness risk is low because the -inf-safe combine and empty-chunk handling already exist. The expected headline effect (~0.8%) is below run drift, so judge it on the kernel row, not the headline.

### Justification
Mechanism: at B=1 the fused kernel is latency-bound, not bandwidth-bound. It moves 512 KB in 10.3 us (~0.05 TB/s, floor 0.1 us) on 32 CTAs of 132 SMs. Each program runs a dependent chain: q load, norm/rope, then per tile a page-table load, a dependent K/V gather, the dot and the online update, then the partial store, fence, ticket and combine. Halving the per-program tile chain from 2 to 1 dependent page-table+gather step should save roughly one gather latency (~1-1.5 us). Doubling CTAs to 64 still underfills the GPU, so there is no occupancy downside. FINDINGS supports this: the shape is 13.9 us with _sliding_fused_kernel 10.3 us at 100% of the kernel time, and ~3.6 us of the shape is graph/launch overhead that this change cannot touch. The expected directions match TASK.md: sliding decode_1 has weight 46, and the decode_1 load is ~46*13.9 + 16*27.9 ~= 1085 us. Saving ~1.2 us is ~5% on that load and ~0.8% on the 6-load geomean, inside the >=1.5% CI and inside observed drift. Byte-identical sliding decode_1 rows were 13.12, 13.22 and 13.88 us across three runs. The full / batch>1 / verify 'flat' expectations are correct, because the gate is cfg batch==1 and q_len==1 only. WEAKEST LINK: the '~1.5-1.8 us per tile' cost comes from sliding-single-launch-fused-attention's 2/8/16 tiles -> 13.5/23.9/38.5 us. Those three points are three different batches (1/16/32), CTA counts (32/128/128) and byte volumes (0.5/8/16 MB), not a within-shape sweep. With num_stages=2, Triton's pipeliner may already overlap the second tile's page-table/gather with the first tile's math, so the real saving may be well under 1.5 us. The second-weakest link is the combine cost. The last-ticket CTA alone serially reads 32 partials of [16,64] fp32 acc plus m/l (~135 KB vs ~68 KB at S=16; half of it is padding rows 8..15, which are stored and reloaded unmasked) through L2 with .cg. On a single SM that is plausibly +0.3-0.7 us of serial tail, not the stated +0.3 us. Expect -0.3 to -1.2 us on the kernel row. A null result is plausible and still informative: it says split depth is settled at B=1.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate (landed): this is the kernel and gate being tuned. It is the parent's own code, not a confounder. Because the comparison is against master itself (S=16), the delta is attributable to S alone.
- full-decode-cuda-gather-splitkv (landed, current master): it does not touch the sliding path. It does load a load_inline CUDA module and run a self-check at import. Sliding decode_1 read 13.88 us under it vs 13.12 under the previous master with an unchanged kernel (10.5 -> 10.3 us kernel), so the shape row on this base already carries ~0.7 us of non-kernel drift. Compare kernel rows (_sliding_fused_kernel us) to master's 10.3 us, not shape rows to older runs.
- fused-preattention-triton / paired-heads-* / fa3-hdim128-head-pairing (landed): no interaction. Batch-1 sliding decode bypasses _pre_attention_kernel and FA3 entirely on master.

### Interference and A/B needs
No companion is warranted. The run's own parent is master, which already runs the identical kernel at S=16, so this single run is the A/B of S=32 vs S=16 on one shape. No landed hypothesis's tuning changes what S does on this path. A control isolating a sub-change has nothing to isolate: the change is one constexpr (the optional runtime-range combine loop is an implementation detail, and keeping tl.static_range(32) is simpler and fine). Read the outcome from _sliding_fused_kernel's profile row against master's 10.3 us. Triton rows reproduce within ~1%, so a kernel delta of >=0.3 us is resolvable even though the headline is not. If it is flat or slower, the likely cause is the serial 32-way combine. A follow-up (not a companion) could mask the combine's workspace stores/loads to rvalid rows (halving the combine bytes) or move to a tree/parallel combine. Coordination note: the queued sibling sliding-deep-splits-on-cuda-splitkv (p1) edits the same `splits =` line and widens the gate to every decode batch with 16 splits. Write this change as an explicit override, e.g. `splits = 32 if batch == 1 and length == 1 else _num_splits(batch, hkv)`, so that the two still 3-way merge, and so that the sibling's 'S=16 for decode' does not silently undo S=32 at B=1 or vice versa.

### Open questions
- Does Triton's num_stages=2 pipeliner already overlap the 2nd tile's page-table load and gather at S=16? If so, the saving per removed tile is far less than the 1.5-1.8 us inferred from cross-batch data. The kernel row answers this.
- Is the last-ticket combine (one CTA, serial 32 partials, ~135 KB incl. padding rows) now the dominant tail? If S=32 is flat, consider masking partial stores/loads to rvalid (8 of 16 rows) before trying any deeper split.
- The sliding decode_1 shape row is ~3.6 us above the kernel (13.9 vs 10.3 us). That overhead (workspace allocs, graph node cost) is untouched here and may be a bigger lever than split depth.


## Implementation notes

- One line in `_sliding_forward` (worktree `muse_attention_decode/submission.py`, ~line 355):
  `splits = _num_splits(batch, hkv)` became
  `splits = 32 if batch == 1 and length == 1 else _num_splits(batch, hkv)`, with a one-line comment.
  It is written as an explicit batch-1/q_len-1 override, per the review coordination note, so it
  3-way merges with sliding-deep-splits-on-cuda-splitkv. The override branches on config only.
- Kernel unchanged. Both combine loops stay `tl.static_range(S)` (32 unrolled iterations at
  S=32). The review calls this "simpler and fine", and it leaves the S<=16 instantiation
  byte-identical. No constexpr runtime-range branch was added.
- Workspace (`ws_acc` [64, 16, 64] fp32 = 256 KB, `ws_ml` [64, 2, 16]) and the counters
  ([B*Hkv], unchanged) already scale with `splits`. `_num_splits`, BLOCK_N=64, num_warps=4 and
  num_stages=2 are all unchanged.
- Edge case checked by reading: for short contexts (test "batch 1, ctx 10, sliding"),
  chunk = 64 and every split s >= 1 gets start >= seq, so its loop is empty and its partial is
  m=-inf, l=0, acc=0. Split S-1 holds the new keys, so it is finite, and the combine weights
  -inf partials 0. This is the same path S=16 already took.
- Deviation from the spec: none.
- Read the result from the `_sliding_fused_kernel` profile row against master's 10.3 us, not
  from the headline.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-fused-triton-decode1-splits-32 diff`
(1 file, +2/-1, `_sliding_forward` only):

- [x] CUDA-graph capturable: the new branch reads `batch`/`length` from `cfg` (config), not input
      values. It adds no `.item()`/`.cpu()`/`synchronize()`/prints. Allocations are shape-only
      (`torch.empty` sized from config), as before.
- [x] Nothing derived from `data` is cached across calls. The only cache is `_COUNTERS`, keyed by
      (batch, hkv, device), and it holds no data (it is reset in-kernel each launch). The diff
      leaves it unchanged.
- [x] n/a: weights. This problem has no weights, and the diff does not touch prepare or any
      weight handling.
- [x] Activations stay BF16 where SGLang keeps BF16. The KV cache is still stored as E4M3.
      Queries are still bf16 operands (finer than E4M3). The kernel is unchanged.
- [x] Attention sees every key. 32 * 64 = 2048 >= 2047 old keys, and the chunks tile
      [lo, seq-1] contiguously. The new keys are still attended by split S-1. Nothing is skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses
      or network. The diff adds none of these.
- [x] All work happens inside the timed call. There is no output caching or replay.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. There is no `triton.autotune`.

Lint (`okc_lint.py ... --problem muse_attention_decode`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/sliding-fused-triton-decode1-splits-32/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK. The `__pycache__` it left was removed from the worktree.
