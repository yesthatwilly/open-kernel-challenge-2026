---
problem: muse_moe_w4a16
slug: marlin-block-size-m-next-larger
created: 2026-10-23T225502Z
base: master
area: MoE align/dispatch
change: >
  Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in
  (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, which yields 8 at bs=1/8/16, 16 at
  bs=32, 48 at bs=128 and 64 at bs=256. Replace it with the next larger candidate (cap 64): 16 at
  bs=1/8/16, 32 at bs=32, 64 at bs=128, 64 at bs=256 (unchanged, a control). The same block size
  feeds moe_align_block_size and the moe_block_size kwarg of both Marlin calls; nothing else
  changes. Mechanism: Marlin MoE re-reads an expert's whole NVFP4 weight for every m-block of that
  expert, so fewer, larger m-blocks mean fewer weight passes (L2 is cold at every replay, so a
  second block re-reads from HBM), at the cost of more padded rows, whose compute is free while
  the kernel is memory-bound.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "bs=32: ~256 routes over ~13 touched experts, ~20 per expert: block 16 gives 2 weight passes for most experts, block 32 gives 1"
  - shapes: bs=128 *
    direction: faster
    why: "bs=128: block 48 -> 64 cuts m-blocks for hot experts (skewed popularity puts 50-100 routes on the hottest)"
  - shapes: bs=16 *
    direction: flat
    why: "bs=16: ~9 touched experts with ~2 routes each, one block either way"
  - shapes: bs=1 *
    direction: flat
    why: "bs=1: one block per touched expert at either size; padding is compute only"
  - shapes: bs=256 *
    direction: flat
    why: "control: 64 before and after"
risk: low
depends_on: sglang-marlin-path-parity
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T024434Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T024440Z
base_sha: 6b5469bd92f0a43ce5908912b02fb8dee425c3fa
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/marlin-block-size-m-next-larger
worktree: .worktrees/muse_moe_w4a16/marlin-block-size-m-next-larger
mode: check
implemented_at: 2026-10-24T024739Z
impl_commit: 82f0d6b82a41b67d6bb345cf96bfe9395893dcb4
submission_sha256: 75ee78230e1088f981d58c67e6c3a17fa9b6577092aa7267a46640c7eaf82856
norm_sha256: 720215ee0bb4a27760957d101e8325ead34b55b3039f9444514b862e30b29a82
submission_blob: a0ffc54f6d1673a2c83389daf9ea103c36350490
lint_warnings: 1
attempt_file: 2026-10-24T024739Z-check-marlin-block-size-m-next-larger.md
prepared_at: 2026-10-24T024740Z
job: 2639e3ca7f5549de
dispatched_at: 2026-10-24T024747Z
outcome: regression
verdict_detail: regression
speedup: 0.917841731023605
speedup_ci: [0.9040741050582509, 0.931609356988959]
harvested_at: 2026-10-24T035910Z
best_at_harvest: 0.9887547325362586
promotion: null
---

# marlin-block-size-m-next-larger

## Change

Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, which yields 8 at bs=1/8/16, 16 at bs=32, 48 at bs=128 and 64 at bs=256. Replace it with the next larger candidate (cap 64): 16 at bs=1/8/16, 32 at bs=32, 64 at bs=128, 64 at bs=256 (unchanged, a control). The same block size feeds moe_align_block_size and the moe_block_size kwarg of both Marlin calls; nothing else changes. Mechanism: Marlin MoE re-reads an expert's whole NVFP4 weight for every m-block of that expert, so fewer, larger m-blocks mean fewer weight passes (L2 is cold at every replay, so a second block re-reads from HBM), at the cost of more padded rows, whose compute is free while the kernel is memory-bound.

## Expected

- `bs=32 *` faster: bs=32: ~256 routes over ~13 touched experts, ~20 per expert: block 16 gives 2 weight passes for most experts, block 32 gives 1
- `bs=128 *` faster: bs=128: block 48 -> 64 cuts m-blocks for hot experts (skewed popularity puts 50-100 routes on the hottest)
- `bs=16 *` flat: bs=16: ~9 touched experts with ~2 routes each, one block either way
- `bs=1 *` flat: bs=1: one block per touched expert at either size; padding is compute only
- `bs=256 *` flat: control: 64 before and after

## Risk

Risks: Marlin's thread_m_blocks grows with moe_block_size, so a bigger block changes the kernel config (tile shape, smem) and could lose occupancy at small M; bs=1/8 could regress slightly from doubled A-tile padding. Pure dispatch knob: no correctness or rule exposure. If this is a gain, the obvious follow-up ab-tune is 64 everywhere.

## Adversarial review

Reviewed 2026-10-24T024434Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Cheap, low-risk, rule-clean tuning knob whose single run is interpretable per shape because the only code change is the block_size_m pick and the SGLang column is the built-in control; but the stated mechanism (fewer weight passes on a memory-bound kernel) does not hold for the scored shapes, so a gain or loss must be read as 'Marlin's next thread_m_blocks config wins/loses at this M', not as 'fewer HBM passes'. No companions needed.

### Justification
In the parity run (fb2cc736613747df; FINDINGS' profile block is the stale starter) Marlin is 53% of bs=1, 71% bs=8, 75% bs=16, 92% bs=32, 96% bs=128/256. The mechanism breaks: route counts are miscounted 8x (only this GPU's 32 experts are computed: ~2-3 routes per expert at bs=32, so block 16 is already one m-block; at bs=128 48->64 collapses a second block only for rare experts with 49-64 routes). The kernel is not memory-bound (~1.3 TB/s at bs=128, a quarter of HBM). What the change really does is select Marlin's thread_m_blocks (16->32 is 1->2, 48->64 is 3->4), a tile-shape/occupancy change; bs=1/8/16 go from the 8-row special path to the 16-row path, more likely a small loss than flat. Padded rows cost a little in align and reduce scratch. Weakest link: the premise of multi-block experts on a bandwidth-bound kernel; neither holds, so this is an empirical probe of the next-larger Marlin m-tile config, still worth one run.

### Confounders on the current kernel
- sglang-marlin-path-parity (align with num_experts=32): keeps sorted_ids <= 4096 for every new block size so no fused-padding fill_ launch appears; upstream 256-expert align would add ~2 us at bs=32/128. Inside the CI.
- sglang-marlin-path-parity: the small-batch align kernel's padding work grows with block size, so a few-us decode loss can come from align rather than Marlin; the profile separates them.

### Interference and A/B needs
ab-revert means the 0.006x starter, meaningless; ab-tune (256-expert align) only re-adds a ~2 us launch. Isolating 'fewer m-blocks' from 'different thread_m_blocks' is impossible via the public API and unnecessary: per shape a gain means keep the next-larger block, a loss means keep SGLang's pick. No companions.

### Open questions
- Read per shape: bs=1/8/16 may lose while bs=32 or bs=128 gains; implement the pick as a per-shape table so partial adoption is a one-line follow-up.
- Expected list rests on an 8x miscount; honest expectation is 'unknown, config-dependent' at bs=32/128 and 'flat to slightly slower' at bs=1/8/16.
- Record the Marlin kernel template name per shape (thread_m_blocks) from the profile to confirm which config ran.
- A skewed-test failure at bs=128 would point at thread_m_blocks=4 smem, not the pick.
- If bs=128 wins with 64, stack marlin-workspace-blocks-per-sm-8 explicitly on it rather than running both against the same base.
- Do not fold the '64 everywhere' ab-tune into this run.


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/marlin-block-size-m-next-larger`, `muse_moe_w4a16/submission.py`, uncommitted, on base 6b5469bd92 (diff stat: 1 file, +8 -1).

- SGLang's selection loop is kept verbatim except that it now enumerates, so it yields SGLang's index `idx` (first candidate with `num_tokens*8/32/block_size_m < 0.9`, else the last). The block size actually used is then `BLOCK_SIZE_M_CANDIDATES[min(idx + BLOCK_SIZE_M_STEP(num_tokens), 4)]`.
- `BLOCK_SIZE_M_STEP = lambda num_tokens: 1` is a new module constant beside the candidates. Adopting part of the change per shape is a one-line edit there (e.g. `lambda num_tokens: 0 if num_tokens < 32 else 1`), as the review asked.
- Picks checked offline, SGLang -> new: bs=1/4/8/16: 8 -> 16; bs=32/33: 16 -> 32; bs=128: 48 -> 64; bs=256/257: 64 -> 64 (the control). These match the `change` paragraph.
- The same `block_size_m` still feeds `moe_align_block_size` (over the bank's 32 experts, as on master) and the `moe_block_size` kwarg of both Marlin calls. Align, workspace, caches, and everything else are unchanged. There is no other tuning, and the "64 everywhere" ab-tune is not folded in.
- For the review's open question, the profile should record the Marlin template (thread_m_blocks) for each shape, to confirm which config ran.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-next-larger diff`:

- [x] CUDA-graph capturable: the pick is pure Python on `num_tokens` (a shape) and a constant. There is no `.item()`/`.cpu()`/sync, and no control flow on tensor values.
- [x] Nothing derived from `data` is cached across calls: no new state.
- [x] Weights: n/a. prepare() is untouched, and NVFP4 stays NVFP4 via the same Marlin repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: n/a for KV/queries (MoE problem). Activation dtypes are unchanged.
- [x] Attention sees every key: n/a (no attention in this problem). The diff skips no routes or experts. Padding only grows.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept. `python3 -m py_compile` passes.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`), rc=0:

```
warn  host-sync    L113: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

The one warning was already on master: it is in prepare(), not this diff, so there was nothing to fix.

## Result

- job `2639e3ca7f5549de` (check), status done, passed True, tests 5/5
- speedup 0.9178x, 95% CI [0.9041, 0.9316], score 14652.30828522236 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/marlin-block-size-m-next-larger.result.json`, `muse_moe_w4a16/hypotheses/done/regression/marlin-block-size-m-next-larger.patch`; branch `hyp/muse_moe_w4a16/marlin-block-size-m-next-larger`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 80.23 | 85.76 | 0.9805 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 191.41 | 192.33 | 0.984 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 208.15 | 261.46 | 0.7948 | lost |
| bs=8 load=verify_1 sharding=ep8 | 60 | 152.04 | 155.61 | 0.9701 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 462.46 | 545.16 | 0.8596 | lost |
| bs=256 load=verify_32 sharding=ep8 | 60 | 621.28 | 617.78 | 1.0005 | flat |

lost on decode_32 decode_32; verify_16 verify_16. Expected hit: 3/5. Noise ±1.5%.

## Follow-ups

- record which shapes lost (Attribution) and do not retry this area in this form.

