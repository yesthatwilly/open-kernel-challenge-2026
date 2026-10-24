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
status: queued
review_at: 2026-10-24T024434Z
review_companions: null
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

