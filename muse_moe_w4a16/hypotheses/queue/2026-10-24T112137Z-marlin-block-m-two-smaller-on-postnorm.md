---
problem: muse_moe_w4a16
slug: marlin-block-m-two-smaller-on-postnorm
created: 2026-10-24T112137Z
base: master
area: w13 marlin
change: >
  Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the
  pre-post-norm base, job ed837a4a46d44a95; patch at
  muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current
  master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection
  loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with
  num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it
  stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx -
  2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32
  (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the
  in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master;
  bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at
  SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's
  fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul,
  the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no
  sync, nothing input-dependent.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9"
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
  - shapes: bs=1 *
    direction: flat
    why: "control: unchanged pick (8) and unchanged Triton post-norm"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: "control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-24T112356Z
review_companions: null
hold_reason: "reviewer: dispatch after marlin-block-m-cap-16-port (1c33392914594026) harvests; if its bs=256 row beats two-smaller's by >~3%, use block 16 at bs=256 (an ab-tune or edit before claim), then unhold"
---

# marlin-block-m-two-smaller-on-postnorm

## Change

Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master; bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul, the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no sync, nothing input-dependent.

## Expected

- `bs=128 *` faster: same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9
- `bs=256 *` faster: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
- `bs=1 *` flat: control: unchanged pick (8) and unchanged Triton post-norm
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept

## Risk

Pure restack of two orthogonal measured gains: the block-m change touches only bs>=128, the post-norm change only bs<32, so the expected headline is roughly 1.130 x 1.092 = ~1.23x, and the bs<128 rows must reproduce master's (55/156/207/121 us) within drift. If marlin-block-m-cap-16-port (in the judge, bs=128 48->16 and bs=256 64->16) harvests better than verify-two-smaller at bs=256, the bs=256 pick here should follow it (16 instead of 32) - either retune this before claim or queue an ab-tune companion; bs=128 is 16 in both. Risk is low: the identical selection code already passed all 5 tests (incl. bs=257) on the old base; the only interaction is that bs=128/256 still use SGLang's combine, which is unchanged.

## Adversarial review

Reviewed 2026-10-24T112356Z by an adversarial subagent (model opus).

**Verdict: proceed.** A clean restack of two measured gains on disjoint shapes: block-m two-smaller touches only bs>=128, the landed Triton post-norm only bs<32, and the code is the same one-line pick that already passed all 5 tests including bs=257. One run can be read. The only open item is the bs=256 pick (32 here, 16 in marlin-block-m-cap-16-port, which is still running), and that run's harvest answers it, not a companion.

### Justification
The mechanism is measured on this exact problem. Per-m-block Marlin cost grows with thread_m_blocks. next-larger lost 18-28%, next-smaller gained (bs=128 1.194, bs=256 1.151), and verify-two-smaller gained more (bs=128 468.8->343.7 us, 1.364; bs=256 618.6->468.7 us, 1.32). All of these ran on base fc28840b9d, whose bs>=128 path is byte-identical to master's: the post-norm patch only adds a num_tokens < 32 branch, and bs=128/256 still call SGLang's fused_post_expert_norm_weighted_sum. Master's bs=128/256 rows (475.9/622.2 us, Marlin 95-96%, 4.7x/5.7x SoL of 101/109 us) match the old base's 468.8/618.6 within drift, so the expected ~345/~470 us are well grounded. Headline check: each load is one shape and the score is a geomean over 6 loads, so swapping master's bs=128/256 per-shape speedups (1.001/1.000) for 1.364/1.32 multiplies the headline by (1.364*1.32)^(1/6) ~= 1.10, which predicts about 1.24x. The proposer's ~1.23x is consistent. The controls are right: bs=1/8/16 keep pick 8 plus the Triton post-norm, and bs=32 keeps 16 plus SGLang's combine, so they should reproduce master's 55/121/156/207 us. Weakest link: the bs=256 choice of 32 is not known to be the best point. Decode_layer's cap-16 gained at T=256, and cap-16-port (64->16 at bs=256) is in the judge right now. If 16 beats 32 at bs=256, this run lands a result that is already sub-optimal at one of six loads. Minor: the old-base profile shows bs=256 Marlin at 490.8 us against a 468.7 us shape median (profile and timed passes disagree), so judge this run by the timed median and the paired SGLang ratio, not the kernel sum.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (192f225d8f): not a confounder for the target shapes, since its branch is num_tokens < 32 only and bs>=128 keeps SGLang's combine unchanged. Its rows (bs=1/8/16) are the in-run control and should reproduce master. A drift of more than ~2.5% there means run noise, not interaction. If it were reverted, bs>=128 would be unaffected.
- sglang-marlin-path-parity (a646cb4151): the foundation. block_size_m feeds moe_align_block_size over the 32-expert bank, so padding is 32*(bm-1) rather than 256*(bm-1), plus both Marlin calls. Smaller bm also cuts padded rows, but the old-base measurement had the same local-count align, so the measured gain already includes this interaction and carries over unchanged.

### Interference and A/B needs
No companion is needed. A gain at bs=128/256 can only come from the block-m pick, because nothing else in the timed path changes for those shapes, and the bs<128 rows are a built-in control on the same base. A loss would also be unambiguous: it would mean a bad restack or drift, and the controls would show which. The one real question is bs=256 at 32 versus 16. An ab-tune companion (this branch with the bs>=128 pick replaced by min(pick, 16)) would be the same code as cap-16-port restacked onto post-norm. That is a tuning choice settled by cap-16-port's harvest, which is already in flight. It is not needed to read this run. Recommended sequencing: hold dispatch until cap-16-port (base 150297e862) harvests. If its bs=256 row beats verify-two-smaller's 1.32 by clearly more than drift (more than ~3%), retune this spec's bs=256 pick to 16 before claiming, so that only one restack is spent. Otherwise dispatch as written. bs=128 is 16 in both, so nothing there depends on it.

### Open questions
- Does cap-16-port's bs=256 row (64->16) beat verify-two-smaller's 1.32 (64->32) by more than drift? If yes, change the bs=256 pick to 16 before dispatch, and do not run both restacks.
- The bs=257 test now picks 32 on master's lineage (it passed on the old base). Its post-norm path is SGLang's at bs>=32, which is unchanged, so correctness risk is minimal, but confirm the 5/5 tests pass.
- For the implementer: use BLOCK_SIZE_M_CANDIDATES.index(block_size_m) after the loop, as in the landed patch, so the loop-exhausted case (64) maps to index 4. Do not track idx inside the loop in a way that misses the exhaust case.

