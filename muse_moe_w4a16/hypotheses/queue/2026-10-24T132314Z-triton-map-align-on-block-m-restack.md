---
problem: muse_moe_w4a16
slug: triton-map-align-on-block-m-restack
created: 2026-10-24T132314Z
base: master
area: MoE align/dispatch + block_size_m pick
change: >
  Stack the two measured, shape-disjoint moe wins: start from master
  (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens
  >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained
  4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) unchanged: the _align_small_kernel single launch
  for expert_map gather + align at num_tokens <= 32, with its prepare() warm-up and self-check.
  The result is the 3-way merge of the map-align branch onto master; the only conflict is the
  block_size_m pick, resolved by keeping map-align's _block_size_m() call and applying master's
  num_tokens >= 128 two-step adjustment right after it. The two changes touch disjoint shapes
  (<=32 vs >=128), so each shape should read as its own parent did.
expected:
  - shapes: bs=1 *
    direction: faster
    why: map-align +12.6% at bs=1
  - shapes: bs=8 *
    direction: faster
    why: map-align +6.5%
  - shapes: bs=16 *
    direction: faster
    why: map-align +6.1%
  - shapes: bs=32 *
    direction: faster
    why: map-align +4.5%
  - shapes: bs=128 *
    direction: flat
    why: master block-m pick unchanged
  - shapes: bs=256 *
    direction: flat
    why: master block-m pick unchanged
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T132520Z
review_companions: null
---

# triton-map-align-on-block-m-restack

## Change

Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained 4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) unchanged: the _align_small_kernel single launch for expert_map gather + align at num_tokens <= 32, with its prepare() warm-up and self-check. The result is the 3-way merge of the map-align branch onto master; the only conflict is the block_size_m pick, resolved by keeping map-align's _block_size_m() call and applying master's num_tokens >= 128 two-step adjustment right after it. The two changes touch disjoint shapes (<=32 vs >=128), so each shape should read as its own parent did.

## Expected

- `bs=1 *` faster: map-align +12.6% at bs=1
- `bs=8 *` faster: map-align +6.5%
- `bs=16 *` faster: map-align +6.1%
- `bs=32 *` faster: map-align +4.5%
- `bs=128 *` flat: master block-m pick unchanged
- `bs=256 *` flat: master block-m pick unchanged

## Risk

Minted by churnsubs under the standing autonomy instruction to restack two harvested wins; the merged file is at /home/user/.claude/jobs/7156a0da/tmp/merge/merged.py (map-align branch + 5-line block-m adjust).

## Adversarial review

Reviewed 2026-10-24T132520Z by an adversarial subagent (model opus).

**Verdict: proceed.** The merge is correct. Each shape runs code that one parent already measured: bs<=32 is byte-for-byte the map-align branch's path (1.185x run c2feb68aafb44aeb), and bs>=128 is master's block-m path (1.254x run c00ef7295c154867). One run is enough to read it, with no companion, and nothing in it breaks a rule. Expect about 1.30-1.32x.

### Justification
Merge check: `git merge-tree master hyp/muse_moe_w4a16/triton-map-align-small-m-port` conflicts only in submission.py (plus an attempts/ md add/add). merged.py differs from the map-align branch file only by master's 5-line `num_tokens >= 128` two-step block_size_m adjust, placed after `_block_size_m()`. It differs from master only by the map-align additions: ALIGN_FUSED_MAX_TOKENS, _align_small_kernel and its helpers, the bank 'start' field, the prepare() warm-up and self-check, and the `<=32` branch. `_block_size_m()` is the same loop master inlines. The >=128 adjust never fires on the <=32 path, so bs=1/8/16/32 get the same block_size_m the map-align run used. bs=128/256/257 (and the bs=33 test) take the else path, which is master's gather + moe_align_block_size(...,32), followed by the same adjust. So the per-shape code equals a measured parent everywhere. Mechanism: the map-align run's top_kernels show the align row dropping from 4.4-6.6 us to 1.5-4.4 us and the expert_map gather launch disappearing. But most of the gain is Marlin's own time: bs=1 45.3 to 39.9 us, bs=16 117.0 to 104.9 us. The run log explains it: SGLang's moe_align_block_size(local_ids, bsm, 32) gives the -1 (remote) routes m-blocks of their own (48 of 60 self-check cases have n_pad < r_pad; at T=32 336 vs 560, all-remote 0 vs 256). _align_small gives them none, so Marlin launches fewer m-blocks. The hypothesis describes the change as launch fusion. In fact it is mostly Marlin work removal. That does not affect the stack, because master's block-m change never touches <=32. Shape directions match the weights in TASK.md (all six shapes weight 60). Measured against master's own rows (55.5/120.4/155.9/209.7 us), the map-align rows (50.5/107.9/145.3/204.1 us) are about +10/+11.5/+7/+2.7%. The expected bs=32 +4.5% is therefore optimistic: about +2.7% vs master's bs=32 row, which is within the 1.5-2.5% cross-run drift in LESSONS. Weakest link: bs=32 may read flat, and the expected magnitudes are quoted against the older base rather than master's rows. Direction is not in doubt at bs=1/8/16.

### Confounders on the current kernel
- post-expert-norm-triton-small-m: shared by both parents (map-align branched from 192f225, master descends from it), so the bs<32 combine is the same code in both and is not a confounder. bs=32 uses SGLang's fused_post_expert_norm in both parents too.
- marlin-block-m-two-smaller-on-postnorm: touches only num_tokens>=128, which never reaches _align_small. No interaction on any scored shape. bs>=128 is identical to master, which makes it an in-run control for drift.
- sglang-marlin-path-parity: the base of both. Its 32-expert align (which gives -1 routes m-blocks) is what map-align improves on. It is unchanged above 32 tokens, so no confound.

### Interference and A/B needs
No companion is warranted. The two changes are shape-disjoint by code, not just by expectation: the <=32 branch and the >=128 adjust are mutually exclusive, and every shape runs exactly one parent's measured code. bs=128/256 act as an in-run drift control, since the expected result there is flat against master's c00ef7295c154867 rows. bs=1/8/16 should reproduce the map-align rows (50.5/107.9/145.3 us) within about 2%. If a <=32 shape lands well away from the map-align row, the cause is drift or the environment, not an interaction. An ab-revert of either parent would only re-measure a parent that has already been measured.

### Open questions
- The self-check prints '60 cases, 60 mismatches', but correctness passed 5/5. All the hard mismatches come from the checker's own strictness: capacity differs from SGLang's (232/29 vs 64/8 at T=1, because SGLang's small-batch wrapper sizes differently), and n_pad/blocks differ because SGLang buckets -1 routes. The per-expert route sets agreed in every case. A future edit should downgrade those to info lines so that a real failure stands out. Don't do it in this run: it would change the code being measured.
- Marlin with is_ep=True now sees n_padded that excludes remote-route blocks. Correctness passed on all 5 tests. Do the bs=4/33/257 test shapes also exercise all-remote tokens (n_pad=0 at T<=32)? If they don't, that edge case is covered only by the self-check.
- align-remote-sentinel-ids (queued) also targets how -1 routes are bucketed. It is probably moot at <=32 after this lands, and should be re-scoped to num_tokens>32 (bs=128/256) if it is kept.
- triton-small-m-threshold-32 (queued): confirm it moves SMALL_M_POST_NORM, not ALIGN_FUSED_MAX_TOKENS (already 32), so it does not duplicate this.
- Read bs=32 against master's 209.7 us row, not against the +4.5% figure, which was taken from the older base. A flat bs=32 is consistent with the parent.

