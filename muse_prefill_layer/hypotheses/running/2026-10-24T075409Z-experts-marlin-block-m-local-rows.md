---
problem: muse_prefill_layer
slug: experts-marlin-block-m-local-rows
created: 2026-10-24T075409Z
base: master
area: w13 marlin + w2 marlin (block_size_m pick)
change: >
  Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity
  loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks on the first size where
  num_tokens * TOP_K / LOCAL_EXPERTS / size < 0.9. num_tokens * 8 / 32 is the routes per local
  expert only if every route were local; under EP8 only 32 of the 256 experts are local, so the
  expected local routes per local expert are num_tokens * TOP_K / N_EXPERTS (= T / 32), 8x fewer.
  At T=753 the parity rule computes 188 rows per expert, never breaks, and picks 64 although each
  local expert holds ~23.5 rows on average. Replace LOCAL_EXPERTS with N_EXPERTS in that one
  expression (keep the 0.9 rule and the candidate list). Picks become: T=753 -> 32 (23.5/32 = 0.73
  breaks), T=3012 -> 64 (94/64 = 1.47, unchanged), T=16384 -> 64 (unchanged); tests T=300 -> 16,
  T=600 -> 32, T=753 -> 32, T=1506 -> 64, T=4096 -> 64. The same block_size_m feeds
  moe_align_block_size and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls, as it
  must. Nothing else changes. Mechanism (empirical, from muse_moe_w4a16
  marlin-block-size-m-next-larger, job 2639e3ca7f5549de): Marlin's per-m-block cost grows with
  thread_m_blocks even with the m-block count unchanged (+28% for 1->2, +18% for 3->4 at about one
  block per expert); here 64 -> 32 is thread_m_blocks 4 -> 2 while most local experts (~23 rows)
  still need one block. In the parity profile (job 62025c368c3c4e8e) Marlin is 730/681 us of the
  turn_1 shapes (66%/45%) and streams ~0.9 GB of NVFP4 weights + scales at ~1.24 TB/s, i.e. it is
  far from its 190 us bandwidth floor and not bound by bytes.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 730/681 us at block 64 (thread_m_blocks 4) on ~23.5 rows per local expert; block 32 halves thread_m_blocks with ~1 block per expert for most experts; sister data predicts ~20-30% off Marlin, i.e. ~10-15% per turn_1 shape
  - shapes: "*load=turn_4*"
    direction: flat
    why: T=3012 keeps 64 under both rules (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 keeps 64 under both rules (control)
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T083634Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T083634Z
base_sha: 67649c559d5a07d10b659c84bf80c1b1ec29c879
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/experts-marlin-block-m-local-rows
worktree: .worktrees/muse_prefill_layer/experts-marlin-block-m-local-rows
mode: null
---

# experts-marlin-block-m-local-rows

## Change

Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks on the first size where num_tokens * TOP_K / LOCAL_EXPERTS / size < 0.9. num_tokens * 8 / 32 is the routes per local expert only if every route were local; under EP8 only 32 of the 256 experts are local, so the expected local routes per local expert are num_tokens * TOP_K / N_EXPERTS (= T / 32), 8x fewer. At T=753 the parity rule computes 188 rows per expert, never breaks, and picks 64 although each local expert holds ~23.5 rows on average. Replace LOCAL_EXPERTS with N_EXPERTS in that one expression (keep the 0.9 rule and the candidate list). Picks become: T=753 -> 32 (23.5/32 = 0.73 breaks), T=3012 -> 64 (94/64 = 1.47, unchanged), T=16384 -> 64 (unchanged); tests T=300 -> 16, T=600 -> 32, T=753 -> 32, T=1506 -> 64, T=4096 -> 64. The same block_size_m feeds moe_align_block_size and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls, as it must. Nothing else changes. Mechanism (empirical, from muse_moe_w4a16 marlin-block-size-m-next-larger, job 2639e3ca7f5549de): Marlin's per-m-block cost grows with thread_m_blocks even with the m-block count unchanged (+28% for 1->2, +18% for 3->4 at about one block per expert); here 64 -> 32 is thread_m_blocks 4 -> 2 while most local experts (~23 rows) still need one block. In the parity profile (job 62025c368c3c4e8e) Marlin is 730/681 us of the turn_1 shapes (66%/45%) and streams ~0.9 GB of NVFP4 weights + scales at ~1.24 TB/s, i.e. it is far from its 190 us bandwidth floor and not bound by bytes.

## Expected

- `*load=turn_1*` faster: Marlin pair 730/681 us at block 64 (thread_m_blocks 4) on ~23.5 rows per local expert; block 32 halves thread_m_blocks with ~1 block per expert for most experts; sister data predicts ~20-30% off Marlin, i.e. ~10-15% per turn_1 shape
- `*load=turn_4*` flat: T=3012 keeps 64 under both rules (control)
- `*load=first_turn*` flat: T=16384 keeps 64 under both rules (control)

## Risk

Routing is skewed (router rows aligned with the stream mean, router_bias std 0.055, reference.py _router), so hot local experts at T=753 can exceed 32 rows and need 2 blocks of 32 (2 weight passes at thread_m_blocks 2); if that offsets the per-block saving turn_1 comes in flat, and the sibling experts-marlin-block-m-48-turn1 (one block up to 48 rows) is the comparison. The extrapolation from the sister's two points (tmb 1->2 and 3->4 at tiny M) to tmb 4->2 at ~23 real rows is unmeasured. Tests T=300/600 also change block (16/32); correctness is unaffected (block size is a tiling choice).

## Adversarial review

Reviewed 2026-10-24T083634Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-expression, shape-only change that moves block_size_m 64->32 at T=753 only. It is well justified by the 8x local-row overcount (the same insight that won on decode, marlin-block-m-cap-16) and by Marlin dominating turn_1. The untouched turn_4 and first_turn shapes act as built-in controls, the sibling experts-marlin-block-m-48-turn1 is the natural A/B, and no extra companion is needed.

### Justification
Arithmetic checks out against submission.py lines 273-276. Under the parity rule, T*8/32/size never drops below 0.9 at T=753, so 64 is picked. With N_EXPERTS the per-size values are 23.5/8=2.9, /16=1.47 and /32=0.73, so the loop breaks at 32. T=1506 still breaks at 64 (0.735). T=3012 (1.47) and T=16384 fall through to 64. The scored set therefore changes only the two turn_1 shapes (T=753), and only Python-side selection on a shape int changes. Profile (job 62025c368c3c4e8e): the Marlin pair is 730.1/680.7 us of the turn_1 shapes (66%/45%), and the floors are 257/497 us. In turn_1-load terms, Marlin is about 33.6 ms + 10.9 ms of about 72.6 ms weighted, i.e. 61% of that load. Since the geomean is over three loads, a 20-30% Marlin cut is roughly 12-18% on the turn_1 load and roughly 4-6% on the headline. That is well above the 1.5-2.5% run-to-run drift, so one run can resolve it. Cross-shape scaling supports a padded-work-bound Marlin rather than a byte-bound one. Turn_1 (T=753) costs 730 us against 2248 us at T=3012: 4x the tokens for only about 3x the time. Shrinking the 64-row padding (about 2048 padded rows for about 750 real local routes, under 40% MMA utilisation) should therefore help, and decode's cap-16 result (+8-15% at T=128/256) shows the same mechanism. Weakest link: the cost model. Halving block size halves MMA rows per block, but dequant work per block is unchanged, and every expert pushed past 32 rows by skew (router bias std 0.055, 'soft' dist; Poisson alone gives about 4% of experts above 32 at mean 23.5, and skew makes it more) pays a second full weight pass with a second dequant. If Marlin at this M is dequant/ALU-bound rather than MMA-bound (as LESSONS found for the Triton W4A16 GEMV), the saving shrinks and the 2-block experts can cancel it. The 20-30% figure is also extrapolated from the sister's tiny-M points (tmb 1->2, 3->4) to tmb 4->2 at about 23 real rows, which is unmeasured. Expected directions match TASK.md: turn_4 and first_turn flat, turn_1 faster.

### Confounders on the current kernel
- sglang-pipeline-parity: it is the base that introduced the parity loop being edited. It does not confound, because baseline.py (lines 243-246) uses the identical rule, so the same-run SGLang time is an exact paired control for the old pick. Reverting it makes no sense, because it is the thing being modified.
- No other landed hypothesis touches the experts path (moe-align-side-stream is neutral and not on master), so a turn_1 Marlin delta attributes cleanly to block_size_m.

### Interference and A/B needs
No companion is warranted. Only one landed item exists and it is the base. The change is a single variable (block_size_m at T=753), and turn_4 and first_turn (picks unchanged) are in-run controls for drift. A gain shows that 32 beats 64 at T=753, and a loss shows that 2-block hot experts or per-block dequant dominate. Either outcome can be read from the Marlin kernel's own top_kernels time on the two turn_1 rows. The only real ambiguity, whether 48 would be better, is already covered by the queued sibling experts-marlin-block-m-48-turn1, which should run on the same base so the two are an A/B pair. An ab-tune at block 16 would only answer a further question, not disambiguate this result.

### Open questions
- What is the actual distribution of local rows per expert at T=753 under the 'soft' dist? Count it offline, outside the timed call, from reference.generate_input routing (or from n_padded at block 32 vs 64). It tells how many experts take 2 blocks at 32 and whether 48 is the better pick.
- The profile merges both Marlin calls (count 2). Is the saving split between w13 (size_n=2I) and w2 differently? The profile cannot show this, so read only the combined Marlin time.
- If experts-triton-w4a16-inkernel-fp4 (running) replaces Marlin on the turn shapes, this pick becomes moot for turn_1. Whichever lands second should be rebased onto the winner.
- Turn_4 (T=3012, 94 rows/expert) is left at 64 by this rule even though 48 fits 2 blocks of 96 almost exactly. That belongs to experts-marlin-block-m-48-turn4; do not fold it in here.

