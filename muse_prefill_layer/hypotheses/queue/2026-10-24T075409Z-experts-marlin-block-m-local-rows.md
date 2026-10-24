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
review: required
status: queued
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
