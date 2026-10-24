---
problem: muse_attention_decode
slug: splitkv-l2-128b-prefetch-hint
created: 2026-10-24T183848Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  One change in _SPLITKV_CUDA: the K/V cp_async16 helper used by issue_tile in splitkv_attn_kernel
  emits `cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;` instead of
  `cp.async.cg.shared.global [%0], [%1], 16, %2;`. Add it as a separate helper cp_async16_l2 and
  use it only for the four K/V copies in issue_tile; the q copy in issue_meta and the 4-byte
  page-index copies keep the existing helpers. The L2::128B qualifier is a prefetch-size hint (PTX
  ISA 7.4+, sm_80+): the first 16-byte request to a cache row brings the whole 128-byte line into
  L2, so the second half-row instruction hits. Nothing else changes; rename the module
  muse_full_decode_splitkv_v2.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: each 128-byte K or V row is fetched by two warp instructions 64 bytes apart; a whole-line L2 fill may cut DRAM transactions per row from two to one. Expect 0-5% on kernel A (111.8 / 245.9 us)
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run kernel A
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T184058Z
review_companions: null
hold_reason: "adversarial review: The mechanism rests on a misreading of issue_tile: each warp-wide cp.async already touches all four 32-byte sectors of every 128-byte K/V line (chunks 0,2,4,6 a"
---

# splitkv-l2-128b-prefetch-hint

## Change

One change in _SPLITKV_CUDA: the K/V cp_async16 helper used by issue_tile in splitkv_attn_kernel emits `cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;` instead of `cp.async.cg.shared.global [%0], [%1], 16, %2;`. Add it as a separate helper cp_async16_l2 and use it only for the four K/V copies in issue_tile; the q copy in issue_meta and the 4-byte page-index copies keep the existing helpers. The L2::128B qualifier is a prefetch-size hint (PTX ISA 7.4+, sm_80+): the first 16-byte request to a cache row brings the whole 128-byte line into L2, so the second half-row instruction hits. Nothing else changes; rename the module muse_full_decode_splitkv_v2.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: each 128-byte K or V row is fetched by two warp instructions 64 bytes apart; a whole-line L2 fill may cut DRAM transactions per row from two to one. Expect 0-5% on kernel A (111.8 / 245.9 us)
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run kernel A

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). Overlaps in mechanism with splitkv-full-line-gather-mapping (both target half-line requests); if both run, they are an A/B on the same hypothesis, so the reviewer may hold one.

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: hold.** The mechanism rests on a misreading of issue_tile: each warp-wide cp.async already touches all four 32-byte sectors of every 128-byte K/V line (chunks 0,2,4,6 at offsets 0/32/64/96 on j=0, then 1,3,5,7 on j=1), so DRAM already fetches each row once and an L2::128B hint cannot remove a transaction; the predicted 0-5% is also below the 6% run-to-run drift on full decode_32, so one run cannot be interpreted.

### Justification
A. In master's issue_tile, thread tid handles token tt = tid>>2 and 16-byte chunks c = 2*(tid&3)+j. For j=0 one warp instruction covers 8 tokens x chunks {0,2,4,6}, which are byte offsets 0,32,64,96: one 16-byte piece in each of the line's four 32-byte sectors. Every sector of every row is therefore requested from L2 by the first instruction, and the j=1 instruction (chunks 1,3,5,7) hits the same sectors, which are already in flight or resident. DRAM traffic is already one 128-byte line per K or V row. ROW = 128 and rows sit at kc + row*128, so a line is exactly one row: a 128-byte prefetch brings in nothing new, and because page_size=1 gathers scattered rows, neighbouring rows cannot benefit either. The hypothesis says the two instructions are '64 bytes apart' (each a contiguous half-row), but that is not the code's mapping. The sibling splitkv-full-line-gather-mapping makes the same mistake ('two sectors of each line'). Profile context: kernel A takes 111.8 us at decode_16 (SoL 63.2 us for the whole shape) and 245.9 us at decode_32 (SoL 126.4 us), so it is roughly 1.8-1.9x off the floor and a real target, but this change cannot account for any of that gap. The expected directions are consistent with TASK.md, since only full decode_16/32 run kernel A, and those flat elsewhere. Weakest link: the premise that each row costs two DRAM transactions today. It is false on this code, so the expected effect is about 0 and sits inside the documented 6% median drift on full decode_32. A gain would be noise and a loss would be noise.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv (landed): introduces the exact issue_tile mapping and 16-byte cp.async.cg loads the hypothesis modifies; its interleaved chunk mapping is what already makes every instruction touch all four sectors, so the hint is redundant specifically because of how that patch laid out the copies.
- sliding-deep-splits-on-cuda-splitkv (landed, current best): touches only sliding shapes, but it moved full decode_32 within the run, and the noted ~6% drift on full decode_32 between identical runs swamps a 0-5% prediction.

### Interference and A/B needs
No companion makes a single run readable here, because the problem is the premise and not interference. An ab-revert of full-decode-cuda-gather-splitkv would remove the kernel entirely. A control that changes only the module name would just measure noise. If the author still wants to test L2 promotion, the useful experiment is a non-scored ncu comparison of kernel A's dram__sectors_read and lts__t_sectors_srcunit_tex_op_read, with and without the qualifier. That should show identical DRAM sector counts. A scored check is not the way to test it.

### Open questions
- Hold the sibling splitkv-full-line-gather-mapping on the same grounds? Its stated reason ('two sectors of each line' per instruction) is also wrong. Its remap could still help L2 request coalescing a little (one instruction covering full lines for 4 tokens instead of half of each sector for 8), but it should be re-justified on request count, not DRAM transactions.
- If full-cuda-onelaunch-warpspec-bulk-gather (running) lands and self-checks, kernel A is replaced for these shapes and both splitkv-gather micro-tunings become moot.
- Kernel A's remaining ~1.9x gap to SoL is more likely latency or occupancy bound: 4-stage ring x 16 KB at 2 CTAs/SM gives about 128 KB in flight per SM, plus per-tile __syncthreads and mma.sync compute. A deeper ring or more bytes in flight per SM is a better-founded lever than cache hints.

