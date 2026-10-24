---
problem: muse_attention_decode
slug: splitkv-full-line-gather-mapping
created: 2026-10-24T183847Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64):
  remap threads so that each warp-wide cp.async instruction covers whole 128-byte cache rows
  instead of half rows. Today thread tid copies token tt = tid >> 2, 16-byte chunks c = 2 * (tid &
  3) + {0, 1}, so one warp instruction touches 8 tokens x 64 bytes (two sectors of each line) and
  the other half of every line arrives from the next instruction. New mapping: for j in {0, 1}: tt
  = (tid >> 3) + 32 * j, c = tid & 7; the destination stays dst = tt * ROW + ((c ^ (tt & 7)) << 4)
  for K and TILE * ROW + dst for V, the validity test and the page index read idx_s[(k & 1) *
  CHUNK + t * TILE + tt] are computed per tt as now, and the source is ks + c * 16 / vs + c * 16
  with src_bytes ok ? 16 : 0. Each thread still issues 4 cp.async16 per tile (2 tokens x K and V),
  the ring layout, swizzle, commit/wait counts and all compute are unchanged, so the smem contents
  per stage are byte-identical to master's. Rename the load_inline module to
  muse_full_decode_splitkv_v2 (no stale build). The import self-check runs unchanged.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: kernel A gathers random page-1 rows at 2.94 TB/s (decode_16, 111.8 us) and 2.67 TB/s (decode_32, 245.9 us) against 4.8 TB/s; a full 128-byte line per L2 request should let L2 issue one 128 B DRAM fetch per row instead of two 64 B ones arriving apart, worth an estimated 3-8% on kernel A
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run kernel A
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T184058Z
review_companions: null
---

# splitkv-full-line-gather-mapping

## Change

One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64): remap threads so that each warp-wide cp.async instruction covers whole 128-byte cache rows instead of half rows. Today thread tid copies token tt = tid >> 2, 16-byte chunks c = 2 * (tid & 3) + {0, 1}, so one warp instruction touches 8 tokens x 64 bytes (two sectors of each line) and the other half of every line arrives from the next instruction. New mapping: for j in {0, 1}: tt = (tid >> 3) + 32 * j, c = tid & 7; the destination stays dst = tt * ROW + ((c ^ (tt & 7)) << 4) for K and TILE * ROW + dst for V, the validity test and the page index read idx_s[(k & 1) * CHUNK + t * TILE + tt] are computed per tt as now, and the source is ks + c * 16 / vs + c * 16 with src_bytes ok ? 16 : 0. Each thread still issues 4 cp.async16 per tile (2 tokens x K and V), the ring layout, swizzle, commit/wait counts and all compute are unchanged, so the smem contents per stage are byte-identical to master's. Rename the load_inline module to muse_full_decode_splitkv_v2 (no stale build). The import self-check runs unchanged.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: kernel A gathers random page-1 rows at 2.94 TB/s (decode_16, 111.8 us) and 2.67 TB/s (decode_32, 245.9 us) against 4.8 TB/s; a full 128-byte line per L2 request should let L2 issue one 128 B DRAM fetch per row instead of two 64 B ones arriving apart, worth an estimated 3-8% on kernel A
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run kernel A

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). The rewrite's BULK=0 fallback uses this same full-line mapping, so a gain here also tells stage 2 of that rewrite which copy form to keep.

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** A cheap, rules-clean, byte-identical-smem remap worth one run, but the stated mechanism is wrong: master's mapping already touches all four 32 B sectors of every row in the first instruction, so the realistic win is halving L2 sector requests per row (likely 0-4% on kernel A), not merging DRAM fetches. Judge it on the splitkv_attn_kernel rows, not the headline.

### Justification
Plausible but overstated. The profile supports the target: kernel A is 111.8 us (91.5%) of full decode_16 and 245.9 us (95.9%) of full decode_32, at about 2.9 / 2.7 TB/s against a 63.2 / 126.4 us floor, so it is about 1.8-1.9x off SoL and is the right kernel to attack. Reading master (submission.py lines 572-589), thread tid copies chunks c = 2*(tid&3)+j, so warp instruction j=0 covers chunks {0,2,4,6} of 8 rows, i.e. bytes 0-15, 32-47, 64-79, 96-111: it touches every 32 B sector of the line (half of each), and j=1 fills the other halves. The hypothesis says one instruction touches 'two sectors of each line' and the other half of the line 'arrives from the next instruction' as a separate 64 B DRAM fetch; that is not what happens. The first instruction already misses on all four sectors, so DRAM sees the full line promptly, and the second instruction's sector requests are L2 hits or in-flight miss merges. What the remap really removes is the duplicate L2 sector request (each sector requested twice today, once after), plus some LSU/L1TEX wavefront and MSHR pressure. That matters only if L2 request throughput or miss-merge capacity limits kernel A, not DRAM. At about 2.8 TB/s DRAM, today's L2 sector-request rate is about 5.6 TB/s equivalent, under H100's L2 bandwidth, so the effect is probably small. Weakest link: the 3-8% estimate rests on the wrong DRAM-granularity story. A 0-3% kernel-A change is the realistic prior, and it sits inside the roughly 6% drift seen on full decode_32 between identical runs. Expected directions match TASK.md: only full decode_16/32 run kernel A. Full shapes weigh 16 against sliding's 46, so full decode_16 is about 74% of its load time and full decode_32 about 77%. A 5% kernel-A gain is then about 3.5% on two of six loads, about 1% on the geomean headline, inside the ±1.5% CI. The swizzle stays conflict-free: 8 threads per row write 8 distinct XOR-swizzled 16 B chunks of one 128 B row, 4 rows per instruction. Smem contents are byte-identical, so correctness risk is very low. The validity and page-index reads per tt double, from one per thread to two (tt = (tid>>3)+32j), which adds two smem int loads per tile. That is negligible but not zero.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: defines kernel A, its STAGES=4 ring, 2 CTAs/SM and CHUNK=1024. This is the base, not a confounder, but the gain depends on its in-flight depth. If kernel A is latency/MLP-bound (bytes in flight = 2 CTAs x 3 stages x 16 KB per SM), cutting L2 requests may not move it, while a deeper ring might. A null result says 'not at STAGES=4', not 'mapping irrelevant'.
- sliding-deep-splits-on-cuda-splitkv, fused-preattention-triton, fa3-hdim128-head-pairing, paired-heads-launch-cleanup(+ab-tune), sliding-fused-triton-decode1-gate: none touch kernel A's copy path. Pairing is disabled when the splitkv gate fires, and the pre-attention kernel is about 2 us before it, so no interaction on the shapes that change. Sliding shapes are timed per shape, so they cannot leak into the full-shape rows.

### Interference and A/B needs
No companion warranted. The change is a single indivisible sub-change (the copy mapping), so there is no sub-change to isolate with a control. No landed hypothesis shares its memory path except the base kernel, and reverting that base removes kernel A entirely. Re-running master as a noise control is identical code, which the harness never re-runs. The real ambiguity is noise: a 0-4% kernel-A effect against about 6% shape-median drift. That is better handled by judging the per-kernel top_kernels rows (us of splitkv_attn_kernel on both decode_16 and decode_32 moving the same direction) than by a companion. Coordinate with the sibling splitkv-l2-128b-prefetch-hint (queued, p5): it targets the same 'second half-row' traffic with an L2::128B hint. If this remap gains, the hint is largely moot on top of it, and vice versa. Run them as independent siblings off master, and do not stack them until one shows a kernel-A gain. Also, full-cuda-onelaunch-warpspec-bulk-gather (running) replaces kernel A outright. If it lands and self-checks first, retarget or hold this idea.

### Open questions
- Are the scored full-shape page-table rows really random (FINDINGS says 'random page-1 rows')? If generate_input lays each request's rows out contiguously, adjacent tokens share DRAM pages and the per-line request pattern matters even less.
- Does an ncu run of master kernel A show lts__t_sectors_srcunit_tex_op_read about 2x lts__t_sectors_srcunit_tex_lookup_miss (duplicate sector requests), and is L2 or L1TEX throughput, rather than DRAM, near its limit? That would justify the mechanism. The profile available here only gives kernel us.
- Should the module rename be coordinated? Both this and splitkv-l2-128b-prefetch-hint rename to muse_full_decode_splitkv_v2. That is fine on separate branches but collides if they are ever stacked.

