---
problem: muse_moe_w4a16
slug: combo-wg-occ2-v64-hot-interleave-align2
created: 2026-10-25T161613Z
base: master
area: "combination: wgmma two-CTA/SM (prefill port) x hot-expert tile interleave + shared floor"
change: >
  COMBINATION (conjectural pairing, wgmma side): prefill's two-CTAs-per-SM rega variant ported to
  moe's wgmma kernels at B256 (and as a B128 candidate), paired with moe's deep-ring NMAX
  templating and a new hot-expert tile interleave, on top of the shared floor. Built on master
  a0e52640. SHARED FLOOR (identical in all three combo hypotheses minted 2026-10-25 ~16Z, so its
  rows are measured three times): (F1) decode_1: w13 Marlin use_atomic_add=True at num_tokens == 1
  only. Source w13-marlin-atomic-add-t1-only, branch
  hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port
  W13_ATOMIC_MAX_TOKENS = 1, W13_ATOMIC_TOL, _w13_band, _w13_atomic_self_check and the w13_mode
  kwarg in the w13 moe_wna16_marlin_gemm call of _moe_marlin (w2 keeps master's w2_mode). Measured
  twice: decode_1 1.766x (job 171684977ec1470f) and 1.754x (job 1048c6edf3e54892) against
  master-equivalent 1.698-1.721x in eight runs (+2.5-3.0%; Marlin pair 33.7 vs 35.9 us). Gate: its
  own self-check (rel-L2 vs master <= 1e-2; it measured 8.4e-3) plus a component graph (the w13
  Marlin call alone at T=1, skewed and random local routing) <= 0.98x. Phase F1 start deadline 120
  s, ~3 s. (F2) verify_32: master's OWN align2 win is dead code on the judged path. Master's
  _align2_setup gates class v256 for _moe_marlin (job c05a8f7326004be6: align2 chunk 64 4.70 us vs
  incumbent 6.46 us, ratio 1.374, ON), but kernel_fn sends 256 <= T <= 512 to _moe_wgmma
  (wg_on_256 True), which still runs bank expert_map[topk_ids.to(int32)] plus SGLang
  moe_align_block_size(local_ids, WG_BSM=16, count): the c05a8f73 verify_32 profile shows
  index_elementwise 3.4 + unrolled_elementwise 3.1 + moe_align_block_size_kernel 2.8 +
  count_and_sort_expert_tokens 2.5 = 11.8 us in four launches and no _align2 kernel. Source:
  master a0e52640's own _align_two_phase / _align2_count_kernel / _align2_place_kernel (port
  origin align-two-phase-port-from17, impl commit fb879b90; prefill origin
  align-two-phase-all-t-rega). Change: in _moe_wgmma only, when state['wg_align2'] (a plain int
  chunk, 0 = master) is set, replace the expert_map gather and moe_align_block_size with
  local_ids, (sorted_ids, block_experts, n_padded) = _align_two_phase(topk_ids, start, count,
  WG_BSM, chunk). Gate F2: warm every T in 128..512 at block_size_m 16 for each chunk in (64, 128,
  256, 512) <= the class top; equivalence on T in (128, 191, 192, 256, 257, 384, 512) x three
  local_starts x six routings (_align2_routings): local_ids torch.equal, n_padded equal,
  sorted_ids[:n_padded] and block_experts[:n_padded/16] torch.equal to master's front end
  (capacity tails may differ: the wg kernels read below n_padded only; assert n_padded/16 <=
  WG_MAX_BLOCKS), then full _moe_wgmma output torch.equal (bitwise: same kernels, same tables) on
  two of those routings per T; then the component graph (front end alone: gather+cast+align vs
  align2) <= 0.98x at T=256 (expected ~0.4-0.5x). Applies to whichever wgmma bucket is ON (B256
  today; B128 only if its flag is on). Phase F2 start deadline 150 s, ~15-20 s (Triton constexpr
  warm per T is most of it). (C2a) Two CTAs per SM. Sources: prefill
  experts-rega-low-class-two-ctas-per-sm (branch
  hyp/muse_prefill_layer/experts-rega-low-class-two-ctas-per-sm, impl commit dfb755d3: separate
  template rega_lo_kernel<G, NMAX, STG, OCC> appended to the master source, __launch_bounds__(384,
  OCC), at OCC 2 numRegs exactly 80 with producer setmaxnreg.dec 32 / consumer inc 104,
  cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor checked BEFORE any launch,
  grid = OCC x SM count stored in the state) and moe cuda-nvfp4-wgmma-rega-b128-n32-deep-ring
  (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring, impl commit c30fd0cc:
  wg_kernel<G, NMAX, STG> with Geo<NMAX, STG>, on-device chunk cut every NMAX/16 blocks, buckets
  above NMAX not instantiated, the selector's per-variant 'weight read twice' MB and TB/s prints).
  Build, in a NEW load_inline module (muse_moe_w4a16_wgmma_occ2_v1; master's wg module stays
  loaded and byte for byte), wg2_kernel<G, NMAX=64, STG=3, OCC=2> for both G (w13+silu, w2): smem
  1024 + 3 x (9216 + 2 x 64 x 128) + 64-row epilogue per CTA, two CTAs plus reserves <= 228 KB
  (static_assert and a Python assert). Any of numRegs != 80, localSizeBytes != 0, occupancy < 2 or
  a ptxas spill turns C2 OFF before its first launch (setmaxnreg.inc hangs on a short register
  pool). (C2b, the conjectural dose) Hot-expert tile interleave: prefill's own gate (job
  1a77da43b3304b07) showed the two-CTA effect is real (V64 at 2 CTAs vs the same slab at 1 CTA:
  0.887x on random routing) but is eaten by chunk splitting under skew (pop:2 routing: V64 0.994x,
  V32 1.233x, chunks 33 -> 38 -> 54), so its PATH CHOSEN was master and the 3.4% turn_1 row was
  noise. At NMAX 64 a hot expert with 130 routes becomes 3 chunks instead of 2. In wg2_kernel
  only, change the item decode so that the chunks of one expert are tile-interleaved: item index
  -> (expert group g, tile j, chunk-within-expert c) with c fastest, so the duplicate reads of
  weight tile (e, j) are issued by adjacent items, which run concurrently on neighbouring CTAs and
  hit L2 for the second and third chunk. Numerics are unchanged (each item still computes one
  (chunk, tile) output exactly as master). Variants: V64 (deep-ring item order) and V64i
  (interleaved). CHECKS: master's s5 bucket canary (N16/N32/N64), _wg_self_check and the
  three-state _wg_copy_check on each variant; rel-L2 vs Marlin <= 1e-2 per kernel and the
  copy-check bound as master. SELECTOR: extend master's _wg_select to candidates {Marlin, master
  s5, V64, V64i} for B256 and B128, all with the F2 front end if F2 is ON, full kernel_fn cold-L2
  graphs, 41 interleaved reps, worse of zipf a=0.3 and the judge-like routing; a variant replaces
  master's chosen path for a bucket only at <= 0.98x; print per variant the w13 and w2
  kernel-graph us, achieved TB/s on touched-expert bytes, chunk count and 'weight read twice' MB,
  and one PATH CHOSEN line per bucket. state['wg2_variant_128'] / state['wg2_variant_256'] (ints)
  pick the launch. ROUTING SUMMARY: T == 1 -> F1; 256 <= T <= 512 -> wgmma (master s5 or V64/V64i
  per selector) with F2; 128 <= T <= 255 -> Marlin unless the selector flips it; everything else
  master. Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C2 build 35-40 s (start deadline
  140 s) + canary/self-check/copy check ~10 s + selector ~3 s (start deadline 230 s) = ~120 s of
  300. RULES (every combo phase): kernel_fn branches on shape and prepare-set plain ints/bools
  only; no host sync, no input-value-sized allocation, no data-dependent control flow, no prints,
  no try/except, no per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights
  stay NVFP4 (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation;
  every local route computed; no child processes. The judge prepares TWO states and copies them by
  cloning tensors: every scratch tensor, stream, workspace and error word lives in the state, no
  module-global device pointer, tensor map or tensor, and tensor maps stay encoded per call
  (master's s5 rule). PREPARE ORDER: master's prepare() runs first and unchanged (align
  self-check, _align2_setup, _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 /
  WG_REPLAY_DEADLINE_S 235 / WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs
  AFTER it, measures its own elapsed time from prepare()'s t_prepare, skips itself (flag stays
  master, prints PATH CHOSEN = master (deadline)) if it would start past its start deadline, and
  no phase starts timing past 270 s. GATE RULE (every source): exactness or tolerance check first,
  then cold-L2 CUDA-graph timing vs master's path for that class (>= 256 MB flush before each
  replay, interleaved A/B, >= 21 reps, empty-graph baseline subtracted, median); ON only at
  new/master <= 0.98 on the worse of the judge-like skewed routing and random routing; otherwise
  that class keeps master. Print one PATH CHOSEN line per source per class with both medians and
  the ratio, and keep those lines in the last 16k characters of the log (print them again at the
  very end of prepare, after master's [align2] lines). Print per-phase prepare seconds and the
  total.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "F2 -1.5-2%, plus C2 if V64i passes: w13 ~239-303 us / w2 ~112-148 us at 2 CTAs/SM with duplicate tiles from L2; target verify_32 <= 385 us (-5%). If the selector keeps s5 the row equals F2 alone"
  - shapes: bs=1 *
    direction: faster
    why: "F1: decode_1 +2.5-3%"
  - shapes: bs=128 *
    direction: flat
    why: "B128 needs ~1.25x over master's wgmma to flip from Marlin; the PATH CHOSEN B128 line with four candidates is the readout"
  - shapes: bs=32 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# combo-wg-occ2-v64-hot-interleave-align2

## Change

COMBINATION (conjectural pairing, wgmma side): prefill's two-CTAs-per-SM rega variant ported to moe's wgmma kernels at B256 (and as a B128 candidate), paired with moe's deep-ring NMAX templating and a new hot-expert tile interleave, on top of the shared floor. Built on master a0e52640. SHARED FLOOR (identical in all three combo hypotheses minted 2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1: w13 Marlin use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port W13_ATOMIC_MAX_TOKENS = 1, W13_ATOMIC_TOL, _w13_band, _w13_atomic_self_check and the w13_mode kwarg in the w13 moe_wna16_marlin_gemm call of _moe_marlin (w2 keeps master's w2_mode). Measured twice: decode_1 1.766x (job 171684977ec1470f) and 1.754x (job 1048c6edf3e54892) against master-equivalent 1.698-1.721x in eight runs (+2.5-3.0%; Marlin pair 33.7 vs 35.9 us). Gate: its own self-check (rel-L2 vs master <= 1e-2; it measured 8.4e-3) plus a component graph (the w13 Marlin call alone at T=1, skewed and random local routing) <= 0.98x. Phase F1 start deadline 120 s, ~3 s. (F2) verify_32: master's OWN align2 win is dead code on the judged path. Master's _align2_setup gates class v256 for _moe_marlin (job c05a8f7326004be6: align2 chunk 64 4.70 us vs incumbent 6.46 us, ratio 1.374, ON), but kernel_fn sends 256 <= T <= 512 to _moe_wgmma (wg_on_256 True), which still runs bank expert_map[topk_ids.to(int32)] plus SGLang moe_align_block_size(local_ids, WG_BSM=16, count): the c05a8f73 verify_32 profile shows index_elementwise 3.4 + unrolled_elementwise 3.1 + moe_align_block_size_kernel 2.8 + count_and_sort_expert_tokens 2.5 = 11.8 us in four launches and no _align2 kernel. Source: master a0e52640's own _align_two_phase / _align2_count_kernel / _align2_place_kernel (port origin align-two-phase-port-from17, impl commit fb879b90; prefill origin align-two-phase-all-t-rega). Change: in _moe_wgmma only, when state['wg_align2'] (a plain int chunk, 0 = master) is set, replace the expert_map gather and moe_align_block_size with local_ids, (sorted_ids, block_experts, n_padded) = _align_two_phase(topk_ids, start, count, WG_BSM, chunk). Gate F2: warm every T in 128..512 at block_size_m 16 for each chunk in (64, 128, 256, 512) <= the class top; equivalence on T in (128, 191, 192, 256, 257, 384, 512) x three local_starts x six routings (_align2_routings): local_ids torch.equal, n_padded equal, sorted_ids[:n_padded] and block_experts[:n_padded/16] torch.equal to master's front end (capacity tails may differ: the wg kernels read below n_padded only; assert n_padded/16 <= WG_MAX_BLOCKS), then full _moe_wgmma output torch.equal (bitwise: same kernels, same tables) on two of those routings per T; then the component graph (front end alone: gather+cast+align vs align2) <= 0.98x at T=256 (expected ~0.4-0.5x). Applies to whichever wgmma bucket is ON (B256 today; B128 only if its flag is on). Phase F2 start deadline 150 s, ~15-20 s (Triton constexpr warm per T is most of it). (C2a) Two CTAs per SM. Sources: prefill experts-rega-low-class-two-ctas-per-sm (branch hyp/muse_prefill_layer/experts-rega-low-class-two-ctas-per-sm, impl commit dfb755d3: separate template rega_lo_kernel<G, NMAX, STG, OCC> appended to the master source, __launch_bounds__(384, OCC), at OCC 2 numRegs exactly 80 with producer setmaxnreg.dec 32 / consumer inc 104, cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor checked BEFORE any launch, grid = OCC x SM count stored in the state) and moe cuda-nvfp4-wgmma-rega-b128-n32-deep-ring (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring, impl commit c30fd0cc: wg_kernel<G, NMAX, STG> with Geo<NMAX, STG>, on-device chunk cut every NMAX/16 blocks, buckets above NMAX not instantiated, the selector's per-variant 'weight read twice' MB and TB/s prints). Build, in a NEW load_inline module (muse_moe_w4a16_wgmma_occ2_v1; master's wg module stays loaded and byte for byte), wg2_kernel<G, NMAX=64, STG=3, OCC=2> for both G (w13+silu, w2): smem 1024 + 3 x (9216 + 2 x 64 x 128) + 64-row epilogue per CTA, two CTAs plus reserves <= 228 KB (static_assert and a Python assert). Any of numRegs != 80, localSizeBytes != 0, occupancy < 2 or a ptxas spill turns C2 OFF before its first launch (setmaxnreg.inc hangs on a short register pool). (C2b, the conjectural dose) Hot-expert tile interleave: prefill's own gate (job 1a77da43b3304b07) showed the two-CTA effect is real (V64 at 2 CTAs vs the same slab at 1 CTA: 0.887x on random routing) but is eaten by chunk splitting under skew (pop:2 routing: V64 0.994x, V32 1.233x, chunks 33 -> 38 -> 54), so its PATH CHOSEN was master and the 3.4% turn_1 row was noise. At NMAX 64 a hot expert with 130 routes becomes 3 chunks instead of 2. In wg2_kernel only, change the item decode so that the chunks of one expert are tile-interleaved: item index -> (expert group g, tile j, chunk-within-expert c) with c fastest, so the duplicate reads of weight tile (e, j) are issued by adjacent items, which run concurrently on neighbouring CTAs and hit L2 for the second and third chunk. Numerics are unchanged (each item still computes one (chunk, tile) output exactly as master). Variants: V64 (deep-ring item order) and V64i (interleaved). CHECKS: master's s5 bucket canary (N16/N32/N64), _wg_self_check and the three-state _wg_copy_check on each variant; rel-L2 vs Marlin <= 1e-2 per kernel and the copy-check bound as master. SELECTOR: extend master's _wg_select to candidates {Marlin, master s5, V64, V64i} for B256 and B128, all with the F2 front end if F2 is ON, full kernel_fn cold-L2 graphs, 41 interleaved reps, worse of zipf a=0.3 and the judge-like routing; a variant replaces master's chosen path for a bucket only at <= 0.98x; print per variant the w13 and w2 kernel-graph us, achieved TB/s on touched-expert bytes, chunk count and 'weight read twice' MB, and one PATH CHOSEN line per bucket. state['wg2_variant_128'] / state['wg2_variant_256'] (ints) pick the launch. ROUTING SUMMARY: T == 1 -> F1; 256 <= T <= 512 -> wgmma (master s5 or V64/V64i per selector) with F2; 128 <= T <= 255 -> Marlin unless the selector flips it; everything else master. Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C2 build 35-40 s (start deadline 140 s) + canary/self-check/copy check ~10 s + selector ~3 s (start deadline 230 s) = ~120 s of 300. RULES (every combo phase): kernel_fn branches on shape and prepare-set plain ints/bools only; no host sync, no input-value-sized allocation, no data-dependent control flow, no prints, no try/except, no per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights stay NVFP4 (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation; every local route computed; no child processes. The judge prepares TWO states and copies them by cloning tensors: every scratch tensor, stream, workspace and error word lives in the state, no module-global device pointer, tensor map or tensor, and tensor maps stay encoded per call (master's s5 rule). PREPARE ORDER: master's prepare() runs first and unchanged (align self-check, _align2_setup, _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 / WG_REPLAY_DEADLINE_S 235 / WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs AFTER it, measures its own elapsed time from prepare()'s t_prepare, skips itself (flag stays master, prints PATH CHOSEN = master (deadline)) if it would start past its start deadline, and no phase starts timing past 270 s. GATE RULE (every source): exactness or tolerance check first, then cold-L2 CUDA-graph timing vs master's path for that class (>= 256 MB flush before each replay, interleaved A/B, >= 21 reps, empty-graph baseline subtracted, median); ON only at new/master <= 0.98 on the worse of the judge-like skewed routing and random routing; otherwise that class keeps master. Print one PATH CHOSEN line per source per class with both medians and the ratio, and keep those lines in the last 16k characters of the log (print them again at the very end of prepare, after master's [align2] lines). Print per-phase prepare seconds and the total.

## Expected

- `bs=256 *` faster: F2 -1.5-2%, plus C2 if V64i passes: w13 ~239-303 us / w2 ~112-148 us at 2 CTAs/SM with duplicate tiles from L2; target verify_32 <= 385 us (-5%). If the selector keeps s5 the row equals F2 alone
- `bs=1 *` faster: F1: decode_1 +2.5-3%
- `bs=128 *` flat: B128 needs ~1.25x over master's wgmma to flip from Marlin; the PATH CHOSEN B128 line with four candidates is the readout
- `bs=32 *` flat: control
- `bs=16 *` flat: control

## Risk

Medium risk with a hang hazard that is guarded: setmaxnreg at OCC 2 only launches after numRegs/occupancy checks (the prefill source ran this pattern on the judge without fault), every spin is bounded by master's mbarrier timeout and error word, and any check failure keeps master's s5 kernels. The new code is the item decode (C2b) and the module plumbing; the kernel body is prefill's and the deep-ring branch's.

## Conjecture

Prefill measured ~11% from thread-level parallelism alone (two resident CTAs, same slab) and lost it all to re-reading the weights of experts split into more chunks under skew. This combination tests whether that loss is an ordering artefact: if the extra chunks of a hot expert read each weight tile at the same moment as their sibling, the second read is an L2 hit and the two-CTA gain survives skewed routing. It also tests a second interaction: 264 resident CTAs halve the item granularity per wave at B256 (2048 w13 items / 264 = 7.8 per CTA), which shrinks the last-wave tail that the queued wg-pdl and wg-stream-k primaries target, so a win here predicts a smaller win for those. Confirm with the selector prints: V64i must show a lower 'weight read twice' effective cost (TB/s on touched bytes near V64's random-routing figure) than V64 on the judge-like routing, and PATH CHOSEN B256 = V64i; refute if V64i and V64 time the same (L2 does not catch the duplicates) or both lose to s5 (the moe chunks are already too small for TLP to matter). The bs=256 profile then shows wg2_kernel rows in place of wg_kernel rows.
