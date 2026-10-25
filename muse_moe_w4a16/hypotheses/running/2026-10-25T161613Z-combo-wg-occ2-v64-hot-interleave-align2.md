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
review: proceed
status: running
review_at: 2026-10-25T162203Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T162203Z
base_sha: 757927698c5069efecc0af3dde3ee3553dc1971f
base_blob: 593b238c16387c5ac3774424ee9557882b9fc46a
branch: hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2
worktree: .worktrees/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T190500Z
impl_commit: 195ad2846671802fa04ad1eef44155ef4c063b35
submission_sha256: e1ffa95d1d7c94de8249a6f0837834ac8df76369d21283c41f408d58258b25d9
norm_sha256: 809db5d96e93c4ae71eb08349d36c0a9c3431047e7432f5331531074146f3406
submission_blob: 1f96da53a3e3c191b471b7c49f22e098069ba50c
lint_warnings: 100
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

## Adversarial review

Reviewed 2026-10-25T162203Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed only after the required edits below. The conjectural dose (C2b, V64i) is not new code. Master's decode_item (submission.py:1023-1038) and prefill's rega_lo_kernel decode_item (dfb755d3, line 2330) already order items with the chunk-within-expert index fastest, so V64i == V64, and prefill's skew loss (0.994x) was measured WITH the interleave in place. Drop V64i, and gate C2 on the reference-rebuilt judge routing, not master's zipf. As edited, this is an interpretable F1 + F2 run plus a low-odds C2 B256 candidate that its own prepare lines read out.

### Justification
A. F1 (w13 atomic add at T==1): measured twice on 9080a71, +2.5-3.0% at decode_1. That is roughly +0.45% on the headline, at the edge of the +-1.5% CI, but cheap and self-checked. F2 (align2 front end in _moe_wgmma): sound and well evidenced. The c05a8f73 verify_32 profile shows the 4-launch gather+cast+moe_align_block_size front end (~11.8 us) and no _align2 kernel. Master's v256 gate measured align2 at 4.70 vs 6.46 us, but at bsm 32 for Marlin; wgmma needs bsm 16, and at bsm 16/T=128 align2 LOST (0.781). So F2's own bsm-16 component gate at T=256 is required, and the spec has it. Expect -5 to -7 us of 405 us (-1.3 to -1.7%). The bitwise equivalence gate (sorted_ids[:n_padded], block_experts, n_padded, then full _moe_wgmma torch.equal) is the right strength. C2a (two CTAs/SM, NMAX 64, STG 3): the register arithmetic is right: 32x128 + 104x256 = 30720 = 80x384, and 2x30720 <= 65536. Smem fits: w2 1024 + 3x25600 + 17408 + 8192 static + 1 KB reserve = ~104.4 KB/CTA, 2 CTAs <= 228 KB. Both halves of the kernel have run on the judge: moe n64s7 at OCC1 (c30fd0cc, 168 regs, 0 spill) and prefill rega_lo at OCC2 (80 regs). But the prior evidence says C2 probably will not win at B256 under skew. Prefill measured TLP at ~0.887x on random routing and 0.994x under pop:2 skew. moe's deep-ring log shows the reference-rebuilt judge routing at T=128 has 17/32 touched experts and max 65 routes on one expert (so ~130 at T=256: 3 NMAX-64 chunks vs 2). WEAKEST LINK, C2b: the 'hot-expert tile interleave' already exists. In decode_item, cg -> (f = gf[cg], k = gk[cg]); c = f + local % k; j = local / k. Chunks of one expert are already tile-interleaved, c fastest, in master's s5 kernel, in the deep-ring kernel the hypothesis calls 'V64 (deep-ring item order)', and in prefill's rega_lo_kernel. So V64 and V64i are the same schedule. The Conjecture ('the skew loss is an ordering artefact') is already refuted by prefill's own gate, which ran with this order. The run cannot test it as written. V64 == V64i timing would read as 'L2 does not catch the duplicates' for the wrong reason. Second weak link, the gate routing: master's B256 'judge-like' routing is a calibrated zipf a=0.3 (_wg_select, submission.py:2386-2445; 457 local routes, all 32 experts touched, max ~30 routes per expert at T=256). NMAX 64 never splits a chunk under it or under random routing. So a 'worse of zipf and random' gate would see only the TLP gain and could ship V64 into the judge's hot-expert routing, where prefill says the gain vanishes. That is exactly the mis-forecast mode the caller asked about. The selector is trustworthy only if the gating set includes the reference.generate_input rebuild (_wg_judge_ids from c30fd0cc: dist normal, busiest-GPU rotation, 8 fixed-seed draws, ~17 touched experts). On trust in general: at B256 master's selector forecast the judge well (Marlin pair 497-500 vs judged 483 us, ratio 1.047-1.054x forecast vs 1.06x judged row). At B128 it did not: calibrated Marlin pair 381 vs judged 306 us (24% off), and the forced s5 B128 lost on the judge. So the gate can be trusted at B256 with the reference rebuild in the set, and cannot be trusted at B128. Expected rows: 'w13 ~239-303 / w2 ~112-148 us' does not state a gain. Master is already 239.2 / 119.0 us at bs=256. Reaching <= 385 us needs C2 to save ~13 us on top of F2. If PATH CHOSEN B256 = s5 (the likely outcome), expect verify_32 ~ -1.5% and decode_1 ~ +2.5%. Both are within or near the CI for a single run, so read attribution from the profile rows, not the headline.

### Confounders on the current kernel
- align-two-phase-port-from17 (landed, a0e52640): F2 reuses its _align_two_phase/_align2_count/_align2_place kernels and extends them to the wgmma path. Its scored verify_32 '+13.4%' was measured against 9080a71 and is really the wgmma bucket's gain (align2 never ran at T=256 on the judged path). The harvest must attribute against c05a8f73 / 4e4a0e26, never 9080a71, or F2 will be credited with wgmma's win again.
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (on master's lineage via 40e2e860; incumbent at B256): C2's selector compares against it, and F2 changes the front end both candidates share. The selector must time master s5 WITH F2 when F2 is ON (the spec says so), or C2's ratio absorbs F2's ~6 us.
- w2-marlin-atomic-add-small-m (landed): F1 pairs w13 atomic add with w2's atomic add at T==1. F1's two measurements were on 9080a71, which already had the w2 atomic, so this is consistent. If w2_mode falls back to master at prepare, F1's self-check tolerance still holds, but the +2.5% was never measured in that configuration.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: these set Marlin's bsm (16 at T=128, 32 at T=256) on the Marlin candidate the selector also times. They do not touch the wgmma rows, so the B256 attribution is not confounded.
- post-expert-norm-triton-small-m / sglang-marlin-path-parity / align-parallel-per-expert-verify: no shared launches with F1/F2/C2 on the shapes they move (the bs=256 combine is SGLang's fused_post_expert_norm in both paths). Not confounding.

### Interference and A/B needs
No companion is warranted. The three parts are separable inside one run. F1 is bs=1 only. F2 shows in the bs=256 profile as two _align2 launches replacing index_elementwise + unrolled_elementwise + moe_align_block_size_kernel + count_and_sort (~11.8 us), and C2 as wg2_kernel rows replacing wg_kernel rows. The PATH CHOSEN lines with both medians say which ran. Both a gain and a loss can be read from the profile and the prepare log without another scored run. The one question a companion could answer (does the existing c-fastest order already make L2 catch duplicate tile reads?) costs nothing extra if the V64i slot becomes a print-only de-interleaved diagnostic (V64s, chunk index slowest) timed in the same selector. A companion run would buy nothing more.

### Open questions
- REQUIRED EDIT 1: drop V64i as a 'new interleave'. It is master's existing decode_item order. Either remove the candidate, or replace it with a print-only V64s (c slowest: item -> (expert group, chunk, tile)) timed under the reference-rebuilt routing. V64s vs V64 then measures what the existing interleave already buys from L2 and actually tests the Conjecture's mechanism. Rewrite the Conjecture to the testable claim: 'OCC2 TLP at NMAX 64 survives the judge's hot-expert split at B256'.
- REQUIRED EDIT 2: C2's gating routings must include the reference.generate_input rebuild (_wg_judge_ids, ported from c30fd0cc, fixed CPU seed, touched-expert count printed). Master has no such routing: its 'judge-like' is a zipf a=0.3 with all 32 experts touched, and NMAX 64 never splits under it. Gate = worse of {reference rebuild, zipf a=0.3}, with random printed only. Print chunk count and 'weight read twice' MB per candidate per routing, and assert the chosen candidate is the fastest printed on the gating routing.
- REQUIRED EDIT 3: make B128 print-only for C2: never enabled from this run. The B128 forecast is untrustworthy (calibrated Marlin pair 381 vs judged 306 us; forced s5 lost on the judge). Prior B128 wgmma variants sat at 0.80-0.81x of Marlin, and ~11% TLP cannot reach the 1.02x gate. B128 is also the scope of queued wg-two-ctas-per-sm-port-b128-n32, so do not pre-empt it.
- Selector multiple-comparison bias: picking the min of {s5, V64} (plus a diagnostic) at a 0.98 margin on 41 reps is mild winner's curse. B256's forecast error has been ~1-3%, so keep 0.98 but require the win on BOTH gating routings, as written.
- Does nvcc exist on this workstation for an offline sm_90a compile of the new module (no GPU needed)? That is the only pre-submit check available for the numRegs == 80 / no-spill / setmaxnreg-warning conditions.
- Fix the expected rows: master is already at w13 239.2 / w2 119.0 us at bs=256, so 'w13 ~239-303 / w2 ~112-148' is not a gain. State the C2 target as w13+w2 <= ~345 us (-13 us) if V64 is chosen. Otherwise verify_32 is expected to be F2 alone (~ -1.5%).
- If C2 loses under the reference rebuild, as prefill predicts, the follow-up that could make two CTAs pay under skew is w13 at OCC2 with NMAX 128 and STG 2 (1024 + 2x41984 + 18432 + 8192 + 1024 ~ 110 KB/CTA, fits 2/SM; w2's 34816-B epilogue does not fit). That keeps hot experts in 2 chunks and is worth a separate hypothesis, not part of this run.


## Implementation notes

Base: worktree base 757927698c, then 3-way merged (git merge-file, base = 757927698c's file) with master f7eac2c3 (wg-pdl-gather-w13-w2-chain landed) at the coordinator's request; PDL kept everywhere. 4 conflict hunks (`_wg_gemm`, `_wg_copy_check` signature/docstring/flags) resolved by combining both: `_wg_gemm` keeps master's pdl / wg_ext_pdl logic and adds `variant`; `_wg_copy_check(..., pdl=False, variant=0)` sets both `wg_pdl` and the wg2 variant flags. Everything below runs AFTER master's prepare() work, which is unchanged (`_combo_phases`, order F1 -> C2 build -> F2 -> C2 checks + selector; each phase self-deadlined, try/except in prepare only, flags stay master on any failure; PATH CHOSEN lines re-printed after master's [align2] lines with per-phase seconds and the total).

- F1 (w13 Marlin atomic at T==1, port of f29a0b9e): `_moe_marlin(..., w13_mode, poison)`; `state['w13_mode']` in {master, atomic, atomic_zero}; kernel_fn passes it, `_moe_marlin` applies it only at num_tokens <= W13_ATOMIC_MAX_TOKENS (1). Gate `_f1_setup`: poisoned output buffer, 3 skewed (8 local routes) + 3 random T=1 inputs, rel-L2 vs current master <= 1e-2 AND (sibling review fix) vs a local fp32 reference (reference.ref_kernel's math, no import) <= 1e-2; master's own fp32 error printed. Then the w13 Marlin call alone as cold-L2 graphs (`_align2_gate_time`, 41 interleaved reps, empty baseline subtracted), ON at worse(skewed, random) cand/master <= 0.98. Start deadline 120 s.
- F2 (align2 front end in `_moe_wgmma`): new `_wg_front(topk_ids, bank, front)`; `state['wg_align2']` int (0 master gather+moe_align_block_size, >0 align2 chunk, -1 `_align_parallel` bsm 16 as the sibling review's extra candidate). Gate `_f2_setup` (sibling review fix applied): warm align2 for every T 128..512 x chunks (64,128,256,512) at bsm 16 (+ `_align_parallel` bsm 16); equivalence on T (128,191,192,256,257,384,512) x 3 local_starts x six routings: `_align_parallel` bsm 16 checked per expert against SGLang (master's `_align_self_check` logic, `_f2_semantic`), then each align2 chunk torch.equal on all four outputs against `_align_parallel` bsm 16 (same capacity), plus wgmma capacity limits (n_padded/16, block table <= WG_MAX_BLOCKS, rows <= WG_CAP_ROWS); then full `_moe_wgmma` output BITWISE equal (torch.equal, no rel-L2 fallback, per the caller) to master's front end on random + skewed routings per T; then the front end alone at T=256 as cold-L2 graphs on the reference rebuild and random routing, ON at worse ratio <= 0.98 for the fastest candidate (asserted fastest printed). Runs only if master's `wgmma_ok`; applies to whichever wgmma bucket is ON.
- C2 (two CTAs/SM): separate module `muse_moe_w4a16_wgmma_occ2_v1` built after master's setup from master's wg head verbatim (now including the PDL griddep helpers) + new body: `wg2_kernel<G, NMAX=64, STG=3, OCC=2, ORD>` = master's s5 kernel with c30fd0cc's Geo<NMAX,STG> ring and chunk cut every NMAX/16 blocks (buckets above NMAX never instantiated; an unreachable bucket > BMAX sets error word 3 and is skipped by producer and consumers alike) and dfb755d3's OCC-2 pieces (`__launch_bounds__(384, 2)`, setmaxnreg dec 32 / inc 104, GeoLo/SmemLo static_asserts ported incl. `32*128 + 104*256 == 80*384`, register-file and 2-CTA smem budgets, w2 95232 B / w13 87040 B dynamic). Own launch marker 0x57473232. Same PDL call sites as master (griddep_wait before the first slab TMA and before the consumers' loop, launch_dependents at the end) and the same w2-only `cudaLaunchKernelEx` programmatic-serialization launch when `state['wg_pdl']`. Per-state tensor maps: encoded per call from the state's own wg_xs/wg_hs into a stack-local grid constant; no file-scope or static variable (the encode entry point is resolved per call); `_wg_audit_cpp` run on the new source (CPU check here: 0 findings, master 0). `state['wg2_grid']` = 2 x SMs.
- Hang safety (review edit 4): before any wg2 launch, `wg2_init` reads cudaFuncGetAttributes / occupancy for all four instantiations; a variant is usable only if numRegs == 80, localSizeBytes == 0, CTAs/SM >= 2, static <= 8192 and 2 x (static + dynamic + 1 KB) <= 228 KB, binaryVersion 90. The build is first `-Xptxas -Werror`; any ptxas line mentioning setmaxnreg = C2 OFF; another warning allows one plain rebuild only if before 160 s. V64 failing = C2 OFF with no wg2 launch ever made.
- Checks (`_c2_checks`): V64: master's bucket canary on N16/N32/N64 (`n_buckets=3`), `_wg_self_check(variant=1)`, three-state `_wg_copy_check(variant=1, pdl=state wg_pdl)` (real state's flags incl. wg_pdl saved/restored). V64s: bucket canary only (print-only). Master's check functions gained default-preserving `variant` args only.
- Review edit 1: V64i dropped (it is master's decode_item order). Added print-only V64s (ORD 1: chunk index slowest, tile fastest; CPU emulation confirms both orders are bijections). The selector prints V64s/V64 on the judge rebuild as the in-run L2 test. Conjecture restated: two CTAs per SM at NMAX 64 still pays at B256 once the judge's hot experts are split into extra chunks (refuted if V64/master's choice > 0.98 on the reference rebuild).
- Review edit 2: `_wg_judge_ids` ported from c30fd0cc (Student-t popularity, busiest-GPU rotation, 8 fixed-seed draws, nearest 17 touched). C2 gates on the WORSE of that routing and zipf a=0.3; random printed only.
- Review edit 3: B128 timed and printed, never enabled (only `wg2_variant_256` can become 1; `wg2_variant_128` stays 0).
- Selector `_c2_select`: full-path graphs (Marlin, s5, V64, V64s) with the final F2 front end and master's PDL setting, cold L2, 41 interleaved reps, empty baseline subtracted; per candidate and routing: touched experts, max routes, chunks, experts split, weight read twice MB, w13/w2 kernel-graph us and TB/s on touched bytes. B256 moves to V64 only if healthy (err 0, marker, finite, rel <= 2e-2 vs Marlin), V64 / master's chosen path <= 0.98 on both gating routings AND V64 is the fastest selectable printed candidate on both (asserted). If master chose Marlin and V64 wins, wg_on_256 is set True with V64.
- Review edit 5: nvcc/ptxas are NOT available on this node (no nvcc or ptxas anywhere, no torch), so no offline sm_90a compile / -Xptxas -v regs/spill report was possible; the runtime numRegs/localSizeBytes gate is the only check.
- Expected rows (review): master is already w13 239.2 / w2 119.0 us at bs=256; C2 target is w13+w2 <= ~345 us if V64 is chosen, else verify_32 is F2 alone (~ -1.5%); decode_1 F1 ~ +2.5%.
- Deviation: the C2 build is ordered before F2 (build deadline 140 s < F2's 150 s); V64s gets only the canary (not self-check / copy check) to save prepare time since it is never selectable.

## Rules check

- [x] CUDA-graph capturable: kernel_fn branches on shape and prepare-set plain values only (w13_mode str, wg_align2 / wg2_variant_* ints, wg_on_* / wg_pdl bools); new kernel-path code (`_wg_front`, `_moe_wgmma`, `_wg_gemm`, `wg2_gemm`) has no .item/.cpu/sync/print; align2/_align_parallel allocate shape-sized tensors only.
- [x] Nothing derived from data cached: tensor maps encoded per call from the state's own scratch; no module/static device pointer (audit 0 findings for both sources).
- [x] Weights: wg2 reads master's lossless retiled NVFP4 bank (register dequant only); NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps them; fp32 accumulation; F1 only changes Marlin's reduction mode (fp32 reduce kept).
- [x] n/a: attention (MoE problem).
- [x] No timing patches, harness access, reference import (fp32 ref written locally), subprocesses (load_inline as master), network.
- [x] All work inside the timed call; no output caching; every local route computed.
- [x] No prints, mutable state or try/except in kernel_fn (trace prints only when prepare passes a tracer).
- [x] Judge's two states: everything per state (wg_xs, wg_hs, wg_err, wg2_grid int, ext modules shared); copy check on A/deepcopy B/clone C for V64.

okc_lint: `OK with warnings` (only prepare-side host-sync and uint8 scratch dtype warnings, same classes as master). py_compile OK; custom undefined-name scan 0 suspects; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus). Read only: no GPU, nvcc or torch on this node. **Verdict: dispatch-with-fixes.**

- **Fix 1 (gate that would almost always fail): F1 tolerances raised from 1e-2 to 1.25e-2** (`W13_ATOMIC_TOL`, `W13_FP32_TOL`). The source measured 8.4e-3 vs master on ONE routing. The sibling combo's judge run (b395f5b6, same F1 gate) logged a worst-of-6 vs master of 1.07e-2 and vs fp32 of 8.55e-3. At 1e-2 F1 would keep master. 1.25e-2 matches the sibling's audited value and is still inside the judge's 1.5e-2.
- **Fix 2 (fallback cascade): phase order changed to F1 -> F2 -> C2 build -> C2 checks + selector, as the spec orders it.** The builder had moved the C2 build ahead of F2. Builds on the judge take 31-51 s (master's module: 37.2 s), and wg2 has 4 instantiations. A failed -Werror build followed by the plain rebuild (allowed up to 160 s) would have pushed F2 past its 150 s start deadline, so a C2 build problem could silently cost F2, the surest gain. C2 still has its own 140 s build deadline, and its checks and selector now see the final front end.
- **Merge (f7eac2c3 PDL):** every `-` line in diff(master, worktree) is a combo signature or call-site edit. No PDL logic is lost. `_wg_gemm` keeps master's `pdl` / `wg_ext_pdl` dispatch for variant 0 and passes `pdl` to `wg2_gemm`, which takes it in the right position. `_wg_copy_check` keeps `st["wg_pdl"] = bool(pdl)`, adds the variant flags and the WG2_MAGIC marker, and shares both ext modules in the deepcopy memo. With every flag at master, kernel_fn issues master's launches byte for byte.
- **C2 isolation and hang safety:** wg2 builds as a separate load_inline name, after master's module is already loaded. TORCH_CUDA_ARCH_LIST is restored in `finally`. A build exception, a setmaxnreg ptxas line, an attribute-gate failure (numRegs == 80, localSizeBytes 0, CTAs/SM >= 2, smem fit, binaryVersion 90), or any raise in a phase leaves `wg2_variant_*` = 0 and restores `wg_on_256`, and no wg2 launch is ever made before the gate. The register split is exact: 32x128 + 104x256 = 80x384, 2 CTAs <= 64K. setmaxnreg.inc therefore cannot starve once numRegs == 80. Every mbarrier spin is bounded by master's 2^26-cycle timeout and error word, and an unreachable bucket > BMAX is skipped by the producer and consumers alike. PDL with 2 CTAs/SM cannot deadlock, because launch_dependents fires only after every primary CTA is resident. The wg2 source passes `_wg_audit_cpp` with 0 findings (recomputed here), its braces balance, and it closes one anonymous namespace.
- **Prepare time:** master, including its PDL stage, logged 60.1 s (667b64e3, b395f5b6). Expected: F1 ~1-3 s, F2 ~15-25 s, C2 build ~45-60 s (~100 s more with a rebuild), checks ~15-25 s, selector ~5-10 s, so ~150-180 s. Start deadlines (C2 build 140 s, checks 200 s, selector 230 s, timing 270 s) bound the worst case to about 250 s, under 300 s.
- Rules: kernel_fn branches only on shape and prepare-set ints and strings. `_wg_front` / `_align_two_phase` allocate shape-sized buffers only, with no sync or print. Weights stay NVFP4 (register dequant only). Tensor maps are encoded per call. The judge's state copy shares `wg2_ext` like `wg_ext`. okc_lint: OK with warnings (prepare-side only). py_compile OK, the AST undefined-name scan found 0, and __pycache__ was removed.

Residual risk: the wg2 module has never been compiled. If ptxas does not land exactly on 80 registers, or spills, C2 turns itself off and the run reads as F1 + F2 only. A device fault inside a wg2 launch (as opposed to a hang) would still be sticky and would fail the run.
