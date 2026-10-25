---
problem: muse_moe_w4a16
slug: combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2
created: 2026-10-25T161613Z
base: master
area: "combination: naive top-row stack (decode_1 w13 atomic, decode_32 align+post-norm, verify_32 wgmma align2 front end)"
change: >
  COMBINATION (naive top-row stack): every per-shape winner that beat master on its own row and
  never reached master, each behind its own prepare-time gate, built on master a0e52640 (wgmma
  bucket gate scored 1.3727 + align-two-phase-port-from17). SHARED FLOOR (identical in all three
  combo hypotheses minted 2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1:
  w13 Marlin use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only,
  branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port
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
  warm per T is most of it). (C1a) decode_32: _align_parallel at 17 <= T <= 32. Source
  align-parallel-from-17-tokens, branch hyp/muse_moe_w4a16/align-parallel-from-17-tokens, impl
  commit c9c2db89 (job ed6d884240084d9c: decode_32 1.041x vs 1.031-1.035x in eleven
  master-equivalent runs; profile _align_parallel_kernel 2.6 us vs _align_small_kernel 4.1-4.3 us
  at T=32). Do NOT lower ALIGN_FUSED_MAX_TOKENS (it also names the align2 d32 incumbent). Instead
  add a third candidate to master's d32 class: state['d32_align'] in {0 = master's choice
  (align_small or align2 chunk, as _align2_setup set it), 1 = _align_parallel}; _moe_marlin calls
  _align_parallel for 17 <= T <= 32 when it is 1. Warm _align_parallel for every T 17..32 at the
  block_size_m each T gets (8 and 16), equivalence torch.equal on all four outputs against
  _align_small on (17, 24, 28, 29, 32) x three local_starts x six routings, component graph at
  T=32 (align alone) <= 0.98x vs master's d32 choice (expected ~0.6-0.7x). Start deadline 160 s,
  ~8 s. (C1b) decode_32: Triton post-norm at T == 32. Source post-norm-triton-hoisted-decode32,
  branch hyp/muse_moe_w4a16/post-norm-triton-hoisted-decode32, impl commit 4aab0187: port
  _post_norm_combine_hoisted_kernel and HOISTED_POST_NORM_TOKENS = 32 (job 47d186dd91974d84:
  decode_32 1.037x; kernel 3.8 us vs SGLang _post_expert_norm_weighted_sum 3.6-4.0 us plus its
  wrapper's extra launches). state['d32_post_norm'] in {0 = SGLang fused (master), 1 = hoisted};
  only num_tokens == 32 on the Marlin path. Gate: rel-L2 vs an fp32 post-norm of the same y <=
  1.5x SGLang's own error (floor 1e-5), on six routings incl. rows with zero local routes, then a
  component graph (post-norm alone, T=32) <= 0.98x. Start deadline 170 s, ~5 s. ROUTING SUMMARY: T
  == 1 -> Marlin with w13 atomic (F1) if ON; 17 <= T <= 32 -> master Marlin with align from C1a
  and, at T == 32, post-norm from C1b; 256 <= T <= 512 (and B128 if ON) -> master wgmma with the
  align2 front end (F2); every other T and every OFF flag -> master byte for byte. Expected
  prepare: master ~50 s + F1 3 s + F2 15-20 s + C1a 8 s + C1b 5 s = ~85 s of 300. RULES (every
  combo phase): kernel_fn branches on shape and prepare-set plain ints/bools only; no host sync,
  no input-value-sized allocation, no data-dependent control flow, no prints, no try/except, no
  per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights stay NVFP4
  (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation; every
  local route computed; no child processes. The judge prepares TWO states and copies them by
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
  - shapes: bs=1 *
    direction: faster
    why: "F1: decode_1 45.3 -> ~44.0 us (+2.5-3%, two prior runs at 1.754/1.766x vs master-equivalent 1.70-1.72x); Marlin row ~33.7 us"
  - shapes: bs=256 *
    direction: faster
    why: "F2: the four SGLang front-end rows (11.8 us) are replaced by _align2_count + _align2_place (~4.7 us): verify_32 404.9 -> ~397 us (-1.5 to -2%); wg_kernel rows unchanged"
  - shapes: bs=32 *
    direction: faster
    why: "C1a+C1b: align 4.1 -> 2.6 us and the hoisted post-norm, decode_32 203.8 -> ~201 us (+1-1.5%; each source measured +0.5-1% alone)"
  - shapes: bs=16 *
    direction: flat
    why: "control: master byte for byte"
  - shapes: bs=8 *
    direction: flat
    why: "control: master byte for byte"
  - shapes: bs=128 *
    direction: flat
    why: "control: Marlin (B128 selector keeps Marlin; F2 only applies if wg_on_128)"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2

## Change

COMBINATION (naive top-row stack): every per-shape winner that beat master on its own row and never reached master, each behind its own prepare-time gate, built on master a0e52640 (wgmma bucket gate scored 1.3727 + align-two-phase-port-from17). SHARED FLOOR (identical in all three combo hypotheses minted 2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1: w13 Marlin use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port W13_ATOMIC_MAX_TOKENS = 1, W13_ATOMIC_TOL, _w13_band, _w13_atomic_self_check and the w13_mode kwarg in the w13 moe_wna16_marlin_gemm call of _moe_marlin (w2 keeps master's w2_mode). Measured twice: decode_1 1.766x (job 171684977ec1470f) and 1.754x (job 1048c6edf3e54892) against master-equivalent 1.698-1.721x in eight runs (+2.5-3.0%; Marlin pair 33.7 vs 35.9 us). Gate: its own self-check (rel-L2 vs master <= 1e-2; it measured 8.4e-3) plus a component graph (the w13 Marlin call alone at T=1, skewed and random local routing) <= 0.98x. Phase F1 start deadline 120 s, ~3 s. (F2) verify_32: master's OWN align2 win is dead code on the judged path. Master's _align2_setup gates class v256 for _moe_marlin (job c05a8f7326004be6: align2 chunk 64 4.70 us vs incumbent 6.46 us, ratio 1.374, ON), but kernel_fn sends 256 <= T <= 512 to _moe_wgmma (wg_on_256 True), which still runs bank expert_map[topk_ids.to(int32)] plus SGLang moe_align_block_size(local_ids, WG_BSM=16, count): the c05a8f73 verify_32 profile shows index_elementwise 3.4 + unrolled_elementwise 3.1 + moe_align_block_size_kernel 2.8 + count_and_sort_expert_tokens 2.5 = 11.8 us in four launches and no _align2 kernel. Source: master a0e52640's own _align_two_phase / _align2_count_kernel / _align2_place_kernel (port origin align-two-phase-port-from17, impl commit fb879b90; prefill origin align-two-phase-all-t-rega). Change: in _moe_wgmma only, when state['wg_align2'] (a plain int chunk, 0 = master) is set, replace the expert_map gather and moe_align_block_size with local_ids, (sorted_ids, block_experts, n_padded) = _align_two_phase(topk_ids, start, count, WG_BSM, chunk). Gate F2: warm every T in 128..512 at block_size_m 16 for each chunk in (64, 128, 256, 512) <= the class top; equivalence on T in (128, 191, 192, 256, 257, 384, 512) x three local_starts x six routings (_align2_routings): local_ids torch.equal, n_padded equal, sorted_ids[:n_padded] and block_experts[:n_padded/16] torch.equal to master's front end (capacity tails may differ: the wg kernels read below n_padded only; assert n_padded/16 <= WG_MAX_BLOCKS), then full _moe_wgmma output torch.equal (bitwise: same kernels, same tables) on two of those routings per T; then the component graph (front end alone: gather+cast+align vs align2) <= 0.98x at T=256 (expected ~0.4-0.5x). Applies to whichever wgmma bucket is ON (B256 today; B128 only if its flag is on). Phase F2 start deadline 150 s, ~15-20 s (Triton constexpr warm per T is most of it). (C1a) decode_32: _align_parallel at 17 <= T <= 32. Source align-parallel-from-17-tokens, branch hyp/muse_moe_w4a16/align-parallel-from-17-tokens, impl commit c9c2db89 (job ed6d884240084d9c: decode_32 1.041x vs 1.031-1.035x in eleven master-equivalent runs; profile _align_parallel_kernel 2.6 us vs _align_small_kernel 4.1-4.3 us at T=32). Do NOT lower ALIGN_FUSED_MAX_TOKENS (it also names the align2 d32 incumbent). Instead add a third candidate to master's d32 class: state['d32_align'] in {0 = master's choice (align_small or align2 chunk, as _align2_setup set it), 1 = _align_parallel}; _moe_marlin calls _align_parallel for 17 <= T <= 32 when it is 1. Warm _align_parallel for every T 17..32 at the block_size_m each T gets (8 and 16), equivalence torch.equal on all four outputs against _align_small on (17, 24, 28, 29, 32) x three local_starts x six routings, component graph at T=32 (align alone) <= 0.98x vs master's d32 choice (expected ~0.6-0.7x). Start deadline 160 s, ~8 s. (C1b) decode_32: Triton post-norm at T == 32. Source post-norm-triton-hoisted-decode32, branch hyp/muse_moe_w4a16/post-norm-triton-hoisted-decode32, impl commit 4aab0187: port _post_norm_combine_hoisted_kernel and HOISTED_POST_NORM_TOKENS = 32 (job 47d186dd91974d84: decode_32 1.037x; kernel 3.8 us vs SGLang _post_expert_norm_weighted_sum 3.6-4.0 us plus its wrapper's extra launches). state['d32_post_norm'] in {0 = SGLang fused (master), 1 = hoisted}; only num_tokens == 32 on the Marlin path. Gate: rel-L2 vs an fp32 post-norm of the same y <= 1.5x SGLang's own error (floor 1e-5), on six routings incl. rows with zero local routes, then a component graph (post-norm alone, T=32) <= 0.98x. Start deadline 170 s, ~5 s. ROUTING SUMMARY: T == 1 -> Marlin with w13 atomic (F1) if ON; 17 <= T <= 32 -> master Marlin with align from C1a and, at T == 32, post-norm from C1b; 256 <= T <= 512 (and B128 if ON) -> master wgmma with the align2 front end (F2); every other T and every OFF flag -> master byte for byte. Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C1a 8 s + C1b 5 s = ~85 s of 300. RULES (every combo phase): kernel_fn branches on shape and prepare-set plain ints/bools only; no host sync, no input-value-sized allocation, no data-dependent control flow, no prints, no try/except, no per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights stay NVFP4 (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation; every local route computed; no child processes. The judge prepares TWO states and copies them by cloning tensors: every scratch tensor, stream, workspace and error word lives in the state, no module-global device pointer, tensor map or tensor, and tensor maps stay encoded per call (master's s5 rule). PREPARE ORDER: master's prepare() runs first and unchanged (align self-check, _align2_setup, _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 / WG_REPLAY_DEADLINE_S 235 / WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs AFTER it, measures its own elapsed time from prepare()'s t_prepare, skips itself (flag stays master, prints PATH CHOSEN = master (deadline)) if it would start past its start deadline, and no phase starts timing past 270 s. GATE RULE (every source): exactness or tolerance check first, then cold-L2 CUDA-graph timing vs master's path for that class (>= 256 MB flush before each replay, interleaved A/B, >= 21 reps, empty-graph baseline subtracted, median); ON only at new/master <= 0.98 on the worse of the judge-like skewed routing and random routing; otherwise that class keeps master. Print one PATH CHOSEN line per source per class with both medians and the ratio, and keep those lines in the last 16k characters of the log (print them again at the very end of prepare, after master's [align2] lines). Print per-phase prepare seconds and the total.

## Expected

- `bs=1 *` faster: F1: decode_1 45.3 -> ~44.0 us (+2.5-3%, two prior runs at 1.754/1.766x vs master-equivalent 1.70-1.72x); Marlin row ~33.7 us
- `bs=256 *` faster: F2: the four SGLang front-end rows (11.8 us) are replaced by _align2_count + _align2_place (~4.7 us): verify_32 404.9 -> ~397 us (-1.5 to -2%); wg_kernel rows unchanged
- `bs=32 *` faster: C1a+C1b: align 4.1 -> 2.6 us and the hoisted post-norm, decode_32 203.8 -> ~201 us (+1-1.5%; each source measured +0.5-1% alone)
- `bs=16 *` flat: control: master byte for byte
- `bs=8 *` flat: control: master byte for byte
- `bs=128 *` flat: control: Marlin (B128 selector keeps Marlin; F2 only applies if wg_on_128)

## Risk

Medium risk: four ports, all of measured code, each behind its own gate and falling back to master per class. Expected headline ~+0.8-0.9% (1.3727 -> ~1.384), inside the +-1.5% CI, so this run is mainly the attribution of three small row wins measured together; it beats the board only with favourable drift. Failure modes are known: F1's rel-L2 8.4e-3 is inside 1e-2; align2 and _align_parallel are bit-identical by construction and checked with torch.equal; the hoisted post-norm already passed the judge.

## Conjecture

The sources are shape-disjoint, so the naive prediction is that their row gains add (decode_1 +2.5%, decode_32 +1-1.5%, verify_32 +1.5-2%). The interaction under test is between adjacent small launches and the selectors that time them: (1) at decode_32 the align and the post-norm sit at the two ends of the Marlin pair; if their savings come from the same graph-node gap rather than kernel time, the stacked row gains only ~1% instead of ~1.5%; (2) at verify_32 the B256 selector in master's _wg_setup timed wgmma with the SGLang front end, so with align2 the real margin over Marlin grows from ~1.05x to ~1.07x, and re-running _wg_select after F2 (print it as a diagnostic line, do not let it flip a bucket ON that master had OFF without the 0.98x rule) shows whether B128 moves toward a flip. Confirm or refute with: the bs=256 profile (the index_elementwise / unrolled_elementwise / moe_align_block_size / count_and_sort rows gone, _align2_count and _align2_place present, wg_kernel rows within +-2%); the bs=32 profile (_align_parallel_kernel and _post_norm_combine_hoisted_kernel rows, Marlin row unchanged); the bs=1 Marlin row near 33.7 us; and the PATH CHOSEN lines F1, F2, C1a, C1b. If decode_32's shape time improves by less than the sum of the two kernel deltas, the gap hypothesis holds.
