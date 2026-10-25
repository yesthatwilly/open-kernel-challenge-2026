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
review: proceed
status: queued
review_at: 2026-10-25T162036Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T162036Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, but F2's equivalence gate has to be rewritten before the build. As written it compares align2's tables torch.equal against SGLang's moe_align_block_size, and they never match on any routing with remote routes, so F2 would always gate OFF and verify_32 would read flat for a spec reason. The wgmma kernels are layout-agnostic, so align2's tables are correct for them. The fix is a semantic check (master's own per_expert logic) plus a bitwise check on the _moe_wgmma output.

### Justification
A. F2 mechanism is real. kernel_fn sends 256..512 tokens to _moe_wgmma whenever wg_on_256 is set (master's logs show B256 WGMMA in all four archived runs). _moe_wgmma still runs expert_map[topk_ids.to(int32)] and SGLang's moe_align_block_size(local_ids, 16, count). Meanwhile master's align2 v256 flag (chunk 64, 4.70 vs 6.46 us at bsm 32) only reaches _moe_marlin, which is dead code at bs=256. Replacing a ~11.8 us 4-launch front end with a ~4.7 us 2-launch one is about 7 us of ~405 us (~1.7%). That is below the row-level drift, so the profile rows are the evidence and the headline is not. EXTRA_NOTE check, which is the weakest link. SGLang's moe_align_block_size(local_ids, bsm, 32) gives remote (-1) routes padded m-blocks of their own, with block_experts = -1. Master's self-check logs this as r_pad > n_pad DIFF in most cases (e.g. T=8 all-remote n_pad=0 vs r_pad=64; triton-map-align-on-block-m-restack's review counted 48/60 cases DIFF). SGLang's within-expert order also comes from atomics. align2, like _align_small and _align_parallel, drops -1 routes, orders routes ascending within an expert, uses capacity n + count*(bsm-1) instead of SGLang's n + (count+1)*(bsm-1), and writes -1 past the used blocks. So n_padded, sorted_ids[:n_padded] and block_experts[:n_padded/16] are NOT torch.equal to master's wgmma front end on any routing with a remote route, and every judged routing has them (32 of 256 experts local). Is this a correctness bug in the outputs? No. I read the wgmma consumers. (1) wg_gather returns for slot >= n_padded and for be outside [0, n_experts), and zero-fills sentinel slots (sid >= num_routes). align2's sentinel is n = numel, the same value SGLang uses. (2) wg_kernel reads min(n_padded/16, blocks_len, MAXB) block_experts and treats any id outside [0, n_experts) as non-local. (3) Its chunks are maximal runs of one local expert cut every 8 blocks, so per-expert chunk sizes and the run-time wgmma N choice match SGLang's layout exactly; only block positions shift. (4) The w2 epilogue scatters to cache3[route], and the post-norm masks by local_ids, which is identical. (5) The host checks hold: at T=512, align2 has 286 blocks (<= WG_MAX_BLOCKS 384), and sorted_ids.numel() 4576 <= WG_CAP_ROWS 8192, with block_experts.numel()*16 >= sorted_ids.numel(). Swap-AB rows are independent dot products over the same K order, so the _moe_wgmma output should be bitwise equal. The table-equality step of the gate is the part that is wrong. Required amendment: drop torch.equal on n_padded, sorted_ids and block_experts. Replace it with master's _align_self_check per_expert semantics against SGLang: per-expert sorted route multisets equal; per-expert block counts equal over the be >= 0 blocks; n_padded == 16 * sum of local blocks; block_experts[:used] all >= 0 and non-decreasing, with -1 for the tail; sentinel fill (count of n == len - real, none > n); local_ids torch.equal; n_padded/16 <= WG_MAX_BLOCKS. Then keep full _moe_wgmma output torch.equal against master's _moe_wgmma, falling back to rel-L2 <= 1e-3 if torch.equal ever fails on an order-only cause; never gate on finiteness alone. Equivalently, compare align2 at bsm 16 torch.equal against _align_parallel at bsm 16, which master already validates per-expert against SGLang, and that suffices. F1 (w13 atomic at T=1): measured twice on the judge, +2.5-3%. However, its gate is rel-L2 vs master (8.4e-3), and master already carries its own error vs fp32. At T=1 the only row is the whole output, under a 1.5% bound. Require F1's self-check to also measure rel-L2 vs an fp32 reference (C1b already has a dequant reference to build on) and stay OFF above ~1.0e-2 vs fp32. C1a/C1b: each was measured at about +0.5-1% at decode_32, and the gate ratios are plausible (_align_parallel 2.6 vs _align_small 3.55-4.3 us; align2 d32 is OFF on master at ratio 1.000, so the incumbent is _align_small). Expected directions agree with TASK.md: all six loads have weight 60, and only bs=1/32/256 should move. The ~+0.85% headline is inside the ±1.5% CI. The conjecture's decode_32 'graph-gap' test is weak: one run cannot tell 'less than the sum of kernel deltas' from ±1-2% row drift. Only the profile rows (kernel us) can tell kernel savings from gap savings, and only roughly.

### Confounders on the current kernel
- align-two-phase-port-from17: provides the _align_two_phase kernels F2 reuses. Its v256 gate and timing ran at bsm 32 against _align_parallel on the Marlin path, not at bsm 16 against the SGLang front end. Its 1.374 ratio does not transfer, so F2 needs its own bsm-16 warm, equivalence and timing (the hypothesis does plan this).
- align-parallel-per-expert-verify / triton-map-align-on-block-m-restack: they established that the dropped-remote-blocks layout is the master norm on the Marlin path. Their logs (r_pad > n_pad) are why F2's torch.equal-vs-SGLang gate cannot pass. On the wgmma path, removing SGLang's -1 blocks also shortens the block_experts scan and the gather grid's work, so a small part of any verify_32 gain could come from that rather than launch count. It is visible as wg_gather / wg_kernel rows moving by more than 0, and is not a reason for a companion.
- triton-map-align-on-block-m-restack (the _align_small at <=32 path): C1a replaces exactly this at 17..32. On that base most of the earlier align gains were Marlin work removed (fewer m-blocks), but _align_parallel and _align_small produce identical layouts. So C1a's gain is align kernel time only, and the Marlin row should be unchanged; if it moves, that is drift.
- post-expert-norm-triton-small-m: sets SMALL_M_POST_NORM = 32, so T == 32 still uses SGLang's fused norm. C1b replaces exactly that one shape, so there is no overlap, but C1b and C1a land on the same shape (decode_32) and cannot be separated by the headline.
- w2-marlin-atomic-add-small-m: w2 at T=1 already runs atomic (state w2_mode). F1 adds atomic on w13 at T=1, so both Marlin calls at decode_1 now use atomic reduction, and F1's error stacks on w2's. This is why F1's gate must be vs fp32, not vs master alone.
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate (BEST, master's wgmma path): F2 changes only its front end. If wg_on_256 comes out False in this run (the selector timed under prepare noise), F2 is moot and verify_32 runs Marlin with align2 v256 as on master. Read the B256 PATH CHOSEN line before attributing anything at bs=256.

### Interference and A/B needs
No companion is warranted. The three sources are shape-disjoint: F1 is T==1, C1a/C1b are 17..32 and T==32, and F2 is the wgmma bucket only. Each has its own prepare-time gate that prints PATH CHOSEN with medians, and bs=8/16/128 are byte-for-byte master controls. Attribution comes from the profile rows: at bs=256 the index_elementwise / unrolled_elementwise / moe_align_block_size / count_and_sort rows should be gone and _align2_count / _align2_place present; at bs=32, _align_parallel_kernel and _post_norm_combine_hoisted_kernel should appear; the bs=1 Marlin row should sit near 33.7 us. An ab-revert of any landed hypothesis would not make a gain or a loss easier to read. The only intra-shape ambiguity is C1a vs C1b at decode_32, and the kernel rows already separate it; a control would cost a scored run to settle a <1% split that is inside drift. The real risk to interpretability is F2 silently gating OFF because of the table torch.equal; the required gate amendment fixes that, not a companion.

### Open questions
- F2 gate (required change): replace the torch.equal on n_padded / sorted_ids[:n_padded] / block_experts[:n_padded/16] vs SGLang's moe_align_block_size with per-expert semantic equality (master's _align_self_check per_expert logic), or with torch.equal against _align_parallel at bsm 16. Then keep the bitwise _moe_wgmma output check against master's front end. Under the spec as written, every routing with a remote route fails and F2 is always OFF.
- Does the bsm-16 align2 at T=256 actually beat SGLang's front end by about 2x in the component graph? It was measured at bsm 32 against _align_parallel, not against the 4-launch SGLang path. Also time _align_parallel at bsm 16 as a second candidate: it is already validated, and at v128 it beat align2 (ratio 0.781).
- Does master's prepare finish early enough (< 120 s) for all four combo phases to start? The log's per-phase seconds will tell.
- Is wg_on_256 True in this run? If the B256 selector flips to Marlin, F2 is moot and bs=256 measures master's Marlin+align2 path.
- F1: what is the rel-L2 vs fp32 at T=1 on the skewed routing with both w13 and w2 atomic?

