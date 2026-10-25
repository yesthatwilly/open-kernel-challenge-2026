---
problem: muse_prefill_layer
slug: turn-align-two-phase-chunk-prefix
created: 2026-10-25T110643Z
base: master
area: "MoE align/dispatch on T <= 4096 (Marlin turn_1 + ssdq turn_4): _align_parallel replaced by a two-launch chunk-count + prefix-place align"
change: >
  Built on master (ssdq s1b landed). Only _align_parallel's implementation changes, at its two
  call sites on T <= 4096: _experts_marlin (turn_1 T=753, block_size_m 64) and _experts_ssdq
  (turn_4 T=3012, block 128). _experts_triton (T > 4096) keeps SGLang's moe_align_block_size
  unchanged, so first_turn is untouched and the Triton grid (sized by sorted_ids capacity) cannot
  move. EVIDENCE (archive ef281fb8b7304240.profile.txt): _align_parallel_kernel is 16.7/16.8 us at
  turn_1 and 68.3 us at turn_4 (2.1% of sliding turn_4, more than gather 33 us or the weighted sum
  24 us), and it runs serially on the main stream between the router join and the expert GEMMs.
  The cause is structural: its grid is local_count + 1 = 33 programs and EVERY program streams all
  n = 8T route ids twice (count pass, then place pass) in CHUNK=512 pieces with a [512 x 32]
  compare, so time grows with n on 33 SMs (24096 ids x 2 passes x 33 programs at T=3012). CHANGE:
  replace it with a two-launch align that reads each id a constant number of times, keeping the
  exact output contract (same four tensors, same shapes: local_ids [T,8], sorted_ids [cap = n +
  32*(bsm-1)], block_experts [cdiv(cap, bsm)], n_padded [1]; same contents BIT FOR BIT: routes of
  expert e in ascending route index, padding slots = n, block_experts = e then -1 past the used
  blocks, n_padded = total). Launch A, grid (nchunks = cdiv(n, CHUNK_A),) with CHUNK_A = 1024:
  each program counts its chunk's routes per local expert (the same local = id - local_start in
  [0, local_count) test) and stores cnt[chunk, 0:32] int32 into a scratch tensor allocated in the
  wrapper with torch.empty((nchunks, 32)) (shape-sized, not value-sized). Launch B, grid (nchunks
  + 1,): every program loads the whole cnt table (<= 32 x 32 ints at T=4096), forms totals, padded
  = ceil to bsm, seg_start = exclusive cumsum, and its own chunk prefix pre[e] = sum of cnt over
  earlier chunks; program i < nchunks then places its chunk's routes at seg_start[e] + pre[e] +
  (rank of the route among chunk routes of expert e, via a per-expert cumsum over the chunk),
  which reproduces ascending order exactly, and writes local_ids for its chunk; program nchunks
  writes each expert's padding slots, block_experts (e for its blocks, -1 past used_blocks up to
  n_blocks), the sentinel n into [total, cap), and n_padded. No atomics, no host sync, no
  value-dependent allocation; both launches graph-capturable. prepare(): compile for bsm 64 and
  128, then require torch.equal on all four outputs against master's _align_parallel (kept in the
  file, unchanged) at T = 9, 200, 300, 753, 1100, 1506, 3012, 4096 x routings random, skewed (4
  fixed hot local experts), all-one-expert, all_remote, plus local_start 0 on 0-31/-1 ids as both
  call sites pass; any mismatch prints '[align2] OFF: <first mismatch>' and master's kernel stays.
  GATE: cold-L2 graph-replay medians of master's _align_parallel vs the new pair at T=753/bsm 64
  and T=3012/bsm 128 (>= 9 interleaved alternations, empty-graph baseline subtracted); ON per T
  class only if new <= 0.98x master (must win, LESSONS 04:36Z); print both medians and one
  '[align2] PATH CHOSEN' line per class as the last align lines (they print before the ssdq lines;
  keep ssdq's PATH CHOSEN last). Do not gate on data_ptr or pointer equality anywhere (the judge
  clones state tensors between copies).
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: _align_parallel_kernel 68.3 us -> target <= 10 us for the two launches (each id read twice in total instead of 66 times, on ~25-26 programs per launch); sliding turn_4 -55 us (~1.8%), full turn_4 ~1.2%
  - shapes: "*load=turn_1*"
    direction: faster
    why: _align_parallel_kernel 16.7/16.8 us -> target <= 7 us; ~1% of sliding turn_1 and ~0.7% of full turn_1, on the load that is still only 1.015x
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: "_experts_triton keeps SGLang's moe_align_block_size"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# turn-align-two-phase-chunk-prefix

## Change

Built on master (ssdq s1b landed). Only _align_parallel's implementation changes, at its two call sites on T <= 4096: _experts_marlin (turn_1 T=753, block_size_m 64) and _experts_ssdq (turn_4 T=3012, block 128). _experts_triton (T > 4096) keeps SGLang's moe_align_block_size unchanged, so first_turn is untouched and the Triton grid (sized by sorted_ids capacity) cannot move. EVIDENCE (archive ef281fb8b7304240.profile.txt): _align_parallel_kernel is 16.7/16.8 us at turn_1 and 68.3 us at turn_4 (2.1% of sliding turn_4, more than gather 33 us or the weighted sum 24 us), and it runs serially on the main stream between the router join and the expert GEMMs. The cause is structural: its grid is local_count + 1 = 33 programs and EVERY program streams all n = 8T route ids twice (count pass, then place pass) in CHUNK=512 pieces with a [512 x 32] compare, so time grows with n on 33 SMs (24096 ids x 2 passes x 33 programs at T=3012). CHANGE: replace it with a two-launch align that reads each id a constant number of times, keeping the exact output contract (same four tensors, same shapes: local_ids [T,8], sorted_ids [cap = n + 32*(bsm-1)], block_experts [cdiv(cap, bsm)], n_padded [1]; same contents BIT FOR BIT: routes of expert e in ascending route index, padding slots = n, block_experts = e then -1 past the used blocks, n_padded = total). Launch A, grid (nchunks = cdiv(n, CHUNK_A),) with CHUNK_A = 1024: each program counts its chunk's routes per local expert (the same local = id - local_start in [0, local_count) test) and stores cnt[chunk, 0:32] int32 into a scratch tensor allocated in the wrapper with torch.empty((nchunks, 32)) (shape-sized, not value-sized). Launch B, grid (nchunks + 1,): every program loads the whole cnt table (<= 32 x 32 ints at T=4096), forms totals, padded = ceil to bsm, seg_start = exclusive cumsum, and its own chunk prefix pre[e] = sum of cnt over earlier chunks; program i < nchunks then places its chunk's routes at seg_start[e] + pre[e] + (rank of the route among chunk routes of expert e, via a per-expert cumsum over the chunk), which reproduces ascending order exactly, and writes local_ids for its chunk; program nchunks writes each expert's padding slots, block_experts (e for its blocks, -1 past used_blocks up to n_blocks), the sentinel n into [total, cap), and n_padded. No atomics, no host sync, no value-dependent allocation; both launches graph-capturable. prepare(): compile for bsm 64 and 128, then require torch.equal on all four outputs against master's _align_parallel (kept in the file, unchanged) at T = 9, 200, 300, 753, 1100, 1506, 3012, 4096 x routings random, skewed (4 fixed hot local experts), all-one-expert, all_remote, plus local_start 0 on 0-31/-1 ids as both call sites pass; any mismatch prints '[align2] OFF: <first mismatch>' and master's kernel stays. GATE: cold-L2 graph-replay medians of master's _align_parallel vs the new pair at T=753/bsm 64 and T=3012/bsm 128 (>= 9 interleaved alternations, empty-graph baseline subtracted); ON per T class only if new <= 0.98x master (must win, LESSONS 04:36Z); print both medians and one '[align2] PATH CHOSEN' line per class as the last align lines (they print before the ssdq lines; keep ssdq's PATH CHOSEN last). Do not gate on data_ptr or pointer equality anywhere (the judge clones state tensors between copies).

## Expected

- `*load=turn_4*` faster: _align_parallel_kernel 68.3 us -> target <= 10 us for the two launches (each id read twice in total instead of 66 times, on ~25-26 programs per launch); sliding turn_4 -55 us (~1.8%), full turn_4 ~1.2%
- `*load=turn_1*` faster: _align_parallel_kernel 16.7/16.8 us -> target <= 7 us; ~1% of sliding turn_1 and ~0.7% of full turn_1, on the load that is still only 1.015x
- `*load=first_turn*` unchanged: _experts_triton keeps SGLang's moe_align_block_size

## Risk

Small absolute gain (~0.5-0.9% on the geomean, below one run's resolution): judge it on the _align_parallel_kernel rows (16.7 and 68.3 us in master's profile) and the new two kernel rows, not the headline. Risk is a second graph node (~2-4 us) eating the turn_1 saving; the per-class 0.98x gate handles that. Bit-identical outputs mean no correctness exposure downstream. Stacks with experts-turn4-ssdq-s2-tma-raw-weight-ring (disjoint code) and with any ssdq/rega stage that keeps calling _align_parallel.
