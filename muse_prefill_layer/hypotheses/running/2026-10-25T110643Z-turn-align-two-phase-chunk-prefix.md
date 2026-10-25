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
review: proceed
status: running
review_at: 2026-10-25T115947Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T115947Z
base_sha: f896dc264c4b4ee1df6c4b33556b50f2d4282f9c
base_blob: 712edc5ea5be2dde8f558a016d0877b285340ee4
branch: hyp/muse_prefill_layer/turn-align-two-phase-chunk-prefix
worktree: .worktrees/muse_prefill_layer/turn-align-two-phase-chunk-prefix
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T115947Z by an adversarial subagent (model opus).

**Verdict: proceed.** Still worth running, and the case is stronger on the new master. In the rega profile (2910b2fb4b744865) _align_parallel_kernel is 16.9/17.1 us at turn_1 (2.3%/1.5%) and 70.3/70.9 us at turn_4 (3.1%/1.9%), so its share went up once the expert GEMMs got faster. But the change text is stale and must be re-targeted before building: the live caller is _experts_rega at REGA_BSM=16 (200<T<=4096). Marlin (bsm 64) and ssdq (bsm 128) are now fallbacks only, so compile, equivalence and gate must cover bsm 16.

### Justification
A: The mechanism holds and is unchanged by the merge. Master's _align_parallel_kernel (submission.py:651) runs on local_count+1 = 33 programs. Every program streams all n = 8T ids twice with a [512 x 32] compare (count pass, then place pass), so the work is about 33 x 2 x n, on 33 SMs. A two-launch count/prefix-place reads each id about twice, spread over cdiv(n,1024) programs (24 at T=3012, 6 at T=753). The 70 us down to <=10 us target at turn_4 is plausible. At turn_1 the target is 17 down to about 6-8 us: there are only 6 programs per launch there, plus a second graph node, so the saving is maybe 9-11 us. Re-based expectation on the new master: turn_4 sliding about -60 us of 2295 (~2.6%), full about -60 of 3821 (~1.6%); turn_1 sliding about -10 of 738 (~1.4%), full about -10 of 1177 (~0.9%); first_turn flat, since _experts_triton keeps moe_align_block_size at 50 us. Geomean is about +0.8%, below one run's CI (1.3075-1.3474). So it must be judged on the profile rows (the align row going from 16.9/70.3 us to the two new rows), as the hypothesis itself says. It is not a duplicate. turn-align-remap-on-side-stream-under-pooler (neutral) tried to hide the align, not shrink it. turn-align-sglang32-prenorm-triton-restack swapped in SGLang's align. Neither cut the O(33n) work. WEAKEST LINK: the stale targeting. Every parameter in the change paragraph (call sites _experts_marlin/_experts_ssdq, compile for bsm 64 and 128, gate at T=753/bsm 64 and T=3012/bsm 128) names paths that no longer run at turn_1/turn_4. _experts_rega (submission.py:2872) calls _align_parallel(local_ids, 0, LOCAL_EXPERTS, REGA_BSM=16) unconditionally. If built as written: bsm 16 is never compiled or checked in prepare (risking a JIT compile in the first timed call, or an untested variant), and the gate measures variants that are not dispatched. Second weakest: at bsm 16, cap = n+480 and n_blocks = cdiv(cap,16) = 1536 at T=3012. The block_experts / sentinel tail written by the last program of launch B is larger than at bsm 128 (BLK-looped, cheap, but the tail program must not be serial over 1536 blocks with tiny tiles).

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 (landed, 1.3275): owns the only live T<=4096 call site (bsm 16). The align feeds rega_gather and both rega_gemm launches, and its outputs size the rega grid (rows = block_experts.shape[0]*16). Bit-identical outputs mean no numeric interaction, so any row change is the align itself. However, _rega_setup gates rega against ssdq/Marlin (REGA_GATE_RATIO 0.95) and runs a two-state replay check. Align2 must be decided in prepare BEFORE _ssdq_setup and _rega_setup, so those gates and checks run on the align that will be timed. rega wins by a wide margin, so a stale gate would not flip a path, but a failure inside rega's replay/canary would silently drop the whole 1.33x. Watch for '[rega] PATH CHOSEN' staying ON in the log.
- experts-turn4-ssdq-s1b-dataptr-symbol-fix (landed): now a fallback behind rega at turn_4. The hypothesis's '[align2] lines before ssdq PATH CHOSEN' ordering must now also keep rega's PATH CHOSEN lines last (the harness reads the final lines).
- stack-tonight-five-prefill (landed): introduced _align_parallel and its prepare check _align_par_setup / state['align_par_ok']. align2 is compared bit-for-bit against this kernel, so the reference is sound (it writes every element of all four outputs, including the [total,cap) sentinel and the -1 block tail, so torch.equal on torch.empty outputs will not trip on garbage). Note that _experts_rega ignores align_par_ok. If align2 hooks in only behind that flag, rega will not pick it up.
- sglang-pipeline-parity, experts-hybrid-marlin-turn-triton-first-turn, experts-triton-w4a16-first-turn-blockt256: first_turn / Triton path only, disjoint. No confound, provided align2 is not wired into _experts_triton.

### Interference and A/B needs
No companion is warranted. The outputs are bit-identical to master's align, so downstream kernels see identical inputs and the only expected movement is the align rows themselves, readable directly from the per-shape profile. A gain is unambiguous if the 16.9/70.3 us rows are replaced by smaller rows. A headline loss with smaller align rows would be noise (the effect is below CI). An ab-revert of rega would just re-test the stale bsm 64/128 variants on paths that are no longer shipped. A control has nothing left to isolate once the change is a single kernel swap with a per-class gate.

### Open questions
- REQUIRED RE-TARGET before build: replace '_experts_marlin (bsm 64) and _experts_ssdq (bsm 128)' with '_experts_rega (REGA_BSM=16, 200<T<=4096), keeping Marlin/ssdq as fallbacks'. Compile align2 for bsm 16 (and 64, 128 if it is swapped globally). Run the torch.equal equivalence suite at bsm 16 for every listed T. Gate at T=753/bsm 16 and T=3012/bsm 16 (the classes rega dispatches: low T<=1024, high above).
- Where does align2 setup run? It must be before _ssdq_setup/_rega_setup (next to _align_par_setup at submission.py:609), so rega's gate and two-state replay check exercise the timed align.
- At T=753 there are only 6 chunks of 1024. Would CHUNK_A=512 (12 programs) beat 1024 for turn_1? The gate will decide, but the builder could time both.
- Should first_turn's moe_align_block_size (~50 us, 0.3%) also be swapped? No. Out of scope, and align-local-parallel-first-turn was already neutral.

