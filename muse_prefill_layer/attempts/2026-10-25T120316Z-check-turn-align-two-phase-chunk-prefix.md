# muse_prefill_layer · 2026-10-25T120316Z · check · turn-align-two-phase-chunk-prefix

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T110643Z-turn-align-two-phase-chunk-prefix.md` (branch `hyp/muse_prefill_layer/turn-align-two-phase-chunk-prefix`).

- **Job:** `d792e5c9d6d94f02` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `44addb9d`

## Goal
Built on master (ssdq s1b landed). Only _align_parallel's implementation changes, at its two call sites on T <= 4096: _experts_marlin (turn_1 T=753, block_size_m 64) and _experts_ssdq (turn_4 T=3012, block 128). _experts_triton (T > 4096) keeps SGLang's moe_align_block_size unchanged, so first_turn is untouched and the Triton grid (sized by sorted_ids capacity) cannot move. EVIDENCE (archive ef281fb8b7304240.profile.txt): _align_parallel_kernel is 16.7/16.8 us at turn_1 and 68.3 us at turn_4 (2.1% of sliding turn_4, more than gather 33 us or the weighted sum 24 us), and it runs serially on the main stream between the router join and the expert GEMMs. The cause is structural: its grid is local_count + 1 = 33 programs and EVERY program streams all n = 8T route ids twice (count pass, then place pass) in CHUNK=512 pieces with a [512 x 32] compare, so time grows with n on 33 SMs (24096 ids x 2 passes x 33 programs at T=3012). CHANGE: replace it with a two-launch align that reads each id a constant number of times, keeping the exact output contract (same four tensors, same shapes: local_ids [T,8], sorted_ids [cap = n + 32*(bsm-1)], block_experts [cdiv(cap, bsm)], n_padded [1]; same contents BIT FOR BIT: routes of expert e in ascending route index, padding slots = n, block_experts = e then -1 past the used blocks, n_padded = total). Launch A, grid (nchunks = cdiv(n, CHUNK_A),) with CHUNK_A = 1024: each program counts its chunk's routes per local expert (the same local = id - local_start in [0, local_count) test) and stores cnt[chunk, 0:32] int32 into a scratch tensor allocated in the wrapper with torch.empty((nchunks, 32)) (shape-sized, not value-sized). Launch B, grid (nchunks + 1,): every program loads the whole cnt table (<= 32 x 32 ints at T=4096), forms totals, padded = ceil to bsm, seg_start = exclusive cumsum, and its own chunk prefix pre[e] = sum of cnt over earlier chunks; program i < nchunks then places its chunk's routes at seg_start[e] + pre[e] + (rank of the route among chunk routes of expert e, via a per-expert cumsum over the chunk), which reproduces ascending order exactly, and writes local_ids for its chunk; program nchunks writes each expert's padding slots, block_experts (e for its blocks, -1 past used_blocks up to n_blocks), the sentinel n into [total, cap), and n_padded. No atomics, no host sync, no value-dependent allocation; both launches graph-capturable. prepare(): compile for bsm 64 and 128, then require torch.equal on all four outputs against master's _align_parallel (kept in the file, unchanged) at T = 9, 200, 300, 753, 1100, 1506, 3012, 4096 x routings random, skewed (4 fixed hot local experts), all-one-expert, all_remote, plus local_start 0 on 0-31/-1 ids as both call sites pass; any mismatch prints '[align2] OFF: <first mismatch>' and master's kernel stays. GATE: cold-L2 graph-replay medians of master's _align_parallel vs the new pair at T=753/bsm 64 and T=3012/bsm 128 (>= 9 interleaved alternations, empty-graph baseline subtracted); ON per T class only if new <= 0.98x master (must win, LESSONS 04:36Z); print both medians and one '[align2] PATH CHOSEN' line per class as the last align lines (they print before the ssdq lines; keep ssdq's PATH CHOSEN last). Do not gate on data_ptr or pointer equality anywhere (the judge clones state tensors between copies).

## What changed, and where
Base: worktree base f896dc26 had the same muse_prefill_layer/submission.py as master 3707e7b0 (no diff), so no merge was needed. Edited only the worktree's muse_prefill_layer/submission.py (+259/-4). No GPU here: nothing was compiled or run; every check below runs in prepare().

What changed:
- New Triton kernels `_align2_count_kernel` (launch A, grid nchunks: per-chunk, per-local-expert counts into a shape-sized `torch.empty((nchunks, 32))` scratch) and `_align2_place_kernel` (launch B, grid nchunks+1: each program loads the count table; chunk programs write local_ids and place routes at seg_start + earlier-chunk counts + in-chunk rank (2-D cumsum); the tail program writes per-expert padding slots, the sentinel n in [total, cap), block_experts, n_padded). block_experts is computed per block as "number of experts whose segment ends at or before slot b*BSM", -1 at or past total, which equals master's e / -1 contract. `_align_two_phase(topk_ids, local_start, local_count, bsm, chunk)` is the wrapper. `_align_parallel` is untouched.
- `_align_dispatch(local_ids, bsm, state)` is now called by `_experts_rega` (bsm 16, the live caller), `_experts_ssdq` (128) and `_experts_marlin` (64, inside the align_par_ok branch). `_experts_triton` is unchanged.
- `_align2_setup(state, device)` runs right after `_align_par_setup`, before `_silu_sorted_setup`, `_ssdq_setup` and `_rega_setup`. It compiles bsm 16/64/128 x chunk 512/1024, checks torch.equal on all four outputs against `_align_parallel` for T = 9, 200, 300, 753, 1100, 1506, 3012, 4096 x {random mapped, random raw ids, skewed, one-expert, all-remote, local_start 8} for every combo, and gates cold-L2 graph medians (15 interleaved alternations, empty-graph baseline subtracted, rotated and flipped order) at (16, T753, low), (16, T3012, high), (64, T753, low), (128, T3012, high). ON only if best chunk <= 0.98x master. It prints per-gate lines and one `[align2] PATH CHOSEN` line, then returns before the ssdq and rega setups, whose PATH CHOSEN lines remain last.

Adversarial-review items:
1. Live caller: `_experts_rega` now goes through `_align_dispatch` (applied; it ignores align_par_ok as before, and align2 is only decided ON when align_par_ok is on).
2. Validate at bsm 16: compile, bit-identity and gates at bsm 16 for T=753 and T=3012 (applied; bsm 64/128 fallback sites also covered and consistent).
3. Setup order: next to `_align_par_setup`, before `_ssdq_setup`/`_rega_setup` (applied), so rega's gate, canary and replay check see the timed align. `[rega]` ON is not forced by me; it appears in the run log to verify.
4. State: decision is `state["align2_ok"]`, a dict with string keys "bsm:class" -> chunk size (0/absent = master). It is added to REGA_STATE_KEYS so rega's copied states carry it. No module-global decision, no tensors in globals.
5. Chunk size: both 512 and 1024 are compiled, checked and timed per gate class; the faster one is stored (512 gives 12 programs at T=753, 1024 gives 6).
Spec corrections applied: re-target to bsm 16 (review overrides the stale marlin/ssdq targeting); dispatch is restricted to T > 200 (the equivalence suite's lower bound); Marlin/ssdq retained as gated fallbacks.
Deviations: block_experts uses the segment-end count formula instead of per-expert loops (same values); the pointer-equality gate is not used anywhere.
Risks: Triton compile of the 2-D axis-0 cumsum on a [512|1024, 32] tile with 8 warps (register pressure); any compile error or mismatch just turns align2 OFF (master's align stays). The serial tail program (padding, sentinel over cap ~24.6k slots at bsm 16, 1536 blocks) could eat the gain at T=753, which the gate decides.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path adds only two Triton launches and torch.empty scratch of shape-derived size; no .item()/.cpu()/sync/prints in it (the .item() calls are in prepare()-only helpers).
- [x] Nothing derived from `data` is cached across calls: cnt is allocated per call; the ON/OFF decision is shape-class state set in prepare().
- [x] Weights: n/a: untouched.
- [x] Activations/KV/queries: n/a: only int32 routing indices change, with bit-identical outputs to master's align.
- [x] Attention sees every key: n/a: no attention change.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess or fork: checked in the diff, none.
- [x] All work inside the timed call; no output caching or replay.

okc_lint output: "OK with warnings: muse_prefill_layer/submission.py". The warnings are host-sync/dtype notes, all in prepare() (including the new .item() at the first-mismatch helper) or pre-existing. undef.py reported []; py_compile passed; __pycache__ removed.

## Risks / what could fail
low
Small absolute gain (~0.5-0.9% on the geomean, below one run's resolution): judge it on the _align_parallel_kernel rows (16.7 and 68.3 us in master's profile) and the new two kernel rows, not the headline. Risk is a second graph node (~2-4 us) eating the turn_1 saving; the per-class 0.98x gate handles that. Bit-identical outputs mean no correctness exposure downstream. Stacks with experts-turn4-ssdq-s2-tma-raw-weight-ring (disjoint code) and with any ssdq/rega stage that keeps calling _align_parallel.

## Result (after the run)
- job `d792e5c9d6d94f02` (check), status done, passed True, tests 5/5
- speedup 1.3369x, 95% CI [1.3168, 1.3569], score 289788.31154642784 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.3274587526558561, "ci": [1.3075468713660183, 1.347370633945694], "job": "2910b2fb4b744865"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-port-t4096.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.result.json`, `muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.patch`; branch `hyp/muse_prefill_layer/turn-align-two-phase-chunk-prefix`
