# muse_moe_w4a16 · 2026-10-25T185808Z · submit · combo-smallm-two-stream-halves-x-w13-splitk

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T161613Z-combo-smallm-two-stream-halves-x-w13-splitk.md` (branch `hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk`).

- **Job:** `9a455a2aae7842cb` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `f53e0bef`

## Goal
COMBINATION (conjectural pairing, small-M Marlin side): two-stream expert halves paired with w13 split-K atomic add (w2 atomic is already on master) at 8 <= T <= 32, on top of the shared floor. Built on master a0e52640. SHARED FLOOR (identical in all three combo hypotheses minted 2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1: w13 Marlin use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port W13_ATOMIC_MAX_TOKENS = 1, W13_ATOMIC_TOL, _w13_band, _w13_atomic_self_check and the w13_mode kwarg in the w13 moe_wna16_marlin_gemm call of _moe_marlin (w2 keeps master's w2_mode). Measured twice: decode_1 1.766x (job 171684977ec1470f) and 1.754x (job 1048c6edf3e54892) against master-equivalent 1.698-1.721x in eight runs (+2.5-3.0%; Marlin pair 33.7 vs 35.9 us). Gate: its own self-check (rel-L2 vs master <= 1e-2; it measured 8.4e-3) plus a component graph (the w13 Marlin call alone at T=1, skewed and random local routing) <= 0.98x. Phase F1 start deadline 120 s, ~3 s. (F2) verify_32: master's OWN align2 win is dead code on the judged path. Master's _align2_setup gates class v256 for _moe_marlin (job c05a8f7326004be6: align2 chunk 64 4.70 us vs incumbent 6.46 us, ratio 1.374, ON), but kernel_fn sends 256 <= T <= 512 to _moe_wgmma (wg_on_256 True), which still runs bank expert_map[topk_ids.to(int32)] plus SGLang moe_align_block_size(local_ids, WG_BSM=16, count): the c05a8f73 verify_32 profile shows index_elementwise 3.4 + unrolled_elementwise 3.1 + moe_align_block_size_kernel 2.8 + count_and_sort_expert_tokens 2.5 = 11.8 us in four launches and no _align2 kernel. Source: master a0e52640's own _align_two_phase / _align2_count_kernel / _align2_place_kernel (port origin align-two-phase-port-from17, impl commit fb879b90; prefill origin align-two-phase-all-t-rega). Change: in _moe_wgmma only, when state['wg_align2'] (a plain int chunk, 0 = master) is set, replace the expert_map gather and moe_align_block_size with local_ids, (sorted_ids, block_experts, n_padded) = _align_two_phase(topk_ids, start, count, WG_BSM, chunk). Gate F2: warm every T in 128..512 at block_size_m 16 for each chunk in (64, 128, 256, 512) <= the class top; equivalence on T in (128, 191, 192, 256, 257, 384, 512) x three local_starts x six routings (_align2_routings): local_ids torch.equal, n_padded equal, sorted_ids[:n_padded] and block_experts[:n_padded/16] torch.equal to master's front end (capacity tails may differ: the wg kernels read below n_padded only; assert n_padded/16 <= WG_MAX_BLOCKS), then full _moe_wgmma output torch.equal (bitwise: same kernels, same tables) on two of those routings per T; then the component graph (front end alone: gather+cast+align vs align2) <= 0.98x at T=256 (expected ~0.4-0.5x). Applies to whichever wgmma bucket is ON (B256 today; B128 only if its flag is on). Phase F2 start deadline 150 s, ~15-20 s (Triton constexpr warm per T is most of it). (C3a) Two-stream halves. Source marlin-two-stream-expert-halves-small-m, branch hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m, impl commit ea345883: port _align_small_halves_kernel (grid 2, one program per expert half, its own sorted_ids_h / block_experts_h / n_padded_h / local_ids_h, HALF_SORTED_ALIGN), _experts_two_stream (side.wait_stream(current); half A on the current stream, half B under torch.cuda.stream(side); w13 Marlin, silu_and_mul with expert_ids=local_ids_h, w2 Marlin into the shared cache3; current.wait_stream(side); master's post-norm on the full local_ids), the second Marlin lock workspace and the side stream, both created in prepare and held in the state. Measured: decode_32 1.042x (job 5d630369b1634f77, best decode_32 row of any run) but verify_1 1.316x vs 1.33-1.35x, decode_16 flat; that run predates master's w2 atomic add. (C3b) w13 atomic at T <= 32. Source w13-marlin-atomic-add-on-master-small-m, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m, impl commit c3a84420 (W13_ATOMIC_MAX_TOKENS 32, _w13_band, _w13_atomic_self_check): flat on its own at T = 8/16/32 (1.353/1.281/1.035x), +2.6% at T=1 (that row is F1). In the halves path both Marlin calls of each half take the w13 kwargs of the chosen mode and w2 keeps master's w2_mode (each half writes disjoint route rows of cache1/cache3, so atomic modes stay row-disjoint; if w2_mode is atomic_zero, the zeroing happens once on the current stream before the fork). GATE per T band, bands chosen by shape only: [8, 12] rep T=8 (verify_1 and the bs=4-adjacent range), [13, 24] rep T=16, [25, 32] rep T=32; candidates {master, H = halves, H+A = halves with w13 atomic, A = w13 atomic alone}. Each candidate: halves self-check (per-half routes and block counts equal to moe_align_block_size with the other half set to -1, as the source did), rel-L2 vs master <= 1e-2 and finite on six routings, graph capture with the fork/join (capturable; the source ran it on the judge), then full kernel_fn cold-L2 graphs, 41 interleaved reps, judge-like skewed and random routings; the band takes the lowest median only at <= 0.98x master on the worse routing. Print PATH CHOSEN per band with all four medians. state['sm_mode'] = dict band -> int set in prepare; kernel_fn picks by num_tokens band only. ROUTING SUMMARY: T == 1 -> F1; 8 <= T <= 32 -> per band master / H / H+A / A; 256 <= T <= 512 -> wgmma with F2; everything else master. Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C3 compile and warm of the halves kernel for T 8..32 (~10 s, start deadline 150 s) + self-checks ~5 s + gate 3 bands x 4 candidates x 2 routings (~6 s, start deadline 230 s) = ~95 s of 300. RULES (every combo phase): kernel_fn branches on shape and prepare-set plain ints/bools only; no host sync, no input-value-sized allocation, no data-dependent control flow, no prints, no try/except, no per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights stay NVFP4 (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation; every local route computed; no child processes. The judge prepares TWO states and copies them by cloning tensors: every scratch tensor, stream, workspace and error word lives in the state, no module-global device pointer, tensor map or tensor, and tensor maps stay encoded per call (master's s5 rule). PREPARE ORDER: master's prepare() runs first and unchanged (align self-check, _align2_setup, _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 / WG_REPLAY_DEADLINE_S 235 / WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs AFTER it, measures its own elapsed time from prepare()'s t_prepare, skips itself (flag stays master, prints PATH CHOSEN = master (deadline)) if it would start past its start deadline, and no phase starts timing past 270 s. GATE RULE (every source): exactness or tolerance check first, then cold-L2 CUDA-graph timing vs master's path for that class (>= 256 MB flush before each replay, interleaved A/B, >= 21 reps, empty-graph baseline subtracted, median); ON only at new/master <= 0.98 on the worse of the judge-like skewed routing and random routing; otherwise that class keeps master. Print one PATH CHOSEN line per source per class with both medians and the ratio, and keep those lines in the last 16k characters of the log (print them again at the very end of prepare, after master's [align2] lines). Print per-phase prepare seconds and the total.

## What changed, and where
Built 2026-10-25 in worktree `.worktrees/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk` on base 405d804d. Master is 5b1c422e, but `muse_moe_w4a16/submission.py` is byte-identical between the two (the commits in between touch other problems or only md files), so no merge was needed. Only `muse_moe_w4a16/submission.py` was edited: +925 / -7 lines. Not committed, okc not run.

**What changed**
- `_moe_marlin` gains `w13_mode` ("master" / "atomic" / "atomic_zero") and a prepare-only `poison` flag. The poison fills cache13 with 3e4. For `atomic_zero`, the zero_ runs inside the call on the current stream. Defaults keep master's behaviour.
- `_moe_wgmma`: when `state["wg_align2"]` (a plain int chunk) is set, it uses `_align_two_phase(topk_ids, start, count, WG_BSM, chunk)`; otherwise it runs master's gather plus `moe_align_block_size` (F2).
- New combo block before `kernel_fn`:
  - Ported from ea345883: `_align_small_halves_kernel`, `_align_small_halves` and `_align_halves_self_check` (it now returns cases and mismatches).
  - New: `_experts_two_stream` (H / H+A) and `_sm_post_norm`.
  - New gates: `_combo_f1`, `_combo_f2`, `_sm_cross_half`, `_sm_time_band`, `_combo_c3`, `_combo_replay_check`, `_combo_setup`.
- `prepare()`: master's prepare runs unchanged. `_combo_setup` runs after `_wg_setup`. The combo lines print last, after the [align2] lines, then the prepare total.
- `kernel_fn` routes by shape and prepare-set plain values only:
  - T == 1: `state["w13_t1"]`.
  - 8..32: `state["sm_mode"][band rep]` (0 master / 1 H / 2 H+A / 3 A, with A using `state["sm_w13"]`).
  - 128..512: master's wgmma gate (F2 sits inside `_moe_wgmma`).
  - Everything else: master.
- Deadlines from t_prepare:
  - Phase starts: F1 120 s, F2 150 s, C3 150 s.
  - C3 gate start 230 s; no C3 band timing starts past 270 s.
  - Replay check 280 s. Past it, every combo class reverts to master, because unverified flags must not ship.
- Each phase is wrapped in try/except in prepare and falls back to master on failure. Per-phase seconds and the prepare total are printed.

**F1 / F2, identical to the sibling combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 spec plus its review fixes**
- F1 correctness:
  - Candidates are "atomic", then "atomic_zero", with cache13 poisoned.
  - Routings: the six `_align2_routings` plus a judge-like one, at T=1.
  - Requires rel-L2 vs current master <= 1e-2 AND vs an fp32 reference <= 1e-2 (sibling review fix), and finite.
  - The fp32 reference is `_combo_fp32_ref`: fp32 dequant of the used experts, fp32 matmuls, fp32 post-norm. Master's own error vs fp32 is printed too.
- F1 timing: a component graph of the w13 Marlin call alone at T=1 (judge-like and all-local-saturated routings). It is ON at a worse-routing ratio <= 0.98.
- F2 only acts on the wgmma buckets that are ON. If none is, it prints "inert".
- F2 warm and capacity: every T of the ON buckets is warmed for every chunk. Capacity is checked against WG_MAX_BLOCKS and WG_CAP_ROWS.
- F2 table check: torch.equal on all four outputs against `_align_parallel` at bsm 16, never against SGLang's tables (review fix). It runs on F2_TEST_T x three local_starts x six routings. local_ids is also checked against the expert_map gather, and n_padded/16 <= WG_MAX_BLOCKS.
- F2 output check: the full `_moe_wgmma` output must be bitwise equal (torch.equal) to master's front end on two routings per T. There is no rel-L2 fallback, as the user's message requires bitwise. The wg error word is checked too.
- F2 timing: a front-end component graph at T=256 (T=128 if only B128 is ON). Master's gather plus align is compared against each chunk, and `_align_parallel` at bsm 16 is printed as info only. Worse-routing ratio <= 0.98.

**User-binding review edits**
1. Poisoned H+A self-check: the stacked cache1 is filled with 3e4 and the shared cache3 with NaN before any launch. Any atomic_zero (w13 via `sm_w13`, w2 via `sm_w2`) does one zero_() inside `_experts_two_stream` (the kernel_fn path), on the current stream, before the fork, on every call. Nothing is zeroed only in prepare.
2. Serialized cross-half wipe test (`_sm_cross_half`):
   - It runs at T 8/16/32 on two routings, with both halves' w13+silu run first.
   - Then cache3 is filled with random bf16 junk and each half's w2 runs alone, with master's w2_mode kwargs and its own workspace.
   - Every row the half does not own (the other half's routes and the remote routes) must be bitwise unchanged, and its own rows must change.
   - If this fails with the atomic w2 mode, `state["sm_w2"] = "master"` and the line "the halves run w2 NON-ATOMIC" is printed. If it fails non-atomic too, H and H+A are OFF.
3. Graph-safe fork/join:
   - `state["sm_side"]` (a torch.cuda.Stream) and `state["sm_workspace_b"]` (Marlin lock workspace) are created in prepare. This happens after `_wg_setup`, so master's deepcopy copy check never sees a stream.
   - Every buffer is allocated before the fork. The current stream waits on the side stream before the post-norm. There is no record_stream.
   - The gate replays the captured fork/join on a cloned state. `_combo_replay_check` captures kernel_fn on state A and on `_wg_copy_tree(state)` (state C). It then replays A, C, A, C with refilled inputs on four routings, for every class whose flag is on, against master's flags in eager mode (tol 2e-2). A failing class reverts to master.
4. Post-norm masking: `_sm_post_norm` keeps master's choice by shape. Below 32, `_post_norm_combine_kernel` skips local id -1 slots and never loads those rows. At T == 32, SGLang's fused kernel gets slot_expert_ids, and the NaN-poisoned cache3 in the H/H+A self-check decides per band: a failure there turns only band [25, 32] off.
5. Prints: per band, the four-way medians (master / H / H+A / A) on both routings, the worse-routing ratios and PATH CHOSEN. There is a `B256 PATH CHOSEN = ... (wg_on_256 ..., wg_on_128 ...)` line with F2 inert or live, a FINAL flags line, and phase seconds. All of these sit at the very end of the log.

**This md's own review**
- The cross-half test is item 2 above.
- The poisoned self-check is item 1 above.
- The mechanism caveat (no "grid refill") needs no code change. A flat result at 8/16 is expected, and the band keeps master.
- The "four Marlin rows" profile readout is replaced by the printed per-band medians.
- The B256 line is printed so bs=256 can be attributed.

**C3 gate details**
- The halves align self-check is ported from the source and must pass with 0 mismatches.
- A (w13 atomic) flavour: the first of atomic / atomic_zero that passes poisoned rel-L2 <= 1e-2 vs master at all three representative T on 7 routings. H+A uses the same flavour.
- H and H+A are checked per band, poisoned, on 7 routings, rel-L2 <= 1e-2 and finite.
- Timing per band: each passed candidate's full kernel_fn is captured (a capture test first), then cold-L2 graph medians are taken with `_align2_gate_time`: 41 interleaved replays, a 256 MB flush, and the empty-graph baseline subtracted. Routings are judge-like and random. A band takes the lowest worse-routing ratio only at <= 0.98.

**Deviations**
- The A flavour is chosen once for all bands, not per band.
- The candidates' graph correctness is proven by the final A/C replay check of the chosen flags, not of every candidate. Each candidate is still capture-tested.
- The replay-check tolerance is 2e-2, not 1e-2. Its job is catching copy and graph bugs, which give errors much larger than 0.1, and F1's own correctness check already holds 1e-2.
- The judge-like routing (`_combo_judge_ids`) is written from TASK.md's description of the "normal" busiest routing. reference.py is not imported.
- No new CUDA or PTX is added, so no arch flags are changed. Only Triton and SGLang ops are used.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's call graph (`_moe_marlin`, `_moe_wgmma`, `_experts_two_stream`, `_align_small_halves`, `_sm_post_norm`, `_align_two_phase`). There is no allocation or control flow on input values: branches are on num_tokens and prepare-set ints/strings only. The fork/join is captured on both states in prepare.
- [x] Nothing derived from `data` is cached across calls. All buffers are allocated per call. The state holds only prepare-made streams, workspaces and flags.
- [x] Weights: no new weight storage. The halves read master's Marlin bank, and NVFP4 stays NVFP4. The fp32 dequant exists only in the prepare-time reference.
- [x] Activations: BF16 where SGLang keeps BF16 (cache1/2/3 bf16, fp32 reduce as master). KV cache: n/a (MoE problem).
- [x] Attention sees every key: n/a (no attention). Every local route is computed: the halves cover experts 0-15 and 16-31, and the align self-check proves the per-half route sets.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. torch.cuda.Stream is not a process.
- [x] All work happens inside the timed call; no output caching or replay. The atomic_zero memsets run inside kernel_fn on every call.
- [x] kernel_fn has no prints, no try/except and no mutable state. `poison` is only ever passed by prepare.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
OK with warnings: muse_moe_w4a16/submission.py
(102 warnings, all `host-sync ... (outside kernel_fn's call graph; fine in prepare())` or pre-existing `dtype` uint8 scratch/flush allocations)
```
py_compile: OK; __pycache__ removed.

## Risks / what could fail
medium
Medium risk: two measured ports whose failure modes are known (the halves ran on the judge, 5/5 tests; w13 atomic passed its self-check at T <= 32 with rel-L2 under 1e-2); the new element is only their pairing and the per-band gate. Error budget: w13 atomic adds ~8e-3 rel-L2 against master at T=1 and the check bound is 1.5% global, so the gate must reject any band whose candidate exceeds 1e-2 vs master.

## Result (after the run)
- job `9a455a2aae7842cb` (submit), status done, passed True, tests 5/5
- speedup 1.3887x, 95% CI [1.3678, 1.4095], score 9684.573782125924 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3865522395057346, "ci": [1.3546070184906125, 1.395864085043626], "job": "b395f5b6097348c7"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-two-phase-port-from17.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/combo-smallm-two-stream-halves-x-w13-splitk.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/combo-smallm-two-stream-halves-x-w13-splitk.patch`; branch `hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk`
