---
problem: muse_prefill_layer
slug: combo-a-master-normgather-naive-stack
created: 2026-10-25T161233Z
base: master
area: "restack: pre-expert norm + rega gather fused (all rega classes) on master 1.4587; naive per-shape winner stack"
change: >
  COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved
  3-way merge
  /tmp/claude-1000/-home-user-git-musespark/410338d7-535d-4133-b5f9-f8ed4ec9d3cb/scratchpad/m3/master_plus_normgather.py
  (master 020566aa / blob of muse_prefill_layer/submission.py with md5 82c368f4..., i.e. master
  1.4587 contiguous-KV FA3 + rega first-turn + align2, plus pre-expert-norm-fused-rega-gather from
  hyp/muse_prefill_layer/pre-expert-norm-fused-rega-gather impl bd60e2db, scored edf68fa3131c4bf8
  1.4600). Copy it over muse_prefill_layer/submission.py in the worktree and change nothing else.
  Verify the diff against master contains only the norm-gather hunks (the _norm_gather_setup block
  after _fa3c_setup, the fused _norm_gather_kernel Triton kernel, the prepare() call after
  _fa3c_setup, and the _experts_rega dispatch that uses the fused norm+gather when state flags say
  so); every master string (_REGA_*, _rega_setup, _fa3c_*, align2) must be byte-identical to
  master. SOURCES AND ROUTING: (1) master (all paths, unchanged). (2)
  pre-expert-norm-fused-rega-gather, bd60e2db, _norm_gather_kernel + _norm_gather_setup: replaces
  state['pre_expert_norm'](expert_input) followed by master's rega_gather with one fused
  RMSNorm*(1+w) + gather-to-sorted-slots Triton kernel, ONLY where rega is ON: T in (200,1024]
  low, (1024,4096] high, (4096,16384] ft; every other T and every non-rega path runs master's norm
  + gather. Its own prepare gate stays as built: self-check within 1 ulp with round_w=True on
  random/skewed routings per class, whole-expert-output rel L2 <= 1e-3, two-state replay rel L2 <=
  1e-3, then cold-L2 graph timing fused vs pair at ON <= 0.9 (it measured 0.348 / 0.329 / 0.327).
  It must print one '[norm-gather] PATH CHOSEN' line covering the three classes. NOT included, and
  why (all from the prepare logs, not the row deltas): experts-rega-producer-scale-expand-all-t
  (6be2a8d3) printed PATH CHOSEN master in low/high/ft (sx/master 1.099 / 1.033 / 1.038),
  experts-rega-pingpong-consumers-all-t (5fbaec9a) printed PATH CHOSEN master in all three (1.471
  / 1.179 / 1.239), experts-rega-low-class-two-ctas-per-sm (1a77da43) printed PATH CHOSEN low:
  master, qkv-gate-columns-side-stream (b8b8536c) was ON only at the sliding high class at 0.959
  with a flat row. Their row 'wins' therefore measured master and are run-to-run noise (about
  +-3-4% per row; contiguous-KV also moved sliding first_turn +2.3% where it cannot act). PREPARE
  BUDGET: master ends its own setup at about 124.3-124.8 s (before-rega 76.9, build 38.8-39.3,
  checks 8 s; fa3c setup 1.1 s), norm-gather setup 0.7 s, total about 125.5 s of 300 s.
  REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s are untouched. No nvcc module is
  added. RULES: unchanged from the two scored runs this file merges (both passed the judge);
  kernel_fn adds only a dict lookup on prepare-set bools keyed by the T class.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "fused norm+gather saves ~146 us per ft call (216.9 -> 71.0 us gate): sliding ft 10947 -> ~10800 us (~1.3%), full ft rows ~0.8-1.2%"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class 42.8 -> 14.0 us: sliding turn_4 2153 -> ~2124 (~1.3%), full ~0.8%"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low class 14.8 -> 5.2 us: sliding turn_1 720 -> ~711 (~1.3%), full ~0.8%"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-a-master-normgather-naive-stack

## Change

COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /tmp/claude-1000/-home-user-git-musespark/410338d7-535d-4133-b5f9-f8ed4ec9d3cb/scratchpad/m3/master_plus_normgather.py (master 020566aa / blob of muse_prefill_layer/submission.py with md5 82c368f4..., i.e. master 1.4587 contiguous-KV FA3 + rega first-turn + align2, plus pre-expert-norm-fused-rega-gather from hyp/muse_prefill_layer/pre-expert-norm-fused-rega-gather impl bd60e2db, scored edf68fa3131c4bf8 1.4600). Copy it over muse_prefill_layer/submission.py in the worktree and change nothing else. Verify the diff against master contains only the norm-gather hunks (the _norm_gather_setup block after _fa3c_setup, the fused _norm_gather_kernel Triton kernel, the prepare() call after _fa3c_setup, and the _experts_rega dispatch that uses the fused norm+gather when state flags say so); every master string (_REGA_*, _rega_setup, _fa3c_*, align2) must be byte-identical to master. SOURCES AND ROUTING: (1) master (all paths, unchanged). (2) pre-expert-norm-fused-rega-gather, bd60e2db, _norm_gather_kernel + _norm_gather_setup: replaces state['pre_expert_norm'](expert_input) followed by master's rega_gather with one fused RMSNorm*(1+w) + gather-to-sorted-slots Triton kernel, ONLY where rega is ON: T in (200,1024] low, (1024,4096] high, (4096,16384] ft; every other T and every non-rega path runs master's norm + gather. Its own prepare gate stays as built: self-check within 1 ulp with round_w=True on random/skewed routings per class, whole-expert-output rel L2 <= 1e-3, two-state replay rel L2 <= 1e-3, then cold-L2 graph timing fused vs pair at ON <= 0.9 (it measured 0.348 / 0.329 / 0.327). It must print one '[norm-gather] PATH CHOSEN' line covering the three classes. NOT included, and why (all from the prepare logs, not the row deltas): experts-rega-producer-scale-expand-all-t (6be2a8d3) printed PATH CHOSEN master in low/high/ft (sx/master 1.099 / 1.033 / 1.038), experts-rega-pingpong-consumers-all-t (5fbaec9a) printed PATH CHOSEN master in all three (1.471 / 1.179 / 1.239), experts-rega-low-class-two-ctas-per-sm (1a77da43) printed PATH CHOSEN low: master, qkv-gate-columns-side-stream (b8b8536c) was ON only at the sliding high class at 0.959 with a flat row. Their row 'wins' therefore measured master and are run-to-run noise (about +-3-4% per row; contiguous-KV also moved sliding first_turn +2.3% where it cannot act). PREPARE BUDGET: master ends its own setup at about 124.3-124.8 s (before-rega 76.9, build 38.8-39.3, checks 8 s; fa3c setup 1.1 s), norm-gather setup 0.7 s, total about 125.5 s of 300 s. REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s are untouched. No nvcc module is added. RULES: unchanged from the two scored runs this file merges (both passed the judge); kernel_fn adds only a dict lookup on prepare-set bools keyed by the T class.

## Expected

- `*load=first_turn*` faster: fused norm+gather saves ~146 us per ft call (216.9 -> 71.0 us gate): sliding ft 10947 -> ~10800 us (~1.3%), full ft rows ~0.8-1.2%
- `*load=turn_4*` faster: high class 42.8 -> 14.0 us: sliding turn_4 2153 -> ~2124 (~1.3%), full ~0.8%
- `*load=turn_1*` faster: low class 14.8 -> 5.2 us: sliding turn_1 720 -> ~711 (~1.3%), full ~0.8%

## Risk

## Why this is the naive stack
The per-shape winner map, re-based on the shared base (all five runs share the align2 base 704.38 / 1097.89 / 2098.36 / 3644.6 / 12078.49 / 11285.6 / 15310.56 / 18819.51 us; deltas are drift-corrected): the only source not on master whose path was actually ON in the judge is norm-gather (all three rega classes). Contiguous-KV is master. Scale-expand, ping-pong and two-CTAs were gated OFF in every class, so they are not stacked anywhere. Expected headline ~1.47 (master 1.4587 x ~1.009 weighted). This run is also the paired control for combination B (same file minus the low-class two-CTA port).

## Conjecture
Norm-gather's +4.5% sliding first_turn in edf68fa3 was measured on the pre-contiguous-KV master, where full-layer first_turn spent ~5% longer in FA3. The conjecture is that its saving is a fixed ~146 us per ft call that is independent of the attention path (they are on different sides of the attention boundary, the same stream, no shared buffers), so the stack is purely additive: full first_turn rows should show contiguous-KV's +3.4/+4.2/+4.7% plus ~1% on top, and sliding first_turn should land ~1.3% above master, not 4.5% (the remaining 3% of the edf68fa3 row was drift; contiguous-KV, which cannot act on sliding layers, moved that row +2.3% in its own run). Confirm with: the '[norm-gather] GATE' lines (fused ~71 / 14 / 5 us), the '[fa3 contig] PATH CHOSEN' line (B at the three first_turn shapes), and in the profile the absence of master's pre_expert_norm + rega_gather rows and a _norm_gather_kernel row of ~71 us at first_turn. Refuted if a first_turn row is slower than master by more than noise with both PATH CHOSEN lines as above (that would mean an L2 interaction between the fa3c contiguous K/V buffers and the fused gather's working set, which never ran together).

## Builder notes
Starting point is a file, not a branch: copy the merge, lint, py_compile, and the CPU dry-run of every PATH CHOSEN branch (ON and OFF) per LESSONS. Grep the diff for bare names in f-strings.
