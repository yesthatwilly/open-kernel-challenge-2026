---
problem: muse_prefill_layer
slug: combo-a-master-normgather-naive-stack
created: 2026-10-25T161233Z
base: master
area: "restack: pre-expert norm + rega gather fused (all rega classes) on master 1.4587; naive per-shape winner stack"
change: >
  COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved
  3-way merge
  /home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py
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
review: proceed
status: running
review_at: 2026-10-25T161512Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T161512Z
base_sha: 982cc0e7665f7a510a5b0128e83949922fd6b456
base_blob: 9fb1f7c4b4282dd7461c6cdba926fe3ebf12128e
branch: hyp/muse_prefill_layer/combo-a-master-normgather-naive-stack
worktree: .worktrees/muse_prefill_layer/combo-a-master-normgather-naive-stack
mode: null
restack_of: null
stale_base: null
---

# combo-a-master-normgather-naive-stack

## Change

COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py (master 020566aa / blob of muse_prefill_layer/submission.py with md5 82c368f4..., i.e. master 1.4587 contiguous-KV FA3 + rega first-turn + align2, plus pre-expert-norm-fused-rega-gather from hyp/muse_prefill_layer/pre-expert-norm-fused-rega-gather impl bd60e2db, scored edf68fa3131c4bf8 1.4600). Copy it over muse_prefill_layer/submission.py in the worktree and change nothing else. Verify the diff against master contains only the norm-gather hunks (the _norm_gather_setup block after _fa3c_setup, the fused _norm_gather_kernel Triton kernel, the prepare() call after _fa3c_setup, and the _experts_rega dispatch that uses the fused norm+gather when state flags say so); every master string (_REGA_*, _rega_setup, _fa3c_*, align2) must be byte-identical to master. SOURCES AND ROUTING: (1) master (all paths, unchanged). (2) pre-expert-norm-fused-rega-gather, bd60e2db, _norm_gather_kernel + _norm_gather_setup: replaces state['pre_expert_norm'](expert_input) followed by master's rega_gather with one fused RMSNorm*(1+w) + gather-to-sorted-slots Triton kernel, ONLY where rega is ON: T in (200,1024] low, (1024,4096] high, (4096,16384] ft; every other T and every non-rega path runs master's norm + gather. Its own prepare gate stays as built: self-check within 1 ulp with round_w=True on random/skewed routings per class, whole-expert-output rel L2 <= 1e-3, two-state replay rel L2 <= 1e-3, then cold-L2 graph timing fused vs pair at ON <= 0.9 (it measured 0.348 / 0.329 / 0.327). It must print one '[norm-gather] PATH CHOSEN' line covering the three classes. NOT included, and why (all from the prepare logs, not the row deltas): experts-rega-producer-scale-expand-all-t (6be2a8d3) printed PATH CHOSEN master in low/high/ft (sx/master 1.099 / 1.033 / 1.038), experts-rega-pingpong-consumers-all-t (5fbaec9a) printed PATH CHOSEN master in all three (1.471 / 1.179 / 1.239), experts-rega-low-class-two-ctas-per-sm (1a77da43) printed PATH CHOSEN low: master, qkv-gate-columns-side-stream (b8b8536c) was ON only at the sliding high class at 0.959 with a flat row. Their row 'wins' therefore measured master and are run-to-run noise (about +-3-4% per row; contiguous-KV also moved sliding first_turn +2.3% where it cannot act). PREPARE BUDGET: master ends its own setup at about 124.3-124.8 s (before-rega 76.9, build 38.8-39.3, checks 8 s; fa3c setup 1.1 s), norm-gather setup 0.7 s, total about 125.5 s of 300 s. REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s are untouched. No nvcc module is added. RULES: unchanged from the two scored runs this file merges (both passed the judge); kernel_fn adds only a dict lookup on prepare-set bools keyed by the T class.

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

## Adversarial review

Reviewed 2026-10-25T161512Z by an adversarial subagent (model opus).

**Verdict: proceed.** The merge is semantically correct. I checked it against both parents: master 020566aa (md5 82c368f4 matches) plus only the norm-gather hunks; bd60e2db plus only the fa3c hunks; it py_compiles. The interaction it tests (fixed ~146/29/10 us norm+gather saving, additive across the attention boundary) is plausible, and the profile and PATH CHOSEN lines can read it even though the ~0.9% headline gain sits inside the +-1.5% CI.

### Justification
A. MERGE VERIFIED, not just textually. diff(master, merged) has exactly six hunks, all from norm-gather. (1) `import inspect`. (2) state['pre_expert_w'] = weights['pre_expert_norm'] in bf16, a weight and not input-derived. (3) prepare order is _ssdq_setup -> _rega_setup -> fa3c_lines = _fa3c_setup(...) -> _norm_gather_setup(state, device, t_prepare0) -> sync, then align2_lines and fa3c_lines are printed again. So both new setups run after _rega_setup, norm-gather comes after fa3c, and fa3c_lines is still printed. (4) _experts(pre_norm=False) applies the module norm on every non-fused branch: rega OFF, ssdq, marlin, triton, and rega ON with norm_gather OFF. Every path except the fused one is therefore byte-identical to master's behaviour. (5) _experts_rega(prenorm=) swaps rega_gather for _norm_gather only when prenorm is set. (6) The NG_* block and kernel_fn's call changes to _experts(expert_input, ..., pre_norm=True). diff(bd60e2db source, merged) has four hunks: the two fa3c prepare lines, the _fa3c_* block and the FA3 call site. All _REGA_*, _rega_setup, align2 and _fa3c_* strings are untouched. Deadlines: FA3C_START_DEADLINE_S 235, FA3C_CLASS_DEADLINE_S 265, NG_BUDGET_S 270, REGA_FT_START_DEADLINE_S 150 and REGA_COPY_DEADLINE_S 255, all unchanged. The prepare estimate is ~125.5 s of 300 s. The rega gating dispatch at line ~3783 calls _experts without pre_norm, so rega's own gates still time master's normed input. Mechanism: the fused kernel reads raw rows once and writes the normed sorted slots, removing one full [T, 4096] bf16 write+read and a launch. The edf68fa3 gate measured cold-L2 graph medians of pair 216.9/42.8/14.8 us against fused 71.0/14.0/5.2 us. Against the FINDINGS rows (sliding ft 10947, turn_4 2153, turn_1 720 us) that is ~1.3% on sliding and ~0.8-1.2% on full, which matches `expected`, all directions faster. WEAKEST LINK: the headline. A weighted ~0.9% gain is well inside the CI half-width (~+-1.5%) and inside the stated +-3-4% per-row noise. So the score cannot confirm additivity, and the sliding-ft 'remaining 3% was drift' claim cannot be checked from rows either. Both can only be read from the profile: _norm_gather_kernel ~71/14/5 us replacing the pre_expert_norm + rega_gather rows, and the FA3 rows on full first_turn unchanged against master's job 1cc92a43. A second weak point is that the gate timings are cold-L2 isolated graphs. In-layer, the pair's norm output may partly hit L2 (pooler output 16384x4096 bf16 = 128 MiB at ft, so not at ft; but 6/24 MiB at turn_1/turn_4 does fit the 50 MB L2). The real turn_1/turn_4 saving may therefore be smaller than the gate ratio implies.

### Confounders on the current kernel
- fa3-full-contiguous-kv-triton-gather-gated (020566aa, master): the only source never co-run with norm-gather. It writes contiguous K/V buffers (persistent in state['fa3c']['route']) before FA3 on full layers. Norm-gather runs after o-proj, boundary and pooler on the same stream with no shared buffers. An L2 or allocator interaction is possible in principle but unlikely; only full-layer rows could show it, and sliding rows act as the internal control.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128: norm-gather exists only where rega_ok[class] is ON, and replaces rega_gather's input. If a rega gate flips OFF in this run (prepare noise near the 0.95 ft ratio is not a concern at 0.554), norm-gather silently becomes a no-op for that class. Read the [rega] PATH CHOSEN lines before attributing any row.
- align-two-phase-all-t-rega: produces the sorted_ids/block_experts/n_padded that the fused kernel consumes. The norm-gather self-check runs _align_dispatch on the live state, so it is already validated against align2. No confound to the timing beyond the shared base.
- sglang-pipeline-parity: defines state['pre_expert_norm'] (the module whose effective weight is bf16, hence round_w=True). The fused path's numerics are checked against it within 1 ulp; there is no timing confound.

### Interference and A/B needs
No companion. Checks are spent and this is a scored submit, so any companion would also cost a scored run. None would make the result less ambiguous than the run's own logs already do. The interaction under test (norm-gather saving vs contiguous-KV FA3) is separated inside this single run: sliding layers are untouched by fa3c and carry norm-gather alone, while full layers carry both. The kernel-level readout (profile rows _norm_gather_kernel vs master's pre_expert_norm + rega_gather, and the FA3 rows vs 1cc92a43) attributes each side directly. An ab-revert of fa3c would just reproduce edf68fa3 (already scored, 1.4600), and an ab-revert of norm-gather is master (1cc92a43, 1.4587). Both arms of the 2x2 already exist on the board, so this run completes the factorial. Combination B also uses this run as its paired control. Judge it on: [norm-gather] PATH CHOSEN = fused in all three classes, the [fa3 contig] PATH CHOSEN line equal to master's (B at the three full first_turn shapes), profile rows as above, and no first_turn row slower than master by more than noise. A headline within CI of 1.4587-1.4600 is the expected outcome and is not a refutation.

### Open questions
- '[norm-gather] setup took 0.7s' in edf68fa3 looks low for three classes of self-checks plus 21-alternation cold-L2 timing (maybe Triton cache hits or a timer quirk). It is harmless under the 270 s budget, but confirm the per-class GATE lines print in full again here.
- At turn_1/turn_4 the pooler output (6/24 MiB) fits in L2, so the in-layer pair cost may be below the cold-L2 gate's 14.8/42.8 us. Does the profile show the expected ~10/29 us saving, or less?
- If the full first_turn FA3 rows differ from 1cc92a43 beyond noise while the [fa3 contig] PATH CHOSEN line is identical, that is the L2/allocator interaction the conjecture names. It would need a follow-up, not a revert.

