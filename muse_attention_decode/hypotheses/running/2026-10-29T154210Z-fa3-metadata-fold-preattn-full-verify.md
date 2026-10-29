---
problem: muse_attention_decode
slug: fa3-metadata-fold-preattn-full-verify
created: 2026-10-29T154210Z
base: master
mint_blob: "2eb03f76ceefa9bf81e28616ced3b33be3a70443"
best_at_mint: 1.634
area: FA3 call (scheduler metadata folded into pre-attention, full verify)
change: >
  Fold FA3's varlen scheduler metadata (what flash::prepare_varlen_num_blocks_kernel<NumWarps,
  Sort> writes, 2.8-3.4 us per call on the board profile) into master's existing
  _pre_attention_kernel launch on the three full-layer verify rows (window 0, q_len 8, page 1,
  tp8, B = 1 unpaired and B = 16 / 32 paired), and pass the result as scheduler_metadata= to
  flash_attn_with_kvcache so FA3 skips its in-call prepare node
  (params.skip_scheduler_metadata_computation). Nothing is cached: the metadata is recomputed from
  this call's cache_seqlens on every call, into a fresh torch.empty int32 buffer. Only shape-only
  constants are frozen at import. (1) LAYOUT, for this build (the profile names
  prepare_varlen_num_blocks_kernel<1|2, true>, i.e. upstream hopper/flash_prepare_scheduler.cu
  with batch sort; causal non-local so varlen_sort_batches and head_swizzle are both on): an int32
  vector of 4*b_rounded + 1 (b_rounded = round_up(B, 4)) laid out as [0, b_r) num_splits_dynamic,
  [b_r, 2b_r) num_m_blocks, [2b_r, 3b_r) varlen_batch_idx, [3b_r, 4b_r) num_nheads_in_l2, [4b_r]
  tile_count_semaphore. Confirm the size at import from the library's own get_scheduler_metadata
  output (the shape is checked by mha_fwd's CHECK_SHAPE). (2) SEMANTICS to port (one program, B <=
  32): n_blocks = ceil(cache_seqlens[b] / BLOCK_N), m_blocks = ceil(L * (QHPK if PACKGQA else 1) /
  BLOCK_M); total = sum over b of m_blocks * n_blocks; blocks_per_sm = int(ceilf(float(total) *
  1.1f * float(NUM_HEAD) / float(NUM_SM))) in fp32 exactly; splits = max(min(ceil(n_blocks /
  blocks_per_sm), NUM_SPLITS_STATIC), 1) (1 when NUM_SPLITS_STATIC == 1); num_nheads_in_l2 =
  16/8/4/2/1 by the first n_blocks * k <= MAX_KVBLOCKS_IN_L2 (with the n_blocks the upstream code
  uses at that point), times QHPK when not PACKGQA, capped at NUM_HEAD; sort key (causal) =
  n_blocks * BLOCK_N - m_blocks * BLOCK_M; batches are placed in descending key order, stable on
  batch index (cub BlockMergeSort is a merge sort; implement it as rank_b = #{j: key_j > key_b} +
  #{j < b: key_j == key_b} over a [32, 32] compare tile, no tl.sort) and each vector is written at
  the sorted (virtual) index, with varlen_batch_idx[rank_b] = b; pad slots [B, b_r) of each vector
  are written 0; the semaphore is written 0 every call (the prepare kernel zeroes it; with
  skip=true nothing else does, so this is required for graph replay). Port the exact statement
  order from the upstream file the profile names (where n_blocks is reduced by splits relative to
  the nheads_in_l2 and key computation) rather than from this paragraph; step (4) is the arbiter.
  (3) WHERE: add constexpr META (default False, so every other route's compile is unchanged) plus
  the constexprs above and pointers seqlens_ptr / meta_ptr to _pre_attention_kernel; launch it
  with grid (tokens + 1, hq + hkv) on the routed rows; program (t == tokens, h == 0) computes and
  stores the metadata with 1 warp, programs (t == tokens, h > 0) return at once, every other
  program is unchanged. In kernel_fn, under a config-and-shape-only predicate _fm_gate(cfg,
  page_table.shape[1]) that reads an import-frozen MappingProxyType _FM_ROUTE keyed by (B,
  max_pages) -> the constexpr tuple, allocate meta = torch.empty(4*b_r + 1, int32), launch the
  pre-attention kernel with META=True, and pass metadata = meta to FA3 in place of None (verify_1
  unpaired today passes data["scheduler_metadata"], which is None on verify; B = 16 / 32 paired
  pass None). Nothing else changes: q layout, pairing, window/causal args, descale, epilogues. (4)
  IMPORT GATE (_fm_setup, per row B in {1, 16, 32} at the judge's shape: ctx 80000 coherent, its
  page-table width): (a) recover the shape-only constants (BLOCK_M, BLOCK_N, PACKGQA, NUM_HEAD =
  h_k-of-call if PACKGQA else hq, NUM_SPLITS_STATIC, NUM_SM = multiProcessorCount,
  MAX_KVBLOCKS_IN_L2 = 8 MiB / (BLOCK_N * (hd + hd) * 1)) by matching a torch emulation of (2)
  against sgl_kernel.flash_attn.get_scheduler_metadata called with exactly the arguments mha_fwd
  derives for this call (as fa3-scheduler-metadata-side-stream did: num_heads_k and headdim of the
  actual paired / unpaired views, max_seqlen_k = page_table.shape[1] * page_size, causal=True,
  window (-1, -1) after mha_fwd's clamps, qkv_dtype e4m3, num_splits=0) over >= 64 draws of the
  judge's length distribution plus edge sets (all-equal lengths for sort ties, lengths below
  BLOCK_N, lengths == L, one long + many short); every candidate tuple that matches all samples
  must agree on a 64-draw holdout, else the row is not routed. (b) The Triton META program's
  output equals the library's on every valid slot ([0, B) of each vector, and the semaphore) for
  all samples. (c) FA3 output and the full kernel_fn output with the folded metadata are
  torch.equal to master's (metadata None) on >= 8 draws, and two consecutive replays of one
  captured graph are equal (semaphore reset). (d) Race: alternating CUDA-graph replays of master's
  kernel_fn step vs the fold on cloned judge-shape inputs, two 31-rep rounds; route when both
  rounds are >= 1.02x at B = 1 and >= 1.005x at B = 16 / 32 with the IQR bound master's gates use;
  else that row stays master. Print one PATH CHOSEN line per row with the measured step times, the
  prepare-node delta and why. Fallback everywhere = master unchanged.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: "prepare_varlen<1,true> 2.8 us node gone, ~2.0 us net after the extra pre-attn program: 30.1 -> ~28.1 us; verify_1 load 1120.8 us, 0.238%/us -> +0.48% headline"
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: prepare<1,true> 3.1 us; f1043d2c measured precomputed metadata -2.2 us in-graph at B=16; ~2.0 us net of 162.7 (1.2%); 0.076%/us -> +0.15%
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: prepare<2,true> 3.4 us; f1043d2c -2.4 us at B=32; ~2.2 us of 302.1 (0.7%, inside drift); 0.044%/us -> +0.10%. Sum ~0.73% headline
  - shapes: "*layer=sliding*"
    direction: same
    why: "untouched: META=False compile, sliding routes unchanged"
  - shapes: "*load=decode_*"
    direction: same
    why: "untouched: full decode is on the stream-K / c320 routes, never this FA3 call"
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "fa3-scheduler-metadata-side-stream: that run still launched prepare_varlen via get_scheduler_metadata on a forked side stream (an extra node plus fork/join edges; full verify_1 lost 0.7 us, sliding lost 6-9%); this launches no scheduler kernel and no stream at all: the metadata is computed by one extra program inside the existing _pre_attention_kernel launch, removing the prepare node, on full verify rows only (sliding untouched). It is the precomputed-metadata saving the claim's reopen cites (f1043d2c: -2.2/-2.4 us at B=16/32), taken legally by per-call recompute, outside the attn-fixed rewrite by user request."
target: check
value_gain: 0.73
value_p: 0.5
value_ev: 0.36
value_cost: 300k tokens, 1 judge runs
evidence: >
  board 4e386647 profile: prepare_varlen_num_blocks_kernel<1,true> 2.8 us (full verify_1, 10.3% of
  30.1 us), <1,true> 3.1 us (verify_16, 162.7 us), <2,true> 3.4 us (verify_32, 302.1 us); anatomy
  f1043d2c profile.log:115-116: precomputed scheduler metadata 80k 148.3 vs 150.4 us (B=16, -2.2)
  and 275.3 vs 277.8 (B=32, -2.4) in-graph; side-stream 42964d938c5842f4 full verify_1 30.23 ->
  30.94 us (moving, not removing, the node loses); row rates from attn-fixed.json: verify_1
  0.238%/us, verify_16 0.076%/us, verify_32 0.044%/us
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T155002Z
review_companions: null
review_best: 1.6339575246416982
review_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
stage: claimed
claimed_at: 2026-10-29T155002Z
base_sha: e18a2af19d9a60bde63f1ca29903b996d02a80a5
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/fa3-metadata-fold-preattn-full-verify
worktree: .worktrees/muse_attention_decode/fa3-metadata-fold-preattn-full-verify
mode: null
restack_of: null
stale_base: null
---

# fa3-metadata-fold-preattn-full-verify

## Change

Fold FA3's varlen scheduler metadata (what flash::prepare_varlen_num_blocks_kernel<NumWarps, Sort> writes, 2.8-3.4 us per call on the board profile) into master's existing _pre_attention_kernel launch on the three full-layer verify rows (window 0, q_len 8, page 1, tp8, B = 1 unpaired and B = 16 / 32 paired), and pass the result as scheduler_metadata= to flash_attn_with_kvcache so FA3 skips its in-call prepare node (params.skip_scheduler_metadata_computation). Nothing is cached: the metadata is recomputed from this call's cache_seqlens on every call, into a fresh torch.empty int32 buffer. Only shape-only constants are frozen at import. (1) LAYOUT, for this build (the profile names prepare_varlen_num_blocks_kernel<1|2, true>, i.e. upstream hopper/flash_prepare_scheduler.cu with batch sort; causal non-local so varlen_sort_batches and head_swizzle are both on): an int32 vector of 4*b_rounded + 1 (b_rounded = round_up(B, 4)) laid out as [0, b_r) num_splits_dynamic, [b_r, 2b_r) num_m_blocks, [2b_r, 3b_r) varlen_batch_idx, [3b_r, 4b_r) num_nheads_in_l2, [4b_r] tile_count_semaphore. Confirm the size at import from the library's own get_scheduler_metadata output (the shape is checked by mha_fwd's CHECK_SHAPE). (2) SEMANTICS to port (one program, B <= 32): n_blocks = ceil(cache_seqlens[b] / BLOCK_N), m_blocks = ceil(L * (QHPK if PACKGQA else 1) / BLOCK_M); total = sum over b of m_blocks * n_blocks; blocks_per_sm = int(ceilf(float(total) * 1.1f * float(NUM_HEAD) / float(NUM_SM))) in fp32 exactly; splits = max(min(ceil(n_blocks / blocks_per_sm), NUM_SPLITS_STATIC), 1) (1 when NUM_SPLITS_STATIC == 1); num_nheads_in_l2 = 16/8/4/2/1 by the first n_blocks * k <= MAX_KVBLOCKS_IN_L2 (with the n_blocks the upstream code uses at that point), times QHPK when not PACKGQA, capped at NUM_HEAD; sort key (causal) = n_blocks * BLOCK_N - m_blocks * BLOCK_M; batches are placed in descending key order, stable on batch index (cub BlockMergeSort is a merge sort; implement it as rank_b = #{j: key_j > key_b} + #{j < b: key_j == key_b} over a [32, 32] compare tile, no tl.sort) and each vector is written at the sorted (virtual) index, with varlen_batch_idx[rank_b] = b; pad slots [B, b_r) of each vector are written 0; the semaphore is written 0 every call (the prepare kernel zeroes it; with skip=true nothing else does, so this is required for graph replay). Port the exact statement order from the upstream file the profile names (where n_blocks is reduced by splits relative to the nheads_in_l2 and key computation) rather than from this paragraph; step (4) is the arbiter. (3) WHERE: add constexpr META (default False, so every other route's compile is unchanged) plus the constexprs above and pointers seqlens_ptr / meta_ptr to _pre_attention_kernel; launch it with grid (tokens + 1, hq + hkv) on the routed rows; program (t == tokens, h == 0) computes and stores the metadata with 1 warp, programs (t == tokens, h > 0) return at once, every other program is unchanged. In kernel_fn, under a config-and-shape-only predicate _fm_gate(cfg, page_table.shape[1]) that reads an import-frozen MappingProxyType _FM_ROUTE keyed by (B, max_pages) -> the constexpr tuple, allocate meta = torch.empty(4*b_r + 1, int32), launch the pre-attention kernel with META=True, and pass metadata = meta to FA3 in place of None (verify_1 unpaired today passes data["scheduler_metadata"], which is None on verify; B = 16 / 32 paired pass None). Nothing else changes: q layout, pairing, window/causal args, descale, epilogues. (4) IMPORT GATE (_fm_setup, per row B in {1, 16, 32} at the judge's shape: ctx 80000 coherent, its page-table width): (a) recover the shape-only constants (BLOCK_M, BLOCK_N, PACKGQA, NUM_HEAD = h_k-of-call if PACKGQA else hq, NUM_SPLITS_STATIC, NUM_SM = multiProcessorCount, MAX_KVBLOCKS_IN_L2 = 8 MiB / (BLOCK_N * (hd + hd) * 1)) by matching a torch emulation of (2) against sgl_kernel.flash_attn.get_scheduler_metadata called with exactly the arguments mha_fwd derives for this call (as fa3-scheduler-metadata-side-stream did: num_heads_k and headdim of the actual paired / unpaired views, max_seqlen_k = page_table.shape[1] * page_size, causal=True, window (-1, -1) after mha_fwd's clamps, qkv_dtype e4m3, num_splits=0) over >= 64 draws of the judge's length distribution plus edge sets (all-equal lengths for sort ties, lengths below BLOCK_N, lengths == L, one long + many short); every candidate tuple that matches all samples must agree on a 64-draw holdout, else the row is not routed. (b) The Triton META program's output equals the library's on every valid slot ([0, B) of each vector, and the semaphore) for all samples. (c) FA3 output and the full kernel_fn output with the folded metadata are torch.equal to master's (metadata None) on >= 8 draws, and two consecutive replays of one captured graph are equal (semaphore reset). (d) Race: alternating CUDA-graph replays of master's kernel_fn step vs the fold on cloned judge-shape inputs, two 31-rep rounds; route when both rounds are >= 1.02x at B = 1 and >= 1.005x at B = 16 / 32 with the IQR bound master's gates use; else that row stays master. Print one PATH CHOSEN line per row with the measured step times, the prepare-node delta and why. Fallback everywhere = master unchanged.

## Expected

- `*layer=full load=verify_1 *` faster: prepare_varlen<1,true> 2.8 us node gone, ~2.0 us net after the extra pre-attn program: 30.1 -> ~28.1 us; verify_1 load 1120.8 us, 0.238%/us -> +0.48% headline
- `*layer=full load=verify_16 *` faster: prepare<1,true> 3.1 us; f1043d2c measured precomputed metadata -2.2 us in-graph at B=16; ~2.0 us net of 162.7 (1.2%); 0.076%/us -> +0.15%
- `*layer=full load=verify_32 *` faster: prepare<2,true> 3.4 us; f1043d2c -2.4 us at B=32; ~2.2 us of 302.1 (0.7%, inside drift); 0.044%/us -> +0.10%. Sum ~0.73% headline
- `*layer=sliding*` same: untouched: META=False compile, sliding routes unchanged
- `*load=decode_*` same: untouched: full decode is on the stream-K / c320 routes, never this FA3 call

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** board 4e386647 profile: prepare_varlen_num_blocks_kernel<1,true> 2.8 us (full verify_1, 10.3% of 30.1 us), <1,true> 3.1 us (verify_16, 162.7 us), <2,true> 3.4 us (verify_32, 302.1 us); anatomy f1043d2c profile.log:115-116: precomputed scheduler metadata 80k 148.3 vs 150.4 us (B=16, -2.2) and 275.3 vs 277.8 (B=32, -2.4) in-graph; side-stream 42964d938c5842f4 full verify_1 30.23 -> 30.94 us (moving, not removing, the node loses); row rates from attn-fixed.json: verify_1 0.238%/us, verify_16 0.076%/us, verify_32 0.044%/us
- **Gain if it works:** 0.73% headline; **p** 0.5; **EV** 0.36% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 0.73% < 2.5%; EV 0.36% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge runs.

## Risk

User-requested under 'run all you want that youre able to' (2026-10-29 1504Z); below the value bar: ~0.73% headline is under the judge margin (1.24%), so it cannot land alone and is read per row (prepare row gone, full verify rows -2 us) or stacked later. Correctness risk: the port must match this build's prepare_varlen_num_blocks_kernel<NumWarps, Sort=true> bit for bit (statement order, fp32 ceilf, stable descending sort, nheads_in_l2 rule, semaphore zero); the import gate compares against the library's own get_scheduler_metadata and FA3 output torch.equal, and falls back to master on any mismatch, so the failure mode is 'no route' (flat), not wrong output. Version risk: if the installed build differs from upstream (vector count, layout, pad semantics), gate (a) finds no matching tuple and nothing routes. Perf risk: the extra program (one warp, B <= 32, [32,32] rank tile) must not lengthen the pre-attention launch by more than ~0.3 us; the race catches it. The side-stream attempt showed full verify_1 losing 0.7 us when the prepare kernel was moved rather than removed; here no kernel and no stream edge is added. Not input-derived caching: only shape-only constants (block sizes, static split cap, SM count) are frozen per (B, page-table width); metadata is rebuilt every call.

## Adversarial review

Reviewed 2026-10-29T155002Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. Fresh (board 1.6340, blob 2eb03f76) and user-requested; differs in mechanism from the refuted side-stream attempt (CLAIMS.md:53): it removes FA3's prepare node rather than moving it, the saving that claim's reopen measured. The headline is unreadable (0.73% inside the CI and the ~1% swing from flipping sliding-verify gates, CLAIMS.md:64), so judge per row: full verify_1, the profile's prepare row, and the in-run race lines. Before building, fix two places where an unchecked mistake could give wrong output instead of no route: the split cap (NUM_SPLITS_STATIC) and the guard on the t == tokens programs.

### Justification
Every full-verify row on 4e386647 and f1043d2c shows prepare_varlen_num_blocks_kernel<1|2,true>: 2.7-2.8 us (B=1), 3.1 (B=16), 3.4-3.5 (B=32). Passing scheduler_metadata sets skip_scheduler_metadata_computation and removes the node (proved by full decode_1 with the judge's metadata). f1043d2c measured the in-graph saving at -2.1 (B=16) / -2.5 us (B=32), ~70% of the row because FA3 PDL-overlaps part of prepare. The stated layout matches upstream mha_fwd but cannot be verified on this node (no sgl_kernel), so gates (a) size match and (b) bitwise match arbitrate; a differing build means no route. verify_1 uses spread 'fixed' (length 80000 at B=1), so the B=1 bitwise gate checks one point; route keys (B, max_pages) are known at import. Value arithmetic checks out (0.238/0.076/0.044 %/us). WEAKEST LINK: the B=1 row carries 0.48 of the 0.73% and is unmeasured; pre-attention there is 1.4-1.6 us so the metadata program is least hidden and removing prepare loses its prologue overlap: net 1.2-1.7 us as likely as 2.0. B=32's 0.7% sits at the 1.005x race bar, so its route may flip. Prior difference real (side-stream 42964d93 still launched prepare; fa3-scheduler-metadata-full-verify never ran). CLAIMS.md:42 does not apply.

### Confounders on the current kernel
- fused-preattention-triton: its span (1.4-1.6 us B=1, 2.7-4.1 us B=16/32) sets how much the metadata program hides; not an attribution confounder because the race is paired in-run.
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup: set the geometry (B=16/32 paired view num_heads_k 1, hd 128; B=1 unpaired hkv 2, hd 64); constants must be recovered per row from the actual views.
- s2b / sw17 sliding verify gates (CLAIMS.md:58, 60, 64): flip between runs, ~+-1% headline; read full verify rows, the profile and PATH CHOSEN lines, never the headline.
- FA3's own PDL launch after prepare: part of the row already hidden; f1043d2c accounts for it at B=16/32, not at B=1.

### Interference and A/B needs
No companions; gate (d) alternates graph replays of master's step vs the fold on cloned judge-shape inputs and prints step times and the prepare delta per row; the profile shows whether the prepare row is gone and FA3 main/combine stayed within ~1% of 4e386647. Full verify_1 (30.1 -> ~28.1 us, ~6.6%) is readable from one run; verify_16 (1.2%) and verify_32 (0.7%) only from race lines plus profile. Record as a per-row [shelf] (a stacker, e.g. with CLAIMS.md:65), not a board candidate.

### Open questions
- Tie order in varlen_batch_idx: try both tie rules (stable on batch index vs other) on the all-equal-length edge set and adopt the one that matches the library.
- fp32 blocks_per_sm: FA3 built with --use_fast_math; Triton div.full.f32 may differ by 1 ulp near integers; memory-safe only with a correct cap.
- Gate's get_scheduler_metadata call must pass cu_seqlens_q and max_seqlen_k = page_table.shape[1]*page_size.
- Print the metadata program's added pre-attention time (META on vs off) in PATH CHOSEN to separate fold overhead from lost PDL overlap at B=1.
- Consider >= 1.00x bars at B=16/32 once gates (a)-(c) pass (bitwise-equal output, a node removed).
- Confirm the harvest matcher treats direction 'same' as 'flat'.

