---
problem: muse_moe_w4a16
slug: align-two-phase-port-from17
created: 2026-10-25T145952Z
base: master
area: MoE align/dispatch
change: >
  Built on master (9080a71, submission_blob 60dfe35d); MoE align only, at 17 <= num_tokens <= 512
  (decode_32 bs=32, verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128
  tests). num_tokens <= 16 keeps master's _align_small byte for byte, and Marlin, silu_and_mul,
  the w2 atomic path and the post-norm tail are untouched. MECHANISM: master's profile (job
  7a0d900a1dd34666) shows _align_small_kernel at 4.2 us for decode_32 (one program building a
  [256, 32] one-hot and storing a CAP_BLOCK=1024 sentinel tile) and _align_parallel_kernel at 4.2
  us (bs=128) and 6.9 us (bs=256), where each of its 33 programs streams all T*8 route ids twice
  with a [512, 32] compare. Prefill replaced the same _align_parallel_kernel with a two-phase
  align (a count kernel, one program per CHUNK of route ids writing a [nchunks, 32] count table,
  then a place kernel, grid nchunks + 1, that prefix-sums the table, places its own chunk's routes
  at segment start + earlier chunks' count + in-chunk rank, and one tail program writing padding
  slots, the sentinel tail, block_experts and n_padded). It produced bit-identical outputs and cut
  align from 69 to 9 us at T=3012 and from 396 to 23 us at T=16384 (LESSONS 2026-10-25, prefill
  align-two-phase-all-t-rega, job 4e6f815a8dd945ba). Here the work per program drops from all T*8
  ids to one chunk, so the T=128/256 align should fall to about 2.5-3.5 us across both launches.
  CHANGE: (1) Port verbatim from
  muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.patch:
  _align2_count_kernel, _align2_place_kernel and _align_two_phase (keep DIST=0, the single tail
  program fills the sentinel, since cap is at most about 5.1k slots here; drop the stage/kinfo
  diagnostics or leave them unused), plus ALIGN2_WARPS, ALIGN2_BLKT=256 and ALIGN2_SENT=2048. The
  kernels take raw global topk_ids with local_start/local_count exactly as master's
  _align_small/_align_parallel do, so pass bank['start'] and bank['count']. Use moe's own bound
  constants ALIGN2_MIN_TOKENS = 17 and ALIGN2_MAX_TOKENS = 512 (LESSONS: never derive a new path's
  gate from ALIGN_FUSED_MAX_TOKENS). (2) Three T classes by shape only: 'd32' 17..32 (incumbent
  _align_small, bsm from _block_size_m), 'v128' 33..191 (incumbent _align_parallel, bsm after the
  two-smaller pick, 16 at T=128 and the bs=33 test), 'v256' 192..512 (incumbent _align_parallel,
  bsm 32 at T=256/257). (3) New prepare() step _align2_setup(state, device), run after
  _align_self_check and BEFORE _w2_atomic_self_check (so the w2 self-check at T=32 exercises the
  align that will be timed). For each class and each CHUNK in (64, 128, 256, 512) with CHUNK <=
  the class's T*8 rounded up to a power of 2: compile on dummy ids, then check equivalence with
  torch.equal on all four outputs (local_ids, sorted_ids, block_experts, n_padded) against the
  class incumbent called with the same bsm, over T in {17, 24, 32} / {33, 57, 120, 128, 191} /
  {192, 256, 257, 512}, local_start in {0, 32, 224}, and the five routings of master's
  _align_self_check (random, skewed, all-remote, all-one-expert, all-local) plus an
  all-local-saturated case (every route to local experts, so the tail writes the most blocks).
  Compile and check EVERY block_size_m a class can reach, enumerated in prepare() by running
  kernel_fn's own pick (_block_size_m plus the two-smaller rule at T >= 128) over every T in the
  class: d32 sees bsm 8 and 16, v128 sees 16, 32 and 48 (T about 108..127 picks 48), v256 sees 32;
  add one equivalence T per reachable bsm. Nothing may JIT-compile inside kernel_fn. Any mismatch
  on any bsm disables that (class, CHUNK) and prints the first differing index. (4) Time each
  surviving candidate against the incumbent at the class's representative T (32, 128, 256) on
  skewed synthetic routing: capture each as its own CUDA graph, flush L2 with a 256 MB buffer read
  before every replay, interleave 41 replays of each, subtract an empty-graph baseline, take
  medians. Enable the class only with the fastest CHUNK, and only if incumbent_median /
  align2_median >= 1.05 (a win by a margin, not a tie; LESSONS 2026-10-25). Assert the stored
  choice is the printed winner. Store state['align2_ok'] = {'d32': chunk or 0, 'v128': chunk or 0,
  'v256': chunk or 0}, a plain dict of ints (the judge copies state, so no pointers or data_ptr
  checks). (5) kernel_fn / _forward: when expert_map is not None and ALIGN2_MIN_TOKENS <=
  num_tokens <= ALIGN2_MAX_TOKENS and state['align2_ok'][cls] > 0, call
  _align_two_phase(data['topk_ids'], bank['start'], bank['count'], block_size_m, chunk); otherwise
  master's branch unchanged. The count table is torch.empty((nchunks, 32), int32) sized by T only;
  no host sync, no prints, no try/except, nothing cached from inputs. (6) Print at the very end of
  prepare() (the archived log is a ~15 KB tail): one line per class, [align2] PATH CHOSEN
  class=<cls> T=<T> bsm=<bsm> = align2 chunk <c> | master, incumbent <x> us, align2 <y> us, ratio
  <r>, equivalence <n> cases ok.
expected:
  - shapes: bs=256 load=verify_32 sharding=ep8
    direction: up
    why: _align_parallel_kernel 6.9 us -> about 3 us across the two align2 launches (one chunk per program instead of all 2048 ids twice in each of 33 programs); about -3 to -4 us of 459 us, +0.5-0.9% on the row. Read the _align2_* rows against 6.9 us in the profile.
  - shapes: bs=128 load=verify_16 sharding=ep8
    direction: up
    why: _align_parallel_kernel 4.2 us -> about 2.5 us; +0.3-0.6% on a 315 us row. The gate keeps master if the prepare timing does not show >= 1.05x.
  - shapes: bs=32 load=decode_32 sharding=ep8
    direction: up
    why: "_align_small_kernel 4.2 us (one program, [256,32] one-hot, 1024-slot sentinel tile) -> about 2.5-3 us; +0.5-0.8% on the weakest-ratio load (1.034x). align-parallel-from-17-tokens moved this row +0.7% with the 33-program kernel."
  - shapes: bs=1 load=decode_1 sharding=ep8
    direction: flat
    why: "T <= 16 keeps master's _align_small byte for byte."
  - shapes: bs=16 load=decode_16 sharding=ep8
    direction: flat
    why: untouched (T <= 16).
  - shapes: bs=8 load=verify_1 sharding=ep8
    direction: flat
    why: untouched (T <= 16).
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T150410Z
review_companions: null
---

# align-two-phase-port-from17

## Change

Built on master (9080a71, submission_blob 60dfe35d); MoE align only, at 17 <= num_tokens <= 512 (decode_32 bs=32, verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128 tests). num_tokens <= 16 keeps master's _align_small byte for byte, and Marlin, silu_and_mul, the w2 atomic path and the post-norm tail are untouched. MECHANISM: master's profile (job 7a0d900a1dd34666) shows _align_small_kernel at 4.2 us for decode_32 (one program building a [256, 32] one-hot and storing a CAP_BLOCK=1024 sentinel tile) and _align_parallel_kernel at 4.2 us (bs=128) and 6.9 us (bs=256), where each of its 33 programs streams all T*8 route ids twice with a [512, 32] compare. Prefill replaced the same _align_parallel_kernel with a two-phase align (a count kernel, one program per CHUNK of route ids writing a [nchunks, 32] count table, then a place kernel, grid nchunks + 1, that prefix-sums the table, places its own chunk's routes at segment start + earlier chunks' count + in-chunk rank, and one tail program writing padding slots, the sentinel tail, block_experts and n_padded). It produced bit-identical outputs and cut align from 69 to 9 us at T=3012 and from 396 to 23 us at T=16384 (LESSONS 2026-10-25, prefill align-two-phase-all-t-rega, job 4e6f815a8dd945ba). Here the work per program drops from all T*8 ids to one chunk, so the T=128/256 align should fall to about 2.5-3.5 us across both launches. CHANGE: (1) Port verbatim from muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.patch: _align2_count_kernel, _align2_place_kernel and _align_two_phase (keep DIST=0, the single tail program fills the sentinel, since cap is at most about 5.1k slots here; drop the stage/kinfo diagnostics or leave them unused), plus ALIGN2_WARPS, ALIGN2_BLKT=256 and ALIGN2_SENT=2048. The kernels take raw global topk_ids with local_start/local_count exactly as master's _align_small/_align_parallel do, so pass bank['start'] and bank['count']. Use moe's own bound constants ALIGN2_MIN_TOKENS = 17 and ALIGN2_MAX_TOKENS = 512 (LESSONS: never derive a new path's gate from ALIGN_FUSED_MAX_TOKENS). (2) Three T classes by shape only: 'd32' 17..32 (incumbent _align_small, bsm from _block_size_m), 'v128' 33..191 (incumbent _align_parallel, bsm after the two-smaller pick, 16 at T=128 and the bs=33 test), 'v256' 192..512 (incumbent _align_parallel, bsm 32 at T=256/257). (3) New prepare() step _align2_setup(state, device), run after _align_self_check and BEFORE _w2_atomic_self_check (so the w2 self-check at T=32 exercises the align that will be timed). For each class and each CHUNK in (64, 128, 256, 512) with CHUNK <= the class's T*8 rounded up to a power of 2: compile on dummy ids, then check equivalence with torch.equal on all four outputs (local_ids, sorted_ids, block_experts, n_padded) against the class incumbent called with the same bsm, over T in {17, 24, 32} / {33, 57, 120, 128, 191} / {192, 256, 257, 512}, local_start in {0, 32, 224}, and the five routings of master's _align_self_check (random, skewed, all-remote, all-one-expert, all-local) plus an all-local-saturated case (every route to local experts, so the tail writes the most blocks). Compile and check EVERY block_size_m a class can reach, enumerated in prepare() by running kernel_fn's own pick (_block_size_m plus the two-smaller rule at T >= 128) over every T in the class: d32 sees bsm 8 and 16, v128 sees 16, 32 and 48 (T about 108..127 picks 48), v256 sees 32; add one equivalence T per reachable bsm. Nothing may JIT-compile inside kernel_fn. Any mismatch on any bsm disables that (class, CHUNK) and prints the first differing index. (4) Time each surviving candidate against the incumbent at the class's representative T (32, 128, 256) on skewed synthetic routing: capture each as its own CUDA graph, flush L2 with a 256 MB buffer read before every replay, interleave 41 replays of each, subtract an empty-graph baseline, take medians. Enable the class only with the fastest CHUNK, and only if incumbent_median / align2_median >= 1.05 (a win by a margin, not a tie; LESSONS 2026-10-25). Assert the stored choice is the printed winner. Store state['align2_ok'] = {'d32': chunk or 0, 'v128': chunk or 0, 'v256': chunk or 0}, a plain dict of ints (the judge copies state, so no pointers or data_ptr checks). (5) kernel_fn / _forward: when expert_map is not None and ALIGN2_MIN_TOKENS <= num_tokens <= ALIGN2_MAX_TOKENS and state['align2_ok'][cls] > 0, call _align_two_phase(data['topk_ids'], bank['start'], bank['count'], block_size_m, chunk); otherwise master's branch unchanged. The count table is torch.empty((nchunks, 32), int32) sized by T only; no host sync, no prints, no try/except, nothing cached from inputs. (6) Print at the very end of prepare() (the archived log is a ~15 KB tail): one line per class, [align2] PATH CHOSEN class=<cls> T=<T> bsm=<bsm> = align2 chunk <c> | master, incumbent <x> us, align2 <y> us, ratio <r>, equivalence <n> cases ok.

## Expected

- `bs=256 load=verify_32 sharding=ep8` up: _align_parallel_kernel 6.9 us -> about 3 us across the two align2 launches (one chunk per program instead of all 2048 ids twice in each of 33 programs); about -3 to -4 us of 459 us, +0.5-0.9% on the row. Read the _align2_* rows against 6.9 us in the profile.
- `bs=128 load=verify_16 sharding=ep8` up: _align_parallel_kernel 4.2 us -> about 2.5 us; +0.3-0.6% on a 315 us row. The gate keeps master if the prepare timing does not show >= 1.05x.
- `bs=32 load=decode_32 sharding=ep8` up: _align_small_kernel 4.2 us (one program, [256,32] one-hot, 1024-slot sentinel tile) -> about 2.5-3 us; +0.5-0.8% on the weakest-ratio load (1.034x). align-parallel-from-17-tokens moved this row +0.7% with the 33-program kernel.
- `bs=1 load=decode_1 sharding=ep8` flat: T <= 16 keeps master's _align_small byte for byte.
- `bs=16 load=decode_16 sharding=ep8` flat: untouched (T <= 16).
- `bs=8 load=verify_1 sharding=ep8` flat: untouched (T <= 16).

## Risk

Risk low: every class is gated on bit-identical outputs against its incumbent and a >= 1.05x cold-L2 graph-replay win, so a failed gate measures master. The expected gain per row is 0.3-0.9%, below the +-1.5% single-run resolution of the headline; judge it on the per-shape _align2 kernel rows against master's 4.2/4.2/6.9 us and the paired ratio, not the headline. Overlaps the range of cuda-align-kernel-a-from17 (hold; a CUDA one-CTA align) with a different mechanism: this ports the Triton two-phase align prefill already proved bit-identical. Two launches instead of one add one graph node (~0.3 us, LESSONS), already included in the estimate. bsm 48 occurs only at unscored T of about 108..127 (SGLang's pick below the two-smaller rule); PAD_BLOCK uses next_power_of_2(bsm), as in prefill, and the equivalence suite covers it.

## Adversarial review

Reviewed 2026-10-25T150410Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run it as a single scored submit with no companion. It ports a Triton align that prefill already showed was bit-identical. Each T class is gated in prepare() on torch.equal plus a 1.05x cold-L2 graph win, and that prints its own in-situ A/B, so the per-kernel profile rows and the [align2] lines can be read even though the headline effect (about 0.1-0.2% geomean) is far below the single-run CI. Before building, fix one gap: prepare() must compile every NCH_P2 variant.

### Justification
A. The mechanism is plausible at bs=128/256. Master's _align_parallel_kernel has 33 programs, and each streams all T*8 ids twice with a [512,32] one-hot. At T=256 that is 2048 ids x 2 passes per program, which matches 6.9 us in master's profile (4.0-4.2 us at bs=128 in FINDINGS). The two-phase form cuts each program's work to one CHUNK, and prefill measured 69->9 us at T=3012, bit-identical. I read the ported kernels (muse_prefill_layer/.../align-two-phase-all-t-rega.patch) against moe's incumbents. Placement keeps ascending route order (pre-chunk counts + in-chunk rank). block_experts from e_of = #(seg_end <= b*BSM) agrees with the incumbent when experts have zero routes. local_ids uses the same mask. The sentinel/padding/placement stores are disjoint. So bit-identity on raw global ids with local_start is credible. The bsm enumeration matches the code: d32 sees 8 (T<=28) and 16 (T 29-32). v128 sees 16 (33-57 and 128-172), 32 (58-115 and 173-191) and 48 (116-127; the hypothesis says ~108, but prepare enumerates it anyway). v256 sees 32 only. The expected directions agree with TASK.md: each load counts equally in the geomean, and the align is about 0.9-2.3% of each touched row, so the predicted +0.3-0.9% per row is the right order. That is about 0.1-0.2% on the headline, well inside the +-1.5% CI, and the hypothesis says so itself. WEAKEST LINK: the d32 class. The incumbent there is one launch at 4.2 us, and the two-phase form adds a second graph node to work that is only 136-256 ids (2-4 chunks at CHUNK 64). Two minimum-size Triton kernels plus the inter-node gap on H200 cost roughly 3-4 us, so the 1.05x gate will probably leave d32 on master, and the decode_32 'up' prediction (the load with the weakest ratio) is the claim least likely to hold. A second link: the 2.5-3.5 us estimate at T=128/256 is extrapolated from prefill's T>=3012. At T=128 (1024 ids) launch latency rather than work dominates, so v128 may also fail the 1.05 margin. Either way the gate turns a failed prediction into a measurement of master, not a regression.

### Confounders on the current kernel
- triton-map-align-on-block-m-restack / marlin-block-m-two-smaller-on-postnorm: these set block_size_m (16 at T=128, 32 at T=256). That fixes PAD_BLOCK, n_blocks and cap, so the align cost depends on them. This confound is benign: the hypothesis compares at the same bsm the timed path uses and enumerates bsm from kernel_fn's own pick, so the measured delta is the align's alone.
- align-parallel-per-expert-verify: this landed the incumbent at T>32, so the comparison is align2 against exactly that kernel. Nothing interacts beyond the comparison itself.
- triton-map-align-small-m-port: this landed the d32 incumbent (_align_small). Its single-launch shape is the reason d32 is the hardest class to win, which is a property of the baseline and not an interaction.
- w2-marlin-atomic-add-small-m: at T<=32, _w2_atomic_self_check runs _forward. Putting _align2_setup before it (as specified) means the w2 mode is chosen with the align that will be timed. If the order were reversed, d32 could time an align/w2 pairing that was never checked. Keep the specified order.
- Not landed but relevant: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate is BEST.json (1.3727x) but is NOT on master (master blob 60dfe35d is 9080a71, measured at 1.3445x). It runs its own align at WG_BSM for 128<=T<=512. This run's headline must be read against master's 1.3445x, not BEST, or it will look like a ~2% loss. If the wgmma rewrite or restack-bucket-gate-on-master-align-atomic lands first, the v128/v256 classes only matter for buckets that fall back to Marlin, and the two edit the same _forward branch.

### Interference and A/B needs
No companion is warranted. Two things already disambiguate a gain or a loss: (a) the prepare() gate prints, per class, the incumbent us, the align2 us, the ratio and the equivalence count, measured interleaved in the same process as an in-situ A/B; (b) the submit profile reports _align2_count/_align2_place rows to set against master's 4.2/4.2/6.9 us rows. An ab-revert of any landed hypothesis would not make this clearer, because the change is a bit-identical substitute for one kernel at a fixed bsm. A control that isolated the count or place launch would not ship and would only repeat what the profile rows already show. The headline cannot resolve 0.1-0.2% in any case, and a companion would cost a second scored run without adding information. Read the result as follows: classes the gate enabled plus profile rows below master's mean a real (small) gain; all classes printed as master means the run measured master (an unproven or flat result, not a failure).

### Open questions
- Does the d32 class pass the 1.05x gate at all with two launches over 2-4 chunks? If it does not, the decode_32 prediction is moot and only v128/v256 are tested.
- Should CHUNK 32 be a candidate for d32/v128? At n=136-1024, CHUNK 64 still leaves only 3-16 programs, so a smaller chunk or fewer warps (ALIGN2_WARPS=8 is sized for prefill's 512-1024 chunks) may be what decides the small classes.
- Merge order with the wgmma rewrite (restack-bucket-gate-on-master-align-atomic is queued): the v128/v256 benefit disappears for any bucket where wgmma takes over with its own WG_BSM align. Should a later stacked hypothesis point align2 at that path?
- The headline will be classified against BEST (1.3727x, not on master). Make sure the reader compares it with master's 1.3445x and the per-kernel rows.

