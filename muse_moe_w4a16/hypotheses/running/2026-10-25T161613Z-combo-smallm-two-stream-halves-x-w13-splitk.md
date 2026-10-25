---
problem: muse_moe_w4a16
slug: combo-smallm-two-stream-halves-x-w13-splitk
created: 2026-10-25T161613Z
base: master
area: "combination: small-M Marlin two-stream halves x w13 split-K atomic + shared floor"
change: >
  COMBINATION (conjectural pairing, small-M Marlin side): two-stream expert halves paired with w13
  split-K atomic add (w2 atomic is already on master) at 8 <= T <= 32, on top of the shared floor.
  Built on master a0e52640. SHARED FLOOR (identical in all three combo hypotheses minted
  2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1: w13 Marlin
  use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only, branch
  hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port
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
  warm per T is most of it). (C3a) Two-stream halves. Source
  marlin-two-stream-expert-halves-small-m, branch
  hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m, impl commit ea345883: port
  _align_small_halves_kernel (grid 2, one program per expert half, its own sorted_ids_h /
  block_experts_h / n_padded_h / local_ids_h, HALF_SORTED_ALIGN), _experts_two_stream
  (side.wait_stream(current); half A on the current stream, half B under torch.cuda.stream(side);
  w13 Marlin, silu_and_mul with expert_ids=local_ids_h, w2 Marlin into the shared cache3;
  current.wait_stream(side); master's post-norm on the full local_ids), the second Marlin lock
  workspace and the side stream, both created in prepare and held in the state. Measured:
  decode_32 1.042x (job 5d630369b1634f77, best decode_32 row of any run) but verify_1 1.316x vs
  1.33-1.35x, decode_16 flat; that run predates master's w2 atomic add. (C3b) w13 atomic at T <=
  32. Source w13-marlin-atomic-add-on-master-small-m, branch
  hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m, impl commit c3a84420
  (W13_ATOMIC_MAX_TOKENS 32, _w13_band, _w13_atomic_self_check): flat on its own at T = 8/16/32
  (1.353/1.281/1.035x), +2.6% at T=1 (that row is F1). In the halves path both Marlin calls of
  each half take the w13 kwargs of the chosen mode and w2 keeps master's w2_mode (each half writes
  disjoint route rows of cache1/cache3, so atomic modes stay row-disjoint; if w2_mode is
  atomic_zero, the zeroing happens once on the current stream before the fork). GATE per T band,
  bands chosen by shape only: [8, 12] rep T=8 (verify_1 and the bs=4-adjacent range), [13, 24] rep
  T=16, [25, 32] rep T=32; candidates {master, H = halves, H+A = halves with w13 atomic, A = w13
  atomic alone}. Each candidate: halves self-check (per-half routes and block counts equal to
  moe_align_block_size with the other half set to -1, as the source did), rel-L2 vs master <= 1e-2
  and finite on six routings, graph capture with the fork/join (capturable; the source ran it on
  the judge), then full kernel_fn cold-L2 graphs, 41 interleaved reps, judge-like skewed and
  random routings; the band takes the lowest median only at <= 0.98x master on the worse routing.
  Print PATH CHOSEN per band with all four medians. state['sm_mode'] = dict band -> int set in
  prepare; kernel_fn picks by num_tokens band only. ROUTING SUMMARY: T == 1 -> F1; 8 <= T <= 32 ->
  per band master / H / H+A / A; 256 <= T <= 512 -> wgmma with F2; everything else master.
  Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C3 compile and warm of the halves kernel
  for T 8..32 (~10 s, start deadline 150 s) + self-checks ~5 s + gate 3 bands x 4 candidates x 2
  routings (~6 s, start deadline 230 s) = ~95 s of 300. RULES (every combo phase): kernel_fn
  branches on shape and prepare-set plain ints/bools only; no host sync, no input-value-sized
  allocation, no data-dependent control flow, no prints, no try/except, no per-call mutable state,
  no timing, nothing cached from inputs; NVFP4 weights stay NVFP4 (register dequant only), BF16
  activations where SGLang keeps them, fp32 accumulation; every local route computed; no child
  processes. The judge prepares TWO states and copies them by cloning tensors: every scratch
  tensor, stream, workspace and error word lives in the state, no module-global device pointer,
  tensor map or tensor, and tensor maps stay encoded per call (master's s5 rule). PREPARE ORDER:
  master's prepare() runs first and unchanged (align self-check, _align2_setup,
  _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 / WG_REPLAY_DEADLINE_S 235 /
  WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs AFTER it, measures its own
  elapsed time from prepare()'s t_prepare, skips itself (flag stays master, prints PATH CHOSEN =
  master (deadline)) if it would start past its start deadline, and no phase starts timing past
  270 s. GATE RULE (every source): exactness or tolerance check first, then cold-L2 CUDA-graph
  timing vs master's path for that class (>= 256 MB flush before each replay, interleaved A/B, >=
  21 reps, empty-graph baseline subtracted, median); ON only at new/master <= 0.98 on the worse of
  the judge-like skewed routing and random routing; otherwise that class keeps master. Print one
  PATH CHOSEN line per source per class with both medians and the ratio, and keep those lines in
  the last 16k characters of the log (print them again at the very end of prepare, after master's
  [align2] lines). Print per-phase prepare seconds and the total.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "C3: halves measured +1% alone at T=32; with split-K w13 the halved grids refill, target decode_32 <= 199 us (+2%)"
  - shapes: bs=8 *
    direction: faster
    why: "C3: halves lost 1.5% here alone (underfilled half grids); H+A is the test, target verify_1 <= 109 us (+2%), else the band keeps master"
  - shapes: bs=16 *
    direction: faster
    why: "C3: as bs=8; target decode_16 <= 143 us (+2%), else master"
  - shapes: bs=1 *
    direction: faster
    why: "F1: decode_1 +2.5-3%"
  - shapes: bs=256 *
    direction: faster
    why: "F2: verify_32 -1.5-2%"
  - shapes: bs=128 *
    direction: flat
    why: "control: Marlin"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T162149Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T162149Z
base_sha: 405d804d6756dfee33cc29cbda79b8057d85fbe1
base_blob: 593b238c16387c5ac3774424ee9557882b9fc46a
branch: hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk
worktree: .worktrees/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk
mode: null
restack_of: null
stale_base: null
---

# combo-smallm-two-stream-halves-x-w13-splitk

## Change

COMBINATION (conjectural pairing, small-M Marlin side): two-stream expert halves paired with w13 split-K atomic add (w2 atomic is already on master) at 8 <= T <= 32, on top of the shared floor. Built on master a0e52640. SHARED FLOOR (identical in all three combo hypotheses minted 2026-10-25 ~16Z, so its rows are measured three times): (F1) decode_1: w13 Marlin use_atomic_add=True at num_tokens == 1 only. Source w13-marlin-atomic-add-t1-only, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only, impl commit f29a0b9e: port W13_ATOMIC_MAX_TOKENS = 1, W13_ATOMIC_TOL, _w13_band, _w13_atomic_self_check and the w13_mode kwarg in the w13 moe_wna16_marlin_gemm call of _moe_marlin (w2 keeps master's w2_mode). Measured twice: decode_1 1.766x (job 171684977ec1470f) and 1.754x (job 1048c6edf3e54892) against master-equivalent 1.698-1.721x in eight runs (+2.5-3.0%; Marlin pair 33.7 vs 35.9 us). Gate: its own self-check (rel-L2 vs master <= 1e-2; it measured 8.4e-3) plus a component graph (the w13 Marlin call alone at T=1, skewed and random local routing) <= 0.98x. Phase F1 start deadline 120 s, ~3 s. (F2) verify_32: master's OWN align2 win is dead code on the judged path. Master's _align2_setup gates class v256 for _moe_marlin (job c05a8f7326004be6: align2 chunk 64 4.70 us vs incumbent 6.46 us, ratio 1.374, ON), but kernel_fn sends 256 <= T <= 512 to _moe_wgmma (wg_on_256 True), which still runs bank expert_map[topk_ids.to(int32)] plus SGLang moe_align_block_size(local_ids, WG_BSM=16, count): the c05a8f73 verify_32 profile shows index_elementwise 3.4 + unrolled_elementwise 3.1 + moe_align_block_size_kernel 2.8 + count_and_sort_expert_tokens 2.5 = 11.8 us in four launches and no _align2 kernel. Source: master a0e52640's own _align_two_phase / _align2_count_kernel / _align2_place_kernel (port origin align-two-phase-port-from17, impl commit fb879b90; prefill origin align-two-phase-all-t-rega). Change: in _moe_wgmma only, when state['wg_align2'] (a plain int chunk, 0 = master) is set, replace the expert_map gather and moe_align_block_size with local_ids, (sorted_ids, block_experts, n_padded) = _align_two_phase(topk_ids, start, count, WG_BSM, chunk). Gate F2: warm every T in 128..512 at block_size_m 16 for each chunk in (64, 128, 256, 512) <= the class top; equivalence on T in (128, 191, 192, 256, 257, 384, 512) x three local_starts x six routings (_align2_routings): local_ids torch.equal, n_padded equal, sorted_ids[:n_padded] and block_experts[:n_padded/16] torch.equal to master's front end (capacity tails may differ: the wg kernels read below n_padded only; assert n_padded/16 <= WG_MAX_BLOCKS), then full _moe_wgmma output torch.equal (bitwise: same kernels, same tables) on two of those routings per T; then the component graph (front end alone: gather+cast+align vs align2) <= 0.98x at T=256 (expected ~0.4-0.5x). Applies to whichever wgmma bucket is ON (B256 today; B128 only if its flag is on). Phase F2 start deadline 150 s, ~15-20 s (Triton constexpr warm per T is most of it). (C3a) Two-stream halves. Source marlin-two-stream-expert-halves-small-m, branch hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m, impl commit ea345883: port _align_small_halves_kernel (grid 2, one program per expert half, its own sorted_ids_h / block_experts_h / n_padded_h / local_ids_h, HALF_SORTED_ALIGN), _experts_two_stream (side.wait_stream(current); half A on the current stream, half B under torch.cuda.stream(side); w13 Marlin, silu_and_mul with expert_ids=local_ids_h, w2 Marlin into the shared cache3; current.wait_stream(side); master's post-norm on the full local_ids), the second Marlin lock workspace and the side stream, both created in prepare and held in the state. Measured: decode_32 1.042x (job 5d630369b1634f77, best decode_32 row of any run) but verify_1 1.316x vs 1.33-1.35x, decode_16 flat; that run predates master's w2 atomic add. (C3b) w13 atomic at T <= 32. Source w13-marlin-atomic-add-on-master-small-m, branch hyp/muse_moe_w4a16/w13-marlin-atomic-add-on-master-small-m, impl commit c3a84420 (W13_ATOMIC_MAX_TOKENS 32, _w13_band, _w13_atomic_self_check): flat on its own at T = 8/16/32 (1.353/1.281/1.035x), +2.6% at T=1 (that row is F1). In the halves path both Marlin calls of each half take the w13 kwargs of the chosen mode and w2 keeps master's w2_mode (each half writes disjoint route rows of cache1/cache3, so atomic modes stay row-disjoint; if w2_mode is atomic_zero, the zeroing happens once on the current stream before the fork). GATE per T band, bands chosen by shape only: [8, 12] rep T=8 (verify_1 and the bs=4-adjacent range), [13, 24] rep T=16, [25, 32] rep T=32; candidates {master, H = halves, H+A = halves with w13 atomic, A = w13 atomic alone}. Each candidate: halves self-check (per-half routes and block counts equal to moe_align_block_size with the other half set to -1, as the source did), rel-L2 vs master <= 1e-2 and finite on six routings, graph capture with the fork/join (capturable; the source ran it on the judge), then full kernel_fn cold-L2 graphs, 41 interleaved reps, judge-like skewed and random routings; the band takes the lowest median only at <= 0.98x master on the worse routing. Print PATH CHOSEN per band with all four medians. state['sm_mode'] = dict band -> int set in prepare; kernel_fn picks by num_tokens band only. ROUTING SUMMARY: T == 1 -> F1; 8 <= T <= 32 -> per band master / H / H+A / A; 256 <= T <= 512 -> wgmma with F2; everything else master. Expected prepare: master ~50 s + F1 3 s + F2 15-20 s + C3 compile and warm of the halves kernel for T 8..32 (~10 s, start deadline 150 s) + self-checks ~5 s + gate 3 bands x 4 candidates x 2 routings (~6 s, start deadline 230 s) = ~95 s of 300. RULES (every combo phase): kernel_fn branches on shape and prepare-set plain ints/bools only; no host sync, no input-value-sized allocation, no data-dependent control flow, no prints, no try/except, no per-call mutable state, no timing, nothing cached from inputs; NVFP4 weights stay NVFP4 (register dequant only), BF16 activations where SGLang keeps them, fp32 accumulation; every local route computed; no child processes. The judge prepares TWO states and copies them by cloning tensors: every scratch tensor, stream, workspace and error word lives in the state, no module-global device pointer, tensor map or tensor, and tensor maps stay encoded per call (master's s5 rule). PREPARE ORDER: master's prepare() runs first and unchanged (align self-check, _align2_setup, _w2_atomic_self_check, _wg_setup with its WG_SETUP_BUDGET_S 200 / WG_REPLAY_DEADLINE_S 235 / WG_TIMING_DEADLINE_S 255 untouched); each combo phase below runs AFTER it, measures its own elapsed time from prepare()'s t_prepare, skips itself (flag stays master, prints PATH CHOSEN = master (deadline)) if it would start past its start deadline, and no phase starts timing past 270 s. GATE RULE (every source): exactness or tolerance check first, then cold-L2 CUDA-graph timing vs master's path for that class (>= 256 MB flush before each replay, interleaved A/B, >= 21 reps, empty-graph baseline subtracted, median); ON only at new/master <= 0.98 on the worse of the judge-like skewed routing and random routing; otherwise that class keeps master. Print one PATH CHOSEN line per source per class with both medians and the ratio, and keep those lines in the last 16k characters of the log (print them again at the very end of prepare, after master's [align2] lines). Print per-phase prepare seconds and the total.

## Expected

- `bs=32 *` faster: C3: halves measured +1% alone at T=32; with split-K w13 the halved grids refill, target decode_32 <= 199 us (+2%)
- `bs=8 *` faster: C3: halves lost 1.5% here alone (underfilled half grids); H+A is the test, target verify_1 <= 109 us (+2%), else the band keeps master
- `bs=16 *` faster: C3: as bs=8; target decode_16 <= 143 us (+2%), else master
- `bs=1 *` faster: F1: decode_1 +2.5-3%
- `bs=256 *` faster: F2: verify_32 -1.5-2%
- `bs=128 *` flat: control: Marlin

## Risk

Medium risk: two measured ports whose failure modes are known (the halves ran on the judge, 5/5 tests; w13 atomic passed its self-check at T <= 32 with rel-L2 under 1e-2); the new element is only their pairing and the per-band gate. Error budget: w13 atomic adds ~8e-3 rel-L2 against master at T=1 and the check bound is 1.5% global, so the gate must reject any band whose candidate exceeds 1e-2 vs master.

## Conjecture

Splitting the experts into two concurrent halves halves each Marlin call's m-block count, so each call's grid under-fills the 132 SMs; that is why the halves lost at T=8 and were flat at T=16 and won only at T=32, where each half still has enough blocks. Marlin's atomic-add mode raises per-call parallelism by splitting K across more CTAs, which was useless on the full grid (flat at 8/16/32) because the grid was already full. The pairing predicts the opposite of either alone: H+A should beat both H and A at T=8 and 16, because split-K refills exactly the SMs that halving emptied, and the two streams hide the atomic path's reduction tail behind the other half's mainloop. A second interaction: the halves were measured before master's w2 atomic add, so H alone may now behave differently at T=32. Confirm with the per-band PATH CHOSEN lines (H+A chosen at the 8 and 16 bands, with medians H+A < min(H, A) by >= 2%) and the bs=8/16 profiles (four Marlin rows, two act_and_mul rows, _align_small_halves_kernel); refute if A and H+A time the same as master at T=8/16 (the small-M Marlin pair is bandwidth-bound, not grid-bound) or H+A loses to H at T=32.

## Adversarial review

Reviewed 2026-10-25T162149Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion: the per-band four-way gate (master / H / H+A / A) with printed medians makes the conjecture confirmable or refutable inside this one run, and the shared floor F1/F2 rows are attributable by shape. Before it ships, fix two correctness gaps. (1) The H+A self-check must poison its buffers. (2) A deterministic cross-half test must show that w2's atomic-add Marlin call does not clobber the other half's rows of the shared cache3; otherwise the halves path keeps w2 non-atomic. Every zero_() must stay inside kernel_fn.

### Justification
A. The floor parts are well grounded. F1 (w13 atomic at T==1) was measured twice at +2.5-3.0% on decode_1, beyond the run-to-run spread of 1.698-1.721x. F2 is a real dead-code finding. Master's kernel_fn sends 256<=T<=512 to _moe_wgmma whenever wg_on_256 is set, and _moe_wgmma still runs expert_map[topk_ids.to(int32)] plus moe_align_block_size at WG_BSM=16, so master's align2 v256 win (4.70 vs 6.46 us, chosen in c05a8f73) never reaches the judged verify_32 path. Replacing about 11.8 us of front end in four launches with two launches is worth about 1-1.5% of verify_32's 405 us, which matches the stated expectation. The F2 gate is strong: torch.equal on the tables below n_padded, n_padded/16 <= WG_MAX_BLOCKS, and a bitwise _moe_wgmma output check. The C3 conjecture (H+A beats both H and A at T=8/16) is plausible but argued from the wrong mechanism, and that is the WEAKEST LINK. SGLang/vLLM's Marlin MoE kernel sizes its grid from the SM count (persistent, stream-K-like over m-blocks x n-tiles x k-slices, with lock workspace max_blocks_per_sm=4), not from the work. Halving the experts therefore does not leave SMs idle inside a call. It deepens per-column K slicing, and the two concurrent full-grid kernels compete for SMs. use_atomic_add adds no CTAs and does not 'refill' anything. It replaces the lock-ordered fp32 global reduce with atomics. A mechanism that survives is this: halving deepens the lock-serialized reduce chain, and atomics shorten it. That predicts the same sign (H+A < H at T=8/16), so the run can still confirm or refute, but a win should not be read as 'grid refill'. Priors are weak. A alone was flat at 8/16/32 (1.353/1.281/1.035x). H alone was -1.5%, flat and +1% at 8/16/32 before w2 atomic landed. The small-M Marlin pair runs at about 35% of the bandwidth floor (105 us vs 37 us at bs=8), so the gap is real but not clearly reduce-bound. Expected directions: bs=1 faster (F1) and bs=256 faster (F2) are justified. bs=32 faster is plausible at about 1%. bs=8 and bs=16 'faster' should read 'faster, or flat because the band keeps master'; flat is the likeliest outcome. bs=128 flat is a valid control (Marlin, align class v128 kept as master). The refutation criteria in the Conjecture are readable from the PATH CHOSEN medians. The criterion '>=2% H+A < min(H, A)' applies to prepare-time medians, which is the right instrument. Prepare budget is fine: c05a8f73 had prepare at 48.7 s after the wg setup, so the 120/150/230 s start deadlines leave ample margin.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m: the halves were measured BEFORE this landed. In the halves path both halves now run w2 with master's w2_mode='atomic' (no memset; c05a8f73 log: 'PATH CHOSEN = atomic') on ONE shared cache3 from two streams. The no-memset mode passed only on the aliased full path. If Marlin's moe wrapper or kernel zeroes or initialises C beyond the rows of its own routes, half B wipes half A's rows, which is a race and not a timing effect. Any change in H vs its source measurement can also come from this interaction rather than from the halving itself. The four-way in-run gate isolates H vs H+A vs A on the current w2 mode, so attribution survives, but correctness needs the cross-half test below.
- align-two-phase-port-from17: F2 reuses its _align_two_phase kernels at a new block_size_m (16) and in a new caller (_moe_wgmma). Its own v128 class LOST at bsm=16 (0.781x), so F2's per-T warm and the T=256 component gate are mandatory, not optional. The halves path replaces master's 8..32 align (_align_small, d32 class = master) with _align_small_halves_kernel, so part of any H delta (a few us) is the align swap, not the Marlin concurrency.
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate: F2 only acts if this run's selector turns wg_on_256 ON, and its margin is thin (1.047/1.055x vs a 1.02 threshold). If B256 flips to Marlin in this run, F2 is inert, verify_32 runs master's Marlin path with align2 v256, and a verify_32 move would come from the selector, not F2. Read the B256 PATH CHOSEN line before attributing the bs=256 row.
- post-expert-norm-triton-small-m: at bs<32 it reads y = cache3. In the halves path cache3 is a separate torch.empty buffer (not aliased to cache13), so remote-route rows hold allocator junk, possibly NaN bit patterns from graph-pool reuse. The combine kernel must mask by local_ids for these rows. The source passed 5/5 so it likely does, but the poisoned self-check should cover it.
- triton-map-align-on-block-m-restack / marlin-block-m picks: each half reuses the full-T block_size_m. With about half the routes per half, a smaller bsm could suit better, so a flat H/H+A result at T=8/16 is conditional on this pick. This is not a reason for a companion now.

### Interference and A/B needs
No companion is warranted. The hypothesis already carries its own in-run A/B. For each band it prints the medians of master, H, H+A and A (41 interleaved cold-L2 graph reps on two routings) and gates at <=0.98x on the worse routing. That answers the conjecture directly: H+A < min(H, A) by >=2% at T=8/16 confirms it, and A ~ H+A ~ master refutes it. It also does not depend on reading the scored headline. The floor parts F1 (bs=1) and F2 (bs=256) are disjoint by shape from C3 (T 8..32), and their own PATH CHOSEN lines say whether they fired. An ab-revert of w2-marlin-atomic-add-small-m (halves with w2 non-atomic) would be informative only if H+A unexpectedly loses. Even then it would be the fallback configuration the cross-half test may force anyway, so it should be a follow-up, not a second scored run. Caveat on reading the result: picking the lowest of three noisy candidates at 0.98 gives a winner's-curse bias. Treat a band chosen at 0.97-0.98 as weak, and confirm it with the judged bs=8/16/32 rows, whose own CI is about ±1.5%.

### Open questions
- Does SGLang's moe_wna16_marlin_gemm with use_atomic_add=True initialise C itself, by an internal full-tensor zero_() or by having slice 0 store plainly? Master's w2 'atomic' no-memset mode passing on an aliased, non-zero cache3 proves only that the result is correct on the full path. The cross-half test above decides whether that holds when two calls share one C.
- With both halves' Marlin kernels launched at a grid sized from the SM count on separate streams, are they actually co-resident, or does the hardware run them mostly back to back? If an nsys or profile timeline is available from the source run (job 5d630369b1634f77), the overlap fraction would say whether H's +1% at T=32 came from concurrency or from the align swap.
- Expected 'faster' at bs=8 and bs=16 should be read as 'faster or flat (band keeps master)'. Given A flat and H -1.5%/flat, master is the most likely PATH CHOSEN for both bands, and that outcome still refutes the conjecture cleanly.
- The 'four Marlin rows' profile readout is not available: the profile aggregates every Marlin launch into one row by kernel name. Use the PATH CHOSEN medians and launch counts instead.

