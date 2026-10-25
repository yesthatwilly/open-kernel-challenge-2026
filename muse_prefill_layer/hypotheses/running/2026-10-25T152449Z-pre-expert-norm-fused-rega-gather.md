---
problem: muse_prefill_layer
slug: pre-expert-norm-fused-rega-gather
created: 2026-10-25T152449Z
base: master
area: pre-expert norm (MuseSpark1xRMSNorm) fused into the rega row gather
change: >
  Built on current master (submission blob a2fe2a75, align-two-phase-all-t-rega landed; no
  depends_on). On the rega expert path only (_experts dispatches to _experts_rega: 200 < T <=
  16384 when state['rega_ok'][class] is ON, which is every scored shape: turn_1 T=753, turn_4
  T=3012, first_turn T=16384), fuse the pre-expert RMSNorm into the row gather. Today kernel_fn
  runs x = state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm over D_EXPERT=4096, eps
  RMS_EPS, weight applied as 1 + w; profiled as at::native vectorized_layer_norm: 76.5-78.5 us at
  T=16384, 18.0 us at T=3012) and _experts_rega then calls ext.rega_gather(x, sorted_ids,
  block_experts, n_padded, num_slots, LOCAL_EXPERTS, xs) (rega_gather_kernel: one 256-thread CTA
  per slot over the whole sorted_ids capacity, most CTAs exit early; 168.9-183.8 us at T=16384,
  27.6-28.0 us at T=3012, 6.8-7.1 us at T=753; it moves ~272 MB at T=16384, a 57 us floor at 4.8
  TB/s, so it runs at ~1.5 TB/s). Replace both with ONE Triton kernel _norm_gather_kernel(x_raw,
  w_eff, sorted_ids, block_experts, n_padded, num_routes, xs, eps, H=4096, BLOCK=4096,
  num_warps=8): a persistent grid of min(cdiv(capacity, 1), 132 * 8) programs; each program loads
  n_padded once from the device and loops slot = pid, pid + nprog, ... < n_padded (a device-side
  loop bound, not host control flow); for each slot: be = block_experts[slot // REGA_BSM]; if be <
  0 or be >= LOCAL_EXPERTS skip the slot entirely (rega never reads remote blocks; this keeps
  rega_gather's exact write set); sid = sorted_ids[slot]; if 0 <= sid < num_routes: row =
  x_raw[sid >> 3] (4096 bf16), ss = sum(row.f32 ** 2), rstd = 1 / sqrt(ss / 4096 + eps), xs[slot]
  = (row.f32 * rstd * w_eff.f32).to(bf16); else xs[slot] = 0. w_eff is the effective (1 + w)
  weight the module multiplies by: read it in prepare() from state['pre_expert_norm'] (its
  precomputed effective weight, as prepare already does for post_expert via
  _get_effective_weight()), keep its dtype, store as state['pre_expert_w_eff'] (contiguous; a
  weight, not input-derived). Before writing, print the module's forward source once in prepare
  (inspect.getsource, read-only) so the reviewer can see the formula order; match it (fp32 math,
  rstd from the mean of squares + eps, multiply by x then by w, one bf16 rounding). kernel_fn:
  compute expert_input = pooler GEMM as today; if _experts would take the rega path for this T AND
  state['norm_gather_ok'][class] (class = 'low' T <= REGA_LOW_MAX, 'high' T <= MARLIN_MAX_TOKENS,
  'ft' above; shape-only branch), call _experts_rega(expert_input, ..., prenorm=True), which
  launches _norm_gather_kernel in place of ext.rega_gather and does not call the norm; otherwise
  run master's exact code (norm, then _experts). Marlin / ssdq / Triton paths, the align, both
  rega_gemm calls, the weighted sum, every other kernel unchanged. PREPARE GATE (after
  _rega_setup, so it measures what _experts really dispatches): per class at T = 753, 3012, 16384
  on the existing synthetic rega cases (_rega_case 'soft' and 'skewed'), (1) self-check: xs from
  the fused kernel vs master's composed path (module norm, then ext.rega_gather) on the same
  sorted_ids/block_experts/n_padded: identical write set (compare the gathered rows of local
  blocks, and verify by a canary fill of xs that remote-block and past-n_padded rows are
  untouched), max difference <= 1 bf16 ulp on >= 99.9% of elements and <= 2 ulp everywhere, rel L2
  <= 1e-3; then the whole _experts_rega output with prenorm=True vs master's at rel L2 <= 1e-3,
  alternating two separately allocated states (the judge prepares 2 copies: nothing module-global,
  the weight lives in state); (2) timing: the pair (norm + rega_gather) vs the fused kernel, each
  captured as a CUDA graph, interleaved over 21 alternations, each replay preceded by a 256 MiB
  flush write, empty-graph baseline subtracted, medians printed; ON only if fused <= 0.90x the
  pair. Print '[norm-gather] GATE <class> T=<T> pair <us> fused <us> ratio <r>: ON/OFF' and a
  final '[norm-gather] PATH CHOSEN' line after rega's PATH CHOSEN lines. Any exception, mismatch
  or a slow ratio leaves the class OFF = master's path byte-identical. Expected prepare cost < 10
  s (3 Triton compiles of one kernel are shape-agnostic: do_not_specialize num_routes; capacity
  only sizes the grid).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "removes vectorized_layer_norm (76.5-78.5 us) and replaces rega_gather (169-184 us at ~1.5 TB/s) with one persistent fused pass of ~272 MB (~65-80 us at 3.5-4 TB/s): -150 to -190 us of 11120-18350 us (sliding first_turn ~1.4-1.7%, full rows 0.8-1.3%)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "norm 18.0 us + gather 27.6-28.0 us -> ~12 us fused: -30 to -35 us of 2151 (sliding) / 3714 (full), 0.8-1.6%"
  - shapes: "*load=turn_1*"
    direction: flat
    why: "norm ~5 us + gather ~7 us -> ~4 us: -5 to -8 us of 747/1137 (0.5-1%), likely inside run noise"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T153001Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T153001Z
base_sha: e15152e9078a60b4c86e9eaf4e8b71da18b05e66
base_blob: a2fe2a75893c60d4f68cf94fcce8ce8f9b708a71
branch: hyp/muse_prefill_layer/pre-expert-norm-fused-rega-gather
worktree: .worktrees/muse_prefill_layer/pre-expert-norm-fused-rega-gather
mode: null
restack_of: null
stale_base: null
---

# pre-expert-norm-fused-rega-gather

## Change

Built on current master (submission blob a2fe2a75, align-two-phase-all-t-rega landed; no depends_on). On the rega expert path only (_experts dispatches to _experts_rega: 200 < T <= 16384 when state['rega_ok'][class] is ON, which is every scored shape: turn_1 T=753, turn_4 T=3012, first_turn T=16384), fuse the pre-expert RMSNorm into the row gather. Today kernel_fn runs x = state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm over D_EXPERT=4096, eps RMS_EPS, weight applied as 1 + w; profiled as at::native vectorized_layer_norm: 76.5-78.5 us at T=16384, 18.0 us at T=3012) and _experts_rega then calls ext.rega_gather(x, sorted_ids, block_experts, n_padded, num_slots, LOCAL_EXPERTS, xs) (rega_gather_kernel: one 256-thread CTA per slot over the whole sorted_ids capacity, most CTAs exit early; 168.9-183.8 us at T=16384, 27.6-28.0 us at T=3012, 6.8-7.1 us at T=753; it moves ~272 MB at T=16384, a 57 us floor at 4.8 TB/s, so it runs at ~1.5 TB/s). Replace both with ONE Triton kernel _norm_gather_kernel(x_raw, w_eff, sorted_ids, block_experts, n_padded, num_routes, xs, eps, H=4096, BLOCK=4096, num_warps=8): a persistent grid of min(cdiv(capacity, 1), 132 * 8) programs; each program loads n_padded once from the device and loops slot = pid, pid + nprog, ... < n_padded (a device-side loop bound, not host control flow); for each slot: be = block_experts[slot // REGA_BSM]; if be < 0 or be >= LOCAL_EXPERTS skip the slot entirely (rega never reads remote blocks; this keeps rega_gather's exact write set); sid = sorted_ids[slot]; if 0 <= sid < num_routes: row = x_raw[sid >> 3] (4096 bf16), ss = sum(row.f32 ** 2), rstd = 1 / sqrt(ss / 4096 + eps), xs[slot] = (row.f32 * rstd * w_eff.f32).to(bf16); else xs[slot] = 0. w_eff is the effective (1 + w) weight the module multiplies by: read it in prepare() from state['pre_expert_norm'] (its precomputed effective weight, as prepare already does for post_expert via _get_effective_weight()), keep its dtype, store as state['pre_expert_w_eff'] (contiguous; a weight, not input-derived). Before writing, print the module's forward source once in prepare (inspect.getsource, read-only) so the reviewer can see the formula order; match it (fp32 math, rstd from the mean of squares + eps, multiply by x then by w, one bf16 rounding). kernel_fn: compute expert_input = pooler GEMM as today; if _experts would take the rega path for this T AND state['norm_gather_ok'][class] (class = 'low' T <= REGA_LOW_MAX, 'high' T <= MARLIN_MAX_TOKENS, 'ft' above; shape-only branch), call _experts_rega(expert_input, ..., prenorm=True), which launches _norm_gather_kernel in place of ext.rega_gather and does not call the norm; otherwise run master's exact code (norm, then _experts). Marlin / ssdq / Triton paths, the align, both rega_gemm calls, the weighted sum, every other kernel unchanged. PREPARE GATE (after _rega_setup, so it measures what _experts really dispatches): per class at T = 753, 3012, 16384 on the existing synthetic rega cases (_rega_case 'soft' and 'skewed'), (1) self-check: xs from the fused kernel vs master's composed path (module norm, then ext.rega_gather) on the same sorted_ids/block_experts/n_padded: identical write set (compare the gathered rows of local blocks, and verify by a canary fill of xs that remote-block and past-n_padded rows are untouched), max difference <= 1 bf16 ulp on >= 99.9% of elements and <= 2 ulp everywhere, rel L2 <= 1e-3; then the whole _experts_rega output with prenorm=True vs master's at rel L2 <= 1e-3, alternating two separately allocated states (the judge prepares 2 copies: nothing module-global, the weight lives in state); (2) timing: the pair (norm + rega_gather) vs the fused kernel, each captured as a CUDA graph, interleaved over 21 alternations, each replay preceded by a 256 MiB flush write, empty-graph baseline subtracted, medians printed; ON only if fused <= 0.90x the pair. Print '[norm-gather] GATE <class> T=<T> pair <us> fused <us> ratio <r>: ON/OFF' and a final '[norm-gather] PATH CHOSEN' line after rega's PATH CHOSEN lines. Any exception, mismatch or a slow ratio leaves the class OFF = master's path byte-identical. Expected prepare cost < 10 s (3 Triton compiles of one kernel are shape-agnostic: do_not_specialize num_routes; capacity only sizes the grid).

## Expected

- `*load=first_turn*` faster: removes vectorized_layer_norm (76.5-78.5 us) and replaces rega_gather (169-184 us at ~1.5 TB/s) with one persistent fused pass of ~272 MB (~65-80 us at 3.5-4 TB/s): -150 to -190 us of 11120-18350 us (sliding first_turn ~1.4-1.7%, full rows 0.8-1.3%)
- `*load=turn_4*` faster: norm 18.0 us + gather 27.6-28.0 us -> ~12 us fused: -30 to -35 us of 2151 (sliding) / 3714 (full), 0.8-1.6%
- `*load=turn_1*` flat: norm ~5 us + gather ~7 us -> ~4 us: -5 to -8 us of 747/1137 (0.5-1%), likely inside run noise

## Risk

Area overlap: the gather is part of the rega expert path, so an experts-front agent may also touch rega_gather; this hypothesis changes only the launch that feeds rega's xs and keeps ext.rega_gather for every OFF class. Risk low: Triton, shape-only branch, no new CUDA build, bit-near parity by construction, prepare gate with self-check + interleaved flushed graph timing (the selector style that predicted the judge in restack-unpooler-slice-cluster-dsmem-m8). Read the result on the replaced kernels' own rows (vectorized_layer_norm + rega_gather vs _norm_gather_kernel), not the headline: the headline gain is ~1%, at the edge of resolution. If the module's forward is not F.rms_norm-shaped (the getsource print shows otherwise), the 1-ulp bound will fail and the gate keeps master: read the self-check line before concluding.

## Adversarial review

Reviewed 2026-10-25T153001Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. The saving is real but is mostly not where the hypothesis puts it. rega_gather's 169-184 us at first_turn is dominated by ~115k dead CTAs (capacity 8T+480 slots, n_padded ~16.6k), not by a 1.5 TB/s copy, so a persistent fused kernel should save ~150-180 us/ft shape and ~30 us/turn_4. The single run reads cleanly on the three profile rows. Its one real performance risk is the fixed 132*8 persistent grid exceeding the resident CTA count.

### Justification
Profile data on master (bff8e36788a04917.profile.txt) confirms the rows. rega_gather_kernel is 180.6/183.8/180.4/168.9 us on the four first_turn shapes, 28.0/27.6 at turn_4 and 7.1/6.8 at turn_1. at::native vectorized_layer_norm is 76.5-78.5 us at ft and 18.0 at sliding turn_4; at turn_1 it is outside the top list. The mechanism is better than stated. Master's own prepare log shows rega_gather at T=8192 'local' (65536 active slots, ~1.07 GB r+w) taking 276 us eager, about 3.9 TB/s, while T=16384 random (16368 active slots, 268 MB) takes 166 us. The copy itself is therefore near bandwidth, and ~100-110 us of the ft row is the launch/retire cost of ~115k CTAs that only read n_padded and exit (one 256-thread CTA per capacity slot). A persistent grid bounded by the device-side n_padded removes that cost. Folding the norm in also removes a full 268 MB round trip (the 77 us norm row at ~3.5 TB/s), because the fused kernel moves the same ~268 MB the gather already moves: T local-route rows read plus written. A target of ~70-90 us fused vs ~255 us pair at ft (ratio ~0.3) and ~10-15 us vs ~46 us at turn_4 is physically reachable. The 0.90x gate should pass at ft and turn_4, and may be marginal at turn_1 (L2-resident, latency-bound). The expected directions match TASK weights: ft rows -1 to -1.6%, turn_4 ~-1%, turn_1 flat. The headline (~+1%) sits inside the +/-1.5% CI, so the result must be read on rows: vectorized_layer_norm + rega_gather_kernel vs _norm_gather_kernel per shape, plus the prepare GATE lines. WEAKEST LINK: the fixed persistent grid of 132*8 = 1056 programs at num_warps=8 assumes 8 resident 256-thread CTAs per SM, which needs <= 32 registers/thread. A BLOCK=4096 row (16 fp32 per thread), w_eff (another 16 unless re-loaded), the reduction and the address math will likely compile to 40-64 regs, giving 4-6 resident CTAs/SM. Then ~25-50% of the programs start only after the first resident set finishes its whole slot loop, a 1.5-2x tail on a kernel whose whole point is bandwidth. Size the grid from the compiled kernel's n_regs/occupancy (132 * resident CTAs/SM), or sweep the grid multiplier {2,4,8} and num_warps {4,8} inside the existing prepare gate and print each. Secondary: each loop iteration has a dependent scalar chain (block_experts -> sorted_ids -> row load), so per-program memory parallelism is one 8 KB row. That is fine only if residency is high, which is another reason to size the grid by occupancy, not by a constant.

### Confounders on the current kernel
- align-two-phase-all-t-rega (landed, master): it produces sorted_ids/block_experts/n_padded with local-only blocks at BSM 16, which is why n_padded ~ T+pad while capacity stays 8T+480. That ratio creates the dead-CTA cost this change removes. If a later align change shrank sorted_ids capacity to n_padded, rega_gather would get cheaper on its own and this hypothesis's margin would shrink. This is not a confound for this run, because the fused kernel consumes exactly the same align outputs.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128 (landed): they define rega_gather and the xs contract (zeros at sentinel slots of local blocks, untouched elsewhere). The fused kernel must keep that write set, which the canary check covers. They are the base, not an interaction.
- sglang-pipeline-parity (landed): it defines state['pre_expert_norm'] as MuseSpark1xRMSNorm(1+w). The fused kernel's numerics depend on matching that module's formula order and weight dtype, which the getsource print plus the ulp/relL2 gate checks. Matching is a correctness question, not an A/B one.
- stack-tonight-five-prefill (landed: router hilo, align, w2 zero-store, Marlin silu): disjoint kernels. zero_nonlocal_w2 only changes y's allocation in _experts_rega, not xs, so there is no interaction.

### Interference and A/B needs
No companion is warranted. Every replaced kernel has its own row in the archived profile (rega_gather_kernel at all shapes, vectorized_layer_norm at ft and turn_4), and the new _norm_gather_kernel will show as its own row, so a gain or a loss is attributable from this single scored run without a revert. The prepare GATE lines carry an in-process cold-L2 graph A/B per class, and a slow result turns the class OFF and runs master's path byte-identical. One sub-change could in principle be isolated: a persistent grid alone (dead-CTA removal) vs norm fusion. That split is a design question for a follow-up and is not needed to interpret this result. Spending a scored run on it is not justified. Collisions with in-flight work: (a) experts-rega-pingpong-consumers-all-t and experts-rega-producer-scale-expand-all-t both edit _experts_rega, but only the two rega_gemm call sites (rega_gemm -> rega_gemm_pp/sx under a per-class flag). This change edits only the rega_gather call and adds a prenorm argument. The functions are independent at run time (the GEMMs read xs, whatever produced it), and a later stack needs a small hand merge in _experts_rega and at prepare's tail, where all three append post-rega phases and PATH CHOSEN lines. (b) fa3-full-contiguous-kv-triton-gather-gated touches only the attention section of kernel_fn, so its hunk is disjoint from the pre-expert norm call site. (c) Deadline: master's prepare totals 124.4 s with the ft phase starting at ~119.9 s, and the new gate runs after _rega_setup returns. It cannot move the REGA_FT_START_DEADLINE_S (150 s) check, REGA_BUDGET_S (200 s) or REGA_COPY_DEADLINE_S (255 s) of this build. It adds ~5-20 s to the total (3 Triton compiles plus 21 x 2 graph replays x 3 classes plus self-checks), well under the 300 s prepare limit. When stacked with the v3 rega builds, its phase must stay after _rega_setup, so it also cannot push their ft start; add a self-imposed skip (e.g. if prepare already used > 270 s, leave every class OFF) so a slow v3 stack cannot push prepare past 300 s.

### Open questions
- What do the compiled kernel's n_regs and CTAs/SM come out to at num_warps=8, BLOCK=4096? If fewer than 8 CTAs/SM, the 132*8 grid has a tail. Print it and size the grid by occupancy.
- _rega_case has no 'soft' kind: it falls through to _ssdq_case's default (random routing), which is what master's GATE 'ft T=16384 soft' line already uses. Use 'random' (and 'skewed') explicitly so the log is not misleading.
- Is _get_effective_weight() for MuseSpark1xRMSNorm bf16 or fp32, and does the forward compute (x*rstd)*w or x*(rstd*w)? The getsource print answers both. Record it in the attempt file before reading the ulp line.
- At turn_1 (T=753, ~1.2k active slots of ~6.5k capacity) the pair is ~12 us and L2-resident. Expect the 0.90x gate to pass narrowly or fail. Either outcome is fine, but read turn_1 as 'flat by design'.
- Follow-up if this lands: the same dead-CTA pattern may exist in other one-CTA-per-capacity-slot kernels on the rega path (e.g. any scatter or zero-fill keyed by num_slots). Worth a grep before minting the next row-level hypothesis.

