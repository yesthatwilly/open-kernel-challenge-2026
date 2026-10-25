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
review: required
status: queued
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
