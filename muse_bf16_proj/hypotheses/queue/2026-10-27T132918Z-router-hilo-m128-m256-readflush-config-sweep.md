---
problem: muse_bf16_proj
slug: router-hilo-m128-m256-readflush-config-sweep
created: 2026-10-27T132918Z
base: master
area: router GEMM (m=65..256, hilo launch config chosen per m by a read-flush sweep)
change: >
  Built on master (8f233e88 lineage). At m > 32 the router runs torch.zeros + _router_hilo_kernel
  with the static _router_hilo_config: (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m <= 128 -> grid
  2x4x16, and (64, 64, 8) at m=256 -> grid 4x4x8, BLOCK_K 128, 3 stages, 4 warps (profile
  a6424247: hilo 5.0 us at m=128 and 6.9 us at m=256, rows 7.95 / 9.51 against floors 2.21 / 2.68;
  4 / 8 K-steps per CTA, so most of each CTA is pipeline fill, plus 0.52M relaxed fp32 atomics).
  Only one alternative config was ever timed (the tall tile), with one write-flush round. CHANGE:
  make the hilo launch config a per-m state entry. _router_hilo_forward takes (BLOCK_M, BLOCK_N,
  BLOCK_K, SPLIT_K, num_warps, num_stages) from state['router_hilo_cfg'][m_class] when
  state['router_hilo_cfg_ok_<m_class>'] is set (m_class 128 for 33..128, 256 for 129..256;
  cfg/shape-only branching), else today's _router_hilo_config; the kernel body is unchanged (same
  hi+lo fp32 accumulate and atomic_add, only tiling and split order differ). Prepare, after
  _router_prepare, state A: for m in (128, 256) enumerate BLOCK_M {64, 128}, BLOCK_N {32, 64,
  128}, BLOCK_K {64, 128}, SPLIT_K {4, 8, 16, 32} with 8192 % (BLOCK_K * SPLIT_K) == 0 and a grid
  of 64..264 CTAs, num_warps 4 (8 when BLOCK_M 128 and BLOCK_N 128), num_stages {2, 3, 4} that
  compile without OutOfResources (catch it, as _router_prepare does); print per compiled config
  its grid, n_regs/spills, the modeled L2->SM bytes (hi+lo once per M-tile, x once per N-tile) and
  the atomic count (m * 256 * SPLIT_K). Pre-screen every config by read-flush graph replay (15
  reps, empty-graph baseline), keep the best two per m, check each (rel L2 <= 5e-5 vs fp64, the
  power-of-two probe row within 1e-3 as today), then two read-flush _cell_gate rounds against
  today's config (master/cand >= 1.03 in both, cand IQR <= 1.5x max(master IQR, 0.5 us)); route
  the first finalist that passes; _cell_kernels asserts the same two launches as today (fill +
  _router_hilo_kernel). Cap the sweep at ~40 configs per m and ~40 s total, printing what it
  skipped. State B inherits; AB_CASES router m=128/256 map to the new flags. One PATH CHOSEN RH
  line per m with the top-5 pre-screen table and both rounds, re-printed at the end of prepare.
expected:
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "a config with more bytes in flight per SM or fewer atomics: 7.95 -> ~7.3 us if any config clears 3% in both read-flush rounds (hilo 5.0 us kernel at ~2 TB/s)"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "8 K-steps of 64x64 tiles re-reading hi+lo per M-tile (50 MB L2->SM): 9.51 -> ~8.8 us if a config wins; the tall tile lost here on a write-flush pick, this one must win twice on the read flush"
  - shapes: layer=router load=decode_* m=*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=router load=verify_1 m=8
    direction: flat
    why: cluster kernel untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "router-hilo-tall-tile-m65-m256: it timed one hand-picked tall config per m (BLOCK_M 128/256, deeper split-K) against master in a single write-flush round at a 1.02 bar, routed m=256 and lost in the judge (9.72 -> 9.99); this sweeps ~40 configs per m including shorter split-K / wider BLOCK_N / 2-CTA-per-SM shapes, ranks them under the judge-like read flush, and routes only after two read-flush rounds at >= 1.03 with the IQR rule"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%; predecessor on Do-not-try"
---

# router-hilo-m128-m256-readflush-config-sweep

## Change

Built on master (8f233e88 lineage). At m > 32 the router runs torch.zeros + _router_hilo_kernel with the static _router_hilo_config: (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m <= 128 -> grid 2x4x16, and (64, 64, 8) at m=256 -> grid 4x4x8, BLOCK_K 128, 3 stages, 4 warps (profile a6424247: hilo 5.0 us at m=128 and 6.9 us at m=256, rows 7.95 / 9.51 against floors 2.21 / 2.68; 4 / 8 K-steps per CTA, so most of each CTA is pipeline fill, plus 0.52M relaxed fp32 atomics). Only one alternative config was ever timed (the tall tile), with one write-flush round. CHANGE: make the hilo launch config a per-m state entry. _router_hilo_forward takes (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages) from state['router_hilo_cfg'][m_class] when state['router_hilo_cfg_ok_<m_class>'] is set (m_class 128 for 33..128, 256 for 129..256; cfg/shape-only branching), else today's _router_hilo_config; the kernel body is unchanged (same hi+lo fp32 accumulate and atomic_add, only tiling and split order differ). Prepare, after _router_prepare, state A: for m in (128, 256) enumerate BLOCK_M {64, 128}, BLOCK_N {32, 64, 128}, BLOCK_K {64, 128}, SPLIT_K {4, 8, 16, 32} with 8192 % (BLOCK_K * SPLIT_K) == 0 and a grid of 64..264 CTAs, num_warps 4 (8 when BLOCK_M 128 and BLOCK_N 128), num_stages {2, 3, 4} that compile without OutOfResources (catch it, as _router_prepare does); print per compiled config its grid, n_regs/spills, the modeled L2->SM bytes (hi+lo once per M-tile, x once per N-tile) and the atomic count (m * 256 * SPLIT_K). Pre-screen every config by read-flush graph replay (15 reps, empty-graph baseline), keep the best two per m, check each (rel L2 <= 5e-5 vs fp64, the power-of-two probe row within 1e-3 as today), then two read-flush _cell_gate rounds against today's config (master/cand >= 1.03 in both, cand IQR <= 1.5x max(master IQR, 0.5 us)); route the first finalist that passes; _cell_kernels asserts the same two launches as today (fill + _router_hilo_kernel). Cap the sweep at ~40 configs per m and ~40 s total, printing what it skipped. State B inherits; AB_CASES router m=128/256 map to the new flags. One PATH CHOSEN RH line per m with the top-5 pre-screen table and both rounds, re-printed at the end of prepare.

## Expected

- `layer=router load=verify_16 m=128` faster: a config with more bytes in flight per SM or fewer atomics: 7.95 -> ~7.3 us if any config clears 3% in both read-flush rounds (hilo 5.0 us kernel at ~2 TB/s)
- `layer=router load=verify_32 m=256` faster: 8 K-steps of 64x64 tiles re-reading hi+lo per M-tile (50 MB L2->SM): 9.51 -> ~8.8 us if a config wins; the tall tile lost here on a write-flush pick, this one must win twice on the read flush
- `layer=router load=decode_* m=*` flat: m <= 32 untouched
- `layer=router load=verify_1 m=8` flat: cluster kernel untouched

## Risk

Effect: -0.5..-0.7 us on each of two router rows, ~+0.2% headline: sub-margin alone, readable on the router verify_16 / verify_32 rows and the RH gate lines (the pre-screen table also tells whether the hilo kernel is L2-ingest, atomic or pipeline-fill bound: three different next steps). Decode-portable: decode's T=128 'hilo router' cell (W6-B) launches the same kernel. Do not dispatch in the same check as router-zero-pdl-overlap-m9-m256 (same rows); if both win separately, restack. Python/Triton only; compile time is the main cost (each config one Triton compile, ~0.5-1.5 s).
