---
problem: muse_prefill_layer
slug: experts-triton-turn4-persistent-local-align-control-no-persi
created: 2026-10-25T010741Z
base: master
area: w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid
change: >
  Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send
  1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 1024, with a comment); T <= 1024
  keeps _experts_marlin. (3) In _experts_triton, when state['align_par_ok'] is set, replace
  moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS) with _align_parallel(local_ids,
  0, LOCAL_EXPERTS, cfg['BLOCK_T']), which already exists on master for the Marlin path. Extend
  _align_par_setup's self-check to cover block 128 and 256 at T=3012 and 16384 (per-expert route
  sets vs moe_align_block_size). Because no -1 tiles remain, pass zero_nonlocal=False to the w2
  call on this path. Add a prepare() NaN-poison check: fill cache3 with NaN, run the finalize, and
  assert a finite output. Run _check_experts_e2e at T=3012 as well as 8192 (rel L2 <= 2e-2, worst
  row <= 5x). Any failure sets MARLIN_MAX_TOKENS back to 4096 for the run and logs '[turn4-triton]
  OFF: <reason>'. Otherwise log 'ON'. _moe_nvfp4_gemm_kernel's grid stays as on master (no
  persistent loop). Everything else is unchanged.
expected:
  - shapes: "*load=turn_4*"
    direction: flat
    why: ~165 -1 tiles and the 170 MB zero-store removed, est. 50-120 us; kernel row 2.0-2.25 ms vs Marlin 2.12-2.15 ms
  - shapes: "*load=first_turn*"
    direction: faster
    why: local align also removes remote -1 tiles at T=16384; 1-3%
  - shapes: "*load=turn_1*"
    direction: flat
    why: T=753 stays on Marlin
risk: medium
depends_on: experts-triton-turn4-persistent-local-align
priority: 2
kind: control
controls_for: experts-triton-turn4-persistent-local-align
review: required
status: queued
---

# experts-triton-turn4-persistent-local-align-control-no-persi

## Change

Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 1024, with a comment); T <= 1024 keeps _experts_marlin. (3) In _experts_triton, when state['align_par_ok'] is set, replace moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']), which already exists on master for the Marlin path. Extend _align_par_setup's self-check to cover block 128 and 256 at T=3012 and 16384 (per-expert route sets vs moe_align_block_size). Because no -1 tiles remain, pass zero_nonlocal=False to the w2 call on this path. Add a prepare() NaN-poison check: fill cache3 with NaN, run the finalize, and assert a finite output. Run _check_experts_e2e at T=3012 as well as 8192 (rel L2 <= 2e-2, worst row <= 5x). Any failure sets MARLIN_MAX_TOKENS back to 4096 for the run and logs '[turn4-triton] OFF: <reason>'. Otherwise log 'ON'. _moe_nvfp4_gemm_kernel's grid stays as on master (no persistent loop). Everything else is unchanged.

## Expected

- `*load=turn_4*` flat: ~165 -1 tiles and the 170 MB zero-store removed, est. 50-120 us; kernel row 2.0-2.25 ms vs Marlin 2.12-2.15 ms
- `*load=first_turn*` faster: local align also removes remote -1 tiles at T=16384; 1-3%
- `*load=turn_1*` flat: T=753 stays on Marlin

## Risk

Separates the turn_4 Triton-vs-Marlin question and the -1-tile removal from the persistent-grid change still being measured.
