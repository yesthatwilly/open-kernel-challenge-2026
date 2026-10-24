---
problem: muse_prefill_layer
slug: w2-nonlocal-zero-store-off-first-turn
created: 2026-10-24T184923Z
base: master
area: "w2 (Triton NVFP4 engine at T > 4096: remote-block zero store)"
change: >
  Built on master (635cfee). One flag and one prepare() guard. Today ZERO_NONLOCAL_W2 = True makes
  the second _nvfp4_moe_gemm call in _experts_triton (T > 4096 only) store a zero 256x128 bf16
  tile for every (-1 token tile, W tile) pair: at T=16384 about 114K remote slot rows x 4096 x 2 B
  = ~940 MB of stores across ~14K CTAs, which nothing reads if the weighted sum skips remote
  slots. Evidence that it does: fused_post_expert_norm_weighted_sum takes
  slot_expert_ids=local_ids and runs in 122 us at first_turn, under the ~210 us it would need just
  to read all T*8 y rows at 4.8 TB/s; and muse_moe_w4a16 master passes every test with this same
  SGLang kernel over y rows its Marlin never writes. CHANGE: add a prepare() check
  _weighted_sum_skips_remote(state, device): build y [T=4096, 8, 4096] bf16 random, local_ids with
  ~7/8 entries -1, topk_weights random normalized; call fused_post_expert_norm_weighted_sum(y, w,
  norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True,
  slot_expert_ids=local_ids) once with the remote slots of y filled with NaN and once with them
  zero; if the outputs are bitwise equal and finite, set state['zero_nonlocal_w2'] = False, else
  True; print '[w2 zero-store] OFF (weighted sum skips remote slots)' or 'ON: <reason>'. In
  _experts_triton pass state['zero_nonlocal_w2'] instead of the module constant ZERO_NONLOCAL_W2
  to the w2 call. _check_expert_gemms already checks both settings' semantics (rows untouched when
  False); run it for the value chosen. Nothing else changes (the w13 call already passes False;
  the Marlin path is untouched).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "the w2 _moe_nvfp4_gemm_kernel call stops writing ~940 MB of zeros (~200-250 us of the 8149-8873 us expert row): about -1.3 to -1.7% on first_turn shapes"
  - shapes: "*load=turn_*"
    direction: neutral
    why: Marlin path unchanged
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# w2-nonlocal-zero-store-off-first-turn

## Change

Built on master (635cfee). One flag and one prepare() guard. Today ZERO_NONLOCAL_W2 = True makes the second _nvfp4_moe_gemm call in _experts_triton (T > 4096 only) store a zero 256x128 bf16 tile for every (-1 token tile, W tile) pair: at T=16384 about 114K remote slot rows x 4096 x 2 B = ~940 MB of stores across ~14K CTAs, which nothing reads if the weighted sum skips remote slots. Evidence that it does: fused_post_expert_norm_weighted_sum takes slot_expert_ids=local_ids and runs in 122 us at first_turn, under the ~210 us it would need just to read all T*8 y rows at 4.8 TB/s; and muse_moe_w4a16 master passes every test with this same SGLang kernel over y rows its Marlin never writes. CHANGE: add a prepare() check _weighted_sum_skips_remote(state, device): build y [T=4096, 8, 4096] bf16 random, local_ids with ~7/8 entries -1, topk_weights random normalized; call fused_post_expert_norm_weighted_sum(y, w, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) once with the remote slots of y filled with NaN and once with them zero; if the outputs are bitwise equal and finite, set state['zero_nonlocal_w2'] = False, else True; print '[w2 zero-store] OFF (weighted sum skips remote slots)' or 'ON: <reason>'. In _experts_triton pass state['zero_nonlocal_w2'] instead of the module constant ZERO_NONLOCAL_W2 to the w2 call. _check_expert_gemms already checks both settings' semantics (rows untouched when False); run it for the value chosen. Nothing else changes (the w13 call already passes False; the Marlin path is untouched).

## Expected

- `*load=first_turn*` faster: the w2 _moe_nvfp4_gemm_kernel call stops writing ~940 MB of zeros (~200-250 us of the 8149-8873 us expert row): about -1.3 to -1.7% on first_turn shapes
- `*load=turn_*` neutral: Marlin path unchanged

## Risk

Single-factor measurement of the zero-store cost. It is a strict subset of moe-align-local-routes-parallel-port and first-turn-moe-dead-work-stack at first_turn (those remove the -1 blocks altogether); worth running on its own because experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) keeps the zero-store by design ('w2 still zero-stores them'), so this row tells that rewrite whether to drop it. Risk: if the guard finds NaN leaking it keeps the zero-store and the run measures master (the log line says so).
