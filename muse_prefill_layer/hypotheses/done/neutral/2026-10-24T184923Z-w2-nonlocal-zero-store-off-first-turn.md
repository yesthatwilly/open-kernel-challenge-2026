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
status: done
stage: harvested
claimed_at: 2026-10-24T231849Z
base_sha: 27a44fb6ced1bfd563c16ae512b70c9222e795f8
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/w2-nonlocal-zero-store-off-first-turn
worktree: .worktrees/muse_prefill_layer/w2-nonlocal-zero-store-off-first-turn
mode: check
implemented_at: 2026-10-24T231931Z
impl_commit: 749edd39f465d8f7a10b9f03dfb2ca122002e43c
submission_sha256: 0c199e69da2c065b8ddcf6418157c8e034c27c8ac3efc5196e615a27961bce4d
norm_sha256: 7ad0a1bd699965b5fcf833a09614dcd2854b5b6c322487ee10c2b89abae8fe01
submission_blob: 39f26f165918e7ccc5c37ebbaed148b6b28cff71
lint_warnings: 6
attempt_file: 2026-10-24T231931Z-check-w2-nonlocal-zero-store-off-first-turn.md
prepared_at: 2026-10-24T231931Z
job: 9b731599dfa24278
dispatched_at: 2026-10-24T231940Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0609256881613394
speedup_ci: [1.0450118028389193, 1.0768395734837592]
harvested_at: 2026-10-24T232736Z
best_at_harvest: 1.0591912383762452
promotion: null
---

# w2-nonlocal-zero-store-off-first-turn

## Change

Built on master (635cfee). One flag and one prepare() guard. Today ZERO_NONLOCAL_W2 = True makes the second _nvfp4_moe_gemm call in _experts_triton (T > 4096 only) store a zero 256x128 bf16 tile for every (-1 token tile, W tile) pair: at T=16384 about 114K remote slot rows x 4096 x 2 B = ~940 MB of stores across ~14K CTAs, which nothing reads if the weighted sum skips remote slots. Evidence that it does: fused_post_expert_norm_weighted_sum takes slot_expert_ids=local_ids and runs in 122 us at first_turn, under the ~210 us it would need just to read all T*8 y rows at 4.8 TB/s; and muse_moe_w4a16 master passes every test with this same SGLang kernel over y rows its Marlin never writes. CHANGE: add a prepare() check _weighted_sum_skips_remote(state, device): build y [T=4096, 8, 4096] bf16 random, local_ids with ~7/8 entries -1, topk_weights random normalized; call fused_post_expert_norm_weighted_sum(y, w, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) once with the remote slots of y filled with NaN and once with them zero; if the outputs are bitwise equal and finite, set state['zero_nonlocal_w2'] = False, else True; print '[w2 zero-store] OFF (weighted sum skips remote slots)' or 'ON: <reason>'. In _experts_triton pass state['zero_nonlocal_w2'] instead of the module constant ZERO_NONLOCAL_W2 to the w2 call. _check_expert_gemms already checks both settings' semantics (rows untouched when False); run it for the value chosen. Nothing else changes (the w13 call already passes False; the Marlin path is untouched).

## Expected

- `*load=first_turn*` faster: the w2 _moe_nvfp4_gemm_kernel call stops writing ~940 MB of zeros (~200-250 us of the 8149-8873 us expert row): about -1.3 to -1.7% on first_turn shapes
- `*load=turn_*` neutral: Marlin path unchanged

## Risk

Single-factor measurement of the zero-store cost. It is a strict subset of moe-align-local-routes-parallel-port and first-turn-moe-dead-work-stack at first_turn (those remove the -1 blocks altogether); worth running on its own because experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) keeps the zero-store by design ('w2 still zero-stores them'), so this row tells that rewrite whether to drop it. Risk: if the guard finds NaN leaking it keeps the zero-store and the run measures master (the log line says so).

## Implementation notes

Built in the main thread 2026-10-24 ~23:40Z with self-review (user approved; no new agents before the reset), as specified:
- `_weighted_sum_skips_remote(state, device)` (prepare only): T=4096, ~7/8 remote slots, y remote rows NaN vs 0, outputs must be finite and bitwise equal; prints `[w2 zero-store] OFF ...` or `ON: <reason>`; any exception -> ON (master behaviour).
- prepare(): `state["zero_nonlocal_w2"] = not _weighted_sum_skips_remote(...)` before `_check_expert_gemms(bank, weights, state["zero_nonlocal_w2"])` (new `zero_w2` parameter, default the module constant), so the GEMM check and `_check_experts_e2e` both run for the chosen setting.
- `_experts_triton`: the w2 call passes `state["zero_nonlocal_w2"]`. w13 and the Marlin path unchanged.

## Rules check

- [x] CUDA-graph capturable: kernel_fn gains only a dict lookup of a prepare()-chosen bool.
- [x] Fresh inputs: the flag is derived from synthetic data in prepare(), never from `data`.
- [x] Weights unchanged; NVFP4 stays NVFP4.
- [x] Activations BF16 as in SGLang; KV FP8; attention untouched.
- [x] Every key attended (attention untouched).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added: the probe is a correctness check in prepare(), like the existing ones.
- [x] okc_lint: OK, no errors.

## Result

- job `9b731599dfa24278` (check), status done, passed True, tests 5/5
- speedup 1.0609x, 95% CI [1.0450, 1.0768], score 365156.48958542885 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/w2-nonlocal-zero-store-off-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/w2-nonlocal-zero-store-off-first-turn.patch`; branch `hyp/muse_prefill_layer/w2-nonlocal-zero-store-off-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1079.54 | 1089.33 | 1.0014 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1419.09 | 1475.73 | 0.9991 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3334.9 | 3363.64 | 0.9989 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4937.98 | 4800.68 | 0.9985 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15386.32 | 14759.08 | 0.9875 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14275.3 | 14349.09 | 1.0038 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18165.94 | 17603.8 | 1.0023 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22332.92 | 21727.63 | 1.0112 | flat |

no shape moved outside noise. Expected hit: 0/2. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: w2-nonlocal-zero-store-off-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

