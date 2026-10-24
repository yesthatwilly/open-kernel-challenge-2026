---
problem: muse_decode_layer
slug: moe-small-t-cuda-swapab-port
created: 2026-10-24T184335Z
base: master
area: w13 marlin + silu-mul + w2 marlin (T<=32)
change: >
  Port muse_moe_w4a16 cuda-nvfp4-swapab-mma-persistent-small-m (impl commit f8b3f72, branch
  hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m, moe job 8f198afc3bf14f6f) into
  decode _experts at num_tokens <= 32. Copy verbatim from f8b3f72:muse_moe_w4a16/submission.py:
  ALIGN_FUSED_MAX_TOKENS, _align_small_kernel, _block_size_m, _align_small, _align_self_check (the
  f8b3f72 version with bsm_of and tag0: _cuda_setup tests its return value, and LESSONS records
  that the older version always reported a capacity mismatch, so do not substitute another copy),
  and the whole CUDA block from CUDA_MAX_TOKENS through _cuda_setup (CUDA_TAG, CUDA_SK13,
  CUDA_SK2, CUDA_THREADS, CUDA_STAGES, _CUDA_CPP, _CUDA_SRC, _cuda_nt, _cuda_bsm, _build_cuda,
  _nibbles, _pack_nibbles, _code_dims, _scale_dims, _inverse, _tile_codes, _tile_scales,
  _cuda_bank, _decode_nvfp4, _row_rel, _cuda_self_check, _cuda_setup), plus moe _moe renamed
  _moe_small_t. Adapt only names: decode state["bank"] gains "start": 0 and "count": 32 (decode
  hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are
  0..31, which is start 0); in _moe_small_t and the self-check, moe state["workspace"] ->
  bank["workspace"], state["norm_w"] -> state["post_expert_w"], state["post_norm_w"] ->
  state["post_expert_norm_w"]; weights keys w13, w13_sf, w13_gs, w2, w2_sf, w2_gs are identical in
  both problems. prepare(): after state is built, call _cuda_setup(weights, {(0, 32):
  state["bank"]}, state, device); it compiles with load_inline (about 41 s), retiles losslessly
  (NVFP4 codes and E4M3 scales stay bytes, round trips asserted), self-checks against the Marlin
  path and prints "[cuda-w4a16] PATH CHOSEN = CUDA|MARLIN". _experts(x, topk_weights, topk_ids,
  state): if num_tokens <= CUDA_MAX_TOKENS and state.get("cuda_ok"): return _moe_small_t(x,
  topk_ids, topk_weights, state["bank"], state, True)[0], passing the shifted ids (not
  bank["expert_map"][ids]: _align_small maps them itself); it runs align at BSM 8/16/32, the
  w13+silu launch, the w2 launch and master post-norm tail (_post_norm_combine_kernel below 32,
  fused_post_expert_norm_weighted_sum at 32). Otherwise master byte for byte; T > 32 untouched.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent Marlin w13+w2 40.7 us (sliding) / 44.9 us (full) + act_and_mul + align ~6 us for 1-2 local experts (~25-50 MB of NVFP4 at 4.8 TB/s = 5-10 us); M1 replaced sum <= 46 us, M2 <= 28 us: -15..-25 us, +10-17%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "Marlin 97.5 / 108.9 us at ~1.2 TB/s, 46% of sliding decode_16; M2 <= 60 us: -30..-40 us, +13-17% sliding"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "Marlin 147 / 146 us, 50% of sliding decode_32 (the 1.000x shape); M2 <= 90 us: -40..-55 us, +15-20%"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "Marlin 116 / 96.6 us at T=8; M2 <= 70 us: +15-20%"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: T=128 outside the gate
  - shapes: "*load=verify_32 *"
    direction: flat
    why: T=256 outside the gate
risk: high
depends_on: null
priority: 4
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13 + silu_and_mul + Marlin w2 + SGLang align at T<=32
review: required
status: queued
---

# moe-small-t-cuda-swapab-port

## Change

Port muse_moe_w4a16 cuda-nvfp4-swapab-mma-persistent-small-m (impl commit f8b3f72, branch hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m, moe job 8f198afc3bf14f6f) into decode _experts at num_tokens <= 32. Copy verbatim from f8b3f72:muse_moe_w4a16/submission.py: ALIGN_FUSED_MAX_TOKENS, _align_small_kernel, _block_size_m, _align_small, _align_self_check (the f8b3f72 version with bsm_of and tag0: _cuda_setup tests its return value, and LESSONS records that the older version always reported a capacity mismatch, so do not substitute another copy), and the whole CUDA block from CUDA_MAX_TOKENS through _cuda_setup (CUDA_TAG, CUDA_SK13, CUDA_SK2, CUDA_THREADS, CUDA_STAGES, _CUDA_CPP, _CUDA_SRC, _cuda_nt, _cuda_bsm, _build_cuda, _nibbles, _pack_nibbles, _code_dims, _scale_dims, _inverse, _tile_codes, _tile_scales, _cuda_bank, _decode_nvfp4, _row_rel, _cuda_self_check, _cuda_setup), plus moe _moe renamed _moe_small_t. Adapt only names: decode state["bank"] gains "start": 0 and "count": 32 (decode hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are 0..31, which is start 0); in _moe_small_t and the self-check, moe state["workspace"] -> bank["workspace"], state["norm_w"] -> state["post_expert_w"], state["post_norm_w"] -> state["post_expert_norm_w"]; weights keys w13, w13_sf, w13_gs, w2, w2_sf, w2_gs are identical in both problems. prepare(): after state is built, call _cuda_setup(weights, {(0, 32): state["bank"]}, state, device); it compiles with load_inline (about 41 s), retiles losslessly (NVFP4 codes and E4M3 scales stay bytes, round trips asserted), self-checks against the Marlin path and prints "[cuda-w4a16] PATH CHOSEN = CUDA|MARLIN". _experts(x, topk_weights, topk_ids, state): if num_tokens <= CUDA_MAX_TOKENS and state.get("cuda_ok"): return _moe_small_t(x, topk_ids, topk_weights, state["bank"], state, True)[0], passing the shifted ids (not bank["expert_map"][ids]: _align_small maps them itself); it runs align at BSM 8/16/32, the w13+silu launch, the w2 launch and master post-norm tail (_post_norm_combine_kernel below 32, fused_post_expert_norm_weighted_sum at 32). Otherwise master byte for byte; T > 32 untouched.

## Expected

- `*load=decode_1 *` faster: incumbent Marlin w13+w2 40.7 us (sliding) / 44.9 us (full) + act_and_mul + align ~6 us for 1-2 local experts (~25-50 MB of NVFP4 at 4.8 TB/s = 5-10 us); M1 replaced sum <= 46 us, M2 <= 28 us: -15..-25 us, +10-17%
- `*load=decode_16 *` faster: Marlin 97.5 / 108.9 us at ~1.2 TB/s, 46% of sliding decode_16; M2 <= 60 us: -30..-40 us, +13-17% sliding
- `*load=decode_32 *` faster: Marlin 147 / 146 us, 50% of sliding decode_32 (the 1.000x shape); M2 <= 90 us: -40..-55 us, +15-20%
- `*load=verify_1 *` faster: Marlin 116 / 96.6 us at T=8; M2 <= 70 us: +15-20%
- `*load=verify_16 *` flat: T=128 outside the gate
- `*load=verify_32 *` flat: T=256 outside the gate

## Risk

CONDITIONAL: dispatch only after moe job 8f198afc3bf14f6f has harvested with "[cuda-w4a16] PATH CHOSEN = CUDA" in its log and its bs<=32 shapes faster than moe master; if it fell back or lost, hold this file. Milestones from the decode profile at T<=32: the Marlin rows and act_and_mul are gone and the w4a16 kernel rows present (else the path silently fell back, read the PATH CHOSEN line first); M1 replaced-kernel sum <= the Marlin + act_and_mul + align sum per shape; M2 <= 0.6x of it. Budget: 41 s compile on top of decode prepare inside the 15-minute run. Overlap: small-t-launch-minimal-dense-chain-pdl (in judge) rewrites the pre-expert chain and align at T<=32 but keeps Marlin; read the replaced rows, not the headline. At T=1 this competes with moe-bs1-fused-persistent-port; if both moe runs win, give T=1 to the fused one and gate this to 2 <= T <= 32. Stage 2: fold the pre-expert norm and the post-norm into the two launches.

## Milestones

Rewrite stage 1 of `Marlin w13 + silu_and_mul + Marlin w2 + SGLang align at T<=32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
