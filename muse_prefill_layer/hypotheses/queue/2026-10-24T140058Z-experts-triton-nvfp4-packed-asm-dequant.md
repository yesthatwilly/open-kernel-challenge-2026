---
problem: muse_prefill_layer
slug: experts-triton-nvfp4-packed-asm-dequant
created: 2026-10-24T140058Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant)"
change: >
  Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4,
  experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_kernel in
  muse_prefill_layer/submission.py changes; the shape gate (Marlin at T<=4096, Triton above),
  EXPERT_CONFIGS, the nibble order prepare() stores, the scale tile construction (sb_lo/sb_hi),
  the epilogue, _experts_* and kernel_fn stay byte-identical. Today each code byte is loaded as
  one int32 lane (b = tl.load(b_ptrs).to(tl.int32)) and every E2M1 element costs ~6 scalar int ops
  (two and/shift/or chains), an int32->uint16 truncation, a bitcast and a bf16 multiply that
  Triton lowers as fp32 mul plus two converts: roughly 10-12 SASS instructions per weight element,
  repeated per token tile. LESSONS records this engine as per-tile dequant-overhead bound (losses
  shrinking with T; BLOCK_T 256 gaining by halving redundant dequants). Replace the lines from b =
  tl.load(...) through w_hi = ... with one tl.inline_asm_elementwise call (pack=4, is_pure=True)
  on the raw uint8 code tile [BLOCK_W, HALF] (4 bytes = one 32-bit register) plus the two
  broadcast scale tiles sb_lo, sb_hi (bf16; pack 4 = two bf16x2 registers each), returning a tuple
  (w_lo, w_hi) of bf16 [BLOCK_W, HALF]. Inside the asm, per pair of bytes: prmt.b32 spreads bytes
  (0,1) and (2,3) into the low byte of each 16-bit half (t); the low-nibble bf16x2 is ((t << 6) &
  0x01C001C0) | ((t << 12) & 0x80008000) and the high-nibble one ((t << 2) & 0x01C001C0) | ((t <<
  8) & 0x80008000) (shl.b32 + lop3.b32: the same bit placement as today, E2M1 * 2^-126 with e=0
  codes on bf16 subnormals), then mul.rn.bf16x2 (sm_90; or fma.rn.bf16x2 with a -0.0 addend) by
  the matching scale register. About 3 instructions per element instead of 10-12, and the product
  stays exact as today (<= 6 significant bits). If the Triton build rejects a tuple dtype, use two
  asm calls (one per nibble half) on the same packed input. Keep today's Python dequant behind a
  module constant PACKED_DEQUANT = True (False restores today's code exactly) so a control can
  flip it. _check_expert_gemms is unchanged and must pass for every EXPERT_CONFIGS entry (it
  compares against nvfp4.dequantize of the original packing, including the +-0.5 subnormal codes;
  if the bf16x2 multiply flushes subnormals the check says so, and the fix is to place codes at a
  larger power of two and fold the difference into SCALE_BIAS / EPILOGUE_BIAS). Diagnostics in
  prepare() only: extend the existing report print with the number of ttg.convert_layout and
  ttg.local_alloc ops in compiled.asm["ttgir"] for each variant, so the log shows whether the
  dequantized tile still round-trips through shared memory before the register-LHS wgmma (that
  decides what stage 6 restructures). No BF16 weight copy anywhere (dequant stays in registers per
  the rules ruling), no autotune, no host sync, nothing cached from inputs.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "replaced kernels: _moe_nvfp4_gemm_kernel x2 is 8149-8873 us per first_turn shape (57% of sliding first_turn, 14.3 ms); the w13+w2 floor is 1.65 TFLOP = 1.67 ms at 989 TFLOP/s, so the kernel is ~5x its floor. Cutting per-element dequant from ~10-12 to ~3 instructions should take w13+w2 to ~5-6 ms (Milestone: under 7.0 ms on every first_turn shape), 2.5-3.5 ms off a 14-22 ms shape, 15-25% on the first_turn load"
  - shapes: "*load=turn_*"
    direction: neutral
    why: T<=4096 still runs Marlin; that code path is untouched
risk: medium
depends_on: experts-hybrid-marlin-turn-triton-first-turn
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: required
status: queued
---

# experts-triton-nvfp4-packed-asm-dequant

## Change

Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_kernel in muse_prefill_layer/submission.py changes; the shape gate (Marlin at T<=4096, Triton above), EXPERT_CONFIGS, the nibble order prepare() stores, the scale tile construction (sb_lo/sb_hi), the epilogue, _experts_* and kernel_fn stay byte-identical. Today each code byte is loaded as one int32 lane (b = tl.load(b_ptrs).to(tl.int32)) and every E2M1 element costs ~6 scalar int ops (two and/shift/or chains), an int32->uint16 truncation, a bitcast and a bf16 multiply that Triton lowers as fp32 mul plus two converts: roughly 10-12 SASS instructions per weight element, repeated per token tile. LESSONS records this engine as per-tile dequant-overhead bound (losses shrinking with T; BLOCK_T 256 gaining by halving redundant dequants). Replace the lines from b = tl.load(...) through w_hi = ... with one tl.inline_asm_elementwise call (pack=4, is_pure=True) on the raw uint8 code tile [BLOCK_W, HALF] (4 bytes = one 32-bit register) plus the two broadcast scale tiles sb_lo, sb_hi (bf16; pack 4 = two bf16x2 registers each), returning a tuple (w_lo, w_hi) of bf16 [BLOCK_W, HALF]. Inside the asm, per pair of bytes: prmt.b32 spreads bytes (0,1) and (2,3) into the low byte of each 16-bit half (t); the low-nibble bf16x2 is ((t << 6) & 0x01C001C0) | ((t << 12) & 0x80008000) and the high-nibble one ((t << 2) & 0x01C001C0) | ((t << 8) & 0x80008000) (shl.b32 + lop3.b32: the same bit placement as today, E2M1 * 2^-126 with e=0 codes on bf16 subnormals), then mul.rn.bf16x2 (sm_90; or fma.rn.bf16x2 with a -0.0 addend) by the matching scale register. About 3 instructions per element instead of 10-12, and the product stays exact as today (<= 6 significant bits). If the Triton build rejects a tuple dtype, use two asm calls (one per nibble half) on the same packed input. Keep today's Python dequant behind a module constant PACKED_DEQUANT = True (False restores today's code exactly) so a control can flip it. _check_expert_gemms is unchanged and must pass for every EXPERT_CONFIGS entry (it compares against nvfp4.dequantize of the original packing, including the +-0.5 subnormal codes; if the bf16x2 multiply flushes subnormals the check says so, and the fix is to place codes at a larger power of two and fold the difference into SCALE_BIAS / EPILOGUE_BIAS). Diagnostics in prepare() only: extend the existing report print with the number of ttg.convert_layout and ttg.local_alloc ops in compiled.asm["ttgir"] for each variant, so the log shows whether the dequantized tile still round-trips through shared memory before the register-LHS wgmma (that decides what stage 6 restructures). No BF16 weight copy anywhere (dequant stays in registers per the rules ruling), no autotune, no host sync, nothing cached from inputs.

## Expected

- `*load=first_turn*` faster: replaced kernels: _moe_nvfp4_gemm_kernel x2 is 8149-8873 us per first_turn shape (57% of sliding first_turn, 14.3 ms); the w13+w2 floor is 1.65 TFLOP = 1.67 ms at 989 TFLOP/s, so the kernel is ~5x its floor. Cutting per-element dequant from ~10-12 to ~3 instructions should take w13+w2 to ~5-6 ms (Milestone: under 7.0 ms on every first_turn shape), 2.5-3.5 ms off a 14-22 ms shape, 15-25% on the first_turn load
- `*load=turn_*` neutral: T<=4096 still runs Marlin; that code path is untouched

## Risk

Risk: ptxas may already pack part of today's chain, in which case the gain is smaller (the per-variant ttgir counts and n_regs in the log tell); inline asm with pack=4 on uint8 needs 4 consecutive bytes per thread in the code tile's layout (Triton packs them or inserts shuffles); bf16x2 subnormal handling must pass _check_expert_gemms. If the log shows the dequantized tile goes through shared memory before the dot, stage 6 should restructure the K loop rather than tune.

## Milestones

Rewrite stage 5 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
