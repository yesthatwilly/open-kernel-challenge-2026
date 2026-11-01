---
problem: muse_decode_layer
slug: marlin-hot-cold-two-class-bsm-t128-t256
created: 2026-10-25T212731Z
base: master
area: w13 marlin + w2 marlin (block_size_m per expert class, 128 <= T <= 256)
change: >
  At 128 <= tokens <= 256 only (verify_16 T=128, verify_32 T=256; new constants
  TWO_CLASS_MIN_TOKENS = 128 / TWO_CLASS_MAX_TOKENS = 256 and state['two_class_<T>']), run each
  Marlin GEMM as two launches over disjoint expert classes instead of one: COLD experts (local
  route count <= HOT_ROWS = 16) at master's block_size_m 16, HOT experts (count > 16) at a LARGER
  block H in {32, 48, 64} picked by the selector. Mechanism: Marlin streams an expert's whole
  weight once per m-block, so a hot expert with 40 rows at bsm 16 streams its ~28 MB three times;
  at bsm 48 once. A uniformly larger block lost 18-28% (moe next-larger) because it pads every
  cold expert, and a uniformly smaller block lost at T=128 (moe marlin-block-m-8-verify16, 391 vs
  341 us) because hot experts re-read more: the two classes take the good side of both.
  IMPLEMENTATION in _experts at that gate (everything else master): (1) one new Triton kernel over
  the T*8 ids (one program, <= 2048 ids): per-local-expert counts (32 ints), then ids_cold /
  ids_hot = the (id - local_start) % 256 ids with the other class's routes replaced by 255
  (non-local for _align_parallel, which treats ids >= 32 as remote); (2) two _align_parallel calls
  (bsm 16 on ids_cold, bsm H on ids_hot), capacities from shape only; (3) w13: two
  moe_wna16_marlin_gemm calls writing the same cache1 (route rows t*8+k are disjoint between
  classes), then silu_and_mul ONCE over cache1 with the FULL local_ids; (4) w2: two calls writing
  the same cache3 view, disjoint rows; post-norm unchanged with the full local_ids. VARIANT S
  (side stream): the hot-class calls on state['side_stream'] with event fork/join, since the hot
  class is a few experts that underfill the grid alone. SELECTOR per T in {128, 256}: candidates H
  in {32, 48, 64} x {serial, side-stream} vs master's whole _experts (ids -> out), decode-like
  soft routing and zipf skew (the rega combo's selector used 163/208 local routes at T=128 and
  320/480 at T=256), worse-of-two >= 1.02x; print per routing the hot-class size (experts, rows)
  and '[2c] PATH CHOSEN T=<t>: H=<h> <mode> X us vs master Y us'. Self-check rel-L2 vs master and
  an fp32-dequant reference <= max(1.5x master's own error, 1.25e-2), cache1/cache3 poisoned
  before the calls (proves every route row is written by exactly one class), two-state refilled
  replays. If wg2 (wg2-occ2-b256-decode-port) or ffn-front-t128-t256-concat-route lands first,
  time against the incumbent as routed. HARD RULES (every path): CUDA-graph capturable; kernel_fn
  has no host sync, no timing, no prints, no try/except, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow and no per-call mutable Python state;
  gates read config/shape and this state's own prepare() flags only; nothing cached from inputs;
  NVFP4 weights stay NVFP4 (lossless repack allowed, dequant in registers only), BF16 stays BF16;
  no child processes. The judge prepares TWO state copies and rotates them: every device pointer,
  TMA map, scratch tensor and ticket lives in the state, never in a module global. Every new CUDA
  build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a),
  -std=c++20 -O3 -lineinfo, a renamed load_inline module, wgmma/setmaxnreg bodies under
  __CUDA_ARCH_FEAT_SM90_ALL, plain bulk copies with no createpolicy / L2::cache_hint operand, a
  numRegs / localSizeBytes / smem print per instantiation (a spill line is a bug: fix before
  spending the check), an in-process canary with synchronize + 'launching'/'ok' prints after each
  first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master
  for that cell. Self-check by capturing once and replaying with refilled inputs while alternating
  two states; force every bucket/class (zipf-skewed, random-local, one-expert, all-remote) and
  name the first non-finite tensor. Selectors time new vs incumbent by interleaved CUDA-graph
  replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both
  medians, enable a cell only when the new path WINS by >= 1.02x on the worse of the two routings,
  and assert the enabled choice equals the printed winner. Print compile seconds per module and
  cumulative prepare() wall time; prepare must stay under 270 s.
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: Marlin pair 314.8 / 272.9 us (sliding/full, profile a903b9d0) = 66% of the verify_16 load, ~1.4 TB/s vs a ~101 us expert floor; -5..-15% on Marlin = -15..-45 us per row, verify_16 load -3..-10% (~+0.5..+1.6% headline)
  - shapes: "*load=verify_32 *"
    direction: faster
    why: "Marlin pair 397.2 / 375.5 us (59% of the load) vs ~109 us floor; hot experts are larger here (320/480 local routes): -5..-12%, verify_32 load -3..-7% (~+0.5..+1.2%)"
  - shapes: "*load=decode_*"
    direction: flat
    why: "control: T<=32 dense chain untouched"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "control: T=8 untouched"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "marlin-block-m-cap-16: that lowered every expert's block to 16 at T>=128 (gain 8-15%); moe marlin-block-m-8-verify16 lowered it further and regressed (391 vs 341 us) from hot-expert re-reads; moe marlin-two-class-block-m-decode32 (expired, never ran) split COLD experts to bsm 8 at T=17..32. This keeps cold experts at 16 and gives only hot experts a LARGER block so their weights stream once, at T=128/256, with an optional side stream for the hot class: never tried"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: hotcold landed (H=-16)"
---

# marlin-hot-cold-two-class-bsm-t128-t256

## Change

At 128 <= tokens <= 256 only (verify_16 T=128, verify_32 T=256; new constants TWO_CLASS_MIN_TOKENS = 128 / TWO_CLASS_MAX_TOKENS = 256 and state['two_class_<T>']), run each Marlin GEMM as two launches over disjoint expert classes instead of one: COLD experts (local route count <= HOT_ROWS = 16) at master's block_size_m 16, HOT experts (count > 16) at a LARGER block H in {32, 48, 64} picked by the selector. Mechanism: Marlin streams an expert's whole weight once per m-block, so a hot expert with 40 rows at bsm 16 streams its ~28 MB three times; at bsm 48 once. A uniformly larger block lost 18-28% (moe next-larger) because it pads every cold expert, and a uniformly smaller block lost at T=128 (moe marlin-block-m-8-verify16, 391 vs 341 us) because hot experts re-read more: the two classes take the good side of both. IMPLEMENTATION in _experts at that gate (everything else master): (1) one new Triton kernel over the T*8 ids (one program, <= 2048 ids): per-local-expert counts (32 ints), then ids_cold / ids_hot = the (id - local_start) % 256 ids with the other class's routes replaced by 255 (non-local for _align_parallel, which treats ids >= 32 as remote); (2) two _align_parallel calls (bsm 16 on ids_cold, bsm H on ids_hot), capacities from shape only; (3) w13: two moe_wna16_marlin_gemm calls writing the same cache1 (route rows t*8+k are disjoint between classes), then silu_and_mul ONCE over cache1 with the FULL local_ids; (4) w2: two calls writing the same cache3 view, disjoint rows; post-norm unchanged with the full local_ids. VARIANT S (side stream): the hot-class calls on state['side_stream'] with event fork/join, since the hot class is a few experts that underfill the grid alone. SELECTOR per T in {128, 256}: candidates H in {32, 48, 64} x {serial, side-stream} vs master's whole _experts (ids -> out), decode-like soft routing and zipf skew (the rega combo's selector used 163/208 local routes at T=128 and 320/480 at T=256), worse-of-two >= 1.02x; print per routing the hot-class size (experts, rows) and '[2c] PATH CHOSEN T=<t>: H=<h> <mode> X us vs master Y us'. Self-check rel-L2 vs master and an fp32-dequant reference <= max(1.5x master's own error, 1.25e-2), cache1/cache3 poisoned before the calls (proves every route row is written by exactly one class), two-state refilled replays. If wg2 (wg2-occ2-b256-decode-port) or ffn-front-t128-t256-concat-route lands first, time against the incumbent as routed. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; NVFP4 weights stay NVFP4 (lossless repack allowed, dequant in registers only), BF16 stays BF16; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, TMA map, scratch tensor and ticket lives in the state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), -std=c++20 -O3 -lineinfo, a renamed load_inline module, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL, plain bulk copies with no createpolicy / L2::cache_hint operand, a numRegs / localSizeBytes / smem print per instantiation (a spill line is a bug: fix before spending the check), an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states; force every bucket/class (zipf-skewed, random-local, one-expert, all-remote) and name the first non-finite tensor. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x on the worse of the two routings, and assert the enabled choice equals the printed winner. Print compile seconds per module and cumulative prepare() wall time; prepare must stay under 270 s.

## Expected

- `*load=verify_16 *` faster: Marlin pair 314.8 / 272.9 us (sliding/full, profile a903b9d0) = 66% of the verify_16 load, ~1.4 TB/s vs a ~101 us expert floor; -5..-15% on Marlin = -15..-45 us per row, verify_16 load -3..-10% (~+0.5..+1.6% headline)
- `*load=verify_32 *` faster: Marlin pair 397.2 / 375.5 us (59% of the load) vs ~109 us floor; hot experts are larger here (320/480 local routes): -5..-12%, verify_32 load -3..-7% (~+0.5..+1.2%)
- `*load=decode_*` flat: control: T<=32 dense chain untouched
- `*load=verify_1 *` flat: control: T=8 untouched

## Risk

Both cells routing at the low end gives ~+1% (must stack, e.g. with ffn-front-t128-t256-concat-route on the same shapes); the high end clears the margin alone. Risks: extra launches (count/mask + second align + two Marlin calls each: ~4 graph nodes, ~3-6 us) eat the gain if the hot class is tiny; Marlin's workspace locks shared by two concurrent launches in variant S (give the side-stream calls their own workspace from marlin_make_workspace). No new CUDA; Triton + SGLang Marlin only, so no fault path.
