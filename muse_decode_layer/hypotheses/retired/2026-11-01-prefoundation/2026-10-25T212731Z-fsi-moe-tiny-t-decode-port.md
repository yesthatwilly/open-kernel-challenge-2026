---
problem: muse_decode_layer
slug: fsi-moe-tiny-t-decode-port
created: 2026-10-25T212731Z
base: master
area: w13 marlin + silu-mul + w2 marlin + post-expert combine (1 <= T <= 8)
change: >
  Stage 1 of a split-I fused MoE for decode at 1 <= tokens <= 8 (DIAGNOSIS action 1; decode port
  of moe's queued cuda-fused-split-i-moe-tiny-t, which has the full kernel spec: read it). Every
  other T runs master byte for byte. WHERE: master's dense chain at T <= 8 calls _experts(xn, tw,
  lids, state, align=align) with kernel R's outputs (xn bf16 [T, 4096] pre-expert-normed, local
  ids int32 -1 for remote, tw fp32, align = (sorted_ids, block_experts, n_padded) at block_size_m
  8). At T <= 8 an expert has at most T <= 8 routes (a token's 8 ids are distinct), so every
  active local expert is exactly ONE 8-row block of R's align: R's align is already the unit list
  and no extra align launch is needed. Add _fsi_experts(xn, tw, lids, align, state) with _experts'
  exact output contract (out bf16 [T, 4096]) and call it instead of _experts when 1 <= tokens <= 8
  and this state's flag for the T class is on; the Marlin bank stays for every other T and as the
  fallback. MECHANISM: y = W2 . (silu(W1g x) * (W1u x)) is linear in the I-columns of h, so a work
  unit u = (active block b, I-chunk j of KC columns) computes h[:, j] over the full K = 4096,
  applies silu * up, rounds h to bf16 into shared memory, and immediately streams W2[:, j] (all
  4096 H rows) into an fp32 per-state scratch y32 with red.global.add.v4.f32 for the real routes
  only. No unit waits on another (no flags, spins, grid barrier or h round trip), so 2 active
  experts give 128 units at KC=64 and 6 give 384, which fills 132 SMs where Marlin's per-expert
  tiling does not. KERNEL: new module muse_decode_fsi_v1 built from muse_moe_w4a16/submission.py
  at moe BEST commit 64f18b03 (_WG_SRC_HEAD / _WG_SRC_BODY: TMA mbarrier ring, producer warpgroup
  setmaxnreg.dec 40, two consumer warpgroups inc 232, register-A wgmma m64n16k16 = the
  judge-proven N16 bucket, gated silu epilogue, stmatrix w2 epilogue, wg_gather_kernel);
  fsi_kernel is persistent (grid = SM count, u = blockIdx.x, += gridDim.x), one flat stage
  sequence per unit through one ring (STAGES 4): 32 w13 stages (q13 tile j, K = 4096) then 16 w2
  stages (the j-th 64-k record of each of the 32 q2 row tiles), so w2 bytes are in flight while
  the w13 epilogue runs; the w2 B operand is the 16 x 64 bf16 h tile the w13 epilogue wrote in the
  B-slab swizzle (rows 8..15 zero). WEIGHTS: build moe's lossless wg bank for the 32 local experts
  in prepare (q13 gated 128-row tiles = 64 gate + 64 up h-columns, q2 128-row tiles, 4608-byte
  records per 64-k block; copy _wg_nibbles / _wg_pack_nibbles and the bank builder verbatim) and
  print its size (~0.9 GB). GATHER: wg_gather into a per-state 16-row-per-block slab (8 real + 8
  zero rows). COMBINE: one Triton kernel with _post_norm_combine_kernel's formula (out[t] = sum
  over local k of tw * rmsnorm(y * (1 + post_norm_w)), POST_NORM_EPS, fp32) reading y32 rows t*8+k
  and writing zeros back into every row it read (zero-after-read keeps y32 zero for the next
  replay; y32 fp32 [64, 4096] per state, zeroed once in prepare). Launches per call: gather,
  fsi_kernel, combine (master: Marlin w13, act_and_mul, Marlin w2, post-norm). KC = 64 and KC =
  128 instantiations; the selector picks per class. CLASSES {1} and {2..8}. CORRECTNESS: at T in
  1, 2, 4, 7, 8 against an fp32-dequant reference, bound max(2x master's own error vs that
  reference, 1e-2) aggregate and 5x per row (LESSONS: never a fixed bound against another
  approximate kernel), plus two-state refilled replays proving y32 returns to zero. SELECTOR per
  class: master's whole expert block (xn -> out, w2 atomic included at T=1) vs gather + fsi +
  combine; print '[fsi] PATH CHOSEN class=<c> KC=<k>: fsi X us vs master Y us (zipf), X' vs Y'
  (random); fsi kernel Z us, W TB/s of weight bytes'. The first measurement is an okc check, never
  a scored submit. STAGES: s1 this file; s2 fsi-moe-midt-decode-port (9 <= T <= 32); s3 tuning
  from the s1 log (N8 m64n8k16 at T=1, a third consumer warpgroup, ring depth). Fallback at every
  stage: the class stays on Marlin. HARD RULES (every path): CUDA-graph capturable; kernel_fn has
  no host sync, no timing, no prints, no try/except, no allocation sized by input values
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
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "replaced: Marlin pair 42.5 / 42.6 us + act_and_mul ~1.5 + post-norm ~2.9 (profile a903b9d0, sliding/full) for ~2 local experts (~57 MB) vs a ~11.4 us floor; Marlin runs ~1.3 TB/s because 1-2 experts' N-tiles cannot fill 132 SMs. M1: PATH CHOSEN = FSI for class {1} with the self-check passing. M2: gather+fsi+combine <= 0.9x master's block (<= ~42 us). M3: fsi kernel <= 22 us (2.6 TB/s). At M3 each decode_1 row loses ~20 us: decode_1 load -16%, ~+2.7% headline"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: Marlin pair 85.1 / 85.4 us for ~6 local experts (~179 MB, floor ~37 us). M2 <= 80 us; M3 fsi <= 55 us: ~-30 us per row, verify_1 load -16..-19%, ~+3% headline. Risk: ~1.3 M fp32 red.v4 at KC=64 (half at KC=128); the selector picks KC"
  - shapes: "*load=decode_16 *"
    direction: flat
    why: "control: T=16 outside the gate"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: T=32 outside the gate"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "control: T=128 outside the gate"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "control: T=256 outside the gate"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
prior: "moe-bs1-fused-persistent-port: that design kept the w13->w2 dependency across CTAs with grid-wide flags and spins (bimodal in moe, 50.9 vs 48.4 us) and an mma.sync mainloop; this one has no cross-unit dependency at all (each unit owns an I-chunk end to end and adds partial y with red.add) and reuses moe's judge-proven TMA ring + register-A wgmma N16. Also differs from moe cuda-nvfp4-wgmma-rega-tiny-m-bucket16 (split-K, phase boundary kept, 64 w13 N-tiles could not fill 132 SMs: tied 46.0 vs 45.5 us at T=1)"
rewrite_stage: 1
replaces: Marlin w13 + act_and_mul + Marlin w2 (w2 use_atomic_add at T=1) + _post_norm_combine_kernel at 1 <= tokens <= 8 (decode_1, verify_1, both layers)
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: Engine A replaced Marlin at T<=8"
---

# fsi-moe-tiny-t-decode-port

## Change

Stage 1 of a split-I fused MoE for decode at 1 <= tokens <= 8 (DIAGNOSIS action 1; decode port of moe's queued cuda-fused-split-i-moe-tiny-t, which has the full kernel spec: read it). Every other T runs master byte for byte. WHERE: master's dense chain at T <= 8 calls _experts(xn, tw, lids, state, align=align) with kernel R's outputs (xn bf16 [T, 4096] pre-expert-normed, local ids int32 -1 for remote, tw fp32, align = (sorted_ids, block_experts, n_padded) at block_size_m 8). At T <= 8 an expert has at most T <= 8 routes (a token's 8 ids are distinct), so every active local expert is exactly ONE 8-row block of R's align: R's align is already the unit list and no extra align launch is needed. Add _fsi_experts(xn, tw, lids, align, state) with _experts' exact output contract (out bf16 [T, 4096]) and call it instead of _experts when 1 <= tokens <= 8 and this state's flag for the T class is on; the Marlin bank stays for every other T and as the fallback. MECHANISM: y = W2 . (silu(W1g x) * (W1u x)) is linear in the I-columns of h, so a work unit u = (active block b, I-chunk j of KC columns) computes h[:, j] over the full K = 4096, applies silu * up, rounds h to bf16 into shared memory, and immediately streams W2[:, j] (all 4096 H rows) into an fp32 per-state scratch y32 with red.global.add.v4.f32 for the real routes only. No unit waits on another (no flags, spins, grid barrier or h round trip), so 2 active experts give 128 units at KC=64 and 6 give 384, which fills 132 SMs where Marlin's per-expert tiling does not. KERNEL: new module muse_decode_fsi_v1 built from muse_moe_w4a16/submission.py at moe BEST commit 64f18b03 (_WG_SRC_HEAD / _WG_SRC_BODY: TMA mbarrier ring, producer warpgroup setmaxnreg.dec 40, two consumer warpgroups inc 232, register-A wgmma m64n16k16 = the judge-proven N16 bucket, gated silu epilogue, stmatrix w2 epilogue, wg_gather_kernel); fsi_kernel is persistent (grid = SM count, u = blockIdx.x, += gridDim.x), one flat stage sequence per unit through one ring (STAGES 4): 32 w13 stages (q13 tile j, K = 4096) then 16 w2 stages (the j-th 64-k record of each of the 32 q2 row tiles), so w2 bytes are in flight while the w13 epilogue runs; the w2 B operand is the 16 x 64 bf16 h tile the w13 epilogue wrote in the B-slab swizzle (rows 8..15 zero). WEIGHTS: build moe's lossless wg bank for the 32 local experts in prepare (q13 gated 128-row tiles = 64 gate + 64 up h-columns, q2 128-row tiles, 4608-byte records per 64-k block; copy _wg_nibbles / _wg_pack_nibbles and the bank builder verbatim) and print its size (~0.9 GB). GATHER: wg_gather into a per-state 16-row-per-block slab (8 real + 8 zero rows). COMBINE: one Triton kernel with _post_norm_combine_kernel's formula (out[t] = sum over local k of tw * rmsnorm(y * (1 + post_norm_w)), POST_NORM_EPS, fp32) reading y32 rows t*8+k and writing zeros back into every row it read (zero-after-read keeps y32 zero for the next replay; y32 fp32 [64, 4096] per state, zeroed once in prepare). Launches per call: gather, fsi_kernel, combine (master: Marlin w13, act_and_mul, Marlin w2, post-norm). KC = 64 and KC = 128 instantiations; the selector picks per class. CLASSES {1} and {2..8}. CORRECTNESS: at T in 1, 2, 4, 7, 8 against an fp32-dequant reference, bound max(2x master's own error vs that reference, 1e-2) aggregate and 5x per row (LESSONS: never a fixed bound against another approximate kernel), plus two-state refilled replays proving y32 returns to zero. SELECTOR per class: master's whole expert block (xn -> out, w2 atomic included at T=1) vs gather + fsi + combine; print '[fsi] PATH CHOSEN class=<c> KC=<k>: fsi X us vs master Y us (zipf), X' vs Y' (random); fsi kernel Z us, W TB/s of weight bytes'. The first measurement is an okc check, never a scored submit. STAGES: s1 this file; s2 fsi-moe-midt-decode-port (9 <= T <= 32); s3 tuning from the s1 log (N8 m64n8k16 at T=1, a third consumer warpgroup, ring depth). Fallback at every stage: the class stays on Marlin. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; NVFP4 weights stay NVFP4 (lossless repack allowed, dequant in registers only), BF16 stays BF16; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, TMA map, scratch tensor and ticket lives in the state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), -std=c++20 -O3 -lineinfo, a renamed load_inline module, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL, plain bulk copies with no createpolicy / L2::cache_hint operand, a numRegs / localSizeBytes / smem print per instantiation (a spill line is a bug: fix before spending the check), an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states; force every bucket/class (zipf-skewed, random-local, one-expert, all-remote) and name the first non-finite tensor. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x on the worse of the two routings, and assert the enabled choice equals the printed winner. Print compile seconds per module and cumulative prepare() wall time; prepare must stay under 270 s.

## Expected

- `*load=decode_1 *` faster: replaced: Marlin pair 42.5 / 42.6 us + act_and_mul ~1.5 + post-norm ~2.9 (profile a903b9d0, sliding/full) for ~2 local experts (~57 MB) vs a ~11.4 us floor; Marlin runs ~1.3 TB/s because 1-2 experts' N-tiles cannot fill 132 SMs. M1: PATH CHOSEN = FSI for class {1} with the self-check passing. M2: gather+fsi+combine <= 0.9x master's block (<= ~42 us). M3: fsi kernel <= 22 us (2.6 TB/s). At M3 each decode_1 row loses ~20 us: decode_1 load -16%, ~+2.7% headline
- `*load=verify_1 *` faster: T=8: Marlin pair 85.1 / 85.4 us for ~6 local experts (~179 MB, floor ~37 us). M2 <= 80 us; M3 fsi <= 55 us: ~-30 us per row, verify_1 load -16..-19%, ~+3% headline. Risk: ~1.3 M fp32 red.v4 at KC=64 (half at KC=128); the selector picks KC
- `*load=decode_16 *` flat: control: T=16 outside the gate
- `*load=decode_32 *` flat: control: T=32 outside the gate
- `*load=verify_16 *` flat: control: T=128 outside the gate
- `*load=verify_32 *` flat: control: T=256 outside the gate

## Risk

Headline: about +6.5% if both classes reach M3, about +3% at half that; far above the 1.24% margin. Weakest links: (1) the dequant consumer: moe's NVFP4 kernels stream 1.5-2.1 TB/s (LESSONS), so M3 needs the consumer at ~2.6 TB/s; M2 needs only parity-plus because the phase boundary, act_and_mul, the cache1/cache2 round trips and one launch disappear. (2) fp32 red throughput at verify_1. (3) Build: a ~1,000-line wgmma TU cannot be compiled here; a CUDA fault in the canary poisons the context (LESSONS), so the canary runs a single tiny unit first with prints. (4) Nondeterministic fp32 summation order: fine for the judge's 3-15% bounds; print run-to-run spread. The first measurement is an okc check.

## Milestones

Rewrite stage 1 of `Marlin w13 + act_and_mul + Marlin w2 (w2 use_atomic_add at T=1) + _post_norm_combine_kernel at 1 <= tokens <= 8 (decode_1, verify_1, both layers)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
