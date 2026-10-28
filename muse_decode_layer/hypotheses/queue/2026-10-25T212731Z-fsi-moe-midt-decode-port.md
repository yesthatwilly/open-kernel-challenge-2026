---
problem: muse_decode_layer
slug: fsi-moe-midt-decode-port
created: 2026-10-25T212731Z
base: master
area: w13 marlin + silu-mul + w2 marlin + post-expert combine (9 <= T <= 32)
change: >
  Stage 2 of fsi-moe-tiny-t-decode-port (DIAGNOSIS action 2; decode port of moe's queued
  cuda-fused-split-i-moe-midt-cluster-reduce: read it for the kernel spec). Build on stage 1's
  branch and reuse its module, bank, gather, combine and selector; extend the gate to a SEPARATE
  class pair {9..16} and {17..32} with its own bound constants FSI_MID_MIN_TOKENS = 9 /
  FSI_MID_MAX_TOKENS = 32 and its own per-state flags (LESSONS 2026-10-25 stacking note: never
  derive this gate from DL_MAX_TOKENS or stage 1's constant). WHAT IS NEW: (1) tokens on N use the
  judge-proven N32 bucket (wgmma m64n32k16); at T <= 32 an expert has at most 32 routes, but R's
  align uses block_size_m 8 (T=16) or 16 (T=32), so one expert can span several blocks: derive the
  expert list on device from block_experts (consecutive equal entries = one expert) in the gather
  launch, and gather ALL of an expert's routes into one 32-row slab, so every weight byte is read
  exactly once per call whatever the routing (Marlin re-streams an expert per m-block). (2) KC =
  128 or 256 (2 or 4 adjacent q13 tiles per unit; 16-32 chunks x ~9-20 active experts = 144-640
  units), picked by the selector. (3) The partial-y reduction goes through a thread-block cluster
  CL = 4 (cudaLaunchKernelEx with a cluster dimension; grid rounded down to a multiple of 4;
  co-residency checked with cudaOccupancyMaxActiveClusters before the first launch, else the
  no-cluster stage-1 reduction is used): the four CTAs of a cluster take four I-chunks of the same
  expert, each writes its w2 partial for one 128-row H tile to its own smem (128 x 32 fp32),
  cluster.sync, rank r sums rows [32r, 32r+32) of the four partials over DSMEM (mapa +
  ld.shared::cluster, fixed rank order) and issues red.global.add.v4.f32 for real routes only; the
  producer warpgroup does not join the cluster barrier, so the ring keeps streaming. That cuts
  global reds 4x. (4) y32 per state grows to fp32 [256, 4096]; the stage-1 combine
  (zero-after-read) also replaces SGLang's fused_post_expert_norm_weighted_sum at T=32.
  CORRECTNESS at T = 9, 16, 17, 32 (and T=33 proven untouched), routings zipf, random-local, one
  expert with T routes, all-remote, fp32-dequant reference, bound max(2x master's error, 1e-2)
  aggregate and 5x per row, two-state refilled replays. SELECTOR per class vs master's whole
  expert block (R's align -> out), zipf and random, worse-of-two >= 1.02x; PATH CHOSEN per class
  with KC, cluster on/off, fsi event time and TB/s. If marlin-bsm8-t17-t32-w13-atomic-t1 landed
  first, time the {17..32} class against Marlin at block 8 (the incumbent as routed). First
  measurement is an okc check. Fallback: the class stays on Marlin. HARD RULES (every path):
  CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow and
  no per-call mutable Python state; gates read config/shape and this state's own prepare() flags
  only; nothing cached from inputs; NVFP4 weights stay NVFP4 (lossless repack allowed, dequant in
  registers only), BF16 stays BF16; no child processes. The judge prepares TWO state copies and
  rotates them: every device pointer, TMA map, scratch tensor and ticket lives in the state, never
  in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only
  -gencode arch=compute_90a,code=sm_90a), -std=c++20 -O3 -lineinfo, a renamed load_inline module,
  wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL, plain bulk copies with no createpolicy
  / L2::cache_hint operand, a numRegs / localSizeBytes / smem print per instantiation (a spill
  line is a bug: fix before spending the check), an in-process canary with synchronize +
  'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback:
  <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with
  refilled inputs while alternating two states; force every bucket/class (zipf-skewed,
  random-local, one-expert, all-remote) and name the first non-finite tensor. Selectors time new
  vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps,
  empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS
  by >= 1.02x on the worse of the two routings, and assert the enabled choice equals the printed
  winner. Print compile seconds per module and cumulative prepare() wall time; prepare must stay
  under 270 s.
expected:
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "Marlin pair 94.9 / 73.6 us (sliding/full, profile a903b9d0) vs moe floor ~52.5 us. M1 PATH CHOSEN = FSI for {9..16}; M2 replaced block <= 0.9x master; M3 <= 1.4x floor (~75 us incl. combine): -20..-25 us per row, decode_16 load -8..-10%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "Marlin pair 122.2 / 113.4 us vs floor ~70.8 us; decode_32 is the weakest load (1.206x paired), Marlin 41% of it. M3 <= ~100 us: -20..-30 us per row, decode_32 load -7..-10%. Both classes at M3: ~+3..+3.5% headline"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: "stage 1's class (or Marlin): unchanged by this stage"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "stage 1's class: unchanged by this stage"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "control: T=128"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "control: T=256"
risk: high
depends_on: fsi-moe-tiny-t-decode-port
priority: 2
kind: rewrite
controls_for: null
prior: "marlin-two-stream-halves-port: that split the experts across two streams but kept Marlin's per-m-block tiling and the w13->w2 phase boundary (moe 0.968-0.980 at T 8-32); this removes the boundary and streams each expert's weights exactly once per call, with a cluster DSMEM pre-reduction of the partial y"
rewrite_stage: 2
replaces: Marlin w13 + act_and_mul + Marlin w2 + post-expert combine (_post_norm_combine_kernel at T<32, SGLang fused_post_expert_norm_weighted_sum at T=32) at 9 <= tokens <= 32 (decode_16, decode_32)
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: Engine A v4c at T=9..32"
---

# fsi-moe-midt-decode-port

## Change

Stage 2 of fsi-moe-tiny-t-decode-port (DIAGNOSIS action 2; decode port of moe's queued cuda-fused-split-i-moe-midt-cluster-reduce: read it for the kernel spec). Build on stage 1's branch and reuse its module, bank, gather, combine and selector; extend the gate to a SEPARATE class pair {9..16} and {17..32} with its own bound constants FSI_MID_MIN_TOKENS = 9 / FSI_MID_MAX_TOKENS = 32 and its own per-state flags (LESSONS 2026-10-25 stacking note: never derive this gate from DL_MAX_TOKENS or stage 1's constant). WHAT IS NEW: (1) tokens on N use the judge-proven N32 bucket (wgmma m64n32k16); at T <= 32 an expert has at most 32 routes, but R's align uses block_size_m 8 (T=16) or 16 (T=32), so one expert can span several blocks: derive the expert list on device from block_experts (consecutive equal entries = one expert) in the gather launch, and gather ALL of an expert's routes into one 32-row slab, so every weight byte is read exactly once per call whatever the routing (Marlin re-streams an expert per m-block). (2) KC = 128 or 256 (2 or 4 adjacent q13 tiles per unit; 16-32 chunks x ~9-20 active experts = 144-640 units), picked by the selector. (3) The partial-y reduction goes through a thread-block cluster CL = 4 (cudaLaunchKernelEx with a cluster dimension; grid rounded down to a multiple of 4; co-residency checked with cudaOccupancyMaxActiveClusters before the first launch, else the no-cluster stage-1 reduction is used): the four CTAs of a cluster take four I-chunks of the same expert, each writes its w2 partial for one 128-row H tile to its own smem (128 x 32 fp32), cluster.sync, rank r sums rows [32r, 32r+32) of the four partials over DSMEM (mapa + ld.shared::cluster, fixed rank order) and issues red.global.add.v4.f32 for real routes only; the producer warpgroup does not join the cluster barrier, so the ring keeps streaming. That cuts global reds 4x. (4) y32 per state grows to fp32 [256, 4096]; the stage-1 combine (zero-after-read) also replaces SGLang's fused_post_expert_norm_weighted_sum at T=32. CORRECTNESS at T = 9, 16, 17, 32 (and T=33 proven untouched), routings zipf, random-local, one expert with T routes, all-remote, fp32-dequant reference, bound max(2x master's error, 1e-2) aggregate and 5x per row, two-state refilled replays. SELECTOR per class vs master's whole expert block (R's align -> out), zipf and random, worse-of-two >= 1.02x; PATH CHOSEN per class with KC, cluster on/off, fsi event time and TB/s. If marlin-bsm8-t17-t32-w13-atomic-t1 landed first, time the {17..32} class against Marlin at block 8 (the incumbent as routed). First measurement is an okc check. Fallback: the class stays on Marlin. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; NVFP4 weights stay NVFP4 (lossless repack allowed, dequant in registers only), BF16 stays BF16; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, TMA map, scratch tensor and ticket lives in the state, never in a module global. Every new CUDA build is sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a or only -gencode arch=compute_90a,code=sm_90a), -std=c++20 -O3 -lineinfo, a renamed load_inline module, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL, plain bulk copies with no createpolicy / L2::cache_hint operand, a numRegs / localSizeBytes / smem print per instantiation (a spill line is a bug: fix before spending the check), an in-process canary with synchronize + 'launching'/'ok' prints after each first launch, and on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line that keeps master for that cell. Self-check by capturing once and replaying with refilled inputs while alternating two states; force every bucket/class (zipf-skewed, random-local, one-expert, all-remote) and name the first non-finite tensor. Selectors time new vs incumbent by interleaved CUDA-graph replay (256 MB L2 flush before each, >= 41 reps, empty-graph baseline subtracted), print both medians, enable a cell only when the new path WINS by >= 1.02x on the worse of the two routings, and assert the enabled choice equals the printed winner. Print compile seconds per module and cumulative prepare() wall time; prepare must stay under 270 s.

## Expected

- `*load=decode_16 *` faster: Marlin pair 94.9 / 73.6 us (sliding/full, profile a903b9d0) vs moe floor ~52.5 us. M1 PATH CHOSEN = FSI for {9..16}; M2 replaced block <= 0.9x master; M3 <= 1.4x floor (~75 us incl. combine): -20..-25 us per row, decode_16 load -8..-10%
- `*load=decode_32 *` faster: Marlin pair 122.2 / 113.4 us vs floor ~70.8 us; decode_32 is the weakest load (1.206x paired), Marlin 41% of it. M3 <= ~100 us: -20..-30 us per row, decode_32 load -7..-10%. Both classes at M3: ~+3..+3.5% headline
- `*load=decode_1 *` flat: stage 1's class (or Marlin): unchanged by this stage
- `*load=verify_1 *` flat: stage 1's class: unchanged by this stage
- `*load=verify_16 *` flat: control: T=128
- `*load=verify_32 *` flat: control: T=256

## Risk

Depends on stage 1 so the TU, bank, canary and combine are proven before the cluster path is added; churnsubs may claim it as soon as stage 1 has a measured check, whatever its score (rewrite rule). Weakest links: the cluster launch and DSMEM reduce are new on this judge for this module; fp32 red volume at T=32 (~250 K red.v4 per call at KC=256 with CL=4); the dequant consumer limit (~2 TB/s) still caps M3. First measurement is an okc check.

## Milestones

Rewrite stage 2 of `Marlin w13 + act_and_mul + Marlin w2 + post-expert combine (_post_norm_combine_kernel at T<32, SGLang fused_post_expert_norm_weighted_sum at T=32) at 9 <= tokens <= 32 (decode_16, decode_32)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
