---
problem: muse_prefill_layer
slug: experts-rega-low-class-two-ctas-per-sm
created: 2026-10-25T125554Z
base: master
area: "w13 + w2 expert GEMMs at 200 < T <= 1024 (turn_1, rega low class): two resident CTAs per SM with a 64- or 32-route slab, to hide consumer dequant latency"
change: >
  Rewrite stage 3 of the rega line, built on master a1a5497d
  (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX
  = 1024, i.e. turn_1 T=753 and the T=300/753 tests). 'high' (turn_4) and 'ft' (first_turn) keep
  master's instantiations byte for byte. WHY: turn_1 is the weakest load (1.348x, against 1.428x
  for turn_4 and 1.444x for first_turn), and rega is 48% of it. In job 07a069bc69ba42a8 the low
  class ran rega_kernel<true,false> (w13+silu) at 282.4/257.8 us and rega_kernel<false,false> (w2)
  at 146.0/125.2 us. At T=753 every one of the 32 local experts is touched, so the kernels stream
  ~604 MB (w13) and ~302 MB (w2) of NVFP4 codes+scales, about 2.1 TB/s against 4.8 (floor ~126 +
  63 us). muse_moe_w4a16 has already ruled out bytes in flight
  (cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: STAGES 7/10 flat, 1.83 -> 1.87 TB/s) and
  per-warpgroup dequant/MMA overlap (s3 variant B, double-buffered A, no faster). LESSONS line 183
  says 'look at the consumer/dequant'. What remains is latency hiding. One 384-thread CTA per SM
  gives each scheduler only 2 consumer warps, and their dequant -> wgmma -> wait<0> chain is
  serial. The CHANGE doubles resident consumer warps, with two CTAs per SM. (1) Template the
  kernel on the slab width NMAX and pipeline depth STG, as the moe branch did (git show
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring:muse_moe_w4a16/submission.py, its
  Geo<NMAX,STG>, chunk cut CH = NMAX/16 blocks, consumer buckets above NMAX not instantiated). Add
  an occupancy parameter OCC (1 or 2) that sets __launch_bounds__(THREADS, OCC). Master's
  instantiations are <G, BIG, NMAX=128, STG=4, OCC=1> and must stay identical. (2) Add two
  low-class candidates for each G: V64 = <G, false, 64, 3, 2> and V32 = <G, false, 32, 4, 2>.
  Per-CTA smem: V64 dynamic 1024 + 3 x (9216 + 2 x 64 x 128) + 64 x PITCH (w2 95232 B), V32
  dynamic 1024 + 4 x 17408 + 32 x PITCH (w2 79360 B), each plus ~14.3 KB static. Two CTAs plus the
  1 KB per-CTA reserve fit in 228 KB; static_assert it. The epilogue staging is NMAX rows, not
  128. (3) Registers at OCC=2. Compile so that numRegs is exactly 80 (__launch_bounds__(384, 2)),
  producer setmaxnreg.dec 24 and consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 = 29696 <= 384
  x 80). Before any launch, prepare() reads cudaFuncGetAttributes and
  cudaOccupancyMaxActiveBlocksPerMultiprocessor for each candidate. It requires numRegs == 80,
  localSizeBytes == 0 and 2 CTAs/SM; otherwise that candidate is OFF. setmaxnreg.inc hangs if the
  pool is short (see the comment near master line 3610), so never launch an unchecked candidate.
  Print regs, local bytes, smem and CTAs/SM per instantiation. (4) Launch with grid = 2 x SM count
  for the OCC=2 candidates. Store this in the state as an int next to rega_grid, never as a module
  global. Keep the kernel persistent, with items strided by gridDim.x as now. (5) Dispatch:
  _experts_rega passes a variant int read from state['rega_low_variant'] (0 = master, 1 = V64, 2 =
  V32), only when T <= REGA_LOW_MAX. rega_gemm selects the instantiation by (which, variant,
  block-table class) on the host from shapes and that int. The judge prepares once and copies the
  state by cloning tensors, so nothing may depend on data_ptr equality; maps are still encoded per
  call. (6) Self-check each candidate with master's existing low-class machinery: eager check
  against the fp32 reference at T=300 and 753 (random, skewed, one-expert), error word 0, launch
  markers, finite output, then the A/B/C copy/replay check with refilled inputs. Any miss turns
  only that candidate OFF. (7) Gate at T=753 against master's low class (rega NMAX128), using
  cold-L2 graph medians, >= 9 interleaved alternations and the empty-graph baseline subtracted.
  Take the WORSE ratio of two routings: uniform random, and a popularity-skewed routing with about
  1.8x the random local-route count (the judge's soft first_turn routing had ~29k local routes
  against ~16k random at T=16384). ON only at <= 0.95. Of passing candidates, pick the lower
  median. Print per candidate the w13 and w2 kernel-graph us and achieved TB/s (touched-expert
  bytes / time). Print '[rega] PATH CHOSEN low: master|V64|V32' among the final prepare() lines.
  kernel_fn rules: no host sync, no prints, no try/except, no per-call mutable state, dispatch on
  T and prepare-set ints only. NVFP4 weights stay NVFP4 (register dequant only), BF16 activations,
  fp32 accumulation. Build exactly as master (TORCH_CUDA_ARCH_LIST=9.0a, -gencode
  arch=compute_90a,code=sm_90a, wgmma/setmaxnreg under __CUDA_ARCH_FEAT_SM90_ALL). Use a new
  extension name (muse_prefill_rega_v3), keeping master's v2 -> v1 build fallback chain.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low-class rega w13+w2 428/383 us at ~2.1 TB/s (floor ~189 us on 906 MB) -> ~290-330 us if two resident CTAs lift it to ~3 TB/s: sliding turn_1 777 -> ~660-700 us (-10 to -15%), full 1136 -> ~1030-1060"
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: high class keeps master NMAX128/STG4/OCC1 byte-identical
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: ft class keeps master instantiation byte-identical
risk: high
depends_on: experts-turn-rega-s4-first-turn-n128
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024
review: required
status: queued
---

# experts-rega-low-class-two-ctas-per-sm

## Change

Rewrite stage 3 of the rega line, built on master a1a5497d (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX = 1024, i.e. turn_1 T=753 and the T=300/753 tests). 'high' (turn_4) and 'ft' (first_turn) keep master's instantiations byte for byte. WHY: turn_1 is the weakest load (1.348x, against 1.428x for turn_4 and 1.444x for first_turn), and rega is 48% of it. In job 07a069bc69ba42a8 the low class ran rega_kernel<true,false> (w13+silu) at 282.4/257.8 us and rega_kernel<false,false> (w2) at 146.0/125.2 us. At T=753 every one of the 32 local experts is touched, so the kernels stream ~604 MB (w13) and ~302 MB (w2) of NVFP4 codes+scales, about 2.1 TB/s against 4.8 (floor ~126 + 63 us). muse_moe_w4a16 has already ruled out bytes in flight (cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: STAGES 7/10 flat, 1.83 -> 1.87 TB/s) and per-warpgroup dequant/MMA overlap (s3 variant B, double-buffered A, no faster). LESSONS line 183 says 'look at the consumer/dequant'. What remains is latency hiding. One 384-thread CTA per SM gives each scheduler only 2 consumer warps, and their dequant -> wgmma -> wait<0> chain is serial. The CHANGE doubles resident consumer warps, with two CTAs per SM. (1) Template the kernel on the slab width NMAX and pipeline depth STG, as the moe branch did (git show hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring:muse_moe_w4a16/submission.py, its Geo<NMAX,STG>, chunk cut CH = NMAX/16 blocks, consumer buckets above NMAX not instantiated). Add an occupancy parameter OCC (1 or 2) that sets __launch_bounds__(THREADS, OCC). Master's instantiations are <G, BIG, NMAX=128, STG=4, OCC=1> and must stay identical. (2) Add two low-class candidates for each G: V64 = <G, false, 64, 3, 2> and V32 = <G, false, 32, 4, 2>. Per-CTA smem: V64 dynamic 1024 + 3 x (9216 + 2 x 64 x 128) + 64 x PITCH (w2 95232 B), V32 dynamic 1024 + 4 x 17408 + 32 x PITCH (w2 79360 B), each plus ~14.3 KB static. Two CTAs plus the 1 KB per-CTA reserve fit in 228 KB; static_assert it. The epilogue staging is NMAX rows, not 128. (3) Registers at OCC=2. Compile so that numRegs is exactly 80 (__launch_bounds__(384, 2)), producer setmaxnreg.dec 24 and consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 = 29696 <= 384 x 80). Before any launch, prepare() reads cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor for each candidate. It requires numRegs == 80, localSizeBytes == 0 and 2 CTAs/SM; otherwise that candidate is OFF. setmaxnreg.inc hangs if the pool is short (see the comment near master line 3610), so never launch an unchecked candidate. Print regs, local bytes, smem and CTAs/SM per instantiation. (4) Launch with grid = 2 x SM count for the OCC=2 candidates. Store this in the state as an int next to rega_grid, never as a module global. Keep the kernel persistent, with items strided by gridDim.x as now. (5) Dispatch: _experts_rega passes a variant int read from state['rega_low_variant'] (0 = master, 1 = V64, 2 = V32), only when T <= REGA_LOW_MAX. rega_gemm selects the instantiation by (which, variant, block-table class) on the host from shapes and that int. The judge prepares once and copies the state by cloning tensors, so nothing may depend on data_ptr equality; maps are still encoded per call. (6) Self-check each candidate with master's existing low-class machinery: eager check against the fp32 reference at T=300 and 753 (random, skewed, one-expert), error word 0, launch markers, finite output, then the A/B/C copy/replay check with refilled inputs. Any miss turns only that candidate OFF. (7) Gate at T=753 against master's low class (rega NMAX128), using cold-L2 graph medians, >= 9 interleaved alternations and the empty-graph baseline subtracted. Take the WORSE ratio of two routings: uniform random, and a popularity-skewed routing with about 1.8x the random local-route count (the judge's soft first_turn routing had ~29k local routes against ~16k random at T=16384). ON only at <= 0.95. Of passing candidates, pick the lower median. Print per candidate the w13 and w2 kernel-graph us and achieved TB/s (touched-expert bytes / time). Print '[rega] PATH CHOSEN low: master|V64|V32' among the final prepare() lines. kernel_fn rules: no host sync, no prints, no try/except, no per-call mutable state, dispatch on T and prepare-set ints only. NVFP4 weights stay NVFP4 (register dequant only), BF16 activations, fp32 accumulation. Build exactly as master (TORCH_CUDA_ARCH_LIST=9.0a, -gencode arch=compute_90a,code=sm_90a, wgmma/setmaxnreg under __CUDA_ARCH_FEAT_SM90_ALL). Use a new extension name (muse_prefill_rega_v3), keeping master's v2 -> v1 build fallback chain.

## Expected

- `*load=turn_1*` faster: low-class rega w13+w2 428/383 us at ~2.1 TB/s (floor ~189 us on 906 MB) -> ~290-330 us if two resident CTAs lift it to ~3 TB/s: sliding turn_1 777 -> ~660-700 us (-10 to -15%), full 1136 -> ~1030-1060
- `*load=turn_4*` unchanged: high class keeps master NMAX128/STG4/OCC1 byte-identical
- `*load=first_turn*` unchanged: ft class keeps master instantiation byte-identical

## Risk

Risk: high, but gated. A null, where two CTAs/SM give no lift, is itself a finding: it would mean the limiter is in the per-item path (decode_item and table scans, epilogue, the single producer thread's TMA issue), not latency. It is printed as per-candidate TB/s. Ways it can fail: the register cap of 80 spills (then that candidate is OFF, printed); NMAX=32 splits experts with more than 32 routes into extra chunks that re-read the weight tile (from L2 when adjacent; V64 avoids most of this); build time for 4 extra instantiations (master prepare is about 115-130 s; budget 200 s for rega, and ft has its own 150 s start deadline, so print phase timings and skip the new candidates first if the budget runs short, never ft). Attribution: the low-class rows get new template names (rega_kernel<G,false,64,3,2> or <G,false,32,4,2>) on turn_1 only. turn_4 and first_turn rows must stay within ~2-3% drift. Read the turn_1 kernel rows, not the headline; LESSONS warns that turn_1 drift is large (up to 7-10% on untouched kernels). Independent of align-two-phase-all-t-rega (different area and kernels). Both build on master.

## Milestones

Rewrite stage 3 of `rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
