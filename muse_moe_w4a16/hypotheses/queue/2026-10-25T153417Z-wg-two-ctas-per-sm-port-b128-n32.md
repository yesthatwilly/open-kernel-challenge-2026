---
problem: muse_moe_w4a16
slug: wg-two-ctas-per-sm-port-b128-n32
created: 2026-10-25T153417Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Port of prefill's experts-rega-low-class-two-ctas-per-sm (done/neutral, gained 3.4% on sliding
  turn_1, job in
  muse_prefill_layer/hypotheses/done/neutral/experts-rega-low-class-two-ctas-per-sm.*) to moe's
  B128 bucket only (128 <= T <= 255: verify_16 bs=128 and the skewed bs=128 test). B256, T < 128
  and T > 512 keep master byte for byte. MECHANISM: master's wg_kernel runs one 384-thread CTA per
  SM (1 producer + 2 consumer warpgroups), so each scheduler has only 2 consumer warps whose
  dequant -> 8 wgmma -> wait<0> chain is serial; moe already ruled out bytes in flight
  (b128-n32-deep-ring: STAGES 7/10 flat). At T=128 almost every chunk is N16 or N32 (~7.5 local
  routes per expert on average), so a 32-row slab is enough and two CTAs fit per SM. CHANGE: (1)
  template wg_kernel on NMAX (slab rows), STG (ring stages) and OCC (__launch_bounds__(384, OCC));
  master's instantiations stay <NMAX=128, STG=4, OCC=1> byte for byte. (2) Add V32 = <NMAX=32,
  STG=4, OCC=2> for both kernels (w13+silu, w2): on-device chunk cut every 2 blocks of 16 rows
  instead of 8 (CH = NMAX/16; a hot expert becomes several chunks that re-read its weights,
  adjacent in item order so from L2), buckets above N32 not instantiated, epilogue staging NMAX
  rows. Smem per CTA: 1024 + 4 x (9216 + 2 x 32 x 128) + epilogue, two CTAs plus reserves <= 228
  KB, static_assert it. (3) Registers at OCC=2: numRegs exactly 80 (launch_bounds(384, 2)),
  producer setmaxnreg.dec 24, consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 <= 384 x 80).
  Before any launch prepare() reads cudaFuncGetAttributes and
  cudaOccupancyMaxActiveBlocksPerMultiprocessor: numRegs == 80, localSizeBytes == 0 and 2 CTAs/SM,
  else V32 is OFF (setmaxnreg.inc hangs if the pool is short: never launch an unchecked
  instantiation). (4) grid = 2 x SM count for V32, stored in the state next to wg_grid. (5) Run
  master's s5 bucket canary (N16/N32), self-check and three-state copy check on V32, then extend
  master's _wg_select B128 row to three candidates (Marlin, master wgmma, V32), same cold-L2 graph
  A/B, margin 1.02x on the judge-like routing, assert the choice equals the printed winner, and
  print PATH CHOSEN B128 with all three timings plus each kernel's event time.
  state['wg_b128_variant'] (int) selects; kernel_fn branches on shape and that flag only. BUILD
  (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only
  -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under
  __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy /
  L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes /
  smem (a spill line is a bug: fix before spending the check). No module-global device pointer,
  tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded
  per call from the tensor passed in that call (master's s5 rule; the judge prepares two states
  and copies them by cloning tensors). First launch in-process after a small synthetic canary,
  then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first
  measurement is an okc check, never a scored submit.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "verify_16 runs Marlin today (selector: Marlin full path 368 us vs wgmma 457 us in prepare, judged Marlin pair 296-306 us). Flipping needs V32 ~1.27x faster than master's wgmma kernels; two CTAs/SM doubled resident consumer warps and gave +3.4% on prefill's comparable class, so a flip is uncertain: the PATH CHOSEN B128 line and the V32 event times are the result either way. If it flips, shape 314 -> <= 300 us"
  - shapes: bs=256 *
    direction: flat
    why: "control: B256 keeps master's wgmma instantiation"
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# wg-two-ctas-per-sm-port-b128-n32

## Change

Port of prefill's experts-rega-low-class-two-ctas-per-sm (done/neutral, gained 3.4% on sliding turn_1, job in muse_prefill_layer/hypotheses/done/neutral/experts-rega-low-class-two-ctas-per-sm.*) to moe's B128 bucket only (128 <= T <= 255: verify_16 bs=128 and the skewed bs=128 test). B256, T < 128 and T > 512 keep master byte for byte. MECHANISM: master's wg_kernel runs one 384-thread CTA per SM (1 producer + 2 consumer warpgroups), so each scheduler has only 2 consumer warps whose dequant -> 8 wgmma -> wait<0> chain is serial; moe already ruled out bytes in flight (b128-n32-deep-ring: STAGES 7/10 flat). At T=128 almost every chunk is N16 or N32 (~7.5 local routes per expert on average), so a 32-row slab is enough and two CTAs fit per SM. CHANGE: (1) template wg_kernel on NMAX (slab rows), STG (ring stages) and OCC (__launch_bounds__(384, OCC)); master's instantiations stay <NMAX=128, STG=4, OCC=1> byte for byte. (2) Add V32 = <NMAX=32, STG=4, OCC=2> for both kernels (w13+silu, w2): on-device chunk cut every 2 blocks of 16 rows instead of 8 (CH = NMAX/16; a hot expert becomes several chunks that re-read its weights, adjacent in item order so from L2), buckets above N32 not instantiated, epilogue staging NMAX rows. Smem per CTA: 1024 + 4 x (9216 + 2 x 32 x 128) + epilogue, two CTAs plus reserves <= 228 KB, static_assert it. (3) Registers at OCC=2: numRegs exactly 80 (launch_bounds(384, 2)), producer setmaxnreg.dec 24, consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 <= 384 x 80). Before any launch prepare() reads cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor: numRegs == 80, localSizeBytes == 0 and 2 CTAs/SM, else V32 is OFF (setmaxnreg.inc hangs if the pool is short: never launch an unchecked instantiation). (4) grid = 2 x SM count for V32, stored in the state next to wg_grid. (5) Run master's s5 bucket canary (N16/N32), self-check and three-state copy check on V32, then extend master's _wg_select B128 row to three candidates (Marlin, master wgmma, V32), same cold-L2 graph A/B, margin 1.02x on the judge-like routing, assert the choice equals the printed winner, and print PATH CHOSEN B128 with all three timings plus each kernel's event time. state['wg_b128_variant'] (int) selects; kernel_fn branches on shape and that flag only. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit.

## Expected

- `bs=128 *` faster: verify_16 runs Marlin today (selector: Marlin full path 368 us vs wgmma 457 us in prepare, judged Marlin pair 296-306 us). Flipping needs V32 ~1.27x faster than master's wgmma kernels; two CTAs/SM doubled resident consumer warps and gave +3.4% on prefill's comparable class, so a flip is uncertain: the PATH CHOSEN B128 line and the V32 event times are the result either way. If it flips, shape 314 -> <= 300 us
- `bs=256 *` flat: control: B256 keeps master's wgmma instantiation
- `bs=1 *` flat: control
- `bs=32 *` flat: control

## Risk

Not a duplicate of the queued pingpong-consumers / producer-scale-expand (one CTA, consumer schedule / instruction count) or b256-n64-deep-ring (B256 only, one CTA, deeper ring); this adds occupancy at B128 only. It is the one lever prefill measured as a gain on the same kernel. If pingpong lands first, rebase and keep both selectable; the selector then compares them.
