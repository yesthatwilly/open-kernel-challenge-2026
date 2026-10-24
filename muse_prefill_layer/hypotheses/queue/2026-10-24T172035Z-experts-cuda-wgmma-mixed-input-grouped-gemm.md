---
problem: muse_prefill_layer
slug: experts-cuda-wgmma-mixed-input-grouped-gemm
created: 2026-10-24T172035Z
base: master
area: "w13 marlin + w2 marlin (CUDA sm_90a warp-specialized wgmma W4A16 grouped GEMM, stage 1: first_turn)"
change: >
  Replace the expert GEMM engine with a hand-written Hopper (sm_90a) mixed-input grouped GEMM in
  CUDA, compiled with torch.utils.cpp_extension.load_inline at import (CUTLASS/CuTe headers from
  the image may be used for the wgmma/TMA atoms; raw inline-PTX wgmma is fine and compiles
  faster). This is not the queued experts-dequant-grouped-mm-cutlass (that one materializes a
  per-call BF16 weight copy, held on the rules) and not another Triton variant: the NVFP4 codes
  are dequantized in registers only, as the rules and LESSONS require, and the MMA is wgmma with
  the dequantized weight tile as the register-resident A operand (wgmma.mma_async ... .bf16 RS
  form, swap-AB as in master: weight rows on M, tokens on N, activations from smem). Activations
  stay BF16 (W4A16, as SGLang's Marlin path keeps them; no FP8 activations, the rules forbid it).
  WHY THIS SUBSYSTEM: on master the expert GEMMs are 37-67% of every scored shape and far off
  their floor on every load. Local routes are about T (T*8 routes over 8 GPUs), so the w13+w2
  floor is max(905 MB of codes+scales at 4.8 TB/s = 189 us, 100.7 MFLOP per route at 989 TFLOP/s):
  turn_1 189 us (weight-stream bound), turn_4 305 us, first_turn 1.67 ms (math bound). Master runs
  752/654 us (turn_1, Marlin), 2217/2131 us (turn_4, Marlin, 7x the floor) and 8.15-8.87 ms
  (first_turn, Triton, ~200 TFLOP/s, 5x the floor). A team at 1.79x on a geomean over three loads
  must have pulled all three loads to ~1.7-1.9x; with everything else in the layer near its floor
  (dense nvjet GEMMs at ~800 TFLOP/s, norm-residual kernels at ~4 TB/s), the arithmetic only
  closes if their experts run at ~600 TFLOP/s on first_turn and turn_4 and ~3 TB/s on turn_1, i.e.
  a real tensor-core W4A16 engine. Triton cannot get there: the stage 2-5 kernels are bound by
  tl.dot's lowering (synchronous wgmma after a full in-register dequant, no producer/consumer
  split). KERNEL DESIGN (all stages): persistent grid of one CTA per SM (132), warp-specialized: 1
  producer warpgroup (setmaxnreg down to 40) and 2 consumer warpgroups (setmaxnreg up to 232).
  Work item = (token block from moe_align_block_size, weight tile of 128 rows = 64 per consumer
  WG); work list ordered weight-tile-major within each expert so consecutive items on neighbouring
  SMs reuse the weight tile from L2; items whose block expert is -1 skip the MMA (w2 still
  zero-stores them, as ZERO_NONLOCAL_W2 does today) and items past n_padded exit. Producer: weight
  codes and e4m3 scales by TMA (2-D tensor maps over the bank viewed as [E*N, K/2] uint8 and [E*N,
  K/16] e4m3) into a 4-5 stage ring with mbarriers; activation rows gathered by cp.async 16 B (row
  = sorted_id // top_k for w13, sorted_id for w2) into 128 B-swizzled smem in the canonical wgmma
  B layout, signalled with cp.async.mbarrier.arrive.noinc. Consumers per k64 step: one LDS.128 of
  codes per thread (prepare() stores the codes, losslessly permuted, in each thread's wgmma
  A-fragment order, so 16 bytes = that thread's 32 nibbles of a 64x64 tile; scales likewise as one
  LDS.64), dequant E2M1 to BF16 by bit placement or prmt (the identity master already checks),
  times the block scale (HMUL2), then wgmma async; dequant of step i+1 overlaps the wgmma of step
  i (wgmma.wait_group 1). The two consumer WGs ping-pong so the tensor core never idles on ALU.
  Epilogue: fp32 acc times global scale times 2^7 (master's SCALE_BIAS convention), bf16,
  transposed through smem (stmatrix) and stored to the output rows by sorted id (slot order),
  exactly master's output contract. Host wrapper has the signature of _nvfp4_moe_gemm (a, w, sf,
  gs, out, sorted_ids, expert_ids, n_padded, top_k, num_valid, cfg, zero_nonlocal) so it is a
  drop-in. STAGE PLAN: stage 1 (this file) = first_turn only (T > MARLIN_MAX_TOKENS, i.e.
  _experts_triton): both _nvfp4_moe_gemm calls go to the CUDA kernel with BLOCK_T 256 (wgmma
  m64n256k16; acc 128 regs/thread per consumer WG), moe_align_block_size at 256 as today;
  silu_and_mul, the post-expert weighted sum, the Marlin path at T <= 4096 and everything else
  unchanged. Stage 2 = turn_4 (T=3012, ~94 rows/expert): token tile 96 or 128 (m64n96 / m64n128),
  moe_align at that block, replaces Marlin at T <= 4096 above 1024 tokens. Stage 3 = turn_1 and
  the tests (~24 rows/expert): token tile 32 (m64n32), deeper weight ring (8 stages) so ~200 KB of
  weight bytes are in flight per SM, replaces Marlin everywhere. Stage 4 = fuse silu-mul into the
  w13 epilogue (consumer WG0 takes gate rows n0.., WG1 the matching up rows 4096+n0..; combine
  through smem) and drop cache1. Each later stage is its own --depends-on hypothesis. FALLBACK AND
  SELF-CHECK: module-level try: load_inline(...) with a compile-time log line; any exception sets
  CUDA_EXPERTS_OK = False and the code runs master's Triton kernel unchanged. In prepare(), after
  the existing _check_expert_gemms: (a) the same one-hot dequant check run through the CUDA kernel
  on the CUDA bank (every N row, one full K chunk, the -1 block and the past-n_padded block) and
  (b) _check_experts_e2e with _experts_triton routed to the CUDA kernel vs Marlin at T=8192 (rel
  L2 <= 2e-2, worst row <= 5x). Any failure prints the reason and flips the flag off for the run.
  Log exactly one line "[cuda expert gemm] ON" or "[cuda expert gemm] OFF: <reason>", plus
  regs/smem/compile seconds. Compile must fit the 15-minute run: target under 90 s; if the CuTe
  templates are slow, use inline-PTX wgmma. MILESTONES (read from the profile's top_kernels; the
  new kernel shows under its own name, e.g. _nvfp4_wgmma_grouped_kernel, 2 calls): M0 the log says
  ON and the 5 tests pass; M1 first_turn w13+w2 under 6.0 ms (beats the 8.15-8.87 ms Triton
  incumbent by 1.4x); M2 under 3.5 ms (~470 TFLOP/s); stretch 2.5 ms (~650 TFLOP/s, CUTLASS
  mixed-input class). Below M1 the stage still teaches the next one (judge it on the kernel row,
  not the headline).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Replaced: _moe_nvfp4_gemm_kernel x2 at 8873 us (full ctx0), 8324 us (sliding ctx32768, weight 138, the heaviest row of the problem), 8339 us (full ctx32768), 8149 us (full ctx65536): 37-58% of each shape. Floor for w13+w2 is the math, 16384 local routes x 100.7 MFLOP = 1.65 TFLOP = 1.67 ms at 989 TFLOP/s (the shapes' floors are 5518/5663/7710/9848 us). At the M2 target of ~3.0-3.5 ms the shapes go 15386->~10.5k, 14275->~9.4k, 18166->~13.3k, 22333->~17.5k us: ~1.5x on the first_turn load, ~1.15x on the headline geomean (1.059 -> ~1.2x); at the 2.5 ms stretch ~1.25x."
  - shapes: "*load=turn_1*"
    direction: neutral
    why: stage 1 keeps Marlin at T <= 4096 (752.4/653.9 us vs a 189 us weight-stream floor); stage 3 targets ~300 us
  - shapes: "*load=turn_4*"
    direction: neutral
    why: stage 1 keeps Marlin at T <= 4096 (2217/2131 us vs a 305 us math floor); stage 2 targets ~600-700 us
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: _moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3
review: required
status: queued
---

# experts-cuda-wgmma-mixed-input-grouped-gemm

## Change

Replace the expert GEMM engine with a hand-written Hopper (sm_90a) mixed-input grouped GEMM in CUDA, compiled with torch.utils.cpp_extension.load_inline at import (CUTLASS/CuTe headers from the image may be used for the wgmma/TMA atoms; raw inline-PTX wgmma is fine and compiles faster). This is not the queued experts-dequant-grouped-mm-cutlass (that one materializes a per-call BF16 weight copy, held on the rules) and not another Triton variant: the NVFP4 codes are dequantized in registers only, as the rules and LESSONS require, and the MMA is wgmma with the dequantized weight tile as the register-resident A operand (wgmma.mma_async ... .bf16 RS form, swap-AB as in master: weight rows on M, tokens on N, activations from smem). Activations stay BF16 (W4A16, as SGLang's Marlin path keeps them; no FP8 activations, the rules forbid it).
WHY THIS SUBSYSTEM: on master the expert GEMMs are 37-67% of every scored shape and far off their floor on every load. Local routes are about T (T*8 routes over 8 GPUs), so the w13+w2 floor is max(905 MB of codes+scales at 4.8 TB/s = 189 us, 100.7 MFLOP per route at 989 TFLOP/s): turn_1 189 us (weight-stream bound), turn_4 305 us, first_turn 1.67 ms (math bound). Master runs 752/654 us (turn_1, Marlin), 2217/2131 us (turn_4, Marlin, 7x the floor) and 8.15-8.87 ms (first_turn, Triton, ~200 TFLOP/s, 5x the floor). A team at 1.79x on a geomean over three loads must have pulled all three loads to ~1.7-1.9x; with everything else in the layer near its floor (dense nvjet GEMMs at ~800 TFLOP/s, norm-residual kernels at ~4 TB/s), the arithmetic only closes if their experts run at ~600 TFLOP/s on first_turn and turn_4 and ~3 TB/s on turn_1, i.e. a real tensor-core W4A16 engine. Triton cannot get there: the stage 2-5 kernels are bound by tl.dot's lowering (synchronous wgmma after a full in-register dequant, no producer/consumer split).
KERNEL DESIGN (all stages): persistent grid of one CTA per SM (132), warp-specialized: 1 producer warpgroup (setmaxnreg down to 40) and 2 consumer warpgroups (setmaxnreg up to 232). Work item = (token block from moe_align_block_size, weight tile of 128 rows = 64 per consumer WG); work list ordered weight-tile-major within each expert so consecutive items on neighbouring SMs reuse the weight tile from L2; items whose block expert is -1 skip the MMA (w2 still zero-stores them, as ZERO_NONLOCAL_W2 does today) and items past n_padded exit. Producer: weight codes and e4m3 scales by TMA (2-D tensor maps over the bank viewed as [E*N, K/2] uint8 and [E*N, K/16] e4m3) into a 4-5 stage ring with mbarriers; activation rows gathered by cp.async 16 B (row = sorted_id // top_k for w13, sorted_id for w2) into 128 B-swizzled smem in the canonical wgmma B layout, signalled with cp.async.mbarrier.arrive.noinc. Consumers per k64 step: one LDS.128 of codes per thread (prepare() stores the codes, losslessly permuted, in each thread's wgmma A-fragment order, so 16 bytes = that thread's 32 nibbles of a 64x64 tile; scales likewise as one LDS.64), dequant E2M1 to BF16 by bit placement or prmt (the identity master already checks), times the block scale (HMUL2), then wgmma async; dequant of step i+1 overlaps the wgmma of step i (wgmma.wait_group 1). The two consumer WGs ping-pong so the tensor core never idles on ALU. Epilogue: fp32 acc times global scale times 2^7 (master's SCALE_BIAS convention), bf16, transposed through smem (stmatrix) and stored to the output rows by sorted id (slot order), exactly master's output contract. Host wrapper has the signature of _nvfp4_moe_gemm (a, w, sf, gs, out, sorted_ids, expert_ids, n_padded, top_k, num_valid, cfg, zero_nonlocal) so it is a drop-in.
STAGE PLAN: stage 1 (this file) = first_turn only (T > MARLIN_MAX_TOKENS, i.e. _experts_triton): both _nvfp4_moe_gemm calls go to the CUDA kernel with BLOCK_T 256 (wgmma m64n256k16; acc 128 regs/thread per consumer WG), moe_align_block_size at 256 as today; silu_and_mul, the post-expert weighted sum, the Marlin path at T <= 4096 and everything else unchanged. Stage 2 = turn_4 (T=3012, ~94 rows/expert): token tile 96 or 128 (m64n96 / m64n128), moe_align at that block, replaces Marlin at T <= 4096 above 1024 tokens. Stage 3 = turn_1 and the tests (~24 rows/expert): token tile 32 (m64n32), deeper weight ring (8 stages) so ~200 KB of weight bytes are in flight per SM, replaces Marlin everywhere. Stage 4 = fuse silu-mul into the w13 epilogue (consumer WG0 takes gate rows n0.., WG1 the matching up rows 4096+n0..; combine through smem) and drop cache1. Each later stage is its own --depends-on hypothesis.
FALLBACK AND SELF-CHECK: module-level try: load_inline(...) with a compile-time log line; any exception sets CUDA_EXPERTS_OK = False and the code runs master's Triton kernel unchanged. In prepare(), after the existing _check_expert_gemms: (a) the same one-hot dequant check run through the CUDA kernel on the CUDA bank (every N row, one full K chunk, the -1 block and the past-n_padded block) and (b) _check_experts_e2e with _experts_triton routed to the CUDA kernel vs Marlin at T=8192 (rel L2 <= 2e-2, worst row <= 5x). Any failure prints the reason and flips the flag off for the run. Log exactly one line "[cuda expert gemm] ON" or "[cuda expert gemm] OFF: <reason>", plus regs/smem/compile seconds. Compile must fit the 15-minute run: target under 90 s; if the CuTe templates are slow, use inline-PTX wgmma.
MILESTONES (read from the profile's top_kernels; the new kernel shows under its own name, e.g. _nvfp4_wgmma_grouped_kernel, 2 calls): M0 the log says ON and the 5 tests pass; M1 first_turn w13+w2 under 6.0 ms (beats the 8.15-8.87 ms Triton incumbent by 1.4x); M2 under 3.5 ms (~470 TFLOP/s); stretch 2.5 ms (~650 TFLOP/s, CUTLASS mixed-input class). Below M1 the stage still teaches the next one (judge it on the kernel row, not the headline).

## Expected

- `*load=first_turn*` faster: Replaced: _moe_nvfp4_gemm_kernel x2 at 8873 us (full ctx0), 8324 us (sliding ctx32768, weight 138, the heaviest row of the problem), 8339 us (full ctx32768), 8149 us (full ctx65536): 37-58% of each shape. Floor for w13+w2 is the math, 16384 local routes x 100.7 MFLOP = 1.65 TFLOP = 1.67 ms at 989 TFLOP/s (the shapes' floors are 5518/5663/7710/9848 us). At the M2 target of ~3.0-3.5 ms the shapes go 15386->~10.5k, 14275->~9.4k, 18166->~13.3k, 22333->~17.5k us: ~1.5x on the first_turn load, ~1.15x on the headline geomean (1.059 -> ~1.2x); at the 2.5 ms stretch ~1.25x.
- `*load=turn_1*` neutral: stage 1 keeps Marlin at T <= 4096 (752.4/653.9 us vs a 189 us weight-stream floor); stage 3 targets ~300 us
- `*load=turn_4*` neutral: stage 1 keeps Marlin at T <= 4096 (2217/2131 us vs a 305 us math floor); stage 2 targets ~600-700 us

## Risk

Biggest risk: a hand-written warp-specialized wgmma RS pipeline (fragment-order code permutation, TMA descriptors, cp.async-into-swizzle, mbarrier phases, setmaxnreg) written with no local GPU; most failure modes are caught by the prepare() self-check and fall back to the incumbent, so the cost of a bug is a check, not a wrong answer. Second risk: compile time inside the 15-minute budget if CUTLASS templates are pulled in. Third: L2 reuse of weight tiles across token blocks at first_turn (2 token blocks of 256 per expert per weight tile) must be scheduled, or the kernel turns bandwidth bound at ~1.8 GB of code re-reads. Whole-headline upside across stages 1-3 is ~1.6-1.8x, which is the only plan in the queue whose arithmetic reaches the board leader.

## Milestones

Rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
