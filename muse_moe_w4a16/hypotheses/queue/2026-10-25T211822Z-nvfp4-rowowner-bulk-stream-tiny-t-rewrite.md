---
problem: muse_moe_w4a16
slug: nvfp4-rowowner-bulk-stream-tiny-t-rewrite
created: 2026-10-25T211822Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1
  bs=1, the bs=1 and bs=4 tests). Every other T runs master byte for byte. New code in its own
  load_inline module (muse_moe_w4a16_rowowner_v1), sm_90 instructions only (cp.async.bulk +
  mbarrier, mma.sync or CUDA-core FMA; no wgmma, no setmaxnreg), -std=c++20 in extra_cuda_cflags
  (LESSONS 159). WHY: decode_1 is the load furthest from its floor (46.4 vs 11.4 us); the Marlin
  pair is 35.4 us on ~56 MB of unique weight bytes = ~1.6 TB/s, while muse_bf16_proj's row-owner
  reduce-free M=1 stream (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc;
  bulk_kernel<16,8192,8,4,2> at 2 CTAs/SM) reached ~3.5 TB/s on m<=32 weight streams and won
  qkv_gate decode_1 1.16x. Every earlier small-M NVFP4 design kept padding to N16, smem staging
  for wgmma, split-K tickets or grid spins (LESSONS 67: none exceeded ~2 TB/s). DESIGN. (1)
  Prepare (untimed, lossless repack): per local expert, re-lay w13 so gate row i and up row i are
  adjacent (one 'pair row' = 2 x 4096 e2m1 codes + 2 x 256 e4m3 scales = 4608 B), and w2 rows
  as-is (each of the 4096 output rows = I/2 code bytes + I/16 scale bytes), each expert's rows
  contiguous so a CTA's span is one contiguous byte range; keep the per-expert global scale. NVFP4
  stays NVFP4 (repack only). (2) Device-side work map (no host branch, graph-safe): kernel A (1
  CTA, or folded into the w13 kernel's prologue) reads topk_ids, maps to local experts (master's
  expert_map), and writes the list of (expert, route rows) for <= 4*8 = 32 routes; CTAs of the w13
  kernel each own a fixed contiguous span of pair rows of ONE active expert, spans sized so
  (active experts x I) / (2 x SMs) rows per CTA. (3) w13 kernel: one producer lane issues 1-D
  cp.async.bulk copies of the CTA's span into a deep smem ring (>= 64 KB in flight per SM,
  mbarrier full/empty), consumers dequantize with master's exact dequant8 bit trick (e2m1 -> bf16
  via mask/shift + fma by scale*2^119) and FMA against the <= 4 token rows of x held in
  registers/smem (bf16x2 products, fp32 accumulate over full K; warp-shuffle reduce within the row
  owner warp only, no cross-CTA reduction), then silu(gate)*up in fp32 with bf16 rounding of gate
  and up first (SGLang parity, as master's silu_mul) and a bf16 h row write. (4) w2 kernel,
  launched after w13 with PDL: the same row-owner stream over w2 rows (K = I), each output element
  y[route, n] produced whole by its owner warp from h, bf16 out into master's cache3 layout so
  master's _post_norm_combine_kernel runs unchanged. No split-K, no atomics, no tickets, no spins
  (the bs=1 slow mode came from cross-CTA reduction; LESSONS 62, 151). (5) Consumer choice:
  implement CUDA-core FMA (variant F) and mma.sync m16n8k16 with tokens padded to 8 (variant M);
  pick per T in the prepare gate. (6) Gate: cold-L2 graph replays of the full T<=4 path vs
  master's path on skewed AND _wg_judge_ids routings at T=1 and T=4, require >= 1.02x on the worse
  routing (LESSONS 164), print kernel-only TB/s per kernel and one PATH CHOSEN line; a canary
  forces every active-expert count (1, 2, 8 local experts) and checks finiteness and relative L2
  vs Marlin with the judge's 1.5% bound (not a self-imposed 1e-2; LESSONS 150). Instrument runs_ns
  for bimodality. Fallback: any build/canary/gate failure keeps master's T<=4 path. First
  measurement is an okc check (new bulk-copy/mbarrier producer can fault or hang).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent Marlin pair 35.4 us (~56 MB, ~1.6 TB/s) of 46.4 us; floor 11.4 us. Milestones: M1 correct + prints kernel-only TB/s; M2 replaced kernels <= 35.4 us (>=1.6 TB/s); M3 >= 2.6 TB/s -> 17-22 us, decode_1 ~28-33 us, +5.5-8% headline"
  - shapes: bs=8 *
    direction: flat
    why: T>4 runs master
  - shapes: "bs=1[0-9]* *"
    direction: flat
    why: T>4 runs master
  - shapes: bs=2* *
    direction: flat
    why: T>4 runs master
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
prior: "cuda-fused-split-i-moe-tiny-t: that queued design keeps the wgmma rega consumer (N16 padding, smem B slab), the measured ~2 TB/s limit that only tied Marlin at T=1 (tiny-m-bucket16 46.0 vs 45.5 us); this changes the consumer (CUDA-core/mma.sync on a register-resident x) and the load path (1-D bulk row spans, no TMA tiles), keeps the w13->w2 boundary with no cross-CTA reduction and no spins (unlike cuda-fused-moe-bs1-persistent-*, bimodal from grid spins)"
rewrite_stage: 1
replaces: Marlin w13 + silu_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2) at 1 <= num_tokens <= 4
review: required
status: queued
---

# nvfp4-rowowner-bulk-stream-tiny-t-rewrite

## Change

Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1 bs=1, the bs=1 and bs=4 tests). Every other T runs master byte for byte. New code in its own load_inline module (muse_moe_w4a16_rowowner_v1), sm_90 instructions only (cp.async.bulk + mbarrier, mma.sync or CUDA-core FMA; no wgmma, no setmaxnreg), -std=c++20 in extra_cuda_cflags (LESSONS 159). WHY: decode_1 is the load furthest from its floor (46.4 vs 11.4 us); the Marlin pair is 35.4 us on ~56 MB of unique weight bytes = ~1.6 TB/s, while muse_bf16_proj's row-owner reduce-free M=1 stream (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc; bulk_kernel<16,8192,8,4,2> at 2 CTAs/SM) reached ~3.5 TB/s on m<=32 weight streams and won qkv_gate decode_1 1.16x. Every earlier small-M NVFP4 design kept padding to N16, smem staging for wgmma, split-K tickets or grid spins (LESSONS 67: none exceeded ~2 TB/s). DESIGN. (1) Prepare (untimed, lossless repack): per local expert, re-lay w13 so gate row i and up row i are adjacent (one 'pair row' = 2 x 4096 e2m1 codes + 2 x 256 e4m3 scales = 4608 B), and w2 rows as-is (each of the 4096 output rows = I/2 code bytes + I/16 scale bytes), each expert's rows contiguous so a CTA's span is one contiguous byte range; keep the per-expert global scale. NVFP4 stays NVFP4 (repack only). (2) Device-side work map (no host branch, graph-safe): kernel A (1 CTA, or folded into the w13 kernel's prologue) reads topk_ids, maps to local experts (master's expert_map), and writes the list of (expert, route rows) for <= 4*8 = 32 routes; CTAs of the w13 kernel each own a fixed contiguous span of pair rows of ONE active expert, spans sized so (active experts x I) / (2 x SMs) rows per CTA. (3) w13 kernel: one producer lane issues 1-D cp.async.bulk copies of the CTA's span into a deep smem ring (>= 64 KB in flight per SM, mbarrier full/empty), consumers dequantize with master's exact dequant8 bit trick (e2m1 -> bf16 via mask/shift + fma by scale*2^119) and FMA against the <= 4 token rows of x held in registers/smem (bf16x2 products, fp32 accumulate over full K; warp-shuffle reduce within the row owner warp only, no cross-CTA reduction), then silu(gate)*up in fp32 with bf16 rounding of gate and up first (SGLang parity, as master's silu_mul) and a bf16 h row write. (4) w2 kernel, launched after w13 with PDL: the same row-owner stream over w2 rows (K = I), each output element y[route, n] produced whole by its owner warp from h, bf16 out into master's cache3 layout so master's _post_norm_combine_kernel runs unchanged. No split-K, no atomics, no tickets, no spins (the bs=1 slow mode came from cross-CTA reduction; LESSONS 62, 151). (5) Consumer choice: implement CUDA-core FMA (variant F) and mma.sync m16n8k16 with tokens padded to 8 (variant M); pick per T in the prepare gate. (6) Gate: cold-L2 graph replays of the full T<=4 path vs master's path on skewed AND _wg_judge_ids routings at T=1 and T=4, require >= 1.02x on the worse routing (LESSONS 164), print kernel-only TB/s per kernel and one PATH CHOSEN line; a canary forces every active-expert count (1, 2, 8 local experts) and checks finiteness and relative L2 vs Marlin with the judge's 1.5% bound (not a self-imposed 1e-2; LESSONS 150). Instrument runs_ns for bimodality. Fallback: any build/canary/gate failure keeps master's T<=4 path. First measurement is an okc check (new bulk-copy/mbarrier producer can fault or hang).

## Expected

- `bs=1 *` faster: incumbent Marlin pair 35.4 us (~56 MB, ~1.6 TB/s) of 46.4 us; floor 11.4 us. Milestones: M1 correct + prints kernel-only TB/s; M2 replaced kernels <= 35.4 us (>=1.6 TB/s); M3 >= 2.6 TB/s -> 17-22 us, decode_1 ~28-33 us, +5.5-8% headline
- `bs=8 *` flat: T>4 runs master
- `bs=1[0-9]* *` flat: T>4 runs master
- `bs=2* *` flat: T>4 runs master

## Risk

High risk: new bulk-copy producer and mbarrier ring can hang (bounded mbar_wait with an error word, as master). bf16 LESSONS 167: the row-owner layout serialises per-token FMA behind one owner, so its time grows with m (+0.6/+6/+25 us at m=1/16/32 at 1 CTA/SM) and the pure stream reached only 2.54 TB/s cold at 1 CTA/SM; hence T<=4 only and 2 CTAs/SM like the winning qkv instance. NVFP4 adds dequant ALU per byte (2 codes per byte vs 0.5 bf16 elements per byte): judge it by printed TB/s, and if variant F is ALU-bound at T=4, stage 2 restricts to T=1 or uses mma.sync. Stage 2 (if M2 met, M3 not): 2-expert interleaved spans and a w2 prefetch under w13 via PDL.

## Milestones

Rewrite stage 1 of `Marlin w13 + silu_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2) at 1 <= num_tokens <= 4`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
