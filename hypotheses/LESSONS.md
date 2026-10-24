# Cross-problem lessons

Hand-maintained by churnsubs. Each problem's `FINDINGS.md` is regenerated from its own results.
This file holds what carries over between problems: measured facts about the hardware, the judge
and the SGLang kernels, with the run that showed each one. Read it before `/hypothesize`, and
add to it when a harvest teaches something general. The evidence for every line is in
`<problem>/hypotheses/done/<outcome>/<slug>.{md,result.json}`, and the full okc record and
profile are in `<problem>/hypotheses/archive/<slug>/`.

## Measurement

- Identical code drifts ~1.5-2.5% per shape between runs (decode expert-id-map-single-kernel moved
  every shape +1-2.5% uniformly; qkv_gate decode_1 ranged 0.981-1.025 across five master-equivalent
  bf16 runs). An effect under ~1.5% on the geomean cannot be resolved by one run. Read per-shape
  rows, the paired ratio against SGLang in the same run, and the replaced kernel's own time
  (`top_kernels`), not the headline alone.
- Triton kernels' times reproduce within ~1% between runs; cuBLAS split-K rows swing more.
- Reported shape time minus the profiled kernel sum is a real per-graph-node cost: ~2-4 us for
  nvjet+splitKreduce pairs, ~0.3 us for a single Triton launch (bf16 pooler/unpooler runs).
- Resubmitting byte-identical code gets a new job id (it is not deduplicated) and costs a check.

## Launches and fusion

- Fusing a handful of tiny kernels was flat on decode-sized layer shapes (decode
  cache-store-fused-fp8-cast, expert-id-map-single-kernel, moe align-single-token-ep, prefill
  moe-align-side-stream, all neutral). It pays only where launches dominate the shape:
  attention_decode's fused pre-attention gave 1.127x (dfee6f2939974bad) on 17-36 us shapes.

## GEMMs (bf16, H200)

- The one large bf16 win: the router as one 3-D split-K Triton kernel (grid cdiv(M,BM) x N/BN x
  SPLIT_K, ~128 CTAs, fp32 atomic_add into a zeroed output, BF16 hi+lo split), 10.5-17.7 us to
  5.7-9.3 us at every m, 1.151x (bf16 router-triton-hilo-fused-control-splitk-all-m).
  A full-K grid of only 8 CTAs at m=128/256 was 50-98 us (router-triton-hilo-fused, 0.940x): fill
  the grid.
- Triton full-K weight streams: BLOCK_N=64, BLOCK_K=256, 4 stages (128 KB in flight per CTA)
  reached 3.73 TB/s at m=16 and beat nvjet. BLOCK_N=32 (64 KB in flight) did 3.3 TB/s, and
  BLOCK_N=16 regressed 1.6x (qkv-gate-triton-skinny-m32). At m=32 the penalty is not the 32-row tile:
  two 16-row tiles were worse (unpooler BLOCK_M=16 control).
- Split-K ticket-semaphore reduce (masked partial store, debug_barrier, one gpu-scope acq_rel
  atomic_add ticket, winner reloads .cg, fixed-order sum, atomic_xchg reset) passes correctness
  (bf16 pooler 1.166x and qkv-gate 1.143x vs 1.151x master: both neutral, within drift).
- cuBLAS vs cublasLt backend preference changes nothing on this image (bf16-cublaslt-backend-toggle).
- FMA-broadcast (no tl.dot) GEMV was 10x slower (unpooler-slice-triton-gemv).

## Marlin MoE (W4A16 NVFP4)

- Per-m-block cost grows with block_size_m (thread_m_blocks). The next-larger block lost 18-28%
  (moe marlin-block-size-m-next-larger, 0.918x). Capping at 16 gained 8-15% at T=128/256
  (decode marlin-block-m-cap-16, 1.062x). SGLang's T*8/32 rule overcounts rows per local expert
  8x under EP8.
- Knobs that did nothing: workspace blocks per SM, use_fp32_reduce=False, use_atomic_add.
- Prefill stage-1 Triton W4A16 grouped GEMM (in-register arithmetic NVFP4 dequant, 128x128x64 tiles,
  SGLang fused_moe_kernel skeleton) is correct but ~2.0 ms for w13+w2 at turn_1 and 3.3 ms at turn_4,
  0.689x overall (experts-triton-w4a16-inkernel-fp4, 218515e2d0b14ce7). Losses shrink with T (0.46x
  turn_1 sliding to 0.90x first_turn), so the cost is per-tile dequant overhead, not the MMA.
- A Triton W4A16 grouped GEMV with arithmetic FP4 dequant was 2-2.6x slower than Marlin at small M
  (triton-w13-grouped-gemv-small-m). It was dequant-ALU-bound with too few busy CTAs.

## Attention (FA3, FP8 KV, page size 1)

- num_splits is only a cap under varlen dynamic split, and FA3 already sizes splits on the GPU
  (decode fa3-num-splits-* reviews).
- FA3 sits at ~600 TFLOP/s on every full prefill shape whatever the CTA count (fa3-batched-mode review).
- Full-attention decode reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit.
- Copying KV into a contiguous buffer per call cost more than it saved on prefill
  (fa3-contiguous-kv-gather, 0.988x).

## Rules rulings (user)

- No per-call BF16 copy of NVFP4 weights (dequant in registers inside the kernel only).
