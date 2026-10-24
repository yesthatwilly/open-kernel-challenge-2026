---
problem: muse_prefill_layer
slug: router-triton-hilo-splitk-turn
created: 2026-10-24T075454Z
base: master
area: router GEMM
change: >
  For tokens <= 4096 only, replace the router's two cuBLAS calls on the side stream (logits =
  torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t,
  out_dtype=fp32)) with one Triton launch ported from master's muse_bf16_proj/submission.py
  (_router_hilo_kernel / _router_forward): grid (cdiv(M, BLOCK_M), 256 // BLOCK_N, SPLIT_K), each
  program loads its x tile once per BLOCK_K step and does acc = tl.dot(x, hi, acc); acc =
  tl.dot(x, lo, acc) in fp32, then tl.atomic_add (sem relaxed) of its K-partial into an fp32 [M,
  256] output zeroed with torch.zeros right before the launch, all inside the existing
  torch.cuda.stream(side_stream) block. prepare() additionally keeps hi and lo as [256, 8192]
  row-major bf16 (state router_hi / router_lo, the same split as today, untransposed). Config for
  this M range, from tokens only: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64, num_warps 8, num_stages 3,
  SPLIT_K 16 for M <= 1024 (6 x 2 x 16 = 192 CTAs at M=753, K-slice 512) and SPLIT_K 4 for 1024 <
  M <= 4096 (24 x 2 x 4 = 192 CTAs at M=3012, K-slice 2048); do_not_specialize M; prepare()
  pre-compiles both buckets on synthetic bf16 x so the timed call never compiles. For tokens >
  4096 (first_turn) the cuBLAS pair stays: there it runs as nvjet 128x256 at ~485 TFLOP/s (283 us
  for 2 calls on 137 GFLOP, profile 62025c368c3c4e8e) on a side branch far shorter than the 1.4 ms
  pooler, so nothing is gained. FusedMuseSparkRouting, the stream fork/join and everything else
  are unchanged. Why at these M: cuBLAS is inefficient here: at M=753 the pair runs as split-K
  nvjet 64x96 (33-40 us) plus cublasLt splitKreduce (45.6 us on turn_1 full) for 6.3 GFLOP (~80
  TFLOP/s), and the moe-align-side-stream review found this side branch (~80-90 us incl. routing)
  exceeds the main branch it overlaps (pooler ~40-50 us + pre-expert norm 7 us), so it is on the
  critical path; at M=3012 it is nvjet 64x104 at 132 us for 25 GFLOP (~190 TFLOP/s), concurrent
  with the pooler and sharing its SMs.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "side branch drops from ~80-90 us to ~25-30 us (one launch + memset, ~6 GFLOP at ~300 TFLOP/s), below the ~50 us main branch, so up to ~40 us leaves the critical path: ~2-3% of 1079/1435 us; the bound is P+N-S, smaller where cuBLAS skipped the splitKreduce"
  - shapes: "*load=turn_4*"
    direction: faster
    why: router 132 us -> ~80 us running beside the pooler; fewer SM-cycles stolen from it; ~0.5-1.5%, likely inside noise
  - shapes: "*load=first_turn*"
    direction: flat
    why: cuBLAS pair kept above 4096 tokens (control)
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# router-triton-hilo-splitk-turn

## Change

For tokens <= 4096 only, replace the router's two cuBLAS calls on the side stream (logits = torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t, out_dtype=fp32)) with one Triton launch ported from master's muse_bf16_proj/submission.py (_router_hilo_kernel / _router_forward): grid (cdiv(M, BLOCK_M), 256 // BLOCK_N, SPLIT_K), each program loads its x tile once per BLOCK_K step and does acc = tl.dot(x, hi, acc); acc = tl.dot(x, lo, acc) in fp32, then tl.atomic_add (sem relaxed) of its K-partial into an fp32 [M, 256] output zeroed with torch.zeros right before the launch, all inside the existing torch.cuda.stream(side_stream) block. prepare() additionally keeps hi and lo as [256, 8192] row-major bf16 (state router_hi / router_lo, the same split as today, untransposed). Config for this M range, from tokens only: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64, num_warps 8, num_stages 3, SPLIT_K 16 for M <= 1024 (6 x 2 x 16 = 192 CTAs at M=753, K-slice 512) and SPLIT_K 4 for 1024 < M <= 4096 (24 x 2 x 4 = 192 CTAs at M=3012, K-slice 2048); do_not_specialize M; prepare() pre-compiles both buckets on synthetic bf16 x so the timed call never compiles. For tokens > 4096 (first_turn) the cuBLAS pair stays: there it runs as nvjet 128x256 at ~485 TFLOP/s (283 us for 2 calls on 137 GFLOP, profile 62025c368c3c4e8e) on a side branch far shorter than the 1.4 ms pooler, so nothing is gained. FusedMuseSparkRouting, the stream fork/join and everything else are unchanged. Why at these M: cuBLAS is inefficient here: at M=753 the pair runs as split-K nvjet 64x96 (33-40 us) plus cublasLt splitKreduce (45.6 us on turn_1 full) for 6.3 GFLOP (~80 TFLOP/s), and the moe-align-side-stream review found this side branch (~80-90 us incl. routing) exceeds the main branch it overlaps (pooler ~40-50 us + pre-expert norm 7 us), so it is on the critical path; at M=3012 it is nvjet 64x104 at 132 us for 25 GFLOP (~190 TFLOP/s), concurrent with the pooler and sharing its SMs.

## Expected

- `*load=turn_1*` faster: side branch drops from ~80-90 us to ~25-30 us (one launch + memset, ~6 GFLOP at ~300 TFLOP/s), below the ~50 us main branch, so up to ~40 us leaves the critical path: ~2-3% of 1079/1435 us; the bound is P+N-S, smaller where cuBLAS skipped the splitKreduce
- `*load=turn_4*` faster: router 132 us -> ~80 us running beside the pooler; fewer SM-cycles stolen from it; ~0.5-1.5%, likely inside noise
- `*load=first_turn*` flat: cuBLAS pair kept above 4096 tokens (control)

## Risk

Expected effect is small (2-3% on turn_1 shapes, at the edge of the +-1.5% interval); queued because it is the one lever on the turn_1 side-branch limiter the moe-align-side-stream review named, and it reuses the one big proven kernel (bf16_proj 1.151x). Atomic fp32 accumulation order is nondeterministic run to run; routing tolerance (0.05 of the 8th best, near-tie slack) absorbs fp32-ulp differences, as it did on bf16_proj and decode. If experts-marlin-block-m-* lands first and shrinks Marlin, turn_1 becomes more sensitive to this branch, so the gain may grow when stacked. torch.zeros adds one memset node on the side stream.
