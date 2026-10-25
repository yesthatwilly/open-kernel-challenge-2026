---
problem: muse_bf16_proj
slug: pooler-k-halves-batched-gemm-m65-m256
created: 2026-10-25T153141Z
base: master
area: pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only
  (scored rows verify_16 m=128 and verify_32 m=256; no test uses this band), replace master's path
  with a user-level split-K of 2. Master's path is torch.mm on the NN weight at m=65..128 (nvjet
  128x64 splitK 20.2 us + cublasLt splitKreduce 3.0 us) and _bf16_gemm_dispatch_impl at m=256
  (nvjet 128x64 TNT 30.4 us). WHY: pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) do the
  same FLOPs and bytes, but cuBLAS runs the long-K, narrow-N pooler worse. At m=256 the pooler
  takes 31.3 us (128x64 tiles, about 550 TFLOP/s) while the unpooler takes 24.5 us (128x128
  tiles); at m=128 it is 22.9 vs 18.9 us. Two K-halves of the pooler have the unpooler's tile
  count: n 4096 x k 4096 x 2 batches gives the same 128 CTAs of 128x128 (m=256) or 128x64 (m=128)
  tiles, each over k=4096, as the unpooler's one GEMM. CHANGE: xh = x.view(m, 2, 4096).permute(1,
  0, 2), a strided view [2, m, 4096] with strides (4096, 8192, 1) and no copy. For the weight,
  either candidate T, a TN view of state['pooler'] as [2, 4096, 4096] with strides (4096, 8192, 1)
  used transposed, or candidate N, the existing NN weight state['pooler_t'].view(2, 4096, 4096)
  (contiguous halves, no new memory). Then p = torch.bmm(xh, w_halves) into [2, m, 4096] bf16, and
  out = torch.add(p[0], p[1]): two graph nodes, with the add reading 4 MB and writing 2 MB at
  m=256. Precision: each half rounds once to bf16 (about 0.2% per element, L2 well under 1%). The
  single-nonzero rows stay exact, because the other half is exactly 0. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The
  profiler assertion must show exactly one nvjet/cutlass GEMM kernel plus one elementwise add, and
  no copy kernel; if torch inserts a contiguous() copy of xh, print it and keep master. Every
  other layer and m stays byte-identical.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: nvjet 128x64 30.4 us / row 31.33, compute floor 17.37; batched halves should match the unpooler m=256 nvjet 128x128 (24.8 kernel) + add ~1.2 -> row ~26
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: NN splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; halves should match unpooler m=128 (row 18.95) + add ~1.0 -> row ~20
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: m <= 32 untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-k-halves-batched-gemm-m65-m256

## Change

Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only (scored rows verify_16 m=128 and verify_32 m=256; no test uses this band), replace master's path with a user-level split-K of 2. Master's path is torch.mm on the NN weight at m=65..128 (nvjet 128x64 splitK 20.2 us + cublasLt splitKreduce 3.0 us) and _bf16_gemm_dispatch_impl at m=256 (nvjet 128x64 TNT 30.4 us). WHY: pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) do the same FLOPs and bytes, but cuBLAS runs the long-K, narrow-N pooler worse. At m=256 the pooler takes 31.3 us (128x64 tiles, about 550 TFLOP/s) while the unpooler takes 24.5 us (128x128 tiles); at m=128 it is 22.9 vs 18.9 us. Two K-halves of the pooler have the unpooler's tile count: n 4096 x k 4096 x 2 batches gives the same 128 CTAs of 128x128 (m=256) or 128x64 (m=128) tiles, each over k=4096, as the unpooler's one GEMM. CHANGE: xh = x.view(m, 2, 4096).permute(1, 0, 2), a strided view [2, m, 4096] with strides (4096, 8192, 1) and no copy. For the weight, either candidate T, a TN view of state['pooler'] as [2, 4096, 4096] with strides (4096, 8192, 1) used transposed, or candidate N, the existing NN weight state['pooler_t'].view(2, 4096, 4096) (contiguous halves, no new memory). Then p = torch.bmm(xh, w_halves) into [2, m, 4096] bf16, and out = torch.add(p[0], p[1]): two graph nodes, with the add reading 4 MB and writing 2 MB at m=256. Precision: each half rounds once to bf16 (about 0.2% per element, L2 well under 1%). The single-nonzero rows stay exact, because the other half is exactly 0. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The profiler assertion must show exactly one nvjet/cutlass GEMM kernel plus one elementwise add, and no copy kernel; if torch inserts a contiguous() copy of xh, print it and keep master. Every other layer and m stays byte-identical.

## Expected

- `layer=pooler load=verify_32 m=256` faster: nvjet 128x64 30.4 us / row 31.33, compute floor 17.37; batched halves should match the unpooler m=256 nvjet 128x128 (24.8 kernel) + add ~1.2 -> row ~26
- `layer=pooler load=verify_16 m=128` faster: NN splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; halves should match unpooler m=128 (row 18.95) + add ~1.0 -> row ~20
- `layer=* load=decode_* m=*` unchanged: m <= 32 untouched

## Risk

Overlaps the cells of the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (pooler m=256) and of bf16-verify16-cutlass-m128-onetile-splitk-rewrite (pooler m=128), but this is pure cuBLAS dispatch at about 30 lines; if it wins, the CUTLASS gates measure against it. Not a retry of pooler-nn-layout-widen-m256 (flat), which kept one long-K GEMM.
