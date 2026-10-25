---
problem: muse_bf16_proj
slug: qkv-k-split-batched-gemm-m65-m256
created: 2026-10-25T153141Z
base: master
area: qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only,
  which covers the scored rows verify_16 m=128 and verify_32 m=256. The m=512 test stays on master
  (m > 256). WHY: qkv_gate (n 2304, k 8192) is the cuBLAS cell furthest from its floor in ratio.
  At m=128 it is 16.58 us against a floor of 8.42, as nvjet 96x64 splitK 12.7 us plus cublasLt
  splitKreduce 3.1 us plus a node. At m=256 it is 19.71 against 9.77, as nvjet 96x64 2x4 TNN 19.9
  us, about 490 TFLOP/s. cuBLAS's own split-K at m=128 pays a separate 3.1 us reduce kernel over
  fp32 partials. CHANGE: the pooler-k-halves mechanism applied to qkv, with S in {2, 4} K-slices.
  xs = x.view(m, S, 8192 // S).permute(1, 0, 2) is a strided view with no copy. ws =
  state['qkv_gate'].view(2304, S, 8192 // S).permute(1, 0, 2) is used transposed, or a contiguous
  [S, 8192 // S, 2304] NN repack made once in prepare(); the repack is lossless and adds 37.7 MB
  per S. p = torch.bmm(xs, ws_t) gives [S, m, 2304] bf16; then out = torch.add(p[0], p[1]) for
  S=2, or p.sum(0) for S=4. Precision: S bf16 roundings, about 0.2-0.4% per element and L2 well
  under 1%. Single-nonzero rows are exact. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Candidates
  per m: S=2 TN, S=2 NN, S=4 TN and S=4 NN. Route the fastest only if it is >= 1.02x master at
  that m; the m=128 and m=256 decisions are separate and printed. The profiler assertion must show
  one GEMM kernel plus one add/reduce and no copy. Every other layer and m stays byte-identical.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "nvjet splitK 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; a bf16 add over 2x0.6 MB (~0.8 us) replaces the 3.1 us fp32 reduce: row ~14"
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: "nvjet 96x64 19.9 / row 19.71, floor 9.77; shorter K per batch lets cuBLAS pick a fuller tile grid: row ~16-17 if a candidate wins, else master"
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: m <= 32 untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-k-split-batched-gemm-m65-m256

## Change

Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only, which covers the scored rows verify_16 m=128 and verify_32 m=256. The m=512 test stays on master (m > 256). WHY: qkv_gate (n 2304, k 8192) is the cuBLAS cell furthest from its floor in ratio. At m=128 it is 16.58 us against a floor of 8.42, as nvjet 96x64 splitK 12.7 us plus cublasLt splitKreduce 3.1 us plus a node. At m=256 it is 19.71 against 9.77, as nvjet 96x64 2x4 TNN 19.9 us, about 490 TFLOP/s. cuBLAS's own split-K at m=128 pays a separate 3.1 us reduce kernel over fp32 partials. CHANGE: the pooler-k-halves mechanism applied to qkv, with S in {2, 4} K-slices. xs = x.view(m, S, 8192 // S).permute(1, 0, 2) is a strided view with no copy. ws = state['qkv_gate'].view(2304, S, 8192 // S).permute(1, 0, 2) is used transposed, or a contiguous [S, 8192 // S, 2304] NN repack made once in prepare(); the repack is lossless and adds 37.7 MB per S. p = torch.bmm(xs, ws_t) gives [S, m, 2304] bf16; then out = torch.add(p[0], p[1]) for S=2, or p.sum(0) for S=4. Precision: S bf16 roundings, about 0.2-0.4% per element and L2 well under 1%. Single-nonzero rows are exact. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Candidates per m: S=2 TN, S=2 NN, S=4 TN and S=4 NN. Route the fastest only if it is >= 1.02x master at that m; the m=128 and m=256 decisions are separate and printed. The profiler assertion must show one GEMM kernel plus one add/reduce and no copy. Every other layer and m stays byte-identical.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: nvjet splitK 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; a bf16 add over 2x0.6 MB (~0.8 us) replaces the 3.1 us fp32 reduce: row ~14
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 96x64 19.9 / row 19.71, floor 9.77; shorter K per batch lets cuBLAS pick a fuller tile grid: row ~16-17 if a candidate wins, else master
- `layer=* load=decode_* m=*` unchanged: m <= 32 untouched

## Risk

Same mechanism as pooler-k-halves-batched-gemm-m65-m256, on a separate layer and rows, so each is attributed per row. The prepare-time cublasLt algo pick on qkv m=256 did not carry over to the judge (LESSONS). This selector times graph replays with a cold L2, as the ouns selector did, which predicted the judge.
