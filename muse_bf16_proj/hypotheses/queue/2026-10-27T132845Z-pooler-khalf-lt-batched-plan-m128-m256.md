---
problem: muse_bf16_proj
slug: pooler-khalf-lt-batched-plan-m128-m256
created: 2026-10-27T132845Z
base: master
area: pooler GEMM (m=128/256, K-halves batched GEMM on a prepare-picked cuBLASLt plan)
change: >
  Built on master (8f233e88 lineage, khalf b2 landed). Today's khalf cell at m=128 runs
  torch.bmm(xh, pooler_t.view(2, 4096, 4096), out_dtype=float32) through cuBLAS's own heuristic
  (profile a6424247: nvjet_sm90_tss_128x64_64x8_1x2_h_bz_NNT 21.4 us, i.e. ~3.1 TB/s on 67 MB
  against the 4.8 TB/s floor 14.6 us) and then _add2_cast_kernel (1.8 us). The bmm algo has never
  been chosen by measurement. CHANGE, in the khalf section only: (1) add to a new C++-only Lt
  extension (copy master's _LT_CPP MuseLtDesc/candidates/run pattern, new LT_EXT_NAME, no nvcc) a
  strided-batched descriptor for exactly the GEMM torch.bmm issues: batch 2, A = the pooler_t half
  [4096 k, 4096 n] (batch stride 4096*4096), B = the x half (row stride 8192, batch stride 4096),
  D fp32 contiguous [2, m, 4096] (batch stride m*4096), compute/scale fp32, via
  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT and CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET; candidates()
  returns up to 16 heuristic algos at split-K 1 for workspace caps 0 and 1 MB, run() launches one
  with a per-state workspace. (2) _khalf_forward_lt(x, state, m): that Lt call into a fresh fp32
  [2, m, 4096] then the unchanged _add2_cast_kernel. (3) Two cells, own flags: khalf_lt128_ok
  (m=128; incumbent = whatever routes today, i.e. khalf b2 heuristic bmm when pooler_khalf128_ok
  else NN torch.mm) and khalf_lt256_ok (m=256; incumbent NN torch.mm on pooler_t under
  nn_p256_ok). Per cell: read-flush 15-rep graph-replay pre-screen of every candidate algo (print
  the top 5 with algo id, tile, stages, net us and the heuristic algo's rank), _cell_check of the
  best two (rel L2 <= 3e-3 vs fp64, probe row exact, bitwise repeat), then the KHALF rule on the
  best passing one: KHALF_ROUNDS read-flush _cell_gate rounds, master/cand >= 1.03 in each and
  cand IQR <= 1.5x max(master IQR, 0.5 us), then _cell_kernels asserting exactly one GEMM with no
  splitK/reduce in its name plus _add2_cast_kernel. kernel_fn: pooler at cfg m == 256 with
  khalf_lt256_ok -> _khalf_forward_lt BEFORE the NN branch; at m == 128 khalf_lt128_ok is checked
  before today's khalf branch. State B inherits flags and algo rows; AB_CASES maps pooler
  m=128/256 to the new flags first. One PATH CHOSEN KL line per m (both rounds' net us, chosen
  algo), re-printed at the end of prepare. m=128 runs after _khalf_prepare so its incumbent is
  known; budget: own deadline like KHALF_START_DEADLINE_S.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "a measured Lt plan for the 2-batch K-half GEMM vs the heuristic nvjet 128x64 (21.4 us at ~3.1 TB/s): 21.87 -> ~20.5 us if a plan reaches ~3.4 TB/s; the gate line shows the plan-vs-heuristic delta even when it does not route"
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: K-halves with fp32 partials at m=256 lost only through torch add+cast (cand b 0.934-0.948; bf16-partials cand a 1.022-1.042); one Triton add-cast plus a measured batched plan may clear 1.03 vs NN 29.0 us; otherwise the read-flush gate line closes the fp32 K-split question at m=256
  - shapes: layer=pooler load=decode_* m=*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "pooler-khalf-m128-fp32-partials-triton-castadd: landed b2 with torch.bmm's heuristic algo at m=128 only; this keeps b2's partials and add-cast but picks the batched GEMM's algo from cuBLASLt's list by a read-flush pre-screen, and extends it to m=256, where pooler-k-halves-fp32-partials-restack-m128-m256's fp32 candidate lost only with torch's two-kernel add+cast (0.934-0.948)"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.5% (self-declared below margin)"
---

# pooler-khalf-lt-batched-plan-m128-m256

## Change

Built on master (8f233e88 lineage, khalf b2 landed). Today's khalf cell at m=128 runs torch.bmm(xh, pooler_t.view(2, 4096, 4096), out_dtype=float32) through cuBLAS's own heuristic (profile a6424247: nvjet_sm90_tss_128x64_64x8_1x2_h_bz_NNT 21.4 us, i.e. ~3.1 TB/s on 67 MB against the 4.8 TB/s floor 14.6 us) and then _add2_cast_kernel (1.8 us). The bmm algo has never been chosen by measurement. CHANGE, in the khalf section only: (1) add to a new C++-only Lt extension (copy master's _LT_CPP MuseLtDesc/candidates/run pattern, new LT_EXT_NAME, no nvcc) a strided-batched descriptor for exactly the GEMM torch.bmm issues: batch 2, A = the pooler_t half [4096 k, 4096 n] (batch stride 4096*4096), B = the x half (row stride 8192, batch stride 4096), D fp32 contiguous [2, m, 4096] (batch stride m*4096), compute/scale fp32, via CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT and CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET; candidates() returns up to 16 heuristic algos at split-K 1 for workspace caps 0 and 1 MB, run() launches one with a per-state workspace. (2) _khalf_forward_lt(x, state, m): that Lt call into a fresh fp32 [2, m, 4096] then the unchanged _add2_cast_kernel. (3) Two cells, own flags: khalf_lt128_ok (m=128; incumbent = whatever routes today, i.e. khalf b2 heuristic bmm when pooler_khalf128_ok else NN torch.mm) and khalf_lt256_ok (m=256; incumbent NN torch.mm on pooler_t under nn_p256_ok). Per cell: read-flush 15-rep graph-replay pre-screen of every candidate algo (print the top 5 with algo id, tile, stages, net us and the heuristic algo's rank), _cell_check of the best two (rel L2 <= 3e-3 vs fp64, probe row exact, bitwise repeat), then the KHALF rule on the best passing one: KHALF_ROUNDS read-flush _cell_gate rounds, master/cand >= 1.03 in each and cand IQR <= 1.5x max(master IQR, 0.5 us), then _cell_kernels asserting exactly one GEMM with no splitK/reduce in its name plus _add2_cast_kernel. kernel_fn: pooler at cfg m == 256 with khalf_lt256_ok -> _khalf_forward_lt BEFORE the NN branch; at m == 128 khalf_lt128_ok is checked before today's khalf branch. State B inherits flags and algo rows; AB_CASES maps pooler m=128/256 to the new flags first. One PATH CHOSEN KL line per m (both rounds' net us, chosen algo), re-printed at the end of prepare. m=128 runs after _khalf_prepare so its incumbent is known; budget: own deadline like KHALF_START_DEADLINE_S.

## Expected

- `layer=pooler load=verify_16 m=128` faster: a measured Lt plan for the 2-batch K-half GEMM vs the heuristic nvjet 128x64 (21.4 us at ~3.1 TB/s): 21.87 -> ~20.5 us if a plan reaches ~3.4 TB/s; the gate line shows the plan-vs-heuristic delta even when it does not route
- `layer=pooler load=verify_32 m=256` faster: K-halves with fp32 partials at m=256 lost only through torch add+cast (cand b 0.934-0.948; bf16-partials cand a 1.022-1.042); one Triton add-cast plus a measured batched plan may clear 1.03 vs NN 29.0 us; otherwise the read-flush gate line closes the fp32 K-split question at m=256
- `layer=pooler load=decode_* m=*` flat: m <= 32 untouched
- `layer=qkv_gate *` flat: untouched

## Risk

Effect: pooler m=128 -1 us is 0.23%; m=256 -1.5 us is 0.28%: sub-margin alone, readable on the two pooler rows and the KL gate lines, stack candidate. Decode-portable: decode's T=128 P runs the khalf/NN pooler cell (W6-B) and T=256 the NN path. Risks: the strided-batch layout attributes on a column-major Lt view are easy to get transposed (the _cell_check catches it; print each layout's rows/cols/ld/batch stride); the heuristic's first algo may equal torch.bmm's (then the pre-screen ties and the gate rejects: print the bmm kernel name next to each finalist). Interacts with the queued khalf-gate-three-rounds-pooled control only through khalf's gate statistics; this cell uses the same KHALF rule so the control's verdict applies to it too. Keep fp32 partials: bf16 partials are ruled out (DISTILL 3.4).
