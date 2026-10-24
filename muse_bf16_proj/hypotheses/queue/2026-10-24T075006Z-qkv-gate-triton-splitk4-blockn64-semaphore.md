---
problem: muse_bf16_proj
slug: qkv-gate-triton-splitk4-blockn64-semaphore
created: 2026-10-24T075006Z
base: master
area: qkv GEMM
change: >
  Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only
  (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl
  (nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT 8.1-10.8 us + cublasLt::splitKreduce_kernel
  1.7-2.0 us; reported 13.8-14.4 us vs a 7.9 us floor, the largest reported-minus-kernel gap of
  any layer) with ONE Triton launch of a split-K kernel with a bf16 semaphore epilogue: grid = (N
  // BLOCK_N, SPLIT_K) = (2304/64, 4) = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K-slice
  2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0),
  num_warps=4, num_stages=4 (24-32 KB smem per stage, so 2 CTAs fit per SM and all 144 CTAs are
  co-resident in one wave); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major
  [N, K] weight (the router's transposed addressing, no repack), fp32 acc via tl.dot(x_tile,
  w_tile, acc). Epilogue: store the fp32 partial to a persistent scratch state['qkv_ws'] =
  torch.empty((4, 32, 2304), fp32) from prepare(); tl.debug_barrier(); one scalar ticket =
  tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['qkv_cnt'] =
  torch.zeros(36, int32) from prepare(); the CTA drawing ticket == SPLIT_K-1 reloads the 4
  partials with cache_modifier='.cg', sums them in fixed order 0..3, casts once to bf16,
  masked-stores [m, 64] into out = torch.empty((m, 2304), bf16) and resets the counter with
  tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm both BLOCK_M binaries in prepare() on synthetic x (m in
  1, 8, 16, 32) then re-zero qkv_cnt. qkv_gate at m >= 128 (and the m=512 outlier test) and all
  other layers keep master's path. This is the router's 3-D split-K design (filled grid, short
  K-slices) applied to qkv_gate, NOT the regressed full-K skinny stream.
expected:
  - shapes: layer=qkv_gate load=decode_1 *
    direction: faster
    why: "14.4 us: 37.7 MB at 3.5-3.7 TB/s is 10.2-10.8 us in one node; the splitKreduce node and the ~3 us reported-minus-kernel gap go; target <= 12 us"
  - shapes: layer=qkv_gate load=verify_1 *
    direction: faster
    why: m=8, same BLOCK_M=16 binary, 14.0 us
  - shapes: layer=qkv_gate load=decode_16 *
    direction: faster
    why: 13.8 us; nvjet main kernel is 8.1 us here (4.66 TB/s) so the margin is the reduce node and gap only; expect a smaller gain
  - shapes: layer=qkv_gate load=decode_32 *
    direction: flat
    why: BLOCK_M=32 Triton binaries carried a ~2.4 us penalty on unpooler; flat-or-slower is the honest call
  - shapes: layer=qkv_gate load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
  - shapes: "*layer=router*"
    direction: flat
    why: master router untouched; control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-gate-triton-splitk4-blockn64-semaphore

## Change

Built on master (dba4ffd; router untouched). For layer == 'qkv_gate' and cfg m <= 32 only (n=2304, k=8192, 37.7 MB bf16), replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT 8.1-10.8 us + cublasLt::splitKreduce_kernel 1.7-2.0 us; reported 13.8-14.4 us vs a 7.9 us floor, the largest reported-minus-kernel gap of any layer) with ONE Triton launch of a split-K kernel with a bf16 semaphore epilogue: grid = (N // BLOCK_N, SPLIT_K) = (2304/64, 4) = 144 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=4 (K-slice 2048, 16 K-steps), BLOCK_M=16 for m<=16 and 32 for 17..32 (one M-block, rows >= m masked to 0), num_warps=4, num_stages=4 (24-32 KB smem per stage, so 2 CTAs fit per SM and all 144 CTAs are co-resident in one wave); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [N, K] weight (the router's transposed addressing, no repack), fp32 acc via tl.dot(x_tile, w_tile, acc). Epilogue: store the fp32 partial to a persistent scratch state['qkv_ws'] = torch.empty((4, 32, 2304), fp32) from prepare(); tl.debug_barrier(); one scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['qkv_cnt'] = torch.zeros(36, int32) from prepare(); the CTA drawing ticket == SPLIT_K-1 reloads the 4 partials with cache_modifier='.cg', sums them in fixed order 0..3, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 2304), bf16) and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm both BLOCK_M binaries in prepare() on synthetic x (m in 1, 8, 16, 32) then re-zero qkv_cnt. qkv_gate at m >= 128 (and the m=512 outlier test) and all other layers keep master's path. This is the router's 3-D split-K design (filled grid, short K-slices) applied to qkv_gate, NOT the regressed full-K skinny stream.

## Expected

- `layer=qkv_gate load=decode_1 *` faster: 14.4 us: 37.7 MB at 3.5-3.7 TB/s is 10.2-10.8 us in one node; the splitKreduce node and the ~3 us reported-minus-kernel gap go; target <= 12 us
- `layer=qkv_gate load=verify_1 *` faster: m=8, same BLOCK_M=16 binary, 14.0 us
- `layer=qkv_gate load=decode_16 *` faster: 13.8 us; nvjet main kernel is 8.1 us here (4.66 TB/s) so the margin is the reduce node and gap only; expect a smaller gain
- `layer=qkv_gate load=decode_32 *` flat: BLOCK_M=32 Triton binaries carried a ~2.4 us penalty on unpooler; flat-or-slower is the honest call
- `layer=qkv_gate load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=router*` flat: master router untouched; control

## Risk

Why this is not a retry of qkv-gate-triton-skinny-m32 (0.924x, 22-26 us, ~1.6 TB/s): that kernel was full-K with BLOCK_N=16 (each CTA streamed 16 rows x 8192 K with 32 KB in flight per SM). This one uses the two levers the landed router proved and the unpooler control measured: BLOCK_N=64 tiles (3.73 TB/s on unpooler) and a split-K grid with short slices that keeps 144 CTAs each with 4 x 16 KB weight stages in flight (~9 MB fleet-wide, above Little's law for 4.8 TB/s). Main risk: nvjet is unusually efficient on qkv_gate (8.1 us = 4.66 TB/s at m=16), so the bar is high; the gain must come mostly from removing the reduce node and the gap. 144 CTAs on 132 SMs means 12 SMs host two CTAs; if the profile shows a 2x tail, the one-axis retune is SPLIT_K=8/num_stages=3 (288 CTAs, ~2.2 per SM) or BLOCK_N=128/SPLIT_K=8 (144 CTAs, larger tiles). Semaphore correctness risk as in pooler-triton-splitk2-blockn64-semaphore; a broken ordering fails the m=1 qkv_gate test. If both land they should share one kernel; they are minted independently so each attributes on its own.
