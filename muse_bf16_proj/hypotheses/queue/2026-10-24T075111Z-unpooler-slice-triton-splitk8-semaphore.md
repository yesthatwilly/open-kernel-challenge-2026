---
problem: muse_bf16_proj
slug: unpooler-slice-triton-splitk8-semaphore
created: 2026-10-24T075111Z
base: master
area: unpooler GEMM
change: >
  Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the
  sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only, replace
  _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0 us +
  cublasLt::splitKreduce_kernel 1.5 us; 7.0-7.1 us reported vs a 1.7-1.8 us floor, the worst ratio
  to floor of any bf16 shape) with ONE Triton launch shaped like the landed router kernel: grid =
  (1024 // BLOCK_N, SPLIT_K) = (16, 8) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K-slice
  512, 4 K-steps), BLOCK_M=16 (rows >= m masked to 0), num_warps=4, num_stages=3, so each CTA
  reads 64 rows x 512 K x 2 B = 64 KB, the same bytes per CTA as the router's 32 rows x 512 K x
  (hi + lo); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [8192, 4096]
  weight's first 1024 rows (row stride 4096, no copy of the slice, no repack), fp32 acc =
  tl.dot(x_tile, w_tile, acc). Epilogue with a bf16 output and no memset: store the fp32 partial
  to persistent scratch state['unpool_ws'] = torch.empty((8, 16, 1024), fp32) from prepare();
  tl.debug_barrier(); scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel',
  scope='gpu') on state['unpool_cnt'] = torch.zeros(16, int32) from prepare(); the CTA drawing
  ticket == 7 reloads the 8 partials (cache_modifier='.cg'), sums in fixed order, casts once to
  bf16, masked-stores [m, 64] into out = torch.empty((m, 1024), bf16), and resets the counter with
  tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm the binary in prepare() on synthetic x (m in 1, 5, 8)
  and re-zero unpool_cnt afterwards. Unpooler with n == 8192 (m >= 16) and every other layer keep
  master's path.
expected:
  - shapes: layer=unpooler load=decode_1 *
    direction: faster
    why: "7.1 us row: the router kernel moves the same 8.4 MB with 128 CTAs of 64 KB each in 3.3 us; one node instead of nvjet + splitKreduce; target <= 5 us"
  - shapes: layer=unpooler load=verify_1 *
    direction: faster
    why: m=8, same binary, 7.0 us row
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: n=8192 keeps cuBLAS; control
  - shapes: layer=unpooler load=verify_16 *
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

# unpooler-slice-triton-splitk8-semaphore

## Change

Built on master (dba4ffd; router untouched). For layer == 'unpooler' with cfg n == 1024 (the sliced case, m <= 8: k=4096, an 8.4 MB bf16 read of W[:1024]) only, replace _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT 4.0 us + cublasLt::splitKreduce_kernel 1.5 us; 7.0-7.1 us reported vs a 1.7-1.8 us floor, the worst ratio to floor of any bf16 shape) with ONE Triton launch shaped like the landed router kernel: grid = (1024 // BLOCK_N, SPLIT_K) = (16, 8) = 128 CTAs, BLOCK_N=64, BLOCK_K=128, SPLIT_K=8 (K-slice 512, 4 K-steps), BLOCK_M=16 (rows >= m masked to 0), num_warps=4, num_stages=3, so each CTA reads 64 rows x 512 K x 2 B = 64 KB, the same bytes per CTA as the router's 32 rows x 512 K x (hi + lo); weight tiles read as [BLOCK_K, BLOCK_N] directly from the row-major [8192, 4096] weight's first 1024 rows (row stride 4096, no copy of the slice, no repack), fp32 acc = tl.dot(x_tile, w_tile, acc). Epilogue with a bf16 output and no memset: store the fp32 partial to persistent scratch state['unpool_ws'] = torch.empty((8, 16, 1024), fp32) from prepare(); tl.debug_barrier(); scalar ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel', scope='gpu') on state['unpool_cnt'] = torch.zeros(16, int32) from prepare(); the CTA drawing ticket == 7 reloads the 8 partials (cache_modifier='.cg'), sums in fixed order, casts once to bf16, masked-stores [m, 64] into out = torch.empty((m, 1024), bf16), and resets the counter with tl.atomic_xchg(cnt_ptr + pid_n, 0). Warm the binary in prepare() on synthetic x (m in 1, 5, 8) and re-zero unpool_cnt afterwards. Unpooler with n == 8192 (m >= 16) and every other layer keep master's path.

## Expected

- `layer=unpooler load=decode_1 *` faster: 7.1 us row: the router kernel moves the same 8.4 MB with 128 CTAs of 64 KB each in 3.3 us; one node instead of nvjet + splitKreduce; target <= 5 us
- `layer=unpooler load=verify_1 *` faster: m=8, same binary, 7.0 us row
- `layer=unpooler load=decode_16 *` flat: n=8192 keeps cuBLAS; control
- `layer=unpooler load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=router*` flat: master router untouched; control

## Risk

Not a retry of unpooler-slice-triton-gemv (0.880x): that one used an FMA broadcast GEMV without tl.dot and ran 70.8 us at m=8. This copies the only Triton design proven fast on an 8.4 MB read here (the landed router: tl.dot, BLOCK_K=128, K-slice 512, 128 CTAs, 3 stages, 3.3 us kernel) and changes only what the bf16 output needs (semaphore reduce instead of zeros + fp32 atomics + a cast kernel, which would be 3 nodes and lose). Per-shape ~30% on 2 shapes, each ~13% of its load: ~+4% on decode_1 and verify_1, ~+1.3% geomean. Risks: semaphore ordering correctness (the m=5 unpooler test and the m=7.. tests exercise it); the serial last-CTA reduce reads 8 x 16 x 64 x 4 B = 32 KB, ~0.3 us; the row stride is 4096 while the weight tensor is [8192, 4096], so the kernel must take stride_wn from the full weight, not assume a contiguous slice. If it lands, the same kernel is the shared base for pooler-triton-splitk2-blockn64-semaphore and qkv-gate-triton-splitk4-blockn64-semaphore.
