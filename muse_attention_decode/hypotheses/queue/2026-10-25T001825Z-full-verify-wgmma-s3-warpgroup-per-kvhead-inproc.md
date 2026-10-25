---
problem: muse_attention_decode
slug: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
created: 2026-10-25T001825Z
base: master
area: full-layer attention (verify, splitkv A8/B8)
change: >
  Stage 3 of the full-attention rewrite, built on the stage-2 code (branch
  hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebased onto
  current master; stage 2 regressed so it is not on master). Gate unchanged: _splitkv_verify_gate
  = full, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1. Keep stage 2's stream-K over
  64-token tiles, TP prefix, page-index prefetch, slot layout and masking math; replace the
  per-tile compute of A8 per REWRITE_READ_2026-10-25 section 2. New kernel
  splitkv_verify_wgmma_kernel in its OWN load_inline module muse_full_verify_wgmma_v1 (so a failed
  build cannot disable master's decode module), 256 threads = 2 warpgroups, __launch_bounds__(256,
  2), grid 2 x SM. WARPGROUP g OWNS KV HEAD g: its 64 rows (8 q heads x 8 block tokens) are
  exactly one wgmma M=64 tile. RING: all 256 threads still issue the 16-byte cp.async.cg gather
  for each 64-token tile (4-stage, 128 B K row + 128 B V row per token), but stores use the Hopper
  128B swizzle (16-byte chunk index XOR (row & 7), 1024-byte-aligned stage base), and completion
  is tracked by a per-stage FULL mbarrier (cp.async.mbarrier.arrive.noinc) and a per-stage EMPTY
  mbarrier both warpgroups arrive on when done; the thread issuing tile n+4 waits on EMPTY[stage].
  There is NO __syncthreads inside the tile loop. QK^T: wgmma.mma_async m64n64k32 f32.e4m3.e4m3, A
  = the request's fp8 Q block for head g from smem (64 rows x 64 dims, K-major, 64B swizzle,
  loaded once per segment), B = the K tile from the ring via descriptor with start offset 64*g
  bytes inside the 128-byte swizzled row (k-steps 2g, 2g+1 of the row, as CUTLASS advances
  descriptors within a swizzle atom); 2 instructions per tile; wgmma.fence / commit_group /
  wait_group 0 before reading S. Mask (stage 2 causal t <= seq_b - 8 + j, applied on any tile
  overlapping [seq_b - 8, seq_b), not just the last), online softmax exp2 domain per row from the
  wgmma accumulator layout (quad shuffles). PV in FP8 (as FA3): P rounded to e4m3 and
  byte-permuted (prmt) from the C-fragment layout to the wgmma register-A layout as FA3's fp8 path
  does; B = V^T: each warpgroup transposes ITS OWN head's 64x64 fp8 V half of the tile into a
  private 4 KB Vt smem buffer (tokens contiguous per dim, 128B swizzle; 32-bit loads + prmt +
  32-bit stores, double-buffered) synchronised by a 128-thread named barrier (bar.sync id, 128),
  never a CTA barrier; wgmma m64n64k32 with A from registers, B = Vt descriptor, O accumulated in
  fp32 (32 regs). This drops stage 2's cooperative f16 V-widen, its extra __syncthreads per tile,
  and each warp's full 48-mma tile. SHRINK B8: fold the combine + norm-gate into the kernel with a
  per-request ticket (atomicAdd after __threadfence; the last segment's CTA LSE-merges its
  request's slots, times v_scale, scaleless RMSNorm over 64 + sigmoid(gate), stores bf16 [B*8,
  1024], resets ticket[b] = 0; ticket is a module-level int32[64] zero buffer from import) so B8's
  launch and its (grid+B) x 8x16x64 fp32 re-read of all slots by a separate kernel go away; if the
  ticket path fails its self-check, keep stage 2's B8 and print that. BUILD/SAFETY (LESSONS
  2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a only (-gencode arch=compute_90a,code=sm_90a), all
  wgmma/mbarrier code under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets a
  device error flag and returns (no __trap), the flag read back after the canary (non-zero ->
  fallback), -lineinfo; print ptxas regs/smem/local bytes; first launch + self-check (stage 2's
  verify cases incl. grids 3/7/61, seq % 64 in 1..7, lengths < 8, kv_scale 1 and 0.25, two replays
  for the ticket reset) run IN-PROCESS (never in a child process: the rules statically reject
  submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25,
  corrected)): first confirm the sm_90a image actually loaded (log
  torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for
  splitkv_verify_wgmma_kernel; anything else turns the path OFF), then a tiny synthetic canary
  launch of splitkv_verify_wgmma_kernel (synthetic B=2, short lengths), each followed by
  torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any
  failure: full verify falls back to master (unpaired FA3 path), printing 'PATH CHOSEN: full
  verify master FA3 (fallback: <reason>)'; success prints 'PATH CHOSEN: full verify wgmma stream-K
  v1 (ticket|B8)'. Full decode (master splitkv v1) and all sliding shapes stay byte for byte. The
  first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky
  fault cannot be recovered in-process).
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "incumbent (master FA3 main+combine+prepare+norm-gate) 162.7 us, stage 2 A8+B8 241.7, floor 63.4. M1: 16/16 tests and < 241.7. M2 (read's gate): main kernel <= 128 us (1.15x stage-1 decode A) and replaced total < 162.7 (beats master). M3: <= 95 us (1.5x floor)"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "incumbent 301.3 us, stage 2 465.5, floor 126.8. M1: < 465.5. M2: main kernel <= 282 us and total < 301.3. M3: <= 190 us"
  - shapes: "*load=decode_* *layer=sliding* *layer=full load=verify_1 *"
    direction: flat
    why: outside _splitkv_verify_gate; master code
risk: high
depends_on: full-verify-cuda-streamk-splitkv
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8"
review: required
status: queued
---

# full-verify-wgmma-s3-warpgroup-per-kvhead-inproc

## Change

Stage 3 of the full-attention rewrite, built on the stage-2 code (branch hyp/muse_attention_decode/full-verify-cuda-streamk-splitkv, impl_commit ee7dd20, rebased onto current master; stage 2 regressed so it is not on master). Gate unchanged: _splitkv_verify_gate = full, q_len 8, 1 < batch <= 64, tp8, head_dim 64, page 1. Keep stage 2's stream-K over 64-token tiles, TP prefix, page-index prefetch, slot layout and masking math; replace the per-tile compute of A8 per REWRITE_READ_2026-10-25 section 2. New kernel splitkv_verify_wgmma_kernel in its OWN load_inline module muse_full_verify_wgmma_v1 (so a failed build cannot disable master's decode module), 256 threads = 2 warpgroups, __launch_bounds__(256, 2), grid 2 x SM. WARPGROUP g OWNS KV HEAD g: its 64 rows (8 q heads x 8 block tokens) are exactly one wgmma M=64 tile. RING: all 256 threads still issue the 16-byte cp.async.cg gather for each 64-token tile (4-stage, 128 B K row + 128 B V row per token), but stores use the Hopper 128B swizzle (16-byte chunk index XOR (row & 7), 1024-byte-aligned stage base), and completion is tracked by a per-stage FULL mbarrier (cp.async.mbarrier.arrive.noinc) and a per-stage EMPTY mbarrier both warpgroups arrive on when done; the thread issuing tile n+4 waits on EMPTY[stage]. There is NO __syncthreads inside the tile loop. QK^T: wgmma.mma_async m64n64k32 f32.e4m3.e4m3, A = the request's fp8 Q block for head g from smem (64 rows x 64 dims, K-major, 64B swizzle, loaded once per segment), B = the K tile from the ring via descriptor with start offset 64*g bytes inside the 128-byte swizzled row (k-steps 2g, 2g+1 of the row, as CUTLASS advances descriptors within a swizzle atom); 2 instructions per tile; wgmma.fence / commit_group / wait_group 0 before reading S. Mask (stage 2 causal t <= seq_b - 8 + j, applied on any tile overlapping [seq_b - 8, seq_b), not just the last), online softmax exp2 domain per row from the wgmma accumulator layout (quad shuffles). PV in FP8 (as FA3): P rounded to e4m3 and byte-permuted (prmt) from the C-fragment layout to the wgmma register-A layout as FA3's fp8 path does; B = V^T: each warpgroup transposes ITS OWN head's 64x64 fp8 V half of the tile into a private 4 KB Vt smem buffer (tokens contiguous per dim, 128B swizzle; 32-bit loads + prmt + 32-bit stores, double-buffered) synchronised by a 128-thread named barrier (bar.sync id, 128), never a CTA barrier; wgmma m64n64k32 with A from registers, B = Vt descriptor, O accumulated in fp32 (32 regs). This drops stage 2's cooperative f16 V-widen, its extra __syncthreads per tile, and each warp's full 48-mma tile. SHRINK B8: fold the combine + norm-gate into the kernel with a per-request ticket (atomicAdd after __threadfence; the last segment's CTA LSE-merges its request's slots, times v_scale, scaleless RMSNorm over 64 + sigmoid(gate), stores bf16 [B*8, 1024], resets ticket[b] = 0; ticket is a module-level int32[64] zero buffer from import) so B8's launch and its (grid+B) x 8x16x64 fp32 re-read of all slots by a separate kernel go away; if the ticket path fails its self-check, keep stage 2's B8 and print that. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a only (-gencode arch=compute_90a,code=sm_90a), all wgmma/mbarrier code under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets a device error flag and returns (no __trap), the flag read back after the canary (non-zero -> fallback), -lineinfo; print ptxas regs/smem/local bytes; first launch + self-check (stage 2's verify cases incl. grids 3/7/61, seq % 64 in 1..7, lengths < 8, kv_scale 1 and 0.25, two replays for the ticket reset) run IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for splitkv_verify_wgmma_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of splitkv_verify_wgmma_kernel (synthetic B=2, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check. Any failure: full verify falls back to master (unpaired FA3 path), printing 'PATH CHOSEN: full verify master FA3 (fallback: <reason>)'; success prints 'PATH CHOSEN: full verify wgmma stream-K v1 (ticket|B8)'. Full decode (master splitkv v1) and all sliding shapes stay byte for byte.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=full load=verify_16*` faster: incumbent (master FA3 main+combine+prepare+norm-gate) 162.7 us, stage 2 A8+B8 241.7, floor 63.4. M1: 16/16 tests and < 241.7. M2 (read's gate): main kernel <= 128 us (1.15x stage-1 decode A) and replaced total < 162.7 (beats master). M3: <= 95 us (1.5x floor)
- `*layer=full load=verify_32*` faster: incumbent 301.3 us, stage 2 465.5, floor 126.8. M1: < 465.5. M2: main kernel <= 282 us and total < 301.3. M3: <= 190 us
- `*load=decode_* *layer=sliding* *layer=full load=verify_1 *` flat: outside _splitkv_verify_gate; master code

## Risk

supersedes full-verify-wgmma-s3-warpgroup-per-kvhead, whose change requires a child-process self-check

## Milestones

Rewrite stage 3 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master); stage-2 splitkv_verify_kernel A8 + splitkv_verify_combine B8`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
