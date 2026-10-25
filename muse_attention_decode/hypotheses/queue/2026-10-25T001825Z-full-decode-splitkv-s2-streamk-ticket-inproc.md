---
problem: muse_attention_decode
slug: full-decode-splitkv-s2-streamk-ticket-inproc
created: 2026-10-25T001825Z
base: master
area: full-layer attention (decode, splitkv kernel A/B)
change: >
  Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A +
  splitkv_combine_kernel B in load_inline module muse_full_decode_splitkv_v1; gate _splitkv_gate =
  full, q_len 1, 1 < batch <= 64, tp8, page 1 is unchanged). Per REWRITE_READ_2026-10-25 section
  1, keep the 2-CTA/SM, 256-thread, all-thread cp.async.cg 4-stage ring and the m16n8k16 f16
  compute exactly; change only the work split and the epilogue, in a NEW module
  muse_full_decode_splitkv_v2 (new kernel splitkv_attn_sk_kernel; stage-1 A/B stay compiled in v1
  as the fallback). (1) STREAM-K over 64-token tiles instead of 1024-token chunks: every CTA loads
  cache_seqlens[0..B-1] to smem, computes tiles_b = ceil(seq_b/64) and the prefix TP[b] (TP[B] =
  Ttot), and owns the flat contiguous tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math),
  which may span requests; page indices for the next 16 tiles are prefetched into double-buffered
  smem with 4-byte cp.async (zero-fill past each request end) and the ring runs across segment
  boundaries (q rows of the next request fetched into a second smem slot when a segment starts).
  The contiguous per-CTA range is also the L2/TLB-locality change for decode_32 (each CTA walks
  one contiguous page-table run instead of strided 1024-chunks). Each CTA writes one fp32 partial
  (acc[16,64], m[16], l[16]) per request segment at slot c + b (slots <= grid + B; workspace
  allocated per call from grid and batch); a segment that covers a whole request skips the
  workspace. (2) FOLD COMBINE + NORM-GATE INTO A with a per-request ticket: after writing a
  partial, __threadfence(), then one thread does atomicAdd(&ticket[b], 1); the CTA that draws
  nseg_b - 1 (nseg_b = number of CTAs whose range intersects request b, recomputed from TP) does
  __threadfence(), LSE-merges the request's slots (exact inverse c_lo..c_hi of floor(c*Ttot/grid),
  empty segments weight 0), times v_scale, then the stage-1 B epilogue math (scaleless RMSNorm
  over 64, eps 1e-5, times sigmoid(gate) from qkv, fp32) and stores bf16 to the [B,1,1024] output,
  then resets ticket[b] = 0 so graph replay starts clean. ticket is a module-level int32[64] zero
  tensor allocated once at import (not derived from inputs). Launches on full decode_16/32 drop
  from three (pre-attn, A, B) to two. (3) Decode_32 bandwidth: grid stays 2 x SM count and 4
  stages for the timed path; the import-time self-check additionally prints eager diagnostic
  timings at synthetic B=16 and B=32 x 80000 of stage-1 A+B, stream-K without ticket (+B), and
  stream-K with ticket, and of 5 stages (80 KB ring, still 2 CTA/SM) — diagnostics only, they do
  not choose the path. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (-gencode
  arch=compute_90a,code=sm_90a only), -lineinfo; run the first launch and the self-check (B in
  {2,16,32,64}, ragged lengths incl. seq % 64 in 1..7, lengths < 64, one request spanning every
  CTA, kv_scale 1 and 0.25, grids 3/7/device, two consecutive replays to prove the ticket reset)
  IN-PROCESS (never in a child process: the rules statically reject submissions that start
  processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the
  sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc
  -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for splitkv_attn_sk_kernel; anything else turns the path OFF), then a
  tiny synthetic canary launch of splitkv_attn_sk_kernel (synthetic B=2, short lengths), each
  followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the
  self-check; if the build, the image check, the canary or the self-check fails, fall back to
  master (stage-1 A+B) and print 'PATH CHOSEN: full decode splitkv v1 (fallback: <reason>)'; on
  success print 'PATH CHOSEN: full decode streamk+ticket v2'. Every other shape keeps master code
  byte for byte. The first measurement of this hypothesis must be an `okc check`, never a scored
  submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced kernels master A+B 111.4+7.9 = 119.3 us, floor 63.2. M1: 16/16 tests and <= 119.3. M2: <= 108 us (B launch gone ~-8, tail removed). M3: <= 95 us (1.5x floor)"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced kernels master A+B 245.5+8.6 = 254.1 us at 2.67 TB/s, floor 126.4. M1: <= 254.1. M2: <= 222 us (ticket -8.6 plus contiguous stream-K ranges back to ~2.9 TB/s). M3: <= 190 us"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside _splitkv_gate
risk: medium
depends_on: full-decode-cuda-gather-splitkv
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)"
review: required
status: queued
---

# full-decode-splitkv-s2-streamk-ticket-inproc

## Change

Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A + splitkv_combine_kernel B in load_inline module muse_full_decode_splitkv_v1; gate _splitkv_gate = full, q_len 1, 1 < batch <= 64, tp8, page 1 is unchanged). Per REWRITE_READ_2026-10-25 section 1, keep the 2-CTA/SM, 256-thread, all-thread cp.async.cg 4-stage ring and the m16n8k16 f16 compute exactly; change only the work split and the epilogue, in a NEW module muse_full_decode_splitkv_v2 (new kernel splitkv_attn_sk_kernel; stage-1 A/B stay compiled in v1 as the fallback). (1) STREAM-K over 64-token tiles instead of 1024-token chunks: every CTA loads cache_seqlens[0..B-1] to smem, computes tiles_b = ceil(seq_b/64) and the prefix TP[b] (TP[B] = Ttot), and owns the flat contiguous tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math), which may span requests; page indices for the next 16 tiles are prefetched into double-buffered smem with 4-byte cp.async (zero-fill past each request end) and the ring runs across segment boundaries (q rows of the next request fetched into a second smem slot when a segment starts). The contiguous per-CTA range is also the L2/TLB-locality change for decode_32 (each CTA walks one contiguous page-table run instead of strided 1024-chunks). Each CTA writes one fp32 partial (acc[16,64], m[16], l[16]) per request segment at slot c + b (slots <= grid + B; workspace allocated per call from grid and batch); a segment that covers a whole request skips the workspace. (2) FOLD COMBINE + NORM-GATE INTO A with a per-request ticket: after writing a partial, __threadfence(), then one thread does atomicAdd(&ticket[b], 1); the CTA that draws nseg_b - 1 (nseg_b = number of CTAs whose range intersects request b, recomputed from TP) does __threadfence(), LSE-merges the request's slots (exact inverse c_lo..c_hi of floor(c*Ttot/grid), empty segments weight 0), times v_scale, then the stage-1 B epilogue math (scaleless RMSNorm over 64, eps 1e-5, times sigmoid(gate) from qkv, fp32) and stores bf16 to the [B,1,1024] output, then resets ticket[b] = 0 so graph replay starts clean. ticket is a module-level int32[64] zero tensor allocated once at import (not derived from inputs). Launches on full decode_16/32 drop from three (pre-attn, A, B) to two. (3) Decode_32 bandwidth: grid stays 2 x SM count and 4 stages for the timed path; the import-time self-check additionally prints eager diagnostic timings at synthetic B=16 and B=32 x 80000 of stage-1 A+B, stream-K without ticket (+B), and stream-K with ticket, and of 5 stages (80 KB ring, still 2 CTA/SM) — diagnostics only, they do not choose the path. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (-gencode arch=compute_90a,code=sm_90a only), -lineinfo; run the first launch and the self-check (B in {2,16,32,64}, ragged lengths incl. seq % 64 in 1..7, lengths < 64, one request spanning every CTA, kv_scale 1 and 0.25, grids 3/7/device, two consecutive replays to prove the ticket reset) IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for splitkv_attn_sk_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of splitkv_attn_sk_kernel (synthetic B=2, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check; if the build, the image check, the canary or the self-check fails, fall back to master (stage-1 A+B) and print 'PATH CHOSEN: full decode splitkv v1 (fallback: <reason>)'; on success print 'PATH CHOSEN: full decode streamk+ticket v2'. Every other shape keeps master code byte for byte.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=full load=decode_16*` faster: replaced kernels master A+B 111.4+7.9 = 119.3 us, floor 63.2. M1: 16/16 tests and <= 119.3. M2: <= 108 us (B launch gone ~-8, tail removed). M3: <= 95 us (1.5x floor)
- `*layer=full load=decode_32*` faster: replaced kernels master A+B 245.5+8.6 = 254.1 us at 2.67 TB/s, floor 126.4. M1: <= 254.1. M2: <= 222 us (ticket -8.6 plus contiguous stream-K ranges back to ~2.9 TB/s). M3: <= 190 us
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside _splitkv_gate

## Risk

supersedes full-decode-splitkv-s2-streamk-ticket, whose change requires a child-process self-check

## Milestones

Rewrite stage 2 of `full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
