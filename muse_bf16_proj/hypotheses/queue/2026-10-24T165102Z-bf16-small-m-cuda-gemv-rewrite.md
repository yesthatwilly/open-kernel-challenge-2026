---
problem: muse_bf16_proj
slug: bf16-small-m-cuda-gemv-rewrite
created: 2026-10-24T165102Z
base: master
area: bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream)
change: >
  Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the
  incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; attention's only >Triton
  bandwidth came from a load_inline CUDA kernel (full-decode-cuda-gather-splitkv, compiled at
  import in 41 s). Stage 1: one hand-written CUDA kernel, built with
  torch.utils.cpp_extension.load_inline at import (no network, cached in the run's own build dir;
  on any compile/load failure print '[gemv] load_inline failed' and keep master's routing), for
  layer in {qkv_gate, o, pooler} at 1 <= cfg m <= 8 and the unpooler slice (cfg n == 1024, m <= 8,
  the first 1024 rows of the full weight read in place, row stride 4096). Design: swap-AB
  mma.sync.aligned.m16n8k16 bf16 -> fp32 with the WEIGHT as the A operand (16 weight rows per MMA)
  and the <= 8 tokens as B (n8 exactly covers m <= 8; columns >= m zero-filled), so no MMA work is
  wasted on padding rows. Grid: persistent, 2 CTAs per SM (264), each CTA owns a contiguous range
  of (64-weight-row x K-chunk) work items; split-K only where N/64 < 264 work items need it (qkv
  36 n-tiles x K-split 8; pooler 64 x 4; o 128 x 2; unpooler slice 16 x 16), chosen statically per
  layer. Weight bytes in flight: a 4-stage cp.async.cg (16-byte, L1-bypass) or cp.async.bulk ring
  in shared memory of >= 32 KB per stage per CTA (>= 256 KB in flight per SM, double master's
  Triton 128 KB per CTA at 1 CTA/SM), consumed by 4 warps each owning 16 weight rows, ldmatrix
  from a swizzled tile to avoid bank conflicts. x (<= 8 x 8192 bf16, 128 KB) is loaded once per
  CTA's K-range into shared memory. Reduce: fp32 partials into a prepare()-allocated workspace
  [SPLIT_K, 8, n] with master's ticket semaphore (store, __threadfence, one atomicAdd ticket per
  n-tile, last CTA sums in fixed order, writes bf16, resets its counter). prepare(): compile
  check, allocate workspaces/counters (zeroed), warm m in 1..8 per layer, run a self-check against
  x @ W.T on synthetic x (rel L2 <= 1e-3, else print 'SELF-CHECK FAILED' and disable the path; a
  constant fixed in prepare, never input-derived), and print per layer the warm-L2 cudaEvent
  kernel time and effective TB/s for m=1 and m=8. kernel_fn routes only those shapes; m >= 9 and
  every router shape keep master.
expected:
  - shapes: layer=pooler m=1
    direction: faster
    why: "incumbent kernel 19.0 us (67.1 MB, 3.5 TB/s) vs floor 14.0. Milestones: M1 correct and self-check passes; M2 pooler m=1 kernel <= 17.5 us (>= 3.85 TB/s, past the Triton ceiling); M3 <= 16 us (4.2 TB/s)"
  - shapes: layer=pooler m=8
    direction: faster
    why: incumbent 19.3 us; same milestones (m8 MMA is free under the BW bound)
  - shapes: layer=qkv_gate m=1
    direction: faster
    why: incumbent 11.8 us (37.7 MB, 3.2 TB/s) vs floor 7.9; M2 <= 10.5 us
  - shapes: layer=qkv_gate m=8
    direction: faster
    why: incumbent 12.0 us; M2 <= 10.5 us
  - shapes: layer=o m=1
    direction: faster
    why: incumbent 4.6 us (16.8 MB, 3.65 TB/s) vs floor 3.5; M2 <= 4.2 us
  - shapes: layer=o m=8
    direction: faster
    why: incumbent 5.2 us; M2 <= 4.4 us
  - shapes: layer=unpooler m=1
    direction: faster
    why: slice incumbent 4.5 us (8.4 MB, 1.9 TB/s) vs floor 1.7; latency-bound, so 264 CTAs with all bytes issued up front is the lever; M2 <= 3.2 us
  - shapes: layer=unpooler m=8
    direction: faster
    why: slice incumbent 4.6 us; M2 <= 3.3 us
  - shapes: "* m=16"
    direction: flat
    why: master path untouched
  - shapes: "* m=32"
    direction: flat
    why: untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "Triton m<=8 bf16 paths: pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler-slice _bf16_splitk_kernel (decode_1 m=1 and verify_1 m=8)"
review: required
status: queued
---

# bf16-small-m-cuda-gemv-rewrite

## Change

Built on master (8b9ad43). LESSONS: Triton weight streams top out at ~3.7 TB/s here, and the incumbent m<=8 kernels run 1.9-3.6 TB/s against 4.8 TB/s floors; attention's only >Triton bandwidth came from a load_inline CUDA kernel (full-decode-cuda-gather-splitkv, compiled at import in 41 s). Stage 1: one hand-written CUDA kernel, built with torch.utils.cpp_extension.load_inline at import (no network, cached in the run's own build dir; on any compile/load failure print '[gemv] load_inline failed' and keep master's routing), for layer in {qkv_gate, o, pooler} at 1 <= cfg m <= 8 and the unpooler slice (cfg n == 1024, m <= 8, the first 1024 rows of the full weight read in place, row stride 4096). Design: swap-AB mma.sync.aligned.m16n8k16 bf16 -> fp32 with the WEIGHT as the A operand (16 weight rows per MMA) and the <= 8 tokens as B (n8 exactly covers m <= 8; columns >= m zero-filled), so no MMA work is wasted on padding rows. Grid: persistent, 2 CTAs per SM (264), each CTA owns a contiguous range of (64-weight-row x K-chunk) work items; split-K only where N/64 < 264 work items need it (qkv 36 n-tiles x K-split 8; pooler 64 x 4; o 128 x 2; unpooler slice 16 x 16), chosen statically per layer. Weight bytes in flight: a 4-stage cp.async.cg (16-byte, L1-bypass) or cp.async.bulk ring in shared memory of >= 32 KB per stage per CTA (>= 256 KB in flight per SM, double master's Triton 128 KB per CTA at 1 CTA/SM), consumed by 4 warps each owning 16 weight rows, ldmatrix from a swizzled tile to avoid bank conflicts. x (<= 8 x 8192 bf16, 128 KB) is loaded once per CTA's K-range into shared memory. Reduce: fp32 partials into a prepare()-allocated workspace [SPLIT_K, 8, n] with master's ticket semaphore (store, __threadfence, one atomicAdd ticket per n-tile, last CTA sums in fixed order, writes bf16, resets its counter). prepare(): compile check, allocate workspaces/counters (zeroed), warm m in 1..8 per layer, run a self-check against x @ W.T on synthetic x (rel L2 <= 1e-3, else print 'SELF-CHECK FAILED' and disable the path; a constant fixed in prepare, never input-derived), and print per layer the warm-L2 cudaEvent kernel time and effective TB/s for m=1 and m=8. kernel_fn routes only those shapes; m >= 9 and every router shape keep master.

## Expected

- `layer=pooler m=1` faster: incumbent kernel 19.0 us (67.1 MB, 3.5 TB/s) vs floor 14.0. Milestones: M1 correct and self-check passes; M2 pooler m=1 kernel <= 17.5 us (>= 3.85 TB/s, past the Triton ceiling); M3 <= 16 us (4.2 TB/s)
- `layer=pooler m=8` faster: incumbent 19.3 us; same milestones (m8 MMA is free under the BW bound)
- `layer=qkv_gate m=1` faster: incumbent 11.8 us (37.7 MB, 3.2 TB/s) vs floor 7.9; M2 <= 10.5 us
- `layer=qkv_gate m=8` faster: incumbent 12.0 us; M2 <= 10.5 us
- `layer=o m=1` faster: incumbent 4.6 us (16.8 MB, 3.65 TB/s) vs floor 3.5; M2 <= 4.2 us
- `layer=o m=8` faster: incumbent 5.2 us; M2 <= 4.4 us
- `layer=unpooler m=1` faster: slice incumbent 4.5 us (8.4 MB, 1.9 TB/s) vs floor 1.7; latency-bound, so 264 CTAs with all bytes issued up front is the lever; M2 <= 3.2 us
- `layer=unpooler m=8` faster: slice incumbent 4.6 us; M2 <= 3.3 us
- `* m=16` flat: master path untouched
- `* m=32` flat: untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched
- `layer=router *` flat: untouched

## Risk

High risk: a hand-written kernel, judged on the replaced kernels' times against its milestones, not the headline. Compile time counts toward the 10-minute run (attention's load_inline took 41 s); keep one translation unit, one template instantiated per layer. Overlaps bf16-m16-swapab-wgmma-extend on the m=1/m=8 shapes: build on master as written, and if that Triton swap-AB lands first, measure stage 2 against it. The read rules allow the bf16 weights unchanged (no repack needed); any swizzled repack in prepare() must stay bf16 and lossless. Later stages: extend to m=16 with n16 (two m16n8 B fragments), and try TMA 1-D bulk copies if cp.async stalls below 4 TB/s.

## Milestones

Rewrite stage 1 of `Triton m<=8 bf16 paths: pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler-slice _bf16_splitk_kernel (decode_1 m=1 and verify_1 m=8)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
