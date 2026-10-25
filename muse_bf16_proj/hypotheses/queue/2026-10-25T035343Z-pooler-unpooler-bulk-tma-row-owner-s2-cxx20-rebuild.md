---
problem: muse_bf16_proj
slug: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild
created: 2026-10-25T035343Z
base: master
area: pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel)
change: >
  Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built
  on its commit (8ba8624, claim bases a rewrite-on-rewrite on the parent commit); no design
  change. Stage 1 (job cd39d36e) never compiled: the nvcc compile of the .cu failed with "#error
  C++20 or later compatible compiler is required to use PyTorch", then "[bulk] DISABLED: no
  extension; master paths", so its 1.183 is master plus batch drift and the design is unmeasured.
  The host main.o compile carried -std=c++20, but the .cu was given only
  -gencode=arch=compute_90a,code=sm_90a. Change: add "-std=c++20" to that load_inline call
  extra_cuda_cflags (keep -gencode arch=compute_90a,code=sm_90a, -O3, -lineinfo; keep
  TORCH_CUDA_ARCH_LIST=9.0a), and rename the module so it rebuilds. Wrap the load_inline in the
  existing try/except so that on a build failure it prints the LAST ~60 LINES OF THE NVCC STDERR
  (the exception text / build log tail, not only the first #error) before "[bulk] DISABLED", so a
  second build error is named in one check. After the build, before any other work, print ptxas
  regs/spill/smem per kernel and assert the PATH CHOSEN line names the bulk kernel for each
  (layer, m) it gates; keep the existing self-check (rel L2 vs master) and fallback unchanged.
  Everything else (repack, ring, gates, other layers) is stage 1 code byte for byte. First
  measurement is an okc check, never a scored submit.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: incumbent 19.0 us (_bf16_splitk_kernel ~3.5 TB/s); floor 14.0; M1 builds and PATH CHOSEN = bulk; M2 below 19.0; M3 <= 16.5
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: incumbent 19.14; floor 14.0
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: incumbent 19.85; floor 14.1
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 20.0 row; floor 14.1
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: incumbent nvjet 18.5; floor 14.1
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 18.7; floor 14.1
  - shapes: layer=qkv_gate * layer=o * layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: pooler m<=32 (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m=32) and unpooler-full m=16/32 (nvjet 64x16, _bf16_swapab_kernel)
review: required
status: queued
---

# pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild

## Change

Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built on its commit (8ba8624, claim bases a rewrite-on-rewrite on the parent commit); no design change. Stage 1 (job cd39d36e) never compiled: the nvcc compile of the .cu failed with "#error C++20 or later compatible compiler is required to use PyTorch", then "[bulk] DISABLED: no extension; master paths", so its 1.183 is master plus batch drift and the design is unmeasured. The host main.o compile carried -std=c++20, but the .cu was given only -gencode=arch=compute_90a,code=sm_90a. Change: add "-std=c++20" to that load_inline call extra_cuda_cflags (keep -gencode arch=compute_90a,code=sm_90a, -O3, -lineinfo; keep TORCH_CUDA_ARCH_LIST=9.0a), and rename the module so it rebuilds. Wrap the load_inline in the existing try/except so that on a build failure it prints the LAST ~60 LINES OF THE NVCC STDERR (the exception text / build log tail, not only the first #error) before "[bulk] DISABLED", so a second build error is named in one check. After the build, before any other work, print ptxas regs/spill/smem per kernel and assert the PATH CHOSEN line names the bulk kernel for each (layer, m) it gates; keep the existing self-check (rel L2 vs master) and fallback unchanged. Everything else (repack, ring, gates, other layers) is stage 1 code byte for byte. First measurement is an okc check, never a scored submit.

## Expected

- `layer=pooler load=decode_1 m=1` faster: incumbent 19.0 us (_bf16_splitk_kernel ~3.5 TB/s); floor 14.0; M1 builds and PATH CHOSEN = bulk; M2 below 19.0; M3 <= 16.5
- `layer=pooler load=verify_1 m=8` faster: incumbent 19.14; floor 14.0
- `layer=pooler load=decode_16 m=16` faster: incumbent 19.85; floor 14.1
- `layer=pooler load=decode_32 m=32` faster: incumbent swap-AB 20.0 row; floor 14.1
- `layer=unpooler load=decode_16 m=16` faster: incumbent nvjet 18.5; floor 14.1
- `layer=unpooler load=decode_32 m=32` faster: incumbent swap-AB 18.7; floor 14.1
- `layer=qkv_gate * layer=o * layer=router *` flat: untouched

## Risk

Base is the stage-1 commit on master 0a2ef67, not the unlanded restack. Read untouched small-m rows against batch drift (+2-6% in the 03:41Z harvest); only rows whose kernel changed count. The design itself is the risk: every earlier stream (Triton, TMA tiles, CUDA gemv) stopped near 3.5 TB/s.

## Milestones

Rewrite stage 2 of `pooler m<=32 (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m=32) and unpooler-full m=16/32 (nvjet 64x16, _bf16_swapab_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
