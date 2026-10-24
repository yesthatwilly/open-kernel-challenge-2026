---
problem: muse_bf16_proj
slug: bf16-restack-m16-swapab-router-swapab-m32
created: 2026-10-24T174421Z
base: master
area: restack (m<=16 swap-AB GEMMs + router swap-AB)
change: >
  Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a
  3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16
  kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3.
  Its patch is in done/neutral/bf16-m16-swapab-wgmma-extend.patch on branch
  hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend. (2) router-swapab-wgmma-all-m, but gated to
  m<=32 only. Above that, keep master's BLOCK_M=64 hilo router kernel, which already emits wgmma
  (the m=128/256 rows read -2%/flat). Patch on branch
  hyp/muse_bf16_proj/router-swapab-wgmma-all-m. The harvest analysis found the merge-file of the
  two clean. Do NOT include bf16-cublaslt-algo-pick (it lost at m=256). No other change.
expected:
  - shapes: layer=qkv_gate load=* m=1
    direction: faster
    why: m16 swap-AB, qkv 18% faster kernel
  - shapes: layer=o load=* m=8
    direction: faster
    why: m16 swap-AB
  - shapes: layer=router load=* m=32
    direction: faster
    why: router swap-AB +3-5% at m<=32
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-restack-m16-swapab-router-swapab-m32

## Change

Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16 kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3. Its patch is in done/neutral/bf16-m16-swapab-wgmma-extend.patch on branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend. (2) router-swapab-wgmma-all-m, but gated to m<=32 only. Above that, keep master's BLOCK_M=64 hilo router kernel, which already emits wgmma (the m=128/256 rows read -2%/flat). Patch on branch hyp/muse_bf16_proj/router-swapab-wgmma-all-m. The harvest analysis found the merge-file of the two clean. Do NOT include bf16-cublaslt-algo-pick (it lost at m=256). No other change.

## Expected

- `layer=qkv_gate load=* m=1` faster: m16 swap-AB, qkv 18% faster kernel
- `layer=o load=* m=8` faster: m16 swap-AB
- `layer=router load=* m=32` faster: router swap-AB +3-5% at m<=32
- `layer=* load=verify_32 m=256` flat: untouched

## Risk

Restack of measured pieces; expected headline about 1.215-1.222 vs 1.196 (harvest analysis). Run as a scored experiment; if it gains, it is the board entry.
