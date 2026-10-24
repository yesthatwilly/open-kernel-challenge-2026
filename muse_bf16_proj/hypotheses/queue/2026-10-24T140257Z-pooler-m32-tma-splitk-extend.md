---
problem: muse_bf16_proj
slug: pooler-m32-tma-splitk-extend
created: 2026-10-24T140257Z
base: master
area: pooler GEMM (m=17..32)
change: >
  Built on master (658d63b). Extend the combo's pooler Triton path to 17 <= cfg m <= 32 (the
  scored pooler decode_32 m=32 row) using the ONE m=32 Triton kernel that has measured a win:
  _bf16_splitk_tma_kernel copied verbatim from branch
  hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream (job 4e89311eefdc4bc3: pooler decode_32 20.8
  vs 22.1 us, 1.061x; kernel 21.0 us vs master's nvjet 20.2 + splitKreduce 2.0 us). Keep its
  configuration at BLOCK_M=32: grid (64, 2) = 128 CTAs, BLOCK_N 64, SPLIT_K 2, BLOCK_K 256, the
  weight via a Triton 3.8 TensorDescriptor.from_tensor(W, [64, 256]) built once in prepare() from
  the pooler weight, x a masked tl.load, num_stages 4 (with that branch's OutOfResources retry at
  3, recorded as a constant), and its ticket-semaphore reduce. Master's m <= 16 pooler path
  (_bf16_splitk_kernel, cp.async BLOCK_K 256, from the BLOCK_K=256 control, the combo's pick) is
  unchanged; only a new gate `layer == 'pooler' and 17 <= cfg m <= 32` plus the existing shape
  preconditions routes to the TMA kernel, placed before the cuBLAS fallback. Own state keys:
  state['pooler_tma_desc'], state['pooler_tma_ws'] fp32 [2, 32, 4096], state['pooler_tma_cnt']
  int32 zeros [64]; warm m in 17..32 in prepare() on synthetic x and re-zero the counter after.
  Nothing else changes (qkv_gate/o/unpooler at m=32 stay cuBLAS, their Triton m=32 rows all lost:
  0.858x, 0.907x, 0.91-0.97x).
expected:
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: master 21.6 us (nvjet 20.2 + 2.0 us reduce node) vs floor 14.1; the TMA kernel measured 20.8 us on this row (1.061x paired) in 4e89311eefdc4bc3; Triton reproduces within ~1%
  - shapes: layer=pooler load=decode_1 m=1
    direction: flat
    why: master m<=16 path unchanged
  - shapes: layer=pooler load=decode_16 m=16
    direction: flat
    why: unchanged
  - shapes: layer=pooler load=verify_1 m=8
    direction: flat
    why: unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-m32-tma-splitk-extend

## Change

Built on master (658d63b). Extend the combo's pooler Triton path to 17 <= cfg m <= 32 (the scored pooler decode_32 m=32 row) using the ONE m=32 Triton kernel that has measured a win: _bf16_splitk_tma_kernel copied verbatim from branch hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream (job 4e89311eefdc4bc3: pooler decode_32 20.8 vs 22.1 us, 1.061x; kernel 21.0 us vs master's nvjet 20.2 + splitKreduce 2.0 us). Keep its configuration at BLOCK_M=32: grid (64, 2) = 128 CTAs, BLOCK_N 64, SPLIT_K 2, BLOCK_K 256, the weight via a Triton 3.8 TensorDescriptor.from_tensor(W, [64, 256]) built once in prepare() from the pooler weight, x a masked tl.load, num_stages 4 (with that branch's OutOfResources retry at 3, recorded as a constant), and its ticket-semaphore reduce. Master's m <= 16 pooler path (_bf16_splitk_kernel, cp.async BLOCK_K 256, from the BLOCK_K=256 control, the combo's pick) is unchanged; only a new gate `layer == 'pooler' and 17 <= cfg m <= 32` plus the existing shape preconditions routes to the TMA kernel, placed before the cuBLAS fallback. Own state keys: state['pooler_tma_desc'], state['pooler_tma_ws'] fp32 [2, 32, 4096], state['pooler_tma_cnt'] int32 zeros [64]; warm m in 17..32 in prepare() on synthetic x and re-zero the counter after. Nothing else changes (qkv_gate/o/unpooler at m=32 stay cuBLAS, their Triton m=32 rows all lost: 0.858x, 0.907x, 0.91-0.97x).

## Expected

- `layer=pooler load=decode_32 m=32` faster: master 21.6 us (nvjet 20.2 + 2.0 us reduce node) vs floor 14.1; the TMA kernel measured 20.8 us on this row (1.061x paired) in 4e89311eefdc4bc3; Triton reproduces within ~1%
- `layer=pooler load=decode_1 m=1` flat: master m<=16 path unchanged
- `layer=pooler load=decode_16 m=16` flat: unchanged
- `layer=pooler load=verify_1 m=8` flat: unchanged
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=unpooler *` flat: untouched
- `layer=router *` flat: untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched

## Risk

The only m in 17..32 Triton row with a measured win (the combo dropped it because it took the cp.async BLOCK_K 256 kernel, which read 1.014x at m=32). Small: ~1.2 us x 62 on one of 30 shapes, ~+0.3% headline, inside the CI, so it is judged by the pooler decode_32 row and kernel time, exactly as the combo was. Cheap to implement (verbatim kernel). Overlaps bf16-m32-swapab-wgmma-triton on this one row; if both run, the faster pooler m=32 row goes into the next combo. Its BLOCK_M=32 compile was 255 regs / mma.sync, so it is the floor for a Triton mma.sync m=32 pooler, not a stepping stone.
