---
problem: muse_bf16_proj
slug: restack-unpooler-slice-cluster-dsmem-m8
created: 2026-10-25T042040Z
base: master
area: unpooler GEMM (sliced n=1024, m<=8)
change: >
  Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun
  kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite (job 3f3330f7:
  16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows of W[:1024], rank r owns K [512r,
  512r+512), all 64 KB in flight, DSMEM reduce, transposed bf16 store, torch.empty, one node).
  Keep its load_inline scaffold (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 in
  extra_cuda_cflags replacing any other -std), self-check at m in {1, 5, 8} and
  print-and-fall-back. Compile only the unpooler instantiation (NT=8); do NOT route o (o lost
  +14..+48% to the incumbent). Gate the route on a COLD-L2 prepare timing (flush a buffer > 50 MB
  before each event, median of 20, synthetic x): enable only if the cluster beats the restack's
  own unpooler-slice path (_swapab16_forward for unpooler at m<=8) at m=1 and m=8; print both. The
  previous run gated on warm eager times. In kernel_fn insert, before the swapab16 check: if layer
  == 'unpooler' and cfg n == 1024 and 1 <= m <= 8 and state['oun_ok'] -> cluster forward.
  Everything else byte-identical to the restack.
expected:
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: restack row 6.15; oun read 5.02 on master (-15.5%, z -33); floor 1.7
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: restack row 6.00; oun read 5.56 on master (-9.4%, z -21)
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: full weight, untouched
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: untouched
  - shapes: layer=unpooler load=verify_* m=*
    direction: flat
    why: m>=128 untouched (verify_1 excepted above)
  - shapes: layer=o *
    direction: flat
    why: o not routed
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-unpooler-slice-cluster-dsmem-m8

## Change

Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite (job 3f3330f7: 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows of W[:1024], rank r owns K [512r, 512r+512), all 64 KB in flight, DSMEM reduce, transposed bf16 store, torch.empty, one node). Keep its load_inline scaffold (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 in extra_cuda_cflags replacing any other -std), self-check at m in {1, 5, 8} and print-and-fall-back. Compile only the unpooler instantiation (NT=8); do NOT route o (o lost +14..+48% to the incumbent). Gate the route on a COLD-L2 prepare timing (flush a buffer > 50 MB before each event, median of 20, synthetic x): enable only if the cluster beats the restack's own unpooler-slice path (_swapab16_forward for unpooler at m<=8) at m=1 and m=8; print both. The previous run gated on warm eager times. In kernel_fn insert, before the swapab16 check: if layer == 'unpooler' and cfg n == 1024 and 1 <= m <= 8 and state['oun_ok'] -> cluster forward. Everything else byte-identical to the restack.

## Expected

- `layer=unpooler load=decode_1 m=1` faster: restack row 6.15; oun read 5.02 on master (-15.5%, z -33); floor 1.7
- `layer=unpooler load=verify_1 m=8` faster: restack row 6.00; oun read 5.56 on master (-9.4%, z -21)
- `layer=unpooler load=decode_16 m=16` flat: full weight, untouched
- `layer=unpooler load=decode_32 m=32` flat: untouched
- `layer=unpooler load=verify_* m=*` flat: m>=128 untouched (verify_1 excepted above)
- `layer=o *` flat: o not routed
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Kill: unpooler decode_1/verify_1 rows not below restack 6.15/6.00, or the cold-L2 gate disables the route (then the run measures the restack; the printed cold numbers decide whether the design is dead). Estimated headline +0.003.
