---
problem: muse_bf16_proj
slug: pooler-skinny-tiled-weight-layout
created: 2026-10-23T225303Z
base: master
area: pooler GEMM
change: >
  Lossless weight repack for the parent's skinny kernel: in prepare(), for the pooler weight only,
  build state['pooler_tiled'] = W.view(N/BLOCK_N, BLOCK_N, K).transpose(1, 2).contiguous(), i.e. a
  [N/32, K, 32] bf16 tensor where program i's whole working set is one contiguous 512 KB block
  laid out K-major, with the same values and dtype (bf16 stays bf16, nothing rounded). In the
  Triton kernel from pooler-triton-skinny-m32, change only the weight tl.load addressing: the tile
  for K-step j is the contiguous [BLOCK_K, BLOCK_N] slab at base + i*K*BLOCK_N +
  j*BLOCK_K*BLOCK_N, loaded directly in the [BLOCK_K, BLOCK_N] orientation tl.dot(x, w) wants, so
  no in-register transpose and every warp's loads are 128-byte-aligned fully sequential lines
  instead of 32 strided row segments of 512 bytes. Everything else (grid, BLOCK sizes, num_stages,
  x path, store) stays exactly as the parent. The original W is kept in state for m >= 128
  (cuBLAS) so memory grows by 67 MB once, untimed.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: same bytes, strictly sequential per-program stream and no transpose; a few percent of the parent if the parent is memory-pipeline bound
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: same
  - shapes: layer=pooler load=decode_32 *
    direction: faster
    why: same
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: same at m=8
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 reads the original W via cuBLAS; control
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: untouched; control
risk: low
depends_on: pooler-triton-skinny-m32
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-skinny-tiled-weight-layout

## Change

Lossless weight repack for the parent's skinny kernel: in prepare(), for the pooler weight only, build state['pooler_tiled'] = W.view(N/BLOCK_N, BLOCK_N, K).transpose(1, 2).contiguous(), i.e. a [N/32, K, 32] bf16 tensor where program i's whole working set is one contiguous 512 KB block laid out K-major, with the same values and dtype (bf16 stays bf16, nothing rounded). In the Triton kernel from pooler-triton-skinny-m32, change only the weight tl.load addressing: the tile for K-step j is the contiguous [BLOCK_K, BLOCK_N] slab at base + i*K*BLOCK_N + j*BLOCK_K*BLOCK_N, loaded directly in the [BLOCK_K, BLOCK_N] orientation tl.dot(x, w) wants, so no in-register transpose and every warp's loads are 128-byte-aligned fully sequential lines instead of 32 strided row segments of 512 bytes. Everything else (grid, BLOCK sizes, num_stages, x path, store) stays exactly as the parent. The original W is kept in state for m >= 128 (cuBLAS) so memory grows by 67 MB once, untimed.

## Expected

- `layer=pooler load=decode_1 *` faster: same bytes, strictly sequential per-program stream and no transpose; a few percent of the parent if the parent is memory-pipeline bound
- `layer=pooler load=decode_16 *` faster: same
- `layer=pooler load=decode_32 *` faster: same
- `layer=pooler load=verify_1 *` faster: same at m=8
- `layer=pooler load=verify_16 *` flat: m=128 reads the original W via cuBLAS; control
- `*layer=qkv_gate*` flat: untouched; control

## Risk

Pure layout tune on the parent, so it attributes to addressing alone; if it is flat the parent's limit is elsewhere (occupancy or latency) and the tiled layout should not be carried to the sibling layers. Rules: repacking weights in prepare() is explicitly allowed and this is a permutation of the same bf16 values (lossless). The repack must use the same BLOCK_N the kernel is launched with; a mismatch is a wrong answer, caught by the m=7 pooler test. Risk is low but so is the expected size of the effect; if the parent lands at >= 4.3 TB/s skip this.
