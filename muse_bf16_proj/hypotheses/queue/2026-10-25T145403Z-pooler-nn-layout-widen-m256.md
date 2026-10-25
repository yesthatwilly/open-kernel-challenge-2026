---
problem: muse_bf16_proj
slug: pooler-nn-layout-widen-m256
created: 2026-10-25T145403Z
base: master
area: pooler GEMM (m=129..256, cuBLAS NN layout)
change: >
  Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128,
  submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous()
  ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at
  NN_GATED_MIN_M=65 <= cfg m <= NN_GATED_MAX_M=128 to torch.mm(x, state[layer + '_t']) in
  kernel_fn. Widen that gate for the POOLER ONLY to cfg m <= 256: add a new constant
  NN_POOLER_MAX_M = 256 (keep NN_GATED_MAX_M = 128 for the unpooler) and change the kernel_fn
  condition so that layer == 'pooler' uses NN_POOLER_MAX_M as its upper bound while layer ==
  'unpooler' keeps 128. Mirror the same per-layer bound in _ab_component (so the in-process A/B
  self-check attributes a pooler m in 129..256 cell to 'nn_ok' and can switch it off) and, if
  cheap, add m=256 pooler to the self-check's captured cells. No new weight copy (pooler_t already
  exists), no new kernel, no other layer or m changes: qkv_gate and o stay TN at every m (the
  probe LOST qkv_gate m=256, 0.943x), unpooler m=256 stays TN (probe 1.012x, noise), and every m
  <= 128 path is byte-identical to master. The gate branches on cfg layer and m only. Evidence:
  bf16-m128-nn-weight-layout-probe (job 2dc5dce0) ran pooler m=256 through NN at 29.73 us row vs
  SGLang 30.74 (paired 1.034x, nvjet 128x64_64x8_1x2 NNT 29.5 us); the TN incumbent on the same
  row read 0.986x (restack 3-cell, nvjet TNT 30.4 us), 0.999x (router-cluster-s2, 28.5 us) and
  0.9998x (pooler-unpooler-m128-nn-layout-gated) in three other runs. The review of
  pooler-unpooler-m128-nn-layout-gated named this widening as its follow-up. The attempt file
  records the pooler verify_32 nvjet kernel name (must carry the NN suffix, e.g. _NNT) and us from
  the profile; if the name is still TNT the run is void.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: "NN-layout nvjet read 1.034x paired vs SGLang in the probe against 0.986-1.000x for the TN incumbent in three runs: about -0.8 to -1.4 us on a 31.3 us row (floor 17.4)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: flat
    why: already NN on master; unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched (stays TN; NN lost at m=256)
  - shapes: layer=unpooler *
    direction: flat
    why: untouched; unpooler keeps the 128 bound
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-nn-layout-widen-m256

## Change

Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at NN_GATED_MIN_M=65 <= cfg m <= NN_GATED_MAX_M=128 to torch.mm(x, state[layer + '_t']) in kernel_fn. Widen that gate for the POOLER ONLY to cfg m <= 256: add a new constant NN_POOLER_MAX_M = 256 (keep NN_GATED_MAX_M = 128 for the unpooler) and change the kernel_fn condition so that layer == 'pooler' uses NN_POOLER_MAX_M as its upper bound while layer == 'unpooler' keeps 128. Mirror the same per-layer bound in _ab_component (so the in-process A/B self-check attributes a pooler m in 129..256 cell to 'nn_ok' and can switch it off) and, if cheap, add m=256 pooler to the self-check's captured cells. No new weight copy (pooler_t already exists), no new kernel, no other layer or m changes: qkv_gate and o stay TN at every m (the probe LOST qkv_gate m=256, 0.943x), unpooler m=256 stays TN (probe 1.012x, noise), and every m <= 128 path is byte-identical to master. The gate branches on cfg layer and m only. Evidence: bf16-m128-nn-weight-layout-probe (job 2dc5dce0) ran pooler m=256 through NN at 29.73 us row vs SGLang 30.74 (paired 1.034x, nvjet 128x64_64x8_1x2 NNT 29.5 us); the TN incumbent on the same row read 0.986x (restack 3-cell, nvjet TNT 30.4 us), 0.999x (router-cluster-s2, 28.5 us) and 0.9998x (pooler-unpooler-m128-nn-layout-gated) in three other runs. The review of pooler-unpooler-m128-nn-layout-gated named this widening as its follow-up. The attempt file records the pooler verify_32 nvjet kernel name (must carry the NN suffix, e.g. _NNT) and us from the profile; if the name is still TNT the run is void.

## Expected

- `layer=pooler load=verify_32 m=256` faster: NN-layout nvjet read 1.034x paired vs SGLang in the probe against 0.986-1.000x for the TN incumbent in three runs: about -0.8 to -1.4 us on a 31.3 us row (floor 17.4)
- `layer=pooler load=verify_16 m=128` flat: already NN on master; unchanged
- `layer=qkv_gate *` flat: untouched (stays TN; NN lost at m=256)
- `layer=unpooler *` flat: untouched; unpooler keeps the 128 bound
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Expected headline effect is small (verify_32 is the weakest load at 1.075x; ~1 us of 94.5 us there is ~+0.2% geomean), so judge on the paired pooler verify_32 ratio, not the headline. Risk: the probe was one run on an older master; cuBLAS NN may pick the same 128x64 tile at m=256 and the 1.034x could be partly drift. Downside is bounded to one row (worst case ~0.97x, the qkv m=256 NN loss magnitude). Shape coverage: the test m=130 unpooler is unaffected (unpooler bound stays 128); no pooler test sits in 129..256, so nearby-shape behaviour equals cuBLAS NN, the same math. Overlap: queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 also targets pooler m=256; if both ever stack, branch order decides, so this keeps its own gate constant NN_POOLER_MAX_M.
