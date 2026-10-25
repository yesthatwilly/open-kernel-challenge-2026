---
problem: muse_bf16_proj
slug: unpooler-full-m16-swapab16-blockt16
created: 2026-10-25T145403Z
base: master
area: unpooler GEMM (full width n=8192, m=9..16, swap-AB BLOCK_T 16)
change: >
  Built on master (blob 353e3b19). On master the full-width unpooler (n=8192, k=4096, 67.1 MB) at
  9 <= m <= 16 is the only m <= 32 bf16 cell still on cuBLAS: _swapab16_key returns '' for
  unpooler with cfg n != 1024, _unpool_shape_ok needs n == 1024, _swapab_route_ok needs m >= 17,
  so kernel_fn falls through to _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT,
  17.6 us kernel, row 18.6 us vs SGLang 18.4, 0.991x paired in the master run). Change, in section
  (6) only: add a key 'unpooler_full': (8192, 4096, 1, 256, 4) to SWAPAB16_CFG (grid (128, 1),
  SPLIT_K 1 so no workspace or tickets, BLOCK_K 256, 4 stages; the same per-layer constants
  master's 17..32 swap-AB uses for the unpooler, SWAPAB_CFG['unpooler']); change _swapab16_key so
  that layer == 'unpooler' returns 'unpooler_slice' when cfg n == 1024 and 'unpooler_full' when
  cfg n == 8192, else ''; in _swapab16_prepare map key 'unpooler_full' to state['unpooler'] (layer
  = 'unpooler' for both unpooler keys) and warm m in 9..16 for it (m_top 16), with the existing
  OutOfResources fallback to 3 stages. _swapab16_route_ok already requires tuple(w.shape) == (n,
  k) for non-slice keys and 1 <= cfg m <= 16; _swapab16_buffers returns (None, None) for the new
  key. The kernel is master's unchanged _bf16_swapab_kernel with BLOCK_T=16 (wgmma m64n16k16),
  exactly as master runs it for pooler/qkv_gate/o at m <= 16. Note the ouns cluster branch is
  checked first in kernel_fn and only takes n == 1024, so it is unaffected. Every other cell is
  byte-identical to master. The attempt file records the unpooler decode_16 kernel name
  (_bf16_swapab_kernel expected) and us, and the prepare log's [swapab16] unpooler_full PTX line
  (wgmma > 0, no spills).
expected:
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: unpooler-full-m16-swapab-route (BLOCK_T 32 swap-AB at m=16, job 05782267) read 1.022x paired on this row while nvjet reads 0.991-1.009x; BLOCK_T 16 halves the masked token tile (wgmma n16), which made qkv 18% and o 10-19% faster at m<=16 (bf16-m16-swapab-wgmma-extend)
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: "m=32 stays on master's BLOCK_T 32 swap-AB"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: n=1024 slice, ouns cluster path unchanged
  - shapes: layer=unpooler load=decode_1 m=1
    direction: flat
    why: n=1024 slice, ouns cluster path unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
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
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# unpooler-full-m16-swapab16-blockt16

## Change

Built on master (blob 353e3b19). On master the full-width unpooler (n=8192, k=4096, 67.1 MB) at 9 <= m <= 16 is the only m <= 32 bf16 cell still on cuBLAS: _swapab16_key returns '' for unpooler with cfg n != 1024, _unpool_shape_ok needs n == 1024, _swapab_route_ok needs m >= 17, so kernel_fn falls through to _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT, 17.6 us kernel, row 18.6 us vs SGLang 18.4, 0.991x paired in the master run). Change, in section (6) only: add a key 'unpooler_full': (8192, 4096, 1, 256, 4) to SWAPAB16_CFG (grid (128, 1), SPLIT_K 1 so no workspace or tickets, BLOCK_K 256, 4 stages; the same per-layer constants master's 17..32 swap-AB uses for the unpooler, SWAPAB_CFG['unpooler']); change _swapab16_key so that layer == 'unpooler' returns 'unpooler_slice' when cfg n == 1024 and 'unpooler_full' when cfg n == 8192, else ''; in _swapab16_prepare map key 'unpooler_full' to state['unpooler'] (layer = 'unpooler' for both unpooler keys) and warm m in 9..16 for it (m_top 16), with the existing OutOfResources fallback to 3 stages. _swapab16_route_ok already requires tuple(w.shape) == (n, k) for non-slice keys and 1 <= cfg m <= 16; _swapab16_buffers returns (None, None) for the new key. The kernel is master's unchanged _bf16_swapab_kernel with BLOCK_T=16 (wgmma m64n16k16), exactly as master runs it for pooler/qkv_gate/o at m <= 16. Note the ouns cluster branch is checked first in kernel_fn and only takes n == 1024, so it is unaffected. Every other cell is byte-identical to master. The attempt file records the unpooler decode_16 kernel name (_bf16_swapab_kernel expected) and us, and the prepare log's [swapab16] unpooler_full PTX line (wgmma > 0, no spills).

## Expected

- `layer=unpooler load=decode_16 m=16` faster: unpooler-full-m16-swapab-route (BLOCK_T 32 swap-AB at m=16, job 05782267) read 1.022x paired on this row while nvjet reads 0.991-1.009x; BLOCK_T 16 halves the masked token tile (wgmma n16), which made qkv 18% and o 10-19% faster at m<=16 (bf16-m16-swapab-wgmma-extend)
- `layer=unpooler load=decode_32 m=32` flat: m=32 stays on master's BLOCK_T 32 swap-AB
- `layer=unpooler load=verify_1 m=8` flat: n=1024 slice, ouns cluster path unchanged
- `layer=unpooler load=decode_1 m=1` flat: n=1024 slice, ouns cluster path unchanged
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Not a duplicate of done/regression unpooler-full-m16-swapab-route: that run used the BLOCK_T 32 kernel on an older master (8b9ad43) and its regression label came from untouched rows drifting (qkv/o/pooler/router lost together); its own target row gained 1.022x. This uses the BLOCK_T 16 variant and is built on current master. Risk medium rather than low: the pooler, the other 67 MB stream, was flat with BLOCK_T 16 (19.2 us at m=16), so the kernel may land at ~17.5-18.4 us, about nvjet's 17.6; the gain on one row is ~1-3% (decode_16 load ~+0.3-0.9%, headline ~+0.1%). Downside is one row. No m=9..16 full-width unpooler test exists (m=5 uses the slice, m=130/1000 stay on cuBLAS), so correctness rests on master's already-proven swap-AB kernel with SPLIT_K 1; prepare should run one fp32 self-check at m=16 and drop the key from swapab16_stages if rel L2 > 1e-2.
