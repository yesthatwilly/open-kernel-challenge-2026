---
problem: muse_bf16_proj
slug: qkv-rowowner-a2-widen-m8-m16
created: 2026-10-25T211857Z
base: master
area: qkv GEMM (m=8/16, bulk row-owner kernel)
change: >
  DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to
  bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; decode_1 12.67 -> 10.54
  us). NT=8 is the padded token count and bulk_run already accepts 1 <= m <= NT, so (1) m == 8:
  route qkv_gate at cfg m == 8 onto the SAME compiled A2 instance (verify_1 qkv now runs
  _bf16_swapab_kernel SPLIT_K 4, 9.6 us kernel, row 12.35). (2) m == 16: add one instance
  bulk_kernel<16, 8192, 16, 3, 2> (NT=16 doubles the x slot: SLOT = 16384 + 16*1040 = 33,024 B,
  S=3 gives 99,328 B, still 2 CTAs/SM; S=4 would not fit two CTAs) for decode_16 (swapab16, 9.5 us
  kernel, row 12.64). Bump BULK_EXT_NAME to a new name since the instance set changes; keep
  -std=c++20 in the nvcc flags and the bounded mbarrier waits (BULK_WAIT_NS). In BULK_CELLS-style
  config add two cells with their own flags bulk_qkv8_ok and bulk_qkv16_ok (appended to
  NEW_CELL_FLAGS, inherited by state B, AB_CASES entries for qkv m=8 and m=16); generalise
  _bulk_route_ok from cfg m == BULK_M to the cell's own m (exact equality, cfg/shape/flag only).
  Each cell goes through master's existing canary (flushed LAUNCH/DONE, synced, error word), the
  rel-L2/probe-row correctness check, the occupancy query (disable the m=16 cell unless it reports
  2 CTAs/SM), then the gate. Gate (hardened from today's evidence): copy master's _cell_gate model
  selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41
  interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent
  interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's
  IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at
  m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged
  9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one
  torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN
  line per cell prints both medians, both IQRs and the kernels. Print per cell the kernel-only
  TB/s next to BULK_REF_TBS. The m=1 cells, all other layers and m=32 stay exactly as master.
expected:
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: one reduce-free launch at 2 CTAs/SM vs SPLIT_K 4 ticket swap-AB 9.6 us kernel; target ~11.0 us row from 12.35
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: same at NT=16; target ~11.8 from 12.64, gate keeps master if the per-token FMA cost eats it
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: flat
    why: A2 m=1 cell unchanged
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild: that row-owner lost +1 us at m=8 and +6 us at m=16 on pooler/unpooler at 1 CTA/SM; this widens master's qkv A2 instance, which runs 2 CTAs/SM and won 1.16x at m=1, against the split-K-4 swap-AB incumbent, behind a per-cell gate"
review: required
status: queued
---

# qkv-rowowner-a2-widen-m8-m16

## Change

DIAGNOSIS Action 4. Built on master 36c26199. Master routes qkv_gate at cfg m == 1 to bulk_kernel<16, 8192, 8, 4, 2> (A2, grid 144, 99,072 B smem, 2 CTAs/SM; decode_1 12.67 -> 10.54 us). NT=8 is the padded token count and bulk_run already accepts 1 <= m <= NT, so (1) m == 8: route qkv_gate at cfg m == 8 onto the SAME compiled A2 instance (verify_1 qkv now runs _bf16_swapab_kernel SPLIT_K 4, 9.6 us kernel, row 12.35). (2) m == 16: add one instance bulk_kernel<16, 8192, 16, 3, 2> (NT=16 doubles the x slot: SLOT = 16384 + 16*1040 = 33,024 B, S=3 gives 99,328 B, still 2 CTAs/SM; S=4 would not fit two CTAs) for decode_16 (swapab16, 9.5 us kernel, row 12.64). Bump BULK_EXT_NAME to a new name since the instance set changes; keep -std=c++20 in the nvcc flags and the bounded mbarrier waits (BULK_WAIT_NS). In BULK_CELLS-style config add two cells with their own flags bulk_qkv8_ok and bulk_qkv16_ok (appended to NEW_CELL_FLAGS, inherited by state B, AB_CASES entries for qkv m=8 and m=16); generalise _bulk_route_ok from cfg m == BULK_M to the cell's own m (exact equality, cfg/shape/flag only). Each cell goes through master's existing canary (flushed LAUNCH/DONE, synced, error word), the rel-L2/probe-row correctness check, the occupancy query (disable the m=16 cell unless it reports 2 CTAs/SM), then the gate. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Print per cell the kernel-only TB/s next to BULK_REF_TBS. The m=1 cells, all other layers and m=32 stay exactly as master.

## Expected

- `layer=qkv_gate load=verify_1 m=8` faster: one reduce-free launch at 2 CTAs/SM vs SPLIT_K 4 ticket swap-AB 9.6 us kernel; target ~11.0 us row from 12.35
- `layer=qkv_gate load=decode_16 m=16` faster: same at NT=16; target ~11.8 from 12.64, gate keeps master if the per-token FMA cost eats it
- `layer=qkv_gate load=decode_1 m=1` flat: A2 m=1 cell unchanged

## Risk

Effect if both route: -1.3 us at 0.343 %/us (verify_1) plus ~-0.8 us at 0.27 %/us (decode_16), about +0.45-0.7% headline: below the margin, must stack (combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16). New CUDA instance with mbarrier rings: fault/hang risk, so the first measurement is an okc check. The gate decides per cell, so a slow m=16 costs nothing.
