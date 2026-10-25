---
problem: muse_bf16_proj
slug: unpooler-k-halves-batched-gemm-m65-m128
created: 2026-10-25T153141Z
base: master
area: unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192)
  and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on
  master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel /
  18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x
  full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is
  then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight
  bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m,
  2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the
  existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out =
  torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64
  N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate
  on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact
  single-nonzero rows, two bf16 roundings). Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every
  other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.
expected:
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "NN nvjet 20.4 kernel / 18.95 row, floor 14.64; halving x L2 traffic should give a kernel of ~16.5-17.5 plus an add of ~1.0: row ~17.5-18.3"
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T203723Z
review_companions: null
---

# unpooler-k-halves-batched-gemm-m65-m128

## Change

Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel / 18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m, 2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out = torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64 N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact single-nonzero rows, two bf16 roundings). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.

## Expected

- `layer=unpooler load=verify_16 m=128` faster: NN nvjet 20.4 kernel / 18.95 row, floor 14.64; halving x L2 traffic should give a kernel of ~16.5-17.5 plus an add of ~1.0: row ~17.5-18.3
- `layer=* load=decode_* m=*` unchanged: untouched

## Risk

Weakest of the four primaries (best case about 1.4 us on one row, about +0.3% headline). Its value is diagnostic: it tests the x-re-read limiter that bf16-verify16-cutlass-m128-onetile-splitk-rewrite is built to remove. If cuBLAS keeps 128x64 tiles for the halves, the add makes it lose, and the selector should keep master.

## Adversarial review

Reviewed 2026-10-25T203723Z by an adversarial subagent (model opus).

**Verdict: proceed.** Real, untried variant of pooler-k-halves-batched-gemm-m65-m256 (own rows 1.063/1.123 paired; selector and profiler machinery worked under a scored submit). Weaker case: the add costs ~2 us (not 1), so the batched GEMM must reach <= ~17.3 us vs 19.3-19.4 today (~11%) from a tile change alone. The prepare-time selector makes a loss safe and readable.

### Justification
master unpooler verify_16 m=128: torch.mm(x, unpooler_t) under nn_ok, nvjet 128x64 NNT 19.3-19.4 us, row 18.9-19.15, floor 14.64. 128 CTAs of 128x64 over k=4096 pull ~128 MB of x via L2 vs 67 MB weight; 2-batch k=2048 128x128 halves it; L2 limiter plausible, not shown. The add moves 4 MB read / 2 MB write (pooler m=256 analogue 2.1 us); realistic row 18.5-19.3. Sibling at m=128 picked 128x64 (bmm 20.1 vs 19.4 us); at m=256 128x128. Weakest link: an ~11% GEMM speedup from a tile change on the shape where cuBLAS prefers 128x64. Prior (missing field): pooler-k-halves-batched-gemm-m65-m256; difference is real (tile-deficit fix vs x-L2 re-read on a matched tile).

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (landed): owns unpooler_t and shared nn_ok (pooler m=128 and unpooler m=128); give the path its own flag (unpooler_khalf_ok) in NEW_CELL_FLAGS checked before nn_ok in _ab_component; a miss falls back to master's NN mm.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (landed): follow the NEW_CELL_FLAGS/_ab_flags/B-state pattern; select once on state A.
- sglang-gemm-dispatch-parity: fallback if nn_ok off; record nn_ok in the attempt file.
- pooler-nn-layout-widen-m256: insert the route without reordering the nn_p256_ok condition.

### Interference and A/B needs
No companion: the selector times master's exact path vs both candidates (N view, T view) with cold L2 and names the kernel; log medians+IQR, bmm-only median, kernel names, xh.stride(), final flags; 'refuted at prepare-time' if master kept.

### Open questions
- Port _khalf_forward/_khalf_select/relaxed _khalf_names_ok from the sibling branch/patch.
- Stack the sibling pooler k-halves later (separate hypothesis).
- Correct expected row to ~18.5-19.3; break-even bmm <= ~17.3 us.
- Does cuBLAS pick 128x128 for (b=2, m=128, n=8192, k=2048)? If 128x64, follow up with a pinned cuBLASLt algo.
- Add prior: pooler-k-halves-batched-gemm-m65-m256.
- Overlaps queued bf16-verify16-cutlass-m128-onetile-splitk-rewrite (re-baseline).

