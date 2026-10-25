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
review: required
status: queued
---

# unpooler-k-halves-batched-gemm-m65-m128

## Change

Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel / 18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m, 2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out = torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64 N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact single-nonzero rows, two bf16 roundings). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.

## Expected

- `layer=unpooler load=verify_16 m=128` faster: NN nvjet 20.4 kernel / 18.95 row, floor 14.64; halving x L2 traffic should give a kernel of ~16.5-17.5 plus an add of ~1.0: row ~17.5-18.3
- `layer=* load=decode_* m=*` unchanged: untouched

## Risk

Weakest of the four primaries (best case about 1.4 us on one row, about +0.3% headline). Its value is diagnostic: it tests the x-re-read limiter that bf16-verify16-cutlass-m128-onetile-splitk-rewrite is built to remove. If cuBLAS keeps 128x64 tiles for the halves, the add makes it lose, and the selector should keep master.
