---
problem: muse_bf16_proj
slug: b12-unpooler-slice-m1-r16-rowowner
created: 2026-10-26T221215Z
base: master
area: unpooler GEMM (sliced n=1024, m=1, bulk12 12-bit row-owner)
change: >
  Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024
  rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. CUDA (_B12_CUDA): two new
  instances of the unchanged b12_kernel template, B12Inst<16, 4096, 8, 1, 4> (inst 6) and
  B12Inst<16, 4096, 8, 1, 8> (inst 7): CKK 512, 8 chunks of 12 KB per CTA, all 8 in flight from
  the start, grid 1024/16 = 64 CTAs, dynamic smem 108,800 B (every static_assert holds: CHUNK
  12288, NS 32 divisible by CW, RING 10496 a multiple of 128), plus their b12_run, b12_unpack
  (inst 6/7 -> B12Inst<16,4096,...>::unpack) and b12_setup switch cases; bump B12_EXT_NAME.
  Python: B12_CELLS['unpooler'] = ('U12s', 1024, 4096, 16, ((6, 8, 1, 4), (7, 8, 1, 8)),
  'b12_unpool_ok'); in _b12_prepare and in the second state's re-encode take w =
  state['unpooler'][:1024] (a contiguous view) for this cell instead of state[layer]; _b12_forward
  already allocates (1, spec n); B12_FLAGS / NEW_CELL_FLAGS / the A/B inheritance pick the cell up
  from B12_CELLS, and _ab_component already maps the AB case (unpooler, 1024, 4096, 1) to the b12
  flag when it is on. kernel_fn: move the _b12_route_ok check above the _ouns_route_ok branch (b12
  routes the unpooler only with its own flag on, cfg m == 1 and n == 1024). Gate: the incumbent is
  today's route, the ouns cluster / DSMEM split-K kernel (ouns_kernel<8,64,512,8,4>, 4.4 us, row
  4.97 us in 3834ceec). Because the slice fits in L2, add flush_mode='read' to _cell_gate
  (flush.view(torch.int32).sum() in place of flush.zero_(), the judge's kind of flush; default
  unchanged for every other caller), gate each instance under both flushes and route the faster
  one only if both its write and read ratios are <= 0.98, then the usual profiler kernel-name
  assertion. One PATH CHOSEN line with both ratios. m=2..8 stay on ouns; the pooler / qkv / o
  cells are untouched.
expected:
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: "6.3 MB of 12-bit weights on 64 whole-K row-owner CTAs (no DSMEM reduce) vs ouns's 8.4 MB cluster split-K: 4.97 -> ~4.0-4.5 us if the read-flush gate routes it"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: m=8 stays on ouns
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "restack-unpooler-slice-cluster-dsmem-m8: that cluster/DSMEM split-K kernel (ouns) is today's m<=8 slice route; this replaces only its m=1 cell with the judge-proven bulk12 row-owner on 12-bit weights (6.3 MB instead of 8.4 MB, whole K per CTA, no DSMEM reduce), gated under a judge-like read flush"
review: required
status: queued
---

# b12-unpooler-slice-m1-r16-rowowner

## Change

Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024 rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. CUDA (_B12_CUDA): two new instances of the unchanged b12_kernel template, B12Inst<16, 4096, 8, 1, 4> (inst 6) and B12Inst<16, 4096, 8, 1, 8> (inst 7): CKK 512, 8 chunks of 12 KB per CTA, all 8 in flight from the start, grid 1024/16 = 64 CTAs, dynamic smem 108,800 B (every static_assert holds: CHUNK 12288, NS 32 divisible by CW, RING 10496 a multiple of 128), plus their b12_run, b12_unpack (inst 6/7 -> B12Inst<16,4096,...>::unpack) and b12_setup switch cases; bump B12_EXT_NAME. Python: B12_CELLS['unpooler'] = ('U12s', 1024, 4096, 16, ((6, 8, 1, 4), (7, 8, 1, 8)), 'b12_unpool_ok'); in _b12_prepare and in the second state's re-encode take w = state['unpooler'][:1024] (a contiguous view) for this cell instead of state[layer]; _b12_forward already allocates (1, spec n); B12_FLAGS / NEW_CELL_FLAGS / the A/B inheritance pick the cell up from B12_CELLS, and _ab_component already maps the AB case (unpooler, 1024, 4096, 1) to the b12 flag when it is on. kernel_fn: move the _b12_route_ok check above the _ouns_route_ok branch (b12 routes the unpooler only with its own flag on, cfg m == 1 and n == 1024). Gate: the incumbent is today's route, the ouns cluster / DSMEM split-K kernel (ouns_kernel<8,64,512,8,4>, 4.4 us, row 4.97 us in 3834ceec). Because the slice fits in L2, add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of flush.zero_(), the judge's kind of flush; default unchanged for every other caller), gate each instance under both flushes and route the faster one only if both its write and read ratios are <= 0.98, then the usual profiler kernel-name assertion. One PATH CHOSEN line with both ratios. m=2..8 stay on ouns; the pooler / qkv / o cells are untouched.

## Expected

- `layer=unpooler load=decode_1 m=1` faster: 6.3 MB of 12-bit weights on 64 whole-K row-owner CTAs (no DSMEM reduce) vs ouns's 8.4 MB cluster split-K: 4.97 -> ~4.0-4.5 us if the read-flush gate routes it
- `layer=unpooler load=verify_1 m=8` flat: m=8 stays on ouns

## Risk

Medium. New template instances of a judge-proven kernel, so a device fault is unlikely, but a compile error would drop the whole b12 module and with it the landed pooler m=1 win (decode_1 pooler 14.8 -> 17.7 us): run it as a check, never a direct submit, after a quick read-only audit of the instance geometry. Only 64 CTAs stream (96 KB each, one per SM): it wins only if one SM pulls its 96 KB faster than ouns's 128 CTAs pull 64 KB each and reduce through DSMEM; the read-flush gate keeps ouns otherwise. Sizing: decode_1 is 0.40 %/us, so -0.5 to -1 us is +0.2-0.4%, below the 1.24% one-run margin alone. Decode-portable: decode runs the same slice at T=1.
