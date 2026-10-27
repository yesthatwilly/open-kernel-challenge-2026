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
review: proceed
status: queued
review_at: 2026-10-27T095240Z
review_companions: null
---

# b12-unpooler-slice-m1-r16-rowowner

## Change

Built on master blob 2bcfe7d4. Add the unpooler slice at m=1 (n=1024, k=4096: the first 1024 rows of the unpooler weight, 8.4 MB) as a fourth bulk12 m=1 cell. CUDA (_B12_CUDA): two new instances of the unchanged b12_kernel template, B12Inst<16, 4096, 8, 1, 4> (inst 6) and B12Inst<16, 4096, 8, 1, 8> (inst 7): CKK 512, 8 chunks of 12 KB per CTA, all 8 in flight from the start, grid 1024/16 = 64 CTAs, dynamic smem 108,800 B (every static_assert holds: CHUNK 12288, NS 32 divisible by CW, RING 10496 a multiple of 128), plus their b12_run, b12_unpack (inst 6/7 -> B12Inst<16,4096,...>::unpack) and b12_setup switch cases; bump B12_EXT_NAME. Python: B12_CELLS['unpooler'] = ('U12s', 1024, 4096, 16, ((6, 8, 1, 4), (7, 8, 1, 8)), 'b12_unpool_ok'); in _b12_prepare and in the second state's re-encode take w = state['unpooler'][:1024] (a contiguous view) for this cell instead of state[layer]; _b12_forward already allocates (1, spec n); B12_FLAGS / NEW_CELL_FLAGS / the A/B inheritance pick the cell up from B12_CELLS, and _ab_component already maps the AB case (unpooler, 1024, 4096, 1) to the b12 flag when it is on. kernel_fn: move the _b12_route_ok check above the _ouns_route_ok branch (b12 routes the unpooler only with its own flag on, cfg m == 1 and n == 1024). Gate: the incumbent is today's route, the ouns cluster / DSMEM split-K kernel (ouns_kernel<8,64,512,8,4>, 4.4 us, row 4.97 us in 3834ceec). Because the slice fits in L2, add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of flush.zero_(), the judge's kind of flush; default unchanged for every other caller), gate each instance under both flushes and route the faster one only if both its write and read ratios are <= 0.98, then the usual profiler kernel-name assertion. One PATH CHOSEN line with both ratios. m=2..8 stay on ouns; the pooler / qkv / o cells are untouched.

## Expected

- `layer=unpooler load=decode_1 m=1` faster: 6.3 MB of 12-bit weights on 64 whole-K row-owner CTAs (no DSMEM reduce) vs ouns's 8.4 MB cluster split-K: 4.97 -> ~4.0-4.5 us if the read-flush gate routes it
- `layer=unpooler load=verify_1 m=8` flat: m=8 stays on ouns

## Risk

Medium. New template instances of a judge-proven kernel, so a device fault is unlikely, but a compile error would drop the whole b12 module and with it the landed pooler m=1 win (decode_1 pooler 14.8 -> 17.7 us): run it as a check, never a direct submit, after a quick read-only audit of the instance geometry. Only 64 CTAs stream (96 KB each, one per SM): it wins only if one SM pulls its 96 KB faster than ouns's 128 CTAs pull 64 KB each and reduce through DSMEM; the read-flush gate keeps ouns otherwise. Sizing: decode_1 is 0.40 %/us, so -0.5 to -1 us is +0.2-0.4%, below the 1.24% one-run margin alone. Decode-portable: decode runs the same slice at T=1.

## Adversarial review

Reviewed 2026-10-27T095240Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is plausible and the run can be read in isolation. The in-run read-flush gate against ouns, the PATH CHOSEN line and the decode_1 unpooler profile row (b12_kernel<16,4096,...> vs ouns 4.4-4.6 us) attribute the result without any companion. Part of the change paragraph is out of date against today's master and must be rebuilt as described below, or the cell silently stays off or the whole import fails.

### Justification
Mechanism plausible: the unpooler slice (n=1024, k=4096, 8.4 MB raw, floor 1.7 us) runs on ouns_kernel<8,64,512,8,4> at 4.4-4.6 us; bulk12 shrinks the stream to 6.3 MB and drops the DSMEM reduce. Geometry checked against the b12_kernel static_asserts: CKK 512, NCHUNK 8, NS 32, CHUNK 12288, RING 10496, SMEM 108,800 B; 64 CTAs. Weakest link: the landed o cell with the same 96 KB per CTA reaches only ~2.7 TB/s (per-CTA ramp and ingest bound), so the slice may land near o's ~4.6 us and tie ouns. A no-route result is a real possibility. Out of date against master 8f233e88: (1) do not re-add flush_mode='read' to _cell_gate (CELL_FLUSH_MODES, B12_GATE_FLUSH='read' exist); (2) B12_CELLS variants are 5-tuples (inst, S, MINB, CW, R): ('U12s', 1024, 4096, 16, ((6, 8, 1, 4, 16), (7, 8, 1, 8, 16)), 'b12_unpool_ok'); (3) B12_INST_R must gain 6: 16 and 7: 16 (the import-time assert loop raises KeyError otherwise and kills the submission); (4) use master's _b12_pick unchanged (home-R variant on read <= 0.98), not the hypothesis's write AND read rule; (5) bump B12_EXT_NAME. kernel_fn must test _b12_route_ok before _ouns_route_ok.

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (ouns at unpooler slice m<=8) is the incumbent the gate compares against: read the result as 'b12 beats ouns at m=1'.
- bulk12 m=1 landing (5325ea40): new instances go into the same load_inline module; a compile error drops all three b12 cells, so check the '[b12] build' line first.
- read-flush gate landing (8f233e88): qkv_gate and o b12 cells route near the 0.98 threshold, so gate noise can flip other decode_1 m=1 rows; attribute only through the unpooler decode_1 profile row and its gate line.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive: no shared kernel or path.

### Interference and A/B needs
No companion needed: inert on every shape except unpooler cfg m==1 (n==1024) with its own flag on; measured in-run by the gate (flag on vs ouns, read flush), the profiler's kernel assertion and the judged decode_1 unpooler row.

### Open questions
- If the read gate lands near 1.0, is a 2-CTA cluster K-split of the b12 kernel the follow-up, or does that rebuild ouns on 12-bit bytes?
- Print the per-variant read-flush cand us next to the incumbent us for this cell, and include it in the end-of-prepare re-print.
- Does the judged decode_1 unpooler row track the read-flush gate for an L2-sized weight? Expect roughly half the gate's gain (o gated 0.92-0.93, judged 0.97).

