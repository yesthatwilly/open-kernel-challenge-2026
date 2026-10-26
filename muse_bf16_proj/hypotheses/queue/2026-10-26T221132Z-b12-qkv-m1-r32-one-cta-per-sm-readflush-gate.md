---
problem: muse_bf16_proj
slug: b12-qkv-m1-r32-one-cta-per-sm-readflush-gate
created: 2026-10-26T221132Z
base: master
area: qkv GEMM (m=1, bulk12 12-bit row-owner CTA shape)
change: >
  Built on master blob 2bcfe7d4 (section (10) bulk12). Python only, no CUDA change. qkv_gate m=1
  runs b12_kernel<16,8192,6,2,4/8> (R=16, grid 144, 2 CTAs/SM, one m16 tile per k16 step) and was
  judged at 10.54 us, the same as the raw A2 bulk_kernel it replaced (741072dc, 10.54), although
  its prepare gate predicted 0.88 (11.87 vs 13.47 us net, check 0acdca48). Give the qkv cell the
  pooler's existing R=32 instances 0 and 1 (B12Inst<32,8192,8,1,8> and <32,8192,8,1,4>: launch
  grid n/R = 72 CTAs at 1 CTA/SM, 8-stage ring, two m16 tiles per k16 step, so each x B-fragment
  load feeds two MMAs and each warp carries two independent accumulator chains). (a) Let each
  B12_CELLS variant carry its own R: qkv_gate variants become (2,6,2,4,16), (3,6,2,8,16),
  (0,8,1,8,32), (1,8,1,4,32); pooler and o keep their variants with their current R. Add
  B12_INST_R = {0: 32, 1: 32, 2: 16, 3: 16, 4: 64, 5: 64} and use it wherever spec[3] is used as
  the encoding R (the A/B second state's re-encode included). (b) In _b12_prepare encode once per
  distinct R (a local dict R -> (wc, esc, base); rows=32 gives 72 escape blocks, about 36 entries
  each, max ~60 vs the 224 cap; b12_unpack inst 0/1 already decodes any n divisible by 32) and
  load the variant's buffers into state before its canary, _cell_check and _cell_gate; keep only
  the winner's buffers. (c) Add flush_mode='write'|'read' to _cell_gate (default 'write',
  unchanged for every other caller): 'read' replaces flush.zero_() with a full read of the same
  256 MB buffer (flush.view(torch.int32).sum()), which is what the judge does before each replay.
  (d) qkv rule: gate every qkv variant under both flushes; keep today's choice (the best R=16
  variant by the write gate, <= 0.98) unless an R=32 variant passes both gates (<= 0.98) and its
  read-flush ratio is at least 0.03 below the best R=16 variant's read-flush ratio. (e) For the
  pooler and o b12 cells, also print the read-flush ratio of the chosen variant beside the write
  one (diagnostic only, routing unchanged), and re-print every b12 gate line compactly at the end
  of prepare (16 KB log tail). PATH CHOSEN names the variant and both ratios. Nothing else
  changes.
expected:
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: faster
    why: "R=32 bulk12 at 72 CTAs (B-fragment reuse over 2 tiles, 96 KB in flight per CTA) if it wins the read-flush gate: 10.54 -> ~8.5-9.5 us; otherwise today's route"
  - shapes: layer=pooler load=decode_1 m=1
    direction: flat
    why: routing unchanged; read-flush ratio printed as a diagnostic
  - shapes: layer=o load=decode_1 m=1
    direction: flat
    why: routing unchanged; read-flush ratio printed as a diagnostic
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "combo-rowowner-reduce-free-m1-qkv-o-plus-naive: that run fixed qkv m=1's row-owner shape (A2: R=16, 144 CTAs, 2 CTAs/SM), which the landed bulk12 cell copied and which judged flat on 12-bit weights; this tries R=32 at 72 CTAs and 1 CTA/SM on the same 12-bit weights, routed only under a judge-like read-flush gate"
review: required
status: queued
---

# b12-qkv-m1-r32-one-cta-per-sm-readflush-gate

## Change

Built on master blob 2bcfe7d4 (section (10) bulk12). Python only, no CUDA change. qkv_gate m=1 runs b12_kernel<16,8192,6,2,4/8> (R=16, grid 144, 2 CTAs/SM, one m16 tile per k16 step) and was judged at 10.54 us, the same as the raw A2 bulk_kernel it replaced (741072dc, 10.54), although its prepare gate predicted 0.88 (11.87 vs 13.47 us net, check 0acdca48). Give the qkv cell the pooler's existing R=32 instances 0 and 1 (B12Inst<32,8192,8,1,8> and <32,8192,8,1,4>: launch grid n/R = 72 CTAs at 1 CTA/SM, 8-stage ring, two m16 tiles per k16 step, so each x B-fragment load feeds two MMAs and each warp carries two independent accumulator chains). (a) Let each B12_CELLS variant carry its own R: qkv_gate variants become (2,6,2,4,16), (3,6,2,8,16), (0,8,1,8,32), (1,8,1,4,32); pooler and o keep their variants with their current R. Add B12_INST_R = {0: 32, 1: 32, 2: 16, 3: 16, 4: 64, 5: 64} and use it wherever spec[3] is used as the encoding R (the A/B second state's re-encode included). (b) In _b12_prepare encode once per distinct R (a local dict R -> (wc, esc, base); rows=32 gives 72 escape blocks, about 36 entries each, max ~60 vs the 224 cap; b12_unpack inst 0/1 already decodes any n divisible by 32) and load the variant's buffers into state before its canary, _cell_check and _cell_gate; keep only the winner's buffers. (c) Add flush_mode='write'|'read' to _cell_gate (default 'write', unchanged for every other caller): 'read' replaces flush.zero_() with a full read of the same 256 MB buffer (flush.view(torch.int32).sum()), which is what the judge does before each replay. (d) qkv rule: gate every qkv variant under both flushes; keep today's choice (the best R=16 variant by the write gate, <= 0.98) unless an R=32 variant passes both gates (<= 0.98) and its read-flush ratio is at least 0.03 below the best R=16 variant's read-flush ratio. (e) For the pooler and o b12 cells, also print the read-flush ratio of the chosen variant beside the write one (diagnostic only, routing unchanged), and re-print every b12 gate line compactly at the end of prepare (16 KB log tail). PATH CHOSEN names the variant and both ratios. Nothing else changes.

## Expected

- `layer=qkv_gate load=decode_1 m=1` faster: R=32 bulk12 at 72 CTAs (B-fragment reuse over 2 tiles, 96 KB in flight per CTA) if it wins the read-flush gate: 10.54 -> ~8.5-9.5 us; otherwise today's route
- `layer=pooler load=decode_1 m=1` flat: routing unchanged; read-flush ratio printed as a diagnostic
- `layer=o load=decode_1 m=1` flat: routing unchanged; read-flush ratio printed as a diagnostic

## Risk

Medium. Pure Python on instances that have run in every bulk12 check, so no new fault surface; the risk is a wrong route. The write-flush gate flattered bulk12 on qkv and o (predicted 0.88 / 0.87, judged 1.00 / 0.97) but not on the pooler (0.82 vs 0.84): flush.zero_() leaves ~50 MB of dirty lines in L2 and a byte-light kernel evicts fewer of them, which only matters for weights that fit in L2 (qkv 37.7 MB, o 16.8 MB). Hence the read-flush requirement; the pooler / o diagnostics test that explanation against rows already judged, which also tells the 00:00Z decode port whether bulk12 qkv / o at T=1 is worth carrying. 72 CTAs leave 60 SMs idle: R=32 wins only if one SM can pull ~50 GB/s (the pooler cell pulls ~29 GB/s per SM under full load). Sizing: decode_1 is 0.40 %/us, so -1 to -2 us is +0.4-0.8%, below the 1.24% one-run margin alone.
