---
problem: muse_decode_layer
slug: triton-flash-decode-full-verify-paired-kv
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  Stacked on triton-flash-decode-full-paired-kv: extend its paired-KV Triton split-KV kernel pair
  to full-attention verify blocks (cfg['window'] == 0 and cfg['q_len'] == 8), which still run FA3
  after the parent (192 us of 714 us on verify_16 full, 374 us of 1050 us on verify_32 full, same
  ~80K-token KV volume as the decode twins). Each kernel-1 program (grid (B, S), S =
  clamp(ceil(2*132/B), 1, 128) as in the parent) now holds all 8 query tokens of request b: Q2
  becomes a [128, 128] bf16 tile with rows ordered (kv group g, token j, head i within group),
  group 0 rows carrying their q in columns 0-63 and zeros in 64-127, group 1 the reverse, so one S
  = Q2 @ K^T [128, BLOCK_N] and one P @ V [BLOCK_N, 128] serve both kv heads from the same
  128-byte rows. Causal inside the block: key position t is visible to query j iff t <=
  seq_lens[b] + j (only the last tile of the last split masks). Partials become fp32 [B, S, 8, 16,
  64] plus (m, l); kernel 2's grid becomes (B*8, 16) and writes bf16 [T, 16, 64] in request-major
  token order for state['o_norm_gate']. Decode (q_len=1) keeps the parent's code path unchanged;
  sliding layers keep FA3.
expected:
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "FA3 374 us for ~655 MB of KV (~1.75 TB/s); the parent's access pattern at ~3 TB/s is ~220 us, ~15% of the shape"
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: FA3 192 us for ~328 MB; ~80 us saved on a 714 us shape (~11%)
  - shapes: "*layer=full load=verify_1*"
    direction: flat
    why: B=1 full verify is 18.6 us of FA3, latency-bound like decode_1
  - shapes: "*load=decode_*"
    direction: flat
    why: "decode path is the parent's, unchanged; control"
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
risk: high
depends_on: triton-flash-decode-full-paired-kv
priority: 5
kind: rewrite
controls_for: null
review: hold
status: hold
replaces: FA3 on full-attention verify (q_len=8)
rewrite_stage: 2
review_at: 2026-10-24T100146Z
review_companions: null
hold_reason: "adversarial review: Stage 1 measured the paired-KV kernel at about 1.63-1.66 TB/s, which is slower than FA3: 200.8 vs 188.8 us at decode_16 and 394.8 vs 374.4 us at decode_32. That"
---

# triton-flash-decode-full-verify-paired-kv

## Change

Stacked on triton-flash-decode-full-paired-kv: extend its paired-KV Triton split-KV kernel pair to full-attention verify blocks (cfg['window'] == 0 and cfg['q_len'] == 8), which still run FA3 after the parent (192 us of 714 us on verify_16 full, 374 us of 1050 us on verify_32 full, same ~80K-token KV volume as the decode twins). Each kernel-1 program (grid (B, S), S = clamp(ceil(2*132/B), 1, 128) as in the parent) now holds all 8 query tokens of request b: Q2 becomes a [128, 128] bf16 tile with rows ordered (kv group g, token j, head i within group), group 0 rows carrying their q in columns 0-63 and zeros in 64-127, group 1 the reverse, so one S = Q2 @ K^T [128, BLOCK_N] and one P @ V [BLOCK_N, 128] serve both kv heads from the same 128-byte rows. Causal inside the block: key position t is visible to query j iff t <= seq_lens[b] + j (only the last tile of the last split masks). Partials become fp32 [B, S, 8, 16, 64] plus (m, l); kernel 2's grid becomes (B*8, 16) and writes bf16 [T, 16, 64] in request-major token order for state['o_norm_gate']. Decode (q_len=1) keeps the parent's code path unchanged; sliding layers keep FA3.

## Expected

- `*layer=full load=verify_32*` faster: FA3 374 us for ~655 MB of KV (~1.75 TB/s); the parent's access pattern at ~3 TB/s is ~220 us, ~15% of the shape
- `*layer=full load=verify_16*` faster: FA3 192 us for ~328 MB; ~80 us saved on a 714 us shape (~11%)
- `*layer=full load=verify_1*` flat: B=1 full verify is 18.6 us of FA3, latency-bound like decode_1
- `*load=decode_*` flat: decode path is the parent's, unchanged; control
- `*layer=sliding*` flat: FA3 unchanged; control

## Risk

Only worth running if the parent shows its kernel beats FA3 on decode_16/32 full; churnsubs should skip it if the parent comes back neutral or a regression. Risk: 128-row Q2 tiles double register pressure (acc [128, 128] fp32 = 64 KB per CTA): use num_warps=8 and BLOCK_N=64, and fall back to splitting the two kv groups across programs (grid (B, S, 2), [64, 64] tiles, losing the shared 128-byte row) if it spills. Correctness traps: the 1M-token full verify test (batch 1, ctx 1048576, q_len 8) runs through this path with S=128 splits of 8192 tokens; the batch 3 ctx 6000 full q_len 8 dist outliers test checks the causal mask in a short context. Every key is attended.

## Adversarial review

Reviewed 2026-10-24T100146Z by an adversarial subagent (model opus).

**Verdict: hold.** Stage 1 measured the paired-KV kernel at about 1.63-1.66 TB/s, which is slower than FA3: 200.8 vs 188.8 us at decode_16 and 394.8 vs 374.4 us at decode_32. That misses the stage-1 pass bar of 0.8x FA3 and disproves the ~3 TB/s premise this stage's expected gains rest on. This stage only carries that kernel over to verify and fixes nothing about its bandwidth, so it would most likely come back about 5% slower than FA3 on full verify. The hypothesis's own Risk section says to skip it if the parent is neutral. The parent was neutral, so the next stage should be a decode bandwidth fix, and the verify extension should wait for stage 3.

### Justification
Mechanism. Full verify FA3 costs about the same as full decode: 194.3 and 374.7 us in the stage-1 run (e82b11f9d9f64b23), against 191.9 and 374.0 us on master. That is the same ~1.7 TB/s over the same ~328 and ~655 MB of FP8 KV, so verify is gather-bound like decode and not compute-bound. In principle a kernel that beats FA3 on decode would beat it on verify too, and the extra q_len=8 math is cheap: [128,128]x[128,64] bf16 MMAs per 16 KB tile is far below the tensor-core roof. The design detail is sound. One Q2 tile of 8 tokens x 16 heads uses block-diagonal zero padding so both kv heads share a 128-byte row. The acc is 128x128 fp32, about 64 regs/thread at num_warps=8, plus 32 for S at BLOCK_N=64, which is tight but feasible. The split-group fallback is a reasonable plan B.

The expected numbers are where it breaks. Both `faster` rows assume 'the parent's access pattern at ~3 TB/s'. Stage 1 measured that access pattern directly: 328 MB / 200.8 us = 1.63 TB/s and 655 MB / 394.8 us = 1.66 TB/s, about 6% below FA3's 1.75 TB/s. Extending the same kernel to verify reads the same bytes at the same rate. It adds a causal mask, 2x MMA width, num_warps=8 and possibly lower occupancy. The honest prediction is about 205 us vs 194 us on verify_16 full and about 400 us vs 375 us on verify_32 full, which is slower, not the claimed ~80/~155 us saving. The one real stage-1 win was latency-side: decode_1 kernel 21.2 us vs FA3 18.4 + 6.8 combine = 25.2 us. That suggests verify_1 full (FA3 18.7 + 6.8 us) could save ~4 us of a ~245 us shape (~1.6% on a weight-16 shape). This hypothesis marked verify_1 as flat, so the one win it could actually deliver goes unmeasured.

Weakest link: the stage-1 bottleneck is unidentified and unfixed. Both FA3 and the paired-row kernel land at 1.6-1.75 TB/s. So either the 128-byte-row framing was not the lever (the earlier reviewer's sector-granularity caveat), or the parent is short on bytes in flight. At S = ceil(264/B) there are about 2 CTAs per SM. Each holds one 16 KB K+V tile in flight, with indices prefetched only one tile ahead, so about 4 MB is in flight chip-wide. Against a ~1.5-2 us loaded latency for a dependent index-then-gather, that caps near 2-2.7 TB/s before any inefficiency, which is consistent with what was measured. Stage 2 has to raise that, for example with BLOCK_N=128, two-deep K/V prefetch, 4-8 CTAs per SM via more splits, or num_stages tuned with an explicit ring, and show that kernel 1 beats FA3 on decode_16/32 at all. Extending scope first doubles the code paths on a kernel that is not yet faster.

### Confounders on the current kernel
- triton-flash-decode-full-paired-kv (parent, not landed but the base): every result this stage produces inherits its ~1.65 TB/s kernel-1 rate. A verify loss would be the parent's bandwidth deficit, not anything the verify extension did, and a verify gain is not expected. The decode rows are an unchanged control, so they carry the parent's 0.987-0.988x shape deltas forward and would drag the headline whatever verify does.
- sglang-pipeline-parity-router-hilo-m8: defines the FA3 verify call this replaces (causal=True, cu_seqlens_q, the FP8 q cast, scheduler_metadata) and the PDL overlap FA3 enjoys with its neighbours. The plain Triton launches lose that overlap (a few us per shape), which is not the kernel's own fault. This does not confound the kernel-time reading but does bias shape time slightly against the change.
- marlin-block-m-cap-16: no interaction. It changes Marlin block_size_m at T=128/256 (verify_16/32), which are the same shapes this targets, but it only touches the MoE path. It shifts the shape denominators (verify_16 full is now 618.9 us, not the hypothesis's 714, and verify_32 full is 969.9 us, not 1050), so recompute the percentage claims against master. Attention kernel time is unaffected.

### Interference and A/B needs
No companion is warranted, and none would rescue this run. The problem is not attribution. The kernel this stage extends is already measured slower than the incumbent, and nothing in the change targets that. An ab-revert of either landed hypothesis cannot change attention bandwidth. A control that splits the verify path into the two-kv-group grid (B,S,2) would only test register pressure on a path that is gather-bound either way. The right follow-up is a different stage 2: keep the verify path on FA3, fix kernel 1's bytes in flight on full decode, and gate on kernel-1 <= 0.8x FA3 (<= 150/300 us). Re-mint this verify extension as stage 3 once that is met, and make verify_1 full an explicit 'faster' expectation given the decode_1 latency win.

### Open questions
- What limits stage-1 kernel 1 at ~1.65 TB/s: bytes in flight (2 CTAs/SM, one 16 KB tile each), the dependent page_table-to-row gather latency, or random 128-byte row access efficiency on HBM3 with page_size 1? Only ncu or a BLOCK_N/splits sweep on decode can say. That sweep should be the actual stage 2.
- Did the explicit next-tile index prefetch actually overlap in the compiled kernel, or did Triton serialise it under num_stages=3? Without PTX/TTGIR on the judge this is unknown, and it bears directly on whether a deeper ring helps.
- Should the verify extension be re-minted later as stage 3 with verify_1 full as an explicit faster target? Stage 1's decode_1 kernel beat FA3 + combine by ~4 us, which is the one demonstrated win of this design.
- Does the 64 KB fp32 acc at num_warps=8 spill on the judge's Triton? If it does, the (B,S,2) fallback loses the shared 128-byte row, which is this design's whole premise.

