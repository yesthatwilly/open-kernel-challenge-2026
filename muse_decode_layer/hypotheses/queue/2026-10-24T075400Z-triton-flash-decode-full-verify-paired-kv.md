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
review: required
status: queued
stage: 2
replaces: FA3 on full-attention verify (q_len=8)
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
