---
problem: muse_attention_decode
slug: sliding-decode-paired-kvhead-rows
created: 2026-10-25T003735Z
base: master
area: sliding-layer attention (batched decode, KV row gather)
change: >
  One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding
  decode at batch > 1 only (q_len 1; config-only predicate pair = length == 1 and batch > 1 and
  HKV == 2; batch 1 and the S == 1 path keep today code). Today program (b, g, s) gathers kv head
  g alone: a 64-byte K row and a 64-byte V row per token (kvoff = (row * HKV + g) * D + d), and
  its M = 16 query tile holds only GROUP = 8 real rows (8 padding rows). LESSONS: page-size-1 row
  gather is the limit of these kernels, and FA3 head pairing (one 128-byte row per token for both
  kv heads) gained 14-24% on full batch-16/32 shapes. New constexpr PAIR: grid (B, 1, S) with S =
  16 unchanged; the program serves BOTH kv heads: rows r = 0..15 are query heads 0..15 (head = r,
  kv head r // 8, so the M = 16 tile has no padding rows at L = 1); the query is laid out as [16,
  2*D] with each row values in its own kv head half and zeros in the other half (same trick as
  master paired FA3 query, exact: the zero half adds 0 to every logit); each old-key tile loads
  one contiguous 128-byte K row and one 128-byte V row per token: offset row * HKV * D +
  arange(2*D); qk = dot(q[16,128], K^T[128,BLOCK_N]); PV gives acc[16,128] and each row keeps only
  its own half (select by r // 8 when writing the partial and in the epilogue, so ws_acc stays
  [16, D] per partial). The last split builds both kv heads new k/v (norm + rope + E4M3, stored at
  cache_loc for g = 0 and 1 exactly as today) and attends them from registers in the same 128-wide
  form. Ticket counters indexed by b (one per request instead of b*HKV + g; _counters(batch, 1,
  device)). Workspace sized (B * S, 16, D) and (B * S, 2, 16). _gate_epilogue unchanged per row
  (head = r). BLOCK_N 64, num_warps 4, num_stages 2 unchanged. Everything else (full layers,
  sliding verify, sliding decode_1) byte for byte.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "master _sliding_fused_kernel 12.9 us (floor 1.65): half the gather instructions per token (one 128 B K and V row instead of two 64 B rows from two programs) and no padding query rows; expect 10-11.5 us"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: master 22.1 us (floor 3.29), 1024 programs -> 512 with full 128-byte rows; expect 16-19 us
  - shapes: "*layer=sliding load=decode_1 * *layer=sliding load=verify_* *layer=full*"
    direction: flat
    why: outside the batch > 1 decode predicate
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-decode-paired-kvhead-rows

## Change

One change inside master _sliding_fused_kernel and its launch in _sliding_forward, sliding decode at batch > 1 only (q_len 1; config-only predicate pair = length == 1 and batch > 1 and HKV == 2; batch 1 and the S == 1 path keep today code). Today program (b, g, s) gathers kv head g alone: a 64-byte K row and a 64-byte V row per token (kvoff = (row * HKV + g) * D + d), and its M = 16 query tile holds only GROUP = 8 real rows (8 padding rows). LESSONS: page-size-1 row gather is the limit of these kernels, and FA3 head pairing (one 128-byte row per token for both kv heads) gained 14-24% on full batch-16/32 shapes. New constexpr PAIR: grid (B, 1, S) with S = 16 unchanged; the program serves BOTH kv heads: rows r = 0..15 are query heads 0..15 (head = r, kv head r // 8, so the M = 16 tile has no padding rows at L = 1); the query is laid out as [16, 2*D] with each row values in its own kv head half and zeros in the other half (same trick as master paired FA3 query, exact: the zero half adds 0 to every logit); each old-key tile loads one contiguous 128-byte K row and one 128-byte V row per token: offset row * HKV * D + arange(2*D); qk = dot(q[16,128], K^T[128,BLOCK_N]); PV gives acc[16,128] and each row keeps only its own half (select by r // 8 when writing the partial and in the epilogue, so ws_acc stays [16, D] per partial). The last split builds both kv heads new k/v (norm + rope + E4M3, stored at cache_loc for g = 0 and 1 exactly as today) and attends them from registers in the same 128-wide form. Ticket counters indexed by b (one per request instead of b*HKV + g; _counters(batch, 1, device)). Workspace sized (B * S, 16, D) and (B * S, 2, 16). _gate_epilogue unchanged per row (head = r). BLOCK_N 64, num_warps 4, num_stages 2 unchanged. Everything else (full layers, sliding verify, sliding decode_1) byte for byte.

## Expected

- `*layer=sliding load=decode_16*` faster: master _sliding_fused_kernel 12.9 us (floor 1.65): half the gather instructions per token (one 128 B K and V row instead of two 64 B rows from two programs) and no padding query rows; expect 10-11.5 us
- `*layer=sliding load=decode_32*` faster: master 22.1 us (floor 3.29), 1024 programs -> 512 with full 128-byte rows; expect 16-19 us
- `*layer=sliding load=decode_1 * *layer=sliding load=verify_* *layer=full*` flat: outside the batch > 1 decode predicate

## Risk

Risk: per-program work doubles while program count halves (B=16: 256 programs of 4 warps on 132 SMs, under one wave per 2 CTAs/SM); if occupancy hurts decode_16, an ab-tune with S=32 at batch 16 is the companion. The 128-wide dot doubles MMA work, but these shapes are gather/latency bound (the fp8-qk mma companion on the CUDA path was flat).
