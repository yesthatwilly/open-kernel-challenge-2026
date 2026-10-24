---
problem: muse_attention_decode
slug: sliding-fused-triton-decode-batched-deep-splits
created: 2026-10-24T140200Z
base: master
area: sliding-layer attention (batched decode, split depth)
change: >
  Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from
  (window and batch == 1 and q_len == 1) to (window and q_len == 1), so sliding decode at every
  batch (scored decode_16 and decode_32; tests batch 3 ctx 2047 page 64 and batch 4 ctx 6000 dist
  edges page 1) takes _sliding_forward, the one-launch _sliding_fused_kernel, instead of
  _pre_attention_kernel + prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine +
  rms_norm_gate (five launches, 16.5-20.1 us GPU busy). (2) Change the split count for decode in
  _sliding_forward: S = 16 at every batch for q_len == 1 (each program owns ceil(2047/16) rounded
  to 64 = 128 old keys, two BLOCK_N=64 tiles), giving grid (B, 2, 16) = 512 CTAs at B=16 and 1024
  at B=32, instead of _num_splits which gave 4 at B=16 and 2 at B=32 (8 and 16 serial tiles per
  program). Keep _num_splits itself unchanged for any other caller and keep B=1 at its current 16
  (so decode_1 is byte-identical in behaviour). Everything else stays as master: the kernel body,
  BLOCK_N=64, num_warps=4, num_stages=2, the ticket-counter in-kernel combine (counters sized
  B*HKV from config, ws_acc/ws_ml sized B*HKV*S*M*D from config: 2 MB at B=32), the gate epilogue,
  and all full-layer and verify paths.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "incumbent 20.66 us (FA3 9.9 + combine 3.2 + prepare 1.9 + pre-attn 1.7 + norm 1.2); the same kernel at S=4 measured 23.87 us (8 serial dependent tiles per program) while at B=1 with S=16 (2 tiles) it is 13.1 us shape / 10.5 us kernel: per-program serial tiles, not bytes (8 MB, 1.7 us floor), set the time, so 2 tiles at 512 CTAs should land ~12-14 us; 46 weight, ~1.12x on the decode_16 load"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: incumbent 24.89 us; S=2 (16 tiles) measured 38.47 us; S=16 at 1024 CTAs (all resident, ~8/SM at 4 warps) ~13-16 us; ~1.08x on the decode_32 load
  - shapes: "*layer=sliding load=decode_1*"
    direction: flat
    why: same gate and S=16 as master
  - shapes: "*layer=full*"
    direction: flat
    why: untouched control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 keeps FA3
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-fused-triton-decode-batched-deep-splits

## Change

Built on master. Two edits, both config-only. (1) In kernel_fn widen the fused sliding gate from (window and batch == 1 and q_len == 1) to (window and q_len == 1), so sliding decode at every batch (scored decode_16 and decode_32; tests batch 3 ctx 2047 page 64 and batch 4 ctx 6000 dist edges page 1) takes _sliding_forward, the one-launch _sliding_fused_kernel, instead of _pre_attention_kernel + prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + rms_norm_gate (five launches, 16.5-20.1 us GPU busy). (2) Change the split count for decode in _sliding_forward: S = 16 at every batch for q_len == 1 (each program owns ceil(2047/16) rounded to 64 = 128 old keys, two BLOCK_N=64 tiles), giving grid (B, 2, 16) = 512 CTAs at B=16 and 1024 at B=32, instead of _num_splits which gave 4 at B=16 and 2 at B=32 (8 and 16 serial tiles per program). Keep _num_splits itself unchanged for any other caller and keep B=1 at its current 16 (so decode_1 is byte-identical in behaviour). Everything else stays as master: the kernel body, BLOCK_N=64, num_warps=4, num_stages=2, the ticket-counter in-kernel combine (counters sized B*HKV from config, ws_acc/ws_ml sized B*HKV*S*M*D from config: 2 MB at B=32), the gate epilogue, and all full-layer and verify paths.

## Expected

- `*layer=sliding load=decode_16*` faster: incumbent 20.66 us (FA3 9.9 + combine 3.2 + prepare 1.9 + pre-attn 1.7 + norm 1.2); the same kernel at S=4 measured 23.87 us (8 serial dependent tiles per program) while at B=1 with S=16 (2 tiles) it is 13.1 us shape / 10.5 us kernel: per-program serial tiles, not bytes (8 MB, 1.7 us floor), set the time, so 2 tiles at 512 CTAs should land ~12-14 us; 46 weight, ~1.12x on the decode_16 load
- `*layer=sliding load=decode_32*` faster: incumbent 24.89 us; S=2 (16 tiles) measured 38.47 us; S=16 at 1024 CTAs (all resident, ~8/SM at 4 warps) ~13-16 us; ~1.08x on the decode_32 load
- `*layer=sliding load=decode_1*` flat: same gate and S=16 as master
- `*layer=full*` flat: untouched control
- `*load=verify_*` flat: q_len 8 keeps FA3

## Risk

Evidence that split depth is the lever: sliding-single-launch-fused-attention ran this exact kernel with S from _num_splits and its sliding decode times tracked serial tiles per program, not bytes: decode_1 S=16 (2 tiles) 13.5 us, decode_16 S=4 (8 tiles) 23.9 us, decode_32 S=2 (16 tiles) 38.5 us, ~1.5-1.8 us per extra tile (a dependent page-table load then K/V gather each tile). Risks: the last-ticket program combines 16 partials of [16,64] fp32 from L2 (64 KB per (b,g)), measured fine at B=1; 1024 CTAs plus 64 counters is more atomics but one per program; if decode_32 is still behind, an ab-tune at S=32 (one tile per program) is the follow-up. Verify is deliberately excluded: the same kernel lost 2-2.6x on verify for a reason not explained by split depth (verify_1 at S=16 was 42 us). Disjoint from sliding-fused-triton-decode1-splits-32 (B=1 only) so the two can be merged by 3-way merge.
