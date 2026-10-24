---
problem: muse_decode_layer
slug: triton-flash-decode-full-oversub-kv-prefetch
created: 2026-10-24T100416Z
base: master
area: FA3 call
change: >
  Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv,
  muse_decode_layer/submission.py) and change only _full_decode_splits and the loop of
  _full_decode_split_kernel, to put more K/V bytes in flight per SM on full DECODE (window == 0,
  q_len == 1). Verify, sliding layers and everything else stay byte-identical to stage 1 (verify
  keeps FA3). Stage 1 measured kernel 1 at 200.8 us (decode_16 full, 1.63 TB/s over ~328 MB) and
  394.8 us (decode_32 full, 1.66 TB/s over ~655 MB), against FA3 188.8 / 374.4 us. Its grid is the
  limiter: S = ceil(264/B) gives B=16: 17x16 = 272 CTAs and B=32: 9x32 = 288 CTAs, about 2 per SM
  (and 288 > 264 leaves a 24-CTA second wave). Each CTA has one 16 KB K+V tile in flight, because
  the loop issues tile i's K/V loads and consumes them in the same iteration (only the page_table
  indices are fetched one tile ahead). Under spread=agentic the split length comes from
  page_table.shape[1], so the CTAs of short requests finish early and leave their slots idle for
  the rest of the single wave. Two changes, both aimed at bytes in flight. (1) Oversubscribe the
  grid: FD_SPLIT_CTAS = 1024 and S = max(1, min(128, 1024 // B)), from B only, so B=1: 128
  (unchanged), B=16: 64, B=32: 32, B=33: 31. That is about 4 resident CTAs per SM over about 2
  waves, and empty or short splits drain and free their slots for the long requests. split_len
  stays ceil(page_table.shape[1]*P / S) rounded up to BLOCK_N (B=16: 1280 tokens = 20 tiles, B=32:
  2560 tokens = 40 tiles). (2) Prefetch one tile ahead in K/V, not only in indices. Before the
  loop, load tile 0's indices, rows, K and V (uint8 [64,128] each), plus tile 1's indices and
  rows. In each iteration, first issue the K and V loads for tile i+1 (its rows are already in
  hand; masked by its valid, other=0), then load tile i+2's page_table indices and rows, then run
  tile i's QK^T, online softmax and PV on the K/V loaded in the previous iteration, then rotate
  (k_cur, v_cur, valid_cur) <- (k_next, v_next, valid_next) and rows_next <- rows_next2. While one
  tile is computed, the next tile's 16 KB and the index load after it are outstanding. At about 4
  CTAs per SM that is about 64 KB in flight per SM (about 8.4 MB chip-wide, about 2x stage 1's ~4
  MB), towards LESSONS' 3.73 TB/s at 128 KB per CTA in bf16. Keep BLOCK_N=64 and num_warps=4. Set
  num_stages=1, because the prefetch is explicit and the compiler must not add its own smem ring
  on top. Everything else is stage 1's code: the block-diagonal [16,128] Q2 with explicit zeros,
  FP8 q, fp8->bf16 K/V, the exp2-domain online softmax with its -inf guard, fp32 partials
  [B,S,16,64] plus (m,l), and combine kernel 2 with BLOCK_S = next_pow2(S) (64 and 32 at B=16 and
  32; prepare() already warms every BLOCK_S up to 128). Every key t < cache_seqlens[b] is still
  visited exactly once. The loop bound and masks are unchanged, so a prefetch of a tile past end
  is fully masked and never dereferenced (-1 entries are never read). No autotune, no host syncs,
  and scratch is allocated per call from shapes only. Report kernel 1 + kernel 2 us per
  full-decode shape and KV bytes / kernel-1 time.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "kernel 1+2 from 200.8 us (1.63 TB/s) to <= 150 us (>= 2.2 TB/s): 1024 CTAs at ~4/SM with one tile prefetched is ~2x stage 1's bytes in flight, and oversubscription stops short agentic requests idling their SM slots; FA3 is 188.8 us, so the shape should beat master's 433.0 us by ~40 us"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "kernel 1+2 from 394.8 us (1.66 TB/s) to <= 300 us (>= 2.2 TB/s): removes the 288-CTA two-wave tail and doubles in-flight bytes; against FA3's 374.4 us the shape should beat master's 601.7 us by ~75 us"
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: "S stays 128 at B=1; the one-ahead K/V prefetch can only shorten stage 1's 21.2 us (FA3 + combine 25.2 us)"
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len=8 stays on FA3, including full verify; control
risk: medium
depends_on: triton-flash-decode-full-paired-kv
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine"
review: required
status: queued
---

# triton-flash-decode-full-oversub-kv-prefetch

## Change

Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv, muse_decode_layer/submission.py) and change only _full_decode_splits and the loop of _full_decode_split_kernel, to put more K/V bytes in flight per SM on full DECODE (window == 0, q_len == 1). Verify, sliding layers and everything else stay byte-identical to stage 1 (verify keeps FA3). Stage 1 measured kernel 1 at 200.8 us (decode_16 full, 1.63 TB/s over ~328 MB) and 394.8 us (decode_32 full, 1.66 TB/s over ~655 MB), against FA3 188.8 / 374.4 us. Its grid is the limiter: S = ceil(264/B) gives B=16: 17x16 = 272 CTAs and B=32: 9x32 = 288 CTAs, about 2 per SM (and 288 > 264 leaves a 24-CTA second wave). Each CTA has one 16 KB K+V tile in flight, because the loop issues tile i's K/V loads and consumes them in the same iteration (only the page_table indices are fetched one tile ahead). Under spread=agentic the split length comes from page_table.shape[1], so the CTAs of short requests finish early and leave their slots idle for the rest of the single wave. Two changes, both aimed at bytes in flight. (1) Oversubscribe the grid: FD_SPLIT_CTAS = 1024 and S = max(1, min(128, 1024 // B)), from B only, so B=1: 128 (unchanged), B=16: 64, B=32: 32, B=33: 31. That is about 4 resident CTAs per SM over about 2 waves, and empty or short splits drain and free their slots for the long requests. split_len stays ceil(page_table.shape[1]*P / S) rounded up to BLOCK_N (B=16: 1280 tokens = 20 tiles, B=32: 2560 tokens = 40 tiles). (2) Prefetch one tile ahead in K/V, not only in indices. Before the loop, load tile 0's indices, rows, K and V (uint8 [64,128] each), plus tile 1's indices and rows. In each iteration, first issue the K and V loads for tile i+1 (its rows are already in hand; masked by its valid, other=0), then load tile i+2's page_table indices and rows, then run tile i's QK^T, online softmax and PV on the K/V loaded in the previous iteration, then rotate (k_cur, v_cur, valid_cur) <- (k_next, v_next, valid_next) and rows_next <- rows_next2. While one tile is computed, the next tile's 16 KB and the index load after it are outstanding. At about 4 CTAs per SM that is about 64 KB in flight per SM (about 8.4 MB chip-wide, about 2x stage 1's ~4 MB), towards LESSONS' 3.73 TB/s at 128 KB per CTA in bf16. Keep BLOCK_N=64 and num_warps=4. Set num_stages=1, because the prefetch is explicit and the compiler must not add its own smem ring on top. Everything else is stage 1's code: the block-diagonal [16,128] Q2 with explicit zeros, FP8 q, fp8->bf16 K/V, the exp2-domain online softmax with its -inf guard, fp32 partials [B,S,16,64] plus (m,l), and combine kernel 2 with BLOCK_S = next_pow2(S) (64 and 32 at B=16 and 32; prepare() already warms every BLOCK_S up to 128). Every key t < cache_seqlens[b] is still visited exactly once. The loop bound and masks are unchanged, so a prefetch of a tile past end is fully masked and never dereferenced (-1 entries are never read). No autotune, no host syncs, and scratch is allocated per call from shapes only. Report kernel 1 + kernel 2 us per full-decode shape and KV bytes / kernel-1 time.

## Expected

- `*layer=full load=decode_16*` faster: kernel 1+2 from 200.8 us (1.63 TB/s) to <= 150 us (>= 2.2 TB/s): 1024 CTAs at ~4/SM with one tile prefetched is ~2x stage 1's bytes in flight, and oversubscription stops short agentic requests idling their SM slots; FA3 is 188.8 us, so the shape should beat master's 433.0 us by ~40 us
- `*layer=full load=decode_32*` faster: kernel 1+2 from 394.8 us (1.66 TB/s) to <= 300 us (>= 2.2 TB/s): removes the 288-CTA two-wave tail and doubles in-flight bytes; against FA3's 374.4 us the shape should beat master's 601.7 us by ~75 us
- `*layer=full load=decode_1*` flat: S stays 128 at B=1; the one-ahead K/V prefetch can only shorten stage 1's 21.2 us (FA3 + combine 25.2 us)
- `*layer=sliding*` flat: FA3 unchanged; control
- `*load=verify_*` flat: q_len=8 stays on FA3, including full verify; control

## Risk

Milestones: kernel 1 + combine <= 150 us at decode_16 full and <= 300 us at decode_32 full (0.8x FA3); decode_1 full <= 25 us. If it misses, read kernel 1's bandwidth. Under ~2 TB/s means either registers capped residency below 4 CTAs/SM (the held next K/V tile adds ~32 regs/thread at num_warps=4) or the gather is latency-bound per row; the next stage is then BLOCK_N=128 with num_warps=8, or a two-deep ring. Combine cost grows with S: partials are B*S*16*64*4 B = 4 MB at B=16/32 (written, then read), about 2-3 us, and kernel 2 reads 64 or 32 partials per (b, h). Count it in the bar. Correctness: the rotation must carry valid with its tile, because a tile past end has valid all false and its qk becomes -inf. Tests on this path: batch 1 ctx 3000 full (page 64, the PAGE constexpr path), batch 33 ctx 20000 full spread mixed (S=31, many empty splits), batch 16 ctx 20000 full, batch 1 ctx 80000 full. Grid and split_len come from B and page_table.shape only; nothing input-derived is cached.

## Milestones

Rewrite stage 2 of `FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
