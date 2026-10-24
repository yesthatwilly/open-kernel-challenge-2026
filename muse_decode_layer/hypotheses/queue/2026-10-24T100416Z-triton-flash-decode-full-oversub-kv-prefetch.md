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
review: proceed
status: queued
review_at: 2026-10-24T104027Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T104027Z by an adversarial subagent (model opus).

**Verdict: proceed.** The stage-2 design (S = 1024 // B oversubscription plus a one-tile-ahead register prefetch of K/V, num_stages=1) is sound and within the rules. It is a reasonable next step for the rewrite, but porting fa3-hdim128-head-pairing to decode_layer is the better use of a check: it is measured, it also covers full verify_16/32, and its estimated headline gain (~2.0%) beats this stage's best case (~1.5%). With 4 decode checks left, dispatch the head-pairing port first and re-base this rewrite's milestone 2 on paired-FA3 time.

### Justification
Mechanism: stage 1's profile (e82b11f9d9f64b23) puts _full_decode_split_kernel at 200.8 us (decode_16 full) and 394.8 us (decode_32 full), about 1.63-1.66 TB/s over ~328/655 MB. Unpaired FA3 is 188.8/374.4 us, and the floors are 137.5/208.1 us for the whole layer, so attention alone is far above its own floor. The grid diagnosis holds: 272/288 CTAs is ~2 per SM, 288 > 264 leaves a tail, and under spread=agentic split_len comes from page_table.shape[1], so the splits of short requests empty out early. S = 1024 // B (64 at B=16, 32 at B=32, still 128 at B=1) lets the empty splits drain and refill the SM slots. That is plausible. Register prefetch costs ~32 extra regs/thread (2 x uint8 [64,128] over 128 threads). Smem is ~32 KB/CTA for the bf16 K/V dot operands. Both fit 4 CTAs/SM on H100, so the 'about 64 KB in flight per SM' claim is credible. Expected directions agree with TASK.md: only the full decode_16/32 shapes change (weight 16 each in their loads), decode_1 stays at S=128, and sliding and verify stay on FA3, so they are controls. WEAKEST LINK: the premise that bytes in flight, not the page-size-1 128-byte row gather, is the limit. LESSONS says full decode 'reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit'. The only paired-row reference point is FA3 with head pairing in muse_attention_decode, at 165.2/307.0 us, about 2.0-2.1 TB/s. The milestone of <= 150/<= 300 us (>= 2.2 TB/s) therefore asks this Triton kernel to beat the best paired gather measured on this hardware, from a stage that is today ~20-30% behind it. A second weak point: stage 1 already ran num_stages=3. If Triton's pipeliner already overlapped those indirect loads, dropping to num_stages=1 with a manual one-tile prefetch may not add bytes in flight. The 2x claim assumes the compiler pipelined nothing. Read kernel-1 bandwidth against stage 1's 1.63/1.66 TB/s to tell. Combine cost: 4 MB of partials written and read, BLOCK_S 64/32, ~2-3 us. That is small and correctly counted in the bar.

### Confounders on the current kernel
- marlin-block-m-cap-16: no confound on the judged quantity. It changes Marlin MoE block_size_m (verify shapes only); the rewrite is judged on _full_decode_split_kernel + combine time in profile top_kernels, which shares no launches or memory traffic with Marlin. Headline-only noise.
- sglang-pipeline-parity-router-hilo-m8: no confound. It sets the surrounding pipeline (FusedQKNorm, cache store, q to FP8) that produces q8/k_cache, but the FP8 q and cache layout are inputs to both stage 1 and stage 2 unchanged, so kernel-time deltas are attributable to this change.

### Interference and A/B needs
No companions. This is a rewrite stage judged on its own kernel time against stage 1's 200.8/394.8 us and FA3's 188.8/374.4 us, and the next stage is the follow-up. A control splitting oversubscription from prefetch (e.g. S=1024//B with stage 1's loop) would attribute the two sub-changes. But with 4 decode checks left it costs more than it tells, and stage 1's profile already gives the before-number for both. Neither landed hypothesis touches attention, so ab-revert or ab-tune would show nothing. If kernel-1 bandwidth comes back under ~2 TB/s, the risk section's next step (BLOCK_N=128/num_warps=8 or a two-deep ring) is the right follow-up, not a companion.

### Open questions
- LEVER COMPARISON (answer: head-pairing is the better lever for decode_layer). In muse_attention_decode, fa3-hdim128-head-pairing took FA3 full decode_16/32 from 194.5/377.9 to 165.2/307.0 us (-15%/-19%) and full verify_16/32 from 198.3/383.1 to 173.0/309.4 us, and lost 27-35% at batch 1 (pad/unpad launches). In decode_layer, FA3 is 188.8/374.4 us on full decode_16/32 and 191.9/374.0 us on full verify_16/32. Scaling by those ratios saves ~29/~71/~25/~72 us x weight 16. That is ~2.5%/5.3%/1.3%/2.6% on the decode_16/decode_32/verify_16/verify_32 load times, about +2.0% on the 6-load geomean, if it is gated on batch > 1 so decode_1 and verify_1 keep the unpaired call. This stage, even at its milestones (150/300 us), saves ~39/~74 us on decode only, about +1.5% on the headline, with far more uncertainty. The pairing port is a ~30-line, already-correct patch (view k/v_cache as [pages, page, Hkv/2, 128], zero-padded FP8 q, slice the output, scheduler_metadata=None). Recommend minting 'fa3-hdim128-head-pairing-full-batched' for decode_layer (full layers, batch > 1, decode + verify) and spending one of the 4 checks on it before this stage.
- If head pairing lands, the incumbent for this rewrite becomes paired FA3 (~160/~304 us est. on decode_16/32), not 188.8/374.4. Stage 1 already uses the same paired 128-byte gather and is ~25-30% behind that, so milestone 2 (<= 150/<= 300 us) barely clears the new incumbent. Should stage 2 still be run, or should the rewrite be parked until paired FA3's kernel bandwidth is read from the profile?
- Does decode_layer's kernel_fn pass the judge's scheduler_metadata to FA3 on full layers? The attention port had to drop it for the paired geometry; check that dropping it in decode_layer does not add a metadata launch that eats the gain at batch 16.
- Did stage 1's num_stages=3 actually pipeline the indirect K/V loads? If it did, switching to num_stages=1 plus manual prefetch adds little; compare kernel-1 bytes/time with stage 1's 1.63/1.66 TB/s to separate the oversubscription effect from the prefetch effect.
- How long are the requests under spread=agentic at B=16/32? If most requests are near 80000 tokens, the 'short splits free their slots' argument is weak and the gain must come from residency alone.

