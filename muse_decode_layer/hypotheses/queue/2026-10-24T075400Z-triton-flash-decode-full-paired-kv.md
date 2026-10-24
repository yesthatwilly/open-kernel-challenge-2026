---
problem: muse_decode_layer
slug: triton-flash-decode-full-paired-kv
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace
  flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1
  layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE
  contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes
  per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from
  shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and
  a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to
  BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows
  page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as
  [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs
  the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128]
  bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold
  heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against
  its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept.
  Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16
  in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32
  m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S,
  16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16
  [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024),
  gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8,
  including full verify) and everything else keep master's code. Same design as
  muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full
  decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32
  full, 1.33 TB/s against a 137 us KV floor).
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len=8 keeps FA3 (full verify included); control
risk: high
depends_on: null
priority: 3
kind: rewrite
controls_for: null
review: proceed
status: queued
stage: 1
replaces: "FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine"
review_at: 2026-10-24T083747Z
review_companions: null
---

# triton-flash-decode-full-paired-kv

## Change

On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1 layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128] bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept. Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16 in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32 m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S, 16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16 [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024), gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8, including full verify) and everything else keep master's code. Same design as muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32 full, 1.33 TB/s against a 137 us KV floor).

## Expected

- `*layer=full load=decode_32*` faster: 655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load
- `*layer=full load=decode_16*` faster: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
- `*layer=full load=decode_1*` flat: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
- `*layer=sliding*` flat: FA3 unchanged; control
- `*load=verify_*` flat: q_len=8 keeps FA3 (full verify included); control

## Risk

Risk: Triton's gather loads through a page table may not reach 3 TB/s; check fa3-full-single-split-probe's result first if it has landed in done/ (if FA3 is gather-pattern-bound the 128-byte-row access is the lever, if MLP-bound the 2-wave grid is). Competes with trtllm-decode-full-attention (queued, same shapes, same call site): whichever lands first is the other's base, so read this one against the base it was built on. Correctness: the block-diagonal Q2 makes the off-group dot products exactly zero contributions only because the zero columns multiply real K values; keep the zero padding explicit, never masked loads of q only. Tests through this path: batch 1 ctx 3000 full, batch 33 ctx 20000 full spread mixed page 1, batch 1 ctx 80000 full page 1, batch 16 ctx 20000 full page 1 (plus any page-64 full decode). Every key in the context is attended; no block skipping. Graph-safe: S and the split length come from B and page_table.shape only; scratch is allocated per call.

## Adversarial review

Reviewed 2026-10-24T083747Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite of the most expensive kernel on the full-decode shapes, and it breaks no rule. Its decode_32 baseline is wrong, though: on this problem FA3 takes 374.4 us of a 601.7 us shape, not 491.6 of 623. Judge it on the replaced kernels' own time against 374.4/188.8/25.2 us, and look hard at whether Triton pipelines the page-table-indirect loads.

### Justification
The mechanism is plausible. Full-decode FA3 is the largest single kernel here. Master's profile (0730135cdcac43b7) shows flash_fwd at 188.8 us of 433.0 us on decode_16 full (42.9%) and 374.4 us of 601.7 us on decode_32 full (59.5%). Against about 328 MB and 655 MB of FP8 KV, that is roughly 1.74-1.75 TB/s, which matches LESSONS ('full-attention decode reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit'). Shape floors are 137.5 and 208.1 us, and the KV-only floor is about 137 us at decode_32, so there is real headroom. Reading each token's two KV heads as one contiguous 128-byte row, with about 270-290 CTAs that stay resident (48 KB of smem each, so about 13.8 MB in flight), is a reasonable way to raise achieved bandwidth.

The expected numbers are inflated. The 491.6/623 us figure is the overlap-inflated muse_attention_decode row (493.3 us 'GPU time' on a 377.9 us shape, which includes PDL wait). It is not this problem's profile. Corrected, 3 TB/s would take decode_32 full from about 374 to 220 us, saving about 155 us per shape. Times weight 16 that is about 2.5 ms of a 21.3 ms decode_32 load, or about 1.13x on that load, not '~20%'. decode_16 full would go from about 189 to 110 us, about 1.07x on its load. Geomean over six loads is about 1.03x, which a single run can still resolve.

The decode_1 'flat' is right in direction but the arithmetic is off. 80,000/128 = 625 tokens is about 10 BLOCK_N=64 tiles, not about 5. The incumbent is FA3 18.4 us plus combine 6.8 us = 25.2 us. A latency-bound 10-tile chain of dependent gathers could land anywhere from about 15 to 30 us, so a few us worse is possible and costs at most about 0.7% on that load.

The weakest link is software pipelining of a doubly indirect load. Each tile loads page_table rows first, and the K/V addresses depend on them. If Triton's pipeliner does not overlap the next tile's index load and gather with the current tile's math (num_stages=3 is no guarantee for indirect loads), each CTA serialises two DRAM latencies per 16 KB tile. That would land at about 2 TB/s, a small gain over FA3 instead of a large one. A secondary uncertainty is that the 128-byte-row argument assumes FA3 wastes or duplicates traffic by reading 64-byte halves per kv-head CTA. With 32-byte sector granularity it may not waste bytes, only request count. The lever may then be grid shape and occupancy rather than row width. fa3-full-single-split-probe would separate the two, but one run of this rewrite cannot.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: this is the base pipeline. It defines the FA3 call site, the FP8 q cast, o_norm_gate and scheduler_metadata usage. It does not confound the result. The change replaces exactly the call it introduced and keeps its q.to(FP8) rounding, so the numerics stay comparable. One caveat: FA3 may currently overlap its neighbours via PDL ('GPU busy' < kernel sum on decode_32: 604.3 vs 629.3 us). Plain Triton launches lose that overlap, which costs a few us that is not the kernel's own fault. Read kernel time, not only shape time.
- marlin-block-m-cap-16: no interaction. At T<=32 (all decode shapes) the cap leaves block_size_m at 8/16 as before, and the change touches only the attention path. The decode shapes this hypothesis targets were flat under that landing (0.996-1.002x).

### Interference and A/B needs
No companion is warranted. This is kind: rewrite and is judged on the replaced kernels' own time (flash_fwd + FlashAttnFwdCombine on full decode) against the incumbent and the floor, and both are readable from one profile. Neither landed hypothesis touches attention memory traffic or numerics in a way an ab-revert would disambiguate. The one sub-change worth isolating, contiguous 128-byte rows versus the grid/occupancy change, is already covered by the queued fa3-full-single-split-probe on the FA3 side. The natural follow-up is stage 2 (index prefetch, BLOCK_N/num_stages tuning), not a companion. Ordering caveat: trtllm-decode-full-attention and the queued fused-preattention-triton touch the same call site and the q production. Whichever lands first becomes the base, so compare this run's kernel time against the FA3 time in its own run's profile.

### Open questions
- The hypothesis is kind: rewrite but has no ## Milestones section. What is stage 1's pass bar? I suggest replaced-kernel time at or below 0.8x FA3 (<=300 us decode_32, <=150 us decode_16) and achieved KV bandwidth reported against the 137 us floor.
- Does the Triton build on the judge image pipeline the in-loop page_table index load ahead of the dependent K/V gather? If not, stage 2 should prefetch the next tile's row indices explicitly (load indices for tile i+1 before computing tile i) or stage all of a split's indices up front.
- Is the combine kernel (B=16/32, S=17/9) cheap enough? FA3's own combine is about 3 us there, and a second Triton launch adds about 0.3 us plus the partials round-trip of about 1 MB.
- Tests with page 64 and full decode (e.g. batch 1 ctx 3000 full, if its default page is 64) exercise the t//P, t%P path. Confirm which full-decode tests use page 64.
- If fa3-full-single-split-probe lands first and shows FA3 is grid- or occupancy-bound rather than gather-pattern-bound, the 128-byte-row framing in this hypothesis's justification should be revised, though the kernel design still stands.

