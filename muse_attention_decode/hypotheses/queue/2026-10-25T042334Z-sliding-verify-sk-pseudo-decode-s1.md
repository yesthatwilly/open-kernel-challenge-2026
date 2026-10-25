---
problem: muse_attention_decode
slug: sliding-verify-sk-pseudo-decode-s1
created: 2026-10-25T042334Z
base: master
area: sliding-layer attention (verify, stream-K pseudo-decode kernel)
change: >
  Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode
  stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64 and the two-level ticket
  at B=1, landed as full-decode1-streamk-s3-two-level-ticket, c3c1b01). Idea: a verify block of 8
  tokens is 8 decode queries, each over its own 2048-key window, so route SLIDING verify (window
  != 0, q_len 8, tp8, head_dim 64, page 1, 1<=B<=32; config-only predicate) through a template
  variant of the stream-K decode kernel over P = 8B pseudo-requests, keeping its gather-bound
  16-row (2 kv heads x 8 q heads) mma.sync design, instead of the 64-row verify tiles that LESSONS
  says are MMA-bound. Kernel edit, new template flag PSEUDO (false = the landed decode code,
  byte-identical): pseudo-request p in [0, 8B) maps to request b = p / 8 and token j = p % 8;
  seq_p = cache_seqlens[b] - 7 + j (keys 0..seq_p-1, so causality inside the block is exact), lo_p
  = max(0, seq_p - window) (matches FA3 window_size=(window-1, 0) as master passes it; every key
  in each window attended), tiles_p = ceil((seq_p - lo_p) / 64) with tile k covering keys lo_p +
  64k .. +63 (page 1, no alignment needed; mask t < seq_p on the last tile). The tile prefix over
  P <= 256 pseudo-requests is built in smem from cache_seqlens exactly as the decode kernel builds
  its per-request prefix (tp_s grows to 257 entries). Page lookup uses page_table row b; the FP8 q
  block is token b*8+j of the q that master _pre_attention_kernel already writes (PAIRED=False,
  ROPE on; it also stores the 8 new K/V rows, so the kernel needs no rope and no new-key build);
  the ticket epilogue norm-gates with qkv row b*8+j and stores out row b*8+j; workspace slots grid
  + P; ticket counters sized for P. Reuse is from L2: consecutive pseudo-requests share 2041 of
  2048 keys and stream-K puts them on neighbouring CTAs at the same time, so DRAM bytes stay ~1x
  (8.4 MB at B=16) while L2 reads are 8x. Hard gate in prepare (LESSONS: PATH CHOSEN must compare
  printed timings): per band B=1, B=2..16, B=17..32 time (cold L2, median of 10, CUDA events) the
  pseudo-decode kernel (single-level and two-level ticket variants, grid SM and 2xSM) against
  master FA3 path (prepare + main + combine + norm-gate, pre-attn excluded from both), print every
  number, and route a band only if its best variant is >= 5% faster; otherwise that band keeps
  FA3. Self-check against an fp32 windowed causal reference: window 2048 with lens < 8, < window,
  < window + 8 and 80000, B = 1/3/16/32, kv_scale 1 and 0.25, grids 3/7/device; on failure only
  this route is disabled (PATH CHOSEN: sliding verify FA3 (fallback: <case, rel L2>)).
  BUILD/SAFETY as the landed kernel (TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, -lineinfo,
  in-process canary, sync+print after each kernel, never a child process). Sliding decode, full
  layers and page 64 untouched. First measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: replaced ~14.3 us (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate ~1.1) of a 16.7 us shape, floor 0.1; 8 pseudo x 32 tiles = ~264 tiles = one tile per CTA at grid 2xSM plus one ticket merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: replaced ~17 us (FA3 10.7 + combine 4.1 + prepare + norm-gate) of 22.7; 67 MB of L2 reads over 8.4 MB of DRAM. M2 shape <= 18 us
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: replaced ~21 us of 28.7; 134 MB L2 reads may make this band lose, then the hard gate keeps FA3 (flat)
  - shapes: "*load=decode_* *layer=full*"
    direction: flat
    why: PSEUDO=false instantiation is the landed decode code; full and decode paths untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate"
review: required
status: queued
---

# sliding-verify-sk-pseudo-decode-s1

## Change

Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64 and the two-level ticket at B=1, landed as full-decode1-streamk-s3-two-level-ticket, c3c1b01). Idea: a verify block of 8 tokens is 8 decode queries, each over its own 2048-key window, so route SLIDING verify (window != 0, q_len 8, tp8, head_dim 64, page 1, 1<=B<=32; config-only predicate) through a template variant of the stream-K decode kernel over P = 8B pseudo-requests, keeping its gather-bound 16-row (2 kv heads x 8 q heads) mma.sync design, instead of the 64-row verify tiles that LESSONS says are MMA-bound. Kernel edit, new template flag PSEUDO (false = the landed decode code, byte-identical): pseudo-request p in [0, 8B) maps to request b = p / 8 and token j = p % 8; seq_p = cache_seqlens[b] - 7 + j (keys 0..seq_p-1, so causality inside the block is exact), lo_p = max(0, seq_p - window) (matches FA3 window_size=(window-1, 0) as master passes it; every key in each window attended), tiles_p = ceil((seq_p - lo_p) / 64) with tile k covering keys lo_p + 64k .. +63 (page 1, no alignment needed; mask t < seq_p on the last tile). The tile prefix over P <= 256 pseudo-requests is built in smem from cache_seqlens exactly as the decode kernel builds its per-request prefix (tp_s grows to 257 entries). Page lookup uses page_table row b; the FP8 q block is token b*8+j of the q that master _pre_attention_kernel already writes (PAIRED=False, ROPE on; it also stores the 8 new K/V rows, so the kernel needs no rope and no new-key build); the ticket epilogue norm-gates with qkv row b*8+j and stores out row b*8+j; workspace slots grid + P; ticket counters sized for P. Reuse is from L2: consecutive pseudo-requests share 2041 of 2048 keys and stream-K puts them on neighbouring CTAs at the same time, so DRAM bytes stay ~1x (8.4 MB at B=16) while L2 reads are 8x. Hard gate in prepare (LESSONS: PATH CHOSEN must compare printed timings): per band B=1, B=2..16, B=17..32 time (cold L2, median of 10, CUDA events) the pseudo-decode kernel (single-level and two-level ticket variants, grid SM and 2xSM) against master FA3 path (prepare + main + combine + norm-gate, pre-attn excluded from both), print every number, and route a band only if its best variant is >= 5% faster; otherwise that band keeps FA3. Self-check against an fp32 windowed causal reference: window 2048 with lens < 8, < window, < window + 8 and 80000, B = 1/3/16/32, kv_scale 1 and 0.25, grids 3/7/device; on failure only this route is disabled (PATH CHOSEN: sliding verify FA3 (fallback: <case, rel L2>)). BUILD/SAFETY as the landed kernel (TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, -lineinfo, in-process canary, sync+print after each kernel, never a child process). Sliding decode, full layers and page 64 untouched. First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_1 *` faster: replaced ~14.3 us (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate ~1.1) of a 16.7 us shape, floor 0.1; 8 pseudo x 32 tiles = ~264 tiles = one tile per CTA at grid 2xSM plus one ticket merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
- `*layer=sliding load=verify_16*` faster: replaced ~17 us (FA3 10.7 + combine 4.1 + prepare + norm-gate) of 22.7; 67 MB of L2 reads over 8.4 MB of DRAM. M2 shape <= 18 us
- `*layer=sliding load=verify_32*` faster: replaced ~21 us of 28.7; 134 MB L2 reads may make this band lose, then the hard gate keeps FA3 (flat)
- `*load=decode_* *layer=full*` flat: PSEUDO=false instantiation is the landed decode code; full and decode paths untouched

## Risk

Kill: if the prepare timing shows the pseudo-decode kernel slower than the FA3 path at B=1 (the most latency-bound band), the line is dead. Risk: 8x L2 read amplification at B=32; tp_s smem growth; correctness at the window lower edge (tests put keys at window edges).

## Milestones

Rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
