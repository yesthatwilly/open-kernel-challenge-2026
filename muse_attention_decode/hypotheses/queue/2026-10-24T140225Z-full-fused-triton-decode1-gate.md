---
problem: muse_attention_decode
slug: full-fused-triton-decode1-gate
created: 2026-10-24T140225Z
base: master
area: full-layer attention (batch-1 decode gate)
change: >
  Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and
  q_len == 1) through master _sliding_fused_kernel via _sliding_forward, replacing the unpaired
  path there (_pre_attention_kernel + prepare_varlen_num_blocks + FA3 main 14.8 us +
  FlashAttnFwdCombine 7.2 us + rms_norm_gate). Kernel edits, all behind new constexprs so the
  sliding instantiation compiles to the same code: (a) FULL: tl.constexpr (= window == 0): lo = 0
  instead of max(seq - (WINDOW - 1), 0), and the old-key validity is tmask only (every old key t <
  seq <= p_row is attended; no window lower edge); ROPE is already off for full layers (rope_theta
  0) and the new key/value is built, stored at cache_loc and attended from registers exactly as on
  sliding; (b) the split combine loops: keep tl.static_range(S) when S <= 16 (sliding, unchanged)
  and use a runtime range(S) loop when S > 16 (a constexpr branch), so a deep split does not
  unroll 64 copies. _sliding_forward chooses, for FULL only, S = 64 and BLOCK_N = 128 (80000 old
  keys -> 1250 per split, ~10 tiles of 128), grid (1, 2, 64) = 128 CTAs (about one per SM),
  num_warps=4, num_stages=3; workspace ws_acc [128, 16, 64] fp32 (512 KB) and ws_ml from config;
  the counters come from _counters(1, 2) as today. Sliding keeps S from _num_splits, BLOCK_N=64,
  num_stages=2. scheduler_metadata is not read on this path. Full batch>1 (paired FA3), full
  verify_1 and every sliding shape are unchanged.
expected:
  - shapes: "*layer=full load=decode_1*"
    direction: faster
    why: incumbent 26.99 us shape / 24.4 us kernels against a 4.3 us floor (20 MB of KV); FA3 combine alone is 7.2 us and prepare 2.8 us; one launch with 128 CTAs streaming ~160 KB each with 3-stage pipelining plus an in-kernel 64-way combine should be ~12-16 us; full is 42% of the decode_1 load, ~1.1-1.15x on it
  - shapes: "*layer=sliding*"
    direction: flat
    why: same constexprs and code for sliding instantiations
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch>1 keeps the paired FA3 path
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 untouched
risk: medium
depends_on: sliding-fused-triton-decode1-gate
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# full-fused-triton-decode1-gate

## Change

Built on master. Route full-attention batch-1 decode (config window == 0 and batch == 1 and q_len == 1) through master _sliding_fused_kernel via _sliding_forward, replacing the unpaired path there (_pre_attention_kernel + prepare_varlen_num_blocks + FA3 main 14.8 us + FlashAttnFwdCombine 7.2 us + rms_norm_gate). Kernel edits, all behind new constexprs so the sliding instantiation compiles to the same code: (a) FULL: tl.constexpr (= window == 0): lo = 0 instead of max(seq - (WINDOW - 1), 0), and the old-key validity is tmask only (every old key t < seq <= p_row is attended; no window lower edge); ROPE is already off for full layers (rope_theta 0) and the new key/value is built, stored at cache_loc and attended from registers exactly as on sliding; (b) the split combine loops: keep tl.static_range(S) when S <= 16 (sliding, unchanged) and use a runtime range(S) loop when S > 16 (a constexpr branch), so a deep split does not unroll 64 copies. _sliding_forward chooses, for FULL only, S = 64 and BLOCK_N = 128 (80000 old keys -> 1250 per split, ~10 tiles of 128), grid (1, 2, 64) = 128 CTAs (about one per SM), num_warps=4, num_stages=3; workspace ws_acc [128, 16, 64] fp32 (512 KB) and ws_ml from config; the counters come from _counters(1, 2) as today. Sliding keeps S from _num_splits, BLOCK_N=64, num_stages=2. scheduler_metadata is not read on this path. Full batch>1 (paired FA3), full verify_1 and every sliding shape are unchanged.

## Expected

- `*layer=full load=decode_1*` faster: incumbent 26.99 us shape / 24.4 us kernels against a 4.3 us floor (20 MB of KV); FA3 combine alone is 7.2 us and prepare 2.8 us; one launch with 128 CTAs streaming ~160 KB each with 3-stage pipelining plus an in-kernel 64-way combine should be ~12-16 us; full is 42% of the decode_1 load, ~1.1-1.15x on it
- `*layer=sliding*` flat: same constexprs and code for sliding instantiations
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch>1 keeps the paired FA3 path
- `*load=verify_*` flat: q_len 8 untouched

## Risk

Analogue of sliding-fused-triton-decode1-gate (sliding decode_1 16.6 -> 13.1 us) on the full layer, where FA3 at batch 1 spends 30% of the shape in its combine. Risks: no test runs full decode at batch 1, so this path is checked only on the scored call (correctness is checked on every timed call); the implementer should reason through FULL masking against the reference carefully (old keys [0, seq-1] plus the new key, causal trivially satisfied at L=1). The last-ticket CTA reads 64 partials x 4 KB per kv head (256 KB) from L2, ~2-3 us; if that dominates, an ab-tune at S=32 halves it. decode_layer triton-flash-decode-full-paired-kv was flat at decode_1 with a separate combine launch and no fused pre-attention; this removes four of five launches as well as the combine.
