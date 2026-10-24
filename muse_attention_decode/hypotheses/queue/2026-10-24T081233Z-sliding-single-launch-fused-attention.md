---
problem: muse_attention_decode
slug: sliding-single-launch-fused-attention
created: 2026-10-24T081233Z
base: master
area: sliding-layer attention (whole-path rewrite)
change: >
  Stage 1 of a rewrite of the sliding-layer path. Built on
  hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding
  layers (config window != 0) kernel_fn launches ONE new Triton kernel, _sliding_fused_kernel, in
  place of _pre_attention_kernel + flash_attn_with_kvcache (its prepare_varlen_num_blocks, main
  and FlashAttnFwdCombine launches) + rms_norm_gate, and returns its bf16 [B, L, Hq*64] output.
  Full layers (window == 0) keep the base's path unchanged (_pre_attention_kernel + FA3 with
  scheduler_metadata + rms_norm_gate). GRID (B, Hkv, S): one program per (request b, kv head g,
  window split s); each program serves the group's 8 query heads for all L block tokens as one M =
  8*L row tile (rows (j, h); decode M = 8 padded to 16 for tl.dot, verify M = 64). S is a
  constexpr chosen from config batch so the launch has ~64-128 CTAs: S = 16 at B=1 (128 keys per
  split), 4 at B=16, 2 at B=32 (cap 16, power of two; S = 1 allowed). RANGE: seq = seq_lens[b];
  old keys are positions [lo, seq-1], lo = max(0, seq - (window-1)) (the union of the L tokens'
  windows; token j at p_j = seq + j attends t in [p_j - window + 1, p_j]); the old range is cut
  into S contiguous chunks of ceil(count/S) rounded up to BLOCK_N (64); a chunk may be empty
  (short-context tests ctx 10 / 2047), which still writes m = -inf, l = 0 and takes a ticket.
  PROLOGUE (every program): load the 8*L query rows from qkv (head h*64), scaleless RMSNorm (eps
  1e-5, mean of 64 squares), interleaved RoPE with pos = positions[b*L+j] and the cached inv_freq
  exactly as in the base kernel, keep fp32 in registers and feed tl.dot as bf16 (finer than E4M3;
  or E4M3 x E4M3 tl.dot as FA3 does; either is allowed and must be decided once, recorded in the
  attempt). Only program s == S-1 also builds the L new key/value rows of head g: RMSNorm+RoPE the
  k row, divide k and v by k_scale/v_scale, cast to E4M3, store them at cache_loc[b*L+j] (the
  entries the judge checks), and keep the E4M3 values in registers; it attends these dequantized
  E4M3 registers (not the fp32 values, so the output matches the entries stored) as one extra
  16-row key block masked to i < L and causal i <= j. No program ever reads a new row back from
  the cache, so there is no store-then-load race; the old rows [lo, seq-1] it reads are never
  written in this call. MAIN LOOP: for each BLOCK_N block of its chunk, rows = page_table[b, t //
  PAGE] * PAGE + t % PAGE (PAGE = config page_size constexpr: 1 scored, 64 in tests, same code),
  load k_cache.view(-1, Hkv, 64)[rows, g] and v likewise as [BLOCK_N, 64] E4M3 (64 contiguous
  bytes per row), num_stages 2-3 so the gathers are in flight, convert to bf16 (exact), S = q @
  k^T in fp32 times sm_scale*qk_scale_factor*k_scale, mask per row j: t >= p_j - window + 1 (t <=
  p_j holds for all old keys), online softmax in fp32 (exp2 with log2e folded), acc += P(bf16) @
  V, v_scale applied once at the end. Every key in each token's window is scored: no skipping.
  COMBINE (in-kernel, S > 1): each program writes (m, l, acc[M, 64]) fp32 to a workspace [B*Hkv,
  S, M, 66], then tl.debug_barrier and t = tl.atomic_add(counter[b*Hkv+g], 1, sem=acq_rel); the
  program that draws t == S-1 reads all S partials (cache_modifier .cg), rescales by exp(m_s -
  m_max), sums, divides by l; then resets counter to 0 for the next replay. The workspace is
  torch.empty per call (graph pool; shape from config only); the int32 counters are a
  zero-initialised buffer allocated once per (B, Hkv, device) before capture and left zeroed by
  every call, holding no input-derived data. With S == 1 the program goes straight to the
  epilogue. EPILOGUE (the last program): per (j, h) row o = acc/l; r = rsqrt(mean(o^2) + 1e-5)
  over 64; load the gate logits qkv[b*L+j, (Hq+2*Hkv)*64 + h*64 : +64], o * r * sigmoid(gate) in
  fp32, cast bf16, store to out[b, j, h*64 : (h+1)*64] (no intermediate bf16 rounding of o: finer
  than SGLang). kv_scale != 1 tests: k_scale folded into the score scale, v_scale into o before
  the norm, divisions applied on the new rows' store. No host syncs, no .item, no value-dependent
  branching or allocation; everything branches only on config (batch, q_len, page_size, window).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: incumbent 16.77 us (SGLang 20.90, floor 0.11); replaced kernels 15.1 us GPU in 5 launches (FA3 8.2, combine 2.5, prepare 1.8, pre-attn 1.5, norm-gate 1.0); 32 CTAs x 128 keys x 128 B = 512 KB in one launch, target <= 6 us replaced time
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: incumbent 20.80 us (SGLang 27.01, floor 1.65); replaced 17.9 us; 128 CTAs stream 8 MB, target <= 7 us
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: incumbent 25.14 us (SGLang 31.41, floor 3.29); replaced 21.2 us; 128 CTAs stream 16 MB, target <= 9 us
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: incumbent 17.12 us (SGLang 22.83, floor 0.12); replaced 16.2 us; M=64 rows per program, same 512 KB, target <= 7 us
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent 22.50 us (SGLang 29.34, floor 1.86); replaced 20.4 us, target <= 8 us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: incumbent 28.72 us (SGLang 34.30, floor 3.72); replaced 26.7 us, target <= 10 us
  - shapes: "*layer=full*"
    direction: flat
    why: full layers keep the base path (pre-attn kernel + FA3 + rms_norm_gate) byte for byte; control
risk: high
depends_on: fused-preattention-triton
priority: 1
kind: rewrite
controls_for: null
stage: 1
replaces: "sliding-layer attention after the QKV projection: _pre_attention_kernel + FA3 (+combine/prepare) + rms_norm_gate"
review: required
status: queued
---

# sliding-single-launch-fused-attention

## Change

Stage 1 of a rewrite of the sliding-layer path. Built on hyp/muse_attention_decode/fused-preattention-triton (1.127x, job dfee6f2939974bad). On sliding layers (config window != 0) kernel_fn launches ONE new Triton kernel, _sliding_fused_kernel, in place of _pre_attention_kernel + flash_attn_with_kvcache (its prepare_varlen_num_blocks, main and FlashAttnFwdCombine launches) + rms_norm_gate, and returns its bf16 [B, L, Hq*64] output. Full layers (window == 0) keep the base's path unchanged (_pre_attention_kernel + FA3 with scheduler_metadata + rms_norm_gate). GRID (B, Hkv, S): one program per (request b, kv head g, window split s); each program serves the group's 8 query heads for all L block tokens as one M = 8*L row tile (rows (j, h); decode M = 8 padded to 16 for tl.dot, verify M = 64). S is a constexpr chosen from config batch so the launch has ~64-128 CTAs: S = 16 at B=1 (128 keys per split), 4 at B=16, 2 at B=32 (cap 16, power of two; S = 1 allowed). RANGE: seq = seq_lens[b]; old keys are positions [lo, seq-1], lo = max(0, seq - (window-1)) (the union of the L tokens' windows; token j at p_j = seq + j attends t in [p_j - window + 1, p_j]); the old range is cut into S contiguous chunks of ceil(count/S) rounded up to BLOCK_N (64); a chunk may be empty (short-context tests ctx 10 / 2047), which still writes m = -inf, l = 0 and takes a ticket. PROLOGUE (every program): load the 8*L query rows from qkv (head h*64), scaleless RMSNorm (eps 1e-5, mean of 64 squares), interleaved RoPE with pos = positions[b*L+j] and the cached inv_freq exactly as in the base kernel, keep fp32 in registers and feed tl.dot as bf16 (finer than E4M3; or E4M3 x E4M3 tl.dot as FA3 does; either is allowed and must be decided once, recorded in the attempt). Only program s == S-1 also builds the L new key/value rows of head g: RMSNorm+RoPE the k row, divide k and v by k_scale/v_scale, cast to E4M3, store them at cache_loc[b*L+j] (the entries the judge checks), and keep the E4M3 values in registers; it attends these dequantized E4M3 registers (not the fp32 values, so the output matches the entries stored) as one extra 16-row key block masked to i < L and causal i <= j. No program ever reads a new row back from the cache, so there is no store-then-load race; the old rows [lo, seq-1] it reads are never written in this call. MAIN LOOP: for each BLOCK_N block of its chunk, rows = page_table[b, t // PAGE] * PAGE + t % PAGE (PAGE = config page_size constexpr: 1 scored, 64 in tests, same code), load k_cache.view(-1, Hkv, 64)[rows, g] and v likewise as [BLOCK_N, 64] E4M3 (64 contiguous bytes per row), num_stages 2-3 so the gathers are in flight, convert to bf16 (exact), S = q @ k^T in fp32 times sm_scale*qk_scale_factor*k_scale, mask per row j: t >= p_j - window + 1 (t <= p_j holds for all old keys), online softmax in fp32 (exp2 with log2e folded), acc += P(bf16) @ V, v_scale applied once at the end. Every key in each token's window is scored: no skipping. COMBINE (in-kernel, S > 1): each program writes (m, l, acc[M, 64]) fp32 to a workspace [B*Hkv, S, M, 66], then tl.debug_barrier and t = tl.atomic_add(counter[b*Hkv+g], 1, sem=acq_rel); the program that draws t == S-1 reads all S partials (cache_modifier .cg), rescales by exp(m_s - m_max), sums, divides by l; then resets counter to 0 for the next replay. The workspace is torch.empty per call (graph pool; shape from config only); the int32 counters are a zero-initialised buffer allocated once per (B, Hkv, device) before capture and left zeroed by every call, holding no input-derived data. With S == 1 the program goes straight to the epilogue. EPILOGUE (the last program): per (j, h) row o = acc/l; r = rsqrt(mean(o^2) + 1e-5) over 64; load the gate logits qkv[b*L+j, (Hq+2*Hkv)*64 + h*64 : +64], o * r * sigmoid(gate) in fp32, cast bf16, store to out[b, j, h*64 : (h+1)*64] (no intermediate bf16 rounding of o: finer than SGLang). kv_scale != 1 tests: k_scale folded into the score scale, v_scale into o before the norm, divisions applied on the new rows' store. No host syncs, no .item, no value-dependent branching or allocation; everything branches only on config (batch, q_len, page_size, window).

## Expected

- `*layer=sliding load=decode_1 *` faster: incumbent 16.77 us (SGLang 20.90, floor 0.11); replaced kernels 15.1 us GPU in 5 launches (FA3 8.2, combine 2.5, prepare 1.8, pre-attn 1.5, norm-gate 1.0); 32 CTAs x 128 keys x 128 B = 512 KB in one launch, target <= 6 us replaced time
- `*layer=sliding load=decode_16 *` faster: incumbent 20.80 us (SGLang 27.01, floor 1.65); replaced 17.9 us; 128 CTAs stream 8 MB, target <= 7 us
- `*layer=sliding load=decode_32 *` faster: incumbent 25.14 us (SGLang 31.41, floor 3.29); replaced 21.2 us; 128 CTAs stream 16 MB, target <= 9 us
- `*layer=sliding load=verify_1 *` faster: incumbent 17.12 us (SGLang 22.83, floor 0.12); replaced 16.2 us; M=64 rows per program, same 512 KB, target <= 7 us
- `*layer=sliding load=verify_16 *` faster: incumbent 22.50 us (SGLang 29.34, floor 1.86); replaced 20.4 us, target <= 8 us
- `*layer=sliding load=verify_32 *` faster: incumbent 28.72 us (SGLang 34.30, floor 3.72); replaced 26.7 us, target <= 10 us
- `*layer=full*` flat: full layers keep the base path (pre-attn kernel + FA3 + rms_norm_gate) byte for byte; control

## Risk

Milestones (judged on the replaced kernels' GPU time from the profile, not the headline): M1 all 16 tests pass and every sliding shape is faster than the incumbent; M2 each sliding shape's single kernel <= 50% of the replaced 15-27 us; M3 within 3x the floor + ~3 us launch on B=16/32. Subsumes triton-window-attention-sliding, fused-sliding-preattn-into-window-kernel and fused-sliding-normgate-epilogue (the three-step stack done at once) and makes fa3-single-split-sliding / fa3-window-trimmed-page-table moot for sliding layers if it lands; churnsubs/reviewer may park those. Risks: (1) verify q_len=8 causal inside the block: the new-key block mask is i <= j and the old-key mask is t >= seq + j - window + 1 per row; the edges test plants keys at p0-2047 (inside for j=0, outside for j>=1) and p0-2048 (outside) and must pass. (2) page_size 64 tests (ctx 10 and ctx 2047 sliding run at page 64): the row formula is generic; if it fails, branch on config page_size != 1 to the incumbent path (allowed: config branch) and say so, scored shapes are page 1. (3) numerics vs FA3: fp32 online softmax and an S-way LSE combine differ from FA3 only in reduction order; bf16 queries are finer than FA3's E4M3; the output bound is 2.5x the E4M3-query error + 1.5%. Attend the E4M3-rounded new k/v, not fp32. Near-zero-row rule: eps inside sqrt over the mean of 64 squares, as reference._rms. (4) the ticket combine: partial stores must be visible before the atomic (acq_rel, gpu scope) and the last program must read with .cg; the counter must be reset to 0 by that program or the next replay deadlocks/mis-combines; if Triton memory ordering proves flaky, fall back to S = 1 (no combine) and accept the B=1 bandwidth cost, recorded as a stage-1 result. Persistent zeroed counters are workspace, not input-derived data, but the reviewer should confirm that reading of the caching rule. (5) register pressure: M=64 x 64 fp32 acc plus a 64x64 score tile at 4 warps; BLOCK_N 32 if Triton spills; decode pads M 8 -> 16. (6) single-program gather throughput at page 1 (64 B rows): bytes in flight per CTA = BLOCK_N x 128 B x num_stages; if B=1 stays above ~6 us, raise S rather than BLOCK_N. Stage 2 candidates: E4M3 tl.dot for QK^T, warp-specialised loads, a persistent grid.

## Milestones

Rewrite stage 1 of `sliding-layer attention after the QKV projection: _pre_attention_kernel + FA3 (+combine/prepare) + rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
