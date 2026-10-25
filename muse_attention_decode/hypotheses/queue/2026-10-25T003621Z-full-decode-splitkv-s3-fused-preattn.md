---
problem: muse_attention_decode
slug: full-decode-splitkv-s3-fused-preattn
created: 2026-10-25T003621Z
base: master
area: full-layer attention (decode, splitkv stream-K kernel prologue)
change: >
  Stage 3 of the full-decode rewrite, built on the stage-2 branch
  hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token
  tiles + per-request ticket combine/norm-gate in splitkv_attn_sk_kernel, module
  muse_full_decode_splitkv_v2), whatever its score. Gate unchanged (_splitkv_gate: full, q_len 1,
  1 < batch <= 64, tp8, head_dim 64, page 1). One change: remove the separate
  _pre_attention_kernel launch on the gated path so full decode_16/32 run ONE kernel
  (REWRITE_READ_2026-10-25 sections 1 and 5 list fused pre-attention as the idea worth keeping on
  this kernel). New kernel splitkv_attn_sk_fused_kernel in a NEW module
  muse_full_decode_splitkv_v3 (v2 stays compiled as the fallback): takes the raw qkv
  pointer/stride, positions, cache_loc, inv_freq, k_scale, v_scale instead of the precomputed FP8
  q. (a) Q: when a CTA starts a request segment it builds that request q block in smem from qkv:
  the exact math of master _pre_attention_kernel for the 16 q heads (_norm_rope: scaleless RMSNorm
  over 64, eps 1e-5, RoPE with the same inv_freq table at positions[b], then the same FP8 cast and
  scale master writes), one warp per 2 heads, fp32 math; the redundancy (each request is touched
  by ~2-17 CTAs) is 2 KB of qkv per segment. (b) New K/V row: the CTA whose range contains request
  b last tile (key seq_b - 1, which is the token being decoded, stored at cache_loc[b]) computes
  the 2 kv heads new k (norm+rope+fp8) and v (fp8) exactly as master, stores them to
  k_cache/v_cache at cache_loc[b] (the cache update stays an output), then __syncthreads() before
  issuing that tile cp.async so the ring reads the stored row (same CTA, L2-coherent .cg loads).
  No other CTA reads key seq_b - 1, so no cross-CTA ordering is needed; the self-check asserts
  that ownership invariant. Launches on full decode_16/32 go from two (pre-attn, A) to one.
  BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image
  check + synthetic canary + self-check with torch.cuda.synchronize() and a print after each
  kernel; never a child process). The self-check additionally compares the stored cache rows of v3
  against master _pre_attention_kernel by rel L2 (not byte equality: LESSONS 2026-10-25, Triton
  sin/cos approximations) and q-dependent outputs by rel L2 < 5e-3. Fallback on any failure: v2
  path (pre-attn + v2 kernel), printing PATH CHOSEN: full decode streamk v2 (fallback: <reason>);
  success prints PATH CHOSEN: full decode streamk fused-preattn v3. Every other shape byte for
  byte. First measurement is an okc check, never a scored submit.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced: _pre_attention_kernel 2.1 us + its launch gap on top of the stage-2 kernel; M1 16/16 tests and <= stage-2 shape time; M2 shape -2..-3 us vs stage 2 (floor 63.2)"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced: _pre_attention_kernel 2.0 us + launch gap; M2 shape -2..-3 us vs stage 2 (floor 126.4)"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside the gate, master code byte for byte
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: _pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)
review: required
status: queued
---

# full-decode-splitkv-s3-fused-preattn

## Change

Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token tiles + per-request ticket combine/norm-gate in splitkv_attn_sk_kernel, module muse_full_decode_splitkv_v2), whatever its score. Gate unchanged (_splitkv_gate: full, q_len 1, 1 < batch <= 64, tp8, head_dim 64, page 1). One change: remove the separate _pre_attention_kernel launch on the gated path so full decode_16/32 run ONE kernel (REWRITE_READ_2026-10-25 sections 1 and 5 list fused pre-attention as the idea worth keeping on this kernel). New kernel splitkv_attn_sk_fused_kernel in a NEW module muse_full_decode_splitkv_v3 (v2 stays compiled as the fallback): takes the raw qkv pointer/stride, positions, cache_loc, inv_freq, k_scale, v_scale instead of the precomputed FP8 q. (a) Q: when a CTA starts a request segment it builds that request q block in smem from qkv: the exact math of master _pre_attention_kernel for the 16 q heads (_norm_rope: scaleless RMSNorm over 64, eps 1e-5, RoPE with the same inv_freq table at positions[b], then the same FP8 cast and scale master writes), one warp per 2 heads, fp32 math; the redundancy (each request is touched by ~2-17 CTAs) is 2 KB of qkv per segment. (b) New K/V row: the CTA whose range contains request b last tile (key seq_b - 1, which is the token being decoded, stored at cache_loc[b]) computes the 2 kv heads new k (norm+rope+fp8) and v (fp8) exactly as master, stores them to k_cache/v_cache at cache_loc[b] (the cache update stays an output), then __syncthreads() before issuing that tile cp.async so the ring reads the stored row (same CTA, L2-coherent .cg loads). No other CTA reads key seq_b - 1, so no cross-CTA ordering is needed; the self-check asserts that ownership invariant. Launches on full decode_16/32 go from two (pre-attn, A) to one. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check with torch.cuda.synchronize() and a print after each kernel; never a child process). The self-check additionally compares the stored cache rows of v3 against master _pre_attention_kernel by rel L2 (not byte equality: LESSONS 2026-10-25, Triton sin/cos approximations) and q-dependent outputs by rel L2 < 5e-3. Fallback on any failure: v2 path (pre-attn + v2 kernel), printing PATH CHOSEN: full decode streamk v2 (fallback: <reason>); success prints PATH CHOSEN: full decode streamk fused-preattn v3. Every other shape byte for byte. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=full load=decode_16*` faster: replaced: _pre_attention_kernel 2.1 us + its launch gap on top of the stage-2 kernel; M1 16/16 tests and <= stage-2 shape time; M2 shape -2..-3 us vs stage 2 (floor 63.2)
- `*layer=full load=decode_32*` faster: replaced: _pre_attention_kernel 2.0 us + launch gap; M2 shape -2..-3 us vs stage 2 (floor 126.4)
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside the gate, master code byte for byte

## Risk

Small absolute win (~2-3 us on 120-250 us shapes) but it completes the one-launch full-decode path; risk is the new-key ownership and Q recomputation correctness, covered by the self-check. If stage 2 fails its gate, this branch still builds on its code and the v1 fallback.

## Milestones

Rewrite stage 3 of `_pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
