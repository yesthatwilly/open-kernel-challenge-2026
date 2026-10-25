---
problem: muse_attention_decode
slug: sliding-verify-wgmma-window-s1
created: 2026-10-25T153027Z
base: master
area: sliding-layer attention (verify, wgmma stream-K windowed kernel)
change: >
  Fresh stage 1 on CURRENT master (no depends_on). This is the held
  full-verify-wgmma-s4-sliding-window idea with its hold reason fixed: it was held because the
  stage-3 kernel it builds on failed its self-check; full-verify-wgmma-s5-qfrag-reload (job
  71e40388a1904937, branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl
  69378d7) made that kernel correct (all 19 bisect cases, DIAG S correct, two-state replays
  5.3e-3). It is slow on FULL verify (0.57x of FA3 at B=16: ~3.8 us per tile per CTA, serial
  per-tile chain), but sliding verify is a different regime: B=16 has only 16 x 33 = 528 tiles, so
  each CTA owns 1-4 tiles and the shape is decided by launch count and one dependent round trip,
  not by per-tile throughput. Master pays 4 launches after pre-attention there (prepare 1.9 + FA3
  10.7 + combine 4.0 + norm-gate 1.4 us at verify_16; 1.8 + 15.0 + 4.2 + 1.6 at verify_32).
  CHANGE: port the s5 kernel splitkv_verify_wgmma_kernel<ticket> verbatim (math, ring, ticket
  combine + norm-gate, error word) into its own load_inline module muse_sliding_verify_wgmma_v1
  and add ONE int argument window (0 = full behaviour, must compile identically). With window > 0:
  per request seq_b = cache_seqlens[b], key range [lo_b, seq_b) with lo_b = max(0, seq_b - 8 -
  (window - 1)), tiles_b = ceil((seq_b - lo_b) / 64), tile k covers keys lo_b + 64k .. +63 (page
  1, no alignment needed); the tile prefix (tile_prefix) and the ticket's per-request tile count
  use tiles_b; page-index prefetch offsets by lo_b. Mask adds the lower edge: key t valid for
  block token j iff t <= seq_b - 8 + j AND t >= seq_b - 8 + j - (window - 1), applied on any tile
  overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it; every
  window key attended). Route: sliding verify (window != 0, q_len 8, page 1, hq 16, hkv 2,
  head_dim 64, rope on, 1 <= B <= 32; config-only) runs master _pre_attention_kernel unchanged
  (ROPE on, PAIRED=False: q FP8 [tokens, 16, 64], new rows stored) then this kernel; nothing else.
  Grid per band chosen in prepare from {2B, 4B, 8B, SM, 2xSM} (capped at 2xSM; config-only), 2
  CTAs/SM as s5. GATE per band (B=1, B=2..16 timed at 16, B=17..32 timed at 32) at batch x 80000
  agentic lengths: interleaved cold-L2 graph replay, empty graph subtracted, median of 31,
  [pre-attn + wgmma window] vs the band's master incumbent (B=1: pre-attn + PSEUDO sk route as
  kernel_fn takes it; B=16/32: pre-attn + FA3 path + rms_norm_gate); route only at >= 1.05x and
  output rel L2 < 3e-2 vs master; print every grid's time. SELF-CHECK vs an fp32 windowed causal
  reference: window 2048, lens < 8, < window, window + 7, window + 8, 80000, B = 1/2/5/16/32
  ragged, kv_scale 1 and 0.25, grids 3/7/device, coherent data, plus the two-state alternating
  graph replay; also run window=0 cases to prove the full instantiation is unchanged. Any miss:
  band stays on master with a PATH CHOSEN line. BUILD as s5 (TORCH_CUDA_ARCH_LIST=9.0a,
  __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, regs/local printed). Full verify, decode and page-64
  tests untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine).
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent shape 22.8 us; replaced kernels prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 = 18.0 us; floor 1.9 us. ~2 tiles per CTA at grid 264. M1 routed; M2 replaced kernels <= 12 us (shape <= 17); M3 <= 8 us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: incumbent shape 29.2 us; replaced prepare 1.8 + FA3 15.0 + combine 4.2 + norm-gate 1.6 = 22.6 us; floor 3.7 us. M2 replaced <= 15 us
  - shapes: "*layer=sliding load=verify_1 *"
    direction: flat
    why: incumbent PSEUDO sk route 10.0 us + pre-attn 1.7 (13.2 shape); route only if the gate shows >= 1.05x, else flat
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)"
review: required
status: queued
---

# sliding-verify-wgmma-window-s1

## Change

Fresh stage 1 on CURRENT master (no depends_on). This is the held full-verify-wgmma-s4-sliding-window idea with its hold reason fixed: it was held because the stage-3 kernel it builds on failed its self-check; full-verify-wgmma-s5-qfrag-reload (job 71e40388a1904937, branch hyp/muse_attention_decode/full-verify-wgmma-s5-qfrag-reload, impl 69378d7) made that kernel correct (all 19 bisect cases, DIAG S correct, two-state replays 5.3e-3). It is slow on FULL verify (0.57x of FA3 at B=16: ~3.8 us per tile per CTA, serial per-tile chain), but sliding verify is a different regime: B=16 has only 16 x 33 = 528 tiles, so each CTA owns 1-4 tiles and the shape is decided by launch count and one dependent round trip, not by per-tile throughput. Master pays 4 launches after pre-attention there (prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 us at verify_16; 1.8 + 15.0 + 4.2 + 1.6 at verify_32). CHANGE: port the s5 kernel splitkv_verify_wgmma_kernel<ticket> verbatim (math, ring, ticket combine + norm-gate, error word) into its own load_inline module muse_sliding_verify_wgmma_v1 and add ONE int argument window (0 = full behaviour, must compile identically). With window > 0: per request seq_b = cache_seqlens[b], key range [lo_b, seq_b) with lo_b = max(0, seq_b - 8 - (window - 1)), tiles_b = ceil((seq_b - lo_b) / 64), tile k covers keys lo_b + 64k .. +63 (page 1, no alignment needed); the tile prefix (tile_prefix) and the ticket's per-request tile count use tiles_b; page-index prefetch offsets by lo_b. Mask adds the lower edge: key t valid for block token j iff t <= seq_b - 8 + j AND t >= seq_b - 8 + j - (window - 1), applied on any tile overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it; every window key attended). Route: sliding verify (window != 0, q_len 8, page 1, hq 16, hkv 2, head_dim 64, rope on, 1 <= B <= 32; config-only) runs master _pre_attention_kernel unchanged (ROPE on, PAIRED=False: q FP8 [tokens, 16, 64], new rows stored) then this kernel; nothing else. Grid per band chosen in prepare from {2B, 4B, 8B, SM, 2xSM} (capped at 2xSM; config-only), 2 CTAs/SM as s5. GATE per band (B=1, B=2..16 timed at 16, B=17..32 timed at 32) at batch x 80000 agentic lengths: interleaved cold-L2 graph replay, empty graph subtracted, median of 31, [pre-attn + wgmma window] vs the band's master incumbent (B=1: pre-attn + PSEUDO sk route as kernel_fn takes it; B=16/32: pre-attn + FA3 path + rms_norm_gate); route only at >= 1.05x and output rel L2 < 3e-2 vs master; print every grid's time. SELF-CHECK vs an fp32 windowed causal reference: window 2048, lens < 8, < window, window + 7, window + 8, 80000, B = 1/2/5/16/32 ragged, kv_scale 1 and 0.25, grids 3/7/device, coherent data, plus the two-state alternating graph replay; also run window=0 cases to prove the full instantiation is unchanged. Any miss: band stays on master with a PATH CHOSEN line. BUILD as s5 (TORCH_CUDA_ARCH_LIST=9.0a, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, regs/local printed). Full verify, decode and page-64 tests untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=sliding load=verify_16 *` faster: incumbent shape 22.8 us; replaced kernels prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 = 18.0 us; floor 1.9 us. ~2 tiles per CTA at grid 264. M1 routed; M2 replaced kernels <= 12 us (shape <= 17); M3 <= 8 us
- `*layer=sliding load=verify_32 *` faster: incumbent shape 29.2 us; replaced prepare 1.8 + FA3 15.0 + combine 4.2 + norm-gate 1.6 = 22.6 us; floor 3.7 us. M2 replaced <= 15 us
- `*layer=sliding load=verify_1 *` flat: incumbent PSEUDO sk route 10.0 us + pre-attn 1.7 (13.2 shape); route only if the gate shows >= 1.05x, else flat
- `*layer=full *` flat: untouched
- `*load=decode_* *` flat: untouched

## Risk

Weight 46 per sliding verify shape: verify_16 load 3657 us, sliding share 1049; -6 us there is ~7.5% of the load. Risk: the s5 per-tile chain (~3.8 us/tile at full) may also hurt here when a CTA owns 3-4 tiles; the per-grid timing table shows that and the gate keeps master. Disjoint from full-verify-wgmma-ws-s1-producer-split (different module, different shapes); if that one lands, a stage 2 here adopts its pipeline. Supersedes held full-verify-wgmma-s4-sliding-window (built on the broken s3 branch) and is not the stage-2 split-KV A8 route of held sliding-verify-cuda-streamk-window (mma.sync A8 was 1.8x slower than FA3 on verify, LESSONS).

## Milestones

Rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8) at 1<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate (B=16/32), and the PSEUDO sk route (B=1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
