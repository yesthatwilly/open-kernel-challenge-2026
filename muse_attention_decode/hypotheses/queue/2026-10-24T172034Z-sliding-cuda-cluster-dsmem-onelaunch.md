---
problem: muse_attention_decode
slug: sliding-cuda-cluster-dsmem-onelaunch
created: 2026-10-24T172034Z
base: master
area: sliding-layer attention (whole-path rewrite, CUDA cluster)
change: >
  Stage 1 of a rewrite of the whole SLIDING-layer path, built on master (no depends_on). Why: the
  sliding layer runs 46x per step in every load, and it costs 13.9-29.3 us per shape against
  floors of 0.1-3.7 us. It is a launch and dependency-chain problem, not a bandwidth one. Batched
  shapes run 5 launches (_pre_attention_kernel, FA3 prepare_varlen_num_blocks, FA3 main,
  FlashAttnFwdCombine, sglang rms_norm_gate). Decode_1 runs the Triton _sliding_fused_kernel at
  10.3 us. That kernel puts one program per (request, kv head, split) and gathers a 64-byte half
  of each 128-byte cache row, so two programs touch every line. It also merges through global
  partials, gpu-scope fences and a ticket. The earlier Triton single launch
  (sliding-single-launch-fused-attention, 0.945x) lost 2-2.6x on verify with 64-row tiles. A 2.7x
  board leader needs every sliding shape at about 6-12 us. New design: ONE CUDA kernel,
  sliding_cluster_kernel<L>, with L in {1, 8} as a template. It goes in its OWN load_inline
  module, muse_sliding_cluster_v1, compiled at import beside the existing splitkv module, so a
  failed build cannot disable full decode. It is launched with cudaLaunchKernelEx and a
  cluster-dimension attribute, which graph capture supports. DECOMPOSITION: one thread-block
  cluster per request, C CTAs per cluster, so the grid is B*C. C comes from config only: B == 1
  gives 16 (set cudaFuncAttributeNonPortableClusterSizeAllowed; if cudaOccupancyMaxActiveClusters
  at import returns 0, use 8), and B >= 2 gives 8. Rank r owns old keys [lo + r*ck, lo +
  (r+1)*ck), where lo = max(0, seq_lens[b] - (window - 1)), count = seq_lens[b] - lo and ck =
  round_up(ceil(count / C), 16). That is about 257 tokens for C = 8 and 2055 keys. One CTA has 256
  threads (8 warps) and serves BOTH kv heads of its tokens, so each token is one full 128-byte K
  row and one 128-byte V row. PROLOGUE, all issued before any wait: cp.async 4-byte loads of the
  chunk's page-table entries; then the 16-byte cp.async gather of the WHOLE chunk into smem in one
  shot (at most 257 x 256 B = 66 KB, plus a 16 B pad per row against bank conflicts; lane -> token
  lane/8, chunk lane%8, so each warp instruction covers 4 full lines). Only the index load stands
  between kernel start and the KV round trip. While the gather flies, each CTA loads the request's
  q rows from qkv (L x 16 x 64 bf16), applies scaleless RMSNorm and interleaved RoPE (angle =
  float(pos) * inv_freq in fp32, then sincosf at full precision, never __sinf: angles reach 8e4
  rad), times qk_scale_factor, rounds to E4M3 (allowed, as FA3) and stores to smem as the MMA A
  operand. The LAST rank also builds the block's L new keys and values: rmsnorm + rope on k, then
  / k_scale and / v_scale, then E4M3. It stores them to k_cache/v_cache at cache_loc, as the
  incumbent does, keeps them in smem as extra key rows of its chunk, and never reads them back
  from the cache, so no other CTA races the store. COMPUTE: QK^T with mma.sync m16n8k32 e4m3 (A is
  the fp8 q fragment, B the raw K bytes). P@V uses m16n8k16 f16, with V widened once per CTA into
  an f16 smem copy. Online softmax runs in the exp2 domain (scale = sm_scale * qk_scale_factor *
  log2e * k_scale from the device pointer, -inf safe). The per-row mask is t <= p_j and t >= p_j -
  (window - 1), applied on the tiles that overlap either window edge (not just the first and
  last), with causal masking inside the new block. For L = 1, each kv head's 8 query rows fill an
  m16 tile (rows 8-15 zero); warp w takes kv head w & 1 and token slice w >> 1, and the CTA merges
  4 warps per head in smem. For L = 8, each kv head has 64 rows (4 m-tiles); warp w takes kv head
  w & 1 and m-tile w >> 1 over all the chunk's tokens, so no intra-CTA merge is needed. MERGE over
  the cluster's distributed shared memory, with no global workspace, atomics or fences: each CTA
  leaves (m, l, unnormalized acc) for its rows in its own smem, then barrier.cluster.arrive/wait.
  Then a reduce-scatter: rank r owns rows [r*R/C, (r+1)*R/C) of the R = 16*L rows. It reads all C
  partials of those rows with mapa + ld.shared::cluster, does the log-sum-exp merge, times
  v_scale, then the master epilogue: scaleless RMSNorm over 64 with eps inside the sqrt, times
  sigmoid(gate) read from qkv, stored bf16 into [B, L, 1024]. A second cluster barrier precedes
  exit, so peers' smem stays live. GATE in kernel_fn, config only: window != 0, rope_theta != 0,
  hq 16, hkv 2, head_dim 64, page_size 1, 1 <= batch <= 64, q_len in {1, 8}, and _SLIDE_OK[q_len].
  Otherwise the master path runs byte for byte (the Triton fused kernel at decode_1, FA3
  elsewhere; page-64 and dp tests stay on master). FALLBACK and SELF-CHECK at import, on synthetic
  data only and never in the timed path: separately for L = 1 and L = 8, with B in {1, 3, 5},
  seq_lens including 10, 2046, 2047, 2049 and 5000, keys planted at both window edges, and
  kv_scale 1 and 1/64. It compares against an FP32 windowed causal reference, rel L2 < 2e-2 and
  all finite, and also checks that the new cache rows equal what master _pre_attention_kernel
  writes on a cloned cache. A failure clears only _SLIDE_OK[L] and prints why. STAGES: (1) this
  one, all six sliding shapes. (2) Tune from the profile: C per batch (4 at B = 32 with 1 CTA per
  SM if the cluster barrier dominates), PDL / programmatic launch to overlap the prologue with the
  previous graph node, and splitting the gather into two halves to overlap compute. (3) Fold in
  whichever full-layer kernel is incumbent so both layer kinds share one module and helpers.
  MILESTONES readable from the profile and log: the kernel appears as sliding_cluster_kernel and
  is the ONLY kernel on every sliding shape; log lines report the build seconds, the chosen C per
  batch, and the self-check rel L2 for L = 1 and L = 8; kernel time per shape against the
  incumbent's kernel sum (decode_1 10.3, decode_16 ~17.9, decode_32 ~21.3, verify_1 ~15.5,
  verify_16 ~20.7, verify_32 ~26.5 us).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: incumbent 13.9 us shape (Triton _sliding_fused_kernel 10.3 us), floor 0.1 us; one cluster launch, three dependent memory round trips (index -> KV -> DSMEM), target kernel <= 4.5 us, shape ~7.5 us; the sliding layer is 59% of the decode_1 load, ~1.37x on it
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: incumbent 20.8 us (FA3 main 9.9 + combine 3.3 + prepare 1.8 + pre-attention + norm-gate), floor 1.6 us; 5 launches -> 1, 8.4 MB gather at ~2.5 TB/s ~3.4 us; target shape ~8.5 us, ~1.24x on the load
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: incumbent 25.2 us (FA3 main 13.2 + combine 3.2 + pre-attn 1.9 + rest), floor 3.3 us; 256 CTAs, 66 KB each, one wave; target shape ~9.5 us, ~1.16x on the load
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: incumbent 17.6 us (FA3 main 7.4 + combine 3.8 + prepare 1.8 + pre-attn + norm-gate), floor 0.1 us; target shape ~8.5 us, ~1.48x on the verify_1 load, whose sliding share is 63%
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: incumbent 22.8 us (FA3 main 10.7 + combine 4.1 + pre-attn 2.7 + rest), floor 1.9 us; 64 rows per kv head on fp8 mma, merge via DSMEM reduce-scatter; target ~10 us, ~1.19x on the load
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "incumbent 29.3 us (FA3 main 14.7 + combine 4.2 + pre-attn 4.2 + rest), floor 3.7 us; target ~12 us, ~1.15x on the load. All six together: ~1.26x on the headline (1.349x -> ~1.70x); decode alone ~1.12x"
  - shapes: "*layer=full*"
    direction: flat
    why: "control: full-layer path untouched (separate module, separate gate)"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "sliding layer at tp8/page 1, every batch, decode and verify: _pre_attention_kernel + FA3 prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate, and the Triton _sliding_fused_kernel at decode_1"
review: required
status: queued
---

# sliding-cuda-cluster-dsmem-onelaunch

## Change

Stage 1 of a rewrite of the whole SLIDING-layer path, built on master (no depends_on). Why: the sliding layer runs 46x per step in every load, and it costs 13.9-29.3 us per shape against floors of 0.1-3.7 us. It is a launch and dependency-chain problem, not a bandwidth one. Batched shapes run 5 launches (_pre_attention_kernel, FA3 prepare_varlen_num_blocks, FA3 main, FlashAttnFwdCombine, sglang rms_norm_gate). Decode_1 runs the Triton _sliding_fused_kernel at 10.3 us. That kernel puts one program per (request, kv head, split) and gathers a 64-byte half of each 128-byte cache row, so two programs touch every line. It also merges through global partials, gpu-scope fences and a ticket. The earlier Triton single launch (sliding-single-launch-fused-attention, 0.945x) lost 2-2.6x on verify with 64-row tiles. A 2.7x board leader needs every sliding shape at about 6-12 us. New design: ONE CUDA kernel, sliding_cluster_kernel<L>, with L in {1, 8} as a template. It goes in its OWN load_inline module, muse_sliding_cluster_v1, compiled at import beside the existing splitkv module, so a failed build cannot disable full decode. It is launched with cudaLaunchKernelEx and a cluster-dimension attribute, which graph capture supports. DECOMPOSITION: one thread-block cluster per request, C CTAs per cluster, so the grid is B*C. C comes from config only: B == 1 gives 16 (set cudaFuncAttributeNonPortableClusterSizeAllowed; if cudaOccupancyMaxActiveClusters at import returns 0, use 8), and B >= 2 gives 8. Rank r owns old keys [lo + r*ck, lo + (r+1)*ck), where lo = max(0, seq_lens[b] - (window - 1)), count = seq_lens[b] - lo and ck = round_up(ceil(count / C), 16). That is about 257 tokens for C = 8 and 2055 keys. One CTA has 256 threads (8 warps) and serves BOTH kv heads of its tokens, so each token is one full 128-byte K row and one 128-byte V row. PROLOGUE, all issued before any wait: cp.async 4-byte loads of the chunk's page-table entries; then the 16-byte cp.async gather of the WHOLE chunk into smem in one shot (at most 257 x 256 B = 66 KB, plus a 16 B pad per row against bank conflicts; lane -> token lane/8, chunk lane%8, so each warp instruction covers 4 full lines). Only the index load stands between kernel start and the KV round trip. While the gather flies, each CTA loads the request's q rows from qkv (L x 16 x 64 bf16), applies scaleless RMSNorm and interleaved RoPE (angle = float(pos) * inv_freq in fp32, then sincosf at full precision, never __sinf: angles reach 8e4 rad), times qk_scale_factor, rounds to E4M3 (allowed, as FA3) and stores to smem as the MMA A operand. The LAST rank also builds the block's L new keys and values: rmsnorm + rope on k, then / k_scale and / v_scale, then E4M3. It stores them to k_cache/v_cache at cache_loc, as the incumbent does, keeps them in smem as extra key rows of its chunk, and never reads them back from the cache, so no other CTA races the store. COMPUTE: QK^T with mma.sync m16n8k32 e4m3 (A is the fp8 q fragment, B the raw K bytes). P@V uses m16n8k16 f16, with V widened once per CTA into an f16 smem copy. Online softmax runs in the exp2 domain (scale = sm_scale * qk_scale_factor * log2e * k_scale from the device pointer, -inf safe). The per-row mask is t <= p_j and t >= p_j - (window - 1), applied on the tiles that overlap either window edge (not just the first and last), with causal masking inside the new block. For L = 1, each kv head's 8 query rows fill an m16 tile (rows 8-15 zero); warp w takes kv head w & 1 and token slice w >> 1, and the CTA merges 4 warps per head in smem. For L = 8, each kv head has 64 rows (4 m-tiles); warp w takes kv head w & 1 and m-tile w >> 1 over all the chunk's tokens, so no intra-CTA merge is needed. MERGE over the cluster's distributed shared memory, with no global workspace, atomics or fences: each CTA leaves (m, l, unnormalized acc) for its rows in its own smem, then barrier.cluster.arrive/wait. Then a reduce-scatter: rank r owns rows [r*R/C, (r+1)*R/C) of the R = 16*L rows. It reads all C partials of those rows with mapa + ld.shared::cluster, does the log-sum-exp merge, times v_scale, then the master epilogue: scaleless RMSNorm over 64 with eps inside the sqrt, times sigmoid(gate) read from qkv, stored bf16 into [B, L, 1024]. A second cluster barrier precedes exit, so peers' smem stays live. GATE in kernel_fn, config only: window != 0, rope_theta != 0, hq 16, hkv 2, head_dim 64, page_size 1, 1 <= batch <= 64, q_len in {1, 8}, and _SLIDE_OK[q_len]. Otherwise the master path runs byte for byte (the Triton fused kernel at decode_1, FA3 elsewhere; page-64 and dp tests stay on master). FALLBACK and SELF-CHECK at import, on synthetic data only and never in the timed path: separately for L = 1 and L = 8, with B in {1, 3, 5}, seq_lens including 10, 2046, 2047, 2049 and 5000, keys planted at both window edges, and kv_scale 1 and 1/64. It compares against an FP32 windowed causal reference, rel L2 < 2e-2 and all finite, and also checks that the new cache rows equal what master _pre_attention_kernel writes on a cloned cache. A failure clears only _SLIDE_OK[L] and prints why. STAGES: (1) this one, all six sliding shapes. (2) Tune from the profile: C per batch (4 at B = 32 with 1 CTA per SM if the cluster barrier dominates), PDL / programmatic launch to overlap the prologue with the previous graph node, and splitting the gather into two halves to overlap compute. (3) Fold in whichever full-layer kernel is incumbent so both layer kinds share one module and helpers. MILESTONES readable from the profile and log: the kernel appears as sliding_cluster_kernel and is the ONLY kernel on every sliding shape; log lines report the build seconds, the chosen C per batch, and the self-check rel L2 for L = 1 and L = 8; kernel time per shape against the incumbent's kernel sum (decode_1 10.3, decode_16 ~17.9, decode_32 ~21.3, verify_1 ~15.5, verify_16 ~20.7, verify_32 ~26.5 us).

## Expected

- `*layer=sliding load=decode_1 *` faster: incumbent 13.9 us shape (Triton _sliding_fused_kernel 10.3 us), floor 0.1 us; one cluster launch, three dependent memory round trips (index -> KV -> DSMEM), target kernel <= 4.5 us, shape ~7.5 us; the sliding layer is 59% of the decode_1 load, ~1.37x on it
- `*layer=sliding load=decode_16 *` faster: incumbent 20.8 us (FA3 main 9.9 + combine 3.3 + prepare 1.8 + pre-attention + norm-gate), floor 1.6 us; 5 launches -> 1, 8.4 MB gather at ~2.5 TB/s ~3.4 us; target shape ~8.5 us, ~1.24x on the load
- `*layer=sliding load=decode_32 *` faster: incumbent 25.2 us (FA3 main 13.2 + combine 3.2 + pre-attn 1.9 + rest), floor 3.3 us; 256 CTAs, 66 KB each, one wave; target shape ~9.5 us, ~1.16x on the load
- `*layer=sliding load=verify_1 *` faster: incumbent 17.6 us (FA3 main 7.4 + combine 3.8 + prepare 1.8 + pre-attn + norm-gate), floor 0.1 us; target shape ~8.5 us, ~1.48x on the verify_1 load, whose sliding share is 63%
- `*layer=sliding load=verify_16 *` faster: incumbent 22.8 us (FA3 main 10.7 + combine 4.1 + pre-attn 2.7 + rest), floor 1.9 us; 64 rows per kv head on fp8 mma, merge via DSMEM reduce-scatter; target ~10 us, ~1.19x on the load
- `*layer=sliding load=verify_32 *` faster: incumbent 29.3 us (FA3 main 14.7 + combine 4.2 + pre-attn 4.2 + rest), floor 3.7 us; target ~12 us, ~1.15x on the load. All six together: ~1.26x on the headline (1.349x -> ~1.70x); decode alone ~1.12x
- `*layer=full*` flat: control: full-layer path untouched (separate module, separate gate)

## Risk

Biggest risk: the reported shape time keeps a fixed ~3.5 us above the kernel sum. Master's decode_1 is one 10.3 us kernel but a 13.9 us shape, so even a perfect 4 us kernel may read ~7.5 us. The targets already assume this; if the residual is larger, every sliding gain shrinks by the same amount. Other risks: (a) Cluster launch: the non-portable C=16 may not be schedulable when co-resident with nothing, so check cudaOccupancyMaxActiveClusters at import and fall back to 8. cudaLaunchKernelEx under graph capture needs the current stream from at::cuda. (b) The m16n8k32 e4m3 mma.sync needs PTX ISA >= 8.4 on sm_90a; a build failure must leave master intact, hence the separate module. (c) Register pressure at L=8 (acc 32 + S + q fragments) with 256 threads: there is no __launch_bounds__ min-blocks constraint, but read the spills from -Xptxas -v. (d) Window-edge masking on interior tiles, and verify rows whose windows differ by up to 7 keys. These are the classic failure; the self-check plants keys at both edges. (e) RoPE precision: use sincosf on an fp32 angle, the same as the Triton kernel, or cache-row checks drift. Interaction: sliding-deep-splits-on-cuda-splitkv (running) and sliding-verify-cuda-streamk-window (queued) change the incumbent on batched decode and verify. Judge this rewrite against whatever master is at dispatch, using each shape's kernel sum.

## Milestones

Rewrite stage 1 of `sliding layer at tp8/page 1, every batch, decode and verify: _pre_attention_kernel + FA3 prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate, and the Triton _sliding_fused_kernel at decode_1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
