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
review: proceed
status: done
review_at: 2026-10-24T172529Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T172705Z
base_sha: 2a1c23034c065ac604a00e741363c6bf9812e9de
base_blob: 1c0f5fa1bfc4e602b62e5e9edd9df6d5652b297b
branch: hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch
worktree: .worktrees/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch
mode: submit
implemented_at: 2026-10-24T175426Z
impl_commit: 593859d3ce482d60c7bb0ec1fb4e4101e7b1ce31
submission_sha256: f5c7fe82954b307189ff8d38afc62a1b8fabe2a962aed8fbd05cdd5e78d23b56
norm_sha256: fde3b2ec6399e2de9350f962623f669f9851968aff48928a7b89a8bbe5f04be8
submission_blob: bd96c94a12cf87fdcdac119d1e0508f009c4120c
lint_warnings: 0
attempt_file: 2026-10-24T181732Z-submit-sliding-cuda-cluster-dsmem-onelaunch.md
prepared_at: 2026-10-24T181732Z
job: 6821e67ff3e749b7
dispatched_at: 2026-10-24T181804Z
outcome: regression
verdict_detail: regression
speedup: 1.22057141323443
speedup_ci: [1.2022628420359136, 1.2388799844329463]
harvested_at: 2026-10-24T184714Z
best_at_harvest: 1.3948738389144513
promotion: null
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

## Adversarial review

Reviewed 2026-10-24T172529Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is a sound rewrite: one cluster launch replaces 5 launches on batched shapes and the 10.3 us Triton kernel at decode_1, with no rules problems, and its 6-12 us targets look reachable against 0.1-3.7 us floors. Two spec defects need fixing before building. The softmax scale applies qk_scale_factor twice, which the self-check would catch, silently disabling the path and wasting the stage. The smem/occupancy budget is understated: the 'one wave at B=32' claim does not hold if the f16 V copy and the fp32 partials take their own smem.

### Justification
Mechanism: FINDINGS backs it. Every sliding shape is dominated by fixed-cost chains, not bytes. decode_16: FA3 main 9.9 + combine 3.3 + prepare 1.8 + pre-attn + norm-gate = 20.8 us against a 1.6 us floor. verify_1: 7.4 + 3.8 + 1.8 = 17.6 us against 0.1 us. decode_1: a single 10.3 us Triton kernel against 0.1 us, using a ticket/fence global merge and half-row (64 B) gathers. Collapsing to one kernel with the merge in DSMEM removes the combine, prepare and norm-gate launches plus the global round trip, so the 'expected faster' directions on all six sliding shapes agree with the weights (46 each) and floors in TASK.md. Full shapes flat is right, since the gate is config-only and the module is separate. Data volume is small: 2047 x 256 B = 524 KB per request, 16.8 MB at B=32, about 3.5 us at 4.8 TB/s. The decode_32/verify_32 targets of ~9.5-12 us are therefore plausible, decode_1 <= 4.5 us is plausible (16 CTAs x ~33 KB each, ~1-1.5 us cold gather), and the speedup arithmetic (sliding share 59-63% of the B=1 loads) holds. WEAKEST LINKS: (1) A correctness defect in the spec. q is multiplied by qk_scale_factor before E4M3 rounding, and then the score scale is again sm_scale * qk_scale_factor * log2e * k_scale. That is 5.473 applied twice. Master's Triton kernel uses q*q_scale with score_scale = sm_scale*LOG2E. The splitkv path folds 5.473 into the scale and does not scale q. Pick one convention. As written, the self-check fails, _SLIDE_OK clears, every sliding shape runs master, and stage 1 reads as flat with no information. (2) Smem and occupancy. The 66 KB figure counts only the fp8 K+V gather (256 B/token x ~256-264 tokens, plus 16 B pads, ~74 KB at L=8 with the last rank's extra rows). It leaves out three things: the f16 V copy (+~67 KB), the fp32 (m, l, acc) partials for R=16L rows (L=8: 128 x 64 x 4 = 32 KB) and the q operand. That is ~175-180 KB per CTA, so 1 CTA/SM. Clusters of 8 must also co-reside in one GPC (H200 GPCs hold ~16-18 SMs), so the B=32 grid of 32 clusters x 8 CTAs likely cannot be resident at once (roughly 14-16 clusters active). It would run 2+ waves and miss the ~9.5/12 us targets. Fix: alias the partials over the consumed K region, and convert V fp8->f16 into registers per fragment instead of a full f16 smem copy. That gets under ~113 KB for 2 CTAs/SM. Also log cudaOccupancyMaxActiveClusters for C=8 at the actual smem size. (3) Minor wording: page-table cp.async loads must complete (wait_group) before the dependent KV gather is issued. 'All issued before any wait' cannot literally hold, but the intended one-index-round-trip chain is right. (4) The m16n8k32 e4m3 mma.sync needs PTX ISA 8.4 / CUDA >= 12.4 for sm_90a. The separate module plus per-L fallback covers a build failure, but a fallback again reads as flat. The log must show the build result so a flat headline is not misread as 'design does not help'.

### Confounders on the current kernel
- None from the landed list (empty). Master's own sliding path is the incumbent: the Triton _sliding_fused_kernel at decode_1, FA3 elsewhere. The stage is judged per shape on kernel time against it, so no landed-change interaction applies.
- Not landed but relevant to judging: sliding-deep-splits-on-cuda-splitkv (running) and sliding-verify-cuda-streamk-window (queued) may move the incumbent on batched decode/verify sliding shapes before dispatch. Compare against the master profile at dispatch time, as the risk section says, not against the FINDINGS numbers quoted in the milestones.

### Interference and A/B needs
No companions. It is a kind: rewrite judged on its own kernel time per shape (profile top_kernels shows sliding_cluster_kernel as the only kernel against the incumbent sum), so attribution is direct and needs no A/B. Nothing has landed that could confound it. The one ambiguity a single run can produce is 'flat everywhere'. That would mean the self-check fell back (likely given the double-scale defect) or the module failed to build, not that the design lost. The required log lines (build seconds, chosen C, self-check rel L2 per L, and which path each sliding shape took) settle that without an extra run. Stage 2 is the follow-up for tuning C and smem.

### Open questions
- Which scale convention will be built: q*5.473 then E4M3 with scale = sm_scale*log2e*k_scale, or unscaled E4M3 q with scale = sm_scale*qk_scale_factor*log2e*k_scale? The spec currently does both.
- What is the real per-CTA smem at L=8 for the last rank (256 old + 8 new rows, f16 V copy, partials, q operand)? What does cudaOccupancyMaxActiveClusters return for C=8 at that size? Does B=32 fit in one wave?
- Is the toolkit on the judge >= CUDA 12.4 (PTX 8.4) so m16n8k32 e4m3 mma.sync compiles for sm_90a? If not, is there an f16/bf16 QK fallback in the same module, rather than falling back to master?
- At B=1, C=16 uses only 16 SMs, against the incumbent's 32 CTAs. Is the cold gather of ~33 KB per CTA latency-bound enough that C=16 with 2 CTAs/SM, or multi-stage cp.async, matters for the <= 4.5 us target?
- For the test cases with ctx < window (batch 1 ctx 10 is page 64 and stays on master, but the self-check's seq_len 10 hits the new kernel), most ranks have empty chunks. Confirm that the m = -inf partials get weight 0 in the DSMEM merge, as master's combine does.


## Implementation notes

Implemented 2026-10-24 in the worktree `muse_attention_decode/submission.py` only (uncommitted, +887/-5). It was built blind: nothing was compiled here. Every master path is unchanged. Only the sliding branch of `kernel_fn` and the import block were touched, plus new code.

- **New module `muse_sliding_cluster_v1`.** A separate `load_inline` with its own `_SLIDE_CPP`/`_SLIDE_CUDA`, built in `_slide_init()` after the splitkv build and self-check. The splitkv module is not touched.
  - Contents: `sliding_cluster_kernel<L>` (L in {1, 8}, 256 threads, `__launch_bounds__(256, 2)`), `slide_setup()` and `slide_attn()`.
  - Launch: `cudaLaunchKernelEx` with a `cudaLaunchAttributeClusterDimension` (C, 1, 1) attribute on the current torch stream. The grid is B*C.
- **Cluster size.** C comes from config only.
  - C is 16 at batch 1 when `cudaOccupancyMaxActiveClusters(C=16)` > 0, measured after setting `cudaFuncAttributeNonPortableClusterSizeAllowed`. Otherwise C is 8.
  - `slide_setup()` returns, for both L: occupancy for C=8 and C=16 at the real smem size, `numRegs`, `localSizeBytes` (spills) and smem bytes. All of these are printed.
  - If C=8 reports <= 0, that L is not used.
- **Scale (review fix 1).** qk_scale_factor is applied exactly once.
  - q is RMS-normalized, rotated (RoPE), multiplied by qk_scale_factor, then rounded to E4M3 with `cvt.rn.satfinite.e4m3x2.f32`.
  - The score scale is `sm_scale*log2e` (host) times `k_scale` (device pointer).
  - The self-check reference uses the same rounding order.
- **Smem (review fix 2).**
  - There is no f16 V copy. V is widened per fragment in registers, using master splitkv's byte_perm + cvt.
  - The fp32 partials, and the L=1 per-warp staging, alias the consumed K region after a `__syncthreads`. static_asserts guard the fit.
  - Per CTA: K 34,816 B + V 34,816 B (272 rows x 128 B, token-XOR swizzle, no pad), plus q (1 KB or 8 KB).
  - Total: 70,656 B at L=1 and 77,824 B at L=8. Both fit 2 CTAs/SM (the target was under 113 KB).
- **Prologue.**
  - Page indices: 8 per thread, loaded with `__ldg` into registers, all issued before use.
  - Gather: one 16-byte `cp.async.cg` pass over the chunk's old K and V rows. Thread -> token tid/8 + 32i, chunk tid%8, so each warp instruction covers 4 full lines.
  - Rows from the last key up to the end of the 16-token tile are zero-stored.
  - While the gather is in flight, every CTA builds the L*16 queries: one warp per head vector, lane = RoPE pair, `sincosf` on an fp32 angle, IEEE `1/sqrtf`.
  - The last rank also builds the L new keys and values. It stores them at `cache_loc` and in smem rows `n_old+i` (swizzled), and never reads them back from the cache.
- **Compute.**
  - QK^T: `mma.sync.m16n8k32.e4m3` on the raw bytes, 2 k-steps per 16-token tile. P@V: `m16n8k16` f16.
  - Softmax: online, exp2 domain, -inf safe.
  - Mask: per row, `e < ntok && p_row-(W-1) <= pos <= p_row`, applied on every tile.
  - L=1: warp w takes kv head w&1 and tiles k ≡ w>>1 (mod 4). The 4 slice warps are merged in smem.
  - L=8: warp w takes kv head w&1 and m-tile w>>1. Rows quad and quad+8 are heads 2mt and 2mt+1 of token quad, so they share one mask. No intra-CTA merge is needed.
- **DSMEM merge.**
  - `cg::this_cluster().sync()`, then rank r owns rows [r*R/C, (r+1)*R/C).
  - It reads the C partials with `cluster.map_shared_rank`, the cooperative-groups API, instead of hand-written mapa/ld.shared::cluster asm.
  - Partials with m=-inf get weight 0 (review open question 5).
  - Then: LSE merge, times v_scale, RMSNorm (eps inside the sqrt), times sigmoid(gate), bf16 store.
  - A second `cluster.sync()` runs before exit.
- **Gate (`_slide_gate`, config only).**
  - Conditions: `0 < window <= 2048`, rope_theta != 0, hq 16, hkv 2, head_dim 64, page 1, 1 <= batch <= 64, and q_len in {1, 8} with a passed self-check for that q_len.
  - Otherwise the master path runs as before: the Triton fused kernel at decode_1, FA3 elsewhere.
- **Fallback (review open question 3).** The CUDA source has `#define SLIDE_FP8QK 1/0`.
  - The fp8-QK build (`muse_sliding_cluster_v1`) is tried first.
  - For any q_len whose build or self-check failed, the f16-QK build (`muse_sliding_cluster_v1_f16qk`) is built and checked. It uses the m16n8k16 path with master's k-permutation.
  - Only if that also fails does the q_len drop to master.
- **Self-check (at import, synthetic data only).** Four cases per L:
  - Coverage: B in {1, 3, 5}; seq 10, 17, 2046, 2047, 2048, 2049, 3000 and 5000; kv_scale 1 and 1/64; aligned keys planted at p-W+1 and p-W.
  - Output: rel L2 < 2e-2 against an FP32 windowed causal reference, and finite.
  - New cache rows: compared against master `_pre_attention_kernel` run on cloned caches. The bar is a byte match >= 99.9% and rel < 1e-2, not exact equality, because the Triton kernel uses approximate sqrt and division.
  - CUDA graph: one capture plus replay must equal the eager output.
  - A failure clears only that L.
- **Log lines.**
  - Build: `[slide] load_inline (fp8qk) compiled in Xs` or `FAILED`.
  - Occupancy: `[slide] fp8qk L=..: cudaOccupancyMaxActiveClusters C=8 -> n, C=16 -> m (smem, regs, local/spill)`.
  - Self-check: `[slide] self-check ... rel L2 ..., cache k match ...` and `graph capture + replay matches`.
  - Decision: `[slide] PATH CHOSEN q_len=1/8: CLUSTER kernel (...)` or `MASTER`.
  - Per shape: one `[slide] path batch=.. q_len=..: ...` line per sliding config that `kernel_fn` sees (host print, once per config).
- **Offline verification.** A numpy emulator of the kernel's per-lane indexing is at `/home/user/.claude/jobs/7156a0da/tmp/slide/emu.py`.
  - It covers the gather swizzle, fragment loads, masks, softmax, the L=1 staging merge, the L=8 partial layout and the DSMEM reduce-scatter.
  - Its MMAs are written from the PTX m16n8k32 and m16n8k16 fragment tables, and its smem is NaN-poisoned.
  - Coverage: L in {1,8} x C in {8,16} x seq in {0,1,10,300,2046,2047,2049,5000}, with planted edge keys.
  - Result: both the fp8 and f16 paths match a float64 reference to ~1e-15. A mutation (window edge off by one) is caught.

Deviations from the spec:
1. Page indices go into registers via `__ldg` instead of 4-byte `cp.async` copies into smem. The dependency is the same single index round trip before the KV gather (review point 3), with less smem and no extra wait_group.
2. There is no 16 B row pad. Master splitkv's token-XOR swizzle avoids the bank conflicts instead, and saves about 4 KB per CTA.
3. The mask is applied on every tile, not only on tiles that overlap a window edge. This is correct everywhere and costs little ALU.
4. The gate adds `window <= 2048`. This bounds the per-CTA chunk to 256 old keys and keeps the smem plan valid.
5. DSMEM uses cooperative_groups `map_shared_rank` and `cluster.sync()` rather than raw `mapa` + `barrier.cluster` asm. These are the same instructions, with less hand-written PTX to get wrong blind.
6. An f16-QK build of the same module is the fallback before master (review open question 3).

Uncertainties:
- (a) Nothing has been compiled. The untested pieces are the e4m3 mma (PTX 8.4), the cooperative_groups cluster API and the `cudaLaunchKernelEx` template on the judge toolchain. The f16 build covers only the first.
- (b) An out-of-bounds bug would raise a sticky error during the import self-check and take the whole run down. Bounds were checked by hand and in the emulator, which asserts on every smem word read.
- (c) Two or three module builds add about 40-80 s to the ~41 s splitkv build.
- (d) C=16 at B=1 is used for decode_1 and verify_1 only if occupancy reports > 0.
- (e) sincosf on angles near 8e4 rad takes the slow (Payne-Hanek) path. It is correct but costs some ALU per head.

### Pre-flight fix (2026-10-24, after an Opus audit)
The new-cache-row check required >= 99.9% identical bytes against master's Triton `_pre_attention_kernel`. That kernel's sin/cos are likely the fast approximations, which at positions 2049-5007 (angles up to ~5000 rad) flip 1-6% of E4M3 bytes in emulation. A single flipped byte in a 128-byte case failed the gate, so the check would likely have switched a correct kernel off. Now: rel L2 < 5e-2 only, with the judge's CACHE_TOL at 8e-2; the byte match is still printed.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch diff`:

- [x] CUDA-graph capturable.
  - The new path in kernel_fn has no `.item()`, `.cpu()`, synchronize or tensor prints. It does `torch.empty` for the output and one `cudaLaunchKernelEx` on the current stream.
  - Branching is on config (`_slide_gate`, `_slide_clusters`) and import-time module state only. The `_slide_note` print is a host string, printed once per config.
  - The self-check also captures and replays once.
- [x] Nothing derived from `data` is cached across calls.
  - `_SLIDE_MOD`, `_SLIDE_C16` and `_SLIDE_TAG` hold import-time build and occupancy results. `_SLIDE_NOTED` holds config tuples only.
  - `_inv_freq` is master's, keyed on theta, and unchanged.
- [x] Weights: n/a, since this problem has no weights and no prepare().
- [x] Activations are BF16 where SGLang keeps BF16, the KV cache is FP8, and queries are at most FP8 E4M3.
  - qkv and the output stay bf16. New K/V are stored E4M3 at cache_loc.
  - The query is rounded to E4M3 (after qk_scale_factor), which is the precision FA3 uses.
  - P@V uses f16 P and V, finer than FA3, with fp32 softmax and accumulation.
- [x] Attention sees every key.
  - Each request's window [max(0, p-2047), p] is split across the C ranks with no gaps (C*ck >= count).
  - The per-row mask is exactly the window edge plus causal. Nothing is skipped by weight.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only import is `torch.utils.cpp_extension.load_inline`, which master already uses. There is no cuobjdump and no process start.
- [x] All work happens inside the timed call, with no output caching or replay. The import-time self-check uses synthetic tensors only, and kernel_fn recomputes everything per call.
- [x] The first-line marker `# okc: problem=muse_attention_decode` is kept, and no triton.autotune was added.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK (the `__pycache__` it left in the worktree was removed).

## Result

- job `6821e67ff3e749b7` (submit), status done, passed True, tests 16/16
- speedup 1.2206x, 95% CI [1.2023, 1.2389], score 3085.685900196175 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/sliding-cuda-cluster-dsmem-onelaunch.result.json`, `muse_attention_decode/hypotheses/done/regression/sliding-cuda-cluster-dsmem-onelaunch.patch`; branch `hyp/muse_attention_decode/sliding-cuda-cluster-dsmem-onelaunch`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.88 | 11.82 | 1.17 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.93 | 27.8 | 1.0117 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 20.84 | 15.31 | 1.3627 | gained |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 123.02 | 122.6 | 1.0037 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 25.2 | 25.84 | 0.9916 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 257.75 | 258.6 | 0.9984 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.63 | 28.45 | 0.6182 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.29 | 30.03 | 1.0056 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.77 | 41.52 | 0.5495 | lost |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.83 | 162.91 | 1.0018 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.35 | 65.57 | 0.4463 | lost |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 301.91 | 301.81 | 1.0007 | flat |

gained on sliding decode_1, decode_16; lost on sliding verify_1, verify_16, verify_32. Expected hit: 3/7. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `sliding layer at tp8/page 1, every batch, decode and verify: _pre_attention_kernel + FA3 prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate, and the Triton _sliding_fused_kernel at decode_1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on sliding-cuda-cluster-dsmem-onelaunch --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.

