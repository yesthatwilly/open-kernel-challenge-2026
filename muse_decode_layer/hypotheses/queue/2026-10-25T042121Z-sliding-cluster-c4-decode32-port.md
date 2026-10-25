---
problem: muse_decode_layer
slug: sliding-cluster-c4-decode32-port
created: 2026-10-25T042121Z
base: master
area: FA3 call (sliding decode, batch 17..32)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's sliding_cluster_kernel
  (fp8qk TU, module muse_sliding_cluster_v2) from branch
  hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc, instantiating ONLY cluster
  size C=4 and gating it to sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config only), i.e.
  sliding decode_32. C=8 lost at B=16 in attention and C=16 was kernel-flat at B=1, so neither is
  built; batch 1 keeps decode_layer's _sliding_fused_kernel and 2..16 keeps master. The kernel
  takes the qkv GEMM output, does q/k RMSNorm + RoPE + FP8 (FA3-equivalent), stores the new k/v
  rows into the FP8 cache, gathers the window per rank, merges the 4 ranks over DSMEM and applies
  the RMSNorm x sigmoid(gate) epilogue, writing the bf16 tensor decode_layer's o GEMM reads; it
  replaces decode_layer's sliding chain between the qkv GEMM and the o GEMM at that band. Map
  decode_layer's qkv column layout, norm weights, rope base and window to the kernel arguments;
  the window covers every key (no skipping). At import call cudaOccupancyMaxActiveClusters for C=4
  and disable the path if < 32. Self-check at B in {17, 32} (lens < window and > window, kv_scale
  1 and 0.25): output vs master's chain, cache rows equal, graph replay. BUILD/SAFETY (LESSONS
  2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags
  are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first
  launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok'
  print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules
  reject starting processes). Any failure keeps the incumbent for the gated shapes and prints
  'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event
  timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower
  (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or
  allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement
  is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "c243a29f: FA3 23.7 + combine 2.8 + prepare 2.2 + pre-attention/cast/store/norm-gate kernels (hidden, ~6-8 us) ~35 us; attention ran the same fused chain at 18.3 us (Triton 22.1). Floor ~3.3. Shape 242.3 -> ~226 (-6%), weight 46. Kill: kernel row > 22 us or shape not < 236"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: outside the band
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: outside the band
  - shapes: "*layer=full*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32
review: proceed
status: queued
review_at: 2026-10-25T042453Z
review_companions: null
---

# sliding-cluster-c4-decode32-port

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's sliding_cluster_kernel (fp8qk TU, module muse_sliding_cluster_v2) from branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc, instantiating ONLY cluster size C=4 and gating it to sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config only), i.e. sliding decode_32. C=8 lost at B=16 in attention and C=16 was kernel-flat at B=1, so neither is built; batch 1 keeps decode_layer's _sliding_fused_kernel and 2..16 keeps master. The kernel takes the qkv GEMM output, does q/k RMSNorm + RoPE + FP8 (FA3-equivalent), stores the new k/v rows into the FP8 cache, gathers the window per rank, merges the 4 ranks over DSMEM and applies the RMSNorm x sigmoid(gate) epilogue, writing the bf16 tensor decode_layer's o GEMM reads; it replaces decode_layer's sliding chain between the qkv GEMM and the o GEMM at that band. Map decode_layer's qkv column layout, norm weights, rope base and window to the kernel arguments; the window covers every key (no skipping). At import call cudaOccupancyMaxActiveClusters for C=4 and disable the path if < 32. Self-check at B in {17, 32} (lens < window and > window, kv_scale 1 and 0.25): output vs master's chain, cache rows equal, graph replay. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_32*` faster: c243a29f: FA3 23.7 + combine 2.8 + prepare 2.2 + pre-attention/cast/store/norm-gate kernels (hidden, ~6-8 us) ~35 us; attention ran the same fused chain at 18.3 us (Triton 22.1). Floor ~3.3. Shape 242.3 -> ~226 (-6%), weight 46. Kill: kernel row > 22 us or shape not < 236
- `*layer=sliding load=decode_1 *` flat: outside the band
- `*layer=sliding load=decode_16*` flat: outside the band
- `*layer=full*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

124 regs + 32 B spill in attention; 103 KB smem/CTA. Check the spill line before trusting.

## Milestones

Rewrite stage 1 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042453Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. This is a single-shape, config-gated rewrite stage whose kernel row (sliding_cluster_kernel) and decode_32 shape time can be read directly, with eleven untouched shapes in the same run as controls. The real risks are that the in-layer saving comes in well under the ~16 us that summing the profile's kernel times predicts, and that a warm-L2 prepare selector picks the wrong path.

### Justification
Plausible on H200. Master's sliding batch>1 chain is still SGLang's: FusedQKNorm, the k/v/q FP8 casts, store_cache, FA3 prepare_varlen_num_blocks 2.2 + main 23.7 + FlashAttnFwdCombine 2.8, then FusedRMSNormGate. Profile c243a29f, sliding decode_32 = 242.3 us against a floor of 84.5. The donor kernel measured 18.3 us at B=32 in muse_attention_decode (job 1990ce53e9d34073: shape 25.84 -> 20.15 us; 124 regs, 32 B spill, 103,424 B smem, MaxActiveClusters C=4 = 62, so the >= 32 gate passes). Geometry matches decode_layer (hq 16, hkv 2, d 64, qkv [T, 2304], page 1, window 2048). The KV window at B=32 is ~16.8 MB, so the floor is ~3.5 us and the kernel sits ~5x above it. Stage-1 milestone 3 (within 1.5x of the floor) is not reachable this stage, which is acceptable for a rewrite. Only sliding decode_32 is gated, so the expected flat directions on every other shape are right. WEAKEST LINK: the ~35 us incumbent figure is a sum of profile rows plus a guessed 6-8 us of 'hidden' glue, and the layer has already shown that this arithmetic overstates the realizable saving. The harvested sliding-fused-batched-decode-port replaced the same chain at decode_32 with a 22.1 us Triton kernel. That predicted about -13 us, yet the shape moved only 246.99 -> 245.52 us (delta 1.03 by paired ratio). If the same discount applies, the cluster kernel's saving is closer to -5..-10 us than -16, and the kill line 'shape not < 236' could fire even though the kernel row is below 22 us. Judge stage 1 on the kernel row against the summed incumbent rows (FA3 + combine + prepare + the casts/store/qknorm/normgate rows in the same profile), not on the 236 line alone.

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl (just landed): disjoint gate (window == 0 only), so there is no timing interaction. It does add a third load_inline module (dense chain 50 s + splitkv 49 s + cluster ~41 s = ~140 s of compile inside the 15-minute run budget), which is a timeout risk, not a confounder.
- small-t-launch-minimal-dense-chain-pdl: its PDL kernels (P, R) sit after the o GEMM and attn_boundary, and the attention-side chain is plain stream order, so the cluster kernel joins no PDL edge. Note that the attention-side PDL attempt (dense-chain-stage2-attn-side-pdl) lost on sliding decode_16/32, which shows this region is sensitive to launch-order changes. A small loss could come from the cluster launch's own setup cost, not from attention.
- sglang-pipeline-parity-router-hilo-m8: defines the incumbent (FusedQKNorm, FusedRMSNormGate, FA3), and the self-check compares against it. That is not a confound, but the reference chain in the self-check must be master's exact kv_scale division and q FP8 cast.
- stack-align-par-sidestream-sliding1 / fa3-hdim128-head-pairing-batched / post-expert-norm-triton-small-t / marlin-block-m-cap-16: gated to batch 1, full layers, the post-expert stage or T>32 respectively. None share the replaced kernels or memory traffic at sliding decode_32, so none confound.

### Interference and A/B needs
No companion is warranted. The gate selects exactly one scored shape (sliding decode_32), and no landed hypothesis touches the kernels it replaces, so reverting one (ab-revert) would measure the same thing. The other eleven shapes are byte-identical in-run controls. The profile names sliding_cluster_kernel as its own row, so a gain, a loss or a fallback can each be read from one run: the kernel row against the summed incumbent rows, plus the PATH CHOSEN and selector-timing lines. A split control is unnecessary because the kernel is one indivisible launch. The only thing that can make a single result ambiguous is the prepare-time selector silently keeping the incumbent, and the log disambiguates that, so it needs a logging requirement, not a second run. The parallel attention re-measure (stack-streamk-ticket-cluster-c16-c4, job 195dcc1209b6457b) does not change the decision. decode_layer's incumbent at B=32 is FA3, not attention's Triton or cluster, so C=4 needs only to beat ~29+ us of FA3-plus-glue, not attention's newer master.

### Open questions
- Prepare-time selector and L2: the B=32 KV window (~16.8 MB) fits in the H200's L2, while the judge cold-flushes L2 before every replay. A back-to-back CUDA-event comparison in prepare therefore measures both paths with warm KV and can mis-rank them. Flush L2 between selector reps (write a buffer larger than L2) or time each path as a graph after a flush, and print both numbers so the log shows whether the selector kept the incumbent.
- The selector must time the incumbent as the WHOLE replaced chain (qk_norm through o_norm_gate, including the casts and store_cache) against the single cluster launch. Timing FA3 alone would bias it toward rejecting the new kernel.
- None of the 13 tests has sliding at 17 <= batch <= 32, so this path is checked only on the timed decode_32 calls. A wrong result surfaces as a failed run after timing begins, not in the tests. The B in {17, 32} self-check against master's chain (rel L2 and exact cache-row equality) is the only early guard, so keep it strict.
- RoPE mapping: decode_layer's sliding layers use rope_theta 500000 with interleaved RoPE via cos_sin positions. Confirm that the donor's _inv_freq/_norm_rope pairing convention matches FusedQKNorm here, as the batch-1 _sliding_fused_kernel already does in this file, and reuse that file's helpers rather than adding a second copy.
- Compile budget: three load_inline modules (~140 s) plus the self-checks. Consider whether the cluster TU can share the splitkv module's build to cut one nvcc invocation; at minimum, print compile seconds per module.
- If attention's job 195dcc1209b6457b re-measures C=4 at B=32 well above 18.3 us, re-read this run's kernel row with that in mind. It still does not block dispatch.

