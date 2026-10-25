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
review: required
status: queued
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
