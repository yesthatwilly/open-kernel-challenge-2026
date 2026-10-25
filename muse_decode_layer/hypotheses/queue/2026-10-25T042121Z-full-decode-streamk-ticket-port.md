---
problem: muse_decode_layer
slug: full-decode-streamk-ticket-port
created: 2026-10-25T042121Z
base: master
area: "FA3 call (full decode, batch > 1): splitkv_attn_kernel + splitkv_combine_kernel"
change: >
  Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed
  restack). Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA
  split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's stream-K
  kernel from branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc
  (splitkv_attn_sk_kernel: stream-K over 64-token tiles of the concatenated page-1 KV of all
  requests, grid 2 x SM count, each CTA owns an equal contiguous tile range; per-request ticket:
  each CTA writes its fp32 partial (acc, m, l), __threadfence(), atomicAdd on ticket[b]; the last
  arrival LSE-merges, applies v_scale and the RMSNorm x sigmoid(gate) epilogue, stores bf16 and
  resets ticket[b] to 0 for graph replay). In decode_layer replace the two calls in
  _splitkv_attn/_splitkv_combine (kernel A + kernel B) with the one stream-K launch at the same
  gate (_splitkv_gate: full layer, q_len 1, page 1, 1 < batch <= 64); it consumes the same q, KV
  cache, page_table, cache_seqlens, v_scale and qkv gate columns master's combine reads, and
  writes the same bf16 out tensor the o GEMM reads. The ticket buffer is a module-level int32 zero
  buffer allocated at import; the partial workspace is sized from config (max batch x grid) in
  prepare. Keep master's kernel A/B compiled as the fallback. Self-check vs master's kernels and
  an fp32 reference at B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, plus
  two consecutive graph replays (ticket reset). BUILD/SAFETY (LESSONS 2026-10-25):
  TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden;
  print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny
  synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the
  kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting
  processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ...
  (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs
  incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z:
  never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn;
  gates read config (batch, q_len, window, page) only. First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced: splitkv_attn 112.7 + combine 8.4 = 121.1 us (profile c243a29f); attention measured -9.3% kernel A from stream-K and -3..5% from the in-kernel ticket combine -> ~106-108 us (KV floor ~63). Shape 306.7 -> ~294. Kill: replaced-kernel time >= 116 us at B=16 in the profile"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced 248.5 + 8.6 = 257.1 us; stream-K removes the agentic tail imbalance at B=32 -> ~226-230 (floor ~126). Shape 479.6 -> ~452. Kill: >= 245 us"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: outside the gate
  - shapes: "*layer=sliding*"
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
rewrite_stage: 3
replaces: splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64
review: required
status: queued
---

# full-decode-streamk-ticket-port

## Change

Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed restack). Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's stream-K kernel from branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles of the concatenated page-1 KV of all requests, grid 2 x SM count, each CTA owns an equal contiguous tile range; per-request ticket: each CTA writes its fp32 partial (acc, m, l), __threadfence(), atomicAdd on ticket[b]; the last arrival LSE-merges, applies v_scale and the RMSNorm x sigmoid(gate) epilogue, stores bf16 and resets ticket[b] to 0 for graph replay). In decode_layer replace the two calls in _splitkv_attn/_splitkv_combine (kernel A + kernel B) with the one stream-K launch at the same gate (_splitkv_gate: full layer, q_len 1, page 1, 1 < batch <= 64); it consumes the same q, KV cache, page_table, cache_seqlens, v_scale and qkv gate columns master's combine reads, and writes the same bf16 out tensor the o GEMM reads. The ticket buffer is a module-level int32 zero buffer allocated at import; the partial workspace is sized from config (max batch x grid) in prepare. Keep master's kernel A/B compiled as the fallback. Self-check vs master's kernels and an fp32 reference at B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, plus two consecutive graph replays (ticket reset). BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## Expected

- `*layer=full load=decode_16*` faster: replaced: splitkv_attn 112.7 + combine 8.4 = 121.1 us (profile c243a29f); attention measured -9.3% kernel A from stream-K and -3..5% from the in-kernel ticket combine -> ~106-108 us (KV floor ~63). Shape 306.7 -> ~294. Kill: replaced-kernel time >= 116 us at B=16 in the profile
- `*layer=full load=decode_32*` faster: replaced 248.5 + 8.6 = 257.1 us; stream-K removes the agentic tail imbalance at B=32 -> ~226-230 (floor ~126). Shape 479.6 -> ~452. Kill: >= 245 us
- `*load=decode_1 *` flat: outside the gate
- `*layer=sliding*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

Kernel proven on the judge in attention (no fault, sm_90a). Decode-layer risk is interface only (gate columns/out layout). Compare against master's split-KV rows (c243a29f).

## Milestones

Rewrite stage 3 of `splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
