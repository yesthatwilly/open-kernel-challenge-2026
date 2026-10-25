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
review: proceed
status: queued
review_at: 2026-10-25T042453Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T042453Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, already-proven kernel port that replaces exactly the two incumbent split-KV kernels (A 112.7 + B 8.4 us at decode_16, 248.5 + 8.6 at decode_32 in c243a29f) at an unchanged config gate. Judge it on the replaced-kernel rows: the headline gain (~0.5-1%) is below the CI width. Port the kernel from attention master c3c1b01 (s2 path = template <4,true,false>) rather than the s2 branch, so the sibling full-decode1-two-level-ticket-port becomes a disjoint batch==1 gate on the same module.

### Justification
A. Mechanism: the hypothesis's replaced work is kernel A (persistent 2 CTA/SM, strided 1024-token chunks) plus kernel B (separate LSE combine + norm-gate). Stream-K over 64-token tiles gives each CTA an equal, contiguous range, which removes the agentic-length tail and the strided page-table walk. The per-request ticket folds B into the last-arriving CTA and drops one graph node. On the same H200 judge, muse_attention_decode measured the same kernel as follows. decode_16: 119.3 us (A+B) -> 108.9 us (s2 e592b1f5) and 110.1 us (s3 2a324178). decode_32: 254.1 -> 226.8 and 225.6. Both runs passed 16/16 tests with no fault on sm_90a. Decode_layer's incumbent rows are nearly identical (A 112.7 vs 111.4; A 248.5 vs 245.5), so the 106-108 and 226-230 us targets transfer directly. The KV floor (~63/126 us) is far above L2, so the surrounding layer's L2 state should not change the kernel's bandwidth. At T=16/32 the dense chain is active and the side stream is unused, so the kernel runs alone on the GPU with the same occupancy assumptions as in attention. Expected shapes and directions are consistent. The gate is unchanged (full, q_len 1, page 1, 1<B<=64, tp8), so decode_1, sliding and verify_* stay flat. Weight 16 per full decode shape means only about -13 us at decode_16 (306.7 -> ~294, ~4%) and about -27 us at decode_32 (479.6 -> ~452, ~6%). That is roughly 0.5-1% on the headline, inside the +/-1.5% CI. Marlin rows on these shapes swing 10-30 us run to run (kv-bw-3p5: Marlin 95.9 -> 106.9 at decode_16, 111.2 -> 144.7 at decode_32), so even the per-shape totals can hide a 9-12 us kernel gain. Read splitkv_attn_sk_kernel's top_kernels row against 121.1/257.1 (kill >= 116/245, as written). WEAKEST LINK: the prepare-time selector. prepare(weights) has no config, so 'timing at the gated batch' must use synthetic B=16/32. With uniform lengths, stream-K's advantage (the ragged tail) is smallest and tickets vs kernel B is a ~8 us difference, so a noisy CUDA-event comparison could silently keep A/B. The run would then read as 'flat' while measuring the incumbent. Use agentic-style ragged synthetic lengths at 80000 ctx, choose per batch (16 and 32 separately) with a small margin, and print PATH CHOSEN per batch so the profile row can be read as which kernel actually ran. A secondary weak point: prepare cannot size the workspace 'from config', so use a fixed MAXB=64 x (grid + B) allocation (~1.3 MB, as in the s3 code's slots <= grid + B).

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl: the incumbent being replaced, not a confounder. The port must keep its q prep (q.view(tokens,16,64).to(FP8).contiguous()), read gate through qkv.stride(0) like splitkv_combine, and keep v_scale and RMS_EPS exactly as kernel B does. A mismatch there shows up as a test failure, not a timing confound.
- small-t-launch-minimal-dense-chain-pdl: active at T=16/32 (DL_MAX_TOKENS=32) but downstream of the o GEMM. The attention kernels are plain <<<>>> launches with no PDL edge, so the only interaction is the one fewer graph node (~1-3 us), which counts as part of this change. It also keeps the side stream off at these shapes, so no concurrent kernels skew stream-K's static partition.
- fa3-hdim128-head-pairing-batched: inactive on the gated shapes (paired requires 'not splitkv'), so no confound.
- stack-align-par-sidestream-sliding1: its side-stream dispatch runs only at T>32 (verify_16/32), outside the gate, so no concurrency with the attention kernel.
- Sibling full-decode1-two-level-ticket-port (queued alongside, not landed): same kernel family, disjoint gate (batch == 1 vs 1 < batch <= 64), so each run attributes cleanly. Attention master c3c1b01 holds both paths in one template: TWO=false is this port's B>1 single-level ticket at grid 2xSM; TWO=true is the sibling's B=1 two-level ticket at grid = SM count. In attention, adding TWO left B>1 flat (108.9/226.8 -> 110.1/225.6 us). Both ports built separately from master will conflict when restacked (same module and gate code), and the second lander must restack. If both take the kernel source from c3c1b01 under one module name, the second becomes a gate/grid edit only. The sibling's text 'batch>1 keeps master kernel A/B' goes stale if this one lands first.

### Interference and A/B needs
No companion is warranted. Kind is rewrite stage 3, judged on the replaced kernels' own rows (splitkv_attn_sk_kernel vs splitkv_attn_kernel + splitkv_combine_kernel), and those rows separate cleanly. The gate is config-only and unchanged, and nothing landed overlaps the gated shapes (pairing is off, PDL is not on the attention edge, the side stream is idle at T<=32). An ab-revert of the incumbent restack would just compare against paired FA3, which is already known. A control isolating stream-K from ticket-fold would only re-measure attention's s2 diagnostics (stream-K with and without ticket), so ask for those in-process prints instead of a separate check. The one ambiguity is which path the prepare selector actually picked. That is a logging requirement (per-batch PATH CHOSEN plus selector timings), not a companion run.

### Open questions
- Will the implementer port from attention master c3c1b01 (s2 kernel plus the unused TWO template path) or from the s2 branch f291b49? c3c1b01 is preferred so the batch-1 sibling reuses the same module.
- Selector inputs: which synthetic length distribution and batch values? Uniform 80000 understates stream-K's benefit. Does it choose per batch (16 vs 32) or once for all gated batches?
- Workspace sizing without config in prepare: fixed MAXB=64 x (2*SMs + 64) slots, or allocated per call as attention s2 did? The hypothesis says 'no allocation in kernel_fn', but master already allocates there.
- The decode_layer profile is dist=soft while attention is dist=coherent. That is the MoE routing distribution and does not affect attention, but confirm the agentic KV-length spread is the same generator in both problems.

