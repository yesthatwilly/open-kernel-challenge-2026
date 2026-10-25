---
problem: muse_decode_layer
slug: full-decode1-two-level-ticket-port
created: 2026-10-25T042121Z
base: master
area: FA3 call (full decode, batch 1)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's
  full-decode1-streamk-s3-two-level-ticket (branch
  hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket; it contains the stage-2
  stream-K kernel) to decode_layer, gated to full layer, q_len 1, page 1, batch == 1 ONLY
  (config). batch > 1 keeps master's split-KV kernel A/B unchanged so this run attributes to full
  decode_1 alone. At batch 1: grid = SM count, 64-token tiles, two-level ticket (groups of G =
  ceil(sqrt(nseg)) CTAs merge into a group slot via gticket[g], the last group merges the group
  slots, applies v_scale + RMSNorm x sigmoid(gate) and stores bf16; both ticket levels reset to 0
  for replay). It replaces, at full decode_1, the FA3 flash_attn_with_kvcache call +
  FlashAttnFwdCombine + the 3 vectorized elementwise pad/unpad copies + FusedRMSNormGate; q comes
  unpaired from the same pre-attention path master's batch>1 split-KV route consumes (extend that
  route's q preparation to batch 1). Self-check at batch 1 with 80000, 4097, 63 and 1 keys,
  kv_scale 1 and 0.25, grids 3/7/SM count, two consecutive replays. BUILD/SAFETY (LESSONS
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
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "profile c243a29f: FA3 18.5 + combine 7.0 + elementwise 3.0 + norm-gate (hidden) ~30 us replaced; attention ran the same kernel at 18.6 us for the whole replaced chain (27.3 -> 22.6 shape). Floor 4.3. Shape 142.7 -> ~132. M1 tests pass; M2 replaced chain <= 20 us; kill: shape not faster than 140 (z>=2)"
  - shapes: "*layer=full load=decode_16*"
    direction: flat
    why: batch>1 keeps master kernel A/B (the two-level kernel was slower at B=16 in attention)
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: batch>1 unchanged
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: FA3 main + FlashAttnFwdCombine + pad/unpad elementwise + FusedRMSNormGate on full decode batch 1
review: required
status: queued
---

# full-decode1-two-level-ticket-port

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's full-decode1-streamk-s3-two-level-ticket (branch hyp/muse_attention_decode/full-decode1-streamk-s3-two-level-ticket; it contains the stage-2 stream-K kernel) to decode_layer, gated to full layer, q_len 1, page 1, batch == 1 ONLY (config). batch > 1 keeps master's split-KV kernel A/B unchanged so this run attributes to full decode_1 alone. At batch 1: grid = SM count, 64-token tiles, two-level ticket (groups of G = ceil(sqrt(nseg)) CTAs merge into a group slot via gticket[g], the last group merges the group slots, applies v_scale + RMSNorm x sigmoid(gate) and stores bf16; both ticket levels reset to 0 for replay). It replaces, at full decode_1, the FA3 flash_attn_with_kvcache call + FlashAttnFwdCombine + the 3 vectorized elementwise pad/unpad copies + FusedRMSNormGate; q comes unpaired from the same pre-attention path master's batch>1 split-KV route consumes (extend that route's q preparation to batch 1). Self-check at batch 1 with 80000, 4097, 63 and 1 keys, kv_scale 1 and 0.25, grids 3/7/SM count, two consecutive replays. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: profile c243a29f: FA3 18.5 + combine 7.0 + elementwise 3.0 + norm-gate (hidden) ~30 us replaced; attention ran the same kernel at 18.6 us for the whole replaced chain (27.3 -> 22.6 shape). Floor 4.3. Shape 142.7 -> ~132. M1 tests pass; M2 replaced chain <= 20 us; kill: shape not faster than 140 (z>=2)
- `*layer=full load=decode_16*` flat: batch>1 keeps master kernel A/B (the two-level kernel was slower at B=16 in attention)
- `*layer=full load=decode_32*` flat: batch>1 unchanged
- `*layer=sliding*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

Shares full decode_1 with running dense-chain-stage2-attn-side-pdl-t1-gate (that one replaces the Q/O GEMMs, this one the attention kernels): read the attention kernel rows, not the headline.

## Milestones

Rewrite stage 1 of `FA3 main + FlashAttnFwdCombine + pad/unpad elementwise + FusedRMSNormGate on full decode batch 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
