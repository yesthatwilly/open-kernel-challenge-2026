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
review: proceed
status: running
review_at: 2026-10-25T042454Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T042454Z
base_sha: 08b1841a86f49e34227afd254a2476f445746929
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/full-decode1-two-level-ticket-port
worktree: .worktrees/muse_decode_layer/full-decode1-two-level-ticket-port
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T042454Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-1 port of a kernel that already beat FA3+combine+norm-gate in muse_attention_decode (22.8 -> 18.6 us kernel time, shape 27.19 -> 22.6). Run it now on current master in parallel with dense-chain-stage2-attn-side-pdl-t1-gate (c6cb53d8); do not wait. Whichever lands second needs an explicit restack that composes the two, because t1-gate's _dl2_attention takes over full decode_1 before master's branch and would make this port dead code. The expected gain is overstated by about 3-5 us.

### Justification
Mechanism: at full decode_1, master runs unpaired FA3 with scheduler_metadata, then FlashAttnFwdCombine, then FusedRMSNormGate. Profile c243a29f (profile 1): FA3 main 18.5 us, combine 7.0 us, norm-gate in the '7 more' rows (~1-1.5 us). That is ~26.5-27 us of attention for 20.5 MB of FP8 KV, against a ~4.3 us floor. The source kernel (splitkv_attn_sk_kernel<4,true,true>, grid = SM count, two-level ticket, fused v_scale + norm-gate epilogue) measured 18.6 us in attention profile 2a324178. There it replaced FA3 14.7 + combine 7.0 + norm-gate 1.1. It passed 16/16 tests, including graph replay and the ticket reset. So M1 is likely, and M2 (<= 20 us replaced chain) is plausible. Shape weight: full decode_1 is weight 16. Directions for decode_16/32, sliding and verify are flat by the config gate (batch == 1, window == 0, q_len 1, page 1), which is consistent. WEAKEST LINK: the savings arithmetic. (1) The 'elementwise 3.0 us (3 calls)' row is not pad/unpad. Master pairs only at batch > 1, so batch 1 has no pad/unpad copies. Those 3 calls are the q/k/v FP8 casts before store_cache and FA3, and the new path still needs them (the split-KV route itself does q.to(FP8).contiguous()). So the replaced set is ~26.5-27 us, not ~30. (2) FA3 main runs 18.5 us inside decode_layer vs 14.7 us standalone, so the layer environment (L2 flushed by ~150 MB of weights, a different launch neighbour) costs ~25% here. The SK kernel will likely pay a similar tax, ~19-23 us, not 18.6. Realistic shape: 142.7 -> ~135-138 us (-5 to -8 us), not ~132. The kill line (not faster than 140 at z >= 2, err ~0.57 us) is still a fair test. Stage-1 caveat: 18.6 us for 20.5 MB is ~1.1 TB/s, about 4x off the floor, so M3 is far away. The attention review already named grid = SM count (1 CTA/SM and ~9.5 tiles each, prologue-dominated) as the open tuning knob.

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl (master b047510): it owns _splitkv_gate, the v1 module (kernels A/B) and the unpaired-q route at 1 < batch <= 64. The port extends the q preparation to batch 1 and adds a second CUDA module (the attention v3 TU). If the builder edits _splitkv_gate or the shared q route rather than adding a separate batch-1 predicate, decode_16/32 could move. Their expected-flat rows (306.7 / 479.6 us) are the guard. Any movement there is a build defect, not an effect of this hypothesis.
- small-t-launch-minimal-dense-chain-pdl (master): at T=1 the attention output goes to the nvjet o GEMM, then attn_boundary, then the PDL chain (P, R, Marlin). The SK kernel ends with a serialized last-group merge on one CTA (~1-2 us tail). Nothing PDL-launched follows it directly, so no overlap changes. Read the attention rows, not the headline, because P/Marlin rows at decode_1 vary by ~1 us between runs.
- fa3-hdim128-head-pairing-batched (master): pairing needs batch > 1, so batch 1 is already unpaired. The new branch must sit before the paired/FA3 else-branch and keep 'paired' false at batch 1. No confound if it does.
- sliding-decode1-fused / stack-align-par-sidestream-sliding1 (master): sliding and T>32 only. The sliding decode_1 row (126.2 us) is an in-run drift control for the target row.
- IN FLIGHT, not landed: dense-chain-stage2-attn-side-pdl-t1-gate (job c6cb53d8ebc2478a). At full T=1 only, it routes kernel_fn through _dl2_attention: Q = dl_qkv_kernel (qkv GEMV + QK-norm + FP8 q + cache store), then FA3, then O = dl_o_kernel, whose prologue does the RMSNorm x sigmoid(gate) and which fuses the o GEMM. Both hypotheses claim FusedRMSNormGate at full decode_1. t1-gate also removes the 3 FP8 casts. Because it is not on master, it does not confound this run's measurement. It does change landing: its predicate fires before master's attention branch, so after it lands a port built on master would be bypassed silently at full decode_1. The gains are sub-additive by ~1-1.5 us (both remove the norm-gate).

### Interference and A/B needs
No companion. This is kind: rewrite, stage 1. Its own replaced-kernel rows decide it: the splitkv_attn_sk_kernel row against FA3 main + FlashAttnFwdCombine + norm-gate in the full decode_1 profile. The same run has two internal controls: the sliding decode_1 row (drift) and full decode_16/32 (master's batch>1 split-KV code, which must be byte-identical). The two sub-decisions (grid SM vs 2xSM, two-level vs single-level ticket) are template or argument switches. The attention build already brackets them with in-process eager diagnostics, so carry the [sk3-diag] printout over. An okc A/B would spend a check to learn what a prepare-time print already shows. WAIT OR STACK vs dense-chain-stage2-attn-side-pdl-t1-gate: run now on master, do not wait. The two runs replace disjoint main kernels (this one FA3 + combine, t1-gate the qkv/o GEMMs and casts), so each attributes cleanly on its own base. Landing rule: whichever lands second must restack onto the new master by hand, and must not be landed from its own patch. If t1-gate lands first, compose at full T=1 as Q(dl_qkv) -> SK two-level (with its norm-gate epilogue) -> dl_o with prologue=False. dl_o_kernel already takes a 'prologue' flag. The alternative is an SK variant without the epilogue feeding dl_o with prologue on. Then re-measure with one check, because the norm-gate saving is counted twice. If this port lands first, t1-gate's _dl2_attention must call the SK path instead of FA3 at full decode_1. If t1-gate has already landed when this is claimed, build on that master with the composition above instead of on today's master. Also coordinate with the queued sibling full-decode-streamk-ticket-port. It ports the same v3 TU for batch > 1, so both should share one module name and one load_inline. Otherwise the second to land carries two copies of splitkv_attn_sk_kernel.

### Open questions
- Does the SK kernel pay the same ~25% in-layer tax as FA3 (14.7 standalone -> 18.5 in layer)? If it lands at ~23 us, M2 (<= 20 us) is missed, and stage 2 should be grid 2xSM or a PDL pre-wait prologue (prefetch q/meta and start the KV ring before griddepcontrol.wait) so the KV stream overlaps the qkv GEMM tail.
- Builder: put the batch-1 predicate as a separate config gate ahead of the paired/FA3 else-branch, and leave _splitkv_gate (1 < batch <= 64) untouched, so decode_16/32 are byte-identical.
- Builder: print PATH CHOSEN for full decode_1 and the [sk3-diag] grid/ticket variants at B=1 x 80000 next to master FA3 main + combine + norm-gate at the same input. If the selector's 'new vs incumbent' margin is under ~2 us, say so: the judge undercounts prepare timings by 20-25% (LESSONS).
- If dense-chain-stage2-attn-side-pdl-t1-gate gains and lands first, mint the composed restack (Q -> SK -> O prologue=False) as this hypothesis's stage 2 rather than landing this branch as is.
- fa3-full-single-split-probe and fa3-num-splits-full-no-metadata (queued, hold) tune FA3 at full decode. If this lands, they no longer apply at batch 1 and should be re-scoped to batch > 1 or retired.

