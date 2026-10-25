---
problem: muse_moe_w4a16
slug: cuda-combine-kernel-b-bs128-only
created: 2026-10-25T011401Z
base: master
area: post-expert norm+combine
change: >
  Port ONLY kernel B (the CUDA post-expert RMSNorm + weighted combine) from
  muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch, and use it only
  for 33 <= num_tokens <= 191 (verify_16 bs=128, the bs=33 and skewed-128 tests) under its own
  constants CUDA_COMBINE_MIN_TOKENS = 33 / CUDA_COMBINE_MAX_TOKENS = 191; at num_tokens >= 192
  (bs=256, 257) keep SGLang's fused_post_expert_norm_weighted_sum, because that run measured
  kernel B slower there (6.2 vs 6.0 us profile; 4.75 vs 3.72 us prepare timing) while faster at
  M=128 (3.7 vs 4.9 us profile; 3.08 vs 3.50 us prepare timing). No PDL (it follows Marlin w2,
  which cannot trigger). Align and everything else stay master. Build with load_inline in prepare
  (no sm_90a instructions); self-check against master's combine at T in {33, 57, 128, 191} with
  random/skewed/all-remote routings (rel-L2 <= 5e-3, finite), else master; print '[cuda-combine-b]
  PATH CHOSEN = CUDA|SGLANG for 33..191 (<reason>)'.
expected:
  - shapes: bs=128 *
    direction: faster
    why: combine row 4.9 -> ~3.7 us (about -1.2 us of 318 us); judge on the combine row
  - shapes: bs=256 *
    direction: flat
    why: "control: SGLang combine kept"
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: hold
status: done
review_at: 2026-10-25T011915Z
review_companions: null
hold_reason: "adversarial review: Hold. This re-measures a kernel row already known from three runs (kernel B 3.5-3.8 us vs SGLang 4.9-5.9 us at bs=128) for about -1.2 us on one of six loads, ro"
outcome: expired
verdict_detail: "retired: stale hold closed per the 2026-10-25 moe diagnosis"
expired_at: 2026-10-25T212024Z
---

# cuda-combine-kernel-b-bs128-only

## Change

Port ONLY kernel B (the CUDA post-expert RMSNorm + weighted combine) from muse_moe_w4a16/hypotheses/done/neutral/cuda-dispatch-combine-pdl-verify.patch, and use it only for 33 <= num_tokens <= 191 (verify_16 bs=128, the bs=33 and skewed-128 tests) under its own constants CUDA_COMBINE_MIN_TOKENS = 33 / CUDA_COMBINE_MAX_TOKENS = 191; at num_tokens >= 192 (bs=256, 257) keep SGLang's fused_post_expert_norm_weighted_sum, because that run measured kernel B slower there (6.2 vs 6.0 us profile; 4.75 vs 3.72 us prepare timing) while faster at M=128 (3.7 vs 4.9 us profile; 3.08 vs 3.50 us prepare timing). No PDL (it follows Marlin w2, which cannot trigger). Align and everything else stay master. Build with load_inline in prepare (no sm_90a instructions); self-check against master's combine at T in {33, 57, 128, 191} with random/skewed/all-remote routings (rel-L2 <= 5e-3, finite), else master; print '[cuda-combine-b] PATH CHOSEN = CUDA|SGLANG for 33..191 (<reason>)'.

## Expected

- `bs=128 *` faster: combine row 4.9 -> ~3.7 us (about -1.2 us of 318 us); judge on the combine row
- `bs=256 *` flat: control: SGLang combine kept
- `bs=1 *` flat: control
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Below headline resolution by design; route-each-shape-to-the-faster-engine (LESSONS prefill stage 4). Cheap, low-risk per-shape part to stack later with cuda-align-kernel-a-from17.

## Adversarial review

Reviewed 2026-10-25T011915Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. This re-measures a kernel row already known from three runs (kernel B 3.5-3.8 us vs SGLang 4.9-5.9 us at bs=128) for about -1.2 us on one of six loads, roughly 0.06% headline. The run cannot be read at the headline and adds nothing at the row level.

### Justification
The mechanism is plausible and already demonstrated. In cuda-dispatch-combine-pdl-verify and both stack-onewarp runs, kernel B (the cdc_pos... row) read 3.7/3.5/3.8 us at bs=128. SGLang's _post_expert_norm_weighted_sum read 4.9 (best), 5.9 (align-parallel-from-17) and 5.3 us on the same shape. The bs>=192 exclusion correctly follows the measured loss at bs=256 (6.0-6.2 vs 6.0-6.4 us profile, 4.75 vs 3.72 us prepare timing). The expected directions match TASK: one equal-weight load out of six moves by -1.2/318 us = -0.4%, which is about -0.06% geomean. verify_16's per-shape err_ns is 2.6-3.0 us, so even the per-shape median cannot resolve a 1.2 us change, and only the profile row can. That row is a single call, and the SGLang combine row alone swings by 1 us between runs of identical code. WEAKEST LINK: the brief itself says 'below headline resolution by design', and the only thing it would measure is a row with three replicates already. The second weakest link is stacking value: 'stack later with cuda-align-kernel-a-from17' is exactly what stack-onewarp-hoisted-dispatchcombine(-from17) already did (kernel A + kernel B at bs>=33, plus two small-M parts), and it came back 1.318/1.324 against a 1.3287 best. Sub-noise parts have not added up on moe today.

### Confounders on the current kernel
- post-expert-norm-triton-small-m: fixes SMALL_M_POST_NORM = 32. The new 33..191 gate is a separate constant, which is correct. Check that 32 stays on SGLang's combine (master behaviour) and that no gate is derived from SMALL_M_POST_NORM (LESSONS 2026-10-25 gate-constant rule).
- marlin-block-m-two-smaller-on-postnorm: changes Marlin w2's block_size_m at bs=128 and so the y write pattern before the combine. Kernel B's 3.5-3.8 us rows were all measured on this base, so it does not confound, but the cache state left by Marlin w2 is shared by both combine engines.

### Interference and A/B needs
No companion is warranted, because no run makes this interpretable at the headline and the row-level answer is already in hand. Reverting or retuning a landed hypothesis would not change that. If kernel B is wanted, it should ride along in a hypothesis whose own effect is large enough to be read. Any such run reads it from the existing rows.

### Open questions
- Is there any non-sub-noise moe hypothesis queued that touches bs=128 (for example a wgmma or verify Marlin rewrite) that kernel B could be folded into as a free part?
- Kernel B was measured with PDL on, as a gdc_wait consumer of Marlin w2. Does its 3.5-3.8 us row survive a plain launch? A prepare-time event timing would answer that without a check.

