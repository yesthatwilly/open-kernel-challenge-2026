---
problem: muse_attention_decode
slug: sliding-cluster-s3-verify-attn-only-phase-timed
created: 2026-10-25T042334Z
base: master
area: sliding-layer attention (verify, cluster kernel)
change: >
  Stage 3 of the sliding cluster rewrite, built on the stage-2 branch
  hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc (sliding_cluster_kernel<L>,
  fp8qk TU), whatever its score; rebase onto current master first (master now carries the landed
  full-decode stream-K kernels; keep both TUs and both prepare blocks, each path with its own gate
  constant). REWRITE_READ section 3 said L=8 may return only with a per-phase timing build that
  names the 13 us fixed chain it paid over L=1. Change (verify only; decode bands stay exactly as
  stage 2): (1) make the L=8 instantiation attention-only: master _pre_attention_kernel runs first
  (ROPE on, PAIRED=False) and writes the FP8 q for the 8 tokens and the 8 new K/V cache rows; the
  kernel reads FP8 q directly (no in-kernel 64-row norm/RoPE/FP8 prologue, no sincosf, so the 32 B
  local stack should go) and reads the new keys from cache like every other key (no last-rank
  new-key build, no barrier.cluster before the gather). DSMEM reduce-scatter and the norm-gate
  epilogue stay. (2) Phase-timing build, prepare only: a debug instantiation records globaltimer
  per CTA after index load, gather landed, QK done, PV done, DSMEM reduce done and epilogue done
  into a device buffer; print the median per phase at B=1, 16, 32. (3) Hard gate: per band (B=1,
  2..16, 17..32) and C in {4, 8, 16} (co-residency permitting), time the cluster L=8 kernel
  against master FA3 sliding-verify path (prepare + main + combine + norm-gate; pre-attn excluded
  from both), cold L2, median of 10; print all; route a band to the fastest cluster C only if >=
  5% faster than FA3, else FA3. Self-check: the stage-1 L=8 cases (window edges, kv_scale 0.25,
  B=1/3/5/16/32) vs fp32 reference; failure disables only verify routing. BUILD/SAFETY as stage 2
  (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First
  measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: replaced ~14.3 us (FA3 7.6 + combine 3.9 + prepare 1.7 + norm-gate); stage 1 with the in-kernel prologue was 23.2. M1 tests pass and phase print names the chain; M2 kernel <= 11 us; M3 <= 8 us
  - shapes: "*layer=sliding load=verify_16* *layer=sliding load=verify_32*"
    direction: faster
    why: replaced ~17 / ~21 us; stage 1 was 36.6 / 55.5. Gate keeps FA3 (flat) where the cluster still loses
  - shapes: "*layer=sliding load=decode_*"
    direction: flat
    why: "stage-2 decode bands unchanged (read against the stage-2 run, not master: B=16 C=8 band is the known +5.7% loss)"
  - shapes: "*layer=full*"
    direction: flat
    why: master code
risk: high
depends_on: sliding-cluster-s2-decode-c4-pdl-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "sliding verify (q_len 8, page 1, B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate"
review: required
status: queued
---

# sliding-cluster-s3-verify-attn-only-phase-timed

## Change

Stage 3 of the sliding cluster rewrite, built on the stage-2 branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc (sliding_cluster_kernel<L>, fp8qk TU), whatever its score; rebase onto current master first (master now carries the landed full-decode stream-K kernels; keep both TUs and both prepare blocks, each path with its own gate constant). REWRITE_READ section 3 said L=8 may return only with a per-phase timing build that names the 13 us fixed chain it paid over L=1. Change (verify only; decode bands stay exactly as stage 2): (1) make the L=8 instantiation attention-only: master _pre_attention_kernel runs first (ROPE on, PAIRED=False) and writes the FP8 q for the 8 tokens and the 8 new K/V cache rows; the kernel reads FP8 q directly (no in-kernel 64-row norm/RoPE/FP8 prologue, no sincosf, so the 32 B local stack should go) and reads the new keys from cache like every other key (no last-rank new-key build, no barrier.cluster before the gather). DSMEM reduce-scatter and the norm-gate epilogue stay. (2) Phase-timing build, prepare only: a debug instantiation records globaltimer per CTA after index load, gather landed, QK done, PV done, DSMEM reduce done and epilogue done into a device buffer; print the median per phase at B=1, 16, 32. (3) Hard gate: per band (B=1, 2..16, 17..32) and C in {4, 8, 16} (co-residency permitting), time the cluster L=8 kernel against master FA3 sliding-verify path (prepare + main + combine + norm-gate; pre-attn excluded from both), cold L2, median of 10; print all; route a band to the fastest cluster C only if >= 5% faster than FA3, else FA3. Self-check: the stage-1 L=8 cases (window edges, kv_scale 0.25, B=1/3/5/16/32) vs fp32 reference; failure disables only verify routing. BUILD/SAFETY as stage 2 (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_1 *` faster: replaced ~14.3 us (FA3 7.6 + combine 3.9 + prepare 1.7 + norm-gate); stage 1 with the in-kernel prologue was 23.2. M1 tests pass and phase print names the chain; M2 kernel <= 11 us; M3 <= 8 us
- `*layer=sliding load=verify_16* *layer=sliding load=verify_32*` faster: replaced ~17 / ~21 us; stage 1 was 36.6 / 55.5. Gate keeps FA3 (flat) where the cluster still loses
- `*layer=sliding load=decode_*` flat: stage-2 decode bands unchanged (read against the stage-2 run, not master: B=16 C=8 band is the known +5.7% loss)
- `*layer=full*` flat: master code

## Risk

Kill: if the attention-only L=8 kernel is still >= 14 us at B=1 in the phase print, the cluster L=8 design is dead for verify. Confound: decode rows carry stage-2 C=8 loss; read only the verify rows.

## Milestones

Rewrite stage 3 of `sliding verify (q_len 8, page 1, B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
