---
problem: muse_decode_layer
slug: sliding-cluster-c4-decode32-s2-k-encode-match
created: 2026-10-25T054127Z
base: master
area: FA3 call (sliding decode, batch 17..32)
change: >
  Stage 1 (run 0fb2c945) was NEVER MEASURED: sliding_cluster_kernel C=4 built (124 regs, 32 B
  spill, 103 KB smem, 62 active clusters), ran without fault and its attention output passed
  (own-rounding rel L2 1.65e-3; vs fp32 6.8e-2, same as master's 6.65-7.3e-2), but the self-check
  failed at B=32 kv_scale=0.25 on the k CACHE bytes only: byte mismatch 4.37% with 'within one
  code False' (B=17 kv_scale 0.25 and B=32 kv_scale 1 passed at 4.0-4.5% mismatch, all within one
  code; v bytes 0 mismatch). The Triton _sliding_fused_kernel splits=16 candidate failed with the
  IDENTICAL k numbers, so the k FP8 encode shared by both differs from master's FusedQKNorm/rope +
  cast + store_cache sequence at some element(s), not the cluster logic. Lever, in order: (1) in
  the self-check, print the worst k mismatches (master byte, new byte, fp32 pre-quant value,
  |x/kv_scale|) so the cause is in the log; (2) make the kernel's k encode reproduce master's
  sequence exactly: round normed k to bf16 before RoPE and again after RoPE where master's
  FusedQKNorm/rope writes bf16, apply 1/kv_scale in the same order and precision master's cast
  uses, and use master's overflow semantics (torch float8_e4m3fn conversion is non-saturating; if
  master emits the NaN code where |k/kv_scale| > 448, emit the same; if master saturates, use
  SATFINITE) and its round-to-nearest-even; (3) require k bytes within one code of master on every
  case, with the overflow class matched exactly. Do NOT loosen the criterion otherwise. Rebase
  onto current master (w2-marlin-atomic-add-t1-port); keep C=4 only, gate sliding layer, q_len 1,
  page 1, 17 <= batch <= 32 (config); batch 1 keeps _sliding_fused_kernel and 2..16 / >32 keep
  master. Add self-check cases with synthetic k magnitudes that cross 448*kv_scale. Print the
  selector timing (sliding_cluster_kernel vs the master chain FusedQKNorm..FusedRMSNormGate at
  B=32) before choosing. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing,
  prints, try/except, allocation on input values, data-dependent control flow or per-call mutable
  state; gates read config only (layer, batch, q_len, window, page); nothing cached from inputs;
  precision per TASK.md; every key in the window attended; no child processes (build and
  self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so
  every buffer, device pointer and descriptor lives in the per-state dict built in prepare(),
  never in a module/static global. Self-check: capture once per state, then alternate graph
  replays of state 0 and state 1 with refilled inputs and compare each to master; any miss falls
  back to master for the gate and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only
  (TORCH_CUDA_ARCH_LIST=9.0a), print regs/smem/local/binaryVersion, first launch a canary +
  synchronize with launching/ok prints. The prepare-time selector prints CUDA-event timing new vs
  master chain at B=32 and keeps master if new is slower. First measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "master chain at B=32 (c243a29f profile): FA3 23.7 + combine 2.8 + prepare 2.2 + norm/rope/cast/store/norm-gate ~6-8 us = ~35 us; attention problem ran the same C=4 chain at 18.3 us. Floor ~3.3. Shape 240.7 -> ~226 (weight 46). M1 self-check passes and PATH CHOSEN = cluster; M2 kernel row <= 22 us; kill: shape not below 236"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: outside the band (drift control)
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
depends_on: sliding-cluster-c4-decode32-port
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32
review: proceed
status: running
review_at: 2026-10-25T054427Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054427Z
base_sha: a3136ce5805b8fcc8f1c58fdd3500b1d54aeb06e
base_blob: e7b58542dc954a2a306648904dbd486bda0c417b
branch: hyp/muse_decode_layer/sliding-cluster-c4-decode32-s2-k-encode-match
worktree: .worktrees/muse_decode_layer/sliding-cluster-c4-decode32-s2-k-encode-match
mode: null
restack_of: null
stale_base: true
---

# sliding-cluster-c4-decode32-s2-k-encode-match

## Change

Stage 1 (run 0fb2c945) was NEVER MEASURED: sliding_cluster_kernel C=4 built (124 regs, 32 B spill, 103 KB smem, 62 active clusters), ran without fault and its attention output passed (own-rounding rel L2 1.65e-3; vs fp32 6.8e-2, same as master's 6.65-7.3e-2), but the self-check failed at B=32 kv_scale=0.25 on the k CACHE bytes only: byte mismatch 4.37% with 'within one code False' (B=17 kv_scale 0.25 and B=32 kv_scale 1 passed at 4.0-4.5% mismatch, all within one code; v bytes 0 mismatch). The Triton _sliding_fused_kernel splits=16 candidate failed with the IDENTICAL k numbers, so the k FP8 encode shared by both differs from master's FusedQKNorm/rope + cast + store_cache sequence at some element(s), not the cluster logic. Lever, in order: (1) in the self-check, print the worst k mismatches (master byte, new byte, fp32 pre-quant value, |x/kv_scale|) so the cause is in the log; (2) make the kernel's k encode reproduce master's sequence exactly: round normed k to bf16 before RoPE and again after RoPE where master's FusedQKNorm/rope writes bf16, apply 1/kv_scale in the same order and precision master's cast uses, and use master's overflow semantics (torch float8_e4m3fn conversion is non-saturating; if master emits the NaN code where |k/kv_scale| > 448, emit the same; if master saturates, use SATFINITE) and its round-to-nearest-even; (3) require k bytes within one code of master on every case, with the overflow class matched exactly. Do NOT loosen the criterion otherwise. Rebase onto current master (w2-marlin-atomic-add-t1-port); keep C=4 only, gate sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config); batch 1 keeps _sliding_fused_kernel and 2..16 / >32 keep master. Add self-check cases with synthetic k magnitudes that cross 448*kv_scale. Print the selector timing (sliding_cluster_kernel vs the master chain FusedQKNorm..FusedRMSNormGate at B=32) before choosing. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only (layer, batch, q_len, window, page); nothing cached from inputs; precision per TASK.md; every key in the window attended; no child processes (build and self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so every buffer, device pointer and descriptor lives in the per-state dict built in prepare(), never in a module/static global. Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with refilled inputs and compare each to master; any miss falls back to master for the gate and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a), print regs/smem/local/binaryVersion, first launch a canary + synchronize with launching/ok prints. The prepare-time selector prints CUDA-event timing new vs master chain at B=32 and keeps master if new is slower. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_32*` faster: master chain at B=32 (c243a29f profile): FA3 23.7 + combine 2.8 + prepare 2.2 + norm/rope/cast/store/norm-gate ~6-8 us = ~35 us; attention problem ran the same C=4 chain at 18.3 us. Floor ~3.3. Shape 240.7 -> ~226 (weight 46). M1 self-check passes and PATH CHOSEN = cluster; M2 kernel row <= 22 us; kill: shape not below 236
- `*layer=sliding load=decode_1 *` flat: outside the band (drift control)
- `*layer=sliding load=decode_16*` flat: outside the band
- `*layer=full*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

If the log shows the mismatch is master emitting NaN/0x7F on overflow and the cache test never hits it, matching it is still required: the self-check compares to master. If master's own encode is not reproducible bit-for-bit within one code, record the printed classes and stop (fallback), the stage is then a diagnosis for stage 3.

## Milestones

Rewrite stage 2 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054427Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. Stage 2 is a self-check fix that turns stage 1's unmeasured kernel into a measurable one at a single config-gated shape. Two details in the spec need sharpening before the build: the overflow threshold torch actually uses, and the extra bf16 rounding before RoPE.

### Justification
The diagnosis is plausible and points at overflow. Master's sequence is FusedQKNorm (bf16 out), then k / k_scale (bf16 / fp32 0-dim gives bf16), then .to(float8_e4m3fn) via c10's non-saturating conversion. At kv_scale 0.25 the divide is an exact x4, so the scale adds no double-rounding hazard beyond the kv_scale 1 case, and that case passed within one code. The one new failure class at B=32 / 0.25 is therefore almost certainly |4k| crossing the E4M3 overflow edge: torch writes NaN (0x7F|sign), while Triton's float8e4nv and the CUDA cvt both saturate to 0x7E. That also explains why the Triton splits=16 candidate failed with IDENTICAL numbers and why v (bytes 0) is clean. A second possible class is sign crossings near zero (0x00/0x80 vs 0x01/0x81), which are a large raw-byte distance but one code in value. Lever (1), printing the worst mismatches, settles which class it is, so it is the right first step. The mechanism for the speed claim is unchanged from stage 1: one cluster launch replaces FusedQKNorm, 3 casts, store_cache, prepare, FA3 23.7, combine 2.8 and norm-gate. It is still unmeasured in this layer. WEAKEST LINK: the realizable saving. The harvested sliding-fused-batched-decode-port replaced the same chain at decode_32 with a 22 us kernel and moved the shape only ~1.03x. The kill line (240.7 -> <236, -2%) is barely above the 1.5% noise floor, and the decode_32 load contributes <0.5% to the headline, so the headline will not read this. Judge on the sliding_cluster_kernel row against the summed incumbent rows (milestone 2), plus the PATH CHOSEN and selector-median lines. Two spec errors to fix in the build. (a) torch's c10 fp32->e4m3fn emits NaN only at |x| >= 480 (fp8_max = 1087<<20), not > 448; values in (448, 480) round to 0x7E. The hardware cvt (cvt.rn.satfinite.e4m3x2.f32, where satfinite is mandatory on sm_90) cannot produce NaN, so emulate it: saturating cvt, then OR in 0x7F with the sign where |x| >= 480 or x is NaN. The synthetic cases should straddle 448, 464 and 480. (b) 'Round normed k to bf16 before RoPE' should be done only if master's FusedQKNorm actually writes an intermediate there. It most likely does norm+rope in fp32 and writes bf16 once, and an extra pre-rope rounding would add mismatches. Let the lever-1 print decide. Note that master already ships the batch-1 _sliding_fused_kernel, which encodes k as fp32 -> satfinite E4M3 with no bf16 round and passes the judge. The judge's cache bound (8% rel L2 vs reference) is far looser than this self-check, and the strict self-check is a safety choice, not a rule.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: defines the incumbent chain (FusedQKNorm, cast, store_cache, FA3, FusedRMSNormGate) that the k-byte self-check is calibrated against. That is the target of the fix, not a timing confound.
- small-t-launch-minimal-dense-chain-pdl: its PDL kernels sit after the o GEMM and the attention-side chain is plain stream order, so the cluster launch joins no PDL edge. The attention-side PDL attempt (dense-chain-stage2-attn-side-pdl) lost on sliding decode_32, so a small loss could come from launch ordering rather than the kernel. Read the kernel row, not only the shape.
- full-decode-splitkv-restack-on-pdl: disjoint gate (window == 0), so no timing interaction. It does add a load_inline module to the 15-minute compile budget (3 modules, ~140 s, plus self-checks with more synthetic cases).
- w2-marlin-atomic-add-t1-port / stack-align-par-sidestream-sliding1 / fa3-hdim128-head-pairing-batched / post-expert-norm-triton-small-t / marlin-block-m-cap-16: gated to T=1, batch-1 sliding, full layers, the post-expert stage or block_m. None touches the replaced kernels or their memory traffic at sliding decode_32.

### Interference and A/B needs
No companion is warranted. The gate selects exactly one scored shape (sliding decode_32), and no landed hypothesis touches the kernels it replaces, so an ab-revert would measure the same thing, and the other eleven shapes are in-run controls. The kernel is a single launch with its own profile row, so a control split would isolate nothing. The only ambiguity a single run can carry is a silent fallback (self-check miss or the selector keeping master), and the PATH CHOSEN line with its reason, the printed worst-mismatch classes and the selector medians resolve it from the log. If the run falls back again, the stage is still a diagnosis that pins the encode class for stage 3, as the hypothesis's Risk says. A second check would spend one of decode's 3 remaining checks for no added interpretability.

### Open questions
- Is the failing class overflow (master 0x7F/0xFF NaN vs new 0x7E/0xFE) or sign-crossing near zero (0x80 vs 0x01)? Define 'within one code' on the sign-magnitude ordinal (or decoded value with -0 == +0), not raw byte difference. Otherwise a ±0 crossing is falsely flagged, and fixing it is a correction, not a loosening.
- Which overflow threshold does master hit in the self-check? Per c10 it should be |x| >= 480. Print it with the worst mismatches and confirm before hard-coding.
- Does FusedQKNorm round to bf16 between norm and rope? If not, do not add that rounding. Only the final bf16 write (and the bf16 result of k / k_scale) are master's roundings.
- Does stage 1's selector timing exist for B=32? It likely never ran because the self-check failed first. If it runs this time, make sure the L2 flush and whole-chain timing are kept, and print compile seconds per module to watch the 15-minute budget.

