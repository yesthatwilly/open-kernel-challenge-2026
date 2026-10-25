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
review: required
status: queued
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
