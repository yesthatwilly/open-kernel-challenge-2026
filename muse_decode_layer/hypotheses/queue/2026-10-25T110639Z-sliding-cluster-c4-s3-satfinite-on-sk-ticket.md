---
problem: muse_decode_layer
slug: sliding-cluster-c4-s3-satfinite-on-sk-ticket
created: 2026-10-25T110639Z
base: master
area: FA3 call (sliding decode, batch 17..32)
change: >
  Stage 3 of the sliding-cluster line. Stage 2 (sliding-cluster-c4-decode32-s2-k-encode-match, job
  7c04214e90354dfa, 1.2803 check) was NEVER MEASURED either: it built (124 regs, 32 B spill, 103
  KB smem, 62 clusters) but its prepare() probe printed 'probe: torch's cast never produced a NaN
  code below 520' and then 'PATH CHOSEN: sliding decode batch 17..32: master chain (fallback:
  torch's E4M3 overflow behaviour is not an edge the kernel can emulate)'. The self-check,
  two-state check and selector never ran, so its 1.2803 is master plus a dead path. The probe
  result means torch's bf16->float8_e4m3fn cast on the judge image does NOT emit NaN in 448..518,
  i.e. it saturates, which is exactly what the kernel's cvt.rn.satfinite.e4m3x2.f32 already does
  with no emulation. BASE: build on branch
  hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64 (submission blob d67278e0, the same
  base the in-flight dense-chain-s3 uses), merging s2's branch
  hyp/muse_decode_layer/sliding-cluster-c4-decode32-s2-k-encode-match (blob f8b86798) with base
  master 62c5a941 by 3-way git merge-file. A trial merge gives exactly 3 conflicts, all
  mechanical: (1) imports: keep sk-ticket's 'import random'; (2) prepare(): keep both, in the
  order _splitkv_prepare(device); _slide_prepare(state, device); _sk_prepare(state, device,
  t_prepare0); (3) kernel_fn's attention else-branch: s2 moved master's inline chain into
  _master_attn(data, state, qkv) while sk-ticket edited that inline chain to route full decode to
  its stream-K kernels; resolve by applying sk-ticket's edits INSIDE _master_attn so kernel_fn
  keeps s2's three-way branch (batch-1 sliding -> _sliding_forward, _slide_gate -> _slide_launch,
  else _master_attn). sk gates on window == 0 and the cluster on window != 0, so the two paths are
  disjoint and the cluster selector still times the unchanged sliding master chain. THE FIX (only
  _slide_probe_nan_thr and its caller change): (a) always print the probed byte table first (x =
  440, 446, 448, 450, 456, 464, 466, 478, 480, 482, 512, 518 both signs, plus +inf, -inf, nan) as
  '[slide-c4] probe bytes: ...'; (b) classify: EDGE (RNE below some E, sign|0x7F at and above it)
  -> nan_thr = E as in s2; SATURATING (every finite |x| >= 448 gives sign|0x7E, no 0x7F anywhere
  for finite input) -> nan_thr = +inf, so the kernel's satfinite cvt is used alone; anything else
  -> print the class and fall back as now. Print '[slide-c4] probe class: EDGE|SATURATING|OTHER
  nan_thr=...'. (c) The synthetic overflow self-check case stays, but its expectation is master's
  own bytes (0x7E in the SATURATING case); keep s2's sign-magnitude ordinal within-one-code rule
  with -0 == +0, the exact-match rule for NaN/0x7E classes, the k mismatch bound < 6% and v
  byte-exact. Everything else in s2 is kept verbatim: enc_mode 0/1 selection with both mode lines
  printed, the worst-6-mismatch print, two-state alternating graph replays with refilled inputs,
  the 320 MB L2-flush interleaved selector requiring the cluster to win by 1.02x over _master_attn
  at B=32, C=4 only, gate sliding, q_len 1, page 1, 17 <= batch <= 32 (config). HARD RULES:
  CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on
  input values, data-dependent control flow or per-call mutable state; gates read config only;
  nothing cached from inputs; precision per TASK.md; every key in the window attended; no child
  processes; per-state buffers/pointers only (never module globals); sm_90a-only build; canary +
  synchronize with launching/ok prints; any miss prints PATH CHOSEN ... (fallback: <reason>).
  First measurement is an okc check. Print compile seconds per load_inline module (dense chain,
  split-KV, sk-ticket, cluster) to watch the 15-minute budget.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "incumbent chain at B=32 is FA3 18.6 + combine 3.3 + prepare_varlen 2.4 + qk-norm/rope/casts/store/norm-gate ~6-8 us (~31-33 us, profile 3fd3fa1f); the same C=4 kernel ran 18.3 us in muse_attention_decode. Shape 240.7 -> ~226 us (weight 46; this is the worst-ratio scored shape, 1.080x vs SGLang). M1: probe class SATURATING and PATH CHOSEN = cluster; M2: sliding_cluster_kernel row <= 22 us; kill: shape not below 236"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "inherited from the sk-ticket base (its check: 308.7 -> 291.9 us); confirms the base, not this stage"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: inherited from the sk-ticket base (473.3 -> 444.5 us)
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: batch 1 keeps _sliding_fused_kernel; drift control
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: batch 16 is outside the 17..32 gate
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8, outside both gates
risk: medium
depends_on: sliding-cluster-c4-decode32-s2-k-encode-match
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32
review: required
status: queued
---

# sliding-cluster-c4-s3-satfinite-on-sk-ticket

## Change

Stage 3 of the sliding-cluster line. Stage 2 (sliding-cluster-c4-decode32-s2-k-encode-match, job 7c04214e90354dfa, 1.2803 check) was NEVER MEASURED either: it built (124 regs, 32 B spill, 103 KB smem, 62 clusters) but its prepare() probe printed 'probe: torch's cast never produced a NaN code below 520' and then 'PATH CHOSEN: sliding decode batch 17..32: master chain (fallback: torch's E4M3 overflow behaviour is not an edge the kernel can emulate)'. The self-check, two-state check and selector never ran, so its 1.2803 is master plus a dead path. The probe result means torch's bf16->float8_e4m3fn cast on the judge image does NOT emit NaN in 448..518, i.e. it saturates, which is exactly what the kernel's cvt.rn.satfinite.e4m3x2.f32 already does with no emulation. BASE: build on branch hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64 (submission blob d67278e0, the same base the in-flight dense-chain-s3 uses), merging s2's branch hyp/muse_decode_layer/sliding-cluster-c4-decode32-s2-k-encode-match (blob f8b86798) with base master 62c5a941 by 3-way git merge-file. A trial merge gives exactly 3 conflicts, all mechanical: (1) imports: keep sk-ticket's 'import random'; (2) prepare(): keep both, in the order _splitkv_prepare(device); _slide_prepare(state, device); _sk_prepare(state, device, t_prepare0); (3) kernel_fn's attention else-branch: s2 moved master's inline chain into _master_attn(data, state, qkv) while sk-ticket edited that inline chain to route full decode to its stream-K kernels; resolve by applying sk-ticket's edits INSIDE _master_attn so kernel_fn keeps s2's three-way branch (batch-1 sliding -> _sliding_forward, _slide_gate -> _slide_launch, else _master_attn). sk gates on window == 0 and the cluster on window != 0, so the two paths are disjoint and the cluster selector still times the unchanged sliding master chain. THE FIX (only _slide_probe_nan_thr and its caller change): (a) always print the probed byte table first (x = 440, 446, 448, 450, 456, 464, 466, 478, 480, 482, 512, 518 both signs, plus +inf, -inf, nan) as '[slide-c4] probe bytes: ...'; (b) classify: EDGE (RNE below some E, sign|0x7F at and above it) -> nan_thr = E as in s2; SATURATING (every finite |x| >= 448 gives sign|0x7E, no 0x7F anywhere for finite input) -> nan_thr = +inf, so the kernel's satfinite cvt is used alone; anything else -> print the class and fall back as now. Print '[slide-c4] probe class: EDGE|SATURATING|OTHER nan_thr=...'. (c) The synthetic overflow self-check case stays, but its expectation is master's own bytes (0x7E in the SATURATING case); keep s2's sign-magnitude ordinal within-one-code rule with -0 == +0, the exact-match rule for NaN/0x7E classes, the k mismatch bound < 6% and v byte-exact. Everything else in s2 is kept verbatim: enc_mode 0/1 selection with both mode lines printed, the worst-6-mismatch print, two-state alternating graph replays with refilled inputs, the 320 MB L2-flush interleaved selector requiring the cluster to win by 1.02x over _master_attn at B=32, C=4 only, gate sliding, q_len 1, page 1, 17 <= batch <= 32 (config). HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only; nothing cached from inputs; precision per TASK.md; every key in the window attended; no child processes; per-state buffers/pointers only (never module globals); sm_90a-only build; canary + synchronize with launching/ok prints; any miss prints PATH CHOSEN ... (fallback: <reason>). First measurement is an okc check. Print compile seconds per load_inline module (dense chain, split-KV, sk-ticket, cluster) to watch the 15-minute budget.

## Expected

- `*layer=sliding load=decode_32*` faster: incumbent chain at B=32 is FA3 18.6 + combine 3.3 + prepare_varlen 2.4 + qk-norm/rope/casts/store/norm-gate ~6-8 us (~31-33 us, profile 3fd3fa1f); the same C=4 kernel ran 18.3 us in muse_attention_decode. Shape 240.7 -> ~226 us (weight 46; this is the worst-ratio scored shape, 1.080x vs SGLang). M1: probe class SATURATING and PATH CHOSEN = cluster; M2: sliding_cluster_kernel row <= 22 us; kill: shape not below 236
- `*layer=full load=decode_16*` faster: inherited from the sk-ticket base (its check: 308.7 -> 291.9 us); confirms the base, not this stage
- `*layer=full load=decode_32*` faster: inherited from the sk-ticket base (473.3 -> 444.5 us)
- `*layer=sliding load=decode_1 *` flat: batch 1 keeps _sliding_fused_kernel; drift control
- `*layer=sliding load=decode_16*` flat: batch 16 is outside the 17..32 gate
- `*load=verify_*` flat: q_len 8, outside both gates

## Risk

If the probe class is OTHER (or the self-check misses again) the path falls back and the run re-measures the sk-ticket restack, the parent of the in-flight dense-chain-s3, so the check is not wasted; the printed byte table and mismatch classes then pin stage 4. Weakest link: realizable saving at the shape level. sliding-fused-batched-decode-port moved sliding decode_32 only 1.03x by replacing the same chain, so judge on the kernel row against the summed incumbent rows (milestone 2), not the headline (decode_32 sliding moves the geomean ~0.8% at best). Compile budget: four load_inline modules (~50 + ~40 + sk + ~41 s).

## Milestones

Rewrite stage 3 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
