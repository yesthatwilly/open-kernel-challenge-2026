---
problem: muse_decode_layer
slug: sliding-fused-batched-decode-port
created: 2026-10-24T184247Z
base: master
area: FA3 call (sliding decode, batch > 1)
change: >
  Port muse_attention_decode master batched sliding-decode path (landed
  sliding-deep-splits-on-cuda-splitkv, job 698d4cd3838c4a63: sliding decode_16 20.7 -> 15.3 us,
  decode_32 24.9 -> 24.2 us there, with 16 splits at every decode batch) to decode_layer kernel_fn
  for sliding-window decode at batch > 1. Copy verbatim from muse_attention_decode/submission.py
  on master: LOG2E, _INV_FREQ and _inv_freq, _COUNTERS and _counters, _num_splits, _norm_rope,
  _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel. Add
  _sliding_forward_batched(data, qkv, cfg), adapted from attention _sliding_forward: the qkv
  operand is master qkv = _gemm(data["attn_input"], dense["qkv_gate"]) ([T, 2304] bf16, pass
  qkv.stride(0)) instead of data["qkv"]; splits = 16 (q_len 1, attention rule); same ws_acc /
  ws_ml / _counters buffers, same launch arguments (positions, cache_loc, seq_lens, page_table,
  inv_freq, k_scale, v_scale, k_cache/v_cache viewed as uint8, HQ, HKV, D, L=1, M, LN, WINDOW=cfg
  window, PAGE=cfg page_size, S=16, BLOCK_N=64, ROPE=bool(rope_theta), EPS=RMS_EPS, num_warps=4,
  num_stages=2); it returns [B, 1, 1024] bf16 that is already QK-normed, RoPEd, cached (FP8 k/v
  stored at cache_loc), attended over the 2048-token window with FP8-rounded queries,
  split-combined and per-head RMSNorm x sigmoid(gate) gated. Gate in kernel_fn on config only:
  fused = window != 0 and q_len == 1 and batch > 1 and hq == 16 and hkv == 2 and d == 64. On that
  branch skip FusedQKNorm, the k/v/q FP8 casts, store_cache, flash_attn_with_kvcache and
  state["o_norm_gate"], and feed o = out.reshape(tokens, hq * d) to the unchanged o GEMM;
  everything after the o GEMM is master. Every other shape keeps master byte for byte, including
  sliding decode at batch 1 (that shape is the in-judge restack decode_1 port). If
  decode-restack-sliding-port-paired-pad-gemms-t1 has landed on master by claim time, its copies
  of these helpers are already in the file: reuse them and only widen its gate to batch > 1 with
  splits 16 at q_len 1, never add a second copy.
expected:
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: "one ~13 us launch replaces FusedQKNorm, three FP8 casts, store_cache, FA3 prepare + main 13.6 us + FlashAttnFwdCombine 3.7 us and FusedRMSNormGate (~28 us of 230 us in a017e10aec544c55): -10..-14 us, +4-6% on a weight-46 shape"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "fused kernel 22.1 us (attention profile) vs FA3 18.3 us + combine + prepare + ~5 glue launches (~33 us): -6..-10 us, +2-4%; this shape is at 1.000x vs SGLang today"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: batch 1 outside the gate (restack covers it)
  - shapes: "*layer=full*"
    direction: flat
    why: untouched control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 outside the gate
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T185011Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T185011Z
base_sha: 90040c7132925d246133b05640f35bf2b6622eb8
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/sliding-fused-batched-decode-port
worktree: .worktrees/muse_decode_layer/sliding-fused-batched-decode-port
mode: check
implemented_at: 2026-10-24T231258Z
impl_commit: 3ddcc4f2a8510a59d4eedab0dacea5bd02078ec8
submission_sha256: b671bd387506452d90e24f197c10a969ff30824efd32c0dba3fd5db5bada9d99
norm_sha256: 4df695a2d87724d8a398580b8b49d2f64f7e0d52e089e3ce56ee4adbf424af36
submission_blob: f99bbe8e814e229a1d528bd8f951814baf1f0aaf
lint_warnings: 20
attempt_file: 2026-10-24T224624Z-check-sliding-fused-batched-decode-port.md
prepared_at: 2026-10-24T224625Z
---

# sliding-fused-batched-decode-port

## Change

Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-on-cuda-splitkv, job 698d4cd3838c4a63: sliding decode_16 20.7 -> 15.3 us, decode_32 24.9 -> 24.2 us there, with 16 splits at every decode batch) to decode_layer kernel_fn for sliding-window decode at batch > 1. Copy verbatim from muse_attention_decode/submission.py on master: LOG2E, _INV_FREQ and _inv_freq, _COUNTERS and _counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel. Add _sliding_forward_batched(data, qkv, cfg), adapted from attention _sliding_forward: the qkv operand is master qkv = _gemm(data["attn_input"], dense["qkv_gate"]) ([T, 2304] bf16, pass qkv.stride(0)) instead of data["qkv"]; splits = 16 (q_len 1, attention rule); same ws_acc / ws_ml / _counters buffers, same launch arguments (positions, cache_loc, seq_lens, page_table, inv_freq, k_scale, v_scale, k_cache/v_cache viewed as uint8, HQ, HKV, D, L=1, M, LN, WINDOW=cfg window, PAGE=cfg page_size, S=16, BLOCK_N=64, ROPE=bool(rope_theta), EPS=RMS_EPS, num_warps=4, num_stages=2); it returns [B, 1, 1024] bf16 that is already QK-normed, RoPEd, cached (FP8 k/v stored at cache_loc), attended over the 2048-token window with FP8-rounded queries, split-combined and per-head RMSNorm x sigmoid(gate) gated. Gate in kernel_fn on config only: fused = window != 0 and q_len == 1 and batch > 1 and hq == 16 and hkv == 2 and d == 64. On that branch skip FusedQKNorm, the k/v/q FP8 casts, store_cache, flash_attn_with_kvcache and state["o_norm_gate"], and feed o = out.reshape(tokens, hq * d) to the unchanged o GEMM; everything after the o GEMM is master. Every other shape keeps master byte for byte, including sliding decode at batch 1 (that shape is the in-judge restack decode_1 port). If decode-restack-sliding-port-paired-pad-gemms-t1 has landed on master by claim time, its copies of these helpers are already in the file: reuse them and only widen its gate to batch > 1 with splits 16 at q_len 1, never add a second copy.

## Expected

- `*layer=sliding load=decode_16 *` faster: one ~13 us launch replaces FusedQKNorm, three FP8 casts, store_cache, FA3 prepare + main 13.6 us + FlashAttnFwdCombine 3.7 us and FusedRMSNormGate (~28 us of 230 us in a017e10aec544c55): -10..-14 us, +4-6% on a weight-46 shape
- `*layer=sliding load=decode_32 *` faster: fused kernel 22.1 us (attention profile) vs FA3 18.3 us + combine + prepare + ~5 glue launches (~33 us): -6..-10 us, +2-4%; this shape is at 1.000x vs SGLang today
- `*layer=sliding load=decode_1 *` flat: batch 1 outside the gate (restack covers it)
- `*layer=full*` flat: untouched control
- `*load=verify_*` flat: q_len 8 outside the gate

## Risk

Proven kernel (attention 1.309x -> 1.395x lineage); the batch-1 port of the same kernel gave +3.4-4.3% on sliding decode_1 here, so the decode_layer glue it removes is real. Overlap: decode-restack-sliding-port-paired-pad-gemms-t1 (in judge) adds the same helpers for batch 1 only; shapes are disjoint, code overlaps (merge, do not duplicate). The fused kernel zeroes _counters once on first eager call and leaves them at 0 (no data carried). Risk: kernel never ran inside the whole layer at batch 16/32; the 13 tests include sliding decode at batch 8 (page 64) and batch 9 (page 1).

## Adversarial review

Reviewed 2026-10-24T185011Z by an adversarial subagent (model opus).

**Verdict: proceed.** Shape-disjoint port of a proven fused sliding-decode kernel onto an attention chain that no landed hypothesis has touched; the sliding decode_16/32 per-shape readouts are cleanly attributable with full/verify/decode_1 shapes as in-run controls, so no companion is needed, though the headline effect (~0.6-1%) will sit inside the +/-1.5% CI.

### Justification
Plausible. Master's sliding attention chain for T=16/32 is still SGLang's (qkv GEMM -> FusedQKNorm -> k/v FP8 casts -> store_cache -> q cast -> FA3 main + combine -> FusedRMSNormGate -> o GEMM); small-t-launch-minimal-dense-chain-pdl only rewrote the post-attention chain, so the ~7 launches the fused kernel removes are all still there. FINDINGS profile for sliding decode_16 (217.1 us, floor 76.7) does not list FA3 among the top 3 (Marlin 126.1, nvjet 26.5, pooler_router 23.9), consistent with FA3 main ~13.6 us + combine 3.7 us + glue; same at decode_32 (247.0 us, floor 84.5). The same Triton kernel already ran inside this layer at batch 1 (sliding-fused-triton-decode1-port: 13/13 tests, sliding decode_1 gain) and in muse_attention_decode at every decode batch with 16 splits, so correctness and graph-safety are established; the 13 tests cover the new gate at batch 8 (ctx 2047) and batch 9 (page 1, mixed spread). Expected directions match the gate: only sliding decode_16/32 change; full, verify and sliding decode_1 are byte-identical controls. Magnitude: -10..-14 us on 217 us is +4.6-6.5% on sliding decode_16 and -6..-10 us on 247 us is +2.5-4% on sliding decode_32, but with weight 46 sliding + 16 full per load that is ~3.5% and ~1.8% of the two load times and ~0.9% on the 6-load geomean, i.e. inside the run's CI; judge it per shape (per-shape err is ~0.7 us). Weakest link: the savings are estimated from an older profile (a017e10aec544c55, 230 us) and from standalone muse_attention_decode timings; at decode_32 the 16-split choice (1024 CTAs, ~4 MB fp32 workspace write+reread) only beat 2 splits by 0.7 us there, so the decode_32 gain rests almost entirely on removed glue launches, not on a faster attention core, and could come in near flat.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (on master, base): touches T<=32 but only after the o GEMM (P/R kernels, PDL on its own launches after attn_boundary); the attention chain the hypothesis replaces is untouched, so no mechanistic interaction; it only shrinks the denominator, making the same absolute saving a slightly larger ratio. Not a confounder for attribution.
- fa3-hdim128-head-pairing-batched (on master): gated window == 0 and batch > 1; the new gate requires window != 0, so the code paths are disjoint. No confound.
- post-expert-norm-triton-small-t, marlin-block-m-cap-16, sglang-pipeline-parity-router-hilo-m8: MoE/router only, downstream and unchanged by this hypothesis; the o GEMM input layout ([T, 1024] bf16) is identical, so nothing downstream sees a different operand. No confound.

### Interference and A/B needs
No companion warranted. The gate is config-only and shape-disjoint: within the same run, sliding decode_16 and decode_32 are the only shapes whose kernels change, while sliding decode_1, all full-layer shapes and all verify shapes are byte-identical to master and serve as in-run noise controls. No landed hypothesis shares the replaced kernels, so an ab-revert would measure the same thing. A split-count control (_num_splits(batch, hkv) = 4/2 instead of 16) would isolate a sub-choice but is not needed to interpret this run: a gain or a loss on the two target shapes is attributable to the fused path as a whole; if decode_32 comes in flat or slower while decode_16 gains, the natural follow-up is a stacked splits-tuning hypothesis, not a companion now.

### Open questions
- The hypothesis's conditional on decode-restack-sliding-port-paired-pad-gemms-t1 is moot: that run was harvested neutral (beats-base-not-best) and is not on master, so the helpers are added fresh; the implementer should not try to merge from the restack branch.
- At decode_32 is 16 splits (1024 CTAs) better than _num_splits = 2 inside the layer, where the qkv GEMM leaves the L2 in a different state than the standalone attention benchmark? Read the fused kernel's own us from the profile against FA3 main + combine (18.3 + 3.7 us).
- Does the fused kernel's split-combine workspace traffic (B*Hkv*16*16*64*4 B, ~4 MB at B=32) show up as extra latency in the following o GEMM's cold-L2 weight read? Check the o GEMM nvjet time on sliding decode_32 versus master.


## Implementation notes

- Worktree muse_decode_layer/submission.py only. Inserted just before kernel_fn, copied verbatim from muse_attention_decode/submission.py (master): _INV_FREQ, _inv_freq, LOG2E, _COUNTERS, _counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel. (_num_splits is copied but unused: splits fixed at 16 for q_len 1.)
- Added _sliding_forward_batched(data, qkv, cfg): attention's _sliding_forward with qkv = the qkv GEMM output (stride(0) passed), splits = 16, same workspaces/counters/launch args.
- kernel_fn: `fused = window != 0 and q_len == 1 and batch > 1 and hq == 16 and hkv == 2 and d == 64` (config only). On it: o = out.reshape(tokens, hq*d) feeds the unchanged o GEMM. Else branch is the master attention code verbatim, only re-indented (diff shows it as changed lines). Everything after is untouched.
- Deviation: none. Per review, no merge from the restack branch.

## Rules check

- [x] CUDA-graph capturable: new path has torch.empty + a Triton launch, no sync; _counters zeroed once on first eager call, branches on config only.
- [x] Nothing derived from `data` cached: _counters/_inv_freq key on (batch, hkv, device) / (theta, d, device); inv_freq is a function of config, counters hold no data (reset to 0 by the kernel).
- [x] Weights: n/a: no weight change; prepare() untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (k/v stored E4M3 at cache_loc); queries bf16 (finer than FP8 cap).
- [x] Attention sees every key: whole 2048-token window plus the new key; no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (checked in diff).
- [x] All work inside the timed call; no output caching or replay.
- [x] Gate shape-disjoint: full, verify (q_len 8) and batch 1 take the master branch.

okc_lint: "OK with warnings" (only host-sync warnings outside kernel_fn's call graph, in prepare/self-check helpers that predate this change; no errors). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading (no GPU on this node). **Verdict: dispatch-with-fixes.**

Checked and found sound:
- The port is verbatim. All eight helpers (_inv_freq, _counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, _sliding_fused_kernel) diff identical to muse_attention_decode master, and nothing in the layer file is duplicated or shadowed.
- Layout plumbing matches. The qkv GEMM output is [T, 2304] bf16 (q 16x64 | k 2x64 | v 2x64 | gate 16x64), the same layout as the attention problem's data["qkv"], and qkv.stride(0) is passed. The k/v cache layout [pages, page_size, 2, 64], positions, cache_loc, seq_lens (the block start), page_table and the k/v scales have the same meaning in both problems. The decode layer's QK-norm and output norm are scaleless (FusedQKNorm, FusedRMSNormGate has_weight=False), and the reference RoPE is interleaved with theta^(-2i/64), all as the kernel computes. The softmax scale sm_scale * qk_scale_factor splits into q_scale x score_scale. The output [B, 1, 1024] is reshaped to [T, 1024] and feeds the unchanged o GEMM.
- The gate reads config only: sliding, q_len 1, batch > 1, 16/2/64 heads. It covers sliding decode_16/32 and the tests at batch 8 (page 64, ctx 2047) and batch 9 (page 1). Decode_1, every full layer and every verify shape stay on master byte for byte. There is no self-check gate that could always fail.
- Window and every-key rule: old keys [seq - 2047, seq - 1] plus the new key make 2048 keys, which equals FA3 window_size=(window - 1, 0). The 16 splits cover the whole range, and an empty split gets weight 0. The new k/v are stored as E4M3 at cache_loc, and the queries are bf16, finer than FP8.
- Capture: allocation depends on shape only. The ticket counters reset in-kernel and hold no data, and inv_freq comes from config.

Fix:
- The new path had no fallback and no PATH CHOSEN line, so a Triton build or launch error would have raised out of kernel_fn and failed the run. The fused call is now wrapped in try/except keyed on config (batch, page, window, rope, scaled). On failure it turns the key off for good, prints "PATH CHOSEN = MASTER (FA3)" and runs the master branch (else became `if not fused`). On first success it prints "PATH CHOSEN = FUSED". Lint OK (only host-sync warnings in prepare-only code). py_compile OK, and __pycache__ was removed.

Residual risk: this kernel has never run inside the layer at batch 16/32. A wrong result would fail the run rather than fall back (there is no numeric self-check), and the decode_32 gain may come out flat if the 16-split workspace traffic cancels the launches saved.
