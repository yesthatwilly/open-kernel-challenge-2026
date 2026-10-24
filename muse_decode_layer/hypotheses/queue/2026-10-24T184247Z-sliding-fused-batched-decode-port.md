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
review: required
status: queued
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
