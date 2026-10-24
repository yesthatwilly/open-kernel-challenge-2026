---
problem: muse_attention_decode
slug: sliding-fused-triton-decode1-gate
created: 2026-10-24T120440Z
base: master
area: sliding-layer attention (batch-1 decode gate)
change: >
  Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence,
  _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim from
  muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.patch
  (impl commit db5fa4d87590) onto current master (bfb7bac, paired full layers, batch>1 pairing
  gate), and call it only when the config says sliding batch-1 decode: at the top of kernel_fn, if
  cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1: return _sliding_forward(data). Every
  other shape (sliding batch 16/32 decode, every sliding verify block, every full layer) keeps
  master exactly: _pre_attention_kernel, FA3, rms_norm_gate or the paired epilogue. The predicate
  reads config only. Mechanism: at sliding decode_1 the master path is five launches
  (pre-attention 2.1 us, prepare_varlen_num_blocks 1.8 us, FA3 main 8.2 us, combine 2.6 us,
  norm_gate 1.1 us = 15.7 us kernel time, 16.6 us reported); the fused kernel did the whole step
  (norm, RoPE, cache store, 16-way split window attention with an in-kernel ticket combine,
  norm-gate) in one 10.2 us launch and 13.5 us reported (1.54x vs SGLang 20.8 us, job
  f389af196ddf47b1). It lost at batch 16/32 and on verify because its grid and M=64 query tile are
  poorly shaped there, which this gate excludes. This is the stage 2 that DECISIONS
  2026-10-24T103755Z named (gate the fused kernel to decode_1 only), filed as a primary on master
  so it keeps the head-pairing gains.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: one 10.2 us launch replaces five (15.7 us kernel sum); measured 13.5 vs 16.6 us, ~19% on the shape, ~12% on the decode_1 load, ~2% geomean
  - shapes: "*layer=full*"
    direction: same
    why: untouched
  - shapes: "*layer=sliding load=decode_16*"
    direction: same
    why: gated out, master path
  - shapes: "*layer=sliding load=decode_32*"
    direction: same
    why: gated out, master path
  - shapes: "*layer=sliding load=verify_*"
    direction: same
    why: gated out, master path
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-fused-triton-decode1-gate

## Change

Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim from muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.patch (impl commit db5fa4d87590) onto current master (bfb7bac, paired full layers, batch>1 pairing gate), and call it only when the config says sliding batch-1 decode: at the top of kernel_fn, if cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1: return _sliding_forward(data). Every other shape (sliding batch 16/32 decode, every sliding verify block, every full layer) keeps master exactly: _pre_attention_kernel, FA3, rms_norm_gate or the paired epilogue. The predicate reads config only. Mechanism: at sliding decode_1 the master path is five launches (pre-attention 2.1 us, prepare_varlen_num_blocks 1.8 us, FA3 main 8.2 us, combine 2.6 us, norm_gate 1.1 us = 15.7 us kernel time, 16.6 us reported); the fused kernel did the whole step (norm, RoPE, cache store, 16-way split window attention with an in-kernel ticket combine, norm-gate) in one 10.2 us launch and 13.5 us reported (1.54x vs SGLang 20.8 us, job f389af196ddf47b1). It lost at batch 16/32 and on verify because its grid and M=64 query tile are poorly shaped there, which this gate excludes. This is the stage 2 that DECISIONS 2026-10-24T103755Z named (gate the fused kernel to decode_1 only), filed as a primary on master so it keeps the head-pairing gains.

## Expected

- `*layer=sliding load=decode_1 *` faster: one 10.2 us launch replaces five (15.7 us kernel sum); measured 13.5 vs 16.6 us, ~19% on the shape, ~12% on the decode_1 load, ~2% geomean
- `*layer=full*` same: untouched
- `*layer=sliding load=decode_16*` same: gated out, master path
- `*layer=sliding load=decode_32*` same: gated out, master path
- `*layer=sliding load=verify_*` same: gated out, master path

## Risk

Risk is low: the kernel already passed all 16 tests including batch 1 ctx 10 sliding and window-edge keys. The _COUNTERS buffer is zeroed from config shape only and every launch leaves it at 0, holds no data. Watch: the fused query path keeps fp32 queries (finer than FP8, allowed); the in-kernel combine uses a gpu-scope fence and ticket. Overlaps fa3-scheduler-metadata-side-stream only on this one shape (that idea moves FA3 prepare off the critical path; here FA3 is not called). If fa3-window-trimmed-page-table (running) gains on sliding decode_1, the expected margin shrinks.
