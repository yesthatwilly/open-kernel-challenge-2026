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
review: proceed
status: queued
review_at: 2026-10-24T120721Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T120721Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a config-gated port of an already-measured, already-correct kernel onto the one shape where it won (sliding decode_1: 13.51 vs 16.56 us, about 18%). Every other shape stays byte-identical to master, so a single run is attributable per shape and needs no companion. The real risk is ordering against the running fa3-window-trimmed-page-table, not interpretability.

### Justification
The mechanism is plausible and measured, not inferred. At sliding decode_1 the shape is pure launch and latency (SoL 0.1 us). The master path is pre-attention, prepare_varlen_num_blocks, the FA3 main kernel, combine and rms_norm_gate. FINDINGS profiles it at FA3 8.2 us, combine 2.5 us and prepare 1.8 us, 16.3 us reported, and bfb7bac's own run has 16.56 us. The fused kernel's one launch measured 13.51 us on this exact shape (sliding-single-launch-fused-attention result.json). That run was built on fused-preattention-triton, whose sliding path is identical to master's: all three later landings touch only window == 0, so the old A/B carries over directly. The arithmetic is consistent with TASK weights. The decode_1 load is about 46*16.56 + 16*28.1 = 1211 us; saving 46*3.05 = 140 us is about 11.6% on that load, so roughly 1.9% on the 6-load geomean (1.269x to about 1.293x). Expected directions (faster on the one shape, same elsewhere) follow from the predicate (window and batch == 1 and q_len == 1, config only). Weakest link: the headline gain of about 1.9% sits just above the judge's +/-1.5% CI and inside the 1.5-2.5% run-to-run drift in LESSONS. As with batch-gate, the classifier may call it neutral even if it works, so judge it on the sliding decode_1 row (its paired ratio against SGLang and the fused kernel's top_kernels time of about 10.2 us) and on the other 11 rows staying flat. Second weakest: the 13.5 us comes from one run; Triton times reproduce within about 1%, so it is likely to hold. The S=16 in-kernel ticket combine adds no extra launch, which keeps the fused kernel's latency lower at B=1.

### Confounders on the current kernel
- fa3-window-trimmed-page-table (running, not landed): it targets the same sliding FA3 path. If it gains at sliding decode_1 (it shrinks FA3's prepare and combine work and its split count, about 4.3 us of the 16.3), this hypothesis's margin against the new master shrinks or vanishes, because the gate bypasses FA3 entirely at that shape. Either claim this on master after that run harvests, or compare its decode_1 sliding row against whichever master it will be promoted onto.
- fused-preattention-triton (landed): not a confounder for the measurement. The gated path replaces _pre_attention_kernel plus FA3 plus rms_norm_gate wholesale at that one shape, so the comparison is fused kernel versus master's full chain, which is exactly the decision. Every other shape still uses it unchanged.
- fa3-hdim128-head-pairing, paired-heads-launch-cleanup, paired-heads-launch-cleanup-ab-tune-batch-gate (landed): full layers only (window == 0); disjoint from the gated shape. The port must not disturb kernel_fn's paired branch: the early return sits before any paired logic, so it cannot.

### Interference and A/B needs
No companion is warranted. The change is a config-only early return that affects exactly one scored shape, and the other 11 shapes run master's code byte for byte. So a gain or a loss on sliding decode_1 can only come from the swapped path, and the flat shapes serve as the in-run drift control. An ab-revert of any landed hypothesis would not disambiguate anything: none of them touches the gated shape's path. A control isolating a sub-change (for example fused attention without the in-kernel combine) would be a new design question, not something needed to read this run. The one real dependency is the running fa3-window-trimmed-page-table. That is a sequencing question: dispatch after it harvests, or rebase onto the new master if it lands. It is not an A/B companion.

### Open questions
- Has fa3-window-trimmed-page-table harvested, and did it move sliding decode_1? If it did, recompute the expected margin against the new master's decode_1 sliding row before claiming, and consider holding if trimmed FA3 comes within about 1 us of 13.5.
- Was the earlier 13.51 us measured under identical judge and evaluator versions? Compare evaluator_sha256 with bfb7bac's run.
- If classify() calls it neutral only on CI overlap, the reviewer's acceptance rule is the per-shape test: sliding decode_1 at least about 10% faster with the other 11 rows within drift, the same reasoning used to merge batch-gate by hand.
- With 5 attention checks left today, this costs one check at a single-shape prediction of about 18%. Decide whether it ranks above the pending reruns.

