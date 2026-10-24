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
status: running
review_at: 2026-10-24T120721Z
review_companions: null
hold_reason: null
stage: dispatched
claimed_at: 2026-10-24T124116Z
base_sha: 4613c6fccb0ad0d891d7c837d1dc24c0465c5aff
base_blob: efc544103e6d303d1f8602caa9439628043a17a7
branch: hyp/muse_attention_decode/sliding-fused-triton-decode1-gate
worktree: .worktrees/muse_attention_decode/sliding-fused-triton-decode1-gate
mode: check
implemented_at: 2026-10-24T124236Z
impl_commit: 5e0c27cca32d3300b0653db04b25fe7a72fb03e1
submission_sha256: 3178c8c418f2d89cbd468deeca07b683922af8f04f2945fbf7cf5e87f185dd2b
norm_sha256: ff853cb8d70fa1861bfe365b397ad2db92085721e9e815e5c20e6057cc40571b
submission_blob: 387dcb6a0e6137b74f7f65b048dce5a3fcbe2008
lint_warnings: 0
attempt_file: 2026-10-24T124236Z-check-sliding-fused-triton-decode1-gate.md
prepared_at: 2026-10-24T124236Z
job: 295727fc68e14093
dispatched_at: 2026-10-24T124248Z
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


## Implementation notes

- Worktree file: `.worktrees/muse_attention_decode/sliding-fused-triton-decode1-gate/muse_attention_decode/submission.py` (base 4613c6f, blob efc5441). Diff: 1 file, 238 insertions, 0 deletions.
- Ported verbatim from `db5fa4d:muse_attention_decode/submission.py`: `LOG2E`, `_COUNTERS`, `_counters`, `_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue`, `_sliding_fused_kernel`, `_sliding_forward`. Inserted between `_pre_attention_kernel` and `_paired_norm_gate_kernel` (same relative place as db5fa4d). Checked mechanically: the text from `LOG2E = ...` to `_sliding_forward`'s `return out` matches the commit byte for byte.
- `kernel_fn`: one early return right after `cfg = data["config"]`, before any paired logic: `if cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1: return _sliding_forward(data)`, plus a two-line comment. The predicate reads config only. Everything after it (pre-attention, FA3, paired gate `batch > 1`, `rms_norm_gate` / `_paired_norm_gate_kernel`) is byte-identical to master.
- At B=1, Hkv=2, `_num_splits` gives S=16 (grid 1x2x16), M=16 (group 8 x L 1), the in-kernel ticket combine is used, `_COUNTERS[(1, 2, dev)]` is the only counter buffer created. It is zeroed once from config shape and reset to 0 by the last-ticket program every launch.
- No deviation from the spec.
- Open question 1 (sequencing): fa3-window-trimmed-page-table has harvested to `done/regression/`, so it did not land and master's sliding FA3 path is the one this gate compares against. The margin estimate stands.
- Open questions 2-4 (evaluator sha match, per-shape acceptance rule, whether it is worth one of today's checks) are for the dispatcher/harvester, not the implementation.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/sliding-fused-triton-decode1-gate diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the added code. `torch.empty` workspaces and grid sizes come from config (batch, q_len, heads, head_dim) only. The gate and `splits > 1` branch on config only. The in-kernel ticket/fence is device-side.
- [x] Nothing derived from `data` is cached across calls: `_COUNTERS` is keyed on (batch, hkv, device), zero-initialised, holds no data and is left at 0 by every launch. `_inv_freq` (unchanged master code) depends on theta/d only.
- [x] n/a: no weights in this problem; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache stored as FP8 E4M3 (`k8`/`v8`, divided by the scales); queries kept bf16 (finer than FP8 E4M3, allowed); output bf16. Softmax/accumulation in fp32.
- [x] Attention sees every key: old keys over `[max(0, seq-2047), seq-1]` split into S chunks that cover the range, plus the new key from registers; the mask only enforces the window edge and causality, no skipping by weight.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (only triton/torch, the inline PTX fence is a memory fence).
- [x] All work happens inside the timed call; no output caching or replay. The new cache rows are written every call.
- [x] First line `# okc: problem=muse_attention_decode` kept; no `triton.autotune` added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
CLEAN: .worktrees/muse_attention_decode/sliding-fused-triton-decode1-gate/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK (exit 0); `__pycache__` removed from the worktree.
