# muse_attention_decode · 2026-10-24T124236Z · check · sliding-fused-triton-decode1-gate

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T120440Z-sliding-fused-triton-decode1-gate.md` (branch `hyp/muse_attention_decode/sliding-fused-triton-decode1-gate`).

- **Job:** `295727fc68e14093` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `5e0c27c`

## Goal
Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim from muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.patch (impl commit db5fa4d87590) onto current master (bfb7bac, paired full layers, batch>1 pairing gate), and call it only when the config says sliding batch-1 decode: at the top of kernel_fn, if cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1: return _sliding_forward(data). Every other shape (sliding batch 16/32 decode, every sliding verify block, every full layer) keeps master exactly: _pre_attention_kernel, FA3, rms_norm_gate or the paired epilogue. The predicate reads config only. Mechanism: at sliding decode_1 the master path is five launches (pre-attention 2.1 us, prepare_varlen_num_blocks 1.8 us, FA3 main 8.2 us, combine 2.6 us, norm_gate 1.1 us = 15.7 us kernel time, 16.6 us reported); the fused kernel did the whole step (norm, RoPE, cache store, 16-way split window attention with an in-kernel ticket combine, norm-gate) in one 10.2 us launch and 13.5 us reported (1.54x vs SGLang 20.8 us, job f389af196ddf47b1). It lost at batch 16/32 and on verify because its grid and M=64 query tile are poorly shaped there, which this gate excludes. This is the stage 2 that DECISIONS 2026-10-24T103755Z named (gate the fused kernel to decode_1 only), filed as a primary on master so it keeps the head-pairing gains.

## What changed, and where
- Worktree file: `.worktrees/muse_attention_decode/sliding-fused-triton-decode1-gate/muse_attention_decode/submission.py` (base 4613c6f, blob efc5441). Diff: 1 file, 238 insertions, 0 deletions.
- Ported verbatim from `db5fa4d:muse_attention_decode/submission.py`: `LOG2E`, `_COUNTERS`, `_counters`, `_num_splits`, `_norm_rope`, `_gpu_fence`, `_online_update`, `_gate_epilogue`, `_sliding_fused_kernel`, `_sliding_forward`. Inserted between `_pre_attention_kernel` and `_paired_norm_gate_kernel` (same relative place as db5fa4d). Checked mechanically: the text from `LOG2E = ...` to `_sliding_forward`'s `return out` matches the commit byte for byte.
- `kernel_fn`: one early return right after `cfg = data["config"]`, before any paired logic: `if cfg["window"] and cfg["batch"] == 1 and cfg["q_len"] == 1: return _sliding_forward(data)`, plus a two-line comment. The predicate reads config only. Everything after it (pre-attention, FA3, paired gate `batch > 1`, `rms_norm_gate` / `_paired_norm_gate_kernel`) is byte-identical to master.
- At B=1, Hkv=2, `_num_splits` gives S=16 (grid 1x2x16), M=16 (group 8 x L 1), the in-kernel ticket combine is used, `_COUNTERS[(1, 2, dev)]` is the only counter buffer created. It is zeroed once from config shape and reset to 0 by the last-ticket program every launch.
- No deviation from the spec.
- Open question 1 (sequencing): fa3-window-trimmed-page-table has harvested to `done/regression/`, so it did not land and master's sliding FA3 path is the one this gate compares against. The margin estimate stands.
- Open questions 2-4 (evaluator sha match, per-shape acceptance rule, whether it is worth one of today's checks) are for the dispatcher/harvester, not the implementation.

## Rules check (each must be true before the run)
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

## Risks / what could fail
low
Risk is low: the kernel already passed all 16 tests including batch 1 ctx 10 sliding and window-edge keys. The _COUNTERS buffer is zeroed from config shape only and every launch leaves it at 0, holds no data. Watch: the fused query path keeps fp32 queries (finer than FP8, allowed); the in-kernel combine uses a gpu-scope fence and ticket. Overlaps fa3-scheduler-metadata-side-stream only on this one shape (that idea moves FA3 prepare off the critical path; here FA3 is not called). If fa3-window-trimmed-page-table (running) gains on sliding decode_1, the expected margin shrinks.

## Result (after the run)
- job `295727fc68e14093` (check), status done, passed True, tests 16/16
- speedup 1.2931x, 95% CI [1.2737, 1.3212], score 2912.693503437914 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2686474414890054, "ci": [1.2496177298666704, 1.2876771531113405], "job": "bd64bf9f9e904f02"}; base result: muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-gate.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-gate.patch`; branch `hyp/muse_attention_decode/sliding-fused-triton-decode1-gate`
