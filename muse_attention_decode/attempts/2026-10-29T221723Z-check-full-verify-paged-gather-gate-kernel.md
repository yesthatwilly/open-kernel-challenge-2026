# muse_attention_decode · 2026-10-29T221723Z · check · full-verify-paged-gather-gate-kernel

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-29T215944Z-full-verify-paged-gather-gate-kernel.md` (branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel`).

- **Job:** `06c051723643462f` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `9f9e177c`

## Goal
Replace master's FA3 call on the full-attention verify rows at B=16 and B=32 (q_len 8, paged FP8 KV, ctx ~80000) with the H100-validated gate kernel from takemyjob hopper/fa3gate 4b2a169 (probes/fa3gate; research/h100-fa3gate-1029.md): a persistent paged-gather attention kernel, 3 stages x 2 CTAs/SM = 64 KB in flight per SM, page indices prefetched a ring ahead, FP8 QK^T by wgmma straight from the swizzled cp.async ring, f16 PV, online softmax, grid-barrier distributed combine. Build it as a separate background-built load_inline extension; route per row (verify_16 full, verify_32 full only; verify_1 and every sliding/decode row keep master) by an in-prepare interleaved race through kernel_fn (read-flushed, cold L2) against master's FA3 path: on only if within the problem tolerance on every draw (fp64 relL2 and the judge metric, which the node measured at 2.5-2.8e-2 vs bounds 0.083/0.085) and <= 0.95 vs master in every round; else master. Inputs fresh per call (page table, lengths); no caching of anything input-derived; no host sync in kernel_fn. Print an [fvg] summary (<= 1.5 KB) at the end of prepare: build s, regs/spill/smem/occupancy, per-row per-round ratios and errors, PATH CHOSEN.

## What changed, and where
**Mechanism:** full verify B=16/32 (q_len 8, page 1, kv_scale 1): FA3 paired call + _paired_norm_gate_kernel replaced by one persistent paged-gather gate kernel (fa3gate v6, 3 stages x 2 CTAs/SM, wgmma FP8 QK / f16 PV, grid-barrier combine with the RMSNorm x sigmoid(gate) epilogue fused), routed per row by an in-prepare race vs master.

- New `[fvg]` section before the module-level `if torch.cuda.is_available():` block (worktree `muse_attention_decode/submission.py`, +1001 lines, file 539,763 B). Hooks: `_fvg_start_build()` is the first statement of that block; `kernel_fn` takes `_fvg_forward` right after the `_SW_ROUTE` check when the frozen `_FVG_ROUTE` has the batch, `_fvg_gate(cfg)` holds (config only) and `_fvg_shape_ok(data)` holds (shape only). `_fvg_setup()` runs last, after `_sw_setup`. The `[fvg]` summary prints after the `[paths]` re-print, so it is the last part of the log.
- CUDA: only fa3gate.cu's `fa3y_kernel<NS=3, MINB=2, MODE, F=0, FN=3>` (cfg 8) was kept, with its helpers copied verbatim, both fences included (the cp.async->wgmma `fence.proxy.async` stays). Dropped: the lineage kernels v1-v5, the SM-split and fenceless variants, `g_dbg`/globaltimer/smid. MODE 1 is the routed mode. MODE 0, 2 and 3 are diag only. Changes from the source:
  1. The early `return` for empty ranges is gone (with a grid barrier it would hang). Ranges are now cut over `Ge = min(G, T)`, and CTAs >= Ge skip the mainloop but still arrive at the barrier and take part in the combine (`cta_of(.., Ge)`).
  2. The barrier spin is bounded (2^22 x 128 ns) and sets error bit 1. Bit 2 means more than 1024 tiles per CTA, which the host shape gate prevents. Bit 4 means a non-sm_90a build, under the `#if __CUDA_ARCH_FEAT_SM90_ALL` guard.
  3. The combine writes the final bf16 output, with RMSNorm and sigmoid(gate) applied in fp32 from qkv.
- Launch: `cudaLaunchKernelEx` with `cudaLaunchAttributeCooperative` only. There is no programmatic-serialization or PDL attribute, and the preceding Triton pre-attention kernel uses no PDL. G = 2 x `multiProcessorCount`. Prepare requires occupancy >= 2, localSizeBytes == 0 and cooperative-launch support, or it keeps master. A capture probe checks that a cooperative launch can be captured; if it cannot, the plain launch is used and the line says so. Each row (16, 32) has its own counters [arrive, depart, error] and its own partials, allocated in prepare from G and the batch. They are re-zeroed after every race step and in `finally`.
- Layout (review confounder): routed rows switch both ends. `_pre_attention_kernel` gets PAIRED=False (the unpaired e4m3 q, the same call master's B=1 verify makes), and the gate writes `[B*8, 1024]` bf16 with the epilogue fused. Neither `_paired_norm_gate_kernel` nor `rms_norm_gate` is launched. This is printed. The race times this whole path through kernel_fn.
- Build: `load_inline` runs in a daemon `threading.Thread` started at the top of the import, with `-gencode=arch=compute_90a,code=sm_90a` in `extra_cuda_cflags`. That flag contains "arch", so torch adds no arch flags, and the thread never touches `TORCH_CUDA_ARCH_LIST`, which the main thread's builds set and restore. The thread is joined with a timeout at a 470 s import-clock rule. If it has not joined, or the build failed, master stays.
- Race, per row and independently, through kernel_fn:
  - Three eager draws: agentic soft keys, agentic with coherent values, and short lengths with T < G (which exercises Ge). The agentic draws force tails (n+8)%64 of 1 and 63 and the longest row, 86000.
  - Each draw is checked against the submission's own fp64 references: e4m3-q relL2 <= 2e-2, judge-style relL2 vs the unrounded ref <= clamp(2.5 x ref error + 1.5%, 3%, 15%), worst row <= 5x the bound, finite. Master's judge metric on the same draw is printed too.
  - Then the two-state `_alt_check` replay, and 3 rounds of interleaved cold-L2 graph replays (route off vs on, empty graph subtracted). A row routes only if the ratio is <= 0.95 in every round.
  - A one-shot diag follows (start rule 500 s): MODE 1/0/2/3 kernel-only us, with implied TB/s at v16/v32.
- Deviations / risks: `kernel_fn` still allocates its q and output with `torch.empty`, as master does; persistent buffers are never allocated in kernel_fn. The kernel was never compiled here (this node has no nvcc), so the first build is the judge's. Node validation was skipped: subagent H100 runs outside a user-approved command are refused. The background nvcc runs alongside master's sequential builds, so if CPU contention delays them, master's own phase gates could shift.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: `_fvg_forward` makes Triton + one ext launch on the current stream; no `.item()/.cpu()/synchronize()` in kernel_fn's call graph (lint: none); branches only on config (`_fvg_gate`) and shapes (`_fvg_shape_ok`); lens/page table read on device; capture verified in prepare by `_alt_check` and the race graphs.
- [x] Nothing derived from `data` is cached across calls: `_FVG_BUF` holds partials/counters sized from G and batch only; `_FVG_ROUTE` frozen at import from synthetic races.
- [x] Weights: n/a: attention problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (read as e4m3, untouched format); queries rounded to FP8 E4M3 (as FA3); P f16 and fp32 O/softmax (finer than FA3); output bf16.
- [x] Attention sees every key: stream-K covers every 64-key tile of every request; only causal / length masking on the last tiles.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import (own fp64 reference), subprocesses (a `threading.Thread` around load_inline, as master's builds) or network.
- [x] All work happens inside the timed call: pre-attention, attention, combine and epilogue all run per call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
warn  host-sync  ... (outside kernel_fn's call graph; fine in prepare())   [x ~110, master's prepare + [fvg] prepare lines]
warn  dtype      L7646: q = torch.full(..., dtype=torch.uint8 ...).view(FP8)   [master, pre-existing]
OK with warnings: .../muse_attention_decode/submission.py
```
`python3 -I -m py_compile`: OK (__pycache__ removed).

## Risks / what could fail
high
H100 -> H200: the kernel is math/latency-bound at the ceiling on the H100 (math-only 233 us at v32 = memory-ceiling time); per-SM structure transfers, absolute TB/s does not; projected H200 e2e 0.82-0.87 (untested; the H200 is held). Every earlier custom full-verify loop lost to FA3 on the judge (CLAIMS); this one is the first measured faster on the same SM. mma.sync e4m3 is emulated on sm_90a (the kernel uses wgmma for QK).

## Result (after the run)
- job `06c051723643462f` (check), status done, passed True, tests 16/16
- speedup 1.6154x, 95% CI [1.5911, 1.6396], score 2331.5390433187868 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/BEST.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel`
