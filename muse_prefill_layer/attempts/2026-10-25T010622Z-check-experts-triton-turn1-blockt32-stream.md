# muse_prefill_layer · 2026-10-25T010622Z · check · experts-triton-turn1-blockt32-stream

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T140154Z-experts-triton-turn1-blockt32-stream.md` (branch `hyp/muse_prefill_layer/experts-triton-turn1-blockt32-stream`).

- **Job:** `8d1a2efec67b4758` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `28e1d99`

## Goal
Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits in muse_prefill_layer/submission.py. (1) MARLIN_MAX_TOKENS from 1024 to 0 (the gate stays, so a control can restore it): every prefill shape and test (all T > 8) runs _experts_triton with stage 5's packed inline-PTX dequant; the Marlin bank and the e2e Triton-vs-Marlin check in prepare() stay as they are (Marlin becomes the check's reference only). (2) Replace the first EXPERT_CONFIGS entry (T <= 1024: turn_1 T=753 and the T=200/300/753 tests) with a bandwidth-shaped config: BLOCK_T=32 (turn_1 puts ~24 local rows on each expert, so one 32-wide token tile per expert and wgmma n32 instead of today's 64 with ~60% padding), BLOCK_W=128, BLOCK_K=128 (fixed by the nibble order), GROUP_SIZE_T=8, num_warps=4, num_stages=4 (~17 KB of codes+scales+activations per stage, ~68 KB in flight per CTA, so 3 CTAs per SM keep ~200 KB of weight bytes in flight per SM to stream HBM). At this T each weight element is dequantized once and the work is a weight stream: w13+w2 read 905 MB of NVFP4 codes and e4m3 scales (189 us at 4.8 TB/s). _check_expert_gemms already sizes its check to max(CHECK_ROWS, BLOCK_T) rows and compiles every entry, so the new config is checked in prepare(); the 4096 and first_turn entries are unchanged. If the prepare log shows spills or smem over budget, fall back to num_stages=3 and say so in the attempt file.

## What changed, and where
Edited only `muse_prefill_layer/submission.py` in the worktree (5 insertions, 2 deletions).
- `MARLIN_MAX_TOKENS` 1024 -> 0 (gate kept, comment added). Every prefill T runs `_experts_triton` with the packed dequant; the Marlin bank and the e2e Triton-vs-Marlin check are untouched. The older prose comment above the constant still describes the 1024 gate (not edited, comment only).
- First `EXPERT_CONFIGS` entry (T <= 1024): BLOCK_T 64 -> 32, num_stages 3 -> 4; BLOCK_W=128, BLOCK_K=128, GROUP_SIZE_T=8, num_warps=4 as specified. The 4096 and first_turn entries are unchanged.
- No deviation from the spec. The num_stages=3 fallback was not applied (no prepare log is available on this node); if the log shows spills or smem over budget, use 3.
- STALE BASE: claim reported stale_base: true. The worktree is on stage 6's commit (de7af5e), which predates prefill master (stack-tonight-five-prefill, 1.0705 check). Built on it as the spec says, not rebased. Master's parallel align and `_silu_mul_sorted_kernel` are absent, so the Marlin comparison (master 768.6/735.9 us at turn_1) is against a different surrounding pipeline; compare against stage 6's own run.
- Context: sibling stage 6 (experts-triton-packed-dequant-turn4, 1.0522, neutral) measured Triton at turn_4 as flat (0.995-1.005 on every shape). REWRITE_READ rates the Triton dequant line as ALU/lowering-bound (~5x floor at first_turn), and the review expects 2 CTAs/SM, not 3. Read n_regs/n_spills/shared for the BLOCK_T=32 variant before attributing.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the diff changes a constant and a config dict; kernel_fn path is unchanged, with no sync/alloc/value-dependent control flow (gate reads x.shape[0] only).
- [x] Nothing derived from `data` is cached across calls: no state added by the diff.
- [x] Weights: no repacking changes; NVFP4 stays NVFP4 (Triton engine dequantizes in-kernel).
- [x] Activations BF16 / KV FP8 / queries: unchanged by the diff.
- [x] Attention sees every key: attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint output:
```
warn  host-sync    L540, L582, L636, L641, L652, L661, L671: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
py_compile passed; __pycache__ removed.

## Risks / what could fail
medium
Risk: the dist=outliers test (T=200) can put far more than 32 rows on one expert; that only adds token tiles (moe_align pads per expert to BLOCK_T), so it is a correctness non-issue but not timed. At ~24 rows per expert the CTA count is 32 experts x 64 W tiles = 2048 for w13 and 1024 for w2, plenty to fill 132 SMs. If the turn_1 rows land between Marlin and the target, an ab-tune on num_stages (3 vs 5) or BLOCK_W 64 is the next step, not a revert.

## Result (after the run)
- job `8d1a2efec67b4758` (check), status done, passed True, tests 5/5
- speedup 0.9425x, 95% CI [0.9283, 0.9566], score 411044.6941669625 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-triton-packed-dequant-turn4.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/experts-triton-turn1-blockt32-stream.result.json`, `muse_prefill_layer/hypotheses/done/regression/experts-triton-turn1-blockt32-stream.patch`; branch `hyp/muse_prefill_layer/experts-triton-turn1-blockt32-stream`
