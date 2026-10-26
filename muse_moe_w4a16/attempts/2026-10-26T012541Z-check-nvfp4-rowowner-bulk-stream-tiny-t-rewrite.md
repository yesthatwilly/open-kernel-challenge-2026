# muse_moe_w4a16 · 2026-10-26T012541Z · check · nvfp4-rowowner-bulk-stream-tiny-t-rewrite

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T211822Z-nvfp4-rowowner-bulk-stream-tiny-t-rewrite.md` (branch `hyp/muse_moe_w4a16/nvfp4-rowowner-bulk-stream-tiny-t-rewrite`).

- **Job:** `c08dbd737e29442d` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `f8aeac2c`

## Goal
Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1 bs=1, the bs=1 and bs=4 tests). Every other T runs master byte for byte. New code in its own load_inline module (muse_moe_w4a16_rowowner_v1), sm_90 instructions only (cp.async.bulk + mbarrier, mma.sync or CUDA-core FMA; no wgmma, no setmaxnreg), -std=c++20 in extra_cuda_cflags (LESSONS 159). WHY: decode_1 is the load furthest from its floor (46.4 vs 11.4 us); the Marlin pair is 35.4 us on ~56 MB of unique weight bytes = ~1.6 TB/s, while muse_bf16_proj's row-owner reduce-free M=1 stream (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc; bulk_kernel<16,8192,8,4,2> at 2 CTAs/SM) reached ~3.5 TB/s on m<=32 weight streams and won qkv_gate decode_1 1.16x. Every earlier small-M NVFP4 design kept padding to N16, smem staging for wgmma, split-K tickets or grid spins (LESSONS 67: none exceeded ~2 TB/s). DESIGN. (1) Prepare (untimed, lossless repack): per local expert, re-lay w13 so gate row i and up row i are adjacent (one 'pair row' = 2 x 4096 e2m1 codes + 2 x 256 e4m3 scales = 4608 B), and w2 rows as-is (each of the 4096 output rows = I/2 code bytes + I/16 scale bytes), each expert's rows contiguous so a CTA's span is one contiguous byte range; keep the per-expert global scale. NVFP4 stays NVFP4 (repack only). (2) Device-side work map (no host branch, graph-safe): kernel A (1 CTA, or folded into the w13 kernel's prologue) reads topk_ids, maps to local experts (master's expert_map), and writes the list of (expert, route rows) for <= 4*8 = 32 routes; CTAs of the w13 kernel each own a fixed contiguous span of pair rows of ONE active expert, spans sized so (active experts x I) / (2 x SMs) rows per CTA. (3) w13 kernel: one producer lane issues 1-D cp.async.bulk copies of the CTA's span into a deep smem ring (>= 64 KB in flight per SM, mbarrier full/empty), consumers dequantize with master's exact dequant8 bit trick (e2m1 -> bf16 via mask/shift + fma by scale*2^119) and FMA against the <= 4 token rows of x held in registers/smem (bf16x2 products, fp32 accumulate over full K; warp-shuffle reduce within the row owner warp only, no cross-CTA reduction), then silu(gate)*up in fp32 with bf16 rounding of gate and up first (SGLang parity, as master's silu_mul) and a bf16 h row write. (4) w2 kernel, launched after w13 with PDL: the same row-owner stream over w2 rows (K = I), each output element y[route, n] produced whole by its owner warp from h, bf16 out into master's cache3 layout so master's _post_norm_combine_kernel runs unchanged. No split-K, no atomics, no tickets, no spins (the bs=1 slow mode came from cross-CTA reduction; LESSONS 62, 151). (5) Consumer choice: implement CUDA-core FMA (variant F) and mma.sync m16n8k16 with tokens padded to 8 (variant M); pick per T in the prepare gate. (6) Gate: cold-L2 graph replays of the full T<=4 path vs master's path on skewed AND _wg_judge_ids routings at T=1 and T=4, require >= 1.02x on the worse routing (LESSONS 164), print kernel-only TB/s per kernel and one PATH CHOSEN line; a canary forces every active-expert count (1, 2, 8 local experts) and checks finiteness and relative L2 vs Marlin with the judge's 1.5% bound (not a self-imposed 1e-2; LESSONS 150). Instrument runs_ns for bimodality. Fallback: any build/canary/gate failure keeps master's T<=4 path. First measurement is an okc check (new bulk-copy/mbarrier producer can fault or hang).

## What changed, and where
Edited only `muse_moe_w4a16/submission.py` in the worktree (+1216 lines; master's code untouched apart from three hooks).

- New load_inline module `muse_moe_w4a16_rowowner_v1` (sm_90 only: cp.async.bulk, mbarrier, mma.sync, griddepcontrol; no wgmma/setmaxnreg; `-O3 -std=c++20 -lineinfo`, TORCH_CUDA_ARCH_LIST=9.0 around the build, arch flags, build time, regs/spills/occupancy printed). Kernels `rowowner_v1_w13<VAR>` and `rowowner_v1_w2<VAR>`, VAR 0 = F (CUDA-core fp32 FMA), 1 = M (mma.sync m16n8k16, tokens padded to 8).
- Repack (prepare, lossless, torch): 16-row units, 256-k records of 2304 B (2048 B codes in A-fragment order + 256 B e4m3 scales); w13 unit = 8 gate rows + the 8 matching up rows, so gate_i and up_i land in the same thread and silu fuses in registers; w2 unit = 16 rows. A numpy port of the pack plus a lane-level emulation of dequant, B layout and mma fragments matched a direct matmul to 1e-17 (scratchpad, not in the repo).
- Each warp owns whole units over full K with a private 4-stage mbarrier ring (lane 0 issues 1-D bulk copies, `__syncwarp` is the empty barrier); no cross-warp/CTA reduction, no atomics, no spins; all mbarrier waits bounded (error word 1). Grid = 2 x SMs CTAs of 4 warps; contiguous balanced unit spans.
- Work map is device-built inside each kernel prologue (shuffle/ballot over topk_ids), so no separate kernel A: it REPLACES master's `_align_small` (answers open Q1). Block 0 writes local_ids in master's [T, 8] layout; y goes to master's cache3 layout; `_post_norm_combine_kernel` unchanged.
- Zero local routes: n_act = 0, all spans empty, no ring is initialised, no wait; w2 warps still execute griddepcontrol.wait; output zeros (open Q2; canary has a zero-local case and requires exact zero).
- w2 PDL: weight copies are issued before `griddepcontrol.wait` (only h waits); w13 triggers launch_dependents after its prologue; smem is ~70 KB/CTA so w2 CTAs co-reside with w13 CTAs (open Q3).
- Span balance: units = n_act x (I/8) split base/rem across all warps, so at most one extra unit per warp, any n_act (open Q4).
- Gate prints F and M at T=1 and T=4 (open Q5); T=2,3 inherit T=4's variant only if canary-clean.

Review fixes applied: (1) selector = two interleaved rounds (41 cold-L2 graph replays each), both >= 1.03 on both local8 and skewed-4hot routings, candidate IQR <= 1.5x master's (master IQR floored at 0.25 us); master timed as kernel_fn with the flag off (its actually routed path, incl. F1/w2 atomic); (2) canary forces active counts 0, 1, 2, 8 and up to 32, finiteness and rel-L2 vs fp32 reference with the judge-side 1.2e-2 cap (and <= 2x master's own error), per-row check, zero-local exact zero, stage diagnostic (h vs fp32, y given kernel h); (3) runs_ns bimodality: p10/p90 printed per cell; (4) every variant's cold-L2 kernel-only us and TB/s printed even when master is kept; (5) compact PATH CHOSEN / gate lines are the last lines of prepare; (6) review wins over spec on LESSONS line numbers (cosmetic).

Other requirements: per-T flags in `state["ro"]` (not module global; state copy clones tensors, ext shared); err word per state; `-lineinfo`; per-launch flushed prints with sync for the first canary case; new phase has its own deadlines (start 150 s, canary 210 s, gate 235 s, copy check 250 s) and prints "skipped: deadline"; PDL falls back to off if the canary only passes without it; any failure leaves master's path for that T.

Deviations: no `_ab_flags`/NEW_CELL_FLAGS pattern exists in this file, so the flag is `state["ro"]["on"][T]`. Nothing was compiled or run (no GPU/nvcc here).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path = `_ro_forward` (torch.empty, two launches, Triton post-norm); no sync/.item/.cpu; branches on shape and prepare flags only.
- [x] Nothing derived from `data` cached: work map rebuilt on device each call; scratch allocated per call.
- [x] Weights repacked losslessly in prepare() only; NVFP4 codes and e4m3 scales unchanged, fp32 global scale x 2^7 exact.
- [x] Activations bf16 (x, h, y bf16 as Marlin path; gate/up rounded to bf16 before silu as SGLang).
- [x] n/a: attention rule, MoE problem; every routed local expert is computed in full.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (lint clean of errors).
- [x] All work inside the timed call; no output caching or replay.
- [x] New code in its own module; own per-state flags; miss falls back to master's exact path; bounded waits; distinct kernel symbols.

okc_lint: `OK with warnings` (only host-sync warnings in prepare-only code); `py_compile` ok; __pycache__ removed.

## Risks / what could fail
high
High risk: new bulk-copy producer and mbarrier ring can hang (bounded mbar_wait with an error word, as master). bf16 LESSONS 167: the row-owner layout serialises per-token FMA behind one owner, so its time grows with m (+0.6/+6/+25 us at m=1/16/32 at 1 CTA/SM) and the pure stream reached only 2.54 TB/s cold at 1 CTA/SM; hence T<=4 only and 2 CTAs/SM like the winning qkv instance. NVFP4 adds dequant ALU per byte (2 codes per byte vs 0.5 bf16 elements per byte): judge it by printed TB/s, and if variant F is ALU-bound at T=4, stage 2 restricts to T=1 or uses mma.sync. Stage 2 (if M2 met, M3 not): 2-expert interleaved spans and a w2 prefetch under w13 via PDL.

## Result (after the run)
- job `c08dbd737e29442d` (check), status done, passed True, tests 5/5
- speedup 1.3887x, 95% CI [1.3678, 1.4095], score 9684.478029666152 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.4020871870995952, "ci": [1.3810558792931014, 1.423118494906089], "job": "31c27e4b8fc84bd2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/combo-wg-occ2-v64-hot-interleave-align2.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/nvfp4-rowowner-bulk-stream-tiny-t-rewrite.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/nvfp4-rowowner-bulk-stream-tiny-t-rewrite.patch`; branch `hyp/muse_moe_w4a16/nvfp4-rowowner-bulk-stream-tiny-t-rewrite`
