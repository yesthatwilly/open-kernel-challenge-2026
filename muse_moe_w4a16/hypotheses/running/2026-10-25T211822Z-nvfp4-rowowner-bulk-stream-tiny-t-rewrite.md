---
problem: muse_moe_w4a16
slug: nvfp4-rowowner-bulk-stream-tiny-t-rewrite
created: 2026-10-25T211822Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1
  bs=1, the bs=1 and bs=4 tests). Every other T runs master byte for byte. New code in its own
  load_inline module (muse_moe_w4a16_rowowner_v1), sm_90 instructions only (cp.async.bulk +
  mbarrier, mma.sync or CUDA-core FMA; no wgmma, no setmaxnreg), -std=c++20 in extra_cuda_cflags
  (LESSONS 159). WHY: decode_1 is the load furthest from its floor (46.4 vs 11.4 us); the Marlin
  pair is 35.4 us on ~56 MB of unique weight bytes = ~1.6 TB/s, while muse_bf16_proj's row-owner
  reduce-free M=1 stream (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc;
  bulk_kernel<16,8192,8,4,2> at 2 CTAs/SM) reached ~3.5 TB/s on m<=32 weight streams and won
  qkv_gate decode_1 1.16x. Every earlier small-M NVFP4 design kept padding to N16, smem staging
  for wgmma, split-K tickets or grid spins (LESSONS 67: none exceeded ~2 TB/s). DESIGN. (1)
  Prepare (untimed, lossless repack): per local expert, re-lay w13 so gate row i and up row i are
  adjacent (one 'pair row' = 2 x 4096 e2m1 codes + 2 x 256 e4m3 scales = 4608 B), and w2 rows
  as-is (each of the 4096 output rows = I/2 code bytes + I/16 scale bytes), each expert's rows
  contiguous so a CTA's span is one contiguous byte range; keep the per-expert global scale. NVFP4
  stays NVFP4 (repack only). (2) Device-side work map (no host branch, graph-safe): kernel A (1
  CTA, or folded into the w13 kernel's prologue) reads topk_ids, maps to local experts (master's
  expert_map), and writes the list of (expert, route rows) for <= 4*8 = 32 routes; CTAs of the w13
  kernel each own a fixed contiguous span of pair rows of ONE active expert, spans sized so
  (active experts x I) / (2 x SMs) rows per CTA. (3) w13 kernel: one producer lane issues 1-D
  cp.async.bulk copies of the CTA's span into a deep smem ring (>= 64 KB in flight per SM,
  mbarrier full/empty), consumers dequantize with master's exact dequant8 bit trick (e2m1 -> bf16
  via mask/shift + fma by scale*2^119) and FMA against the <= 4 token rows of x held in
  registers/smem (bf16x2 products, fp32 accumulate over full K; warp-shuffle reduce within the row
  owner warp only, no cross-CTA reduction), then silu(gate)*up in fp32 with bf16 rounding of gate
  and up first (SGLang parity, as master's silu_mul) and a bf16 h row write. (4) w2 kernel,
  launched after w13 with PDL: the same row-owner stream over w2 rows (K = I), each output element
  y[route, n] produced whole by its owner warp from h, bf16 out into master's cache3 layout so
  master's _post_norm_combine_kernel runs unchanged. No split-K, no atomics, no tickets, no spins
  (the bs=1 slow mode came from cross-CTA reduction; LESSONS 62, 151). (5) Consumer choice:
  implement CUDA-core FMA (variant F) and mma.sync m16n8k16 with tokens padded to 8 (variant M);
  pick per T in the prepare gate. (6) Gate: cold-L2 graph replays of the full T<=4 path vs
  master's path on skewed AND _wg_judge_ids routings at T=1 and T=4, require >= 1.02x on the worse
  routing (LESSONS 164), print kernel-only TB/s per kernel and one PATH CHOSEN line; a canary
  forces every active-expert count (1, 2, 8 local experts) and checks finiteness and relative L2
  vs Marlin with the judge's 1.5% bound (not a self-imposed 1e-2; LESSONS 150). Instrument runs_ns
  for bimodality. Fallback: any build/canary/gate failure keeps master's T<=4 path. First
  measurement is an okc check (new bulk-copy/mbarrier producer can fault or hang).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent Marlin pair 35.4 us (~56 MB, ~1.6 TB/s) of 46.4 us; floor 11.4 us. Milestones: M1 correct + prints kernel-only TB/s; M2 replaced kernels <= 35.4 us (>=1.6 TB/s); M3 >= 2.6 TB/s -> 17-22 us, decode_1 ~28-33 us, +5.5-8% headline"
  - shapes: bs=8 *
    direction: flat
    why: T>4 runs master
  - shapes: "bs=1[0-9]* *"
    direction: flat
    why: T>4 runs master
  - shapes: bs=2* *
    direction: flat
    why: T>4 runs master
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
prior: "cuda-fused-split-i-moe-tiny-t: that queued design keeps the wgmma rega consumer (N16 padding, smem B slab), the measured ~2 TB/s limit that only tied Marlin at T=1 (tiny-m-bucket16 46.0 vs 45.5 us); this changes the consumer (CUDA-core/mma.sync on a register-resident x) and the load path (1-D bulk row spans, no TMA tiles), keeps the w13->w2 boundary with no cross-CTA reduction and no spins (unlike cuda-fused-moe-bs1-persistent-*, bimodal from grid spins)"
rewrite_stage: 1
replaces: Marlin w13 + silu_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2) at 1 <= num_tokens <= 4
review: proceed
status: running
review_at: 2026-10-25T212523Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212549Z
base_sha: 641f0e0099c84be50a9f8705e113a62388a5847a
base_blob: 35b8a851e4ef5fa8fd66b12d09caac88b7ea7615
branch: hyp/muse_moe_w4a16/nvfp4-rowowner-bulk-stream-tiny-t-rewrite
worktree: .worktrees/muse_moe_w4a16/nvfp4-rowowner-bulk-stream-tiny-t-rewrite
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T011350Z
impl_commit: f8aeac2ce84e32fc31c5c54ff673fa9cf8955c2f
submission_sha256: 006510c848cd58ef8832029b1c7b00134f4e0a90c7f19a1668af58f08f6811a8
norm_sha256: 716518602fecdcb24fb5cebb09d1626c452a4ead420bedc11cc986e938285f3c
submission_blob: 9ef29f6f02cb52b468933ae16626fb37854c927e
lint_warnings: 112
---

# nvfp4-rowowner-bulk-stream-tiny-t-rewrite

## Change

Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1 bs=1, the bs=1 and bs=4 tests). Every other T runs master byte for byte. New code in its own load_inline module (muse_moe_w4a16_rowowner_v1), sm_90 instructions only (cp.async.bulk + mbarrier, mma.sync or CUDA-core FMA; no wgmma, no setmaxnreg), -std=c++20 in extra_cuda_cflags (LESSONS 159). WHY: decode_1 is the load furthest from its floor (46.4 vs 11.4 us); the Marlin pair is 35.4 us on ~56 MB of unique weight bytes = ~1.6 TB/s, while muse_bf16_proj's row-owner reduce-free M=1 stream (combo-rowowner-reduce-free-m1-qkv-o-plus-naive, 741072dc; bulk_kernel<16,8192,8,4,2> at 2 CTAs/SM) reached ~3.5 TB/s on m<=32 weight streams and won qkv_gate decode_1 1.16x. Every earlier small-M NVFP4 design kept padding to N16, smem staging for wgmma, split-K tickets or grid spins (LESSONS 67: none exceeded ~2 TB/s). DESIGN. (1) Prepare (untimed, lossless repack): per local expert, re-lay w13 so gate row i and up row i are adjacent (one 'pair row' = 2 x 4096 e2m1 codes + 2 x 256 e4m3 scales = 4608 B), and w2 rows as-is (each of the 4096 output rows = I/2 code bytes + I/16 scale bytes), each expert's rows contiguous so a CTA's span is one contiguous byte range; keep the per-expert global scale. NVFP4 stays NVFP4 (repack only). (2) Device-side work map (no host branch, graph-safe): kernel A (1 CTA, or folded into the w13 kernel's prologue) reads topk_ids, maps to local experts (master's expert_map), and writes the list of (expert, route rows) for <= 4*8 = 32 routes; CTAs of the w13 kernel each own a fixed contiguous span of pair rows of ONE active expert, spans sized so (active experts x I) / (2 x SMs) rows per CTA. (3) w13 kernel: one producer lane issues 1-D cp.async.bulk copies of the CTA's span into a deep smem ring (>= 64 KB in flight per SM, mbarrier full/empty), consumers dequantize with master's exact dequant8 bit trick (e2m1 -> bf16 via mask/shift + fma by scale*2^119) and FMA against the <= 4 token rows of x held in registers/smem (bf16x2 products, fp32 accumulate over full K; warp-shuffle reduce within the row owner warp only, no cross-CTA reduction), then silu(gate)*up in fp32 with bf16 rounding of gate and up first (SGLang parity, as master's silu_mul) and a bf16 h row write. (4) w2 kernel, launched after w13 with PDL: the same row-owner stream over w2 rows (K = I), each output element y[route, n] produced whole by its owner warp from h, bf16 out into master's cache3 layout so master's _post_norm_combine_kernel runs unchanged. No split-K, no atomics, no tickets, no spins (the bs=1 slow mode came from cross-CTA reduction; LESSONS 62, 151). (5) Consumer choice: implement CUDA-core FMA (variant F) and mma.sync m16n8k16 with tokens padded to 8 (variant M); pick per T in the prepare gate. (6) Gate: cold-L2 graph replays of the full T<=4 path vs master's path on skewed AND _wg_judge_ids routings at T=1 and T=4, require >= 1.02x on the worse routing (LESSONS 164), print kernel-only TB/s per kernel and one PATH CHOSEN line; a canary forces every active-expert count (1, 2, 8 local experts) and checks finiteness and relative L2 vs Marlin with the judge's 1.5% bound (not a self-imposed 1e-2; LESSONS 150). Instrument runs_ns for bimodality. Fallback: any build/canary/gate failure keeps master's T<=4 path. First measurement is an okc check (new bulk-copy/mbarrier producer can fault or hang).

## Expected

- `bs=1 *` faster: incumbent Marlin pair 35.4 us (~56 MB, ~1.6 TB/s) of 46.4 us; floor 11.4 us. Milestones: M1 correct + prints kernel-only TB/s; M2 replaced kernels <= 35.4 us (>=1.6 TB/s); M3 >= 2.6 TB/s -> 17-22 us, decode_1 ~28-33 us, +5.5-8% headline
- `bs=8 *` flat: T>4 runs master
- `bs=1[0-9]* *` flat: T>4 runs master
- `bs=2* *` flat: T>4 runs master

## Risk

High risk: new bulk-copy producer and mbarrier ring can hang (bounded mbar_wait with an error word, as master). bf16 LESSONS 167: the row-owner layout serialises per-token FMA behind one owner, so its time grows with m (+0.6/+6/+25 us at m=1/16/32 at 1 CTA/SM) and the pure stream reached only 2.54 TB/s cold at 1 CTA/SM; hence T<=4 only and 2 CTAs/SM like the winning qkv instance. NVFP4 adds dequant ALU per byte (2 codes per byte vs 0.5 bf16 elements per byte): judge it by printed TB/s, and if variant F is ALU-bound at T=4, stage 2 restricts to T=1 or uses mma.sync. Stage 2 (if M2 met, M3 not): 2-expert interleaved spans and a w2 prefetch under w13 via PDL.

## Milestones

Rewrite stage 1 of `Marlin w13 + silu_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2) at 1 <= num_tokens <= 4`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212523Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is DIAGNOSIS action 2 almost word for word. It is the first small-M NVFP4 design with no N16 padding, no wgmma smem staging, no split-K or tickets and no grid spins, and its ~2.6-3.3 TB/s target is physically reachable at T=1. Run it as the stage-1 okc check. Fix two things first: the gate threshold (new lesson: two interleaved rounds >= 1.03 with an IQR bound) and the missing zero-local-route canary.

### Justification
A. Mechanism. decode_1 is 46.4 us against an 11.4 us floor, and the Marlin pair is 35.4 us (85.8%) of it. The ~56 MB checks out against TASK: 8 local experts x 3 x I(1024) x 4096 x 0.5625 B is about 56.6 MB, which matches the 11.4 us floor at 4.8 TB/s, so the judge routing puts all 8 routes local at T=1. Repacking into 4608 B gate/up pair rows (2 x 2048 B codes + 2 x 256 B e4m3 scales) is lossless and lets one 1-D cp.async.bulk per span stream contiguous bytes, with no tensor maps. That also avoids the g_maps per-state failure class. ALU budget: at 3 TB/s on 132 SMs, each SM needs ~13 B/cycle, about 23 FP4 elements/cycle. At T=1, dequant plus one FMA per element is well under 1/4 of issue, so variant F is feasible. At T=4 the fp32-accumulate FMAs reach ~90+/cycle/SM out of 128 lanes plus dequant, so F may be ALU-bound there. The design already says variant M or a T=1-only restriction is the fallback. Reduce-free row ownership removes the cross-CTA reduction that caused the bs=1 slow mode (LESSONS: the w2 atomic cut it from 44/100 to 10/100). It uses no spins, unlike the dead cuda-fused-moe-bs1-persistent-* line. Prior: cuda-fused-split-i-moe-tiny-t (queued) keeps the wgmma rega consumer, and DIAGNOSIS holds that consumer at T<=7 (tiny-m-bucket16 46.0 vs 45.5 us). cuda-nvfp4-swapab-mma-persistent-small-m-inproc used padded swap-AB mma tiles and persistent scheduling and was 1.7-2x slower. The difference from both is real: this changes the consumer (register-resident x, no N16 pad) and the load path (contiguous row spans). It is not on any do-not-try list, and DIAGNOSIS explicitly says a stage on a new consumer is not a re-run. Expected directions match: only bs=1 moves, and bs=4 is a test, not scored. The M3 numbers (17-22 us, decode_1 ~28-33 us, +5.5-8% headline) follow from one of six equally weighted loads going 46 -> ~30 us. WEAKEST LINK: the 3.5 TB/s anchor is a bf16 result (0.5 elements per byte). LESSONS 172 says the 1-CTA/SM row-owner pure stream reached only 2.54 TB/s cold, and only the 2-CTA/SM qkv instance reached ~3.5. NVFP4 does ~3.6x more consumer work per byte (~1.78 elements/byte with scales), and every NVFP4 kernel so far has capped at 1.5-2.1 TB/s because of dequant and consumer issue. Transfer is therefore unproven. Also, ~56 MB at ~3 TB/s is an ~19 us kernel pair, short enough that ramp-up, the w13->w2 drain and tail effects can eat a large share. M2 (beating 35.4 us) is likely; M3 (>= 2.6 TB/s) is the real bet. Minor: the cited LESSONS line numbers are stale (C++20 is line 164, not 159; row-owner is 172, not 167).

### Confounders on the current kernel
- combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 (master) runs w13 use_atomic_add at T=1 (W13_ATOMIC_MAX_TOKENS=1, 0.949 when live, routing-dependent). The prepare gate compares against master's T=1 path, so whether F1 went live in that process sets the bar. The 35.4 us incumbent comes from the board branch, where F1 gated off on skewed routing. Read the replaced kernels against master's own T=1 Marlin rows in the same profile, not against 35.4.
- w2-marlin-atomic-add-small-m (landed, T<=32) removed most of the bs=1 slow mode. The replacement drops it at T<=4, so any bimodality in runs_ns now belongs to the new kernels. That makes runs_ns instrumentation essential, as the design says.
- post-expert-norm-triton-small-m / _post_norm_combine_kernel: kept unchanged, but it reads y[t, s] only where local_ids[t, s] >= 0. Kernel A must emit local_ids in master's [T, 8] layout ([-1] for remote) and write y rows at route index t*8+s. A route-order mismatch is a silent correctness bug, not a perf confounder.
- triton-map-align-on-block-m-restack (_align_small, 1.5 us at decode_1): it is unclear whether kernel A replaces it or runs beside it. If both run, ~1.5 us of front-end stays and the decode_1 row understates the GEMM win. Attribute by kernel time, not by row.

### Interference and A/B needs
No companion. This is a rewrite stage, judged on the replaced kernels' own time in the profile. Both milestones that matter (M1 correct; M2 replaced kernels <= incumbent) can be read from one run, because the prepare gate prints kernel-only TB/s and timings for both variants against master's path, and the profile shows the new kernel names when live. The landed items above change only the incumbent's bar (F1 at T=1), and that bar is measured in the same process. A revert companion would not disambiguate anything the gate print does not already show. The real risk to readability is the tiny-m-bucket16 failure mode: the gate keeps master and the check measures master. So the prepare log must print every variant's T=1 and T=4 cold-L2 kernel-only us and TB/s even when PATH CHOSEN = MASTER, so the next stage can be minted from those numbers.

### Open questions
- Does kernel A replace master's _align_small at T<=4 or run beside it? It must produce local_ids in master's [T, 8] layout for _post_norm_combine_kernel.
- Zero local routes: does the canary force a T=1 or T=4 routing with no local expert? The output must be zeros. _post_norm_combine_kernel stores zeros, but the w13/w2 kernels must not fault or hang on an empty work map, and the mbarrier rings must not deadlock when a CTA has no span. The canary lists counts 1, 2 and 8 only.
- w2 dependency: will the w2 kernel's producer issue its weight bulk copies before griddepcontrol.wait? w2 weights do not depend on h. Prefetching them under the w13 tail is the cheapest stage-1 lever. Check that smem lets w2 CTAs co-reside with w13 CTAs at 2 CTAs/SM, or the PDL overlap is nil.
- Span balance: with 264 CTAs (2 per SM) and 8 active experts at T=1, 33 CTAs per expert x 31 pair rows covers 1023 of 1024 rows. How are remainder rows and non-divisible active-expert counts (3, 5, 7) assigned without a tail CTA doing twice the work?
- If variant F is ALU-bound at T=4 and M is chosen, padding 4 tokens to 8 doubles the B-operand work. Does the gate print both variants at both T so stage 2 can split T=1 (F) from T=2..4 (M)?


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn path = `_ro_forward` (torch.empty, two launches, Triton post-norm); no sync/.item/.cpu; branches on shape and prepare flags only.
- [x] Nothing derived from `data` cached: work map rebuilt on device each call; scratch allocated per call.
- [x] Weights repacked losslessly in prepare() only; NVFP4 codes and e4m3 scales unchanged, fp32 global scale x 2^7 exact.
- [x] Activations bf16 (x, h, y bf16 as Marlin path; gate/up rounded to bf16 before silu as SGLang).
- [x] n/a: attention rule, MoE problem; every routed local expert is computed in full.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (lint clean of errors).
- [x] All work inside the timed call; no output caching or replay.
- [x] New code in its own module; own per-state flags; miss falls back to master's exact path; bounded waits; distinct kernel symbols.

okc_lint: `OK with warnings` (only host-sync warnings in prepare-only code); `py_compile` ok; __pycache__ removed.

## Pre-flight audit

Verdict: dispatch-with-fixes. Emulation: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/emu/rowowner_emu.py (log run_full.log beside it).

- Fix 1 (gate that always fails): the spec and the code assumed I = 1024, but reference.D_EXPERT = 4096 (the 11.4 us floor is 2 active local experts x 28 MB at T=1, not 8 x 7 MB). w2 stages h for T rows per warp, 4 warps: 168704 B of dynamic smem at T=4, but ro_init raised the limit to only 120 KB and sized the host launch for MAXT rows, so every ro_w2 launch raised (invalid argument), the canary failed for every variant and T and the path never ran. Now the limit is 192 KB, the w2 smem and per-warp stride are sized by the actual T (T=1: 70016 B, 3 CTAs/SM and PDL co-residence with w13; T=4: 1 CTA/SM), and ro_init's occupancy probe uses the real 2056-word row.
- Fix 2 (variance veto): RO_IQR_FLOOR_US 0.25 -> 0.5 us (CUDA event timer resolution).
- Fix 3 (gate that may fail a correct kernel): RO_REL_FLOOR 2e-3 -> 4e-3. The emulation shows the chain's own bf16 roundings (gate, up, h, y) give ~3.3e-3 rel-L2 vs an un-rounded fp32 reference, above the old floor, so a lucky master could veto it. The 1.2e-2 cap is unchanged.
- Verified by reading and emulation: mbarrier counts (init 1, one arrive.expect_tx of 2304 B per fill, parity (q/4)&1, every wait bounded by 2^26 cycles then error word 1); w2 pre-wait work touches only w2 records, topk_ids and its own smem, h is read after griddepcontrol.wait; numpy emulation at real shapes (H = I = 4096) of the pack, dequant, A/B/C fragments (M), FMA + xor-shuffle (F), staging, spans (every active count 0..32, grids 228..288 partition exactly), local_ids layout and y routing matched an fp64 reference to 1e-16 for both variants at T=1..4, including 0 active experts (no span, no ring, zero output). Zero-local: master's own path handles all-remote (the F1 and align self-checks run it) and its post-norm stores exact zeros, so the exact-zero canary can pass.
- Not changed: deadlines. Master's prepare is 60-105 s, this phase starts at ~105 s, compile ~40-60 s, canary budget to 210 s; a miss lands on master with a "skipped: deadline" line.

Residual risk: nothing was compiled or run (no nvcc); w2 at T=4 runs 1 CTA/SM (36 KB in flight per SM), so T=4 may fail the 1.03 gate and keep master, which does not affect the scored T=1.
