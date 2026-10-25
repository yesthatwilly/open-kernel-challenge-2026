---
problem: muse_prefill_layer
slug: experts-turn4-cuda-dequant-producer-ss-wgmma
created: 2026-10-25T042410Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a warp-specialized grouped GEMM with an in-kernel dequant-producer warpgroup and SS wgmma"
change: >
  Built on master (a81d19c). At turn_4 the Marlin pair is 2148/2115 us (64%/43% of the
  sliding/full turn_4 shapes) against a 306 us math floor (~3012 local routes x 2 x 50.3 MFLOP),
  ~7x; the Triton register-LHS engine only tied it (2193/2254 us with the padding removed, LESSONS
  03:41Z), and the per-element dequant ALU was shown not to bind (packed-asm dequant moved
  nothing). Both incumbents dequantize and multiply in the same warps. New kernel, a new
  decomposition: tokens on the wgmma M side, weights dequantized ONCE per (expert, N tile, K step)
  by a dedicated producer warpgroup into shared memory, never to global memory (in-kernel dequant,
  per the standing answer; no BF16 copy of any weight exists outside smem). CTA tile 128 routes x
  256 output columns, BLOCK_K 64, one expert per CTA tile via master's _align_parallel(local_ids,
  0, LOCAL_EXPERTS, 128) (local routes only; ~94 routes per expert at T=3012, so most experts are
  one token tile and each weight is dequantized ~1.3 times in all). Warpgroup 0 (producer, 128
  threads): cp.async.bulk the packed E2M1 bytes and e4m3 scales of the [256 x 64] weight tile into
  a 4-stage raw ring, cp.async-gather the 128 activation rows x 64 k (bf16, rows from sorted_ids)
  into a 128B-swizzled A stage, then dequantize the raw weight stage (nibble -> bf16x2, times
  scale x global scale) into a 128B-swizzled bf16 B stage (2 stages, 32 KB each), arriving on an
  mbarrier. Warpgroups 1-2 (consumers): SS wgmma m64n256k16, 64 routes each, fp32 accumulators
  (128 regs), setmaxnreg 232 (producer 160, it needs registers to dequantize); epilogue writes
  bf16 rows to cache1 (w13) / y (w2) at their sorted slots, skipping padded rows. Keep
  _silu_mul_sorted between the two GEMMs and the post-expert weighted sum unchanged. In
  _experts_marlin: when 1024 < T <= 4096 and state['ssdq_ok'], use the new pair; else Marlin.
  Print per-CTA producer-wait vs consumer-wait cycle counters from one prepare launch, so a
  dequant-bound producer is visible. Build rules (LESSONS 2026-10-25): torch load_inline with
  os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before the build (restored after) and only
  -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3 -lineinfo in extra_cuda_cflags; all
  wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device
  error word and returns; print the nvcc flags, compile seconds, and regs / local bytes / smem /
  binaryVersion per kernel. In-process only (never a child process: the rules reject submissions
  that start processes). Prepare canary: one tiny launch per kernel, each followed by
  torch.cuda.synchronize() and a flushed print, then a finiteness check that names the first
  non-finite tensor; self-checks must force every tile/bucket shape (an all-one-expert routing and
  a skewed one, not only random routings) and compare against an fp32 dequant reference with a
  bound calibrated on the incumbent's own error (not a fixed 1e-2 against another approximate
  kernel). Hard gate: interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of
  the new kernels vs the incumbent at the scored T; ON only at <= 0.92x; print both times and PATH
  CHOSEN and assert the chosen one is the faster. Any failure prints one OFF: <reason> line and
  master runs unchanged. The first measurement must be an okc check, never a scored submit. Gate
  shape: T=3012 soft routing, new w13+w2 vs master's Marlin pair.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: Marlin pair 2148/2115 us -> <= 1400 us (Milestone M2), floor 306 us; turn_4 sliding ~3319 -> ~2570 us (-22%), full ~4849 -> ~4100 us (-15%)
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: T=753 <= 1024 keeps Marlin
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps the Triton engine
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)
review: proceed
status: running
review_at: 2026-10-25T043310Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T043310Z
base_sha: 8c0b24db143ab6bf6ff8765c64729c63abac3138
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-turn4-cuda-dequant-producer-ss-wgmma
worktree: .worktrees/muse_prefill_layer/experts-turn4-cuda-dequant-producer-ss-wgmma
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T044809Z
impl_commit: 44ca9c221f6b2daad5320772acc334d5788422e7
submission_sha256: 5e18a107a67cf8b1369e76e334ecc3d3e474e53a02e7989ec327ef3fba894aa0
norm_sha256: d657ebd0be595907a6128d52dfe3bd273810fa729d21269af71d7fce3e387454
submission_blob: d5e8e93c9f001ce31b439299ad78db8bf3b8f614
lint_warnings: 32
attempt_file: 2026-10-25T044809Z-check-experts-turn4-cuda-dequant-producer-ss-wgmma.md
prepared_at: 2026-10-25T044809Z
job: 40541db65f1b4ec6
dispatched_at: 2026-10-25T044817Z
---

# experts-turn4-cuda-dequant-producer-ss-wgmma

## Change

Built on master (a81d19c). At turn_4 the Marlin pair is 2148/2115 us (64%/43% of the sliding/full turn_4 shapes) against a 306 us math floor (~3012 local routes x 2 x 50.3 MFLOP), ~7x; the Triton register-LHS engine only tied it (2193/2254 us with the padding removed, LESSONS 03:41Z), and the per-element dequant ALU was shown not to bind (packed-asm dequant moved nothing). Both incumbents dequantize and multiply in the same warps. New kernel, a new decomposition: tokens on the wgmma M side, weights dequantized ONCE per (expert, N tile, K step) by a dedicated producer warpgroup into shared memory, never to global memory (in-kernel dequant, per the standing answer; no BF16 copy of any weight exists outside smem). CTA tile 128 routes x 256 output columns, BLOCK_K 64, one expert per CTA tile via master's _align_parallel(local_ids, 0, LOCAL_EXPERTS, 128) (local routes only; ~94 routes per expert at T=3012, so most experts are one token tile and each weight is dequantized ~1.3 times in all). Warpgroup 0 (producer, 128 threads): cp.async.bulk the packed E2M1 bytes and e4m3 scales of the [256 x 64] weight tile into a 4-stage raw ring, cp.async-gather the 128 activation rows x 64 k (bf16, rows from sorted_ids) into a 128B-swizzled A stage, then dequantize the raw weight stage (nibble -> bf16x2, times scale x global scale) into a 128B-swizzled bf16 B stage (2 stages, 32 KB each), arriving on an mbarrier. Warpgroups 1-2 (consumers): SS wgmma m64n256k16, 64 routes each, fp32 accumulators (128 regs), setmaxnreg 232 (producer 160, it needs registers to dequantize); epilogue writes bf16 rows to cache1 (w13) / y (w2) at their sorted slots, skipping padded rows. Keep _silu_mul_sorted between the two GEMMs and the post-expert weighted sum unchanged. In _experts_marlin: when 1024 < T <= 4096 and state['ssdq_ok'], use the new pair; else Marlin. Print per-CTA producer-wait vs consumer-wait cycle counters from one prepare launch, so a dequant-bound producer is visible. Build rules (LESSONS 2026-10-25): torch load_inline with os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before the build (restored after) and only -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3 -lineinfo in extra_cuda_cflags; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error word and returns; print the nvcc flags, compile seconds, and regs / local bytes / smem / binaryVersion per kernel. In-process only (never a child process: the rules reject submissions that start processes). Prepare canary: one tiny launch per kernel, each followed by torch.cuda.synchronize() and a flushed print, then a finiteness check that names the first non-finite tensor; self-checks must force every tile/bucket shape (an all-one-expert routing and a skewed one, not only random routings) and compare against an fp32 dequant reference with a bound calibrated on the incumbent's own error (not a fixed 1e-2 against another approximate kernel). Hard gate: interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of the new kernels vs the incumbent at the scored T; ON only at <= 0.92x; print both times and PATH CHOSEN and assert the chosen one is the faster. Any failure prints one OFF: <reason> line and master runs unchanged. The first measurement must be an okc check, never a scored submit. Gate shape: T=3012 soft routing, new w13+w2 vs master's Marlin pair.

## Expected

- `*load=turn_4*` faster: Marlin pair 2148/2115 us -> <= 1400 us (Milestone M2), floor 306 us; turn_4 sliding ~3319 -> ~2570 us (-22%), full ~4849 -> ~4100 us (-15%)
- `*load=turn_1*` unchanged: T=753 <= 1024 keeps Marlin
- `*load=first_turn*` unchanged: T=16384 keeps the Triton engine

## Risk

Milestones on the replaced rows: M0 sm_90a build, 0 B local, canary and every forced routing finite; M1 prepare A/B <= 1.0x Marlin; M2 <= 0.92x so it ships, judge rows < 1500 us. Kill: A/B > 1.0 with the producer-wait counter dominant -> dequant-producer design is bound by one warpgroup's ALU; stage 2 would then split dequant across two producer warpgroups. If consumer-wait dominates, the next stage deepens the raw ring. No relation to the held per-call BF16 dequant family: nothing is materialized in global memory.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T043310Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound new decomposition for the turn_4 Marlin pair (2148/2115 us vs 306 us floor; a <=1400 us pair is ~1.25x on the turn_4 load, ~+7% headline) and compliant with the in-kernel-dequant-only rule (bf16 lives only in smem). Independent of the RS register-A hazard, so it need not wait for moe s3, but the stated setmaxnreg budget is invalid and the producer is overloaded; fix both before building.

### Justification
Mechanism plausible: at T=3012 ~94 local routes per expert, a 128-route M tile wastes ~1.36x MMA but dequantizes each weight ~1.3x total, vs Triton register-LHS which re-dequantizes per token tile and per warp, and Marlin's mma.sync. Per K step (64) a CTA does 128x256x64 MACs (~1000 tensor clocks on two consumer warpgroups) and the producer dequantizes 256x64 = 16K elements (~400-600 issue clocks at 128 threads plus 32 KB of st.shared, ~256 clocks of smem bandwidth) - feasible but tight. Weakest link 1 (hard bug): setmaxnreg 232 for 2 consumer warpgroups + 160 for the producer = 2*128*232 + 128*160 = 79,872 registers > 65,536 per SM; setmaxnreg.inc would stall forever. Valid splits are e.g. 232/40 (no room to dequantize), 200/104, or no setmaxnreg at all (all at the 168 compile cap, which SS consumers at ~140-150 regs fit). Weakest link 2: the producer warpgroup is asked to do the bulk weight copy, a cp.async row-gather of 128x64 bf16 activations, AND the dequant + swizzled st.shared - three jobs on one warpgroup is the likeliest bottleneck. Weakest link 3 (correctness): B stages written by st.shared (generic proxy) and A stages written by cp.async are read by wgmma through the async proxy; a fence.proxy.async.shared::cta after the producer's writes and before the mbarrier arrive is mandatory, otherwise stale/torn tiles appear nondeterministically.

### Confounders on the current kernel
- stack-tonight-five-prefill: provides _align_parallel (used here at bsm 128) and _silu_mul_sorted (kept between the GEMMs). The incumbent the gate times is Marlin with bsm 64 and the same align/silu, so no confound.
- experts-hybrid-marlin-turn-triton-first-turn: MARLIN_MAX_TOKENS gate; the test case q_len 4096 (T=4096) falls in 1024<T<=4096 and will exercise the new path in the judge's tests - good coverage, but the self-check must include T=4096.
- Queued turn-align-remap-on-side-stream-under-pooler moves the same align call at T<=4096 to the side stream with Marlin's bsm; not a measurement confound (separate base), but a merge conflict if both land: this path needs bsm 128.

### Interference and A/B needs
No companion: rewrite stage 1 with a hard prepare gate and printed producer-wait/consumer-wait counters; a gated-off run is master and the counters say which side binds for stage 2.

### Open questions
- Does the producer-wait counter dominate at T=3012? If so stage 2 splits dequant across two producer warpgroups (then setmaxnreg can't be used at all at 384+ threads unless consumers drop).
- Is the w2 GEMM (K=4096 over the silu output) better served by having _silu_mul_sorted write cache2 directly in sorted-row order, so w2's A is a plain TMA tile with no gather?
- Per-expert route counts under the soft distribution at T=3012: how many experts exceed 128 (second token tile => second dequant of all its weights)?


## Implementation notes

Edited only `muse_prefill_layer/submission.py` in the worktree (+886/-3): `import os, time`; a new block before `kernel_fn` (CUDA source, build, tracer, self-check, gate, `_experts_ssdq`, `_silu_rows` Triton kernel); `_experts` routes `SSDQ_MIN_TOKENS (1024) < T <= 4096` to `_experts_ssdq` when `state["ssdq_ok"]`, else Marlin; `prepare()` calls `_ssdq_setup` last (budget guard: skipped if prepare already used 170 s). Everything else is master. Not run (no GPU/nvcc here).

**The review wins over the spec, applied:**
- setmaxnreg dropped entirely (the 232/160 split needs 79,872 regs > 65,536). All 3 warpgroups run at the 168-register `__launch_bounds__(384,1)` cap; ptxas regs / local bytes are printed by `ssdq_info`.
- The activation gather is off the producer: `ssdq_gather` pre-gathers x rows into sorted order (zeros for padding), A is a plain 2-D TMA box (128B swizzle, box 64 k x 128 rows). For w2, A is the silu output already in sorted-row order (answers the review's open question 2): w13's epilogue writes c1 in sorted-row space, `_silu_rows_kernel` (Triton) does silu*up row to row, w2 TMA-reads c2 directly and scatters to y[sorted id].
- `fence.proxy.async.shared::cta` after each producer thread's st.shared and before its mbarrier arrive (full barrier count 129 = 128 producer threads + the TMA thread's expect_tx arrive).
- Every mbarrier wait is bounded (2^26 cycles, then error word 1 and the thread stops waiting); error word 2 = non-sm_90a image ran.
- Self-check includes T=4096 (skewed to 4 experts and random), plus T=1506, 3012 (random) and T=1100 with one local expert, all vs an fp32 dequant reference (original-order weights), bound = max(2 x Marlin's own error vs the same reference, 8e-3), worst row <= 5 x bound, error word 0.
- Gate: turn_4 only. Cold-L2 (256 MB flush) CUDA-graph replays, 9 alternations with swapped order, of the whole expert block (align, GEMMs, silu, weighted sum) at T=3012 soft routing, new vs `_experts_marlin`; ON only at <= 0.92x; prints both medians, `PATH CHOSEN`, and asserts the chosen path is the faster. Any failure prints one `OFF: <reason>` line.

**Deviations from the spec (reasoned):**
- No cp.async.bulk raw ring: the producer prefetches the next 128-k chunk's packed codes (8 x 16 B per thread) and e4m3 scales (8 x 8 B) into registers with `ld.global.nc` one chunk ahead, dequantizes into the swizzled bf16 B stage, so no raw smem and no second mbarrier ring. 4 stages of 48 KB (A 16 KB TMA + B 32 KB) = 193 KB dynamic smem. One 128-k chunk produces two 64-k stages (low nibbles = k0..k0+63, high = k0+64..), matching prepare's nibble order, so each thread's 16 bytes are exactly one scale group per half.
- The gate times the whole expert block, not only the two GEMMs, so the extra gather and silu launches are paid for.
- The path needs master's parallel align (`align_par_ok`); `_align_parallel` at block 128 is not in master's own align self-check, so the e2e self-check is what validates it. When `zero_nonlocal_w2` is True, y is `torch.zeros` (the weighted sum reads remote rows), else `torch.empty`.
- One CTA per (128 sorted rows, 256 columns), non-persistent, grid (N/256, n_blocks): blocks past the used count exit at once (block_experts = -1). Epilogue stores straight from registers (4-byte bf16x2 stores), no smem staging.
- Counters: one profiled prepare launch prints producer-wait (on empty stages) vs consumer-wait (on full stages) cycles per CTA for w13 and w2.

Build: `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored, `extra_cuda_cflags = -O3 -std=c++20 -lineinfo -gencode=arch=compute_90a,code=sm_90a`, wgmma asm under `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)` with an `#else` that sets error word 2; arch flags, compile seconds, regs / local bytes / smem / occupancy / binaryVersion printed. In-process, no child process, plain TMA (no createpolicy / cache-hint variants).

Biggest risk: nothing here has compiled or run. A first-launch fault (descriptor or swizzle mistake in the hand-written B tile, wgmma operand syntax) would poison the CUDA context past the try/except, so the first measurement must be an `okc check`. Secondary: 168-register cap with 128 accumulators (watch local_bytes), and a producer that cannot keep up (the counters say).

## Rules check

- [x] CUDA-graph capturable: `_experts_ssdq` and its kernels allocate only by shape, no host syncs, `ssdq_gemm` encodes the tensor map on the host (no stream work); every `.item()` / synchronize is in prepare-only code (lint warnings are all "outside kernel_fn's call graph"); the tracer is passed only from prepare.
- [x] Nothing derived from `data` is cached: `state` holds the extension, an error word and counters; x, ids and buffers are per call.
- [x] Weights: no new weight copy; the kernel reads the existing NVFP4 bank (nibble-reordered in prepare, lossless); bf16 weight tiles exist only in shared memory for one stage.
- [x] Activations bf16 throughout (gathered rows, c1, c2, y); fp32 accumulation; same bf16 rounding points as the Marlin path.
- [x] Attention: n/a, only the expert GEMMs at 1024 < T <= 4096 change.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (diff grep for subprocess / multiprocessing / fork / autotune: none; load_inline runs in-process).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a-only PTX is inside `__CUDA_ARCH_FEAT_SM90_ALL` guards with an error-word `#else`; arch list set and restored; `-lineinfo`; bounded waits; per-launch announce/sync/print in the self-check.

okc_lint: `OK with warnings: muse_prefill_layer/submission.py` (0 errors; all warnings are host-sync in prepare-only code). `python3 -I -m py_compile` passed; `__pycache__` removed. Not committed, okc not run.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the code and running a numpy emulation. There is no GPU or nvcc here, so nothing was compiled.

**Verdict: dispatch-with-fixes.**

**Two-state judge hazard (setup `copies: 2`): clean.** No module or static global holds a device pointer, a CUtensorMap, a workspace or a buffer. The A-operand tensor map is encoded on the host at every `ssdq_gemm` call, from that call's own `xs`/`c2`. It is passed by value as a `__grid_constant__` kernel parameter, so each graph freezes a map that points at its own graph-pool buffer. `xs`, `c1`, `c2`, `y`, `sorted_ids` and `n_padded` are allocated per call. The error word and counters live in `state`. The only C++ static is the cached `cuTensorMapEncodeTiled` driver entry point, which is host code and not state-specific. A second `prepare()` reloads the same cached `.so` and runs its own gate, so the two states may differ in ON/OFF, and either way is correct.

**Fix (1): two-state replay self-check added.** The moe s3 lesson showed that the self-check used one state and replayed the same inputs, so it could not see this hazard. The new `_ssdq_replay_check` runs after the fp32 self-check and before the counters and the gate. It builds a second state with its own cloned bank (about 0.9 GB, freed afterwards), its own error word and its own counters. It captures one CUDA graph of `_experts_ssdq` per state, each over that state's own static inputs, at T=4096 (a judge test shape). It then runs 4 rounds in A,B / B,A order. Each round refills one state's inputs in place with a fresh routing (random / skewed / one-expert), replays that state's graph, and checks the output against the fp32 reference with the same bound as the self-check (max(2x Marlin's error, 8e-3), worst row <= 5x) and error word 0. Any miss prints `OFF: two-state replay check failed` and keeps Marlin. okc_lint is OK (warnings only, all prepare-side) and py_compile passes.

**Verified correct (no change needed):**
- **B tile matches the descriptor.** The producer's B writes match the SW128 K-major descriptor. The numpy emulation covers the reorder of `_reorder_nibbles`, the `__byte_perm` 0x4140/0x4342 nibble pairs per half, `dq_pair` bit placement and the chunk ^ (row & 7) st.shared layout, read back through the SW128 address XOR at 2 chunks x 2 halves x 256 rows x 64 k. It reproduces the E2M1 values with 0 mismatches.
- **Scales are applied correctly.** Scale group (half*4 + q) is right, the scale times 2^119 is exact in bf16, and the epilogue multiplies by gs*128.
- **Descriptor fields are valid.** Start address >> 4, SBO 1024, layout 1 = 128B swizzle, every base 1024-aligned, and the +32 B k16 advance stays inside the 128 B row.
- **A tile TMA is valid.** The box is 64 x 128 with 128B swizzle, the expect_tx count of 16384 B equals the box, and the full-barrier count of 129 matches 128 producer arrives plus the expect_tx arrive.
- **Barriers and fences are correct.** `fence.proxy.async.shared::cta` comes before every producer arrive. The empty-barrier count of 8 is one arrive per consumer warp, and a stage is released only after `wgmma.wait_group 1` has retired it. Every wait is bounded at 2^26 cycles and then sets error word 1. Neither wait depends on another CTA.
- **Accumulator fragment and epilogue are correct.** The fragment-to-row/column mapping is right for m64n256, `cw` is `warp % 4`, padded rows are skipped in w2 through the sorted_ids sentinel (n = T*8), and blocks past the used count (expert -1) exit uniformly before the first barrier.
- **Row bound and silu indexing are correct.** The bound is `cdiv(T*8 + 32*127, 128)` blocks. The silu kernel runs over rows < n_padded only, which is always a multiple of 16.
- **Rules are met.** kernel_fn's call graph branches only on T and a prepare-time flag, with no sync, print or try/except. The weights stay NVFP4 in global memory, and bf16 weight tiles exist only in shared memory. Activations are bf16 throughout.
- **The turn_4 gate is correct.** It is interleaved, uses a cold L2, makes 9 alternations of CUDA-graph replays of the whole expert block at T=3012, and turns the path ON only at <= 0.92x. The self-check includes T=4096 and runs every routing (random, skewed, one-expert) against the fp32 reference.

**Open risks (not fixable by reading):**
- **Register cap and spills.** The 168-register cap with 128 fp32 accumulators per consumer thread has not been compiled. If `local_bytes > 0` in the `[ssdq] kernel ssdq_kernel:` line, stage 2 should drop to m64n128 (BN 128, 64 accumulator registers) or reinstate a legal setmaxnreg split, for example consumers 216 and producer 72 (2*128*216 + 128*72 = 64,512 <= 65,536).
- **Prepare time.** Two prepares mean two self-checks, two replay checks and two gates. Compile happens once (about 35 s).
- **Untested code.** Nothing has run yet. A first-launch fault, such as a wrong wgmma operand syntax, would still poison the context, so the first measurement must be an okc check.

**Residual risk:** the kernel has never compiled or run, so a first-launch fault or a register spill at the 168-register cap with n256 could make the run unreadable. The okc check log's regs/local_bytes line and the canary line will tell which.
