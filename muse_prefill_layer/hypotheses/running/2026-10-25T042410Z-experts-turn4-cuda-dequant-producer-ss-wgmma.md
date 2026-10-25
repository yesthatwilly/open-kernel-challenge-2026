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
stage: claimed
claimed_at: 2026-10-25T043310Z
base_sha: 8c0b24db143ab6bf6ff8765c64729c63abac3138
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-turn4-cuda-dequant-producer-ss-wgmma
worktree: .worktrees/muse_prefill_layer/experts-turn4-cuda-dequant-producer-ss-wgmma
mode: null
restack_of: null
stale_base: null
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

