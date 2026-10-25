---
problem: muse_prefill_layer
slug: experts-triton-turn1-blockt32-stream
created: 2026-10-24T140154Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config)"
change: >
  Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch
  hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits in
  muse_prefill_layer/submission.py. (1) MARLIN_MAX_TOKENS from 1024 to 0 (the gate stays, so a
  control can restore it): every prefill shape and test (all T > 8) runs _experts_triton with
  stage 5's packed inline-PTX dequant; the Marlin bank and the e2e Triton-vs-Marlin check in
  prepare() stay as they are (Marlin becomes the check's reference only). (2) Replace the first
  EXPERT_CONFIGS entry (T <= 1024: turn_1 T=753 and the T=200/300/753 tests) with a
  bandwidth-shaped config: BLOCK_T=32 (turn_1 puts ~24 local rows on each expert, so one 32-wide
  token tile per expert and wgmma n32 instead of today's 64 with ~60% padding), BLOCK_W=128,
  BLOCK_K=128 (fixed by the nibble order), GROUP_SIZE_T=8, num_warps=4, num_stages=4 (~17 KB of
  codes+scales+activations per stage, ~68 KB in flight per CTA, so 3 CTAs per SM keep ~200 KB of
  weight bytes in flight per SM to stream HBM). At this T each weight element is dequantized once
  and the work is a weight stream: w13+w2 read 905 MB of NVFP4 codes and e4m3 scales (189 us at
  4.8 TB/s). _check_expert_gemms already sizes its check to max(CHECK_ROWS, BLOCK_T) rows and
  compiles every entry, so the new config is checked in prepare(); the 4096 and first_turn entries
  are unchanged. If the prepare log shows spills or smem over budget, fall back to num_stages=3
  and say so in the attempt file.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "replaced kernels at turn_1: Marlin x2 at 653.9 us (full) / 752.4 us (sliding), 45-67% of the shape, ~3.5-4x the 189 us weight-stream floor. Stage 2's Triton engine (old dequant, BLOCK_T 64) ran 0.80 ms here; with packed dequant, no token padding beyond 32 and 4 stages in flight, target w13+w2 at ~300-400 us (Milestone: under 500 us on both turn_1 shapes), 250-450 us off a 1.08-1.42 ms shape, 20-35% on the turn_1 load"
  - shapes: "*load=turn_4*"
    direction: neutral
    why: "T=3012 keeps stage 6's BLOCK_T 128 config"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: T=16384 keeps the BLOCK_T 256 config
risk: medium
depends_on: experts-triton-packed-dequant-turn4
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 7
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: proceed
status: running
review_at: 2026-10-24T165620Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T010440Z
base_sha: de7af5e266755a23304676ec04c4526206066029
base_blob: 0910a52804bc45a2f5ace82707c915f4634d8add
branch: hyp/muse_prefill_layer/experts-triton-turn1-blockt32-stream
worktree: .worktrees/muse_prefill_layer/experts-triton-turn1-blockt32-stream
mode: null
restack_of: null
stale_base: true
---

# experts-triton-turn1-blockt32-stream

## Change

Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits in muse_prefill_layer/submission.py. (1) MARLIN_MAX_TOKENS from 1024 to 0 (the gate stays, so a control can restore it): every prefill shape and test (all T > 8) runs _experts_triton with stage 5's packed inline-PTX dequant; the Marlin bank and the e2e Triton-vs-Marlin check in prepare() stay as they are (Marlin becomes the check's reference only). (2) Replace the first EXPERT_CONFIGS entry (T <= 1024: turn_1 T=753 and the T=200/300/753 tests) with a bandwidth-shaped config: BLOCK_T=32 (turn_1 puts ~24 local rows on each expert, so one 32-wide token tile per expert and wgmma n32 instead of today's 64 with ~60% padding), BLOCK_W=128, BLOCK_K=128 (fixed by the nibble order), GROUP_SIZE_T=8, num_warps=4, num_stages=4 (~17 KB of codes+scales+activations per stage, ~68 KB in flight per CTA, so 3 CTAs per SM keep ~200 KB of weight bytes in flight per SM to stream HBM). At this T each weight element is dequantized once and the work is a weight stream: w13+w2 read 905 MB of NVFP4 codes and e4m3 scales (189 us at 4.8 TB/s). _check_expert_gemms already sizes its check to max(CHECK_ROWS, BLOCK_T) rows and compiles every entry, so the new config is checked in prepare(); the 4096 and first_turn entries are unchanged. If the prepare log shows spills or smem over budget, fall back to num_stages=3 and say so in the attempt file.

## Expected

- `*load=turn_1*` faster: replaced kernels at turn_1: Marlin x2 at 653.9 us (full) / 752.4 us (sliding), 45-67% of the shape, ~3.5-4x the 189 us weight-stream floor. Stage 2's Triton engine (old dequant, BLOCK_T 64) ran 0.80 ms here; with packed dequant, no token padding beyond 32 and 4 stages in flight, target w13+w2 at ~300-400 us (Milestone: under 500 us on both turn_1 shapes), 250-450 us off a 1.08-1.42 ms shape, 20-35% on the turn_1 load
- `*load=turn_4*` neutral: T=3012 keeps stage 6's BLOCK_T 128 config
- `*load=first_turn*` neutral: T=16384 keeps the BLOCK_T 256 config

## Risk

Risk: the dist=outliers test (T=200) can put far more than 32 rows on one expert; that only adds token tiles (moe_align pads per expert to BLOCK_T), so it is a correctness non-issue but not timed. At ~24 rows per expert the CTA count is 32 experts x 64 W tiles = 2048 for w13 and 1024 for w2, plenty to fill 132 SMs. If the turn_1 rows land between Marlin and the target, an ab-tune on num_stages (3 vs 5) or BLOCK_W 64 is the next step, not a revert.

## Milestones

Rewrite stage 7 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165620Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, narrow stage-7 step. It moves turn_1 (T=753, both shapes of the turn_1 load) onto the packed-dequant Triton engine with a token tile matched to ~24 local rows per expert. turn_4 and first_turn run byte-identical stage-6 code, so the run reads cleanly from the turn_1 _moe_nvfp4_gemm_kernel rows against Marlin's 654/752 us. The bandwidth premise is overstated: the kernel is register-limited to 2 CTAs/SM, not 3, and the real ceiling is dequant ALU at ~200-300 us, not the 189 us HBM floor. The <500 us milestone is still plausible.

### Justification
Mechanism. At T=753 about 753 of 6024 routes are local, ~23.5 rows per expert. One 32-wide token tile per expert means each weight element is dequantized about once (twice only for the minority of experts above 32 rows), and wgmma n32 wastes ~25% of the N side instead of ~63% at BLOCK_T 64. The FLOPs are small (~76 GFLOP useful, ~100 GFLOP padded, ~100 us at peak), so the shape is weight-stream plus dequant work. FINDINGS profile: Marlin x2 is 752.4 us (67.4% of sliding turn_1, weight 46) and 653.9 us (45.0% of full turn_1, weight 16). The w13+w2 HBM floor is ~189 us (905 MB at 4.8 TB/s), so Marlin sits 3.5-4x off it. Stage 2/3's Triton engine with the old ~10-12-instruction dequant ran turn_1 at 728-847 us (archive swapab-reglhs 796/823, blockt256 847/728). Scaling that by stage 5's ~3-3.5 instructions per element puts the dequant portion at ~220-300 us, so the 300-400 us target and the <500 us milestone are plausible. The expected directions match TASK.md: turn_4 (T=3012, 4096 entry) and first_turn (None entry) are byte-identical to stage 6, so they are neutral against the parent. turn_1 is the only moving load, and Marlin is 45-67% of its shapes, so 20-35% on that load is arithmetically consistent if the kernel lands at 300-450 us. Weakest link: the occupancy and bytes-in-flight story. (a) Stage 2's log for the sibling BLOCK_T 64 / 4 warps / 3 stages variant shows n_regs=255 and shared=67584. At 128 threads and >170 regs, only 2 CTAs fit per SM by registers. Halving the accumulator to [128,32] frees ~32 regs/thread, but Triton will likely still sit above 170, so '3 CTAs per SM, ~200 KB in flight' is likely 2 CTAs. (b) The code and scale tiles feed inline asm, not tl.dot directly, so it is unverified that Triton's pipeliner multi-buffers them through smem with cp.async rather than prefetching one iteration into registers. The 67584 B at 3 stages fits 3 activation stages plus about 2 code buffers, not 3x(codes+scales+acts). (c) Even fully pipelined, the packed dequant (~26 instructions per 8 elements, plus scale broadcast and convert, over 1.6G weight elements at ~64 int ops/clk/SM) costs ~200-300 us on the ALU pipes. So the kernel is dequant-issue-bound near 1.2-1.6x the HBM floor, not HBM-bound. Read the prepare-log n_regs/n_spills/shared/local_alloc line for the BLOCK_T=32 variant to see which of these binds.

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn (stage 4, the gate): this stage moves its gate to 0. Reverting it is meaningless (that is Marlin at turn_1, i.e. the incumbent the milestone already compares against). Not a confounder; the Marlin rows from stage 6's own run are the A side.
- sglang-pipeline-parity: supplies moe_align_block_size, silu_and_mul and fused_post_expert_norm_weighted_sum around the GEMMs. At T=753 Marlin's pick is block_size_m=32 (23.5/32 < 0.9 breaks at 32), the same as the new BLOCK_T, so the align and silu costs are identical on both sides. One asymmetry: Triton's w2 call zero-stores the ~5271 non-local slots (ZERO_NONLOCAL_W2, ~43 MB, ~9-12 us plus ~5280 trivial CTAs at BLOCK_T 32). Marlin does not pay this. Count it as a fixed in-kernel cost, not an effect of the config.
- experts-triton-w4a16-first-turn-blockt256: its BLOCK_T 256 entry is untouched and only reached at T>4096. No interaction with the turn_1 path.
- Stage 5/6 (parents, not yet landed): the packed dequant is the main lever this stage relies on. Compare turn_1 rows against stage 6's run (same code everywhere else), not against BEST.json/master, which lack stages 5/6. Use turn_4/first_turn kernel rows (byte-identical to stage 6) as the drift control.

### Interference and A/B needs
No companions. This is a rewrite stage judged on the turn_1 _moe_nvfp4_gemm_kernel rows against Marlin's 654/752 us and the 189 us floor. A gain (kernel row below Marlin) is unambiguous, and Triton kernel times reproduce within ~1%. The two bundled edits (gate to 0, new BLOCK_T 32 config) cannot be separated without a check that runs the old BLOCK_T 64 / 3-stage entry with packed dequant at turn_1. That control would only matter if the stage loses to Marlin, and even then the prepare log (n_regs, n_spills, shared, local_alloc for the BLOCK_T=32 variant) tells whether occupancy or spills caused it. The hypothesis already names an ab-tune (num_stages 3/5, BLOCK_W 64) as the next step. That is next-stage material, and prefill checks are scarce. No landed hypothesis's revert or retune would change how this run reads.

### Open questions
- What do n_regs, n_spills, shared and the local_alloc count report for the BLOCK_T=32 / 4 warps / 4 stages variant? If n_regs > 168, the kernel runs 2 CTAs/SM (8 warps), and BLOCK_W 64 or a maxnreg cap is the next lever, not more stages.
- Does Triton multi-buffer the code/scale tiles in smem (local_alloc count > activation buffers), or only the activations? If only the activations, num_stages=4 buys nothing for the weight stream.
- Is ZERO_NONLOCAL_W2 still required? fused_post_expert_norm_weighted_sum receives slot_expert_ids and may already mask the -1 slots. Dropping the zero-store removes ~10-15 us at turn_1 and ~35 us at turn_4 (a separate, later change; verify the masking first).
- Under dist=soft, how many of the 32 local experts exceed 32 rows at T=753? Each one dequantizes its weights twice. If it is many, BLOCK_T 48 is not a power of two for wgmma, but a 64-wide tile for the hottest experts is a future option.

