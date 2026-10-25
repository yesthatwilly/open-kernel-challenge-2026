---
problem: muse_prefill_layer
slug: experts-triton-packed-dequant-turn4
created: 2026-10-24T140133Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4)"
change: >
  Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch
  hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only the
  shape gate in muse_prefill_layer/submission.py changes: MARLIN_MAX_TOKENS goes from 4096 to
  1024, so turn_4 (T=3012) and the T=4096 test run _experts_triton with the existing (4096,
  BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry of EXPERT_CONFIGS and stage 5's
  packed inline-PTX dequant, while turn_1 (T=753) and the T<=1024 tests stay on Marlin. That
  config is already compiled and checked by _check_expert_gemms in prepare(); the end-to-end
  Triton-vs-Marlin check (E2E_CHECK_TOKENS = 8192) is unchanged. At T=3012 about 3012 of the 24096
  routes are local, ~94 rows per expert, so BLOCK_T 128 covers one expert in one token tile and
  each weight element is dequantized exactly once (no redundant dequant), making this the shape
  where cheaper dequant should pay most. If stage 5's log showed register spills for the BLOCK_T
  128 variant, use num_stages=2 for it and say so in the attempt file. Nothing else changes:
  kernel, banks, first_turn config, kernel_fn.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "replaced kernels at turn_4: Marlin x2 at 2131-2217 us (43-64% of the shape). Stage 2's Triton engine with the old dequant already ran turn_4 w13+w2 at ~2.1 ms (on par with Marlin); w13+w2 move 905 MB of NVFP4 codes+scales (floor 189 us at 4.8 TB/s) and 0.30 TFLOP (300 us at 989), so both engines are ~7x their floor. With stage 5's ~3-4x cheaper dequant and one dequant per weight element, target w13+w2 at ~1.0-1.3 ms (Milestone: under 1.6 ms), 0.9-1.2 ms off a 3.3-4.9 ms shape, 20-35% on the turn_4 load"
  - shapes: "*load=turn_1*"
    direction: neutral
    why: T=753 <= 1024 stays on Marlin
  - shapes: "*load=first_turn*"
    direction: neutral
    why: same code as stage 5 at T=16384
risk: medium
depends_on: experts-triton-nvfp4-packed-asm-dequant
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: proceed
status: running
review_at: 2026-10-24T165620Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T003236Z
base_sha: 3c444875aef3f2a85c86b412e22d10ce0a6290dc
base_blob: 526518b54879e7929be2f03a0c8084907596fa70
branch: hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4
worktree: .worktrees/muse_prefill_layer/experts-triton-packed-dequant-turn4
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003418Z
impl_commit: de7af5e266755a23304676ec04c4526206066029
submission_sha256: 05424a69403b2c02e4e13bafd664dea069f501d3c8b82bbd85595eed371c5e5c
norm_sha256: e58f695f11f4803b559e8d0c23c67e51b80393eddb8ddeb844118ad5099d04e7
submission_blob: 0910a52804bc45a2f5ace82707c915f4634d8add
lint_warnings: 7
attempt_file: 2026-10-25T003418Z-check-experts-triton-packed-dequant-turn4.md
prepared_at: 2026-10-25T003418Z
job: ce3cc1a22a244ec7
dispatched_at: 2026-10-25T003428Z
---

# experts-triton-packed-dequant-turn4

## Change

Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only the shape gate in muse_prefill_layer/submission.py changes: MARLIN_MAX_TOKENS goes from 4096 to 1024, so turn_4 (T=3012) and the T=4096 test run _experts_triton with the existing (4096, BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry of EXPERT_CONFIGS and stage 5's packed inline-PTX dequant, while turn_1 (T=753) and the T<=1024 tests stay on Marlin. That config is already compiled and checked by _check_expert_gemms in prepare(); the end-to-end Triton-vs-Marlin check (E2E_CHECK_TOKENS = 8192) is unchanged. At T=3012 about 3012 of the 24096 routes are local, ~94 rows per expert, so BLOCK_T 128 covers one expert in one token tile and each weight element is dequantized exactly once (no redundant dequant), making this the shape where cheaper dequant should pay most. If stage 5's log showed register spills for the BLOCK_T 128 variant, use num_stages=2 for it and say so in the attempt file. Nothing else changes: kernel, banks, first_turn config, kernel_fn.

## Expected

- `*load=turn_4*` faster: replaced kernels at turn_4: Marlin x2 at 2131-2217 us (43-64% of the shape). Stage 2's Triton engine with the old dequant already ran turn_4 w13+w2 at ~2.1 ms (on par with Marlin); w13+w2 move 905 MB of NVFP4 codes+scales (floor 189 us at 4.8 TB/s) and 0.30 TFLOP (300 us at 989), so both engines are ~7x their floor. With stage 5's ~3-4x cheaper dequant and one dequant per weight element, target w13+w2 at ~1.0-1.3 ms (Milestone: under 1.6 ms), 0.9-1.2 ms off a 3.3-4.9 ms shape, 20-35% on the turn_4 load
- `*load=turn_1*` neutral: T=753 <= 1024 stays on Marlin
- `*load=first_turn*` neutral: same code as stage 5 at T=16384

## Risk

Risk: if stage 5 did not cut the per-tile dequant cost (its first_turn kernel rows tell), turn_4 will land near stage 2's 2.1 ms, i.e. flat against Marlin; then this stage should wait for a K-loop restructure. The w2 call's -1 blocks zero-store ~21K non-local rows x 4096 x 2 B (~170 MB, ~35 us) at this T, part of the Triton cost Marlin does not pay.

## Milestones

Rewrite stage 6 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T165620Z by an adversarial subagent (model opus).

**Verdict: proceed.** Moving the gate is a clean, one-constant stage that puts stage 5's packed dequant head to head with Marlin on turn_4 (and on the T=1506 and T=4096 tests), and it reads directly from the turn_4 top_kernels rows. The direction is plausible, but the 1.0-1.3 ms target is optimistic. Measured data puts the dequant share of turn_4's ~2.1 ms at about 0.6 ms, so expect ~1.6-1.8 ms, which is a 0.35-0.55 ms win over Marlin's 2.13-2.22 ms and sits at the edge of Milestone 2's under-1.6 ms target.

### Justification
The mechanism is plausible. At T=3012 about 3012 routes are local, ~94 rows per expert, so BLOCK_T 128 is one token tile per expert at the mean, and Triton's register-LHS wgmma serializes dequant ahead of the MMA, so a cheaper dequant shortens every K step. The prior runs confirm the config is viable at this shape. Stage 2 (swapab-reglhs, scalar dequant, the same 128/128/8w/3s config) ran turn_4 w13+w2 at 2100-2158 us against Marlin's 2131-2217 us on master, and passed all 5 tests with this config at every T. The floor arithmetic checks out (905 MB -> 189 us, 0.30 TFLOP -> 300 us). WEAKEST LINK: the size of the gain. It assumes turn_4 is dequant-bound, and the data says otherwise. (a) The dequant work per weight element is the same at turn_1 and turn_4 (one tile per expert at both), yet stage 2 took ~0.80 ms at turn_1 and ~2.1 ms at turn_4. So ~1.3 ms of turn_4 scales with tokens (MMA, gathered activation loads, epilogue, zero stores), not with dequant. (b) Stage 3 calibrates one full dequant pass. Halving the redundant dequant at first_turn (BLOCK_T 128->256, ~4 -> ~2 passes per element) saved ~1.1-1.3 ms, so one pass over w13+w2 costs about 0.55-0.65 ms on this engine. A 3-4x cheaper dequant then saves ~0.4-0.5 ms at turn_4, which gives ~1.6-1.75 ms, not 1.0-1.3 ms. That is still ~0.4-0.55 ms under Marlin, about 12-16% on sliding turn_4 (3335 us) and 8-11% on full turn_4 (4938 us), roughly +2-3% on the headline. That is readable, but it is not the stated 20-35%. Second weak link: the one-dequant-per-element premise holds only at the mean. dist=soft routing leaves some experts above 128 rows, so they take two tiles, and the mean tile is ~27% padding. Third: if stage 5's prepare() fell back to the scalar dequant (state['packed'] False after the tile check or the e2e check), this run is just stage 2's engine at turn_4. That engine is known to be at par or slower (stage 3's identical turn_4 config ran 2275-2318 us, a 0.97x shape loss), and the result would say nothing about the packed dequant. The expected directions match TASK.md: turn_1 (T=753) stays on Marlin, and first_turn is byte-identical to stage 5. Noise: Triton turn_4 rows moved ~7% between stages 2 and 3 with an identical config (2100-2158 vs 2275-2318 us), so only a change over ~10% in the replaced kernel row is attributable from one run. A 0.4 ms move clears that bar.

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn (stage 4, landed): this stage undoes its turn_4 half by moving MARLIN_MAX_TOKENS from 4096 to 1024. That makes the comparison against master's turn_4 rows a direct Triton-packed vs Marlin A/B, which is the point, not a confound. Both banks stay built in prepare, so memory and L2 state are unchanged.
- experts-triton-w4a16-first-turn-blockt256 (stage 3, landed): it owns the BLOCK_T 256 first_turn config and the 4096 entry this stage reuses. No interaction at turn_4, but its turn_4 Triton rows (2275-2318 us, the same config as stage 2's 2100-2158 us) show ~7% run-to-run drift for this kernel at this shape. Use that as the noise band, not LESSONS' ~1%.
- experts-triton-nvfp4-packed-asm-dequant (parent stage 5, not landed): every first_turn difference against BEST comes from stage 5, not from this stage. The headline mixes both, so attribute from per-shape rows only: turn_4 against master (Marlin), first_turn against stage 5's own run.
- sglang-pipeline-parity (landed): it provides the shared moe_align_block_size, silu_and_mul and fused_post_expert_norm_weighted_sum around both engines. Those are untouched and run on both sides, so they do not confound. Note that silu_and_mul over all 24096 slots, including non-local ones, is in both.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage with a single-constant change, and it is judged on the replaced kernels' rows: master's turn_4 Marlin rows (2131-2217 us) against this run's turn_4 _moe_nvfp4_gemm_kernel row. The scalar-dequant control already exists: stages 2 and 3 ran this same 128/128/8w/3s config at turn_4 with the scalar dequant (2100-2318 us), so a PACKED_DEQUANT=False companion would rerun known data. An ab-revert of stage 4 is this hypothesis's own change. The one thing that would make the result ambiguous is a silent fallback to the scalar dequant, and the log answers that without another run. Read '[expert gemm] packed dequant ON' and '[expert e2e]' in profiles/submission.<job>.log before attributing anything, and if it is OFF, file the run as a stage-2 rerun, not as a packed result.

### Open questions
- Check stage 5's log for n_spills on the packed BLOCK_T=128 variant and for 'packed dequant ON'. If prepare fell back to the scalar dequant, hold this stage until stage 5 is fixed: it would only re-measure stage 2.
- The e2e Triton-vs-Marlin check (E2E_CHECK_TOKENS 8192) exercises only the BLOCK_T 256 config. With the gate at 1024, the 128 config now serves the T=1506 and T=4096 judge tests and the timed turn_4 calls. Consider a second untimed e2e call at T=3012 in prepare(), so a packed-path wiring bug at BLOCK_T 128 falls back instead of failing the judge. The tile check covers only the dequant, not the full silu/w2/finalize chain at this config.
- Once stage 5 harvests, use its first_turn kernel rows against stage 4's 8.1-8.9 ms to recompute the per-pass dequant cost. If stage 5 saved under ~0.4 ms at first_turn (~2 passes there), expect turn_4 to land within the noise band of Marlin (<~0.2 ms).
- Next-stage lever after this one: since ~1.3 ms of turn_4 scales with tokens, not with dequant, the next limit is the per-K-step serialization of dequant -> wgmma within the same warps (no producer/consumer split) and the gathered (sorted_ids // TOP_K) activation loads without TMA. A warp-specialized dequant producer is the step that could reach Milestone 3.
- ZERO_NONLOCAL_W2 zero-stores ~21K non-local rows x 4096 x 2 B (~170 MB, ~35 us) at turn_4. Does fused_post_expert_norm_weighted_sum mask by slot_expert_ids? If it does, the zero store (and Marlin's equivalent) can go in a later stage.
- The grid covers the static EM bound (~24096 + 256*127 slots -> ~442 token tiles x 64 w-tiles for w13). Most CTAs exit early. That launch overhead is likely 10-20 us at turn_4: small, but visible against a ~0.4 ms target.


## Implementation notes

Edited only `muse_prefill_layer/submission.py` in the worktree (16 insertions, 13 deletions).
- `MARLIN_MAX_TOKENS` 4096 -> 1024 (comment updated). turn_4 (T=3012) and the T=1506 / T=4096 tests now run `_experts_triton` with the existing (4096, BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry and stage 5's packed dequant; turn_1 and T<=1024 stay on Marlin.
- Review fix (open question 1): `E2E_CHECK_TOKENS` is now `(8192, 3012)` and `_check_experts_e2e` takes `t`; prepare() runs both (in both the first attempt and the scalar-fallback retry), so a packed-path wiring bug at BLOCK_T 128 falls back to the scalar dequant instead of failing the judge. Untimed, prepare() only.
- num_stages: kept 3. REWRITE_READ_2026-10-25.md reports a w13 BLOCK_T 256 spill (4 spills, 255 regs) but no spill for the BLOCK_T 128 variant, so the num_stages=2 rule does not trigger.
- STALE BASE: claim reported stale_base: true. The worktree is on stage 5's commit (3c444875ae), which predates prefill master (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored). Built on stage 5 as the spec says; not rebased. Master's parallel align and `_silu_mul_sorted_kernel` are therefore absent here, so the turn_4 comparison against master's Marlin rows (2148/2115 us) is against a different surrounding pipeline.
- Context from REWRITE_READ: stage 5 was judged DEAD (packed dequant ON, mean +1.4% at first_turn, within noise; asm added a smem round trip). The review expects only ~1.6-1.8 ms at turn_4, so a gain here is plausible but modest. Check the log for 'packed dequant ON' and both '[expert e2e]' lines before attributing.

## Rules check

- [x] CUDA-graph capturable: diff touches a constant, prepare() and its untimed check only; kernel_fn path unchanged, no new sync/alloc/value-dependent control flow (gate reads x.shape[0] only).
- [x] Nothing derived from `data` is cached across calls: n/a to kernel_fn; the e2e check uses synthetic data inside prepare() only.
- [x] Weights: unchanged (no repacking changes in the diff).
- [x] Activations BF16 / KV FP8 / queries: unchanged by the diff.
- [x] Attention sees every key: attention untouched by the diff.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: the Triton path runs fully inside kernel_fn; no caching added.

okc_lint output:
```
warn  host-sync    L537, L579, L633, L638, L649, L658, L668: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
py_compile passed; __pycache__ removed.
