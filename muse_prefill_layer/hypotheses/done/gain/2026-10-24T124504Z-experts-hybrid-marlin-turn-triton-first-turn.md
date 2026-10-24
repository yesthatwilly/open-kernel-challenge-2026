---
problem: muse_prefill_layer
slug: experts-hybrid-marlin-turn-triton-first-turn
created: 2026-10-24T124504Z
base: master
area: "w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)"
change: >
  Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3
  experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in
  muse_prefill_layer/submission.py change; kernel_fn is byte-identical between parity and master
  and stays so. (1) Restore the parity Marlin path verbatim from sglang-pipeline-parity (commit
  6e616b8a05, muse_prefill_layer/submission.py): its imports (types, scalar_types from
  sgl_kernel.scalar_type, moe_wna16_marlin_gemm, marlin_make_workspace,
  prepare_moe_nvfp4_layer_for_marlin), BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64),
  _marlin_bank(weights) unchanged, and the parity _experts body renamed _experts_marlin
  (block_size_m loop breaking at num_tokens*TOP_K/LOCAL_EXPERTS/block_size_m < 0.9,
  moe_align_block_size at that block size, the two moe_wna16_marlin_gemm calls with the same
  common kwargs: mul_topk_weights=False, is_ep=True, float4_e2m1f, is_k_full=True,
  use_atomic_add=False, use_fp32_reduce=True, workspace max_blocks_per_sm=4, then silu_and_mul and
  fused_post_expert_norm_weighted_sum exactly as parity). (2) Rename master's _experts body to
  _experts_triton, unchanged (EXPERT_CONFIGS, _nvfp4_moe_gemm, ZERO_NONLOCAL_W2 as on master). (3)
  New _experts(x, topk_weights, local_ids, state) with MARLIN_MAX_TOKENS = 4096: if x.shape[0] <=
  MARLIN_MAX_TOKENS call _experts_marlin, else _experts_triton. The gate reads only the token
  count (a shape, equal to config tokens), never tensor values: turn_1 T=753, turn_4 T=3012 and
  every test (largest T=4096) go to Marlin; the four first_turn shapes (T=16384) go to Triton with
  the BLOCK_T 256 config. (4) prepare() builds both layouts: state["marlin"] =
  _marlin_bank(weights) beside state["bank"] = _expert_bank(weights), and still runs
  _check_expert_gemms on the Triton bank (leave EXPERT_CONFIGS intact so the two now-unreachable
  small-T configs are still compiled and checked; no runtime effect). Weights stay NVFP4 in both
  layouts, each kernel dequantizes in registers; no BF16 copy, no autotune, no host sync in
  kernel_fn, nothing derived from inputs cached. Memory cost of the second bank: Marlin-repacked
  codes 768 MiB (w13 512 + w2 256) plus permuted e4m3 scales 96 MiB plus the small workspace,
  about 0.86 GiB, on top of master's nibble-reordered codes (0.75 GiB), cos_sin (512 MiB) and the
  original weights; activations are unchanged (largest is cache13 at T=16384, 2 GiB, Triton path).
  Under 1 GiB extra against an H200's 141 GB, so headroom is not a concern; prepare time grows by
  SGLang's repack (seconds, within 300 s). Print torch.cuda.max_memory_allocated() at the end of
  prepare() so the run log shows the ~0.9 GiB growth.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "w13+w2 return from the Triton kernel (847 us sliding / 729 us full) to Marlin (~730 / 681 us in the parity profile): 50-120 us off a 1.08-1.43 ms shape, master 0.93-0.98x of SGLang back to ~1.0x"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "Triton 2275 / 2318 us vs Marlin ~2250 / 2200 us: 1-5% on the shape, back to parity with SGLang"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: T=16384 > 4096 keeps stage 3 Triton BLOCK_T 256 (w13+w2 8.0-8.7 ms vs Marlin 10.2-12.0 ms) unchanged
risk: low
depends_on: experts-triton-w4a16-first-turn-blockt256
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: proceed
status: done
review_at: 2026-10-24T124746Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T124746Z
base_sha: b1395799d690a0aa9053a787f98c98321be85724
base_blob: f17d41c9a9b0696f20e6679a493da9d4aa6d2934
branch: hyp/muse_prefill_layer/experts-hybrid-marlin-turn-triton-first-turn
worktree: .worktrees/muse_prefill_layer/experts-hybrid-marlin-turn-triton-first-turn
mode: check
implemented_at: 2026-10-24T125053Z
impl_commit: c322454f5a0f28dabba91a7cf546d13eb4d3a230
submission_sha256: ecb9bd833e29d3c4e9a90546156d72d305d53f40471f59c6f8a1edbc984f7f9a
norm_sha256: bf96d2824fa9f5656c73c1770ff12c6244bae0c27fc96d325b87a360d4564300
submission_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
lint_warnings: 5
attempt_file: 2026-10-24T125053Z-check-experts-hybrid-marlin-turn-triton-first-turn.md
prepared_at: 2026-10-24T125053Z
job: fb1e52d2119d4175
dispatched_at: 2026-10-24T125104Z
outcome: gain
verdict_detail: gain
speedup: 1.0599319720006266
speedup_ci: [1.044032992420617, 1.075830951580636]
harvested_at: 2026-10-24T135614Z
best_at_harvest: 1.0228712349305686
promotion: merged
promotion_commit: b92a0f47192b31b8618c3c53aab281b72204f3c4
promotion_attempt: 2026-10-24T163459Z-submit-experts-hybrid-marlin-turn-triton-first-turn.md
---

# experts-hybrid-marlin-turn-triton-first-turn

## Change

Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3 experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in muse_prefill_layer/submission.py change; kernel_fn is byte-identical between parity and master and stays so. (1) Restore the parity Marlin path verbatim from sglang-pipeline-parity (commit 6e616b8a05, muse_prefill_layer/submission.py): its imports (types, scalar_types from sgl_kernel.scalar_type, moe_wna16_marlin_gemm, marlin_make_workspace, prepare_moe_nvfp4_layer_for_marlin), BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64), _marlin_bank(weights) unchanged, and the parity _experts body renamed _experts_marlin (block_size_m loop breaking at num_tokens*TOP_K/LOCAL_EXPERTS/block_size_m < 0.9, moe_align_block_size at that block size, the two moe_wna16_marlin_gemm calls with the same common kwargs: mul_topk_weights=False, is_ep=True, float4_e2m1f, is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, workspace max_blocks_per_sm=4, then silu_and_mul and fused_post_expert_norm_weighted_sum exactly as parity). (2) Rename master's _experts body to _experts_triton, unchanged (EXPERT_CONFIGS, _nvfp4_moe_gemm, ZERO_NONLOCAL_W2 as on master). (3) New _experts(x, topk_weights, local_ids, state) with MARLIN_MAX_TOKENS = 4096: if x.shape[0] <= MARLIN_MAX_TOKENS call _experts_marlin, else _experts_triton. The gate reads only the token count (a shape, equal to config tokens), never tensor values: turn_1 T=753, turn_4 T=3012 and every test (largest T=4096) go to Marlin; the four first_turn shapes (T=16384) go to Triton with the BLOCK_T 256 config. (4) prepare() builds both layouts: state["marlin"] = _marlin_bank(weights) beside state["bank"] = _expert_bank(weights), and still runs _check_expert_gemms on the Triton bank (leave EXPERT_CONFIGS intact so the two now-unreachable small-T configs are still compiled and checked; no runtime effect). Weights stay NVFP4 in both layouts, each kernel dequantizes in registers; no BF16 copy, no autotune, no host sync in kernel_fn, nothing derived from inputs cached. Memory cost of the second bank: Marlin-repacked codes 768 MiB (w13 512 + w2 256) plus permuted e4m3 scales 96 MiB plus the small workspace, about 0.86 GiB, on top of master's nibble-reordered codes (0.75 GiB), cos_sin (512 MiB) and the original weights; activations are unchanged (largest is cache13 at T=16384, 2 GiB, Triton path). Under 1 GiB extra against an H200's 141 GB, so headroom is not a concern; prepare time grows by SGLang's repack (seconds, within 300 s). Print torch.cuda.max_memory_allocated() at the end of prepare() so the run log shows the ~0.9 GiB growth.

## Expected

- `*load=turn_1*` faster: w13+w2 return from the Triton kernel (847 us sliding / 729 us full) to Marlin (~730 / 681 us in the parity profile): 50-120 us off a 1.08-1.43 ms shape, master 0.93-0.98x of SGLang back to ~1.0x
- `*load=turn_4*` faster: Triton 2275 / 2318 us vs Marlin ~2250 / 2200 us: 1-5% on the shape, back to parity with SGLang
- `*load=first_turn*` neutral: T=16384 > 4096 keeps stage 3 Triton BLOCK_T 256 (w13+w2 8.0-8.7 ms vs Marlin 10.2-12.0 ms) unchanged

## Risk

Low risk: both halves have passed every test before (parity ran Marlin at all T, master runs Triton at all T); only the shape gate and holding both banks are new. Judge the stage on the replaced kernels per shape: moe_wna16_marlin at turn_1/turn_4 should match the parity profile within ~2%, the Triton kernel at first_turn should match stage 3 within ~1%. If turn shapes do not return to ~1.0x of SGLang, look for a confound outside the experts, not the gate. The 4096 bound comes from the measured crossover (Triton loses at T=3012, wins at T=16384); T between 4096 and 16384 is unscored and not tuned here. Expected overall: the turn_1 and turn_4 loads move from ~0.95x to ~1.0x, first_turn keeps its gain, so roughly 1.023x -> 1.04x.

## Milestones

Rewrite stage 4 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T124746Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound, low-risk stage-4 gate: put Marlin back where it has won every within-run comparison (turn_1) or tied (turn_4), and keep stage-3 Triton BLOCK_T 256 at T=16384, where it is about 25% faster than Marlin. The one real build hazard is tensor aliasing between the two banks in prepare(). With prefill on its last check, close it by cloning the inputs to _marlin_bank and running the Triton bank and its self-check first.

### Justification
Mechanism: SGLang's baseline is the same Marlin kernel. At T<=4096 the paired ratio against SGLang should therefore return to ~1.00, the parity run's 1.007/1.009 (turn_1) and 1.002/1.003 (turn_4). The turn_1 evidence is strong and consistent. Triton was 0.935/0.952 vs SGLang in stage 2 (00148fbe…) and 0.932/0.943 in stage 3 (b33b87b5), with kernel rows 796/823 and 847/729 us against Marlin's 730/681. At ~24 rows per local expert, the per-tile in-register dequant cannot amortize. The turn_4 evidence is the weakest link and is a near-tie, not a measured crossover. In stage 2, byte-identical Triton code was 1.012/0.999 vs SGLang (kernel 2158/2100 us, below Marlin's 2248/2201). In stage 3 the same code was 0.973/0.981 (2275/2318 us). The hypothesis's 'Triton loses at T=3012' leans on the second run only. The two Triton runs average ~0.99 against Marlin's ~1.00, so sending turn_4 to Marlin is the safe choice (a guaranteed ~1.0 floor), but expect +0-2.5% there, not a clear win. Cross-run kernel comparisons are noisier than LESSONS' '~1%' claim, because every run draws new routing seeds, so expert imbalance changes. The same Triton code moved +5%/+10% at turn_4 and -11% at full turn_1 between stages 2 and 3. Read this stage on the within-run paired ratio vs SGLang per shape, which is self-calibrating here because both sides run Marlin at T<=4096. Do not read it on kernel-µs against the parity profile. The 4096 bound is consistent with TASK: turn_1 T=753, turn_4 T=3012 and all five tests (max T=4096) go to Marlin, and the four first_turn shapes (T=16384) go to Triton. The headline estimate is understated. With load-geomean (0.935·0.975·1.17)^(1/3)≈1.022 today, returning turn_1 to ~1.005 and turn_4 to ~1.0 while first_turn holds ~1.17 gives ≈1.055-1.06, not 1.04. As a rewrite milestone: (1) correctness at T<=4096 is the parity path that already passed; (2) turn shapes at or below the incumbent by construction; (3) floor distance is unchanged (Marlin is 2.8x/1.4x the floor at turn_1 sliding/full), so this stage reaches 'promotable', not 'near floor'. Memory (+~0.86 GiB) and prepare time are non-issues.

### Confounders on the current kernel
- sglang-pipeline-parity: no confound. The restored _experts_marlin is that hypothesis's own code, verbatim, and everything around the experts is byte-identical. Its run is the reference for what turn shapes should read (~1.00 vs SGLang).
- experts-triton-w4a16-first-turn-blockt256: no confound on the T=16384 shapes, where the code is byte-identical (only reached through the new gate). Its T<=4096 configs become unreachable, so turn-shape movement cannot come from them. Residual interaction is limited to prepare(): its bank aliases weights[*_sf/_gs] through .contiguous(), so an in-place change by prepare_moe_nvfp4_layer_for_marlin would corrupt the Triton path (see open_questions).

### Interference and A/B needs
No companion is warranted. Both outcomes read unambiguously without one. At turn shapes both sides of the paired ratio run the same Marlin kernel, so anything other than ~1.00±drift means a wiring bug in the gate (e.g., the wrong bank or block size), not an interaction. At first_turn the kernel row should match stage 3's 8574/8535/7960/8722 us within routing drift, and the shape ratio should match 1.19/1.19/1.14/1.11. An ab-revert of stage 3 would just recreate stage 2 or parity, which are already measured. A control isolating the turn_4 half of the gate (MARLIN_MAX_TOKENS=1024) would resolve a ~1-2% effect that one run cannot resolve anyway. It is not worth a check with prefill at 1 check left. Leave it for a later stage if turn_4 is tuned.

### Open questions
- Aliasing (build must address it): master's _expert_bank stores weights['w13_sf'/'w13_gs'/'w2_sf'/'w2_gs'].contiguous(), which is the same storage as the weights. Parity's _marlin_bank hands the same aliased tensors to prepare_moe_nvfp4_layer_for_marlin. Its source is not available on this node, so whether it mutates scales or global scales in place cannot be verified. If it does, the Triton bank's scales change silently. _check_expert_gemms would not catch it, because it reads sf/gs from the bank, not from the weights. The failure would then surface only at the first_turn timed calls, burning the last check. Cheap fix: pass .clone() tensors into _marlin_bank. Also build state['bank'] and run _check_expert_gemms before _marlin_bank, and assert torch.equal on the four scale tensors before and after.
- Is the prepare() self-check enough, given that no pre-timing test reaches T>4096? Mostly yes. It compiles and checks the BLOCK_T 256 variant's dequant, indexing, -1 and past-n_padded handling. The first_turn path is byte-identical to stage 3, which passed timed-call correctness at T=16384, and every timed call is checked against the reference, so a bug fails loudly rather than scoring. What it does not cover is the new wiring: which bank _experts_triton reads, and moe_align_block_size at cfg BLOCK_T on the >4096 branch. Recommended additional untimed check: one end-to-end call each of _experts_triton and _experts_marlin on a synthetic T=8192 case (random bf16 x, random top-8 local ids and weights, all built in prepare), asserting relative L2 between them below ~1e-2. That covers the aliasing hazard and the branch wiring for a few hundred ms.
- Make sure _experts' dispatch reads x.shape[0] (= config tokens), not num_slots, and that the Marlin branch uses state['marlin'] for the bank and workspace while the Triton branch uses state['bank'].
- T between 4096 and 16384 is untuned and unscored, so the crossover there is unknown. That is acceptable, but note it in FINDINGS so a later turn_4 stage does not assume 4096 was measured.


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-hybrid-marlin-turn-triton-first-turn`, file `muse_prefill_layer/submission.py` only (diff: 137 insertions, 4 deletions, uncommitted). kernel_fn is byte-identical to master (checked with diff).

- Imports restored from parity 6e616b8a05: `types`, `scalar_types` (sgl_kernel.scalar_type), `moe_wna16_marlin_gemm`, `marlin_make_workspace`, `prepare_moe_nvfp4_layer_for_marlin`.
- Constants: `MARLIN_MAX_TOKENS = 4096` (commented as not a measured crossover, per the review) and `BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)`.
- `_marlin_bank(weights)` restored verbatim (byte-identical to parity, checked with diff).
- `_experts_marlin` is the parity `_experts` body verbatim except one line: `bank = state["marlin"]` (was `state["bank"]`), so the Marlin branch reads the Marlin bank and its workspace.
- `_experts_triton` is master's `_experts` body unchanged. It still reads `state["bank"]`, and EXPERT_CONFIGS and ZERO_NONLOCAL_W2 are untouched.
- New `_experts`: `x.shape[0] <= MARLIN_MAX_TOKENS` goes to Marlin, else Triton. It reads the shape only, not num_slots or any tensor value.
- prepare() (build requirement 1): `state["bank"] = _expert_bank(weights)` is built first, and `_check_expert_gemms` runs on it (all three EXPERT_CONFIGS still compiled and checked). Then the four Triton-bank scale tensors are cloned, and `_marlin_bank` is called on a dict of `.clone()`d w13/w2/w13_sf/w2_sf/w13_gs/w2_gs. After the repack, torch.equal asserts that the four bank scales are unchanged, and so are the weights[...] they alias.
- prepare() (build requirement 2): new `_check_experts_e2e(state, device)`.
  - Data: synthetic, weights-only, at T = E2E_CHECK_TOKENS = 8192, built in prepare from a seeded generator. That is randn bf16 x [8192, 4096], a distinct top-8 of 256 (rand().topk) mapped to local 0-31 / -1 as kernel_fn maps them, and positive normalized fp32 weights.
  - Calls: `_experts_triton` once (BLOCK_T 256 config, the first_turn path) and `_experts_marlin` once (block_size_m 64).
  - Asserts: both outputs are finite; relative L2 <= E2E_CHECK_TOL = 1e-2; worst row <= 5x that, with near-zero rows measured against a tenth of the mean row norm (the judge's row rule). Both numbers are printed.
- Tolerance choice, a judgment call: "the reference tolerance" was read as the review's ~1e-2 bound, which is below the tightest judge bound (3%). checks.py is not imported, because it sits beside reference.py and an import could trip the reference-import rule. The metric is reimplemented inline instead.
- `torch.cuda.empty_cache()` runs after the Marlin repack to drop the clones before the e2e check.
- `[prepare] max_memory_allocated` is printed at the end of prepare. It includes the e2e check's transient activations (cache13 at T=8192 is about 1 GiB), so the printed peak is above the steady-state bank growth.
- The module docstring's experts line now names the gate.
- Deviations from the spec: only the cloning, the asserts and the e2e check that the review and brief require. Nothing else.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-hybrid-marlin-turn-triton-first-turn diff`.

- [x] CUDA-graph capturable. kernel_fn is byte-identical to master. The new `_experts` branches only on `x.shape[0]`, a shape. Both branches were capturable before (parity for Marlin, master for Triton). The only new syncs are `torch.cuda.synchronize()` calls in prepare(), and the lint confirms they are outside kernel_fn's call graph. `_check_experts_e2e` uses `float()`/`bool()` on tensors in prepare() only.
- [x] Nothing derived from `data` is cached across calls. State gains only `marlin` (built from weights) and keeps `bank`. The e2e data is synthetic, made in prepare and not stored.
- [x] Weights repacked losslessly in prepare() only. The Marlin bank is SGLang's own NVFP4 repack (E2M1 codes, permuted e4m3 scales, fp32 global scales), and the Triton bank is unchanged. Both stay NVFP4, there is no BF16 expert copy, and the router hi/lo is unchanged.
- [x] Activations BF16 where SGLang keeps BF16. The Marlin path is SGLang's own and the Triton path is unchanged. The KV cache and q to FP8 are unchanged (kernel_fn identical).
- [x] Attention sees every key: n/a in substance, because the attention code is untouched (kernel_fn identical: full window / full context FA3).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new imports are sglang/sgl_kernel modules and `types`. checks.py and reference.py are not imported.
- [x] All work happens inside the timed call; no output caching or replay. The expert GEMMs run per call on either path. The prepare() checks produce nothing that kernel_fn reads.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept.
- [x] No triton.autotune added. EXPERT_CONFIGS stays static and the gate is a fixed constant.

okc_lint output:

```
warn  host-sync    L463: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L505: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L557: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L563: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L570: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` passed. The __pycache__ was removed from the worktree.

- churnsubs: loosened E2E_CHECK_TOL from 1e-2 to 2e-2 before dispatch (still under the judge's tightest 3%): the check compares two bf16 pipelines (Triton vs Marlin), not either against the reference, and a spurious prepare() failure would burn prefill's last check of the day.

## Result

- job `fb1e52d2119d4175` (check), status done, passed True, tests 5/5
- speedup 1.0599x, 95% CI [1.0440, 1.0758], score 365498.83410797897 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0228712349305686, "ci": [1.00752816640661, 1.038214303454527], "job": "b33b87b5083b4d1c"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.result.json
- **outcome: gain** (gain)
- files: `muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.patch`; branch `hyp/muse_prefill_layer/experts-hybrid-marlin-turn-triton-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1141.1 | 1079.54 | 1.0812 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1517.93 | 1419.09 | 1.0691 | gained |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3435.39 | 3334.9 | 1.0301 | gained |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4975.37 | 4937.98 | 1.0223 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15029.51 | 15386.32 | 1.0172 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14426.71 | 14275.3 | 1.0055 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18568.92 | 18165.94 | 0.9923 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22521.73 | 22332.92 | 1.0049 | flat |

gained on sliding turn_1, turn_4; full turn_1. Expected hit: 2/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 4 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-hybrid-marlin-turn-triton-first-turn --stage 5`; a slow or failed first cut is the expected start, not a do-not-retry.
- candidate for promotion; stack further changes with `depends_on` once it lands.

