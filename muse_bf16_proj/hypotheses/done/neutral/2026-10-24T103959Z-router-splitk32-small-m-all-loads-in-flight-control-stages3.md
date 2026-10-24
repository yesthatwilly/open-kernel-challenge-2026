---
problem: muse_bf16_proj
slug: router-splitk32-small-m-all-loads-in-flight-control-stages3
created: 2026-10-24T103959Z
base: master
area: router GEMM
change: >
  From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly
  as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPLIT_K=32) and 17..32 bucket
  (BLOCK_M=32, BLOCK_N=32, SPLIT_K=32), grid (1, 8, 32) = 256 CTAs, K-slice 256, BLOCK_K=128,
  num_warps=4, the torch.zeros output with the relaxed fp32 atomic_add epilogue, the unchanged
  m>32 buckets, and _router_config returning (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES). Change one
  thing: the two small-m buckets return NUM_STAGES=3 (master's ROUTER_NUM_STAGES) instead of 2, so
  every bucket launches with num_stages=3. ROUTER_WARM_MS is unchanged and still compiles both
  specializations in prepare().
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: 256 CTAs with both K-steps issued in the prologue (num_stages-1 = 2 >= the 2 K-iterations) give the full grid-depth effect without the 2-stage restructure
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: same bucket as decode_1
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: same bucket
  - shapes: layer=router load=decode_32 *
    direction: flat
    why: doubled atomics (262K fp32) may offset the gain from the deeper grid at m=32
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: m=128 bucket unchanged
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: m=256 bucket unchanged
risk: low
depends_on: router-splitk32-small-m-all-loads-in-flight
priority: 3
kind: control
controls_for: router-splitk32-small-m-all-loads-in-flight
review: proceed
status: done
review_at: 2026-10-24T104313Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T104313Z
base_sha: 60622cdebc40953738b4429ef80902673cd5f50f
base_blob: a0fcbb70e25fa568ef94f81935748440798a1410
branch: hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3
worktree: .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3
mode: check
implemented_at: 2026-10-24T104354Z
impl_commit: 3a8ae33a5a1e448fd4425cc85c9e4b571abde5c2
submission_sha256: 338da3e88be1bc1d7f6d1f2aba8ba0d35ef91f148a12a23cc08dc930175dbce8
norm_sha256: 69a8d95d184ceefb689de66ef3353b175b0d20bb85ec00666c814860044e215e
submission_blob: efe58306beefac9bf49baced3db1ac91f79ac357
lint_warnings: 0
attempt_file: 2026-10-24T104355Z-check-router-splitk32-small-m-all-loads-in-flight-control-stages3.md
prepared_at: 2026-10-24T104355Z
job: 4685bac7a662454d
dispatched_at: 2026-10-24T104412Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.1518068196113151
speedup_ci: [1.1345297173171454, 1.1690839219054847]
harvested_at: 2026-10-24T115931Z
best_at_harvest: 1.1432568936693237
promotion: null
---

# router-splitk32-small-m-all-loads-in-flight-control-stages3

## Change

From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPLIT_K=32) and 17..32 bucket (BLOCK_M=32, BLOCK_N=32, SPLIT_K=32), grid (1, 8, 32) = 256 CTAs, K-slice 256, BLOCK_K=128, num_warps=4, the torch.zeros output with the relaxed fp32 atomic_add epilogue, the unchanged m>32 buckets, and _router_config returning (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES). Change one thing: the two small-m buckets return NUM_STAGES=3 (master's ROUTER_NUM_STAGES) instead of 2, so every bucket launches with num_stages=3. ROUTER_WARM_MS is unchanged and still compiles both specializations in prepare().

## Expected

- `layer=router load=decode_1 *` faster: 256 CTAs with both K-steps issued in the prologue (num_stages-1 = 2 >= the 2 K-iterations) give the full grid-depth effect without the 2-stage restructure
- `layer=router load=verify_1 *` faster: same bucket as decode_1
- `layer=router load=decode_16 *` faster: same bucket
- `layer=router load=decode_32 *` flat: doubled atomics (262K fp32) may offset the gain from the deeper grid at m=32
- `layer=router load=verify_16 *` flat: m=128 bucket unchanged
- `layer=router load=verify_32 *` flat: m=256 bucket unchanged

## Risk

Separates the SPLIT_K 16->32 grid-depth effect (control vs master) from the num_stages 3->2 effect (parent vs control). Without it, a loss could come from either the atomics or the stage change, and a gain cannot tell the semaphore-reduce follow-up which SPLIT_K to build on.

## Adversarial review

Reviewed 2026-10-24T104313Z by an adversarial subagent (model opus).

**Verdict: proceed.** This control can be built as written: on the parent's branch it is one constant, ROUTER_SMALL_M_NUM_STAGES 2 -> 3. It also isolates the parent's two knobs cleanly. Control vs master changes only SPLIT_K 16->32, with the same tile, stages and smem per CTA. Parent vs control changes only num_stages 3->2 at an identical grid.

### Justification
Mechanism: the parent's worktree diff (60622cd) adds ROUTER_SMALL_M_NUM_STAGES = 2 and has _router_config return a 4-tuple. Setting that constant to 3 (or returning ROUTER_NUM_STAGES for the two small-m buckets) gives exactly this spec. Every bucket then launches with master's num_stages=3. The small-m buckets keep (16,32,32) and (32,32,32), grid (1,8,32) = 256 CTAs, K_SLICE=256, so a 2-iteration constexpr loop of BLOCK_K=128. This makes a better control than the brief claims. Master's small-m buckets already run BLOCK_M=16/32, BLOCK_N=32, BLOCK_K=128, num_stages=3, num_warps=4, so the per-CTA tile, registers and smem (~20-24 KB/stage, ~60-72 KB at 3 stages, 3 CTAs/SM fit in 228 KB) are identical to master. The only thing that changes from master is the grid depth and K-slice (128 CTAs x 4 K-steps -> 256 CTAs x 2 K-steps) and the doubled atomics. With 3 stages and 2 iterations, Triton's pipeliner issues both K-steps' x/hi/lo loads in the prologue (num_stages-1 = 2 >= trip count). That is the parent's stated 'all loads in flight' mechanism in its cleanest form. 256 CTAs at >=2 CTAs/SM on 132 SMs is still one wave. FINDINGS: _router_hilo_kernel takes 3.3/3.3/3.5/4.1 us at m=1/8/16/32 against a ~1.8 us floor for 8.4 MB. The 0.7-0.8 us memset is fixed. So up to ~1.5 us of kernel time is open per row, and Triton kernel times reproduce within ~1% (LESSONS), which is enough to read a 0.3-1 us move. Expected directions are consistent with TASK: the four small-m router rows (weight 62 each) are faster or flat, and m=128/256 are flat because those buckets are untouched (in-run noise controls). Weakest link: the 'faster' claim for m<=16 assumes the router kernel is limited by per-CTA serial depth or MLP rather than fixed latency (launch ramp, first-byte DRAM latency, tail) plus the atomic epilogue. Master's 128 CTAs at 3 stages already put ~5 MB in flight in the prologue, near Little's-law sufficiency. So a flat result here is a real and likely outcome, not a failure of the control. decode_32 'flat' is a hedge: doubling to 262K fp32 atomics into an L2-resident 32 KB output could make it slower.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m (dba4ffd): this is the kernel being retuned, not a confounder. Master's 128-CTA small-m rows (kernel 3.3/3.3/3.5/4.1 us) are the cross-run baseline for the SPLIT_K effect. The comparison crosses runs, so use kernel-us from top_kernels and the in-run m=128/256 router rows to normalise drift, not the headline.
- sglang-gemm-dispatch-parity (965b046): inert on the router path, which no longer uses cuBLAS. Its bf16 nvjet rows swing ~1.5-3.5% between runs and must not be read for or against this control.
- router-hilo-at-m8 (f9dc05d): inert. Its ROUTER_FP32_MAX_TOKENS branch was replaced wholesale by the 3-D Triton kernel, and m=8 takes the m<=16 bucket regardless.

### Interference and A/B needs
No further companions (as instructed), and none are needed. With the parent and master, this control forms a complete 2x2-minus-one on the two knobs. Master = (SPLIT_K 16, stages 3), control = (32, 3), parent = (32, 2). It leaves no unexplained cell, because (16, 2) is not part of the parent's claim. The control reads as follows. If control beats master on m<=16 kernel-us, grid depth is the lever. Parent minus control is then the pure stage effect, at identical grid and atomics. If control is flat vs master but the parent gains, the 2-stage restructure is what matters, not SPLIT_K. Then router-splitk-semaphore-no-memset should stay at SPLIT_K=16, with its 16-way workspace and ticket == 15. If both are flat, the small-m floor is launch + memset + latency, and the memset-removal hypothesis is the remaining lever. If decode_32 loses in the control while m<=16 gains, the 17..32 bucket should keep SPLIT_K=16. Every comparison crosses runs (drift ~1.5-2.5% per row, ~1% for Triton kernels), so judge on _router_hilo_kernel us, not row-us or the geomean. The router-only effect is at most ~+1% geomean, below headline resolution.

### Open questions
- The frontmatter says `base: master`, but the change must be built on the parent's branch hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight (60622cd), where depends_on points. Building from master would lack the 4-tuple _router_config and SPLIT_K=32. The implementer must branch from the parent's impl commit.
- Simplest implementation: set ROUTER_SMALL_M_NUM_STAGES = 3 and leave everything else byte-identical to the parent. The source then differs from master only in the SPLIT_K values and the inert 4-tuple plumbing, so it is not a duplicate of master. Confirm the normalised hash differs from both the parent and master.
- Confirm in the profile that _router_hilo_kernel runs as one wave (256 CTAs, >=2 CTAs/SM) at m=32 with 3 stages. If smem or registers push it to 2 waves, the decode_32 row would mostly measure that, not the atomics.
- Note for readers: control vs master isolates SPLIT_K at fixed per-CTA code. It differs from the parent's framing ('without the 2-stage restructure'), because master also ran 3 stages, only with 4 K-steps.


## Implementation notes

Implemented inline by churnsubs on the parent's commit (60622cd): `ROUTER_SMALL_M_NUM_STAGES` 2 -> 3, so the m <= 32 buckets keep SPLIT_K=32 (grid 1x8x32) with master's num_stages=3. Nothing else differs from the parent. No deviation.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3 diff` (one constant).

- [x] CUDA-graph capturable: the edit is on config values only (a module constant); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 and queries FP8 E4M3: unchanged by this edit (both paths already exist on the parent and passed review).
- [x] Attention sees every key: n/a: no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line marker kept; no triton.autotune added.

okc_lint output:

```
CLEAN: .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3/muse_bf16_proj/submission.py
```

## Result

- job `4685bac7a662454d` (check), status done, passed True, tests 9/9
- speedup 1.1518x, 95% CI [1.1345, 1.1691], score 4194.106092921194 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight-control-stages3.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight-control-stages3.patch`; branch `hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 14.24 | 13.74 | 1.017 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.81 | 6.67 | 1.0015 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 21.43 | 21.48 | 1.001 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 7.06 | 6.88 | 1.044 | gained |
| layer=router load=decode_1 m=1 | 62 | 6.28 | 5.78 | 1.0964 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.29 | 14.01 | 1.0369 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.95 | 6.69 | 1.0247 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 21.58 | 21.11 | 1.0057 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.42 | 18.47 | 1.0137 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.56 | 5.72 | 1.1336 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.68 | 14.47 | 1.0196 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.89 | 7.04 | 1.0081 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 21.81 | 21.99 | 1.0036 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.84 | 19.01 | 1.0182 | flat |
| layer=router load=decode_32 m=32 | 62 | 7.4 | 6.55 | 1.1206 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 14.21 | 14.56 | 1.0036 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.65 | 6.81 | 0.9812 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 21.32 | 21.53 | 0.9957 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.9 | 7.06 | 0.9913 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.99 | 5.28 | 1.1158 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.85 | 17.21 | 0.9714 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.49 | 7.22 | 0.9695 | lost |
| layer=pooler load=verify_16 m=128 | 62 | 24.91 | 25.46 | 0.9459 | lost |
| layer=unpooler load=verify_16 m=128 | 62 | 19.43 | 19.85 | 0.9854 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.15 | 7.65 | 0.9665 | lost |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 19.71 | 0.9956 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.47 | 9.29 | 0.9989 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.02 | 30.47 | 1.0139 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.99 | 24.3 | 1.015 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.49 | 9.25 | 1.0265 | flat |

gained on unpooler decode_1; router decode_1, decode_16, decode_32, verify_1; qkv_gate decode_16; lost on o verify_16; pooler verify_16; router verify_16. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-splitk32-small-m-all-loads-in-flight-control-stages3`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

