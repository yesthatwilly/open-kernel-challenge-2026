---
problem: muse_prefill_layer
slug: experts-marlin-block-m-48-turn1
created: 2026-10-24T075422Z
base: master
area: w13 marlin + w2 marlin (block_size_m pick)
change: >
  Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 <
  num_tokens <= 1024 use block_size_m = 48; every other token count keeps the parity loop's pick
  unchanged (64 at T=3012 and T=16384, and the parity picks at the tests outside the range). In
  the scored set this moves only the two turn_1 shapes (T=753, parity pick 64) and in the tests
  T=600 and T=753. The same block_size_m goes to moe_align_block_size and to the moe_block_size
  kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: Marlin's per-m-block
  cost grows with thread_m_blocks (muse_moe_w4a16 marlin-block-size-m-next-larger measured +18%
  for 3->4 and +28% for 1->2 at about one block per expert), so 64 -> 48 (thread_m_blocks 4 -> 3)
  should cut each block's cost, while 48 rows still hold, in one block, nearly every local expert
  at T=753 (mean ~23.5 rows per local expert, but routing is skewed by the router's alignment with
  the stream mean and router_bias, so hot experts reach 30-50 rows). It is the sibling of
  experts-marlin-block-m-local-rows (block 32 at T=753): 32 has the larger per-block saving but
  sends hot experts to 2 blocks (2 weight passes); 48 trades a smaller per-block saving for fewer
  passes. Run both; the faster lands and the other is its A/B.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 730/681 us at thread_m_blocks 4; 48 rows (tmb 3) keeps ~all local experts at one block; sister 3->4 cost was +18%, so ~12-15% off Marlin, ~6-10% per turn_1 shape
  - shapes: "*load=turn_4*"
    direction: flat
    why: T=3012 keeps the parity 64 (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 keeps the parity 64 (control)
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T083825Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T083826Z
base_sha: 4735619dba9401926cf675fc366d56f69d945921
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/experts-marlin-block-m-48-turn1
worktree: .worktrees/muse_prefill_layer/experts-marlin-block-m-48-turn1
mode: check
implemented_at: 2026-10-24T083936Z
impl_commit: 80dcbc040baf3ff15273366870c8e3756a88a5c2
submission_sha256: e6dc2af631f9accd2027d9dad4ccf60d4f973c6d4208141eb3a4e90def273589
norm_sha256: 99870116ea7ac224540a17a71afcc32f869678c08376a3e79c8cca05a90c0a82
submission_blob: 2ad966454adce5a619a1ebff11848b5463f1295a
lint_warnings: 1
attempt_file: 2026-10-24T083936Z-check-experts-marlin-block-m-48-turn1.md
prepared_at: 2026-10-24T083936Z
job: 0606f849d2f441ad
dispatched_at: 2026-10-24T083946Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0030005750632152
speedup_ci: [0.9879555664372669, 1.0180455836891633]
harvested_at: 2026-10-24T095814Z
best_at_harvest: 1.0038664293875537
promotion: null
---

# experts-marlin-block-m-48-turn1

## Change

Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count keeps the parity loop's pick unchanged (64 at T=3012 and T=16384, and the parity picks at the tests outside the range). In the scored set this moves only the two turn_1 shapes (T=753, parity pick 64) and in the tests T=600 and T=753. The same block_size_m goes to moe_align_block_size and to the moe_block_size kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: Marlin's per-m-block cost grows with thread_m_blocks (muse_moe_w4a16 marlin-block-size-m-next-larger measured +18% for 3->4 and +28% for 1->2 at about one block per expert), so 64 -> 48 (thread_m_blocks 4 -> 3) should cut each block's cost, while 48 rows still hold, in one block, nearly every local expert at T=753 (mean ~23.5 rows per local expert, but routing is skewed by the router's alignment with the stream mean and router_bias, so hot experts reach 30-50 rows). It is the sibling of experts-marlin-block-m-local-rows (block 32 at T=753): 32 has the larger per-block saving but sends hot experts to 2 blocks (2 weight passes); 48 trades a smaller per-block saving for fewer passes. Run both; the faster lands and the other is its A/B.

## Expected

- `*load=turn_1*` faster: Marlin pair 730/681 us at thread_m_blocks 4; 48 rows (tmb 3) keeps ~all local experts at one block; sister 3->4 cost was +18%, so ~12-15% off Marlin, ~6-10% per turn_1 shape
- `*load=turn_4*` flat: T=3012 keeps the parity 64 (control)
- `*load=first_turn*` flat: T=16384 keeps the parity 64 (control)

## Risk

Same knob as experts-marlin-block-m-local-rows at a different value; the two cannot both land as written (they edit the same pick), so whichever loses becomes the comparison point. 48 is a less common Marlin MoE size (thread_m_blocks 3); if the build's moe_wna16_marlin_gemm has no tmb-3 instantiation for this config it should raise at the first test, which is a clean failure, not a silent fallback.

## Adversarial review

Reviewed 2026-10-24T083825Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is one rule-clean knob: the block_size_m pick goes from 64 to 48 for 512<T<=1024. One run reads cleanly per shape, because turn_4 and first_turn act as built-in controls and SGLang's column also uses 64. Whether it wins depends on how many local experts at T=753 exceed 48 rows, and nobody has measured that, so read the result per shape and against the block-32 sibling.

### Justification
The mechanism has evidence behind it and the magnitude is consistent. In muse_moe_w4a16 marlin-block-size-m-next-larger (job 2639e3ca7f5549de), bs=128 went 0.8596x when moving 48->64. At ~4 local rows per expert that is about one block per expert, so a tmb3 block costs ~0.85x a tmb4 block. SGLang itself picks 48 at bs=128 on that problem, so the tmb-3 Marlin instantiation for this NVFP4 config already exists and ran. The 'may raise at the first test' risk is effectively moot. At T=753 Marlin (2 launches) is 730.1 us (66.4%) on the sliding shape and 680.7 us (45.4%) on the full shape. It streams ~0.9 GB at ~1.2 TB/s, so it is not byte-bound. Compute is not the limit either: about 2048 padded rows x 50 MFLOP/row x 2 is roughly 0.2 TFLOP, a few hundred us even on mma.sync. So the per-block config cost is the lever. Expected size: 12-15% off Marlin gives ~8-10% on sliding and ~5.5-7% on full. With weights 46/16 the turn_1 load gains ~8%, roughly +2.5% on the 3-load geomean. That is consistent with the stated 6-10% and far above the 1.5-2.5% per-shape drift. turn_4 (T=3012) and first_turn (T=16384) keep 64 and are true controls. Tests T=600 and T=753 change tiling only. Weakest link: the hypothesis asserts 'hot experts reach 30-50 rows' but never measured it. Each local expert above 48 rows adds a second full weight pass at tmb3. At a ~0.85 per-block ratio, break-even is roughly 32*0.15/0.85 = ~5-6 overflow experts out of 32. Routing is correlated across tokens (ROUTER_ALIGN=0.07 along the shared stream-mean direction plus ROUTER_BIAS_STD=0.055 bias), so the tail can be heavier than Poisson around 23.5. The 7% gap in Marlin time between the sliding and full turn_1 shapes (same T) already shows routing load differs per shape. A second, minor weak link: the sister ratio comes from ~4 real rows per block. At ~23 real rows the per-block cost may scale differently if the kernel is partly smem/latency-bound rather than padded-MMA-bound.

### Confounders on the current kernel
- sglang-pipeline-parity (only landed item): moe_align_block_size runs with N_EXPERTS=256 on remapped ids, where the ~7/8 of routes that are non-local are -1. If SGLang's align buckets -1 routes into skipped (is_ep) blocks, the skipped-block count grows from ~83 to ~110 when moving 64->48. If Marlin's stripe partition counts skipped blocks, part of any delta is load imbalance and not tmb cost. It is not a direction confounder: the muse_moe_w4a16 sister ran the same EP -1 structure, and SGLang's baseline column shares the path. It matters only for attributing the mechanism.
- sglang-pipeline-parity: the Triton local-id remap and the use_fp32_reduce/use_atomic_add settings are unchanged by this hypothesis, and LESSONS records those knobs as no-ops. No interaction expected.

### Interference and A/B needs
No companion is warranted. A gain is unambiguous: one knob changes, only the turn_1 Marlin rows can move, turn_4/first_turn are in-run controls, and the replaced kernel's own time (top_kernels Marlin) shows directly whether the delta is Marlin. A loss is also interpretable: overflow passes outweigh the cheaper blocks at T=753. ab-revert of sglang-pipeline-parity would revert to the starter, which is meaningless. An ab-tune of the cutoff (e.g. 48 only on sliding) branches on layer type, not on T, and would not isolate anything. Splitting 'cheaper tmb3 block' from 'more blocks' is impossible through the public API. The sibling experts-marlin-block-m-local-rows (block 32, same base, same shapes) already serves as the second point on the cost curve, so the pair brackets 64/48/32 without extra runs. Dispatch both on the same master base and compare per-shape Marlin µs directly. The loser is the A/B. Do not stack one on the other.

### Open questions
- How many of the 32 local experts exceed 48 (and 32) rows at T=753 on the scored inputs? If the implementer can compute a routing histogram offline (outside the timed call, e.g. a local script on reference inputs), it would explain a flat or split result. Do not add any device->host read inside kernel_fn.
- Sliding and full turn_1 may split, since their Marlin times differ by 7% at the same T, which suggests different routing skew. A per-layer-type pick (branching on config['window']) is legal, but adopt one only after the split repeats, not on one run.
- Confirm from the profile that the Marlin kernel launched with the tmb-3 template (kernel name or grid) at turn_1, and that turn_4/first_turn kernel times are within ~1% of 62025c368c3c4e8e.
- If both 48-turn1 and 48-turn4 win, combine them as one T->block table in a stacked follow-up. They touch disjoint T ranges, but each was measured against master alone.
- Check moe_align_block_size's sorted_ids/expert_ids buffer sizing at block 48 with num_experts=256 (non-power-of-two block). It is SGLang's own code path at bs=128 on the sister problem, so expect it to be fine.


## Implementation notes

- Where: `_experts` in `muse_prefill_layer/submission.py`, directly after the parity loop over `BLOCK_SIZE_M_CANDIDATES`. Three added lines: `if 512 < num_tokens <= 1024: block_size_m = 48` (plus a one-line comment). `num_tokens` is `x.shape[0]`, i.e. total tokens B*L, a shape, so this is a static branch under graph capture.
- The single `block_size_m` variable already feeds `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` and `common["moe_block_size"]`, which both `moe_wna16_marlin_gemm` calls take, so all three use 48 together. Nothing else changed.
- Before/after picks. The parity loop breaks only when T*8/32/bs < 0.9. That never happens for T >= 231 at bs=64, so master picks 64 for every T here:

  | T (case) | master pick | this branch |
  |---|---|---|
  | 753 (scored turn_1 sliding / full) | 64 | **48** |
  | 3012 (scored turn_4, B=4 x 753) | 64 | 64 |
  | 16384 (scored first_turn, all 4) | 64 | 64 |
  | 300 (test B1 sliding q300) | 64 | 64 |
  | 753 (test B1 ctx3000 full) | 64 | **48** |
  | 1506 (test B2 x 753 sliding) | 64 | 64 |
  | 600 (test B3 x 200 full, outliers) | 64 | **48** |
  | 4096 (test B1 q4096 sliding) | 64 | 64 |

- No deviation from the spec. I did not compute the routing histogram the review's open question asks for, because this node has no GPU or torch.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-marlin-block-m-48-turn1 diff` (+3 lines in `_experts`):
- [x] CUDA-graph capturable: the new branch reads `num_tokens = x.shape[0]`, a shape, not an input value. It adds no `.item()`, `.cpu()`, synchronize or prints.
- [x] Nothing derived from `data` is cached across calls. The diff does not touch `state`.
- [x] Weights: n/a. `prepare()` and the weight bank are untouched, and NVFP4 stays NVFP4.
- [x] Activations: BF16 where SGLang keeps BF16 and the KV cache is FP8. The diff changes only Marlin tiling, not dtypes or the cache.
- [x] Attention sees every key: n/a. Attention is untouched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. The diff adds no imports or calls.
- [x] All work happens inside the timed call. The diff adds no output caching or replay.
- [x] The first-line marker `# okc: problem=muse_prefill_layer` is kept, and there is no triton.autotune.

okc_lint output:
```
warn  host-sync    L261: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
The warning comes from the existing line in `prepare()` and is unchanged from master. `py_compile` passed, and I removed the __pycache__ it left.

## Result

- job `0606f849d2f441ad` (check), status done, passed True, tests 5/5
- speedup 1.0030x, 95% CI [0.9880, 1.0180], score 386244.9430555745 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-marlin-block-m-48-turn1.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-marlin-block-m-48-turn1.patch`; branch `hyp/muse_prefill_layer/experts-marlin-block-m-48-turn1`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1078.66 | 1112.09 | 1.0036 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1434.85 | 1487.59 | 0.983 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3367.03 | 3402.82 | 1.0002 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4949.82 | 4770.61 | 0.9986 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18279.1 | 17328.16 | 0.995 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 16995.2 | 17598.48 | 0.9973 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21535.66 | 20789.9 | 1.0289 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 25286.61 | 24817.75 | 1.0014 | flat |

no shape moved outside noise. Expected hit: 2/3. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-marlin-block-m-48-turn1`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

