---
problem: muse_prefill_layer
slug: silu-mul-sorted-first-turn
created: 2026-10-24T233334Z
base: master
area: silu-mul on the Triton path (T > 4096, first_turn)
change: >
  Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton,
  when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2,
  expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids,
  n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend
  _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us
  at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the
  reason the parent excluded this path, is done (neutral), so nothing is confounded.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: silu-mul over local rows only at first_turn
risk: medium
depends_on: silu-mul-triton-local-rows-marlin-turn
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T233335Z
base_sha: a0eb6398a55163f42feb6a3cdfb8e02b392e090b
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/silu-mul-sorted-first-turn
worktree: .worktrees/muse_prefill_layer/silu-mul-sorted-first-turn
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233357Z
impl_commit: fc7f0a08663ad479ea473c461c282a70c85fd712
submission_sha256: 1c7c6b27c04f1afe3fee01ce659e9e3f56704866761be61a98e60a22eadf6234
norm_sha256: 15e47d79d157b9eebd87478eb8529df1ac369836121c3420d8915881b5c28ebe
submission_blob: 9e85c24636d47e2cf91bb76ca793a5a9894ef683
lint_warnings: 7
attempt_file: 2026-10-25T000329Z-check-silu-mul-sorted-first-turn.md
mode: check
prepared_at: 2026-10-25T000329Z
job: 9528eed511b04f33
dispatched_at: 2026-10-25T000344Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.0636456311695792
speedup_ci: [1.0476909467020354, 1.0796003156371228]
harvested_at: 2026-10-25T003212Z
best_at_harvest: 1.0705080897445196
promotion: null
---

# silu-mul-sorted-first-turn

## Change

Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the reason the parent excluded this path, is done (neutral), so nothing is confounded.

## Expected

- `*load=first_turn*` faster: silu-mul over local rows only at first_turn

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:35Z with user-approved self-review, on silu-mul-triton-local-rows-marlin-turn's implementation commit (claim based it there: parent still running). `_experts_triton`: `_silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n)` when `state["silu_sorted_ok"]`, else master's silu_and_mul. `_silu_sorted_setup` now also checks T=16384 at the Triton path's BLOCK_T (256); a miss turns the sorted silu OFF on both paths. Equivalence: master's silu_and_mul(expert_ids=local_ids) also skips -1 slots; the w13 Triton GEMM leaves -1-tile rows unwritten and w2 zero-stores those tiles without reading cache2, so skipping them changes no read value.

## Rules check

- [x] CUDA-graph capturable: one Triton launch, static grid, prepare()-set gate.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Re-base (2026-10-25 00:0xZ)

Reset priority 2: re-based onto prefill master after stack-tonight-five-prefill landed (it already carries the Marlin-path sorted silu). The parent diff (8+/2-: the Triton-path gate and the T=16384 self-check) applied cleanly. On first_turn the sorted silu now walks master's moe_align_block_size(LOCAL_EXPERTS) output and still masks local_ids >= 0, so it does not depend on which align ran. Rules check above still holds; lint has no errors.

## Result

- job `9528eed511b04f33` (check), status done, passed True, tests 5/5
- speedup 1.0636x, 95% CI [1.0477, 1.0796], score 364222.7153925436 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0705080897445196, "ci": [1.0544504683983518, 1.0865657110906872], "job": "c4d14d62f11e4aaf"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/silu-mul-sorted-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/silu-mul-sorted-first-turn.patch`; branch `hyp/muse_prefill_layer/silu-mul-sorted-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1101.29 | 1107.19 | 1.0003 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1529.55 | 1554.34 | 1.0015 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3318.71 | 3465.97 | 0.9984 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4849.27 | 5119.82 | 0.9986 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14705.32 | 15166.77 | 0.9792 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14264.64 | 14378.67 | 0.9814 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17746.69 | 18942.77 | 0.9897 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21733.75 | 22677.63 | 0.982 | flat |

no shape moved outside noise. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: silu-mul-sorted-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

