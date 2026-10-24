---
problem: muse_moe_w4a16
slug: align-single-token-ep
created: 2026-10-23T225502Z
base: master
area: MoE align/dispatch
change: >
  At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from
  sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-mapped ids (-1 for remote
  experts) instead of moe_align_block_size. baseline.py takes this path only when expert_map is
  None (the 'full' sharding), never under EP; this hypothesis tests whether the single-token
  kernel tolerates -1 ids and is cheaper than the general align (which at 256 or 32 experts runs a
  histogram + cumsum + scatter over a padded sorted_ids buffer) for the 8 ids of one token. Its
  outputs (sorted_ids, block_experts, n_padded) feed both Marlin GEMMs unchanged; all other shapes
  keep moe_align_block_size.
expected:
  - shapes: bs=1 *
    direction: faster
    why: bs=1 is 77 us; the general align is 2-3 launches over a 32- or 256-expert histogram for 8 ids, the single-token kernel is one small launch
  - shapes: bs=8 *
    direction: flat
    why: "control: num_tokens=8 keeps moe_align_block_size"
  - shapes: bs=16 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T032411Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-24T032412Z
base_sha: 7c54c225a5de1458c4cd57d1fea92d4b3895d9f5
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/align-single-token-ep
worktree: .worktrees/muse_moe_w4a16/align-single-token-ep
mode: check
implemented_at: 2026-10-24T032534Z
impl_commit: 626e602e8c42555cd6e31cc1c17c4714b410deea
submission_sha256: cf85d53e18535b069939208616928ffd52a8310ee5d4e0241d53905f2b5a48c7
norm_sha256: 43c2b0c4373e5cfb2c37d4e20dc791c706e54ff607c5da6d9fd53f08ab85515d
submission_blob: 6075ec2ba91b451ab6510be0763073337419e243
lint_warnings: 1
attempt_file: 2026-10-24T032534Z-check-align-single-token-ep.md
prepared_at: 2026-10-24T032534Z
job: a7bb0cecaf8d4de1
dispatched_at: 2026-10-24T032541Z
---

# align-single-token-ep

## Change

At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-mapped ids (-1 for remote experts) instead of moe_align_block_size. baseline.py takes this path only when expert_map is None (the 'full' sharding), never under EP; this hypothesis tests whether the single-token kernel tolerates -1 ids and is cheaper than the general align (which at 256 or 32 experts runs a histogram + cumsum + scatter over a padded sorted_ids buffer) for the 8 ids of one token. Its outputs (sorted_ids, block_experts, n_padded) feed both Marlin GEMMs unchanged; all other shapes keep moe_align_block_size.

## Expected

- `bs=1 *` faster: bs=1 is 77 us; the general align is 2-3 launches over a 32- or 256-expert histogram for 8 ids, the single-token kernel is one small launch
- `bs=8 *` flat: control: num_tokens=8 keeps moe_align_block_size
- `bs=16 *` flat: control: untouched

## Risk

Risks: if moe_align_single_token does not skip -1 ids, the bs=1 pre-timing test fails outright (a cheap, informative failure: log it as a do-not-retry). Superseded on bs=1 if triton-w2-grouped-gemv-small-m lands, since that removes the align from the small-M path; still worth a run as a low-cost probe of SGLang's own fast path. No rule exposure.

## Adversarial review

Reviewed 2026-10-24T032411Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** A cheap, well-isolated probe of SGLang's single-token align under EP, but its stated cost model is wrong: on master the bs=1 align is already SGLang's one-launch small-batch kernel over 32 experts (below 4.9 us, not in the bs=1 top-4), so the realistic upside is ~2-4 us of 80 us on decode_1 only (<1% overall, inside the +/-1.5% CI). Single result is interpretable from the per-shape bs=1 number and its top-kernel list; no companion needed.

### Justification
Mechanism: for one token the 8 ids are distinct, so a single-token align needs no histogram/cumsum and can write 8 blocks (one per id) directly; that is plausibly a smaller kernel than moe_align_block_size's small-batch kernel. But the hypothesis overstates what it replaces. (1) FINDINGS/BEST.json top_kernels for bs=1 on master are Marlin 41.1 us, at::reduce_kernel 5.6, unrolled_elementwise 5.4, elementwise 4.9 -- the align kernel is not in the top four, so it costs <4.9 us; at bs=32 the same kernel (moe_align_block_size_small_batch_expert_kernel) is 6.7 us. (2) The landed parity patch already shrank the align to 32 experts and (per its own md) dropped sorted_ids under the 4096 fused-padding threshold, so 'a 256-expert histogram, 2-3 launches' does not describe master; it is one small-batch launch over 32 counters plus possibly a zeros() for the cumsum buffer. (3) Under EP the single-token kernel will, if it tolerates -1 at all, most likely emit 8 blocks with block_experts = -1 for the 6-7 remote slots (n_padded = 8*block_size_m = 64 rows at block 8) where the general align emits only the 1-2 local blocks; Marlin with is_ep=True skips expert -1 blocks, but it now iterates 8 m-blocks instead of 1-2, so a few hundred ns could come back on the Marlin side. Expected shapes/directions are consistent with TASK.md (only bs=1 changes code; bs=8/16 are true controls since num_tokens==1 is the branch), and the bs=1 pre-timing test exercises exactly this path so correctness is covered. Magnitude: 2-4 us on a 60-weight shape at 80.2 us is 2.5-5% on decode_1, roughly 0.4-0.8% on the geomean -- detectable per shape (bs=1 err_ns is 479 ns, ~0.6%) but invisible in the overall CI, so this run is a per-shape learning, not a submit candidate. Weakest link: the gain budget is the align kernel's own few microseconds, and master already runs its cheapest variant; the hypothesis has not measured what it is removing. Also note master is 3.2 us SLOWER than SGLang at bs=1 (80.2 vs 77.0, outside the 0.48 us error) even with the 32-expert align, so even a full success here only restores parity on decode_1 and the unexplained 16 us of generic at::native kernels (reduce_kernel, unrolled/elementwise) at bs=1 is the bigger target.

### Confounders on the current kernel
- sglang-marlin-path-parity (the only landed item) changed moe_align_block_size's num_experts from 256 to 32, which is the exact kernel this hypothesis replaces at bs=1: the measured delta is 'single-token align vs 32-expert small-batch align', smaller than the 'vs 256-expert align' the hypothesis text argues from. This does not confound attribution (the 32-expert align is master and the right comparator); it only shrinks the expected gain. A flat result means the single-token kernel is not cheaper than the 32-expert small-batch kernel, which is the question that matters for master.
- sglang-marlin-path-parity also fixed is_ep=True, block_size_m=8 at bs=1 and the fused sorted_ids padding; under the single-token path n_padded becomes 8*block_size_m regardless of how many local experts the token hits, so any loss could sit in Marlin's m-block loop (8 blocks, 6-7 skipped as -1) rather than in the align itself. The result's per-kernel list for bs=1 separates these without another run: Marlin time up => block-count cost; align kernel time down => the align won.

### Interference and A/B needs
No companion warranted. The change is a single kernel swap on one shape with two in-file controls (bs=8, bs=16 keep the general align), and both outcomes are interpretable: a gain on decode_1 is attributable to the swap (nothing else differs from master), and a loss is attributable to the swap plus the Marlin 8-block side effect, which the profile's per-kernel times resolve. An ab-revert of the parity align count (256 experts) would answer 'does single-token beat the 256-expert align', which no longer matters since 32 is master. In-flight marlin-block-size-m-next-larger (block 16 at bs=1) does not change the picture (n_padded 128 vs 64, still 8 blocks); in-flight triton-w13-grouped-gemv-small-m and queued triton-w2-grouped-gemv-small-m would supersede this at bs<=8 if they land (the align leaves the small-M path entirely once w2 moves), which the hypothesis already states -- so merge ordering, not a companion, decides whether this stays relevant. A control isolating 'align cost' from 'Marlin 8-block cost' would need a compaction kernel that is itself a new change; not worth a run.

### Open questions
- Does moe_align_single_token exist in the judge's SGLang build and accept ids containing -1? Nothing in this problem has ever executed that import (baseline reaches it only when expert_map is None, never under ep8). An ImportError or a -1 index fault fails the bs=1 pre-timing test and should be logged as do-not-retry.
- What does the single-token kernel emit for a -1 id: a block with block_experts=-1 (Marlin is_ep skips it), or does it drop the slot and compact (n_padded = local_count*block_size_m)? The former costs Marlin 6-7 skipped blocks; the latter is the ideal. Read SGLang's source before building if available.
- What is the align kernel's actual time at bs=1 on master? It is absent from the top-4 (<4.9 us); the hypothesis's '2-3 launches over a 256-expert histogram' describes neither master (32 experts, fused padding) nor the small-batch kernel variant that profiles show. Get the number from the full profile txt of fb2cc736613747df if it is saved.
- Why is master 3.2 us slower than SGLang at bs=1 (80.2 vs 77.0 us, err 0.48 us) despite the 32-expert align that was expected to make bs=1 faster? If the 32-expert small-batch kernel is for some reason slower than the 256-expert one at bs=1 (e.g. different branch of the wrapper), this hypothesis is testing on a shape whose align regression is unexplained.
- The ~16 us of at::native reduce_kernel / unrolled_elementwise / elementwise kernels at bs=1 (20% of the shape) are a larger target than the align; what are they (the .to(int32)/.to(float32) copies, the expert_map gather, the Marlin wrapper's workspace zeroing, torch.zeros for the align cumsum)?


## Implementation notes

- Worktree `.worktrees/muse_moe_w4a16/align-single-token-ep`, file `muse_moe_w4a16/submission.py` only (diff stat: 1 file, +19 / -3).
- kernel_fn align branch: `if num_tokens == 1 and expert_map is None:` became `if num_tokens == 1 and moe_align_single_token is not None:`. The branch is shape-only (num_tokens) plus a module-level constant; it now fires under ep8. It is called exactly as baseline.py calls it, `moe_align_single_token(local_ids, block_size_m)`, on the expert_map-mapped ids (-1 for remote experts). sorted_ids / block_experts / n_padded go straight into both Marlin GEMMs unchanged; n_padded is never read on the host. All other shapes keep `moe_align_block_size(local_ids, block_size_m, bank["count"])`.
- Import hoisted from inside kernel_fn to module level, guarded with try/except ImportError (per review). On success it prints `[align-single-token-ep] bs=1 path: moe_align_single_token`; on failure it sets the name to None, prints `... unavailable (<err>); bs=1 path: moe_align_block_size (fallback)`, and bs=1 runs master's path. Check log_tail for which line appears: a fallback run is a master re-run on bs=1 and says nothing about the hypothesis.
- Docstring gained one paragraph describing the change.
- Deviation: none from the change paragraph. Not verified (no SGLang source or GPU on this node): whether the kernel tolerates -1 ids and whether it compacts or emits -1 blocks. The bs=1 pre-timing test covers correctness; the bs=1 per-kernel profile separates align time from Marlin m-block cost (see review).

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; the new branch is on `num_tokens` (shape) and a module constant; n_padded stays on device. The prints are at import time, strings only.
- [x] Nothing derived from `data` is cached across calls (no new state; module global is set at import from an import result only).
- [x] Weights: unchanged; prepare() untouched, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged; only the int32 align outputs differ. KV cache / queries: n/a, no attention in this problem.
- [x] Attention sees every key: n/a, MoE problem with no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only addition is an import of an SGLang op baseline.py itself imports.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`, exit 0):

```
warn  host-sync    L126: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/align-single-token-ep/muse_moe_w4a16/submission.py
```

The warning is the pre-existing synchronize() in prepare(), untouched by this diff. `py_compile` OK; __pycache__ removed.
