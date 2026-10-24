---
problem: muse_moe_w4a16
slug: marlin-workspace-blocks-per-sm-8
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8)
  and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps the
  max_blocks_per_sm=4 workspace SGLang's MoE runner uses (its dense runner uses 1). The Marlin
  runner sizes the number of resident CTAs it launches from the workspace's lock count (sms *
  max_blocks_per_sm), capped by what the kernel config's shared memory allows, so a larger
  workspace lets more CTAs be resident and more k-slices run in parallel for the verify shapes
  where the w13 grid (2048 rows x 8192 cols) has work to spare. Nothing else changes.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "bs=256: more resident CTAs on a 620 us kernel that is neither at the 109 us bandwidth floor nor near compute peak (~40 GFLOP), i.e. latency/occupancy-bound"
  - shapes: bs=128 *
    direction: faster
    why: same at bs=128 (467 us vs 101 us floor)
  - shapes: bs=1 *
    direction: flat
    why: bs=1 has too few tiles to fill 132 SMs x 4 either way
  - shapes: bs=32 *
    direction: flat
    why: control
risk: low
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T032603Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T032603Z
base_sha: 3a4f9cf09a44d1c6045ca118687902eacfb2b76f
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8
worktree: .worktrees/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8
mode: check
implemented_at: 2026-10-24T032701Z
impl_commit: 8274bfd8743b4c57b172bda45394583589414306
submission_sha256: ce831bb914c2917dfa3aa1bee7c792e2b7ad9953f27a895e4a1aa25d67338589
norm_sha256: 76bb755c486de0ce3b6f91e4378d3b06b86a3013863893cb5cb57e7f2a2be54d
submission_blob: fa8d1da0a06d91f85531897cfa722574bf356738
lint_warnings: 1
attempt_file: 2026-10-24T032702Z-check-marlin-workspace-blocks-per-sm-8.md
prepared_at: 2026-10-24T032702Z
---

# marlin-workspace-blocks-per-sm-8

## Change

In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8) and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps the max_blocks_per_sm=4 workspace SGLang's MoE runner uses (its dense runner uses 1). The Marlin runner sizes the number of resident CTAs it launches from the workspace's lock count (sms * max_blocks_per_sm), capped by what the kernel config's shared memory allows, so a larger workspace lets more CTAs be resident and more k-slices run in parallel for the verify shapes where the w13 grid (2048 rows x 8192 cols) has work to spare. Nothing else changes.

## Expected

- `bs=256 *` faster: bs=256: more resident CTAs on a 620 us kernel that is neither at the 109 us bandwidth floor nor near compute peak (~40 GFLOP), i.e. latency/occupancy-bound
- `bs=128 *` faster: same at bs=128 (467 us vs 101 us floor)
- `bs=1 *` flat: bs=1 has too few tiles to fill 132 SMs x 4 either way
- `bs=32 *` flat: control

## Risk

Risks: the kernel may already cap residency by shared memory (Marlin stages use most of the 228 KB) so the knob is inert and the result flat; a larger lock array is harmless (zeroed once in prepare, reset by the kernel). If flat, record that occupancy is not Marlin's limiter at these shapes; if a gain, the ab-tune is 2 and 16, and the same workspace on w2.

## Adversarial review

Reviewed 2026-10-24T032603Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Cheap, rule-clean and unambiguous either way, but almost certainly a no-op: as far as I can reconstruct the Marlin MoE runner (vLLM's marlin_moe_wna16/ops.cu lineage, which sgl-kernel's sglang::device::marlin_moe::Marlin ports; no source on this machine), the workspace is a lock pool whose size is only CHECKED against the grid, and the grid is sms * blocks_per_sm with blocks_per_sm hard-coded to 1 whenever thread_m_blocks > 1 (block_size_m 48/64, i.e. the bs=128/256 shapes this hypothesis targets) and capped at min(allow_count, 4) otherwise, which is exactly why SGLang's MoE runner builds a 4-per-SM workspace. Run it as a low-priority quota filler; read flat as 'workspace size is not a lever' and do not spend the ab-tune (2/16, w2) unless a verify-shape gain clears the CI.

### Justification
A. Mechanism: the hypothesis asserts that moe_wna16_marlin_gemm sizes its resident-CTA count from workspace.numel(). My recollection of the upstream kernel this is a port of is the opposite: determine_exec_config() returns exec_config_t{1, th_config} as soon as thread_m_blocks > 1, and for thread_m_blocks == 1 computes allow_count = min(regs/SM / regs-per-CTA, smem/SM / (cache_size+1024)) clamped to [1, 4]; blocks = sms * blocks_per_sm; the workspace is then only validated (TORCH_CHECK numel >= required) and indexed as locks for the serial k-slice reduce. Python's marlin_make_workspace_new comment ('we use the num of threadblocks as workspace size') is about sizing the pool to the launch, not the launch to the pool. Consistent with that: the dense runner gets away with a 1-per-SM workspace because its small-M path uses atomic add (no locks), and the MoE runner's 4 matches the kernel's hard cap of 4, which is the only reason that number exists. Under this reading the knob is inert on every shape: bs=128 (block 48, thread_m_blocks 3) and bs=256 (block 64, thread_m_blocks 4) launch 132 CTAs by construction, and bs=1/8/16/32 (block 8/16, thread_m_blocks 1) are already allowed up to 4/SM with the existing workspace. Profile support: the 611.8 us at bs=256 and 495.4 us at bs=128 are the SUM of both Marlin launches (calls=2; w13 and w2 are not separable in the okc profile), so '620 us w13 kernel' overstates the target by roughly a third; the gap to the 108.9/101.3 us floor is real and Marlin at these M is plausibly latency/instruction-bound with one CTA per SM, but the lever for that is the kernel's exec config (thread_m_blocks, stages, thread_n), none of which the Python op exposes, or the per-m-block weight re-read that the in-flight marlin-block-size-m-next-larger attacks. Minor factual slips: 'w13 grid 2048 rows x 8192 cols' counts all routes, but under ep8 only local routes are computed (a few hundred real rows at bs=256; padded rows happen to be about 32 experts x 64); and bs=32 is not a control for the verify shapes, because block 16 puts it in the same thread_m_blocks==1 regime as bs=1/8/16, so the regime split is bs<=32 vs bs>=128, not bs=32 vs bs=256. Expected directions vs TASK: six equally weighted loads, so a verify-only gain of x% moves the geomean by about x/3; a gain would need >= 4-5% on both verify shapes to clear the >= +/-1.5% interval. Weakest link: the stated causal chain (workspace size -> resident CTAs) is, on my reading of the code, simply not there; the hypothesis's own risk clause anticipates 'inert' but blames shared memory, which is the wrong reason and would mislead the follow-up (an ab-tune to 16 would also be inert).

### Confounders on the current kernel
- sglang-marlin-path-parity (the only landed item): it is what introduced the max_blocks_per_sm=4 workspace (baseline's own), the 32-expert align and the whole Marlin launch sequence; nothing in it can manufacture a false gain or loss for a workspace-size change, since the SGLang side of the same run uses the identical 4-per-SM workspace and the only delta is the w13 call's workspace tensor. Its 32-expert align shrinks sorted_ids (bs=256: 18176 -> 4064 entries), which may change how many m-blocks the kernel iterates if it derives that from sorted_ids.numel(), but that affects both arms of this hypothesis equally.

### Interference and A/B needs
No companion is warranted. There is no second sub-change (w2 keeps the old workspace by design, a built-in per-call control), no landed item to revert or retune that interacts with the lock-pool size, and neither outcome is ambiguous: flat everywhere (the expected result) means the runner does not size its grid from the workspace; a gain outside the CI confined to bs=128/256 would be attributable to the change alone (nothing else differs from master, and the SGLang arm is unchanged) and would falsify my code reading; a loss is noise or at most a 4 KB extra zeroed lock array, which cannot cost a measurable amount. The hypothesis's proposed ab-tune (2 and 16, same workspace on w2) is correctly gated on a gain and should stay gated. The in-flight marlin-block-size-m-next-larger is worth noting for a later rebase: it moves bs=32 from block 16 to 32 (thread_m_blocks 1 -> 2), which flips that shape from the up-to-4-CTA/SM regime to the 1-CTA/SM regime, so if it lands first the per-shape reading of this hypothesis changes (bs=32 joins the verify-regime shapes).

### Open questions
- Source check (nothing local, SGLang is not installed here): in sgl-kernel's moe_wna16_marlin port at a55ab7d5, is blocks = sms * exec_cfg.blocks_per_sm with blocks_per_sm = 1 for thread_m_blocks > 1 and min(allow_count, 4) otherwise, and is workspace.numel() only TORCH_CHECKed against that? If the port instead derives blocks_per_sm from workspace.numel() / sms, the hypothesis is live at bs<=32 only (still inert at bs=128/256 unless the > 1 rule is also gone).
- Zero-cost probe the implementer can add in prepare() (untimed, prints allowed): call moe_wna16_marlin_gemm once on a dummy bs=4 and a dummy bs=128 input with a 1-per-SM workspace inside try/except and print the exception text; if the kernel checks workspace against its grid, the message states the required size (= blocks launched) per regime and settles the grid question without spending a run. Dummy inputs there derive from nothing in data, so the fresh-inputs rule is untouched; keep it out of kernel_fn.
- The okc profile merges the two Marlin launches (calls=2). The attempt file should state that the hypothesis's 620/467 us figures are w13+w2 combined, and if nsys grid dimensions are obtainable from the run log, record the w13 grid at bs=1 and bs=256 (132 vs 528 would confirm the regime split).
- If a verify-shape gain does appear, the next question is not 16-per-SM but why: the only way a larger lock pool speeds the kernel is if the port really does launch more CTAs, and then shared memory per CTA (max_shared_mem / blocks_per_sm - 1024) shrinks, which would change the stage count; check correctness margins on the bs=257 and skewed tests in that case.

## Implementation notes

Implemented 2026-10-24 in the worktree's `muse_moe_w4a16/submission.py` (diff: 1 file, +4/-1).

- `prepare()`: adds `state["workspace_w13"] = marlin_make_workspace(device, max_blocks_per_sm=8)` beside the existing `state["workspace"]` (`max_blocks_per_sm=4`, unchanged). Built once, untimed, depends on nothing in `data`.
- `kernel_fn()`: the w13 `moe_wna16_marlin_gemm` call passes `state["workspace_w13"]` as its workspace; the w2 call still passes `ws = state["workspace"]` (4 per SM), the built-in per-call control. Nothing else changes: same block_size_m choice, align, caches, flags, silu_and_mul and post-norm sum.
- No deviation from the spec.
- Expectation (per the adversarial review): the reviewer expects this to be **inert** on every shape. On their reading of the upstream Marlin MoE runner, the launch grid is sized by the kernel's exec config (`blocks = sms * blocks_per_sm`, with `blocks_per_sm = 1` when `thread_m_blocks > 1`, i.e. bs=128/256, and `min(allow_count, 4)` otherwise), and the workspace is only size-checked against that grid and used as the lock pool. A flat result should be read as "workspace size is not a lever", not as "shared memory caps residency"; do not spend the 2/16 ab-tune unless a verify-shape gain clears the CI.
- Note for the attempt file: the 620/467 us figures in the hypothesis are w13+w2 combined (the okc profile reports both Marlin launches together, calls=2).
- Not implemented (suggested follow-up, a separate idea): the reviewer's zero-cost prepare()-time probe, i.e. call `moe_wna16_marlin_gemm` once on dummy bs=4 and bs=128 inputs with a 1-per-SM workspace inside try/except and print the exception text, which would state the required workspace size (= blocks launched) per regime and settle the grid question without a timed run.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8 diff`.

- [x] CUDA-graph capturable: the diff only swaps which prebuilt state tensor the w13 call receives; no `.item()`/`.cpu()`/`synchronize()`/tensor prints, no allocation or control flow on input values added in kernel_fn. (The existing `torch.cuda.synchronize()` is in prepare(), unchanged.)
- [x] Nothing derived from `data` is cached across calls: the new workspace is built in prepare() from the device only.
- [x] Weights: unchanged; repacking still in prepare() via prepare_moe_nvfp4_layer_for_marlin, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (n/a for KV cache / queries: no attention in this problem).
- [x] n/a: attention-key rule does not apply (MoE problem, no attention); routing still computes every local route.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: the workspace is a zeroed lock pool reset by the kernel, not an output.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (the `__pycache__` it left was deleted).
