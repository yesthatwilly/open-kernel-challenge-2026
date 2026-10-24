---
problem: muse_decode_layer
slug: moe-align-parallel-large-t-port
created: 2026-10-24T184511Z
base: master
area: MoE align/dispatch (T>32)
change: >
  Port muse_moe_w4a16 align-parallel-per-expert-verify (patch
  muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check
  45eacf4ca2e849d2: SGLang align 10.6 / 11.4 us -> _align_parallel_kernel 4.0 / 6.9 us at bs=128 /
  256, verify_16 / verify_32 +3-4% there) to decode _experts for num_tokens > 32 only. Copy
  verbatim from the patch: ALIGN_PAR_CHUNK, ALIGN_PAR_WARPS, ALIGN_PAR_BLK, _align_parallel_kernel
  and _align_parallel. In _experts, when num_tokens > 32: replace the two master statements
  `local_ids = bank["expert_map"][topk_ids].contiguous()` and `sorted_ids, block_experts, n_padded
  = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` with `local_ids, (sorted_ids,
  block_experts, n_padded) = _align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)` (decode
  hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are
  0..31: local_start 0, count 32; the kernel writes local_ids itself, -1 for remote slots, exactly
  the expert_map semantics). block_size_m stays master pick with the cap (16 at T=128 and T=256).
  Everything downstream (cache allocation, both Marlin calls, silu_and_mul with
  expert_ids=local_ids, fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids) is
  unchanged. prepare(): compile one variant per block_size_m it can see above 32 tokens (8, 16) on
  dummy ids, then a self-check on synthetic skewed routings at T in 33, 40, 57, 128, 256, 257: per
  local expert, the set of routes in sorted_ids and its block count must equal
  moe_align_block_size(expert_map[ids], bsm, N_EXPERTS) and local_ids must be equal; ignore
  capacity and padding-count differences (LESSONS: never gate on a check that counts the capacity
  line, it always differs). On a real mismatch set state["align_par_ok"] = False and keep master;
  print "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". T <= 32 untouched.
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "SGLang align (~10.6 us in moe at bs=128) plus the expert_map gather launch -> one ~4 us launch: -6..-8 us of 439 / 601 us, +1-1.5%"
  - shapes: "*load=verify_32 *"
    direction: faster
    why: "~11.4 us + gather -> ~6.9 us: -5..-6 us of 623 / 906 us, +0.6-0.9%"
  - shapes: "*load=decode_*"
    direction: flat
    why: T <= 32 untouched
  - shapes: "*load=verify_1 *"
    direction: flat
    why: T=8 untouched
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T185035Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T185036Z
base_sha: 9f6e88b7296c037ca7b6ce685ffdaa4fa7303553
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/moe-align-parallel-large-t-port
worktree: .worktrees/muse_decode_layer/moe-align-parallel-large-t-port
mode: null
implemented_at: 2026-10-24T231252Z
impl_commit: 18c7ee4a2a60eadad634cc9604c9f5d784a5d60e
submission_sha256: 475c36ec4787179bcf460dc2d76121eab621ab0e609fd8529ff3b7cd4982d690
norm_sha256: 76b07b1ceec1bed137f3127441c4067199d8eeef1404b97abc05c18f84b4f60f
submission_blob: d2f6a92e8b7373cb05038d8797e96598f16d3cc8
lint_warnings: 28
---

# moe-align-parallel-large-t-port

## Change

Port muse_moe_w4a16 align-parallel-per-expert-verify (patch muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check 45eacf4ca2e849d2: SGLang align 10.6 / 11.4 us -> _align_parallel_kernel 4.0 / 6.9 us at bs=128 / 256, verify_16 / verify_32 +3-4% there) to decode _experts for num_tokens > 32 only. Copy verbatim from the patch: ALIGN_PAR_CHUNK, ALIGN_PAR_WARPS, ALIGN_PAR_BLK, _align_parallel_kernel and _align_parallel. In _experts, when num_tokens > 32: replace the two master statements `local_ids = bank["expert_map"][topk_ids].contiguous()` and `sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` with `local_ids, (sorted_ids, block_experts, n_padded) = _align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)` (decode hands _experts ids already shifted by (topk_ids - local_start) % 256, so local experts are 0..31: local_start 0, count 32; the kernel writes local_ids itself, -1 for remote slots, exactly the expert_map semantics). block_size_m stays master pick with the cap (16 at T=128 and T=256). Everything downstream (cache allocation, both Marlin calls, silu_and_mul with expert_ids=local_ids, fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids) is unchanged. prepare(): compile one variant per block_size_m it can see above 32 tokens (8, 16) on dummy ids, then a self-check on synthetic skewed routings at T in 33, 40, 57, 128, 256, 257: per local expert, the set of routes in sorted_ids and its block count must equal moe_align_block_size(expert_map[ids], bsm, N_EXPERTS) and local_ids must be equal; ignore capacity and padding-count differences (LESSONS: never gate on a check that counts the capacity line, it always differs). On a real mismatch set state["align_par_ok"] = False and keep master; print "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". T <= 32 untouched.

## Expected

- `*load=verify_16 *` faster: SGLang align (~10.6 us in moe at bs=128) plus the expert_map gather launch -> one ~4 us launch: -6..-8 us of 439 / 601 us, +1-1.5%
- `*load=verify_32 *` faster: ~11.4 us + gather -> ~6.9 us: -5..-6 us of 623 / 906 us, +0.6-0.9%
- `*load=decode_*` flat: T <= 32 untouched
- `*load=verify_1 *` flat: T=8 untouched

## Risk

A proven per-shape win in moe (measured on the align kernel row), ported unchanged; the headline effect (~0.3%) is under resolution, so judge on the align rows at verify_16/32. OVERLAP: moe-chain-cuda-swapab-expert-once-large-t (in judge) replaces the align at T>32 with its own route-list kernel when its path is ON; if it lands ON this is moot, if it falls back to Marlin this is the cheap T>32 fallback. Run it only after that harvest.

## Adversarial review

Reviewed 2026-10-24T185035Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a port of a kernel that was hand-landed on moe master after measuring 10.6/11.4 us of align become 4.0/6.9 us. In decode it stands on its own: the T>32 overlap is gone (moe-chain-cuda-swapab-expert-once-large-t failed tests), and the result can be read from the align and Marlin kernel rows at verify_16/32 without a companion, though the headline will be inside the CI.

### Justification
Mechanism: at T=128/256 (verify_16/32, the only shapes on the T>32 path) master runs the expert_map gather (index_elementwise) and SGLang moe_align_block_size (align kernel plus count_and_sort, and in decode with num_experts=N_EXPERTS=256, not 32). That is 3-4 launches. One 33-program Triton launch replaces them. The moe parent measured this directly: 10.6->4.0 us at bs=128 and 11.4->6.9 us at bs=256, +4.3%/+3.2% per shape, deterministic integer work. So the kernel-level claim is about as safe as a port gets. Size: Marlin is 51-70% of verify_16/32 (272-450 us), while the shapes run 447-911 us against floors of 127-267 us. Saving 4-8 us is +0.6-1.5% per verify shape and roughly +0.3-0.4% on the 6-load geomean, well inside the >=1.5% CI. The hypothesis says this itself, and the run must be judged on the kernel rows. The expected directions agree with TASK.md: verify_16/32 faster, decode_* and verify_1 (T<=32, kernel R path, DL_MAX_TOKENS=32) byte-identical and usable as an in-run drift control. Weakest link 1: the decode numbers are borrowed. FINDINGS lists only the top 3 kernels per shape, so decode's own align+gather time at T>32 is not in evidence. Decode's SGLang align runs the 256-expert variant (a 256-bin cumsum, capacity n+256*(bsm-1)), which may be slower than moe's 32-expert one (so the gain could be larger) and may size Marlin's block list differently. Read the removed rows from this run's profile, not from moe. Weakest link 2: the change keeps `(topk_ids - local_start) % N_EXPERTS` in kernel_fn (a subtract and a remainder launch) and passes local_start=0. Only the gather and the align go away, so expect 2-3 launches removed, not 4. Two smaller points: (a) bsm above 32 tokens is always 16 under the cap (T*8/32/16 < 0.9 never holds for T>=33 with bsm 8), so only the BSM=16 variant is reachable. The moe parent timed bsm 16 at bs=128 but 32 at bs=256, so the T=256 kernel time at BSM 16 (more padding and block_experts stores) is not measured, though it should be within ~1 us. (b) Capacity changes from 256-expert (n+3840 slots, 304 blocks at T=128) to 32-expert (n+480, ~94 blocks). If Marlin's grid or early-exit depends on block_experts.numel() rather than n_padded, the Marlin rows can move too. That still belongs to this change, but the harvester must compare Marlin rows base vs hyp, not only the align rows.

### Confounders on the current kernel
- marlin-block-m-cap-16: fixes bsm=16 at T=128 and T=256 (the moe parent used 16/32). This sets the kernel's padding and block counts and the BSM variant compiled. It does not confound attribution, but the T=256/BSM16 kernel time is unmeasured, and a later retune of the cap must recompile and re-check this kernel.
- sglang-pipeline-parity-router-hilo-m8: sets decode's T>32 contract (expert_map gather plus moe_align_block_size(..., N_EXPERTS=256)) that is being replaced. The new 32-expert capacity and n_padded can change what Marlin iterates over. This is a legitimate part of the change, but it is a second possible source of Marlin-row movement besides the removed launches.
- post-expert-norm-triton-small-t: gated below 32 tokens, so no timing confound at T>32. Correctness dependency only: fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids) and silu_and_mul(expert_ids=local_ids) need int32 contiguous local_ids with exactly -1 for remote routes, which program 32 writes.
- small-t-launch-minimal-dense-chain-pdl: T<=32 only (DL_MAX_TOKENS=32). It does not interact with the T>32 path. Its shapes are this run's untouched drift control.

### Interference and A/B needs
No companion is warranted. The change replaces 3-4 small integer launches on the T>32 main-stream path with one launch. No landed hypothesis sits between that and its measurement: the cap-16 pick only sets BSM, and an ab-revert or ab-tune of it would shift the new kernel by under 1 us and could not flip the reading. The headline gain (~0.3-0.4%) is below resolution, and no A/B companion fixes that. The interpretable signal is in this single run's top_kernels at verify_16/32: the new _align_parallel_kernel row against the removed index_elementwise / moe_align_block_size_kernel / count_and_sort rows from the base profile (81ee6714b7614a99), plus a check that the Marlin rows did not move in the wrong direction. T<=32 shapes are byte-identical and serve as the drift control. A control that splits the gather removal from the align replacement is pointless, because one kernel does both. Harvest rule: hand-land if the log shows '[align-parallel-large-t] ON' with 0 real mismatches, the new row is at or below the sum of the removed rows minus ~2 us at both verify shapes, and the verify_16/32 paired ratios are at or above base, even if classify says neutral (as the moe parent did).

### Open questions
- How long do decode's own gather + 256-expert moe_align_block_size + count_and_sort take at T=128/256? Take this from the base profile's full kernel list, not from FINDINGS' top 3. It is the denominator for judging the new row.
- Is Marlin's work bounded by n_padded on device or by block_experts.numel() / sorted_ids.numel()? If the latter, shrinking capacity from 256-expert to 32-expert sizing could save Marlin time too, and the Marlin rows should be read as part of this change.
- Follow-up, not this run: load local_start from data['local_start'] (an int32 [1] device tensor) inside the kernel and consume raw global topk_ids. That would also drop the subtract and remainder launches (~2 more graph nodes), as kernel R already does at T<=32.
- Should the self-check also run the Marlin y comparison (y_new vs y_ref via _experts with both aligns), as _dl_self_check does for kernel R? That would catch a layout error that the structural check misses, at a few seconds of prepare time.
- The test case 'batch 33, ctx 20000, layer full' (T=33, bsm 16) and the q_len-8 tests at T=24..40 exercise the new path at small n. Confirm the 'ON' line in the log before trusting the timings.


## Implementation notes

Worktree muse_decode_layer/submission.py only (+188/-1). Added ALIGN_PAR_* constants, `_align_parallel_kernel` and `_align_parallel` copied verbatim from muse_moe_w4a16 master (docstring of the wrapper reworded only). `_experts`: when `align is None`, num_tokens >= 33 and `state["align_par_ok"]`, the expert_map gather and moe_align_block_size are replaced by `_align_parallel(topk_ids, 0, LOCAL_EXPERTS, block_size_m)`; otherwise master statements unchanged. T<=32 never reaches the new branch (dense chain path and its master fallback byte-identical). `_align_par_setup` (called in prepare() after `_dl_setup`) warms every block_size_m reachable above 32 tokens (computed from `_block_size_m` over T 33..1024, plus 8 and 16; in practice 16) for int32 and int64 id dtypes, then self-checks T in 33, 40, 57, 128, 256, 257 on random / skewed / remote-heavy routings: local_ids equality, per-expert route set and block count vs moe_align_block_size(expert_map[ids], bsm, 256); capacity/padding ignored. Mismatch or exception -> align_par_ok False (master kept). Prints "[align-parallel-large-t] ON|OFF cases=.. mismatches=..". Deviation: none in the kernel; self-check is structural only (no Marlin y comparison, per review open question left as optional).

## Rules check

- [x] CUDA-graph capturable: new kernel_fn path only launches a Triton kernel and torch.empty allocs on shapes; host syncs/.cpu() are only in prepare()-time setup (checked in diff).
- [x] Nothing derived from `data` is cached across calls (state gets only the prepare-time bool align_par_ok).
- [x] Weights: n/a: no weight changes in the diff.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8: n/a: integer routing work only, no dtype changes.
- [x] Attention sees every key: n/a: attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (diff checked).
- [x] All work happens inside the timed call; no output caching or replay (align computed per call from topk_ids).

okc_lint: "OK with warnings: muse_decode_layer/submission.py" (only host-sync warnings, all in prepare()-time code, noted as fine). py_compile clean, __pycache__ removed.
