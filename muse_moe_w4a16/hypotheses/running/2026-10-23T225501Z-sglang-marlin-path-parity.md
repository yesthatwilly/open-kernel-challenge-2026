---
problem: muse_moe_w4a16
slug: sglang-marlin-path-parity
created: 2026-10-23T225501Z
base: master
area: MoE align/dispatch
change: >
  Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call,
  x @ w13.T over all tokens, masked weighted sum) with SGLang's Marlin pipeline from baseline.py:
  prepare() builds the Marlin bank with prepare_moe_nvfp4_layer_for_marlin over experts
  [local_start, local_start+local_count), the expert_map (-1 for remote experts),
  marlin_make_workspace(device, max_blocks_per_sm=4) and norm_w = (post_norm_w + 1) in bf16;
  kernel_fn runs local_ids = expert_map[topk_ids], the BLOCK_SIZE_M_CANDIDATES 0.9-ratio pick,
  moe_align_block_size, moe_wna16_marlin_gemm for w13 (top_k=8, size_m=num_tokens, size_n=2I),
  silu_and_mul(expert_ids=local_ids), moe_wna16_marlin_gemm for w2 (top_k=1, size_m=num_tokens*8)
  and fused_post_expert_norm_weighted_sum(weight_first=True, slot_expert_ids=local_ids), with
  is_ep=True, use_fp32_reduce=True, use_atomic_add=False, exactly as baseline.py. ONE structural
  change so this is not a verbatim port: moe_align_block_size is called with
  num_experts=bank['count'] (32, the local bank) instead of N_EXPERTS=256. The ids it receives are
  already mapped to [-1] U [0,32) by expert_map, so no id is >= 32 and the only out-of-range value
  is -1, which the baseline already relies on the align kernel to skip at num_experts=256.
  Effects: the align kernel's per-expert cumsum shrinks 256 -> 32 entries, sorted_ids shrinks from
  M*8 + 256*(block_size_m-1) to M*8 + 32*(block_size_m-1) entries (at bs=256, block 64: 18176 ->
  4064, which drops under SGLang's 4096 fuse_sorted_ids_padding threshold and removes the separate
  sorted_ids.fill_ launch), and Marlin's block_experts array shrinks accordingly. Attribution: the
  whole move from 0.006x to ~1.0x is the Marlin pipeline; anything above ~1.0x (expected a few us
  per shape, largest in relative terms at bs=1) is the 32-expert align. Fallback if the align
  wrapper refuses num_experts < 256 or a test fails on it: keep 256 and instead differ from
  baseline.py by hoisting the per-call block_size_m pick and the `common` kwargs into a bs-keyed
  table built in prepare() (host-only, zero GPU effect), and say so in the attempt file so the
  result reads as pure parity.
expected:
  - shapes: "*"
    direction: faster
    why: "starter dequantizes all 32 experts to BF16 every call (index/elementwise kernels are 80% of 37 ms); SGLang's Marlin W4A16 path reads NVFP4 once and only for routed blocks: ~0.006x -> ~1.0x everywhere"
  - shapes: bs=1 *
    direction: faster
    why: "parity plus the 32-expert align: at 77 us total, a shorter align cumsum and smaller sorted_ids are a visible fraction"
  - shapes: bs=256 *
    direction: faster
    why: "parity plus: sorted_ids 18176 -> 4064 entries drops under the 4096 fused-padding threshold, removing the sorted_ids.fill_ launch that bs=256 alone pays"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-23T230616Z
review_companions: null
stage: implemented
claimed_at: 2026-10-23T230616Z
base_sha: 4739730c9a8a22229401f2e07ef91550db8b0152
base_blob: 8eeb6a4778875e37a1d9c55a5e27ee6079999847
branch: hyp/muse_moe_w4a16/sglang-marlin-path-parity
worktree: .worktrees/muse_moe_w4a16/sglang-marlin-path-parity
mode: check
implemented_at: 2026-10-23T231147Z
impl_commit: 6b5469bd92f0a43ce5908912b02fb8dee425c3fa
submission_sha256: 149fe060036b35682f97b6bc5d6e9a1d8e206c9e925a7eba736043723613a5c3
norm_sha256: 5ccbf8c8b17a3a86b37c90e49b77f70e05f67e397aa52c1c8ff73cb713084c58
submission_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
lint_warnings: 1
attempt_file: 2026-10-23T231147Z-check-sglang-marlin-path-parity.md
prepared_at: 2026-10-23T231147Z
---

# sglang-marlin-path-parity

## Change

Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's Marlin pipeline from baseline.py: prepare() builds the Marlin bank with prepare_moe_nvfp4_layer_for_marlin over experts [local_start, local_start+local_count), the expert_map (-1 for remote experts), marlin_make_workspace(device, max_blocks_per_sm=4) and norm_w = (post_norm_w + 1) in bf16; kernel_fn runs local_ids = expert_map[topk_ids], the BLOCK_SIZE_M_CANDIDATES 0.9-ratio pick, moe_align_block_size, moe_wna16_marlin_gemm for w13 (top_k=8, size_m=num_tokens, size_n=2I), silu_and_mul(expert_ids=local_ids), moe_wna16_marlin_gemm for w2 (top_k=1, size_m=num_tokens*8) and fused_post_expert_norm_weighted_sum(weight_first=True, slot_expert_ids=local_ids), with is_ep=True, use_fp32_reduce=True, use_atomic_add=False, exactly as baseline.py. ONE structural change so this is not a verbatim port: moe_align_block_size is called with num_experts=bank['count'] (32, the local bank) instead of N_EXPERTS=256. The ids it receives are already mapped to [-1] U [0,32) by expert_map, so no id is >= 32 and the only out-of-range value is -1, which the baseline already relies on the align kernel to skip at num_experts=256. Effects: the align kernel's per-expert cumsum shrinks 256 -> 32 entries, sorted_ids shrinks from M*8 + 256*(block_size_m-1) to M*8 + 32*(block_size_m-1) entries (at bs=256, block 64: 18176 -> 4064, which drops under SGLang's 4096 fuse_sorted_ids_padding threshold and removes the separate sorted_ids.fill_ launch), and Marlin's block_experts array shrinks accordingly. Attribution: the whole move from 0.006x to ~1.0x is the Marlin pipeline; anything above ~1.0x (expected a few us per shape, largest in relative terms at bs=1) is the 32-expert align. Fallback if the align wrapper refuses num_experts < 256 or a test fails on it: keep 256 and instead differ from baseline.py by hoisting the per-call block_size_m pick and the `common` kwargs into a bs-keyed table built in prepare() (host-only, zero GPU effect), and say so in the attempt file so the result reads as pure parity.

## Expected

- `*` faster: starter dequantizes all 32 experts to BF16 every call (index/elementwise kernels are 80% of 37 ms); SGLang's Marlin W4A16 path reads NVFP4 once and only for routed blocks: ~0.006x -> ~1.0x everywhere
- `bs=1 *` faster: parity plus the 32-expert align: at 77 us total, a shorter align cumsum and smaller sorted_ids are a visible fraction
- `bs=256 *` faster: parity plus: sorted_ids 18176 -> 4064 entries drops under the 4096 fused-padding threshold, removing the sorted_ids.fill_ launch that bs=256 alone pays

## Risk

Parity base for every other muse_moe_w4a16 hypothesis. Risks: (1) sgl_kernel's moe_align_block_size must skip -1 ids at num_experts=32 exactly as it does at 256 (same code path, same -1 values; the five pre-timing tests, incl. bs=1 and dist=skewed, catch any miscount); (2) the SGLang wrapper may have a num_experts-dependent branch (older versions switched kernels at >=224 experts) so the 32 path could be a different, possibly slower align kernel: then the result is still parity +/- a few us and the attempt file must record which branch ran; (3) correctness is Marlin's own (BF16 activations, fp32 reduce), the same the judge's SGLang side passes. Rules: NVFP4 stays NVFP4 (Marlin repack is lossless), activations bf16 where SGLang keeps them, nothing derived from data is cached, no host syncs (block_size_m branches on num_tokens, a shape). Commit-step dedup: the num_experts change alters the stripped AST, so this does not collide with baseline.py.

## Adversarial review

Reviewed 2026-10-23T230616Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** The 32-expert moe_align_block_size call is a genuine GPU-side change (shorter cumsum, 4.5x smaller sorted_ids/block_experts, and at bs=128 and bs=256 the removal of the separate sorted_ids.fill_ launch), not a cosmetic way past the dedup, and the judge's own SGLang column is a built-in A/B control for it because it is the only difference from baseline.py. Its expected signal is below the judge's +/-1.5% resolution on every shape, so the run should be read as 'parity base established' and the align-32 effect as 'not resolved' unless |speedup-1| clears the CI; no companion can sharpen that.

### Justification
A. Mechanism: with expert_map already mapping ids to [-1] U [0,32), passing num_experts=32 shrinks the align kernel's per-expert histogram/cumsum from 256 to 32 entries and sorted_ids from M*8+256*(bsm-1) to M*8+32*(bsm-1). Recomputing the baseline's block_size_m pick (ratio = M/4/bsm, break on first < 0.9): bs=1/8/16 -> 8, bs=32 -> 16, bs=128 -> 48, bs=256 -> 64. sorted_ids at 256 vs 32 experts: bs=1 1800->232, bs=8 1856->288, bs=16 1920->352, bs=32 4096->736, bs=128 13056->2528, bs=256 18176->4064. So the fuse_sorted_ids_padding threshold (<=4096, consistent with my recollection of sglang's wrapper, unverified locally) is crossed at bs=128 as well as bs=256; the hypothesis's expected list omits bs=128 and should be read as covering both verify shapes. The 4064 figure is only 32 below the threshold: if this SGLang commit sizes the buffer as (num_experts+1)*(block_size-1) (a variant that has existed upstream) it is 4127 and bs=256 does not cross; this is unverified. Magnitude: one fill_ of 72 KB (~2-3 us launch) out of 620 us at bs=256 is ~0.4%; at bs=128 ~0.5%; at bs=1 a shorter cumsum inside the same kernel is at most ~1-2 us of 77 us. All of these sit inside the judge's minimum +/-1.5% interval, and the geometric mean over six equally weighted loads of sub-1% effects is also inside it. FINDINGS has no profile of the Marlin pipeline (only the starter at 0.006x, where index/elementwise kernels dominate), so there is no measured align-kernel time to lean on; the SoL gap (SGLang 77 us vs 11.4 us floor at bs=1, 620 vs 109 at bs=256) is dominated by the two Marlin GEMMs, not align. Weakest link: the attribution rule 'anything above ~1.0x is the 32-expert align' is not supportable, because the align delta is below the judge's resolution; a result of 1.01x or 0.99x is noise. The redeeming property is that the SGLang side of the same run is baseline.py, which differs from this submission only in the num_experts argument, so speedup-1.0 (with its CI) IS the align-32 effect, measured under identical graph-replay, cold-L2, order-swapped pairing. Correctness: the ids passed to align are the expert_map-mapped local ids (the baseline passes the same tensor), -1 marks a remote expert, and the baseline's judge-side correctness at num_experts=256 is the only evidence the sgl_kernel align handles -1; whether that handling is independent of num_experts is unverified (SGLang's wrapper historically branched kernels at num_experts >= 224, and a small-num_experts kernel without a -1 guard would index shared memory at [-1], which could show up as extra empty blocks on local expert 31 (slower only) or a bad sorted_ids scatter (wrong output or OOB)). The five pre-timing tests, including bs=1 and dist=skewed, are the gate; a test failure here must not be read as a Marlin problem. Marlin indexes the 32-expert bank by block_experts values in [0,32) either way, so the GEMMs are unaffected by the change except possibly through a smaller max m-block count if the kernel sizes its loop from sorted_ids.numel() (unverified; if so, bs=128/256 get a second small effect). Unverified SGLang-source claims in the hypothesis: the <=4096 fused-padding threshold and its operator, the sorted_ids sizing formula, the -1 handling in the align kernel at 32 experts, and whether the wrapper still has a num_experts-dependent kernel branch at this commit.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
Nothing is landed on master (landed: []), so ab-revert and ab-tune have no target. The only sub-change is the num_experts argument, and its control arm (num_experts=256, i.e. baseline.py) is supplied by the judge for free on every shape in the same run, under the same timing protocol; a `control` companion would be verbatim baseline.py (exempt from the dedup, but it re-measures harness symmetry, already known from the muse_bf16_proj verbatim-baseline check at 1.0028x [0.988, 1.018]) and could not resolve a sub-1% effect against a >=+/-1.5% interval anyway. Neither a gain nor a loss is ambiguous: 0.006x -> ~1.0x is the Marlin pipeline; any deviation from 1.0x outside the CI is the align-32 call and nothing else. No companions.

### Open questions
- At SGLang commit a55ab7d, does moe_align_block_size (fused_moe_triton) still branch on num_experts (historically >= 224 selected sgl_kernel's kernel, else a vLLM-style kernel)? The attempt file should record which kernel ran at 32 (nsys/profile kernel name) so a later align hypothesis is not misattributed.
- Does the sgl_kernel align kernel guard expert_id < 0 explicitly, or does -1 work at 256 by a benign out-of-range path that changes meaning at 32? Only the tests answer this; a failure should be filed against the align call, not the pipeline.
- Exact sorted_ids sizing formula and the fused-padding comparison operator at this commit: 4064 vs a 4096 threshold leaves 32 entries of margin, and (num_experts+1)*(block_size-1) would put bs=256 back over it.
- Does moe_wna16_marlin_gemm size its m-block loop or grid from sorted_ids.numel() (284 -> 64 max blocks at bs=256) or only from num_tokens_post_padded on device? If the former, the verify shapes get a second, slightly larger effect that the hypothesis does not claim.
- The hypothesis's expected list should add bs=128 to the fill_-removal shapes (13056 -> 2528 entries also crosses the threshold); as written it attributes that effect to bs=256 alone.

## Implementation notes

Ported baseline.py (SGLang a55ab7d5, MuseSpark1xFp4Experts._forward_marlin after the pre-expert norm) into the worktree's `muse_moe_w4a16/submission.py`, keeping the starter's first line, module layout and the Weights/Config/Inputs TypedDicts, and the `prepare(weights)` / `kernel_fn(data, state)` signatures (annotated `prepare(weights: Weights) -> dict`, `kernel_fn(data: Inputs, state: dict) -> torch.Tensor`; the dedup strips annotations and docstrings).

- `prepare()`: identical to baseline.py. `_marlin_layer` builds the SimpleNamespace that `prepare_moe_nvfp4_layer_for_marlin` reads for experts [0, 32) (SGLang's own lossless Marlin repack of w13/w2 + block scales + per-expert global scales), the bank dict keyed `(start, count)` with `expert_map` (-1 for remote experts, `arange(32)` for local), `marlin_make_workspace(device, max_blocks_per_sm=4)`, and `norm_w = (post_norm_w.float() + 1).to(bf16)`. The `torch.cuda.synchronize()` there is baseline.py's, untimed.
- `kernel_fn()`: identical to baseline.py except the one change. `local_ids = expert_map[topk_ids.int()]`, fp32 contiguous topk_weights, the BLOCK_SIZE_M_CANDIDATES pick (`num_tokens*8/32/bsm < 0.9` -> break; a shape-only branch), the (dead under ep8) `moe_align_single_token` branch, `cache2`/`cache13` scratch, `common` kwargs (`is_ep=True`, `use_fp32_reduce=True`, `use_atomic_add=False`, `b_q_type=float4_e2m1f`, `is_k_full=True`, `mul_topk_weights=False`, `is_zp_float=False`), `moe_wna16_marlin_gemm` w13 (`top_k=8, size_m=M, size_n=2I, size_k=H`), `silu_and_mul(expert_ids=local_ids.view(-1), expert_step=1)`, `moe_wna16_marlin_gemm` w2 (`top_k=1, size_m=M*8, size_n=H, size_k=I`), `fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=norm_w, eps=1e-8, weight_first=True, slot_expert_ids=local_ids)`.
- The one change (the only line whose stripped AST differs from baseline.py), at the align call in `kernel_fn`:
  - baseline: `moe_align_block_size(local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])`
  - here: `moe_align_block_size(local_ids, block_size_m, bank["count"])` i.e. 32 under ep8.
  Buffer sizing is not done by the caller: SGLang's `fused_moe_triton.moe_align_block_size` wrapper allocates `sorted_ids`/`expert_ids` (block_experts) from `num_experts` itself, so passing 32 is exactly "sized as the wrapper would for 32 experts" with no further code. The hypothesis's fallback (hoisting host-only kwargs) was not needed and was not used.
- `diff` of the code bodies (from `from typing` down) against baseline.py shows only: the two signature annotations, two function docstrings, a three-line comment, and the align argument. `churn.norm_sha256` (docstring/annotation/TypedDict-stripped AST) differs from both baseline.py (5bbaa397...) and the starter (d0428697...): submission 5ccbf8c8....
- Which align kernel path runs at 32: cannot be determined from code on this machine (no SGLang / sgl_kernel source installed; the only local SGLang text is baseline.py). Recollection, unverified: at this era the `fused_moe_triton` wrapper calls `sgl_kernel.moe_align_block_size` for every expert count (the old `>= 224` kernel switch is gone), sizes `sorted_ids` as `topk_ids.numel() + num_experts * (block_size - 1)` (an `(num_experts + 1)` variant has existed upstream) and sets `fuse_sorted_ids_padding = sorted_ids.numel() <= 4096`, else runs `sorted_ids.fill_(topk_ids.numel())` as a separate launch. The attempt file must record the align kernel name and whether a `fill_` launch appears in the run's profile at bs=128/256, since that is the only way to settle it here.
- Not run (no GPU/SGLang locally): correctness rests on the judge's five pre-timing tests; a failure must be filed against the align-32 call, not the Marlin pipeline (see Adversarial review).

## Rules check

Verified against `git diff -- muse_moe_w4a16/submission.py` in the worktree (165 lines: 130 insertions, 35 deletions, one file).

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in `kernel_fn` or anything it calls; the only `synchronize()` is baseline.py's in untimed `prepare()`; the one branch in `kernel_fn` (`block_size_m` pick, `expert_map is None`) is on `num_tokens` (a shape) and a prepare-time constant.
- [x] Nothing derived from `data` is cached across calls: `kernel_fn` writes only locals; `state` is read, never written; no module-level containers.
- [x] Weights: repacked losslessly in `prepare()` only, via SGLang's `prepare_moe_nvfp4_layer_for_marlin` (NVFP4 E2M1 + E4M3 block scales + fp32 global scales stay as they are, only re-laid out); `norm_w` is the bf16 post-norm weight plus one in bf16, the parameterization SGLang itself binds.
- [x] Activations BF16 where SGLang keeps BF16: `x`, `cache1/2/3` are `x.dtype` (bf16), the Marlin GEMMs reduce in fp32 (`use_fp32_reduce=True`) as SGLang does, `topk_weights` stays fp32; no KV cache or queries in this problem.
- [x] Attention sees every key: not applicable (no attention in this problem); all 8 routed experts per token are passed to align/GEMM/weighted sum, no expert or token skipping beyond the ep8 remote-expert mask (-1) that the task defines.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: imports are torch, sgl_kernel and sglang only (plus `types`); no `reference`/`checks`/`baseline` import; lint `harness`/`process`/`network` rules clean.
- [x] All work happens inside the timed call: `kernel_fn` computes the full output from fresh inputs every call; `prepare()` holds only weight-derived state.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_moe_w4a16/submission.py --problem muse_moe_w4a16` -> `warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())` / `OK with warnings: muse_moe_w4a16/submission.py` (exit 0).
