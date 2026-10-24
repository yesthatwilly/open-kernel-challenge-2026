# muse_decode_layer · 2026-10-24T071925Z · check · moe-align-single-kernel-small-t

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-23T225609Z-moe-align-single-kernel-small-t.md` (branch `hyp/muse_decode_layer/moe-align-single-kernel-small-t`).

- **Job:** `0a0af8669ffe4cb0` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `f0f9511`

## Goal
At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one Triton kernel, a single program, that reads the T*8 int32 topk_ids and the local_start device scalar and writes everything Marlin consumes: the int32 local_ids [T, 8] (0..31 or -1, for silu_and_mul and the post-expert sum), sorted_ids (flat token*8+k indices grouped by local expert, each expert padded up to a multiple of block_size_m with the numel sentinel, in the exact layout moe_align_block_size produces), block_experts (the expert id of each m-block, -1 for none) and num_tokens_post_pad (an int32 [1] device scalar). With at most 256 routes and 32 experts this is a counting sort in registers/shared memory, one launch replacing the 3-4 launches moe_align_block_size issues (buffer fills plus its align kernel(s)) and the 1 from the id map. The buffers keep the host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts. T=128/256 keep SGLang's moe_align_block_size as is. block_size_m selection, both Marlin GEMMs, silu_and_mul and the post-expert sum are unchanged. Attribution: the T<=32 shapes only; verify_16/verify_32 are the control.

## What changed, and where
- Worktree `.worktrees/muse_decode_layer/moe-align-single-kernel-small-t`, file `muse_decode_layer/submission.py` only (1 file, +181 / -9), on the parent impl 08e8fcb6.
- New `_align_small_kernel` (Triton, grid (1,), num_warps=4): loads the T*8 int32 topk_ids and the local_start device scalar and writes local_ids [T, 8] ((id - local_start) & 255 if < 32 else -1, the parent's formula). Then it does a counting sort over the 32 local experts. A one-hot [BLOCK_N, 32] gives per-expert counts; padded = roundup(count, bsm); an exclusive cumsum gives each segment's start; an axis-0 cumsum of the one-hot gives each route's rank within its expert. Each local route stores its flat index at start + rank, so indices ascend within an expert and experts ascend, the order SGLang's small-batch path produces. Padding slots inside each segment (a masked [32, bsm] grid) and every slot past the used length get the numel sentinel T*8. block_experts is each m-block's expert, found by a [MB_BLOCK, 32] segment test, and -1 past the used blocks. num_tokens_post_pad is the used length (int32 [1], device only). The three store sets into sorted_ids are disjoint, so no ordering or barrier is needed inside the program.
- New `_align_small(topk_ids, local_start, bsm)` host wrapper: cap = T*8 + 32*(bsm - 1) and n_blocks = cdiv(cap, bsm), from T and bsm only. The constexprs BLOCK_N/CAP_BLOCK/MB_BLOCK are next_power_of_2 of n/cap/n_blocks, and BSM = bsm. n/cap/n_blocks are do_not_specialize. Nothing on the host reads any count.
- `_block_size_m(num_tokens)`: the existing selection loop moved into a helper, unchanged (T<=28 -> 8, T 29..32 -> 16).
- kernel_fn: the branch `if tokens <= ALIGN_FUSED_MAX_TOKENS (32)` is on shape only. It calls `_align_small` at the point where the parent called `_local_expert_ids` (same place in the stream, before pre_expert_norm), and passes the (sorted_ids, block_experts, n_padded) triple into `_experts(..., align)`. `_experts` skips `moe_align_block_size` when it gets that triple. T > 32 (verify_16/verify_32 and the T=33/40 tests) runs the parent's id-map kernel + `moe_align_block_size(local_ids, bsm, 256)` exactly as before. The Marlin calls, silu_and_mul (still on local_ids) and the post-expert sum are unchanged.
- prepare(): warm-compiles `_align_small` on zero ids for every T in 1..32, which covers every BLOCK_N/CAP_BLOCK/MB_BLOCK/BSM variant the branch can reach, including test T=9/24. Then `_align_self_check` runs on synthetic topk_ids (torch.Generator seed 20261024): T in {1,8,16,32} x local_start in {0,32,224} x {random, skewed, all-remote, all-one-expert, all-local}, 60 cases in all. Each case is compared with SGLang's `moe_align_block_size(ref_local, bsm, 256)` on the same ids, checking:
  - local_ids match exactly;
  - for each local expert, the set of real route indices and its block count match;
  - n_padded == bsm * the sum of SGLang's local-expert blocks;
  - used blocks are non-negative and ascending, unused blocks are -1;
  - the sentinel count is right.

  It prints one summary line plus up to 20 mismatch lines; within-expert order differences are reported as "(info)". Printing happens only in prepare(). The check never raises and never changes the path, and it sees no test routing.
- Deviation (forced by the stated capacity; an implementation detail): remote (-1) routes get no padded bucket in sorted_ids/block_experts. My understanding is that SGLang's sgl_kernel under EP shifts ids by +1 and gives -1 a bucket of its own with block_experts -1, which needs up to 33*(bsm-1) of padding. That can overflow T*8 + 32*(bsm-1). Marlin with is_ep=True skips -1 blocks, so dropping them changes only how many m-blocks get skipped, not the results. For that reason the self-check compares per local expert, not slot for slot: the block index layout differs from SGLang's whenever SGLang emits -1 blocks. Capacity stays at the spec's 32-expert sizing; the 256-expert capacity is the queued companion control.
- Not verified, because this node has no GPU, Triton or SGLang source: that the kernel compiles (an int32 tl.cumsum along axis 0 of a [BLOCK_N, 32] tensor, the [32, BSM] scatter), and what the self-check prints. Read the run log for the `[moe-align-single-kernel-small-t] self-check` line. Any mismatch not marked "(info)" means the run says nothing about the hypothesis, even if it passes. The review's open questions (which SGLang path gets replaced, per-kernel durations at T=32, Marlin's host-side m-block count) are left for the attribution step.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds one Triton launch, with buffers sized by shape (from T and bsm). There is no `.item()`/`.cpu()`/`synchronize()` or tensor print in kernel_fn's call graph; lint confirms, and every host sync it flags is in prepare()/_align_self_check. The only new branch is `tokens <= 32` (shape), and n_padded stays on device.
- [x] Nothing derived from `data` is cached across calls: no new state entries; prepare() uses synthetic ids only.
- [x] Weights: unchanged; no weight touched by this diff.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries FP8 E4M3: unchanged, the diff touches only int32 routing metadata.
- [x] Attention sees every key: unchanged (attention code untouched).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds a Triton kernel and calls SGLang's moe_align_block_size (already imported) in prepare().
- [x] All work happens inside the timed call; no output caching or replay: the align runs on every call, and the outputs of the prepare() warm-compile and self-check are thrown away.
- [x] First line `# okc: problem=muse_decode_layer` kept; no triton.autotune; prints only in prepare().

Lint (`okc_lint.py ... --problem muse_decode_layer`, exit 0):

```
warn  host-sync    L281: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L283: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L285: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L303: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L305: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L306: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L435: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L437: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/moe-align-single-kernel-small-t/muse_decode_layer/submission.py
```

Every warning is in prepare()/_align_self_check (untimed). `py_compile` OK; __pycache__ removed.

## Risks / what could fail
medium
Risk: the output contract of moe_align_block_size (sorted_ids padding value, the order of tokens within an expert, whether block_experts covers the padded capacity or only the used blocks, and whether the two Marlin calls read num_tokens_post_pad from the device) must be reproduced exactly; read sglang's moe_align_block_size.py and the Marlin kernel's expectations in the image before writing it and compare the three outputs tensor-for-tensor against SGLang's op on the 13 tests' routings in prepare() (untimed) as a self-check, printing any mismatch. Token order within an expert does not affect correctness (fp32 reduce per row) but should match anyway for a clean A/B. Depends on expert-id-map-single-kernel having landed so the id-map launch is not double counted; if that one is flat but not harmful, this still stacks on it. Do not run together with moe-align-local-expert-count.

## Result (after the run)
- job `0a0af8669ffe4cb0` (check), status done, passed True, tests 13/13
- speedup 1.0431x, 95% CI [1.0275, 1.0588], score 21685.33166202374 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.0620874943175542, "ci": [1.046156181902791, 1.0780188067323175], "job": "0730135cdcac43b7"}; base result: muse_decode_layer/hypotheses/done/neutral/expert-id-map-single-kernel.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch`; branch `hyp/muse_decode_layer/moe-align-single-kernel-small-t`
