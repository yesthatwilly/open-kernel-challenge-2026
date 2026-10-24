---
problem: muse_decode_layer
slug: moe-align-single-kernel-small-t
created: 2026-10-23T225609Z
base: master
area: MoE align/dispatch
change: >
  At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the
  expert-id map kernel from expert-id-map-single-kernel with one Triton kernel, a single program,
  that reads the T*8 int32 topk_ids and the local_start device scalar and writes everything Marlin
  consumes: the int32 local_ids [T, 8] (0..31 or -1, for silu_and_mul and the post-expert sum),
  sorted_ids (flat token*8+k indices grouped by local expert, each expert padded up to a multiple
  of block_size_m with the numel sentinel, in the exact layout moe_align_block_size produces),
  block_experts (the expert id of each m-block, -1 for none) and num_tokens_post_pad (an int32 [1]
  device scalar). With at most 256 routes and 32 experts this is a counting sort in
  registers/shared memory, one launch replacing the 3-4 launches moe_align_block_size issues
  (buffer fills plus its align kernel(s)) and the 1 from the id map. The buffers keep the
  host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts.
  T=128/256 keep SGLang's moe_align_block_size as is. block_size_m selection, both Marlin GEMMs,
  silu_and_mul and the post-expert sum are unchanged. Attribution: the T<=32 shapes only;
  verify_16/verify_32 are the control.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "3-4 launches (~8-10 us) removed from a ~172 us launch-bound shape: ~4-6%"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "same launches on a ~268 us shape: ~3%"
  - shapes: "*load=decode_16*"
    direction: faster
    why: same launches, ~2-3% of 266/431 us
  - shapes: "*load=decode_32*"
    direction: faster
    why: same launches, ~1.5-2%, near the CI
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 keeps the SGLang op; control
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 keeps the SGLang op; control
risk: medium
depends_on: expert-id-map-single-kernel
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T071442Z
review_companions: [moe-align-single-kernel-small-t-control-capacity-256]
stage: implemented
claimed_at: 2026-10-24T071443Z
base_sha: 08e8fcb6d37d8a5b6a21eda279552962cc244ea8
base_blob: d3cbff5d9d9e6de23eb2f6770a99cd93eea093b7
branch: hyp/muse_decode_layer/moe-align-single-kernel-small-t
worktree: .worktrees/muse_decode_layer/moe-align-single-kernel-small-t
mode: null
implemented_at: 2026-10-24T071924Z
impl_commit: f0f9511bac9784936e451d51cb7bf33ba912947d
submission_sha256: b986de6ede76c2509f2de77d6543f0ef8ca0388985fefd9de2837fdb8a6c8e8c
norm_sha256: ccbd37c57b6999979576e6a467c238b8062bb9b14bcc1bf72486a53f25751989
submission_blob: c641d42a3930199e12a8ec66d93e2abd2b81f515
lint_warnings: 8
---

# moe-align-single-kernel-small-t

## Change

At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one Triton kernel, a single program, that reads the T*8 int32 topk_ids and the local_start device scalar and writes everything Marlin consumes: the int32 local_ids [T, 8] (0..31 or -1, for silu_and_mul and the post-expert sum), sorted_ids (flat token*8+k indices grouped by local expert, each expert padded up to a multiple of block_size_m with the numel sentinel, in the exact layout moe_align_block_size produces), block_experts (the expert id of each m-block, -1 for none) and num_tokens_post_pad (an int32 [1] device scalar). With at most 256 routes and 32 experts this is a counting sort in registers/shared memory, one launch replacing the 3-4 launches moe_align_block_size issues (buffer fills plus its align kernel(s)) and the 1 from the id map. The buffers keep the host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts. T=128/256 keep SGLang's moe_align_block_size as is. block_size_m selection, both Marlin GEMMs, silu_and_mul and the post-expert sum are unchanged. Attribution: the T<=32 shapes only; verify_16/verify_32 are the control.

## Expected

- `*load=decode_1*` faster: 3-4 launches (~8-10 us) removed from a ~172 us launch-bound shape: ~4-6%
- `*load=verify_1*` faster: same launches on a ~268 us shape: ~3%
- `*load=decode_16*` faster: same launches, ~2-3% of 266/431 us
- `*load=decode_32*` faster: same launches, ~1.5-2%, near the CI
- `*load=verify_16*` flat: T=128 keeps the SGLang op; control
- `*load=verify_32*` flat: T=256 keeps the SGLang op; control

## Risk

Risk: the output contract of moe_align_block_size (sorted_ids padding value, the order of tokens within an expert, whether block_experts covers the padded capacity or only the used blocks, and whether the two Marlin calls read num_tokens_post_pad from the device) must be reproduced exactly; read sglang's moe_align_block_size.py and the Marlin kernel's expectations in the image before writing it and compare the three outputs tensor-for-tensor against SGLang's op on the 13 tests' routings in prepare() (untimed) as a self-check, printing any mismatch. Token order within an expert does not affect correctness (fp32 reduce per row) but should match anyway for a clean A/B. Depends on expert-id-map-single-kernel having landed so the id-map launch is not double counted; if that one is flat but not harmful, this still stacks on it. Do not run together with moe-align-local-expert-count.

## Adversarial review

Reviewed 2026-10-24T071442Z by an adversarial subagent (model opus).

**Verdict: proceed-with-companions.** The launch-fusion idea is plausible, but the change quietly bundles a second sub-change. The sorted_ids/block_experts capacity drops from SGLang's 256-expert sizing to 32-expert sizing, which changes the m-block count Marlin's persistent scheduler partitions over. Run it with one in-run control that keeps the new kernel but allocates SGLang's 256-expert capacity, so the launch effect and the capacity effect can be told apart.

### Justification
Mechanism: at T<=32 the parent branch (expert-id-map-single-kernel) issues the id-map kernel plus SGLang's moe_align_block_size(local_ids, bsm, 256). With num_experts=256 the sgl op cannot take its small-batch single-kernel path (that path needs few experts), so it issues the multi-kernel path: align/cumsum, count-and-sort, and possibly a separate sorted_ids fill. Replacing 3-5 graph-replayed tiny kernels with one is plausible on H200 at ~1.5-2.5 us per back-to-back tiny kernel, and decode_1 (168-181 us, SoL 40-44 us) is launch-bound. Prior art: in muse_moe_w4a16, align-single-token-ep (one SGLang single-token align kernel at bs=1) measured +4.1% at decode_1 (80.2 -> 77.0 us), so a ~3 us align saving is real at T=1. Weakest links: (1) FINDINGS never shows the align kernels in any top-3 list, so the '8-10 us' saving is assumed, not measured. A realistic net is 3-7 us: 2-4% at decode_1, ~2% at verify_1, ~1-2% at decode_16 sliding, and decode_32 / decode_16 full (430-615 us) inside the per-shape noise. (2) The change says the buffers keep 'the host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts'. But this pipeline, like baseline.py, calls moe_align_block_size with N_EXPERTS=256, so SGLang here allocates T*8 + 256*(bsm-1). Using bsm=8 for T in {1,8,16} and bsm=16 for T=32, the sizes are: T=1 1800 vs 232 slots (225 vs 29 m-blocks); T=8 1856 vs 288 (232 vs 36); T=16 1920 vs 352 (240 vs 44); T=32 4096 vs 736 (256 vs 46). Marlin MoE takes its parallel m-block count from sorted_ids.size(0)/moe_block_size on the host and stripes k/n tiles over that. So shrinking capacity can change Marlin's work partition and early-exit pattern, and Marlin is 22-42% of every T<=32 shape. That is exactly the queued moe-align-local-expert-count mechanism, bundled in here, and it could move Marlin in either direction. A hint it might cost something: muse_moe_w4a16's 32-count master was an unexplained 3.2 us slower than SGLang's 256-count path at bs=1. (3) A single-program Triton counting sort at T=32 (256 routes, per-expert ranks via a one-hot [256,32] cumsum or a [256,256] compare) is not free. It may run 2-4 us itself, which eats much of the saving at decode_32. The directions in `expected` agree with the shape weights: decode_* and verify_1 carry weight 46/16 per layer and verify_16/32 are correctly held as controls. The magnitudes are optimistic by about 2x.

### Confounders on the current kernel
- expert-id-map-single-kernel (unpromoted parent, neutral 1.037x with uniform +1-2.5% drift on every shape): at T<=32 this hypothesis removes the parent's kernel, so the T<=32 delta versus BEST/master covers master's 3 id-map ops plus SGLang's align, collapsed to 1 kernel. At T=128/256 the parent's id-map kernel stays, so verify_16/verify_32 are not pure master controls. They carry the parent's 3->1 id-map change, worth ~0.5% at most on 500-1050 us shapes. Read them as drift controls with that small bias, and compare against the parent's result as well as against BEST.
- sglang-pipeline-parity-router-hilo-m8 (landed): it sets moe_align_block_size's num_experts=256 and the side-stream router/routing that joins before the expert block. The join happens before local_ids, so the fused kernel stays serial on the main stream exactly as the old align was. No overlap interaction. The router change itself touches only T=8 (verify_1), which is upstream of the align and unaffected. Not a real confounder, except that it fixes the 256-expert capacity this hypothesis silently changes.
- Bundled sub-change (not a landed hypothesis, but the main confounder): the 256->32 expert capacity shrink alters Marlin's host-side m-block partition on every T<=32 shape. Without a control, a gain or loss cannot be pinned on launch removal versus a Marlin scheduling change. It also overlaps the queued moe-align-local-expert-count, which tests the same capacity change at large T on a different base and run.

### Interference and A/B needs
A companion is warranted. Without one, both outcomes are ambiguous. A gain at decode_1/verify_1 could come from fewer launches or from Marlin running fewer and differently partitioned m-block slices (225->29 at T=1). A flat or negative result could mean the fused kernel's own time offset the launch saving, or that the capacity shrink hurt Marlin while the fusion helped. Cross-run comparison with moe-align-local-expert-count will not resolve this: it runs on master rather than this branch, it mostly measures large T, and run-to-run drift in this problem is +-1.5-2.5% per shape (see expert-id-map's uniform shift), which matches the expected effect size. The fix is an in-run control on the same branch: the same single kernel, with sorted_ids and block_experts sized exactly as moe_align_block_size(..., 256) sizes them. Then the control minus the parent's result isolates launch fusion at identical Marlin inputs, and the primary minus the control isolates the capacity effect at small T. No ab-revert is needed. The parent's id-map kernel is subsumed at T<=32 anyway, and the landed router change does not interact.

### Companions queued
- `moe-align-single-kernel-small-t-control-capacity-256`: Separates the hypothesis's two sub-changes. Control versus the parent (expert-id-map-single-kernel) gives the launch-fusion effect with Marlin seeing exactly SGLang's tensor extents. Primary versus control, in comparable runs, gives the 256->32 capacity effect on Marlin's m-block partition at small T, which the queued moe-align-local-expert-count can only probe at large T on a different base. Without it, any primary gain or loss at T<=32 could be either mechanism.

### Open questions
- Which path does sgl moe_align_block_size take at num_experts=256 and numel<=256 on the judge's image: the small-batch single kernel or the two-kernel align + count/sort path? Does the Python wrapper issue a separate sorted_ids.fill_ / cumsum zero? This sets whether the saving is 2 or 4 launches. Read the image's fused_moe_triton moe_align_block_size wrapper and sgl_kernel source before building.
- Does moe_wna16_marlin_gemm compute its parallel m-block count from sorted_token_ids.size(0) (host capacity) or only read num_tokens_post_padded on the device? If the host capacity is used, the 256->32 shrink changes Marlin's tile striping on every T<=32 shape, and the control is essential.
- What is the measured duration of the current align kernels and the new single-program kernel at T=32 (256 routes)? Pull them from the full per-kernel profile of the run, not the top-3 summary, and report them in the attribution.
- Why was muse_moe_w4a16's 32-count master 3.2 us slower than SGLang's 256-count path at bs=1? If the capacity shrink is the cause, expect this primary to underperform the control at decode_1.
- Comparison base: report the T<=32 deltas against the parent's result (expert-id-map-single-kernel, job with impl 08e8fcb6) as well as against BEST. The parent showed a uniform +1-2.5% shift that is likely drift.


## Implementation notes

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

## Rules check

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
