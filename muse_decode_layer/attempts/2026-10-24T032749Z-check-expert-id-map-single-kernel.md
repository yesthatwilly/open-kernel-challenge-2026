# muse_decode_layer · 2026-10-24T032749Z · check · expert-id-map-single-kernel

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-23T225353Z-expert-id-map-single-kernel.md` (branch `hyp/muse_decode_layer/expert-id-map-single-kernel`).

- **Job:** `11b8df414a2245ae` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `08e8fcb`

## Goal
Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_ids - data['local_start']) % N_EXPERTS (a sub and a remainder launch) and _experts then gathers bank['expert_map'][topk_ids] (an index launch) to get int32 ids in 0..31 or -1. The new kernel (one program over the T*8 int32 ids, no host-side values) reads topk_ids and the local_start device scalar and writes a single int32 [T, 8] tensor with ((id - local_start) mod 256) where that is < 32 and -1 otherwise, exactly what expert_map produced; that tensor is passed unchanged to moe_align_block_size, silu_and_mul (expert_ids) and fused_post_expert_norm_weighted_sum (slot_expert_ids). topk_weights handling, moe_align_block_size, the Marlin calls and the returned topk_ids are untouched; the expert_map table is no longer needed in the bank. Attribution: 3 launches become 1 on every shape; only measurable on the launch-bound T<=8 shapes.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/expert-id-map-single-kernel`, file `muse_decode_layer/submission.py` only (36 insertions, 8 deletions).

- Added `import triton` / `import triton.language as tl`.
- New `_local_expert_ids_kernel` (`@triton.jit(do_not_specialize=["n"])`, no autotune): masked loads over `n = T*8` ids with a fixed constexpr `BLOCK = LOCAL_ID_BLOCK = 2048`, grid `cdiv(n, 2048)` (one program for every T <= 256; larger T gets a multi-program grid). `local_start` is loaded on device from its pointer (no `.item()`). Wrap is `(id - local_start) & 255` (Triton `%` truncates; `& 255` equals the non-negative mod 256 in two's complement), then `where(local < 32, local, -1)`, which is exactly `expert_map[(id - local_start) % 256]` since expert_map was `arange(32)` then `-1`. Literal 255/32 are used inside the kernel rather than module globals (some Triton versions reject non-constexpr globals).
- New host wrapper `_local_expert_ids(topk_ids, local_start)`: allocates `torch.empty((T, 8), int32)` per call and launches once. `n` is a runtime int and `do_not_specialize=["n"]` stops the divisible-by-16 specialization, so T=1/8/9/... and the scored T=128/256 all share one compiled variant.
- `kernel_fn`: `local_ids = (topk_ids - data["local_start"]) % N_EXPERTS` replaced by `local_ids = _local_expert_ids(topk_ids, data["local_start"])`, same position (main stream, after the side-stream join).
- `_experts`: parameter renamed `topk_ids` -> `local_ids` (it already received the local ids); the `bank["expert_map"][topk_ids].contiguous()` gather is removed and `local_ids` flows unchanged to `moe_align_block_size`, `silu_and_mul(expert_ids=...)` and `fused_post_expert_norm_weighted_sum(slot_expert_ids=...)`.
- `_marlin_bank`: `expert_map` construction and bank entry removed (grep confirms nothing else used it).
- `prepare()`: one warm-up launch on dummy `zeros((1, 8), int32)` ids and `zeros((1,), int32)` local_start before the existing `synchronize()`, so the JIT compile happens outside any timed call. Untimed, nothing data-derived, result discarded.
- Returned `topk_ids` (global ids) unchanged; topk_weights, moe_align_block_size and the Marlin calls untouched.
- Deviation: none in substance. `do_not_specialize` goes beyond the review note (which asked for a runtime `n`) to also neutralize Triton's integer-divisibility specialization of `n`.
- Uncertainty: the warm-up compiles for int32 `topk_ids`; FusedMuseSparkRouting is expected to return int32 (kernel_fn returns it as the int32 output). If it ever returned int64 the kernel still works (load then `.to(int32)`) but would compile a second variant on first use. Not runnable on this node (no GPU/torch/Triton).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's path; `local_start` read on device; the only allocation is `torch.empty((T, 8))` from shapes; grid depends only on `numel()` (shape).
- [x] Nothing derived from `data` is cached across calls: the bank lost `expert_map` and gained nothing; the prepare warm-up uses dummy zeros only.
- [x] Weights: n/a, no weight code changed (only the expert_map index table, not a weight, was dropped).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: n/a, those paths are not modified (integer ids only).
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: only `import triton` added.
- [x] All work happens inside the timed call; no output caching or replay: the id map is computed every call from that call's topk_ids and local_start.
- [x] No triton.autotune; fixed constexpr BLOCK=2048, `n` runtime and not specialized.

Lint (`okc_lint.py ... --problem muse_decode_layer`):

```
warn  host-sync    L273: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/expert-id-map-single-kernel/muse_decode_layer/submission.py
```

`python3 -I -m py_compile`: OK; `__pycache__` removed from the worktree.

## Risks / what could fail
low
Risk: correctness is exact integer arithmetic, so the only failure modes are dtype (moe_align_block_size and silu_and_mul expect int32 ids; FusedMuseSparkRouting already returns int32 since kernel_fn returns topk_ids as the int32 output) and the modulo of a negative difference (use a Python-style non-negative mod, as torch's % does). The effect size is small: if decode_1 moves less than the +-1.5% CI, record it as flat rather than a gain and keep the kernel only if the stacked moe-align-single-kernel-small-t (which subsumes it) lands. Confounder: if the adversarial review finds that torch already fuses sub+remainder into one kernel on this image, the saving is 1 launch, not 2.

## Result (after the run)
- job `11b8df414a2245ae` (check), status done, passed True, tests 13/13
- speedup 1.0374x, 95% CI [1.0219, 1.0530], score 21804.14048700647 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.020038644953596, "ci": [1.004738065279292, 1.0353392246278998], "job": "5755a412dda244a1"}; base result: muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/expert-id-map-single-kernel.result.json`, `muse_decode_layer/hypotheses/done/neutral/expert-id-map-single-kernel.patch`; branch `hyp/muse_decode_layer/expert-id-map-single-kernel`
