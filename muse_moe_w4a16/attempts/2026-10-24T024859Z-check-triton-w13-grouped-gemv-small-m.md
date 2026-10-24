# muse_moe_w4a16 · 2026-10-24T024859Z · check · triton-w13-grouped-gemv-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225501Z-triton-w13-grouped-gemv-small-m.md` (branch `hyp/muse_moe_w4a16/triton-w13-grouped-gemv-small-m`).

- **Job:** `f31a03eb9bc6410f` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `5e969da`

## Goal
For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with two Triton launches that write the same cache1 [M*8, 2I] bf16 in flat slot order (row = token*8 + slot), so silu_and_mul, the w2 Marlin GEMM and the post-norm stay exactly as in parity. (a) compact_routes: grid (32,), program e loads the M*8 <= 64 mapped local_ids, mask = ids == e, cnt[e] = sum(mask), and writes the matching flat slot indices to slot_list[e, cumsum(mask)-1] with a masked store (cnt[e] <= M <= 8 because a token routes to an expert at most once). (b) w13_gemv: grid (32, 2I/BLOCK_N); program (e, n) loads cnt[e] and returns immediately if 0 (untouched experts cost one load), else loads up to BLOCK_M=16 slots, gathers x[slot//8] rows masked by arange < cnt into a [16, BLOCK_K] bf16 tile per K step, and dequantizes the expert's w13 tile on the fly: the uint8 tile [BLOCK_N, BLOCK_K/2] is split into nibbles, E2M1 magnitude from code&7 via the (0,.5,1,1.5,2,3,4,6) grid (arithmetic: e=c>>1, m=c&1, e==0 ? 0.5*m : (1+0.5*m)*2^(e-1)), sign from bit 3, times the fp8 e4m3 block scale (.to(float32), one per 16 elements); the e2m1*e4m3 product has <= 7 significant bits so the cast to bf16 for tl.dot is exact; accumulate in fp32 and multiply by the per-expert global scale w13_gs[e] once after the K loop; store rows at cache1[slot, n-tile] masked by arange < cnt. prepare() keeps, beside the Marlin bank, a second lossless copy of the 32 local experts' raw w13/w13_sf/w13_gs tensors (contiguous, same dtype), optionally nibble-regrouped within each row so the low nibbles of byte j hold element j and the high nibbles element j + H/2 (a K permutation applied identically to the scale index; or use tl.interleave instead). Marlin's moe_align_block_size still runs for the w2 GEMM in this hypothesis. Attribution: any change is confined to bs=1 and bs=8 and to the w13 GEMM; bs=16..256 are untouched controls.

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/triton-w13-grouped-gemv-small-m`, only `muse_moe_w4a16/submission.py` changed (uncommitted; 114 insertions, 5 deletions). First-line marker kept.

- `prepare()`: each bank gains `raw13` = `weights["w13"|"w13_sf"|"w13_gs"][0:32].contiguous()` — the identical NVFP4 uint8 bytes, fp8 e4m3 block scales and fp32 global scales, no permutation (lossless copy beside the Marlin bank; ~576 MB extra).
- `kernel_fn`: `if num_tokens <= 8` (shape branch) calls `_w13_small_m` instead of the w13 `moe_wna16_marlin_gemm`; otherwise the parity call is unchanged. `moe_align_block_size`, `silu_and_mul`, w2 Marlin and the post-norm are untouched on both paths.
- `_compact_routes_kernel` grid (32,), num_warps=1: loads the M*8 (<= 64, BLOCK_S=64, masked) local ids, `hit = ids == e`, `pos = cumsum(hit) - 1`, masked store of the flat slot index to `slot_list[e*16 + pos]`, `cnt[e] = min(sum(hit), 16)`. `cnt` [32] and `slot_list` [32*16] are fresh `torch.empty` allocations inside kernel_fn each call (fixed size, not data-dependent); slot loads in the GEMV are masked by `arange(16) < cnt`, so unwritten padding is never read.
- `_w13_gemv_kernel` grid (32, 8192/32=256), BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, num_warps=4, num_stages=3, fixed constexprs (no autotune). Program returns after one load if `cnt[e]==0`. Per K step: uint8 tile [32,128], lo = code & 0xF (element 2j), hi = code >> 4 (element 2j+1), E2M1 -> fp32 built from bits (exp field e+126, mantissa bit m<<22, e==0 -> 0.5*m, sign bit 3 -> bit 31), times the fp8 block scale gathered at index byte//8 (= (2j)//16), then `tl.interleave(lo, hi)` gives natural K order (no K permutation anywhere) and is cast to bf16 (exact: <= 6 significant bits). x rows `x[slot // 8]` gathered masked by `arange < cnt`, `tl.dot` into fp32 acc. `acc *= w13_gs[e]` once after the K loop (never folded into the bf16 tile). bf16 store to `cache1[slot, n-tile]` masked by `arange < cnt`.
- Remote-expert slots of cache1 are left unwritten, as the Marlin EP path leaves them (cache13 is `torch.empty` in parity too; silu_and_mul and the post-norm skip `-1` ids).
- No local GPU/triton: compiled only with `py_compile`; Triton compile and correctness untested. Next lever if w13_gemv is slow at bs=1: BLOCK_N 16 or split-K (per review).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/triton-w13-grouped-gemv-small-m diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the timed path; `cnt`/`slot_list` are fixed-size allocations; the only branch is `num_tokens <= 8` (shape); the empty-expert exit is a device-side branch inside the kernel.
- [x] Nothing derived from `data` is cached across calls: `cnt` and `slot_list` are rebuilt by `_compact_routes_kernel` every call inside kernel_fn.
- [x] Weights: `raw13` is a byte-identical contiguous slice of `w13`/`w13_sf`/`w13_gs` made in `prepare()` only; NVFP4 stays NVFP4 (uint8 E2M1 + fp8 e4m3 + fp32), dequantized only inside the kernel.
- [x] Activations BF16 where SGLang keeps BF16: x read as bf16, cache1 written bf16 (as Marlin writes it); fp32 is only the accumulator. KV cache / queries: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: MoE problem, no attention; no route/expert is skipped (every local slot is computed; only experts with zero slots exit early, as Marlin skips them).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only `import triton`, `triton.language` and two `@triton.jit` kernels.
- [x] All work happens inside the timed call; no output caching or replay: routing compaction and the GEMV both run in kernel_fn every call.
- [x] No `triton.autotune`: fixed constexpr configs (checked: diff has no `autotune` call).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`), exit 0:

```
warn  host-sync    L216: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/triton-w13-grouped-gemv-small-m/muse_moe_w4a16/submission.py
```

The one warning is the pre-existing `torch.cuda.synchronize()` in `prepare()` (untimed), unchanged by this diff. `python3 -m py_compile` passes.

## Risks / what could fail
high
Risks: Triton compile time counts toward the 15 min run limit (two small kernels, two shapes: fine); the nibble/scale indexing must match nvfp4.py (low nibble = element 2i) or the bs=1 test fails immediately; the extra 'compact_routes' launch adds ~3 us, which the bs=1 gain must exceed; early-exit programs still cost grid scheduling for 32 x (8192/BLOCK_N) programs. Correctness: fp32 accumulation and exact bf16 dequant leave more margin than Marlin's path. Rules: the raw-layout copy is the same NVFP4 bytes, a lossless permutation at most; x stays bf16; no data-derived caching (slot_list/cnt are rebuilt each call inside the timed graph); branching on num_tokens is a shape branch. Nearby shapes (bs 4) take the same path, bs 33 the Marlin path, so final-review reruns see no cliff.

## Result (after the run)
- job `f31a03eb9bc6410f` (check), status done, passed True, tests 5/5
- speedup 0.7506x, 95% CI [0.7393, 0.7618], score 17917.85687820234 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/triton-w13-grouped-gemv-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/regression/triton-w13-grouped-gemv-small-m.patch`; branch `hyp/muse_moe_w4a16/triton-w13-grouped-gemv-small-m`
