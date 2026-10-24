---
problem: muse_moe_w4a16
slug: triton-w13-grouped-gemv-small-m
created: 2026-10-23T225501Z
base: master
area: w13 marlin
change: >
  For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path
  untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with two Triton launches that
  write the same cache1 [M*8, 2I] bf16 in flat slot order (row = token*8 + slot), so silu_and_mul,
  the w2 Marlin GEMM and the post-norm stay exactly as in parity. (a) compact_routes: grid (32,),
  program e loads the M*8 <= 64 mapped local_ids, mask = ids == e, cnt[e] = sum(mask), and writes
  the matching flat slot indices to slot_list[e, cumsum(mask)-1] with a masked store (cnt[e] <= M
  <= 8 because a token routes to an expert at most once). (b) w13_gemv: grid (32, 2I/BLOCK_N);
  program (e, n) loads cnt[e] and returns immediately if 0 (untouched experts cost one load), else
  loads up to BLOCK_M=16 slots, gathers x[slot//8] rows masked by arange < cnt into a [16,
  BLOCK_K] bf16 tile per K step, and dequantizes the expert's w13 tile on the fly: the uint8 tile
  [BLOCK_N, BLOCK_K/2] is split into nibbles, E2M1 magnitude from code&7 via the
  (0,.5,1,1.5,2,3,4,6) grid (arithmetic: e=c>>1, m=c&1, e==0 ? 0.5*m : (1+0.5*m)*2^(e-1)), sign
  from bit 3, times the fp8 e4m3 block scale (.to(float32), one per 16 elements); the e2m1*e4m3
  product has <= 7 significant bits so the cast to bf16 for tl.dot is exact; accumulate in fp32
  and multiply by the per-expert global scale w13_gs[e] once after the K loop; store rows at
  cache1[slot, n-tile] masked by arange < cnt. prepare() keeps, beside the Marlin bank, a second
  lossless copy of the 32 local experts' raw w13/w13_sf/w13_gs tensors (contiguous, same dtype),
  optionally nibble-regrouped within each row so the low nibbles of byte j hold element j and the
  high nibbles element j + H/2 (a K permutation applied identically to the scale index; or use
  tl.interleave instead). Marlin's moe_align_block_size still runs for the w2 GEMM in this
  hypothesis. Attribution: any change is confined to bs=1 and bs=8 and to the w13 GEMM; bs=16..256
  are untouched controls.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "SGLang is 77 us against an 11.4 us floor at bs=1 (1-2 touched experts); Marlin MoE at M=1 with 8-row blocks is far below bandwidth, while a grouped GEMV reading each touched expert's 19 MB of w13 once and skipping the other 30 experts runs near HBM speed"
  - shapes: bs=8 *
    direction: faster
    why: "bs=8: 151 us vs 37.3 us floor; same mechanism, cnt[e] <= 8 fits one BLOCK_M=16 tile so no m-loop"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens=16 > 8 stays on the parity Marlin path"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: sglang-marlin-path-parity
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T024500Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T024500Z
base_sha: 6b5469bd92f0a43ce5908912b02fb8dee425c3fa
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/triton-w13-grouped-gemv-small-m
worktree: .worktrees/muse_moe_w4a16/triton-w13-grouped-gemv-small-m
mode: null
implemented_at: 2026-10-24T024859Z
impl_commit: 5e969da46ab87d84a299f2124a2f42cf9f5109cd
submission_sha256: ea2dcd2ce23c11bd76a9270a774d295a6240ba2dde688e2b1b787007241be514
norm_sha256: 5efcb7aca5a2623591faeda5c1df62dfcaf3e33e5a0af534c8d9600df9f1d46c
submission_blob: 09d727eae616e13de2636faa5071b006980144ee
lint_warnings: 1
---

# triton-w13-grouped-gemv-small-m

## Change

For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with two Triton launches that write the same cache1 [M*8, 2I] bf16 in flat slot order (row = token*8 + slot), so silu_and_mul, the w2 Marlin GEMM and the post-norm stay exactly as in parity. (a) compact_routes: grid (32,), program e loads the M*8 <= 64 mapped local_ids, mask = ids == e, cnt[e] = sum(mask), and writes the matching flat slot indices to slot_list[e, cumsum(mask)-1] with a masked store (cnt[e] <= M <= 8 because a token routes to an expert at most once). (b) w13_gemv: grid (32, 2I/BLOCK_N); program (e, n) loads cnt[e] and returns immediately if 0 (untouched experts cost one load), else loads up to BLOCK_M=16 slots, gathers x[slot//8] rows masked by arange < cnt into a [16, BLOCK_K] bf16 tile per K step, and dequantizes the expert's w13 tile on the fly: the uint8 tile [BLOCK_N, BLOCK_K/2] is split into nibbles, E2M1 magnitude from code&7 via the (0,.5,1,1.5,2,3,4,6) grid (arithmetic: e=c>>1, m=c&1, e==0 ? 0.5*m : (1+0.5*m)*2^(e-1)), sign from bit 3, times the fp8 e4m3 block scale (.to(float32), one per 16 elements); the e2m1*e4m3 product has <= 7 significant bits so the cast to bf16 for tl.dot is exact; accumulate in fp32 and multiply by the per-expert global scale w13_gs[e] once after the K loop; store rows at cache1[slot, n-tile] masked by arange < cnt. prepare() keeps, beside the Marlin bank, a second lossless copy of the 32 local experts' raw w13/w13_sf/w13_gs tensors (contiguous, same dtype), optionally nibble-regrouped within each row so the low nibbles of byte j hold element j and the high nibbles element j + H/2 (a K permutation applied identically to the scale index; or use tl.interleave instead). Marlin's moe_align_block_size still runs for the w2 GEMM in this hypothesis. Attribution: any change is confined to bs=1 and bs=8 and to the w13 GEMM; bs=16..256 are untouched controls.

## Expected

- `bs=1 *` faster: SGLang is 77 us against an 11.4 us floor at bs=1 (1-2 touched experts); Marlin MoE at M=1 with 8-row blocks is far below bandwidth, while a grouped GEMV reading each touched expert's 19 MB of w13 once and skipping the other 30 experts runs near HBM speed
- `bs=8 *` faster: bs=8: 151 us vs 37.3 us floor; same mechanism, cnt[e] <= 8 fits one BLOCK_M=16 tile so no m-loop
- `bs=16 *` flat: control: num_tokens=16 > 8 stays on the parity Marlin path
- `bs=128 *` flat: control: untouched

## Risk

Risks: Triton compile time counts toward the 15 min run limit (two small kernels, two shapes: fine); the nibble/scale indexing must match nvfp4.py (low nibble = element 2i) or the bs=1 test fails immediately; the extra 'compact_routes' launch adds ~3 us, which the bs=1 gain must exceed; early-exit programs still cost grid scheduling for 32 x (8192/BLOCK_N) programs. Correctness: fp32 accumulation and exact bf16 dequant leave more margin than Marlin's path. Rules: the raw-layout copy is the same NVFP4 bytes, a lossless permutation at most; x stays bf16; no data-derived caching (slot_list/cnt are rebuilt each call inside the timed graph); branching on num_tokens is a shape branch. Nearby shapes (bs 4) take the same path, bs 33 the Marlin path, so final-review reruns see no cliff.

## Adversarial review

Reviewed 2026-10-24T024500Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** The mechanism is plausible and the parity profile supports it (Marlin at bs=1 is 41 us for both GEMMs against an ~11 us floor, ~30% of HBM), the change is shape-gated to bs<=8 with four untouched control shapes, and the only landed hypothesis is the base pipeline itself, so there is nothing to revert or retune; the run's own per-kernel profile separates compact_routes from w13_gemv and exposes the w2 Marlin time alone. The real risks are implementation quality (Triton dequant ALU cost, early-exit grid, slot_list sizing) and geomean dilution, not interpretability.

### Justification
At bs=1 the parity result shows one Marlin line at 41.1 us (52.6% of 80.2 us), both GEMMs aggregated; w13 is ~27 us against a ~8 us w13-only floor (~30% of HBM), so a grouped GEMV reading each touched expert once has headroom; at bs=8 Marlin is 107.6 us, w13 share ~72 us against ~24 us. Expected shapes consistent with TASK.md (bs=1/8 faster, bs=16..256 untouched controls). cnt[e] <= M <= 8 fits BLOCK_M=16; cache1 row layout matches Marlin's so silu, w2 and post-norm are unaffected; e2m1 x e4m3 fits bf16 exactly. Weakest link: the Triton dequant speed (~15 ALU ops per weight with only ~256-512 live CTAs) could leave it slower than Marlin's lop3 tricks. At ~60% of HBM expect ~+6% geomean; at ~35-40% only 0.5-1%, inside the CI, so read per-shape medians. Second: +1 net launch and a mostly-empty grid cost a few us.

### Confounders on the current kernel
- sglang-marlin-path-parity (the base): cannot be absent; its 32-expert align knob no longer feeds w13. Parity's unexplained 3 us bs=1 deficit vs SGLang may be partly recovered here; bounded <=0.7% of geomean, cannot flip a gain's sign; the SGLang column is a second yardstick.
- ~16 us of at::native glue at bs=1 of unknown origin: if part lives in the Marlin wrapper, dropping the w13 call removes it too and overstates the GEMV kernel's contribution; the profile shows this directly.

### Interference and A/B needs
No companion. ab-revert restores the 0.006x starter; ab-tune of the align knob resolves nothing; the per-kernel profile decomposes a loss (w13_gemv > ~27 us vs overhead) and a gain is unambiguous since bs=16..256 are untouched. Reading rule: judge on bs=1 and bs=8 medians against both parity (80.2 / 152.0 us) and SGLang (77.0 / 151.1 us); claim the mechanism only if the bs=1 gain exceeds ~5 us.

### Open questions
- Record the standalone w2 Marlin time after this run so the w13 Marlin cost is measured, not inferred.
- Where do the at::native reduce/elementwise launches at bs=1 come from?
- Why is parity's bs=1 3.2 us slower than SGLang?
- slot_list is sized cnt[e] <= 8 but the GEMV loads BLOCK_M=16 slots: pad to 16 columns or mask by arange < cnt before the x gather.
- If the run loses with w13_gemv dominating, the follow-up should change only the dequant.
- If w13_gemv is well above ~12 us at bs=1, smaller BLOCK_N or split-K is the next lever.


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/triton-w13-grouped-gemv-small-m`, only `muse_moe_w4a16/submission.py` changed (uncommitted; 114 insertions, 5 deletions). First-line marker kept.

- `prepare()`: each bank gains `raw13` = `weights["w13"|"w13_sf"|"w13_gs"][0:32].contiguous()` — the identical NVFP4 uint8 bytes, fp8 e4m3 block scales and fp32 global scales, no permutation (lossless copy beside the Marlin bank; ~576 MB extra).
- `kernel_fn`: `if num_tokens <= 8` (shape branch) calls `_w13_small_m` instead of the w13 `moe_wna16_marlin_gemm`; otherwise the parity call is unchanged. `moe_align_block_size`, `silu_and_mul`, w2 Marlin and the post-norm are untouched on both paths.
- `_compact_routes_kernel` grid (32,), num_warps=1: loads the M*8 (<= 64, BLOCK_S=64, masked) local ids, `hit = ids == e`, `pos = cumsum(hit) - 1`, masked store of the flat slot index to `slot_list[e*16 + pos]`, `cnt[e] = min(sum(hit), 16)`. `cnt` [32] and `slot_list` [32*16] are fresh `torch.empty` allocations inside kernel_fn each call (fixed size, not data-dependent); slot loads in the GEMV are masked by `arange(16) < cnt`, so unwritten padding is never read.
- `_w13_gemv_kernel` grid (32, 8192/32=256), BLOCK_M=16, BLOCK_N=32, BLOCK_K=256, num_warps=4, num_stages=3, fixed constexprs (no autotune). Program returns after one load if `cnt[e]==0`. Per K step: uint8 tile [32,128], lo = code & 0xF (element 2j), hi = code >> 4 (element 2j+1), E2M1 -> fp32 built from bits (exp field e+126, mantissa bit m<<22, e==0 -> 0.5*m, sign bit 3 -> bit 31), times the fp8 block scale gathered at index byte//8 (= (2j)//16), then `tl.interleave(lo, hi)` gives natural K order (no K permutation anywhere) and is cast to bf16 (exact: <= 6 significant bits). x rows `x[slot // 8]` gathered masked by `arange < cnt`, `tl.dot` into fp32 acc. `acc *= w13_gs[e]` once after the K loop (never folded into the bf16 tile). bf16 store to `cache1[slot, n-tile]` masked by `arange < cnt`.
- Remote-expert slots of cache1 are left unwritten, as the Marlin EP path leaves them (cache13 is `torch.empty` in parity too; silu_and_mul and the post-norm skip `-1` ids).
- No local GPU/triton: compiled only with `py_compile`; Triton compile and correctness untested. Next lever if w13_gemv is slow at bs=1: BLOCK_N 16 or split-K (per review).

## Rules check

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
