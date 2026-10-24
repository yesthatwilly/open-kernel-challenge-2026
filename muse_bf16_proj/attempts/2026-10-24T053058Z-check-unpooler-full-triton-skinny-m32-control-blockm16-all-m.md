# muse_bf16_proj · 2026-10-24T053058Z · check · unpooler-full-triton-skinny-m32-control-blockm16-all-m

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T032454Z-unpooler-full-triton-skinny-m32-control-blockm16-all-m.md` (branch `hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m`).

- **Job:** `09abdd472faa4dce` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `150b438`

## Goal
From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num_warps=4, fp32 accumulate and the bf16 store exactly as the parent, and change only the M tiling: BLOCK_M is 16 for every m in 9..32. For m > 16 the kernel keeps two fp32 accumulators [16, 64] and per K-step loads the one [64, 256] weight tile once and two [16, 256] x tiles (rows 0..15 and 16..31, rows >= m masked to zero), doing two tl.dot(x_i, tl.trans(w_tile), acc_i); for m <= 16 it is the parent's single-tile path (so unpooler decode_16 is byte-for-byte the parent's kernel and serves as the within-run control). Smem per stage drops to 32 KB weight + 2 x 8 KB x = 48 KB, 192 KB at 4 stages (same as the parent at m=32; if the parent had to drop to 3 stages, this one may keep 4). Two stores, one per 16-row accumulator, masked on rows < m. Warm m in 9..32 in prepare() on synthetic zeros. Pooler and every other layer unchanged from the parent.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m`, `muse_bf16_proj/submission.py` only (diff: 1 file, +53 / -6). Uncommitted.

- New `@triton.jit(do_not_specialize=["M"]) _unpooler_two_tile_kernel`, a SEPARATE kernel as the review's first open question asks: `_pooler_skinny_kernel` is textually unchanged (no new constexpr), so every pooler specialization and the unpooler m <= 16 specialization (BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, 4 stages, 4 warps) are the parent's exact kernels under the same cache key; unpooler decode_16 is the in-run control.
- The two-tile kernel: per K step it loads one [64, 256] weight tile, `tl.trans` once, and two [16, 256] x tiles (`offs_m1 = tl.arange(0, 16)`, `offs_m2 = 16 + tl.arange(0, 16)`), each masked on `offs_m* < M` with `other=0.0`; two `tl.dot` into two fp32 [16, 64] accumulators; two bf16 stores masked on `offs_m1 < M` / `offs_m2 < M`. Same K order as the parent, so per-row arithmetic is the parent's.
- `_unpooler_skinny_gemm`: always BLOCK_M = `UNPOOLER_BLOCK_M` = 16 (new constant); picks `_pooler_skinny_kernel` for m <= 16 and `_unpooler_two_tile_kernel` for 16 < m <= 32. Grid (128), BLOCK_N 64, BLOCK_K 256, num_warps 4, num_stages 4 unchanged.
- Routing (`_unpooler_skinny_applies`, 9 <= m <= 32 at [8192, 4096]), pooler path, router, cuBLAS fallback: unchanged.
- Warmup: `_unpooler_skinny_warmup` already loops m in 9..32 on synthetic zeros through `_unpooler_skinny_gemm`, so it now compiles both kernels (single-tile for 9..16, two-tile for 17..32); only its docstring changed. Module docstring updated with one sentence.
- Deviations: none. Smem: 4 x (32 KB w + 2 x 8 KB x) = 192 KB, same as the parent's BLOCK_M=32 at 4 stages, which compiled and ran; kept 4 stages.
- Not runnable here (no GPU/Triton on this node): compile/smem behavior of the two-tile kernel is unverified until the okc run; a shared-memory failure would surface in prepare().

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; the only new branch is on `x.shape[0]` (a shape); allocation is the same `torch.empty((m, 8192))` as the parent.
- [x] Nothing derived from `data` is cached across calls: no new state entries; warmup uses synthetic zeros and drops outputs.
- [x] Weights: unchanged (no repack); BF16 stays BF16, read as stored; router hi+lo untouched.
- [x] Activations BF16 where SGLang keeps BF16: x bf16 in, bf16 x bf16 tl.dot with fp32 accumulate, bf16 out.
- [x] n/a: attention-key rule (no attention in this problem); no skipping of any weight — every weight tile is loaded and used.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only a Triton kernel and a dispatch line.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept; no `triton.autotune`.
- [x] Masking: second tile masked with `offs_m2 = 16 + tl.arange(0, 16) < M` on both load and store.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile`: OK; `__pycache__` removed from the worktree.

## Risks / what could fail
low
Separates the m=32 row's result into tile (BLOCK_M=32 wgmma tile / smem pipeline) versus streaming bandwidth: if the parent loses at m=32 and this control does not, the family's m=32 problem is the 32-row tile and o-/qkv-gate-triton-skinny should adopt BLOCK_M=16 everywhere; if both lose equally, the plain Triton stream's ceiling (~3.3 TB/s on the pooler) is confirmed on a second shape and the whole skinny-Triton family should be withdrawn.

## Result (after the run)
- job `09abdd472faa4dce` (check), status done, passed True, tests 9/9
- speedup 0.9990x, 95% CI [0.9840, 1.0140], score 4835.620240078633 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1510295903072447, "ci": [1.133764146452636, 1.1682950341618532], "job": "bd9fa14151644cee"}; base result: muse_bf16_proj/hypotheses/done/neutral/unpooler-full-triton-skinny-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/unpooler-full-triton-skinny-m32-control-blockm16-all-m.result.json`, `muse_bf16_proj/hypotheses/done/neutral/unpooler-full-triton-skinny-m32-control-blockm16-all-m.patch`; branch `hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32-control-blockm16-all-m`
