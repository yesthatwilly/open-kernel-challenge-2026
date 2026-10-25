---
problem: muse_moe_w4a16
slug: swapab-w13-silu-epilogue-verify
created: 2026-10-24T140229Z
base: master
area: silu-mul
change: >
  Build on swapab-nvfp4-gemm-verify-rewrite (whatever its score) and fold silu_and_mul into the
  w13 _moe_nvfp4_gemm_kernel epilogue at num_tokens >= 128. In prepare(), permute the Triton
  bank's w13 rows (codes and their e4m3 block scales identically, a lossless row permutation) so
  that within every BLOCK_W=128 row tile, row 2j holds gate row n0+j and row 2j+1 holds up row
  I+n0+j, for n0 = tile*64 and j in [0, 64). Add a constexpr SWIGLU to the kernel. When it is set,
  after the K loop, multiply acc [BLOCK_W, BLOCK_T] by the expert's global scale * 2^7 as now.
  Then do tl.reshape to (64, 2, BLOCK_T), permute to (64, BLOCK_T, 2) and tl.split into gate and
  up [64, BLOCK_T]. Round each to bf16 and back to fp32, so the SwiGLU input matches SGLang's bf16
  cache1. Compute silu(gate)*up in fp32 and store bf16 into cache2[sorted id, tile*64 + j] (row
  stride I). The grid is unchanged (2I/BLOCK_W tiles). Remote (-1) blocks still return without
  writing, and w2's -1 blocks never read A. kernel_fn at num_tokens >= 128 drops the silu_and_mul
  launch and the cache1 view; w2 reads cache2 unchanged. Extend prepare's tile check to the SWIGLU
  variant against dequantized-reference silu(g)*u on the check slice. num_tokens < 128 untouched.
expected:
  - shapes: bs=128 *
    direction: faster
    why: removes act_and_mul (3.9 us) and cache1 local-route rows written then re-read at 2I=8192 wide
  - shapes: bs=256 *
    direction: faster
    why: removes act_and_mul (6.4 us) plus the larger cache1 round trip
  - shapes: bs=32 *
    direction: flat
    why: "control: untouched path"
risk: medium
depends_on: swapab-nvfp4-gemm-verify-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: silu_and_mul (act_and_mul_kernel) and the cache1 round trip at num_tokens >= 128
review: required
status: done
outcome: expired
verdict_detail: "retired: superseded per the 2026-10-25 moe diagnosis (do not run; replaced by the wg2-based line or measured dead)"
expired_at: 2026-10-25T212024Z
---

# swapab-w13-silu-epilogue-verify

## Change

Build on swapab-nvfp4-gemm-verify-rewrite (whatever its score) and fold silu_and_mul into the w13 _moe_nvfp4_gemm_kernel epilogue at num_tokens >= 128. In prepare(), permute the Triton bank's w13 rows (codes and their e4m3 block scales identically, a lossless row permutation) so that within every BLOCK_W=128 row tile, row 2j holds gate row n0+j and row 2j+1 holds up row I+n0+j, for n0 = tile*64 and j in [0, 64). Add a constexpr SWIGLU to the kernel. When it is set, after the K loop, multiply acc [BLOCK_W, BLOCK_T] by the expert's global scale * 2^7 as now. Then do tl.reshape to (64, 2, BLOCK_T), permute to (64, BLOCK_T, 2) and tl.split into gate and up [64, BLOCK_T]. Round each to bf16 and back to fp32, so the SwiGLU input matches SGLang's bf16 cache1. Compute silu(gate)*up in fp32 and store bf16 into cache2[sorted id, tile*64 + j] (row stride I). The grid is unchanged (2I/BLOCK_W tiles). Remote (-1) blocks still return without writing, and w2's -1 blocks never read A. kernel_fn at num_tokens >= 128 drops the silu_and_mul launch and the cache1 view; w2 reads cache2 unchanged. Extend prepare's tile check to the SWIGLU variant against dequantized-reference silu(g)*u on the check slice. num_tokens < 128 untouched.

## Expected

- `bs=128 *` faster: removes act_and_mul (3.9 us) and cache1 local-route rows written then re-read at 2I=8192 wide
- `bs=256 *` faster: removes act_and_mul (6.4 us) plus the larger cache1 round trip
- `bs=32 *` flat: control: untouched path

## Risk

Bounded at ~4-8 us per verify shape (~1.5-2%), so it can sit inside the CI. Judge it on the act_and_mul row vanishing and the w13 kernel row not growing. Risks: the reshape/permute/split of the accumulator may cost register shuffles in the epilogue. The interleaved row order must be applied to the block scales too, or the dequant is silently wrong; the tile check catches that. Only worth running if stage 1 reaches milestone 1 (correct).

## Milestones

Rewrite stage 2 of `silu_and_mul (act_and_mul_kernel) and the cache1 round trip at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
