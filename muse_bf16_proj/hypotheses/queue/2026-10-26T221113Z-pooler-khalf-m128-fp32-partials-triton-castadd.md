---
problem: muse_bf16_proj
slug: pooler-khalf-m128-fp32-partials-triton-castadd
created: 2026-10-26T221113Z
base: master
area: pooler GEMM (m=128, K-split as a 2-batch cuBLAS GEMM)
change: >
  Built on master blob 2bcfe7d4 (bulk12 m=1 landed, scored 3834ceec 1.2664). For layer == 'pooler'
  at cfg m == 128 only (verify_16), port the [khalf] path of
  done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.patch (_khalf_forward,
  _khalf_route_ok, _khalf_names_ok, its hardened two-round gate, own flag pooler_khalf128_ok
  appended to NEW_CELL_FLAGS and inherited by state B, the AB case (pooler, 4096, 8192, 128)
  mapped to that flag in _ab_component when it is on, and the kernel_fn branch placed before
  master's NN torch.mm branch), with the m=256 cell dropped. Today master runs pooler m=128 as
  torch.mm(x, pooler_t): nvjet 128x64 splitK NNT 20.2 us plus a separate cublasLt splitKreduce
  node of 3.0 us, row 22.91 us (3834ceec). Candidates: (b2, preferred) xh = x.view(128, 2,
  4096).permute(1, 0, 2) (strided view, no copy), wh = state['pooler_t'].view(2, 4096, 4096), p =
  torch.bmm(xh, wh, out_dtype=torch.float32) giving [2, 128, 4096] fp32, then ONE new Triton
  kernel _add2_cast_kernel that loads p[0] and p[1] as fp32 with 16-byte vectors, adds once and
  stores bf16 into a fresh torch.empty((128, 4096), bfloat16) (grid cdiv(128*4096, 2048), 4 warps,
  warmed in prepare so the timed call never compiles): one bf16 rounding of an fp32 sum, i.e. the
  numerics of cuBLAS's own split-K reduce. The prior run's fp32 form lost (min ratio 0.991)
  because torch.add(p[0], p[1], out=bf16) lowered to the slow unrolled LoadWithCast elementwise
  kernel. (a) the bf16-partials form exactly as judged twice (torch.bmm in bf16, then p[0] + p[1];
  22.85 -> 21.36 and 21.81 us). Gate per candidate: two interleaved _cell_gate-style rounds (41
  reps, 256 MB flush, empty-graph baseline subtracted), both master/cand >= 1.03 and cand IQR <=
  1.5x master's; route b2 if it passes and is within 1% of a; route a only if b2 fails. Then one
  torch.profiler call asserts exactly two launches: an nvjet GEMM without splitK in its name plus
  the add kernel. One PATH CHOSEN line with both candidates' ratios and medians. Pooler m=256,
  every m <= 32 path and every other layer stay untouched.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "the 3.0 us splitKreduce node becomes one vectorized add: 22.91 -> ~21.0-21.8 us (the bf16-partials form judged 21.36 / 21.81 twice)"
  - shapes: layer=pooler load=verify_32 m=256
    direction: flat
    why: m=256 untouched (the K-halves route flipped in the judge there)
  - shapes: layer=pooler load=decode_* m=*
    direction: flat
    why: m <= 32 paths untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-k-halves-fp32-partials-restack-m128-m256-cf: pooled 1.2636 neutral, yet pooler m=128 won its row in both runs (22.85 -> 21.36 / 21.81) with bf16 partials; now rebased on the bulk12 master, m=128 only (m=256 flipped in the judge), and the fp32 form gets a vectorized Triton add-cast in place of torch's unrolled cast-add that made it lose (0.991)"
review: required
status: hold
hold_reason: "superseded: khalf (b2) landed on master by the campaign 2026-10-27 (commit 8f233e88, branch hyp/bf16-khalf), board 1.2735"
---

# pooler-khalf-m128-fp32-partials-triton-castadd

## Change

Built on master blob 2bcfe7d4 (bulk12 m=1 landed, scored 3834ceec 1.2664). For layer == 'pooler' at cfg m == 128 only (verify_16), port the [khalf] path of done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.patch (_khalf_forward, _khalf_route_ok, _khalf_names_ok, its hardened two-round gate, own flag pooler_khalf128_ok appended to NEW_CELL_FLAGS and inherited by state B, the AB case (pooler, 4096, 8192, 128) mapped to that flag in _ab_component when it is on, and the kernel_fn branch placed before master's NN torch.mm branch), with the m=256 cell dropped. Today master runs pooler m=128 as torch.mm(x, pooler_t): nvjet 128x64 splitK NNT 20.2 us plus a separate cublasLt splitKreduce node of 3.0 us, row 22.91 us (3834ceec). Candidates: (b2, preferred) xh = x.view(128, 2, 4096).permute(1, 0, 2) (strided view, no copy), wh = state['pooler_t'].view(2, 4096, 4096), p = torch.bmm(xh, wh, out_dtype=torch.float32) giving [2, 128, 4096] fp32, then ONE new Triton kernel _add2_cast_kernel that loads p[0] and p[1] as fp32 with 16-byte vectors, adds once and stores bf16 into a fresh torch.empty((128, 4096), bfloat16) (grid cdiv(128*4096, 2048), 4 warps, warmed in prepare so the timed call never compiles): one bf16 rounding of an fp32 sum, i.e. the numerics of cuBLAS's own split-K reduce. The prior run's fp32 form lost (min ratio 0.991) because torch.add(p[0], p[1], out=bf16) lowered to the slow unrolled LoadWithCast elementwise kernel. (a) the bf16-partials form exactly as judged twice (torch.bmm in bf16, then p[0] + p[1]; 22.85 -> 21.36 and 21.81 us). Gate per candidate: two interleaved _cell_gate-style rounds (41 reps, 256 MB flush, empty-graph baseline subtracted), both master/cand >= 1.03 and cand IQR <= 1.5x master's; route b2 if it passes and is within 1% of a; route a only if b2 fails. Then one torch.profiler call asserts exactly two launches: an nvjet GEMM without splitK in its name plus the add kernel. One PATH CHOSEN line with both candidates' ratios and medians. Pooler m=256, every m <= 32 path and every other layer stay untouched.

## Expected

- `layer=pooler load=verify_16 m=128` faster: the 3.0 us splitKreduce node becomes one vectorized add: 22.91 -> ~21.0-21.8 us (the bf16-partials form judged 21.36 / 21.81 twice)
- `layer=pooler load=verify_32 m=256` flat: m=256 untouched (the K-halves route flipped in the judge there)
- `layer=pooler load=decode_* m=*` flat: m <= 32 paths untouched

## Risk

Low: Python plus one elementwise Triton kernel, no new CUDA module, behind its own flag and the hardened two-round gate. Sizing: -1.0 to -1.9 us on verify_16 at 0.228 %/us is +0.23-0.43% on the headline, below the 1.24% one-run margin, so alone it lands only stacked (or under the sprint's beats-the-board rule with the pooler row improved). Rules: b2 is exact up to one bf16 rounding of an fp32 sum; a rounds each K-half to bf16 before the add (rel L2 2.45e-3 vs master 1.66e-3, tests passed in both judged runs) and should route only if review accepts bf16 partials. Supersedes the queued combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16: that file builds on the pre-bulk12 base (it would drop the landed bulk12 m=1 cells) and its qkv m=8/16 cells never routed (0.949 / 0.773), so retire it if this one runs. Decode-portable: decode's verify T=128 pooler is the same GEMM.
