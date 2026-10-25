---
problem: muse_bf16_proj
slug: pooler-k-halves-fp32-partials-restack-m128-m256
created: 2026-10-25T211856Z
base: master
area: pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)
change: >
  DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch
  hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256
  (done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch: _khalf_forward, _khalf_kernel_names,
  _khalf_names_ok, _khalf_select, flag pooler_khalf_ok + dict state['pooler_khalf']) for layer ==
  'pooler' at cfg m in {128, 256} only. Today master runs pooler m=128 as nvjet 128x64 splitK NNT
  20.8 + cublasLt splitKreduce 3.0 us (row 22.85) and m=256 as nvjet 128x64 NNT 28.8 us (row 27.98
  in 741072dc, 29.9-30.05 in today's two other runs). The khalf path: xh = x.view(m, 2,
  4096).permute(1, 0, 2) (strided, no copy), wh = state['pooler_t'].view(2, 4096, 4096) (master's
  existing lossless NN repack, no new weight bytes), p = torch.bmm(xh, wh) -> [2, m, 4096], out =
  p[0] + p[1]. Offer the selector two candidates per m: (a) bf16 partials exactly as ran
  (ce808e5a: nvjet 128x128 NNT 24.7 + add 2.1 at m=256; 20.1 + 1.5 at m=128); (b) fp32 partials: p
  = torch.bmm(xh, wh, out_dtype=torch.float32), then out = torch.empty(m, 4096, bf16);
  torch.add(p[0], p[1], out=out) (one elementwise kernel, fp32 add, one bf16 rounding, the same
  numerics as cuBLAS's own split-K + splitKreduce). If torch.bmm rejects out_dtype on this image,
  print it and offer (a) only. Prefer (b) on a tie within 1%; route (a) only if it beats (b) by >=
  1% (bf16 partials passed the tests already, rel L2 2.45e-3 vs 1.66e-3 master, well under the 1%
  bound). Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph
  replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved
  reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved
  rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two
  picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted
  1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The
  pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the
  chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both
  medians, both IQRs and the kernels. Wire it the way the source branch did: own flag in
  NEW_CELL_FLAGS, inherited by state B via _ab_flags, an AB_CASES entry for pooler m=128 and
  m=256, the kernel_fn branch placed before master's NN torch.mm branch, cfg/shape/flag-only gate,
  no print/try/sync, fresh output. Everything else in master is untouched, m=512/1000 tests stay
  on master.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: K-halves bmm 24.7+2.1 (fp32 add similar) vs nvjet NNT 28.8 kernel; judged 29.94->26.75 on the old base
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: removes the 3.0 us splitKreduce node; judged 23.2->21.43 on the old base
  - shapes: layer=pooler load=decode_* m=*
    direction: flat
    why: m<=32 paths untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-k-halves-batched-gemm-m65-m256: same mechanism, now rebased onto the combo master 36c26199 (bulk row-owner, Lt cell B), with an fp32-partials candidate added and the hardened two-round 1.03 gate; the bf16-partials form already won pooler verify_16 23.2->21.43 and verify_32 29.94->26.75 on the 67b46a57 base"
review: required
status: queued
---

# pooler-k-halves-fp32-partials-restack-m128-m256

## Change

DIAGNOSIS Action 1. Built on master 36c26199 (blob ad8d7254). Port the [khalf] path from branch hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256 (done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch: _khalf_forward, _khalf_kernel_names, _khalf_names_ok, _khalf_select, flag pooler_khalf_ok + dict state['pooler_khalf']) for layer == 'pooler' at cfg m in {128, 256} only. Today master runs pooler m=128 as nvjet 128x64 splitK NNT 20.8 + cublasLt splitKreduce 3.0 us (row 22.85) and m=256 as nvjet 128x64 NNT 28.8 us (row 27.98 in 741072dc, 29.9-30.05 in today's two other runs). The khalf path: xh = x.view(m, 2, 4096).permute(1, 0, 2) (strided, no copy), wh = state['pooler_t'].view(2, 4096, 4096) (master's existing lossless NN repack, no new weight bytes), p = torch.bmm(xh, wh) -> [2, m, 4096], out = p[0] + p[1]. Offer the selector two candidates per m: (a) bf16 partials exactly as ran (ce808e5a: nvjet 128x128 NNT 24.7 + add 2.1 at m=256; 20.1 + 1.5 at m=128); (b) fp32 partials: p = torch.bmm(xh, wh, out_dtype=torch.float32), then out = torch.empty(m, 4096, bf16); torch.add(p[0], p[1], out=out) (one elementwise kernel, fp32 add, one bf16 rounding, the same numerics as cuBLAS's own split-K + splitKreduce). If torch.bmm rejects out_dtype on this image, print it and offer (a) only. Prefer (b) on a tie within 1%; route (a) only if it beats (b) by >= 1% (bf16 partials passed the tests already, rel L2 2.45e-3 vs 1.66e-3 master, well under the 1% bound). Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. Wire it the way the source branch did: own flag in NEW_CELL_FLAGS, inherited by state B via _ab_flags, an AB_CASES entry for pooler m=128 and m=256, the kernel_fn branch placed before master's NN torch.mm branch, cfg/shape/flag-only gate, no print/try/sync, fresh output. Everything else in master is untouched, m=512/1000 tests stay on master.

## Expected

- `layer=pooler load=verify_32 m=256` faster: K-halves bmm 24.7+2.1 (fp32 add similar) vs nvjet NNT 28.8 kernel; judged 29.94->26.75 on the old base
- `layer=pooler load=verify_16 m=128` faster: removes the 3.0 us splitKreduce node; judged 23.2->21.43 on the old base
- `layer=pooler load=decode_* m=*` flat: m<=32 paths untouched

## Risk

Effect: -1.2 to -3 us at 0.184 %/us on verify_32 and -1.4 us at 0.230 %/us on verify_16, about +0.5-0.9% headline: below the ~1.24% one-run margin, so it is expected to land neutral alone and pays inside combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16. Low fault risk (cuBLAS only). Rules: bf16 weights unchanged (master's lossless NN repack), fp32 only in internal partials.
