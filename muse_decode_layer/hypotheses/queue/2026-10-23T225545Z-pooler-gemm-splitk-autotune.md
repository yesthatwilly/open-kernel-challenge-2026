---
problem: muse_decode_layer
slug: pooler-gemm-splitk-autotune
created: 2026-10-23T225545Z
base: master
area: pooler GEMM
change: >
  For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight
  read whose floor is 14 us), pick the GEMM form per scored T in prepare(): candidates are (a)
  SGLang's _bf16_gemm_dispatch_impl as today, (b) F.linear, (c) a 2-way and (d) a 4-way split-K
  done as torch.bmm over K-chunks with out_dtype=torch.float32 partials (x viewed as [S, T,
  8192/S] against a prepared, lossless [S, 8192/S, 4096] transposed-chunk copy of the same bf16
  weight) followed by one sum over S and a cast to bf16, so the GEMM output is bf16 exactly where
  SGLang keeps it and the only extra precision is fp32 partials. prepare() times each candidate at
  T in {1, 8, 16, 32, 128, 256} by capturing a CUDA graph of the candidate and replaying it (not
  eagerly; AGENTS.md warns eager winners lose under graphs), ~50 replays each with a cold-L2
  flush, and records the winner per T; kernel_fn branches on cfg['tokens'] (a shape, allowed) to
  call the winner. The default stays (a) for any T not in the table. qkv, o and unpooler GEMMs are
  untouched. Attribution: whichever shapes change are those whose T picked a non-default form;
  prepare() prints the table so the log shows the choice.
expected:
  - shapes: "*load=decode_16*"
    direction: faster
    why: "m=16: the muse_bf16_proj review measured pooler/unpooler at ~1.5x their 14 us floor on one nvjet kernel; split-K partials over 4x more CTAs should close part of the ~7 us gap (~2-3% of the sliding shape)"
  - shapes: "*load=decode_32*"
    direction: faster
    why: m=32, same regime
  - shapes: "*load=decode_1*"
    direction: flat
    why: "m=1: cuBLAS gemv already near bandwidth; a split-K adds a sum launch on a launch-bound shape, so the autotune should keep (a) here; control"
  - shapes: "*load=verify_32*"
    direction: flat
    why: m=256 is compute-shaped and cuBLAS picks a proper tile; expect (a) to stay
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-gemm-splitk-autotune

## Change

For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in prepare(): candidates are (a) SGLang's _bf16_gemm_dispatch_impl as today, (b) F.linear, (c) a 2-way and (d) a 4-way split-K done as torch.bmm over K-chunks with out_dtype=torch.float32 partials (x viewed as [S, T, 8192/S] against a prepared, lossless [S, 8192/S, 4096] transposed-chunk copy of the same bf16 weight) followed by one sum over S and a cast to bf16, so the GEMM output is bf16 exactly where SGLang keeps it and the only extra precision is fp32 partials. prepare() times each candidate at T in {1, 8, 16, 32, 128, 256} by capturing a CUDA graph of the candidate and replaying it (not eagerly; AGENTS.md warns eager winners lose under graphs), ~50 replays each with a cold-L2 flush, and records the winner per T; kernel_fn branches on cfg['tokens'] (a shape, allowed) to call the winner. The default stays (a) for any T not in the table. qkv, o and unpooler GEMMs are untouched. Attribution: whichever shapes change are those whose T picked a non-default form; prepare() prints the table so the log shows the choice.

## Expected

- `*load=decode_16*` faster: m=16: the muse_bf16_proj review measured pooler/unpooler at ~1.5x their 14 us floor on one nvjet kernel; split-K partials over 4x more CTAs should close part of the ~7 us gap (~2-3% of the sliding shape)
- `*load=decode_32*` faster: m=32, same regime
- `*load=decode_1*` flat: m=1: cuBLAS gemv already near bandwidth; a split-K adds a sum launch on a launch-bound shape, so the autotune should keep (a) here; control
- `*load=verify_32*` flat: m=256 is compute-shaped and cuBLAS picks a proper tile; expect (a) to stay

## Risk

Risk: the gain is bounded by ~7 us per shape on shapes of 260-700 us (1-3%), close to the +-1.5% CI, so the adversarial reviewer may ask for a control run with the autotune forced to (a) if the result is ambiguous; prepare() must print the chosen table so a flat result can be read as 'autotune picked (a) everywhere' versus 'split-K did not help'. The weight chunk copy is a lossless relayout of the same bf16 values (allowed repacking); it doubles pooler weight memory (134 MB), fine. prepare() timing with graph capture/replay for 4 candidates x 6 T values must stay well under the 300 s budget (it is seconds). The fp32-partials-then-sum path must not be mistaken for a precision reduction: it is strictly finer than cuBLAS's bf16 output rounding. Graph-safe: the branch is on T only.
