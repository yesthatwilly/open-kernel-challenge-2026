---
problem: muse_prefill_layer
slug: boundary-overlap-s1-slice-kernel-ffn-tail
created: 2026-10-25T152554Z
base: master
area: FFN boundary + unpooler GEMM (boundary chunks overlapped under unpooler chunks via a slice-writing boundary kernel)
change: >
  Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two
  FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kernel<8192, 512>: 399-402 us
  EACH at T=16384, 76-78 us at T=3012, 21-22 us at T=753; at T=16384 each moves ~1.61 GB = 335 us
  at 4.8 TB/s, so they already run at ~4.0 TB/s and a faster re-implementation cannot pay:
  norm-boundaries-persistent-multirow's Triton variants were 1.17-1.39x SLOWER) OFF the critical
  path, by overlapping them with the compute-bound nvjet GEMMs next to them. That needs a boundary
  kernel that writes into ROW SLICES of preallocated outputs, which the SGLang module (it
  allocates its outputs) cannot do. STAGE 1 (this hypothesis) = (a) the kernel and (b) the FFN
  tail at T > 4096 only. (a) A CUDA kernel built with torch.utils.cpp_extension.load_inline as its
  own small module (extra_cuda_cflags must include -std=c++20 and -O3, plain sm_90 is enough: no
  wgmma; -lineinfo), one function bnd_rows(stream_in fp32 [T,8192], branch bf16 [T,8192], row0,
  nrows, coef tensors, eps values, out_next bf16 [T,8192], out_stream fp32 [T,8192]) on the
  current stream: one CTA per row, 512 threads, 16 elements per thread, every global load of the
  row issued up front as 16-byte vectors (4 x float4 of stream, 2 x uint4 of branch), warp-shuffle
  + shared-memory block reductions, the same two reductions SGLang's boundary does: post-norm of
  the branch (eps POST_NORM_EPS, with the post_ffn (1 + w) input scale for the FFN boundary, none
  for the attention one), the norm-preserving residual mix with the module's precomputed
  coefficients, then the pre-norm of the new stream ((1 + w), RMS_EPS) into out_next. Take the
  exact formulas and the coefficient tensors from master's modules (state['_modules'],
  state['ffn_boundary'] / state['attn_boundary'] and their residual gates' precomputed
  coefficients) and from the Triton port in
  muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.patch, which
  already reproduced the math (stream err/residual 1.7-2.8e-3, out rel L2 1.7-2.3e-3 vs the
  module). Coefficients are weight-derived: build them in prepare into state, never module-global;
  no device pointers cached across calls. (b) FFN tail at T > MARLIN_MAX_TOKENS (first_turn):
  expert_output -> unpooler -> ffn boundary becomes, for NCH chunks of T/NCH rows (NCH in {2, 4},
  picked in prepare): main stream torch.mm(expert_output[rc], dense['unpooler'].t(),
  out=ffn_output[rc]) (contiguous row slices of a preallocated [T,8192] bf16), then
  bnd_stream.wait_stream(main) and, on a new per-state stream state['bnd_stream'],
  bnd_rows(stream, ffn_output, rc, ..., out_next=next_attn_input, out_stream=stream_out); after
  the loop main.wait_stream(bnd_stream); outputs preallocated once per call with torch.empty
  (shape-sized). So boundary chunk c runs under unpooler chunk c+1 (nvjet tiles are a
  non-persistent grid, so the block scheduler interleaves the two, as it does for the router under
  the pooler: 172 us of overlap in profile 5). Outside capture, record_stream every tensor the
  bnd_stream touches. T <= 4096 and the attention boundary stay master in stage 1. FALLBACK GATE
  in prepare (budget <= 40 s including the ~25 s build; master's prepare is 124 s of its own 200 s
  budget): (1) parity: bnd_rows on random rows at T=753 and 16384 vs state['ffn_boundary'] and
  state['attn_boundary']: residual-update rel err <= 5e-3 and next-input rel L2 <= 3e-3, worst row
  <= 5x that, no NaN; run twice alternating two separately allocated states; (2) kernel speed: one
  bnd_rows over all rows at T=16384 vs the module, graph replay, flushed, interleaved: print the
  ratio (milestone M1 below); (3) the tail: graphs of master's tail (unpooler + ffn boundary) vs
  the chunked tail at NCH 2 and 4 on synthetic bf16 expert_output at T=16384, 15 interleaved
  alternations, 256 MiB flush before each replay, empty-graph baseline subtracted; ON with the
  faster NCH only if <= 0.95x master. Any build failure, exception or a slow ratio leaves master
  byte-identical. Print '[bnd] ...' lines and a '[bnd] PATH CHOSEN' line after rega's. LATER
  STAGES (not this one): stage 2 the attention side at T > 4096 (o GEMM -> attention boundary
  chunks on bnd_stream -> pooler chunks with torch.mm(out=expert_input[rc]) waiting per chunk,
  router waiting on the last chunk), stage 3 extend both to turn_4 (T=3012, NCH 2) if stage 1-2
  show the overlap is real.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "incumbent FFN tail at T=16384 = unpooler (nvjet 192x192 share, ~1400 us) + ffn boundary 399-402 us serial; floor of the tail ~ the unpooler alone plus one exposed chunk of boundary (~100-200 us). Target -100 to -250 us per first_turn row (sliding 11120 us: 0.9-2.2%). Milestones: M1 bnd_rows <= 1.05x the SGLang kernel at T=16384 (prepare line); M2 chunked tail <= 0.92x master's tail in the prepare graph timing; M3 the judge's first_turn rows faster with the nvjet unpooler rows + norm_residual rows summing above the shape-time drop (overlap visible as GPU busy < kernel sum)"
  - shapes: "*load=turn_*"
    direction: flat
    why: "T <= 4096 keeps master's tail byte-identical in stage 1"
risk: medium
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "FusedPostNormResidualPreNorm (muse_spark_norm_residual_kernel) x2 on the critical path after the o and unpooler GEMMs at T > 4096 (stage 1: the FFN boundary)"
review: required
status: queued
---

# boundary-overlap-s1-slice-kernel-ffn-tail

## Change

Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kernel<8192, 512>: 399-402 us EACH at T=16384, 76-78 us at T=3012, 21-22 us at T=753; at T=16384 each moves ~1.61 GB = 335 us at 4.8 TB/s, so they already run at ~4.0 TB/s and a faster re-implementation cannot pay: norm-boundaries-persistent-multirow's Triton variants were 1.17-1.39x SLOWER) OFF the critical path, by overlapping them with the compute-bound nvjet GEMMs next to them. That needs a boundary kernel that writes into ROW SLICES of preallocated outputs, which the SGLang module (it allocates its outputs) cannot do. STAGE 1 (this hypothesis) = (a) the kernel and (b) the FFN tail at T > 4096 only. (a) A CUDA kernel built with torch.utils.cpp_extension.load_inline as its own small module (extra_cuda_cflags must include -std=c++20 and -O3, plain sm_90 is enough: no wgmma; -lineinfo), one function bnd_rows(stream_in fp32 [T,8192], branch bf16 [T,8192], row0, nrows, coef tensors, eps values, out_next bf16 [T,8192], out_stream fp32 [T,8192]) on the current stream: one CTA per row, 512 threads, 16 elements per thread, every global load of the row issued up front as 16-byte vectors (4 x float4 of stream, 2 x uint4 of branch), warp-shuffle + shared-memory block reductions, the same two reductions SGLang's boundary does: post-norm of the branch (eps POST_NORM_EPS, with the post_ffn (1 + w) input scale for the FFN boundary, none for the attention one), the norm-preserving residual mix with the module's precomputed coefficients, then the pre-norm of the new stream ((1 + w), RMS_EPS) into out_next. Take the exact formulas and the coefficient tensors from master's modules (state['_modules'], state['ffn_boundary'] / state['attn_boundary'] and their residual gates' precomputed coefficients) and from the Triton port in muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.patch, which already reproduced the math (stream err/residual 1.7-2.8e-3, out rel L2 1.7-2.3e-3 vs the module). Coefficients are weight-derived: build them in prepare into state, never module-global; no device pointers cached across calls. (b) FFN tail at T > MARLIN_MAX_TOKENS (first_turn): expert_output -> unpooler -> ffn boundary becomes, for NCH chunks of T/NCH rows (NCH in {2, 4}, picked in prepare): main stream torch.mm(expert_output[rc], dense['unpooler'].t(), out=ffn_output[rc]) (contiguous row slices of a preallocated [T,8192] bf16), then bnd_stream.wait_stream(main) and, on a new per-state stream state['bnd_stream'], bnd_rows(stream, ffn_output, rc, ..., out_next=next_attn_input, out_stream=stream_out); after the loop main.wait_stream(bnd_stream); outputs preallocated once per call with torch.empty (shape-sized). So boundary chunk c runs under unpooler chunk c+1 (nvjet tiles are a non-persistent grid, so the block scheduler interleaves the two, as it does for the router under the pooler: 172 us of overlap in profile 5). Outside capture, record_stream every tensor the bnd_stream touches. T <= 4096 and the attention boundary stay master in stage 1. FALLBACK GATE in prepare (budget <= 40 s including the ~25 s build; master's prepare is 124 s of its own 200 s budget): (1) parity: bnd_rows on random rows at T=753 and 16384 vs state['ffn_boundary'] and state['attn_boundary']: residual-update rel err <= 5e-3 and next-input rel L2 <= 3e-3, worst row <= 5x that, no NaN; run twice alternating two separately allocated states; (2) kernel speed: one bnd_rows over all rows at T=16384 vs the module, graph replay, flushed, interleaved: print the ratio (milestone M1 below); (3) the tail: graphs of master's tail (unpooler + ffn boundary) vs the chunked tail at NCH 2 and 4 on synthetic bf16 expert_output at T=16384, 15 interleaved alternations, 256 MiB flush before each replay, empty-graph baseline subtracted; ON with the faster NCH only if <= 0.95x master. Any build failure, exception or a slow ratio leaves master byte-identical. Print '[bnd] ...' lines and a '[bnd] PATH CHOSEN' line after rega's. LATER STAGES (not this one): stage 2 the attention side at T > 4096 (o GEMM -> attention boundary chunks on bnd_stream -> pooler chunks with torch.mm(out=expert_input[rc]) waiting per chunk, router waiting on the last chunk), stage 3 extend both to turn_4 (T=3012, NCH 2) if stage 1-2 show the overlap is real.

## Expected

- `*load=first_turn*` faster: incumbent FFN tail at T=16384 = unpooler (nvjet 192x192 share, ~1400 us) + ffn boundary 399-402 us serial; floor of the tail ~ the unpooler alone plus one exposed chunk of boundary (~100-200 us). Target -100 to -250 us per first_turn row (sliding 11120 us: 0.9-2.2%). Milestones: M1 bnd_rows <= 1.05x the SGLang kernel at T=16384 (prepare line); M2 chunked tail <= 0.92x master's tail in the prepare graph timing; M3 the judge's first_turn rows faster with the nvjet unpooler rows + norm_residual rows summing above the shape-time drop (overlap visible as GPU busy < kernel sum)
- `*load=turn_*` flat: T <= 4096 keeps master's tail byte-identical in stage 1

## Risk

Stage 1 of a rewrite of the boundary placement (not of the boundary math): judge it on M1/M2 in the prepare log and the replaced rows (FFN-boundary muse_spark_norm_residual_kernel + unpooler nvjet) at first_turn, not on the headline. Main risks: (1) the overlap may not materialise if the boundary CTAs (512 threads) cannot co-reside with nvjet CTAs and only fill tile-wave tails; the 0.95x gate then keeps master and the stage still delivers M1 (a slice-writing boundary kernel at SGLang speed), which stage 2 needs anyway; (2) chunking the unpooler into M=4096-8192 GEMMs costs wave-quantization tails (cuBLAS may pick another tile); NCH 2 limits that. CUDA build hygiene from LESSONS: -std=c++20 when overriding nvcc flags, grep the log for the build line and PATH CHOSEN, sync+print after each first launch in the self-check so a fault is attributed, first run as a check. No FP8 or precision change: fp32 stream, bf16 outputs as SGLang.

## Milestones

Rewrite stage 1 of `FusedPostNormResidualPreNorm (muse_spark_norm_residual_kernel) x2 on the critical path after the o and unpooler GEMMs at T > 4096 (stage 1: the FFN boundary)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
