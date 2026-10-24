---
problem: muse_prefill_layer
slug: router-hilo-one-gemm-n512
created: 2026-10-24T185248Z
base: master
area: router GEMM (hi and lo halves as one N=512 cuBLAS GEMM)
change: >
  Built on master (635cfee). prepare(): add state['router_hilo_t'] = torch.cat((hi.t(), lo.t()),
  dim=1).contiguous(), bf16 [8192, 512] (the same BF16 hi/lo split master stores as router_hi_t
  and router_lo_t, side by side: a lossless repack of the pair the rules allow). kernel_fn, inside
  the existing `with torch.cuda.stream(side_stream):` block, replace the two lines `logits =
  torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t,
  out_dtype=fp32)` with `both = torch.mm(ffn_input, state['router_hilo_t'],
  out_dtype=torch.float32)` ([T, 512] fp32) and `logits = both[:, :256] + both[:, 256:]` (one
  elementwise add, fresh contiguous [T, 256] fp32). Same math: each logit is still x.hi + x.lo,
  each dot accumulated in fp32 by cuBLAS, summed in fp32. FusedMuseSparkRouting, the stream
  fork/join, the pooler and everything else are unchanged. MECHANISM: today the pair is two cuBLAS
  GEMMs with N=256 plus an in-place add; at T=753 cuBLAS runs them as split-K nvjet tss_64x96
  (32.7 us sliding, 81.8 us full turn_1, plus cublasLt splitKreduce kernels) and at T=3012 as
  tss_64x104 at ~190 TFLOP/s (131.9-139.6 us), all on the side stream sharing SMs with the pooler.
  One GEMM with N=512 gives cuBLAS twice the N tiles to fill the grid without split-K at these M,
  one launch and one reduce fewer. prepare(): one check that logits from the new path equal
  master's two-GEMM logits within 1e-5 relative on synthetic bf16 [753, 8192] and [3012, 8192]
  inputs, and that the routing kernel picks identical top-8 on them; print both paths' kernel
  names via a short torch.profiler pass if cheap, else skip.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "router rows 32.7 us (sliding) and 81.8 us + splitKreduce (full) on the side branch, which at turn_1 full outlasts the ~55-70 us pooler: est. -10 to -40 us, -1 to -2.5% on turn_1 full, ~-0.5% sliding"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "131.9-139.6 us of router GEMMs at ~190 TFLOP/s sharing SMs with the pooler: est. -20 to -60 us, -0.5 to -1.5%"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: router already runs as nvjet 128x256 at ~485 TFLOP/s under a 1.4 ms pooler
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: running
stage: dispatched
claimed_at: 2026-10-24T231740Z
base_sha: 10e97df213c296823faa2881225ed84509e1e385
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/router-hilo-one-gemm-n512
worktree: .worktrees/muse_prefill_layer/router-hilo-one-gemm-n512
mode: check
implemented_at: 2026-10-24T231832Z
impl_commit: dcffe36980df8afe57be826a1f26f654096549fa
submission_sha256: e86b3cfd18fb722b5dbbe9b08aa32555a0f3f54d2d96314e4b3bcb3ba76f3d05
norm_sha256: 94ff691d37c4cd8e749e4ebdbf5d35b9f3ac544108c86ca51b7de82ab52a297d
submission_blob: 1d99c85e29e4eea02f922f841550643a9f1a62f9
lint_warnings: 5
attempt_file: 2026-10-24T231833Z-check-router-hilo-one-gemm-n512.md
prepared_at: 2026-10-24T231833Z
job: f4e3c5f4c306427b
dispatched_at: 2026-10-24T231843Z
---

# router-hilo-one-gemm-n512

## Change

Built on master (635cfee). prepare(): add state['router_hilo_t'] = torch.cat((hi.t(), lo.t()), dim=1).contiguous(), bf16 [8192, 512] (the same BF16 hi/lo split master stores as router_hi_t and router_lo_t, side by side: a lossless repack of the pair the rules allow). kernel_fn, inside the existing `with torch.cuda.stream(side_stream):` block, replace the two lines `logits = torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t, out_dtype=fp32)` with `both = torch.mm(ffn_input, state['router_hilo_t'], out_dtype=torch.float32)` ([T, 512] fp32) and `logits = both[:, :256] + both[:, 256:]` (one elementwise add, fresh contiguous [T, 256] fp32). Same math: each logit is still x.hi + x.lo, each dot accumulated in fp32 by cuBLAS, summed in fp32. FusedMuseSparkRouting, the stream fork/join, the pooler and everything else are unchanged. MECHANISM: today the pair is two cuBLAS GEMMs with N=256 plus an in-place add; at T=753 cuBLAS runs them as split-K nvjet tss_64x96 (32.7 us sliding, 81.8 us full turn_1, plus cublasLt splitKreduce kernels) and at T=3012 as tss_64x104 at ~190 TFLOP/s (131.9-139.6 us), all on the side stream sharing SMs with the pooler. One GEMM with N=512 gives cuBLAS twice the N tiles to fill the grid without split-K at these M, one launch and one reduce fewer. prepare(): one check that logits from the new path equal master's two-GEMM logits within 1e-5 relative on synthetic bf16 [753, 8192] and [3012, 8192] inputs, and that the routing kernel picks identical top-8 on them; print both paths' kernel names via a short torch.profiler pass if cheap, else skip.

## Expected

- `*load=turn_1*` faster: router rows 32.7 us (sliding) and 81.8 us + splitKreduce (full) on the side branch, which at turn_1 full outlasts the ~55-70 us pooler: est. -10 to -40 us, -1 to -2.5% on turn_1 full, ~-0.5% sliding
- `*load=turn_4*` faster: 131.9-139.6 us of router GEMMs at ~190 TFLOP/s sharing SMs with the pooler: est. -20 to -60 us, -0.5 to -1.5%
- `*load=first_turn*` neutral: router already runs as nvjet 128x256 at ~485 TFLOP/s under a 1.4 ms pooler

## Risk

Same area as the held router-triton-hilo-splitk-turn (Triton split-K port) and router-pooler-concat-gemm-turn, but a different, smaller mechanism (stay on cuBLAS, one N=512 call); judge it on the router rows and the turn_1 full shape, where the side branch is most likely on the critical path. The extra [T,256] add kernel replaces master's in-place add, so the launch count drops by one GEMM. Risk: cuBLAS may still pick split-K at M=753, N=512 (then the gain is one launch); expected headline move is small, ~0.3-0.8%.

## Implementation notes

Built in the main thread 2026-10-24 ~23:35Z with self-review (user approved; no new agents before the reset). prepare(): state["router_hilo_t"] = cat(hi.t(), lo.t(), dim=1) bf16 [8192, 512]. kernel_fn side stream: one torch.mm(ffn_input, router_hilo_t, out_dtype=fp32) then logits = both[:, :256] + both[:, 256:]. router_hi_t/lo_t kept in state but unused. Same math per logit (x.hi + x.lo in fp32); the cuBLAS reduction order for the N=512 call may differ by fp32 rounding only.

## Rules check

- [x] CUDA-graph capturable: no host sync, allocation on input values or value-dependent control flow added to kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 and BF16 unchanged; the router hi/lo split is the one the rules allow (hi+lo, fp32 accumulate).
- [x] Activations BF16 as in SGLang; KV FP8; queries unchanged.
- [x] Every key attended (attention untouched).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (prepare() only builds a weight repack).
- [x] okc_lint: OK with warnings (pre-existing prepare-only host syncs).
