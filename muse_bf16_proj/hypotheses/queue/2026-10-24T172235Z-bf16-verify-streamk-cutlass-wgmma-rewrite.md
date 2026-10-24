---
problem: muse_bf16_proj
slug: bf16-verify-streamk-cutlass-wgmma-rewrite
created: 2026-10-24T172235Z
base: master
area: bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma)
change: >
  Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: at m=256 (verify_32,
  the load furthest from its floor: 92.9 us vs 51.8 floor) all four bf16 layers run cuBLAS nvjet
  with data-parallel tiles that cannot fill 132 SMs or are math-bound at well under peak. Measured
  tensor rates from FINDINGS: qkv_gate nvjet 96x64 19.4 us = 9.66 GFLOP -> 498 TFLOP/s (n=2304
  gives only 36-72 output tiles); pooler nvjet 128x64 29.5 us -> 583 TFLOP/s (n=4096: 64 tiles at
  128x128); o nvjet 128x128 8.3 us -> 518 TFLOP/s (k=1024: 16 K-blocks, pipeline fill and epilogue
  dominate); unpooler nvjet 128x128 24.6 us -> 699 TFLOP/s (already near practical peak). A
  persistent stream-K kernel splits the total MAC-loop iterations evenly over exactly 132 CTAs, so
  tile-count quantization stops costing anything, and a TMA + warp-specialized wgmma mainloop runs
  at the 700-750 TFLOP/s a dense sm90 GEMM reaches. No Triton or cuBLASLt-algorithm path does this
  (the cuBLASLt algo pick only re-orders cuBLAS's own candidates and is under a rules heads-up;
  this hypothesis does no timing or algorithm search in prepare at all: the tile config is a fixed
  constant). STAGE 1 (this file): one CUTLASS 3.x sm90 GEMM, compiled once at import with
  torch.utils.cpp_extension.load_inline (nvcc; flags -std=c++17 --expt-relaxed-constexpr -DNDEBUG
  -gencode arch=compute_90a,code=sm_90a, which wgmma and TMA require). Headers: search, in order,
  the CUTLASS/CuTe include dirs shipped inside the image (deep_gemm's package include dir,
  flashinfer's data/3rdparty cutlass include, sgl_kernel / cutlass_library source dirs,
  /usr/local/cutlass/include), print the path used; if none has
  cutlass/gemm/collective/collective_builder.hpp, print '[streamk] no CUTLASS headers, master
  kept' and skip. The kernel: GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop,
  CollectiveEpilogue, cutlass::gemm::StreamKScheduler> via CollectiveBuilder<Sm90,
  OpClassTensorOp, bf16 A RowMajor (x [m,k]) align 8, bf16 B ColumnMajor (W [n,k] row-major read
  as K x N column-major, i.e. a TN GEMM, no repack) align 8, float accumulate, TileShape
  128x128x64, ClusterShape 1x2x1 (the two CTAs along N share each x tile by TMA multicast),
  StageCountAutoCarveout, KernelTmaWarpSpecializedCooperative>, epilogue builder
  TmaWarpSpecializedCooperative writing bf16 RowMajor D with no C (beta 0). ONE instantiation
  shared by all four layers (problem shape is a runtime argument), to keep compile time near one
  minute; print compile seconds. Host wrapper gemm(x, W, out, workspace): builds Arguments {kGemm,
  {m, n, k, 1}, {x, stride, W, stride}, {{1, 0}, nullptr, ..., out, stride}, hw_info with sm_count
  132 cached from prepare}, scheduler args decomposition_mode = StreamK, reduction_mode =
  Deterministic (fixed summation order), calls can_implement (raise on failure, which only happens
  in prepare), initialize(args, workspace_ptr, stream) and run(stream) on
  at::cuda::getCurrentCUDAStream(). Workspace: Gemm::get_workspace_size for m=256 per layer,
  allocated ONCE in prepare() (state['streamk_ws'], max over layers) and reused every call, never
  sized from input values. initialize() may zero the stream-K barrier region with a
  cudaMemsetAsync; under graph capture that is one memset node, which is acceptable for stage 1
  and must be visible in the profile (stage 2 removes it). ROUTING: kernel_fn takes the new path
  for layer in {qkv_gate, o, pooler, unpooler} at 129 <= cfg m <= 256 when
  state['streamk_ok'][layer] is True, cfg n/k match the full weight, and x is contiguous bf16; the
  judge's m=130 unpooler test therefore exercises it (TMA zero-fills the out-of-range rows of the
  last M tile). m = 512/1000 (tests), m <= 128, the router and every other shape keep master byte
  for byte. SELF-CHECK in prepare() per layer (failure prints '[streamk] <layer> DISABLED:
  <reason>' and sets the flag False, a constant fixed in prepare): can_implement OK; synthetic x
  at m in {130, 200, 256}, each run twice through the shared workspace: rel L2 <= 3e-3 vs
  x.double() @ W.double().T, a one-hot power-of-two probe row within 1e-4 (exact weights), finite
  output, and the second run bit-identical to the first (deterministic reduction, barriers reset).
  DIAGNOSTICS (prints only, try/except): headers path, compile seconds, workspace bytes per layer,
  the scheduler's chosen units/CTA count if exposed, and warm-L2 cudaEvent kernel time + TFLOP/s
  per layer at m=256, also measured for master's cuBLAS on the same synthetic x so the log gives
  an in-run A/B per layer. STAGES: stage 2 = the same kernel at 65 <= m <= 128 (all four layers;
  for pooler/unpooler compare with pooler-unpooler-m128-tma-wspec-rewrite's rows and keep the
  faster per shape), adding a 128x64 or 64x128 tile instantiation where the profile shows the x
  re-read or the epilogue dominating, and a self-resetting barrier so the memset node goes away;
  stage 3 = per-layer tile tuning (128x256 with two consumer warpgroups for qkv_gate/pooler to
  halve x re-reads, KernelTmaWarpSpecializedPingpong for o where k=1024 makes the epilogue a large
  share) and folding m=17..64 onto a swap-AB variant if the shared kernel beats the Triton swap-AB
  there. MILESTONES readable from the profile top_kernels (one
  cutlass::device_kernel<...GemmUniversal...StreamK...> row per verify_32 shape, plus at most one
  memset node): M1 all 9 tests pass (m=130 unpooler goes through the new kernel) and prepare logs
  every layer enabled; M2 each verify_32 row at or below its incumbent (qkv 19.6, o 9.0, pooler
  30.5, unpooler 24.5 us); M3 qkv_gate <= 14 us, pooler <= 25 us, o <= 7 us, unpooler <= 23.5 us.
expected:
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 96x64 19.4 us kernel (row 19.6, 498 TFLOP/s on 36-72 tiles); floor 9.8 us (math, 9.66 GFLOP at 989; bytes 7.9 us). Stream-K over 132 CTAs at ~700 TFLOP/s: M3 <= 14 us"
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: incumbent nvjet 128x64 29.5 us (row 30.5, 583 TFLOP/s); floor 17.4 us (math, 17.2 GFLOP). 64 output tiles at 128x128 cannot fill 132 SMs without stream-K; M3 <= 25 us
  - shapes: layer=o load=verify_32 m=256
    direction: faster
    why: incumbent nvjet 128x128 8.3 us (row 9.0, 518 TFLOP/s); floor 4.5 us (math 4.35). k=1024 is 16 K-blocks; TMA epilogue overlap and x multicast; M3 <= 7 us
  - shapes: layer=unpooler load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 128x128 24.6 us (row 24.5, 699 TFLOP/s, already near practical peak); floor 17.4 us. Only the 128-tiles-on-132-SMs quantization is left: M3 <= 23.5 us, a small gain"
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: layer=* load=decode_* m=*
    direction: flat
    why: m <= 32 keeps master
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: keeps master
  - shapes: layer=* load=verify_16 m=128
    direction: flat
    why: m=128 keeps master in stage 1 (stage 2 scope)
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet at m=256 (verify_32) for qkv_gate (nvjet_sm90_tst_96x64_64x10_2x4, 19.4 us), o (nvjet 128x128, 8.3 us), pooler (nvjet 128x64, 29.5 us) and unpooler (nvjet 128x128, 24.6 us)
review: required
status: queued
---

# bf16-verify-streamk-cutlass-wgmma-rewrite

## Change

Built on master (8b9ad43); no dependency on an unharvested hypothesis. WHY: at m=256 (verify_32, the load furthest from its floor: 92.9 us vs 51.8 floor) all four bf16 layers run cuBLAS nvjet with data-parallel tiles that cannot fill 132 SMs or are math-bound at well under peak. Measured tensor rates from FINDINGS: qkv_gate nvjet 96x64 19.4 us = 9.66 GFLOP -> 498 TFLOP/s (n=2304 gives only 36-72 output tiles); pooler nvjet 128x64 29.5 us -> 583 TFLOP/s (n=4096: 64 tiles at 128x128); o nvjet 128x128 8.3 us -> 518 TFLOP/s (k=1024: 16 K-blocks, pipeline fill and epilogue dominate); unpooler nvjet 128x128 24.6 us -> 699 TFLOP/s (already near practical peak). A persistent stream-K kernel splits the total MAC-loop iterations evenly over exactly 132 CTAs, so tile-count quantization stops costing anything, and a TMA + warp-specialized wgmma mainloop runs at the 700-750 TFLOP/s a dense sm90 GEMM reaches. No Triton or cuBLASLt-algorithm path does this (the cuBLASLt algo pick only re-orders cuBLAS's own candidates and is under a rules heads-up; this hypothesis does no timing or algorithm search in prepare at all: the tile config is a fixed constant). STAGE 1 (this file): one CUTLASS 3.x sm90 GEMM, compiled once at import with torch.utils.cpp_extension.load_inline (nvcc; flags -std=c++17 --expt-relaxed-constexpr -DNDEBUG -gencode arch=compute_90a,code=sm_90a, which wgmma and TMA require). Headers: search, in order, the CUTLASS/CuTe include dirs shipped inside the image (deep_gemm's package include dir, flashinfer's data/3rdparty cutlass include, sgl_kernel / cutlass_library source dirs, /usr/local/cutlass/include), print the path used; if none has cutlass/gemm/collective/collective_builder.hpp, print '[streamk] no CUTLASS headers, master kept' and skip. The kernel: GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler> via CollectiveBuilder<Sm90, OpClassTensorOp, bf16 A RowMajor (x [m,k]) align 8, bf16 B ColumnMajor (W [n,k] row-major read as K x N column-major, i.e. a TN GEMM, no repack) align 8, float accumulate, TileShape 128x128x64, ClusterShape 1x2x1 (the two CTAs along N share each x tile by TMA multicast), StageCountAutoCarveout, KernelTmaWarpSpecializedCooperative>, epilogue builder TmaWarpSpecializedCooperative writing bf16 RowMajor D with no C (beta 0). ONE instantiation shared by all four layers (problem shape is a runtime argument), to keep compile time near one minute; print compile seconds. Host wrapper gemm(x, W, out, workspace): builds Arguments {kGemm, {m, n, k, 1}, {x, stride, W, stride}, {{1, 0}, nullptr, ..., out, stride}, hw_info with sm_count 132 cached from prepare}, scheduler args decomposition_mode = StreamK, reduction_mode = Deterministic (fixed summation order), calls can_implement (raise on failure, which only happens in prepare), initialize(args, workspace_ptr, stream) and run(stream) on at::cuda::getCurrentCUDAStream(). Workspace: Gemm::get_workspace_size for m=256 per layer, allocated ONCE in prepare() (state['streamk_ws'], max over layers) and reused every call, never sized from input values. initialize() may zero the stream-K barrier region with a cudaMemsetAsync; under graph capture that is one memset node, which is acceptable for stage 1 and must be visible in the profile (stage 2 removes it). ROUTING: kernel_fn takes the new path for layer in {qkv_gate, o, pooler, unpooler} at 129 <= cfg m <= 256 when state['streamk_ok'][layer] is True, cfg n/k match the full weight, and x is contiguous bf16; the judge's m=130 unpooler test therefore exercises it (TMA zero-fills the out-of-range rows of the last M tile). m = 512/1000 (tests), m <= 128, the router and every other shape keep master byte for byte. SELF-CHECK in prepare() per layer (failure prints '[streamk] <layer> DISABLED: <reason>' and sets the flag False, a constant fixed in prepare): can_implement OK; synthetic x at m in {130, 200, 256}, each run twice through the shared workspace: rel L2 <= 3e-3 vs x.double() @ W.double().T, a one-hot power-of-two probe row within 1e-4 (exact weights), finite output, and the second run bit-identical to the first (deterministic reduction, barriers reset). DIAGNOSTICS (prints only, try/except): headers path, compile seconds, workspace bytes per layer, the scheduler's chosen units/CTA count if exposed, and warm-L2 cudaEvent kernel time + TFLOP/s per layer at m=256, also measured for master's cuBLAS on the same synthetic x so the log gives an in-run A/B per layer. STAGES: stage 2 = the same kernel at 65 <= m <= 128 (all four layers; for pooler/unpooler compare with pooler-unpooler-m128-tma-wspec-rewrite's rows and keep the faster per shape), adding a 128x64 or 64x128 tile instantiation where the profile shows the x re-read or the epilogue dominating, and a self-resetting barrier so the memset node goes away; stage 3 = per-layer tile tuning (128x256 with two consumer warpgroups for qkv_gate/pooler to halve x re-reads, KernelTmaWarpSpecializedPingpong for o where k=1024 makes the epilogue a large share) and folding m=17..64 onto a swap-AB variant if the shared kernel beats the Triton swap-AB there. MILESTONES readable from the profile top_kernels (one cutlass::device_kernel<...GemmUniversal...StreamK...> row per verify_32 shape, plus at most one memset node): M1 all 9 tests pass (m=130 unpooler goes through the new kernel) and prepare logs every layer enabled; M2 each verify_32 row at or below its incumbent (qkv 19.6, o 9.0, pooler 30.5, unpooler 24.5 us); M3 qkv_gate <= 14 us, pooler <= 25 us, o <= 7 us, unpooler <= 23.5 us.

## Expected

- `layer=qkv_gate load=verify_32 m=256` faster: incumbent nvjet 96x64 19.4 us kernel (row 19.6, 498 TFLOP/s on 36-72 tiles); floor 9.8 us (math, 9.66 GFLOP at 989; bytes 7.9 us). Stream-K over 132 CTAs at ~700 TFLOP/s: M3 <= 14 us
- `layer=pooler load=verify_32 m=256` faster: incumbent nvjet 128x64 29.5 us (row 30.5, 583 TFLOP/s); floor 17.4 us (math, 17.2 GFLOP). 64 output tiles at 128x128 cannot fill 132 SMs without stream-K; M3 <= 25 us
- `layer=o load=verify_32 m=256` faster: incumbent nvjet 128x128 8.3 us (row 9.0, 518 TFLOP/s); floor 4.5 us (math 4.35). k=1024 is 16 K-blocks; TMA epilogue overlap and x multicast; M3 <= 7 us
- `layer=unpooler load=verify_32 m=256` faster: incumbent nvjet 128x128 24.6 us (row 24.5, 699 TFLOP/s, already near practical peak); floor 17.4 us. Only the 128-tiles-on-132-SMs quantization is left: M3 <= 23.5 us, a small gain
- `layer=router *` flat: untouched
- `layer=* load=decode_* m=*` flat: m <= 32 keeps master
- `layer=* load=verify_1 m=8` flat: keeps master
- `layer=* load=verify_16 m=128` flat: m=128 keeps master in stage 1 (stage 2 scope)

## Risk

Headline reasoning: at M3, verify_32's bf16 rows drop from 83.6 to ~69.5 us (qkv -5.6, pooler -5.5, o -2.0, unpooler -1.0; router 9.3 unchanged), so the load goes 92.9 -> ~78.8 us, 1.099x -> 1.296x vs SGLang's 102.1: about +2.8% on the 6-load geomean from stage 1 alone. Stage 2 applies the same kernel to verify_16 (qkv 16.8 vs 8.4 floor with a separate 3.0 us splitKreduce node, pooler 25.6 vs 14.6, unpooler 19.8, o 7.1), worth roughly another +3-4%, so the family is ~+6-7% when done: the m>=128 half of the gap to a 1.56x leader, whose rows must sit near 1.3x of the floors everywhere. BIGGEST RISK: building CUTLASS from inside the submission. The headers may not be findable in the image (fallback prints and keeps master, so the run is wasted but harmless), and a CUTLASS 3.x sm90 stream-K instantiation can take 60-120 s of nvcc time against the 10-minute run limit (attention's raw-CUDA load_inline took 41 s): keep exactly one instantiation in stage 1 and print the compile time. Second risk: CUTLASS's stream-K heuristics may choose data-parallel for some shapes (o, unpooler) on their own, which is fine, and the memset node initialize() adds (~1 us) eats part of the o gain; the profile shows both. Overlap: bf16-cublaslt-algo-pick-in-prepare-m128-m256 (running) targets the same m=256 rows by choosing among cuBLAS's own algorithms (and carries the open prepare-time timing heads-up); this hypothesis uses a fixed config and no prepare-time timing, so it is independent of that ruling. Per-shape rows decide which a combo keeps. Rules: bf16 weights unchanged and read in place, bf16 activations, fp32 accumulation with a deterministic stream-K fixup, no host syncs, graph-capturable (workspace preallocated, launch parameters depend only on shapes). Risk high, so check runs only.

## Milestones

Rewrite stage 1 of `cuBLAS nvjet at m=256 (verify_32) for qkv_gate (nvjet_sm90_tst_96x64_64x10_2x4, 19.4 us), o (nvjet 128x128, 8.3 us), pooler (nvjet 128x64, 29.5 us) and unpooler (nvjet 128x128, 24.6 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
