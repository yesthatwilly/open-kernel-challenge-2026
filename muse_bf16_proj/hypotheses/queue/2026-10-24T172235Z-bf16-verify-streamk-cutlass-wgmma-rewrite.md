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
review: proceed
status: queued
review_at: 2026-10-24T172632Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T172632Z by an adversarial subagent (model opus).

**Verdict: proceed.** The judge image does ship CUTLASS. rules/technical-rules.llms.txt (Environment) says submissions run in lmsysorg/sglang:dev-muse-spark-open and may 'import and build on anything in the image, including PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS', with native CUDA via load_inline/nvcc, and attention's load_inline already compiled on the judge in 41 s. So stage 1 can be built and interpreted; proceed with sharp notes on header discovery, cluster orientation and forced stream-K on o/unpooler.

### Justification
Environment (the make-or-break question): nothing in the repo names a CUTLASS include path, the torch version or the nvcc version. TASK.md and AGENTS.md say nothing about headers, and no bf16 log prints a path. What is on record: (1) the official technical rules list CUTLASS, CuTe DSL, DeepGEMM and FlashInfer as present in the image and endorse load_inline (nvcc); (2) muse_attention_decode archive logs 82085a4e82154393 and de486093311d4bb9 show '[splitkv] load_inline compiled in 41.0s/41.5s' on the judge with -arch=sm_90a, so nvcc, ninja and a writable build dir all exist. The cutlass:: kernels in BEST.result.json and FINDINGS (cutlass_80_simt_sgemm, flash::...) are prebuilt binaries, not evidence of headers. Header locations are therefore likely but unverified. The most probable C++ trees are flashinfer/data/cutlass/include (plus flashinfer/data/cutlass/tools/util/include) and deep_gemm/include (cute/ + cutlass/, no tools/util). The brief's 'flashinfer data/3rdparty' path is probably wrong, and the CuTe DSL python package (nvidia-cutlass-dsl) ships no C++ headers. The hypothesis's fallback ('no CUTLASS headers, master kept') makes a miss harmless but wastes the run, so find the headers by glob, not a fixed list. Mechanism: sound. At m=256 the four layers are math-heavy (qkv 9.66 GFLOP against a 7.9 us byte floor; pooler and unpooler 17.2 GFLOP) and nvjet runs them at 498-699 TFLOP/s. qkv_gate (36 tiles of 128x128) and pooler (64 tiles) clearly under-fill 132 SMs, so stream-K on those two is the right lever, and the profile rows (19.4 / 29.5 us) can show the milestone directly. o and unpooler have 128 tiles of 128x128 on 132 SMs, already about one wave, so stream-K buys almost nothing there. For o (16 K-blocks) a forced StreamK decomposition adds fixup and memset cost to an 8.3 us kernel, so M2 for o is at risk. Design issue: ClusterShape 1x2x1 multicasts x, the SMALL operand (256x8192 bf16 = 4 MB), and leaves both M-tiles to fetch the weight separately (37.7-67 MB, the operand that sets the byte floor under a cold L2). With m=256 and TileM=128 there are exactly 2 M-tiles, so ClusterShape 2x1x1 (multicast W across the two M-tiles) makes each weight tile cross DRAM to SM exactly once and halves L2-to-SM weight traffic. It also keeps cluster count = N-tiles (qkv 18, pooler 32, o/unpooler 64 on 66 cluster slots). Expected shapes and directions match TASK floors and weights: only verify_32 rows are targeted, and m<=128 keeps master byte for byte. The m=130 unpooler test is correctly routed and needs TMA OOB zero-fill, which CUTLASS handles. WEAKEST LINK: the qkv_gate M3 target (<=14 us) needs ~690 TFLOP/s sustained in a ~14 us kernel on a cold L2 that must also stream 37.7 MB at ~2.7 TB/s. The fixed costs (prologue/pipeline fill, deterministic stream-K fixup with serialized peers, and the initialize() memset node of ~1-2 us under the graph) are 10-15% of that budget. M2 is plausible; M3 for qkv and o is optimistic.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
Nothing has landed on master's lineage beyond the base the hypothesis names (landed list empty), and the change replaces only the cuBLAS call at 129<=m<=256, which no landed Triton path touches (all are m<=32 plus the router). No landed interaction can produce a gain or loss here. The stage is a rewrite judged on its own profile rows, and the brief already builds an in-run per-layer A/B (cudaEvent warm-L2 time of the CUTLASS kernel vs master's cuBLAS on the same synthetic x). That diagnostic, together with the top_kernels rows, makes both a gain and a loss attributable without a companion. A control isolating stream-K vs data-parallel would be useful but belongs in stage 2 (pass DecompositionMode as a runtime argument and print both in the prepare diagnostics instead of spending a run). In-flight overlap to note, not to companion: bf16-cublaslt-algo-pick-in-prepare-m128-m256 targets the same m=256 rows. If both land, the per-shape rows decide which is kept; do not stack them in one run.

### Open questions
- Header discovery: rather than a fixed list, glob importlib-resolved package dirs (flashinfer, deep_gemm, sgl_kernel, tilelang, cutlass_library) plus /usr/local/cutlass and /usr/include for '**/cutlass/gemm/collective/collective_builder.hpp' with a sibling 'cute/' directory. Print every hit and the CUTLASS_MAJOR/MINOR/PATCH from cutlass/version.h. The StreamK scheduler argument names (decomposition_mode, reduction_mode) moved between 3.x releases, and DeepGEMM's tree may be 4.x.
- Avoid any dependency on cutlass/util/* (packed_stride.hpp, device_memory.h): it lives in tools/util/include, which deep_gemm does not ship. Build strides with cute::make_stride (or include tools/util only if found).
- Switch ClusterShape to 2x1x1 (multicast the weight across the two M-tiles at m=256), or justify 1x2x1. Multicasting x (4 MB) saves little, while W is the floor-setting operand under a cold L2.
- For o and unpooler (128 tiles on 132 SMs), use DecompositionMode::Heuristic, or DataParallel, instead of forced StreamK. Otherwise the fixup plus memset can make o miss M2 (9.0 us). Print the scheduler's chosen mode and unit count per layer.
- Deterministic reduction serializes peer fixups. If M2 is missed on qkv/pooler, test Nondeterministic in stage 2 (correctness bounds do not require bitwise determinism; relax the self-check to finite plus rel L2 then).
- Budget compile time: torch/extension.h plus a CUTLASS 3 sm90 collective is typically 60-150 s. If bf16-small-m-cuda-gemv-rewrite's load_inline later lands alongside, compile both in one load_inline module so the 10-minute run holds.
- Does CUTLASS's StreamK tile scheduler accept a 2-CTA cluster in the installed version together with KernelTmaWarpSpecializedCooperative? If can_implement fails, the fallback to cluster 1x1x1 should be a second compiled instantiation only if compile time allows; otherwise log and keep master.

