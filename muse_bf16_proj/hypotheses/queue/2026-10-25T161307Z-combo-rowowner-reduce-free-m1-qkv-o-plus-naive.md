---
problem: muse_bf16_proj
slug: combo-rowowner-reduce-free-m1-qkv-o-plus-naive
created: 2026-10-25T161307Z
base: master
area: combination (naive stack + bulk row-owner widened to qkv_gate/o m=1)
change: >
  Built on current master (commit 67b46a57, blob f919748b). Everything in
  combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS
  a conjectural widening of source A's bulk row-owner kernel to the other two m=1 weight streams
  that master runs with a split-K reduce or a high shape-minus-kernel gap, each behind its own
  gate: (A2) qkv_gate m == 1 (n 2304, k 8192): master runs _bf16_swapab_kernel via swapab16 with
  SPLIT_K 4 and a ticket reduce, kernel 9.6 us but row 12.67 us (a 3.1 us gap, vs 0.2 us on pooler
  d1). New instantiation bulk_kernel<16, 8192, 8, 0> (CKK 512, MT 1) with a grid of N/R = 144
  CTAs; make the grid a launch argument (gridDim.x*R = N already in the epilogue) and reduce
  BULK_STAGES to 4 for this instantiation (template the stage count) so dyn smem ~ 256 + 4*(16384
  + 8*1040) = 99 KB allows 2 CTAs/SM and all 144 CTAs are resident in one wave; change
  __launch_bounds__(160, 1) to (160, 2) for it. (A3) o m == 1 (n 8192, k 1024): bulk_kernel<64,
  1024, 8, 0> (CKK 128, NCHUNK 8, MT 4), grid 128, 6 stages; master runs swapab16 o SPLIT_K 1
  (kernel 3.9 us, row 5.97 us, 2.1 us gap). Repack qkv_gate and o with _bulk_repack(w, R) per
  state (37.7 MB + 16.8 MB, lossless permutation, round trip asserted bit-exact). Check the
  static_asserts: R*CKK*2 == 16384 holds for (16, 512) and (64, 128). Sources: SOURCE A:
  pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch
  hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45
  (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch,
  bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import
  like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack,
  _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s
  kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only ->
  bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip
  verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence:
  pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master
  1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1
  cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B:
  bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl
  commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of
  torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl
  call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture.
  Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt
  splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK
  12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x),
  so it never applies there. Note for review: this toggle is a host-side library preference set
  and restored within the call (shape-gated, never value-gated); the judge's static check passed
  it once (5ed9ace10d9a4f68). Routing: {pooler, qkv_gate, o} m == 1 -> A/A2/A3 per cell gate, else
  master swapab16; qkv m==128 -> B; everything else master byte-identical (the m=1 qkv_gate TEST
  runs whichever path the gate chose, so A2 must pass the same self-check first). GATES (one per
  source per cell, all AFTER master's own _prepare_state work, master's code and constants
  untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and
  candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps,
  empty-graph baseline subtracted, both medians printed). Route the candidate only if
  candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <=
  3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep
  master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master>
  (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell
  with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the
  selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked
  weights, error words and flags; no module-global device pointers). The second state of master's
  A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and
  the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m)
  line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation
  sized by input values. Canary: run each new instantiation once on synthetic x with the
  bounded-wait error word before any graph capture, print LAUNCH/DONE lines around it (LESSONS
  2026-10-25: a fault in prepare poisons the context; this kernel family already ran clean at
  R=32/64 in the judge). PREPARE BUDGET: bulk build with three MODE-0 instantiations ~45-55 s at
  import; repacks + canaries + three cell gates <= 4 s per state (deadline 200 s); Lt as combo 1
  (deadline 250 s). Total prepare ~15 s plus ~125 s of import-time builds.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: source A, as combo 1
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: source B, as combo 1
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: faster
    why: "conjecture: reduce-free row-owner removes the split-K ticket and the 3.1 us row-kernel gap; row 12.7 -> ~10.5-11 if so"
  - shapes: layer=o load=decode_1 m=1
    direction: unchanged
    why: "control-like cell: master already one-pass (SPLIT_K 1) at ~4.3 TB/s; expect the gate to keep master"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-rowowner-reduce-free-m1-qkv-o-plus-naive

## Change

Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS a conjectural widening of source A's bulk row-owner kernel to the other two m=1 weight streams that master runs with a split-K reduce or a high shape-minus-kernel gap, each behind its own gate: (A2) qkv_gate m == 1 (n 2304, k 8192): master runs _bf16_swapab_kernel via swapab16 with SPLIT_K 4 and a ticket reduce, kernel 9.6 us but row 12.67 us (a 3.1 us gap, vs 0.2 us on pooler d1). New instantiation bulk_kernel<16, 8192, 8, 0> (CKK 512, MT 1) with a grid of N/R = 144 CTAs; make the grid a launch argument (gridDim.x*R = N already in the epilogue) and reduce BULK_STAGES to 4 for this instantiation (template the stage count) so dyn smem ~ 256 + 4*(16384 + 8*1040) = 99 KB allows 2 CTAs/SM and all 144 CTAs are resident in one wave; change __launch_bounds__(160, 1) to (160, 2) for it. (A3) o m == 1 (n 8192, k 1024): bulk_kernel<64, 1024, 8, 0> (CKK 128, NCHUNK 8, MT 4), grid 128, 6 stages; master runs swapab16 o SPLIT_K 1 (kernel 3.9 us, row 5.97 us, 2.1 us gap). Repack qkv_gate and o with _bulk_repack(w, R) per state (37.7 MB + 16.8 MB, lossless permutation, round trip asserted bit-exact). Check the static_asserts: R*CKK*2 == 16384 holds for (16, 512) and (64, 128). Sources: SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45 (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch, bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack, _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only -> bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence: pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master 1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1 cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B: bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture. Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK 12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x), so it never applies there. Note for review: this toggle is a host-side library preference set and restored within the call (shape-gated, never value-gated); the judge's static check passed it once (5ed9ace10d9a4f68). Routing: {pooler, qkv_gate, o} m == 1 -> A/A2/A3 per cell gate, else master swapab16; qkv m==128 -> B; everything else master byte-identical (the m=1 qkv_gate TEST runs whichever path the gate chose, so A2 must pass the same self-check first). GATES (one per source per cell, all AFTER master's own _prepare_state work, master's code and constants untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked weights, error words and flags; no module-global device pointers). The second state of master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation sized by input values. Canary: run each new instantiation once on synthetic x with the bounded-wait error word before any graph capture, print LAUNCH/DONE lines around it (LESSONS 2026-10-25: a fault in prepare poisons the context; this kernel family already ran clean at R=32/64 in the judge). PREPARE BUDGET: bulk build with three MODE-0 instantiations ~45-55 s at import; repacks + canaries + three cell gates <= 4 s per state (deadline 200 s); Lt as combo 1 (deadline 250 s). Total prepare ~15 s plus ~125 s of import-time builds.

## Expected

- `layer=pooler load=decode_1 m=1` faster: source A, as combo 1
- `layer=qkv_gate load=verify_16 m=128` faster: source B, as combo 1
- `layer=qkv_gate load=decode_1 m=1` faster: conjecture: reduce-free row-owner removes the split-K ticket and the 3.1 us row-kernel gap; row 12.7 -> ~10.5-11 if so
- `layer=o load=decode_1 m=1` unchanged: control-like cell: master already one-pass (SPLIT_K 1) at ~4.3 TB/s; expect the gate to keep master

## Risk

Expected: combo 1's ~+0.8% plus up to +0.7% from qkv decode_1 (-2 us of a 47.5 us load). Risk medium: new template instantiations of a kernel that ran clean, behind a canary and a hard gate. ## Conjecture: the row-owner kernel won pooler m=1 not by bandwidth (its cold pure read was only 2.54 TB/s in prepare and ~3.95 TB/s effective in the judge, about master's swap-AB rate on qkv) but by being reduce-free: one CTA owns whole output rows, so there is no split-K ticket, no fp32 workspace round trip and no tail CTA doing the reduce. If so, the gain carries to the cell with the deepest split (qkv_gate, SPLIT_K 4, the largest row-minus-kernel gap at 3.1 us) and NOT to o (SPLIT_K 1, 0.0 reduce, K only 1024, where an 8-chunk ring cannot amortise its fill). A second interaction is occupancy: A2 needs 2 CTAs/SM at 4 stages, so it also tests whether the row-owner stream survives half the ring depth. Confirmed if A2's gate prints cand <= 0.98x and the profile's qkv decode_1 row drops with its row-minus-kernel gap shrinking below ~1 us, while A3's gate keeps master. Refuted if A2 is flat or slower with the same kernel time (then the 3.1 us gap is not the reduce) or if A3 also wins (then the effect is the TMA bulk issue path, not reduce removal).
