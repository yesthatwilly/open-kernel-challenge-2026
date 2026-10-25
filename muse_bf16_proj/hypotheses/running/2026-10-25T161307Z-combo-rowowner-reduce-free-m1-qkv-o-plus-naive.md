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
review: proceed
status: running
review_at: 2026-10-25T161800Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T161801Z
base_sha: a2286ee3352c35c3e80062a62bc74df5393b1d95
base_blob: f919748b5af6e815f319e5d2d828ed68e5b28a71
branch: hyp/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive
worktree: .worktrees/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T161800Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with two required changes. (1) Source B's per-call set/restore of preferred_blas_library('cublaslt') is forbidden: replace it with a cuBLASLt plan fixed in prepare and held in state, or drop that cell. (2) Drop _bulk_forward's state['bulk_route_hit'] write. The A2 smem budget fits 2 CTAs/SM (99,072 B dynamic each). The Conjecture's mechanism (the qkv row-minus-kernel gap comes from the split-K reduce) is already contradicted by master's own profile, so restate the confirm/refute criteria on kernel time before running.

### Justification
A. Sources A and B are carried over unchanged from the sibling combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt and are justified by measured rows. Pooler d1: bulk kernel 17.0 us vs swapab 18.7 us (row 17.28 vs 18.93). qkv m=128: nvjet 48x64 single node 16.2 us vs nvjet 96x64 splitK 12.7 us + splitKreduce 3.0 us. The new parts are A2 (qkv_gate m=1, bulk_kernel<16,8192,8,0>, grid 144, 4 stages, 2 CTAs/SM) and A3 (o m=1, bulk_kernel<64,1024,8,0>, grid 128, 6 stages). Geometry checks out. A2: CKK=512, NCHUNK=16, XS=1040, SLOT=16384+8*1040=24704 (24704%128==0), and R*CKK*2=16384. KSW=8 per consumer warp, kbi<=7, so each chunk is 8 kblocks x 16 rows x 128 B = 16 KB. _bulk_repack's [c][kb][row][64] layout makes each chunk contiguous at (c*NCHUNK+j)*16384 B for any R, so no repack change is needed. The epilogue N=gridDim.x*R=2304 is correct. scratch [4][16][8] fp32 = 2 KB, which aliases the ring safely. A3: CKK=128, NCHUNK=8, XS=272, SLOT=18560 (18560%128==0), KSW=2, MT=4, dyn smem 256+6*18560=111,616 B, 1 CTA/SM, grid 128 = 8192/64. SMEM FIT for the (160,2) variant: dynamic smem = 256 + 4*24704 = 99,072 B per CTA. Adding sm_90's 1 KB per-block reservation gives 2 x 100,096 = 200,192 B, under the 233,472 B (228 KB) per SM. It fits with about 33 KB to spare. Five stages would not fit (2 x (123,776+1,024) = 249,600 B), so 4 is the deepest ring that allows 2/SM. Registers are not binding (65536/320 = 204/thread; acc is 4 floats at MT=1, NN=1). Note that __launch_bounds__(160,2) only caps registers; the smem carveout decides whether the second CTA is co-resident. WEAKEST LINK: the Conjecture's premise. Master's own data contradicts it. Pooler m=1 on master also runs a split-K ticket reduce (SWAPAB16_CFG pooler SPLIT_K 2), yet its row-minus-kernel gap is 0.2 us. o m=1 runs SPLIT_K 1 with no reduce, yet its gap is 2.1 us. Source A's pooler win came entirely from kernel time (18.7 to 17.0 us); the gap went 0.23 to 0.28 us, essentially unchanged. So the gap does not track the reduce. At small row times it is more likely a fixed ramp-up or profile-vs-graph-row artifact, and 'gap shrinks below ~1 us' cannot discriminate. On kernel time alone, A2 must beat 9.6 us for 37.7 MB (3.93 TB/s). The row-owner kernel's best observed effective rate was ~3.95 TB/s (67.1 MB / 17.0 us), which projects to ~9.5 us, about flat. It also gets half the ring depth (4 stages, 64 KB/CTA in flight) and an uneven 144-on-132 placement: 12 SMs host two CTAs and stream 512 KB vs 256 KB. The expected 12.7 to 10.5-11 us is therefore optimistic. A 'master kept' outcome from A2's gate is the most likely result and is a valid refutation, which the run CAN read off the PATH CHOSEN line and the profile. A3 is a sound control-like cell: 8 chunks cannot amortise a 6-stage fill, so the gate should keep master. Expected directions are consistent with TASK floors (qkv d1 7.9 us, o d1 3.5 us, pooler d1 14.0 us, qkv m128 8.4 us). The total upside is about +0.8% from the sibling's cells plus at most ~0.3-0.7% from A2, below the ±1.5% headline CI. Attribution must come from the per-cell gate prints and the profile rows, not the headline.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32: defines the incumbent at qkv_gate/o/pooler m=1 (swapab16: qkv SPLIT_K 4, o SPLIT_K 1, pooler SPLIT_K 2). It is the comparison point for A/A2/A3, not a confounder. However, the qkv incumbent's 3.1 us gap and o's 2.1 us gap come from this path's rows, and the pooler (SPLIT_K 2, 0.2 us gap) shows that the gap is not the reduce.
- sglang-gemm-dispatch-parity: defines the qkv_gate m=128 incumbent (_bf16_gemm_dispatch_impl -> nvjet 96x64 splitK + splitKreduce) that Source B replaces. It is the control for B, and the cell gate compares against it directly.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 / pooler-nn-layout-widen-m256: no shared shapes (unpooler slice m<=8, router m<=8, NN layout m>=65 on pooler/unpooler only, not qkv). They are relevant only through prepare budget and import-time build time: ouns + router_cluster + bulk builds are all inside the 10-minute run limit, so log each build's seconds.
- router/bf16 Triton landings: different layers and shapes, no interaction beyond the shared cold-L2 timing method.

### Interference and A/B needs
No companion. Every new cell is behind its own in-prepare A/B gate, which prints both medians (graph replay, 256 MB flush, 41 interleaved reps) and the chosen path. With the required torch.profiler kernel-name assertion, each scored row is attributable without another run: a routed cell shows bulk_kernel or the Lt kernel in the profile with its own µs, and a non-routed cell is byte-identical to master. The sibling combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt is already the natural 'minus A2/A3' control if both are submitted. Even if it is not, A2's and A3's contributions are read from their PATH CHOSEN lines plus the qkv_gate/o decode_1 rows vs master's 12.67/5.97 us. A third scored run would buy nothing the gate prints do not already give.

### Open questions
- Restate the Conjecture's confirm/refute criteria before running; the hypothesis md can be edited, the code is unaffected. Confirmed: A2's gate routes (cand <= 0.98x) AND the profile's bulk_kernel<16,...> time is < 9.6 us AND the qkv decode_1 row < 12.3 us. Refuted: the gate keeps master, or the kernel time is >= 9.6 us. Drop 'row-minus-kernel gap < 1 us' as the discriminator, because master's pooler (SPLIT_K 2, gap 0.2 us) and o (SPLIT_K 1, gap 2.1 us) already show the gap does not track the reduce. Also drop 'A3 wins means the effect is TMA issue, not reduce removal', since o has no reduce to remove and a win there is just a faster stream.
- With 144 CTAs on 132 SMs at 2/SM, 12 SMs carry double bytes. Print the gate's candidate median next to a projected 37.7 MB / 3.95 TB/s = 9.5 us, so a loss can be split into ring-depth vs imbalance.
- The A2 template adds a stage-count parameter. Make sure bulk_launch's smem formula, the kernel's S, and bulk_inst_info's printed 'dyn smem' all use the same template S, not the BULK_STAGES #define.
- Prepare/import budget: log each load_inline build's wall time (ouns, router_cluster, bulk, Lt) and each gate's seconds. The whole run must finish inside 10 minutes including timing.

