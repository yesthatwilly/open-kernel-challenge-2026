---
problem: muse_bf16_proj
slug: combo-cublaslt-single-node-m128-m256-plus-naive
created: 2026-10-25T161307Z
base: master
area: combination (naive stack + cuBLASLt single-node probe on pooler/unpooler/o m=128/256)
change: >
  Built on current master (commit 67b46a57, blob f919748b). Everything in
  combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS
  a conjectural widening of source B to the other m=128/256 cuBLAS cells, each behind its own
  gate: (B2) pooler m=128 and unpooler (n 8192) m=128: torch.mm(x, state[layer+'_t']) (master's NN
  path) under the cuBLASLt preference; (B3) o m=128: master's _bf16_gemm_dispatch_impl under Lt;
  (B4) pooler m=256 NN (master's nn_p256 path), unpooler m=256 and o m=256 under Lt. qkv_gate
  m=256 is EXCLUDED (Lt lost it 0.920x in the source). Sources: SOURCE A:
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
  it once (5ed9ace10d9a4f68). The source run measured Lt only on the TN layout (pooler v16 0.999x
  TN 64x64, unpooler v16 0.998x TN, o v16 1.012x same kernel); the NN-layout cells are untested,
  which is the point. Routing: pooler m==1 -> A; qkv m==128 -> B; {pooler, unpooler} m==128 -> B2
  if its gate wins else master NN; o m==128 -> B3 else master; {pooler, unpooler, o} m==256 -> B4
  per cell else master; everything else master byte-identical. GATES (one per source per cell, all
  AFTER master's own _prepare_state work, master's code and constants untouched): copy master's
  _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same
  synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline
  subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND
  its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single
  power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one
  'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per
  source per cell, and profile one call of each routed cell with torch.profiler and assert the
  launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives
  in state (two judge copies: per-state repacked weights, error words and flags; no module-global
  device pointers). The second state of master's A/B self-check inherits the first state's flags.
  kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the
  source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is
  per-call mutable state), no syncs, no allocation sized by input values. Each Lt cell's gate
  captures its graph under the preference, so what is timed is what is replayed. PREPARE BUDGET: A
  as combo 1 (~37-40 s import build, <= 2 s/state, deadline 200 s); Lt warm-up + 7 cell gates at
  ~0.3-0.5 s each <= 4 s per state (deadline 250 s). Total prepare ~15 s plus ~115 s of
  import-time builds.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: source A, as combo 1
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: source B, as combo 1
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "conjecture: Lt picks a single-node NN algo, dropping the 3.1 us splitKreduce node; row 23.2 -> ~21 if so, else gate keeps master"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: unchanged
    why: master NN is already one node (19.6 us); Lt may pick a fuller tile; gate decides
  - shapes: layer=* load=verify_32 m=256
    direction: unchanged
    why: "probing cells: pooler/unpooler/o m=256 under Lt, gate decides; qkv m=256 excluded"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T161800Z
review_companions: null
hold_reason: "Review: B3 and part of B4 launch identical kernels under Lt and cuBLAS; pooler/unpooler m=128 under Lt measured slower before; if B2-B4 gate to master it equals the naive combo, so it would spend a scored run re-measuring. Held in favour of the naive and row-owner combos."
outcome: expired
verdict_detail: "retired: superseded by bf16-verify-per-cell-cublaslt-plan-selector (2026-10-25 mint)"
expired_at: 2026-10-25T211947Z
---

# combo-cublaslt-single-node-m128-m256-plus-naive

## Change

Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS a conjectural widening of source B to the other m=128/256 cuBLAS cells, each behind its own gate: (B2) pooler m=128 and unpooler (n 8192) m=128: torch.mm(x, state[layer+'_t']) (master's NN path) under the cuBLASLt preference; (B3) o m=128: master's _bf16_gemm_dispatch_impl under Lt; (B4) pooler m=256 NN (master's nn_p256 path), unpooler m=256 and o m=256 under Lt. qkv_gate m=256 is EXCLUDED (Lt lost it 0.920x in the source). Sources: SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45 (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch, bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack, _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only -> bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence: pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master 1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1 cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B: bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture. Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK 12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x), so it never applies there. Note for review: this toggle is a host-side library preference set and restored within the call (shape-gated, never value-gated); the judge's static check passed it once (5ed9ace10d9a4f68). The source run measured Lt only on the TN layout (pooler v16 0.999x TN 64x64, unpooler v16 0.998x TN, o v16 1.012x same kernel); the NN-layout cells are untested, which is the point. Routing: pooler m==1 -> A; qkv m==128 -> B; {pooler, unpooler} m==128 -> B2 if its gate wins else master NN; o m==128 -> B3 else master; {pooler, unpooler, o} m==256 -> B4 per cell else master; everything else master byte-identical. GATES (one per source per cell, all AFTER master's own _prepare_state work, master's code and constants untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked weights, error words and flags; no module-global device pointers). The second state of master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation sized by input values. Each Lt cell's gate captures its graph under the preference, so what is timed is what is replayed. PREPARE BUDGET: A as combo 1 (~37-40 s import build, <= 2 s/state, deadline 200 s); Lt warm-up + 7 cell gates at ~0.3-0.5 s each <= 4 s per state (deadline 250 s). Total prepare ~15 s plus ~115 s of import-time builds.

## Expected

- `layer=pooler load=decode_1 m=1` faster: source A, as combo 1
- `layer=qkv_gate load=verify_16 m=128` faster: source B, as combo 1
- `layer=pooler load=verify_16 m=128` faster: conjecture: Lt picks a single-node NN algo, dropping the 3.1 us splitKreduce node; row 23.2 -> ~21 if so, else gate keeps master
- `layer=unpooler load=verify_16 m=128` unchanged: master NN is already one node (19.6 us); Lt may pick a fuller tile; gate decides
- `layer=* load=verify_32 m=256` unchanged: probing cells: pooler/unpooler/o m=256 under Lt, gate decides; qkv m=256 excluded

## Risk

Expected: combo 1's ~+0.8% plus 0 to +0.6% if the NN cells take Lt single-node algos. Risk medium: pure dispatch for the B cells, so build risk is A's only. ## Conjecture: the verify_16 gap to the floor is graph-node count, not kernel speed. cuBLAS's heuristic picks split-K plus a separate cublasLt splitKreduce node at m=128 for both qkv (TN) and pooler (NN); the cuBLASLt heuristic chose a one-node 48x64 algo for qkv TN. The conjecture is that the preference, not the layout, drives that choice, so the NN pooler m=128 cell also loses its reduce node under Lt (as LESSONS 04:36Z found for memset nodes at m<=8, but here on a 23 us row). It also tests whether a selector re-decides once its neighbour changed: master's nn_ok A/B self-check now compares NN-under-Lt against TN-under-cuBLAS. Confirmed if the profile's pooler verify_16 row shows one nvjet kernel and no splitKreduce, and the B2 gate line prints cand < 0.98x master. Refuted if Lt keeps splitK+splitKreduce on NN (same kernel names as master), in which case the reduce node is a property of the K=8192 shape and the queued pooler-k-halves-batched-gemm-m65-m256 is the right lever. The m=256 B4 cells are probes: read their gate medians even where master is chosen. Overlaps the cells of the queued qkv-k-split-batched-gemm-m65-m256 and pooler-k-halves-batched-gemm-m65-m256; if either lands first, its path becomes the incumbent the gates compare against.

## Adversarial review

Reviewed 2026-10-25T161800Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed only with source B and the B2-B4 cells rebuilt on a rules-clean mechanism. The per-call save/set/restore of torch.backends.cuda.preferred_blas_library breaks standing policy and must go; in its place, port the prepare-time cuBLASLt extension (_LT_CPP, impl a77cdec6, which already ran clean in scored job 26a8be34), query its heuristic at the workspace caps and descriptor settings torch's Lt path uses, and hold the chosen algo row plus a per-state workspace in state. Drop B3 (o m=128) and the o/unpooler m=256 cells of B4: the toggle profile shows Lt launches the identical nvjet kernel there. With those changes the per-cell gates make the run readable without a companion.

### Justification
A. Source A (bulk row-owner kernel, pooler m=1 only) is justified. Its paired 1.226x vs master's 1.135x on the largest decode_1 cell is a measured kernel-level win (17.0 vs 18.7 us), and gating it to m==1 matches where it won. Source B's qkv_gate m=128 evidence is real: under the Lt preference, profile 20 of 5ed9ace1 shows one nvjet_sm90_tst_48x64_64x15_4x2_h_bz_TNN at 16.2 us with no splitKreduce, against master's 96x64 splitK 12.7 + reduce 3.0 us. The mechanism is NOT 'the preference' as such. The algo-pick log (26a8be34) shows the Lt heuristic at a 32 MiB workspace returns, for qkv m=128, 8 candidates that are all splitK>=2. Its candidate 0 is bit-identical to torch's cuBLAS output (rel_vs_torch=0). So at matched inputs, cublasGemmEx and Lt pick the same algo. The 48x64 one-node kernel therefore comes from a different heuristic input in ATen's bgemm_internal_cublaslt: most likely a smaller Lt workspace cap that excludes the 2.36 MB split-K candidates, or ATen's alignment/descriptor attributes. An extension can reproduce that legally by querying at those caps in prepare(). The B2-B4 conjecture ('Lt drops the reduce node, so pooler NN m=128 goes 23.2 -> ~21') is weakly justified. The same toggle run already showed what Lt does at m=128 on TN. Pooler took a single-node 64x64 tile at 24.1 us (row 25.14), slower than master's NN splitK+reduce row of 23.2. Unpooler went 21.9 us, worse than master's NN 19.6. Dropping the reduce node by giving up split-K leaves a 4096x128 output at 128x64 tiles: 64 CTAs on 132 SMs, which is unlikely to beat split-K plus a 3 us reduce. o m=128 (128x64 TNT), o m=256 (128x128 TNT) and unpooler m=256 (128x128 TNT) ran the identical kernel under Lt and under cuBLAS, so those B3/B4 cells are null by construction. The expected directions in the hypothesis (pooler v16 faster, others unchanged/gate decides) are consistent with TASK weights. verify_16/verify_32 are 2 of 6 loads, and the realistic upside of B2-B4 is ~0-1 us on one cell, below the ~1.5% row noise, so it is readable only from the PATH CHOSEN lines and the profile. Weakest link: whether a rules-clean extension can actually reproduce the 48x64 single-node qkv algo. If the cause is an ATen descriptor attribute rather than the workspace cap, the heuristic ladder may never return it, and B collapses to master. A second weak link is the gate itself. In 26a8be34, a prepare-time graph-replay pick at a 0.97 threshold chose qkv m=256 cand 4 (26.27 vs 28.19 us), and that cell then LOST 0.962 in the judge. Prepare medians of 24 us vs judge rows of 16.8 us show how far that harness was from the judge's. This hypothesis uses the better _ouns_gate method (interleaved, empty-graph baseline subtracted), but at a looser 0.98 threshold.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 (landed, nn_p256_ok): B4's pooler m=256 incumbent is this NN path (nvjet 128x64 NNT 29.5 us). B4 compares NN-under-Lt against it, so a B4 pooler gain is relative to the landed NN path, not to SGLang's TN. A self-check MISS on that cell must switch off the Lt flag, not nn_p256_ok.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (landed, nn_ok for pooler/unpooler m=65..128): B2 replaces exactly the NN cuBLAS call this landed. master's _ab_selfcheck maps a pooler/unpooler m=128 MISS to nn_ok via _ab_component. If B2 routes there, a MISS caused by the Lt algo would switch off the proven NN path and fall back to TN cuBLAS, a confounded loss. _ab_component and _ab_flags must learn the new Lt and bulk flags.
- sglang-gemm-dispatch-parity (landed): defines the _bf16_gemm_dispatch_impl incumbent for qkv m=128 (source B). It is the control, not a confounder, provided the extension builds the same TN problem (a77cdec6 did: rel_vs_torch=0 for cand 0).
- bf16-restack-m16-swapab-router-swapab-m32 / restack-three-cell-wins (pooler m=1 swapab16 incumbent): source A is checked before the swapab16 branch for pooler m==1 only. _ab_component returns '' for pooler m=1, so an A/B self-check MISS on the bulk kernel prints 'nothing to switch off' and the bulk path stays live. Add a bulk flag to _ab_component and _ab_flags.
- Sibling queue item combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt is this hypothesis minus B2-B4. If the B2-B4 gates all choose master (the likely outcome), the two submits are behaviourally identical. Run only one of them.

### Interference and A/B needs
No companion is warranted. Every added cell is behind its own prepare-time gate. Each gate prints a PATH CHOSEN line with both medians and is backed by a torch.profiler kernel-name assertion, so each changed scored row maps to a logged choice. Rows where master was chosen are byte-identical in dispatch. A gain reads as the sum of the cells that chose their candidate. A loss on a chosen cell reads as 'gate timing did not transfer to the judge', as with qkv m=256 in 26a8be34, and that is the finding. Source A was measured alone on this cell. A source-B-only control duplicates the sibling combo. A revert of the landed NN paths would only change B2/B4's incumbent, and the gate already prints that incumbent's median. A companion would cost a scored run and would not change how either outcome is read.

### Open questions
- Does any heuristic query at a workspace cap below the 2.36 MB split-K requirement return the nvjet 48x64_64x15 no-splitK algo for qkv_gate m=128? Print the candidate table per cap. If none does, the cause is an ATen descriptor attribute: replicate bgemm_internal_cublaslt's descriptor (alignment prefs, transposes, ld) before concluding source B is unreachable.
- Should the gate threshold be 0.97 with the _ouns_gate harness, or stay at 0.98? 26a8be34's 0.97 pick on qkv m=256 lost 3.8% in the judge, and the B cells differ by only 1-3 us.
- Should B2 also require the profile to show no splitKreduce node? The conjecture is specifically about node count, so a B2 candidate that wins the gate while keeping split-K tells a different story.
- With B3 and the null B4 cells dropped (o m=128/256, unpooler m=256: identical kernels under Lt in 5ed9ace1), do the remaining probes (pooler/unpooler m=128 NN, pooler m=256 NN) justify a separate scored submit over the sibling combo, or should B2/B4 be folded into the sibling and this slug retired?

