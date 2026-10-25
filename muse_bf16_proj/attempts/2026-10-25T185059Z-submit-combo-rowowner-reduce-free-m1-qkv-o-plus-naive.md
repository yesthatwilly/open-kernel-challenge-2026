# muse_bf16_proj · 2026-10-25T185059Z · submit · combo-rowowner-reduce-free-m1-qkv-o-plus-naive

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T161307Z-combo-rowowner-reduce-free-m1-qkv-o-plus-naive.md` (branch `hyp/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive`).

- **Job:** `741072dca5fc4c9d` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `52c77fb2`

## Goal
Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS a conjectural widening of source A's bulk row-owner kernel to the other two m=1 weight streams that master runs with a split-K reduce or a high shape-minus-kernel gap, each behind its own gate: (A2) qkv_gate m == 1 (n 2304, k 8192): master runs _bf16_swapab_kernel via swapab16 with SPLIT_K 4 and a ticket reduce, kernel 9.6 us but row 12.67 us (a 3.1 us gap, vs 0.2 us on pooler d1). New instantiation bulk_kernel<16, 8192, 8, 0> (CKK 512, MT 1) with a grid of N/R = 144 CTAs; make the grid a launch argument (gridDim.x*R = N already in the epilogue) and reduce BULK_STAGES to 4 for this instantiation (template the stage count) so dyn smem ~ 256 + 4*(16384 + 8*1040) = 99 KB allows 2 CTAs/SM and all 144 CTAs are resident in one wave; change __launch_bounds__(160, 1) to (160, 2) for it. (A3) o m == 1 (n 8192, k 1024): bulk_kernel<64, 1024, 8, 0> (CKK 128, NCHUNK 8, MT 4), grid 128, 6 stages; master runs swapab16 o SPLIT_K 1 (kernel 3.9 us, row 5.97 us, 2.1 us gap). Repack qkv_gate and o with _bulk_repack(w, R) per state (37.7 MB + 16.8 MB, lossless permutation, round trip asserted bit-exact). Check the static_asserts: R*CKK*2 == 16384 holds for (16, 512) and (64, 128). Sources: SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45 (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch, bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack, _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only -> bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence: pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master 1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1 cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B: bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture. Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK 12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x), so it never applies there. Note for review: this toggle is a host-side library preference set and restored within the call (shape-gated, never value-gated); the judge's static check passed it once (5ed9ace10d9a4f68). Routing: {pooler, qkv_gate, o} m == 1 -> A/A2/A3 per cell gate, else master swapab16; qkv m==128 -> B; everything else master byte-identical (the m=1 qkv_gate TEST runs whichever path the gate chose, so A2 must pass the same self-check first). GATES (one per source per cell, all AFTER master's own _prepare_state work, master's code and constants untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked weights, error words and flags; no module-global device pointers). The second state of master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation sized by input values. Canary: run each new instantiation once on synthetic x with the bounded-wait error word before any graph capture, print LAUNCH/DONE lines around it (LESSONS 2026-10-25: a fault in prepare poisons the context; this kernel family already ran clean at R=32/64 in the judge). PREPARE BUDGET: bulk build with three MODE-0 instantiations ~45-55 s at import; repacks + canaries + three cell gates <= 4 s per state (deadline 200 s); Lt as combo 1 (deadline 250 s). Total prepare ~15 s plus ~125 s of import-time builds.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive`, file `muse_bf16_proj/submission.py` only (+~900 / -7, uncommitted). Base a2286ee3. Master 455705cd differs from it only in hypothesis md files, so no merge was needed and everything outside the change is master byte-identical.

**What changed.** The new section (9) sits just before `_prepare_state`.
- **CUDA module `muse_bf16_bulk_rowown_m1_v1`.** Ported from b3e80b45's `_BULK_CUDA`, MODE 0 only (the MODE 1/2 pure-read probes and the 3.8 TB/s kill gate are dropped).
  - `bulk_kernel<R, K, NT, S, MINB>`: the ring depth S and the `__launch_bounds__(160, MINB)` min-CTAs/SM are template parameters. The grid is `n / R`, set at launch from `out.size(1)`. The epilogue keeps `N = gridDim.x * R`.
  - A `BulkInst<...>` struct derives its SMEM from the same template S as the kernel, and `bulk_setup` prints that S. This answers the review's open question on S consistency.
  - Three instances:
    - A, pooler: `<32, 8192, 8, 6, 1>`, grid 128, 123,904 B.
    - A2, qkv_gate: `<16, 8192, 8, 4, 2>`, grid 144, 99,072 B.
    - A3, o: `<64, 1024, 8, 6, 1>`, grid 128, 111,616 B.
  - Static asserts hold for all three. A2: SLOT 24704, scratch 2 KB. A3: SLOT 18560, scratch 8 KB.
  - The source's `static bool attr_set` is gone. `bulk_setup(inst)`, called in prepare() before any launch, sets MaxDynamicSharedMemorySize and **PreferredSharedMemoryCarveout = cudaSharedmemCarveoutMaxShared**. It returns regs, local, static/dyn smem and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.
  - The build happens at import with `TORCH_CUDA_ARCH_LIST=9.0a`, set and then restored, and flags `-O3 -lineinfo -std=c++20`. The arch flags and the build seconds are printed.
- **cuBLASLt module `muse_bf16_lt_qkv128_v1`.** Ported from a77cdec6's `_LT_CPP`. The static descriptor map is replaced by a per-call RAII `MuseLtDesc` (host-only; under graph replay none of it runs), so there is no static or global C++ state. The module is built at import, and its seconds are printed.
- **Python helpers.** `_bulk_repack` (lossless, round trip asserted bit-exact), `_bulk_route_ok`, `_bulk_forward` and `_lt_route_ok`/`_lt_forward`.
- **Shared gate helpers.**
  - `_cell_check`: canary plus correctness, with flushed LAUNCH/DONE prints, a sync and the error word per launch. It checks rel L2 vs fp64 <= 3e-3, a probe row rel <= 1e-4, finiteness, and that run 2 is bit-identical.
  - `_cell_gate`: `_ouns_gate`'s model selector run through `kernel_fn` itself on shallow state copies with the cell flag on vs off. It uses graph replay, a 256 MB flush, 41 interleaved reps and an empty-graph baseline, and prints both medians.
  - `_cell_kernels`: one torch.profiler call that returns the CUDA kernel names.
- **`_bulk_prepare` / `_lt_prepare`.** These run after master's own work in `_prepare_state` (flags None). Each is timed with a `[prepare-time]` line. The per-cell total and the gate seconds are printed. Each cell prints exactly one `PATH CHOSEN <A|A2|A3|B> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)` line, with the reason when it is master.
  - The second state of the A/B check (flags given) inherits the flags and the Lt algo row. It builds its own repacked weights, error word and 1 MiB Lt workspace.
- **`kernel_fn`.** After the ouns line, `_bulk_route_ok` sends pooler/qkv_gate/o at cfg m == 1 to `_bulk_forward` when that cell's flag is set. In the bf16 branch, before master's `_bf16_gemm_dispatch_impl`, qkv_gate m == 128 goes to `_lt_forward` when `lt_qkv128_ok` is set. There are no prints, no state writes and no try/except.
- **A/B check.** `AB_CASES` gains qkv_gate m=1, o m=1 and qkv_gate m=128; pooler m=1 was already there.

**Review fixes applied:**
1. **No per-call `preferred_blas_library` toggle.** Source B's toggle was NOT ported. qkv m=128 uses a prepare-time cuBLASLt plan in state (`lt_algo_qkv128` CPU int64[8] plus a per-state `lt_ws_qkv128`).
   - The heuristic is queried at workspace caps 0 and 1 MiB, and only SPLITK_NUM == 1 candidates within 1 MiB are kept, deduplicated.
   - Each candidate is canaried and checked eagerly, then ranked by graph replay (15 reps, cold L2). The best one is gated against master.
   - When routed, torch.profiler must show exactly ONE kernel whose name contains no "splitk" or "reduce". Otherwise, if no split-K-1 candidate exists, the gate fails or the profiler is unavailable, the cell is dropped and the reason goes on its PATH CHOSEN line.
2. **The `state['bulk_route_hit']` write is removed.** `_bulk_forward` writes nothing to state. The source's route probe that read it is replaced by the profiler kernel-name assertion.
3. **A2 occupancy.** A2 sets the max-shared carveout. `bulk_setup` returns the occupancy query, and the cell is disabled (never launched) if it is < 2, i.e. 1. The value is printed.
4. **Canary before capture.** Every new kernel instance and every Lt candidate is first launched eagerly in `_cell_check`, before any graph capture. The A2 instance never launches at all if its occupancy check fails.
5. **Per-cell flags in the A/B check.** `_ab_flags` carries `bulk_pooler_ok`, `bulk_qkv_ok`, `bulk_o_ok` and `lt_qkv128_ok`. `_ab_component` maps (pooler|qkv_gate|o, m=1) and (qkv_gate, m=128) to their own flags, checked before the NN/router/ouns rules. A miss switches off only that cell, never nn_ok or another cell. The second-state-build-failure path also turns the new flags off.
6. **Confirm/refute criteria restated on kernel time** in the Risk/Conjecture paragraph above:
   - Confirmed: A2 routes AND bulk_kernel<16,...> < 9.6 us AND row < ~12.3 us.
   - Refuted: A2 keeps master, or the kernel time is >= 9.6 us.
   - The "gap below 1 us" test is dropped, as is the "A3 wins means TMA issue" reading.
7. **Timing logs and PATH CHOSEN lines.**
   - Build seconds: `[build-time] bulk ...` and `[build-time] lt ...` (ouns and router_cluster already print theirs).
   - Gate seconds: `[prepare-time] gate <src> ...`, plus the per-cell total, plus the ouns gate tick that already exists.
   - Exactly one PATH CHOSEN line per cell, on every exit path.
   - The review's open question on the 144-on-132 imbalance is also handled: every bulk gate prints the projected `bytes / 3.95 TB/s` time (qkv: 37.7 MB -> 9.55 us) next to the candidate median.

**Deviations from the spec:**
- **Source B is not the toggle.** The review overrides the spec: B uses a cuBLASLt split-K-1 plan picked in prepare. It may not reproduce the toggle's nvjet 48x64 kernel. The gate decides, and a miss keeps master.
- **Bulk-cell profiler failures keep the route.** If torch.profiler raises or returns no CUDA events for a bulk cell, the route is kept, with a printed note: kernel_fn calls our own extension directly, so the launched kernel is not in doubt. A *wrong* kernel name still disables the cell. For B a profiler failure drops the cell, as the review requires.
- **Gate rule.** The gate is cand/master <= 0.98 on net medians, as the spec says (equivalent to master/cand >= 1.0204).
- **Lt workspace.** It is 1 MiB per state, where a77cdec6 used 32 MiB, because split-K-1 algos need none.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive diff` (submission.py only):

- [x] **CUDA-graph capturable.**
  - kernel_fn's new branches call `_bulk_route_ok`/`_lt_route_ok`, which read cfg, shape, dtype, contiguity, data_ptr alignment and prepare()-set state flags.
  - `_bulk_forward` does `torch.empty` plus one `bulk_run` launch. `_lt_forward` does `torch.empty` plus one `cublasLtMatmul` with a prepare()-allocated workspace.
  - There is no `.item()`, sync, print or try/except in kernel_fn's call graph; every lint host-sync hit is outside it.
  - Branching uses cfg and shapes only.
- [x] **Nothing derived from `data` is cached.** The only state writes are in prepare(): flags, repacked weights, error word, algo row and workspace, all from weights and synthetic x. kernel_fn and its callees write nothing to state; the `bulk_route_hit` write was removed.
- [x] **Weights stay BF16, repacked losslessly in prepare() only.** `_bulk_repack` is a pure permutation of bf16 values with a bit-exact round trip asserted, done per state. Lt uses the weight as stored. The router path is untouched.
- [x] **Activations BF16.** bf16 x in and bf16 out, with fp32 accumulation (mma.sync f32 / CUBLAS_COMPUTE_32F). Lt keeps only split-K-1 algos, so there is no reduced-precision reduction. KV cache / queries: n/a (no attention in this problem).
- [x] n/a: there is no attention in muse_bf16_proj, and every weight row and K element is read.
- [x] **No forbidden operations.** No timing patches, harness frames, I/O or secrets, module edits, reference import, subprocesses or network (lint PROCESS/NETWORK/HARNESS checks clean). `load_inline` builds at import as master's ouns/router_cluster do. torch.profiler, CUDAGraph and Event are only *used* in prepare(), never patched.
- [x] **All work happens inside the timed call.** Every output is computed fresh. Only the per-cell route decision and the Lt algo choice, made in untimed prepare() from synthetic data, are reused.
- [x] **Multi-state safety.** First line is still `# okc: problem=muse_bf16_proj`; there is no triton.autotune. There are no device pointers, workspaces or descriptors in module or C++ globals or statics: the source's `static bool attr_set` and the Lt static descriptor map were both removed. Each judge copy has its own repacked weights, error word and Lt workspace. The A/B check captures a graph on two states and alternates replays with refilled x on the new cells too.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_bf16_proj/submission.py --problem muse_bf16_proj`), rc 0:

```
26 x warn host-sync ... (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1483: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2655: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2825: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2836: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2852: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
OK with warnings
```

The uint8 hits are L2-flush and cuBLASLt workspace byte buffers, not activations. `python3 -I -m py_compile` passed, and `__pycache__` was removed. A symtable scan found no undefined global names in the file (LESSONS 2026-10-25 f-string check). No CPU dry run was possible: there is no torch on this node.

## Risks / what could fail
medium
Expected: combo 1's ~+0.8% plus up to +0.7% from qkv decode_1 (-2 us of a 47.5 us load). Risk medium: new template instantiations of a kernel that ran clean, behind a canary and a hard gate. ## Conjecture: the row-owner kernel won pooler m=1 not by bandwidth (its cold pure read was only 2.54 TB/s in prepare and ~3.95 TB/s effective in the judge, about master's swap-AB rate on qkv) but by being reduce-free: one CTA owns whole output rows, so there is no split-K ticket, no fp32 workspace round trip and no tail CTA doing the reduce. If so, the gain carries to the cell with the deepest split (qkv_gate, SPLIT_K 4, the largest row-minus-kernel gap at 3.1 us) and NOT to o (SPLIT_K 1, 0.0 reduce, K only 1024, where an 8-chunk ring cannot amortise its fill). A second interaction is occupancy: A2 needs 2 CTAs/SM at 4 stages, so it also tests whether the row-owner stream survives half the ring depth. Confirmed (criteria restated per the adversarial review, on kernel time) if A2's gate routes (PATH CHOSEN A2 qkv_gate m=1: candidate, cand/master <= 0.98) AND the profile's bulk_kernel<16, 8192, 8, 4, 2> time is < 9.6 us AND the qkv_gate decode_1 row is < ~12.3 us. Refuted if A2's gate keeps master, or the bulk kernel time is >= 9.6 us. (The 'row-minus-kernel gap below ~1 us' test is dropped: master's pooler (SPLIT_K 2, gap 0.2 us) and o (SPLIT_K 1, gap 2.1 us) show the gap does not track the reduce. An A3 win would only mean a faster stream, since o has no reduce to remove.)

## Result (after the run)
- job `741072dca5fc4c9d` (submit), status done, passed True, tests 9/9
- speedup 1.2566x, 95% CI [1.2378, 1.2755], score 3844.2020335760453 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2490645809967658, "ci": [1.2303286122818142, 1.267800549711717], "job": "ee4f89f2f62c42d3"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-nn-layout-widen-m256.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json`, `muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.patch`; branch `hyp/muse_bf16_proj/combo-rowowner-reduce-free-m1-qkv-o-plus-naive`
