---
problem: muse_attention_decode
slug: merge-perkvhead-c320-sv32-tiled
created: 2026-10-25T190812Z
base: master
area: full decode_1 split-KV + sliding verify_32 pre-attention
change: >
  Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch
  hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0d92, scored
  1.5199 in job 52cbfaebab9646f2) onto current attention master 5d528bf3
  (sliding-decode1-cluster-per-kvhead, 1.5120). That branch was built on master 7e128915, before
  per-kvhead landed; a 3-way merge (base 7e128915) has 8 conflicts. Port exactly: (1) the chunk320
  full decode_1 path: the separate muse_full_decode_c320 module, its sequential main-thread build
  phase (flags -O3 -arch=sm_90a), GATE A correctness and timing, and the route flag that replaces
  master's two-level split-KV at full decode_1 only when GATE A passes (ratio <= 0.98); (2) tiled
  pre-attention at SV32 only (sliding verify_32), with its GATE B exactness and timing gate. Leave
  every other tiled class, the neighbour regate of sliding decode B=2..32, and the dropped skF
  regate out; those classes keep master's per-kvhead and sk-window routing unchanged. Per-kvhead's
  sliding decode_1 cluster path stays as on master. Every ported phase runs after master's own
  setup, with its own deadline and the 480 s import start rule, and falls back to master on
  failure or overrun with one PATH CHOSEN line per piece.
expected:
  - shapes: "*layer=full*load=decode_1*"
    direction: faster
    why: c320 measured 0.894-0.892 of master two-level in b9d062aa and 52cbfaeb
  - shapes: "*layer=sliding*load=verify_32*"
    direction: faster
    why: tiled pre-attention at SV32 0.964 in b9d062aa; regate gained sliding verify_32
  - shapes: "*layer=sliding*load=decode_1*"
    direction: same
    why: "keeps per-kvhead's +9.9% that regate's branch lacked"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T191118Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T191118Z
base_sha: 8eca21f5b391cbe0335e7b32676b0f35d9c5ff9e
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled
worktree: .worktrees/muse_attention_decode/merge-perkvhead-c320-sv32-tiled
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T192000Z
impl_commit: 12aaa6b98d05f59638d45c4d0af7e7120394d82b
submission_sha256: fda9e34878124d04a923f6208991ed54de5e6408f6db77365e2ebb79b5067958
norm_sha256: 46c431d8045ec39e8735bd5f4e362ba4e3bf1ee016aaafbc820eef67df25aca7
submission_blob: 6d947fb9ae48e4542cacddb80b8c7506fcc3e42a
lint_warnings: 69
attempt_file: 2026-10-25T192000Z-submit-merge-perkvhead-c320-sv32-tiled.md
prepared_at: 2026-10-25T192000Z
---

# merge-perkvhead-c320-sv32-tiled

## Change

Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0d92, scored 1.5199 in job 52cbfaebab9646f2) onto current attention master 5d528bf3 (sliding-decode1-cluster-per-kvhead, 1.5120). That branch was built on master 7e128915, before per-kvhead landed; a 3-way merge (base 7e128915) has 8 conflicts. Port exactly: (1) the chunk320 full decode_1 path: the separate muse_full_decode_c320 module, its sequential main-thread build phase (flags -O3 -arch=sm_90a), GATE A correctness and timing, and the route flag that replaces master's two-level split-KV at full decode_1 only when GATE A passes (ratio <= 0.98); (2) tiled pre-attention at SV32 only (sliding verify_32), with its GATE B exactness and timing gate. Leave every other tiled class, the neighbour regate of sliding decode B=2..32, and the dropped skF regate out; those classes keep master's per-kvhead and sk-window routing unchanged. Per-kvhead's sliding decode_1 cluster path stays as on master. Every ported phase runs after master's own setup, with its own deadline and the 480 s import start rule, and falls back to master on failure or overrun with one PATH CHOSEN line per piece.

## Expected

- `*layer=full*load=decode_1*` faster: c320 measured 0.894-0.892 of master two-level in b9d062aa and 52cbfaeb
- `*layer=sliding*load=verify_32*` faster: tiled pre-attention at SV32 0.964 in b9d062aa; regate gained sliding verify_32
- `*layer=sliding*load=decode_1*` same: keeps per-kvhead's +9.9% that regate's branch lacked

## Risk

User-requested (2026-10-25 ~19:05Z: 'do the attention hand-merge and submit it'). Runs as a scored submit; checks are spent. Expected to stack per-kvhead (1.5120) with regate's two live wins (1.5199 without per-kvhead). ## Conjecture: per-kvhead's sliding decode_1 clusters and c320's full decode_1 kernel never ran in the same layer graph; they touch different layers, so L2 working sets should not interact, and the gain should be additive. Read the PATH CHOSEN lines for full decode_1 [A chunk320], SV32 tiled and sliding decode_1 per-kvhead.

## Adversarial review

Reviewed 2026-10-25T191118Z by an adversarial subagent (model opus).

**Verdict: proceed.** Justified, safe merge of two independently reproduced row wins (c320 at full decode_1, about -3.7 us, and tiled pre-attention at SV32, about -1.0 to -1.4 us) onto master's per-kvhead (sliding decode_1 10.87 us). Prediction is about 1.53x (range 1.51-1.55), roughly 1% over the 1.5199 board entry, which is inside the CI. The per-shape rows and PATH CHOSEN lines show attribution directly, so no companion is needed. The one real hazard is a merge slip that silently brings back pre-per-kvhead sliding routing.

### Justification
A. The mechanism is measured, not inferred. Full decode_1, weight 16: master per-kvhead (4b608df5) is 22.70 us per shape (sk two-level 19.1 us plus pre-attention 1.4 us). c320 was 19.00 us in 52cbfaeb and 19.76 us in b9d062aa (splitkv_attn_kernel<320> 10.9 us, combine 4.4 us, pre-attention 1.3 us), and GATE A ratios of 0.894 and 0.892 reproduce across two builds. Per-kvhead touched only sliding code (sliding_cluster_kernel PERHEAD, _slide_*, and the _sd_setup_body route.pop(1)), and GATE A compares against the same two-level kernel, so the c320 ratio should carry over. SV32, weight 46: master is 28.83 us with untiled pre-attention at 4.2 us. Tiled runs measured 27.42 and 27.92 us, with _pre_attention_tiled_kernel at 2.3-2.6 us. Sliding decode_1, weight 46: per-kvhead is 10.87 us against 12.04 and 12.62 us in the two branches without it, and the merge keeps it. Arithmetic: the decode_1 load goes from 863.2 to about 804 weighted us (+7.4%), and the verify_32 load from 6167 to about 6102 (+1.06%). The geomean over 6 loads gives 1.5120 x 1.0120 x 1.0018, about 1.533x. The expected shapes and directions match the weights. Full decode_1 is still 4.4x its 4.3 us floor, so there is room left. The conjecture's L2 argument is moot rather than wrong: every shape is its own captured graph, timed from cold L2, so the two layers never share a graph. Weakest link: the SV32 piece is worth about 0.18% of headline, too small to show in the headline and only readable in its row (expect about 27.4-27.9 us against 28.83). The headline gain also rests almost entirely on c320, which is about 1.2%, so a run of 1.51-1.52 would be ambiguous at headline level without the rows. The second weak link is merge fidelity: 8 conflicts in setup and route orchestration.

### Confounders on the current kernel
- sliding-decode1-cluster-per-kvhead (master 5d528bf3): disjoint shapes, but the same import orchestration. Its _sd_setup_body does route.pop(1, None) before freezing _SD_ROUTE, and _slide_init sets _SLIDE_PH. The regate branch snapshots and republishes _SD_ROUTE (_NB_SNAP, MappingProxyType(new_sd)). If that publish or restore is ported, it can overwrite or restore a table built without per-kvhead's changes. Since the neighbour regate is dropped, the port must not touch _SD_ROUTE at all. A wrong merge would show as sliding decode_1 at about 12 us instead of about 10.9 us, or as a missing 'PERHEAD ON' line.
- full-decode1-streamk-s3-two-level-ticket (two-level sk kernel at FD1): this is the incumbent c320 replaces and the GATE A comparator. It is unchanged by per-kvhead, so the 0.89 ratio is not confounded. If GATE A falls back, the row stays at about 22.7 us.
- fused-preattention-triton / paired-heads-launch-cleanup (master _pre_attention_kernel): the tiled kernel replaces this first node only at SV32 (unpaired, sliding). GATE B compares [tiled + rest] against [master + rest] on the merged master, so the gain is self-attributed.
- restack-sliding-decode-sk-window-on-cluster-master (sk window at sliding decode B=17..32): regate's GATE B also probed SD32 tiled and kept master. Leaving every non-SV32 tiled class out keeps this untouched. Any SD32 movement would be noise or a merge error.

### Interference and A/B needs
No companion is warranted. Both ported pieces are shape-gated to rows that per-kvhead does not touch (FD1, SV32), and per-kvhead's only row (SD1) is kept unchanged. Each piece also prints its own PATH CHOSEN line with an in-run A/B ratio, and the profile shows which kernels ran per row. A gain or a loss can therefore be read row by row against three existing references: master 4b608df5 (FD1 22.70, SV32 28.83, SD1 10.87), regate 52cbfaeb (19.00 / 27.42 / 12.04) and naive b9d062aa (19.76 / 27.92 / 12.62). An ab-revert of per-kvhead would only reproduce 52cbfaeb. A control for c320 alone or SV32 alone would cost a scored run to learn what the row already shows.

### Open questions
- Builder checklist for the 8-conflict merge: (1) do not port the _SD_ROUTE publish or snapshot entries, or the neighbour-regate phases P4 and P6. (2) Confirm the log shows '[slide] PATH CHOSEN: sliding B=1 PERHEAD ON', 'PATH CHOSEN: full decode_1 [A chunk320]: route (... ratio <= 0.98)', and 'GATE B SV32: tiled first node', with every other GATE B class on master. (3) Run okc-style local lint, and if possible a local import, to confirm the 480 s start rule and fallback paths.
- Readout rule for the scored run: success means FD1 at about 19-19.8 us, SV32 at about 27.4-27.9 us, and SD1 at about 10.9 us. If the headline lands inside [1.497, 1.543] but those rows hold, the stacking conjecture is confirmed even though the headline is not a statistically proven best. If SD1 is about 12 us, the merge reverted per-kvhead routing; that is a build error, not a result.
- Bookkeeping: BEST.json lists score_us 2490.98 and CI [1.4893, 1.5347], which are master 4b608df5's numbers, against job 52cbfaeb, whose own values are 2477.92 and [1.4971, 1.5427]. Fix this before comparing the merge to the board.
- No scored test covers full decode at B=1 (the tests cover B=4 full decode and B=1/2 full verify). The c320 path is exercised only by GATE A and by the scored shape's per-call reference check. Both of those passed twice, but a c320 regression would surface only at the scored shape.


## Implementation notes

Built 2026-10-25 by a subagent. The file is at muse_attention_decode/submission.py in worktree .worktrees/muse_attention_decode/merge-perkvhead-c320-sv32-tiled, sha256 fda9e348...7958. Nothing is committed.

- **Method.** The build started from master 5d528bf3 and applied `git merge-file` (current = master, base = 7e128915, other = naive 20bc3d00). The 3 conflicts were adjacency only (master's `_SLIDE_PH = False` and the end of `_sd_setup`); both sides were kept. A Python script then narrowed the ported code to SV32. Against master: 6 hunks, +1108/-7.
- **Source per piece.** Both pieces come from naive 20bc3d00, whose base submission.py is byte-identical to 7e128915. Naive adds 6 small hunks and never touches master's sliding tables. Regate (5add0d92) snapshots and republishes `_SD_ROUTE` and threads tiled flags through `_sd_run`/`_sd_case`, which is the hazard the review named, so it was not used. Its c320 CUDA is the same kernel as naive's; only the init entry points differ. Its build uses the same flags and is also sequential.
- **Hunks.**
  1. `_IMPORT_T0` after `import time`. This is the import clock both pieces use.
  2. [B] `_pre_attention_tiled_kernel`, verbatim.
  3. [A] In kernel_fn: `if _C320_ROUTE and _c320_gate(cfg): return _full_decode_c320(data)`. It sits after the sliding-decode block and before master's full path.
  4. [B] The pre-attention launch is wrapped in `if _TILED_ROUTE and _tiled_class(cfg) in _TILED_ROUTE: tiled else: <master's launch, unchanged>`.
  5. The merge block contains:
     - [A] the `muse_full_decode_c320` sources, `_build_c320`, `_c320_ready`, `_c320_gate`, `_c320_launch`, `_full_decode_c320` and GATE A (`_c320_case`, `_c320_gate_a`);
     - [B] `_tiled_class`, which returns "SV32" only for q_len 8, batch 17..32, window and RoPE; `_tiled_pre`, `_tiled_exact`, `_tiled_time`, `_tiled_replay` and `_tiled_cases` (SV32 cases only); and `_tiled_gate_b` over `_TILED_CLASSES = ("SV32",)`;
     - shared: `_cx_clock`, `_cx_may_start`, `_cx_interleaved`, and the phase functions `_cx_phase_a` / `_cx_phase_b`.
  6. Module tail, after `_sd_setup()`: `_cx_phase_a()` and then `_cx_phase_b()`, each in its own try/except.
- **Left out.** The FV1/FV16/FV32/SV1/SV16 tiled classes and the verify-full reference/synthetic helpers. Also every regate piece: the neighbour regate, the skF regate, and the `_NB_*` snapshot/restore.
- **Build.** On the main thread, sequential, after every master build including PERHEAD (`_slide_init`) and `_sd_setup`. `extra_cuda_cflags=['-O3', '-arch=sm_90a']`. TORCH_CUDA_ARCH_LIST is not touched. No thread.
- **Deadlines.** A phase starts only if the import clock plus its deadline is at most 480 s. A phase that finishes past its deadline routes nothing, and an exception falls back for its own piece only. The deadlines are: c320 build 120 s, GATE A 60 s, GATE B (SV32) 40 s.
- **Routes.** `_C320_ROUTE` is a bool set once in a finally block. `_TILED_ROUTE` is `MappingProxyType`, empty or {"SV32": True}. Tickets are re-zeroed after each piece.
- **Log lines.** `PATH CHOSEN: full decode_1 [A chunk320]: ...` and `PATH CHOSEN: sliding verify_32 pre-attention SV32 [B tiled]: ...`. Master's `[slide] PATH CHOSEN: sliding B=1 PERHEAD ON ...` line is unchanged. Note that the wording differs from regate's `GATE B SV32: tiled first node`.

## Rules check

- [x] CUDA-graph capturable: the new kernel_fn paths have no `.item()`/`.cpu()`/`synchronize()`/prints; routing is config-only on frozen import-time values; c320 workspaces are `torch.empty` sized from shapes (verified by builder and Opus audit).
- [x] Nothing derived from `data` is cached across calls (route tables are set at import from synthetic gate inputs only).
- [x] Weights: no weight handling changed.
- [x] Precision: KV stays FP8, queries E4M3, BF16 output, same as master; GATE B holds tiled pre-attention to within 1 ulp on at most 1% of bytes.
- [x] Attention sees every key: c320 is full split-KV over the whole context; the tiled kernel is pre-attention only.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network (the c320 build is torch load_inline at import, as in master).
- [x] All work happens inside the timed call; no output caching or replay.

Builder notes:
- kernel_fn with both routes off makes master's launches, in master's order, with master's arguments. The added code is two config-only predicates on frozen import-time values; the else branch is master's launch byte-for-byte, re-indented.
- The new kernel_fn paths have no host sync, print, try/except, per-call mutable state, data-dependent control flow or input-value-sized allocation. The c320 workspaces are `torch.empty` sized from `page_table.shape[1]` and the batch.
- No module-global per-state device pointers. The c320 module keeps only a host bool (`g_c320_ready`); its workspaces are allocated per call. GATE A and GATE B each run a two-state alternating capture-once replay with refilled inputs.
- Master's CUDA modules and sliding tables (`_SD_ROUTE`, `_SLIDE_*`, `_PS_ROUTE`) are not written by the new code.
- Precision is unchanged. The tiled kernel has the same arithmetic as master's pre-attention and GATE B checks it to within 1 ulp on at most 1% of bytes. c320 uses FP8 KV with e4m3 queries, as master does.
- `okc_lint.py --problem muse_attention_decode`: OK with warnings. All warnings are host-sync calls outside kernel_fn's call graph, plus one dtype warning on GATE B's uint8 poison buffer, which kernel_fn never uses. py_compile passed, and a symtable scan found 0 undefined names (its positive control was detected).
- Not run: no GPU or nvcc is available on this node, so the import, the build and both gates are untested locally.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the code (no GPU on this node). **Verdict: dispatch.** No fixes were needed.

- **Routes off.** With both routes off, kernel_fn matches master. The c320 check short-circuits on `_C320_ROUTE` (False), and an empty `_TILED_ROUTE` is falsy. The else branch is master's `_pre_attention_kernel` launch, byte-identical apart from indentation. The merge block's globals are defined at about line 5941, before the module tail at line 6938, so master's own setup calls to kernel_fn (`_splitkv_self_check`, `_slide_init`, `_sd_setup`) see them already set. An AST scan finds no other module-level calls.
- **Fidelity.** The merge block matches naive 20bc3d00 line for line, apart from: the SV32-only narrowing of `_tiled_class`, `_TILED_CLASSES` and `_tiled_cases`; `_ps_synthetic`/`_ps_reference` in place of the dropped `_cx_verify_*` helpers; a GATE B deadline of 40 s instead of 90 s; and `_cx_setup` split into `_cx_phase_a`/`_cx_phase_b`, each with its own try/finally and PATH CHOSEN line. The master regions around `_SLIDE_PH = False` and the end of `_sd_setup`/`_sd_setup_body` are byte-identical to master. That includes `route.pop(1, None)` and the `_SD_ROUTE` freeze. Nothing in the new code writes `_SD_ROUTE`, `_SLIDE_*`, `_PS_ROUTE`, `_FUSED_ROUTE`, `_B1_PATH` or `_EXT*`.
- **Helper signatures.** On master: `_ps_synthetic(lens, kv_scale, gen, device, max_pages=None, L=8)` and `_ps_reference(data, lens)`. `_splitkv_synthetic`, `_splitkv_reference`, `_sk_prepare_q`, `_alt_sets`, `_alt_check(tag, synth, ref_fn, sets, fn=None, gate=None)`, `_graph_of` and `_rezero_tickets` all match how GATE A and GATE B call them.
- **Gates.** `_c320_gate` is `_shape_gate` (full, q_len 1, tp8, d 64, page 1) with batch == 1, so it covers FD1 only. `_tiled_class` requires q_len 8, tp8, d 64, page 1, batch 17..32, window 2048 and RoPE, so it covers SV32 only, and none of the 16 tests fall in that class. Neither gate can always fail: naive b9d062aa ran this exact gate code and routed both pieces (FD1 ratio 0.892, SV32 ratio 0.964).
- **Build and deadlines.** The c320 build runs on the main thread, sequentially, after `_sd_setup`, with flags ['-O3', '-arch=sm_90a']; no environment variable is touched. A phase starts only if the import clock plus its deadline is at most 480 s. A build that fails, a gate that fails, an exception or a phase that finishes past its deadline each sends that piece to master, sets its route in a finally block, re-zeros the tickets and prints one PATH CHOSEN line. Naive finished its import at 226 s and its whole run at 530 s, and its log reports "GATE B took 1.1s" for six classes, so the 40 s SV32 deadline has wide margin. Per-kvhead adds the PERHEAD build, so expect roughly 560 to 600 s, well under the 900 s limit.
- **Rules.** kernel_fn adds only config predicates on frozen import-time values. The c320 workspaces and q are `torch.empty` sized from shapes (`page_table.shape[1]`, batch). The c320 module holds only a host bool, with no per-state device pointers. Precision is unchanged: FP8 KV, e4m3 queries, BF16 output. Lint: OK with warnings (all host syncs at import, plus GATE B's uint8 poison buffer). py_compile: OK.

Residual risk: c320 has no test coverage at FD1 (B=1 full decode page 1), so a regression would surface only at the scored shape's per-call check; the same code passed there in b9d062aa and 52cbfaeb.
