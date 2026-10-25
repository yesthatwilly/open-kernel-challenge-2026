# muse_attention_decode · 2026-10-25T192000Z · submit · merge-perkvhead-c320-sv32-tiled

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T190812Z-merge-perkvhead-c320-sv32-tiled.md` (branch `hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled`).

- **Job:** `a78f29aeab9b47cb` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `12aaa6b9`

## Goal
Hand-merge two gated pieces of combo-tiled-preattn-neighbour-regate-c320 (branch hyp/muse_attention_decode/combo-tiled-preattn-neighbour-regate-c320, impl 5add0d92, scored 1.5199 in job 52cbfaebab9646f2) onto current attention master 5d528bf3 (sliding-decode1-cluster-per-kvhead, 1.5120). That branch was built on master 7e128915, before per-kvhead landed; a 3-way merge (base 7e128915) has 8 conflicts. Port exactly: (1) the chunk320 full decode_1 path: the separate muse_full_decode_c320 module, its sequential main-thread build phase (flags -O3 -arch=sm_90a), GATE A correctness and timing, and the route flag that replaces master's two-level split-KV at full decode_1 only when GATE A passes (ratio <= 0.98); (2) tiled pre-attention at SV32 only (sliding verify_32), with its GATE B exactness and timing gate. Leave every other tiled class, the neighbour regate of sliding decode B=2..32, and the dropped skF regate out; those classes keep master's per-kvhead and sk-window routing unchanged. Per-kvhead's sliding decode_1 cluster path stays as on master. Every ported phase runs after master's own setup, with its own deadline and the 480 s import start rule, and falls back to master on failure or overrun with one PATH CHOSEN line per piece.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
User-requested (2026-10-25 ~19:05Z: 'do the attention hand-merge and submit it'). Runs as a scored submit; checks are spent. Expected to stack per-kvhead (1.5120) with regate's two live wins (1.5199 without per-kvhead). ## Conjecture: per-kvhead's sliding decode_1 clusters and c320's full decode_1 kernel never ran in the same layer graph; they touch different layers, so L2 working sets should not interact, and the gain should be additive. Read the PATH CHOSEN lines for full decode_1 [A chunk320], SV32 tiled and sliding decode_1 per-kvhead.

## Result (after the run)
- job `a78f29aeab9b47cb` (submit), status done, passed True, tests 16/16
- speedup 1.5347x, 95% CI [1.5117, 1.5577], score 2454.0938154955193 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/merge-perkvhead-c320-sv32-tiled.result.json`, `muse_attention_decode/hypotheses/done/neutral/merge-perkvhead-c320-sv32-tiled.patch`; branch `hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled`
