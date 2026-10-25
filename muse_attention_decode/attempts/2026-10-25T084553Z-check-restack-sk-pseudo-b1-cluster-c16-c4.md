# muse_attention_decode · 2026-10-25T084553Z · check · restack-sk-pseudo-b1-cluster-c16-c4

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T054448Z-restack-sk-pseudo-b1-cluster-c16-c4.md` (branch `hyp/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4`).

- **Job:** `eb6ff0cb701d4def` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `00223bb6`

## Goal
Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A) From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job 94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route. (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job 195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS 2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit (B) for module/static globals holding device pointers or state-sized buffers and move any into per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills; extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Base: worktree base edf4bf2 = current master for muse_attention_decode/submission.py (blob d8769ef, unchanged since); no stale_base merge needed. Built in `muse_attention_decode/submission.py` only (+1722/-40).

What changed
- (A) Took the whole s1 PSEUDO change (impl 6f95ce7, blob-identical base) as is: `splitkv_attn_sk_kernel<..., PSEUDO>`, `Lay<STAGES,256,true>`, `load_prefix_pseudo`, `_ps_setup` gate (>= 1.05x vs FA3, unchanged), `_ps_launch`, `-DSK_PSEUDO=1` build with landed-only rebuild fallback.
- (B) Ported only the sliding cluster TU from 41fcca0 (`_SLIDE_CPP`/`_SLIDE_CUDA`, byte-identical CUDA text, fp8qk with f16qk rebuild fallback) plus its Python glue, nothing of that branch's stream-K. Own constants `_SLIDE_*` (no shared constant with the PSEUDO/stream-K path). C=8 dropped from the Python tables (never routed).
- `kernel_fn`: `if window and q_len == 1: if _slide_gate(cfg): c = _slide_clusters(cfg); return _slide_launch(_SLIDE_MOD[c], data, c); return _sliding_forward(data)`. Pure config-gated dispatch (batch 1 -> C=16, 17..32 -> C=4, else master Triton; verify untouched).

Review fixes / asks applied
1. Strip kernel_fn try/except, `_slide_note` prints and `_SLIDE_NOTED` set (and `_slide_forward`): done, none remain (grep clean).
2. (B) margin: kept config-only (C=16 at B=1, C=4 at B=17..32), no margin gate; `_slide_timing` prints the interleaved cold-L2 (256 MB flush) graph-replay time of cluster vs master `_sliding_forward` with an empty-graph baseline subtracted (round-robin, order flipped each round, median of 20) for each enabled C, information only. PATH CHOSEN lines carry both numbers.
3. Two-state rule for (B)'s self-check: `_slide_two_state_check` builds state A and state B (different lens, cache_loc, page tables, caches, kv scale 1.0 vs 0.25, same shapes), captures the launch ONCE on a third buffer set after a side-stream warm-up, then alternates refill A / replay / compare, refill B / replay / compare, twice, output poisoned with NaN before each replay, each vs the fp32 reference (rel L2 < 2e-2). Any miss drops that C to the f16qk rebuild, then master Triton, with a PATH CHOSEN line naming the cause. The old single-state eager+graph block was removed.
   Also extended (A): `_ps_replay_check` now refills state B then state A again then B again (fresh pages/cache values) after the existing A, A, B sequence, each against its own reference.
4. TU audit for static device pointers: `_SLIDE_CUDA` has no `__device__`/`__constant__` globals, no static locals, no cudaMalloc/cudaMemcpyToSymbol, no cached tensor maps; `slide_attn` reads every pointer from its tensor arguments each launch and allocates only `out` (shape-only, torch.empty). Only host-side function attributes are set (in `slide_setup`, once). Python side: the only module-level tensor is master's `_inv_freq` cache (theta-only, not state data) and master/PSEUDO's self-resetting tickets. Nothing to move.
5. Per-phase wall times: `_phase(...)` prints after splitkv v1, stage-2 build, stage-2 validate, batch-1 validate, pseudo gate + replay, diagnostics, and sliding cluster build/checks/timing, each with running total; `_build_slide` also prints its own compile time.
6. sk_attrs IDENTICAL line with both TUs loaded: after `_slide_init()`, `_EXT2.sk_attrs(0)` is re-read and printed as "landed-kernel check with both TUs loaded ... IDENTICAL to landed / DIFFERS FROM LANDED" (landed 121 regs, 0 local, 95264 B, 2 CTAs/SM). The earlier in-validate line is kept.
7. Open questions: compile budget is printed per phase (cache warmth on the judge unknown; fp8qk build is first, f16qk only on failure). Cache bound: the cache rel < 5e-2 check is unchanged and is run in the eager cases at kv_scale 0.25 and 1.0 (cluster cases include kv_scale 0.25); page-1 kv_scale 0.015625 tests route to master per the review, no new gate case has kv_scale != 1 outside the synthetic ones.

Deviations: none from the change paragraph; (A)'s gate untouched. Nothing was run (no GPU here).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn dispatch has no sync/.item()/prints/try; allocations are shape-only; all sync/prints are in import-time prepare code (diff checked).
- [x] Nothing derived from `data` is cached across calls: no module/static global holds a state pointer, TMA map or state-sized buffer (TU audited); only master's theta-keyed `_inv_freq` and self-resetting tickets.
- [x] Weights: n/a: attention problem, no weights.
- [x] Activations/KV: KV FP8, q rounded to E4M3 once in the cluster kernel (as in 41fcca0), fp32 softmax/merges, bf16 out.
- [x] Attention sees every key: cluster ranks cover [lo, seq) of the window, no skipping; checked against the fp32 reference in the self-check.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (grep for subprocess/fork/multiprocessing/Popen clean; lint network check passes).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a set around each load_inline and restored; arch flags printed; `-lineinfo`; cluster TU is sm_90 (no wgmma); PTX paths are those of the landed/ported TUs.

okc_lint: `OK with warnings` (36 host-sync warnings, all outside kernel_fn's call graph, prepare-time only); no errors. py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Landing candidate. Substituting the two cluster rows into s1's measured loads gives ~1.489x (s1 alone 1.4705; master re-measured 1.432-1.457 across three master-equivalent checks today: 2a3241780c7b4c46, f519261b7c2b435f and c0f1945a108c4aa2 all fell back to master on every row). Do not add sk-pdl-from-preattn: its gate chose plain launch (PDL +0.54 / +3.84 us slower), so that run measured master; its full decode_32 199 us row is below its own 229.6 us kernel row, i.e. a timing artefact, not a gain. Risk: the cluster TU was written before the two-state lesson; the audit above is mandatory.

## Result (after the run)
- job `eb6ff0cb701d4def` (check), status done, passed True, tests 16/16
- speedup 1.4870x, 95% CI [1.4647, 1.5093], score 2532.748975545189 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/restack-sk-pseudo-b1-cluster-c16-c4.result.json`, `muse_attention_decode/hypotheses/done/neutral/restack-sk-pseudo-b1-cluster-c16-c4.patch`; branch `hyp/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4`
