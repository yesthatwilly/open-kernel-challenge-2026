---
problem: muse_attention_decode
slug: restack-sk-pseudo-b1-cluster-c16-c4
created: 2026-10-25T054448Z
base: master
area: "sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4)"
change: >
  Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at
  B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A)
  From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job
  94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as
  is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup
  gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed
  only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at
  B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route.
  (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job
  195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU
  (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode
  (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on
  master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's
  stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS
  2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit
  (B) for module/static globals holding device pointers or state-sized buffers and move any into
  per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state
  refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills;
  extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no
  host sync, no allocation sized by input values (shape/config only), no data-dependent Python
  control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from
  inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window
  attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them:
  no module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0
  by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare
  against an fp32 reference with two states and alternating graph replays with refilled inputs
  (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to
  master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed
  TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)"
  - shapes: "*layer=full *"
    direction: flat
    why: "full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: B=2..16 stays on master Triton (C=8 gated off)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T054818Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T054819Z
base_sha: edf4bf29fe8d509a2574c0becee65138cb7705f2
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4
worktree: .worktrees/muse_attention_decode/restack-sk-pseudo-b1-cluster-c16-c4
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T084553Z
impl_commit: 00223bb62adc15f5b9d56ea5faff2034aaee58b5
submission_sha256: b015a01e0175339af029dc93dc681c896a94682bf5336d611817445cd83becee
norm_sha256: de8fb43c57a6803dc508db9e4e4738a46d1f1b25db01df36959e63bcc7bd9a37
submission_blob: 7eda7a001d411ff41aaf5946ed5c59c840686052
lint_warnings: 36
---

# restack-sk-pseudo-b1-cluster-c16-c4

## Change

Restack onto current master (c3c1b01, full decode stream-K v2 at 1<B<=64 + two-level ticket at B=1) the two measured sliding-row wins that touch disjoint shapes and are not on master. (A) From branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1 (impl 6f95ce7, job 94523c3313444fd8; its base IS master's submission blob d8769ef): take the whole PSEUDO change as is (splitkv_attn_sk_kernel<..., PSEUDO>, Lay<STAGES,256,true>, load_prefix_pseudo, _ps_setup gate, _ps_launch, the -DSK_PSEUDO=1 build with landed-only rebuild fallback). Its gate routed only sliding verify B=1 (grid 8P ticket variant 15.4 vs FA3 19.6 us, 1.277x) and kept FA3 at B=2..16 (0.653x) and B=17..32 (0.482x); keep that gate logic unchanged so the same bands route. (B) From branch hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4 (job 195dcc1209b6457b, built on the old deep-splits base) port ONLY the sliding cluster TU (sliding_cluster_kernel fp8qk, setup/launch for C=16 and C=4) and its routing: sliding decode (window != 0, q_len 1, tp8, page 1) at B=1 -> C=16, B=17..32 -> C=4, B=2..16 and B>32 stay on master Triton _sliding_fused_kernel, every verify untouched by (B). Do NOT take that branch's stream-K v2 (master already has it, newer). Give (B) its own bound constants (LESSONS 2026-10-25: never derive one path's gate from a constant the other edits). Before porting, audit (B) for module/static globals holding device pointers or state-sized buffers and move any into per-call torch.empty (shape-only) or self-resetting tickets; then add the two-state refill/alternating replay self-check for both routes (the s1 _ps_replay_check already refills; extend it to alternate two states). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_1 *` faster: s1 pseudo-decode route: 17.78 -> 13.44 us measured (kernel 9.8 us replaces FA3 7.9 + combine 3.9 + prepare 1.8 + norm-gate)
- `*layer=sliding load=decode_32 *` faster: cluster C=4: 24.33 -> 20.05 us measured (kernel 18.0 vs Triton 22.4)
- `*layer=sliding load=decode_1 *` faster: cluster C=16: 13.15 -> 12.57 us measured (-4.4%; near drift, read the kernel row)
- `*layer=full *` flat: full paths are master's byte-identical code (PSEUDO=false layout check prints IDENTICAL)
- `*layer=sliding load=decode_16 *` flat: B=2..16 stays on master Triton (C=8 gated off)

## Risk

Landing candidate. Substituting the two cluster rows into s1's measured loads gives ~1.489x (s1 alone 1.4705; master re-measured 1.432-1.457 across three master-equivalent checks today: 2a3241780c7b4c46, f519261b7c2b435f and c0f1945a108c4aa2 all fell back to master on every row). Do not add sk-pdl-from-preattn: its gate chose plain launch (PDL +0.54 / +3.84 us slower), so that run measured master; its full decode_32 199 us row is below its own 229.6 us kernel row, i.e. a timing artefact, not a gain. Risk: the cluster TU was written before the two-state lesson; the audit above is mandatory.

## Adversarial review

Reviewed 2026-10-25T054818Z by an adversarial subagent (model opus).

**Verdict: proceed.** Clean restack of two separately measured sliding-row wins on disjoint shapes. (A) applies to master's own blob d8769ef and (B) is gated away from every shape (A) touches, so each row is attributable from the profile and the PATH CHOSEN lines. Before the build: strip (B)'s kernel_fn try/except, prints and mutable set, and state the C=16 margin explicitly so the B=1 band is not silently gated off.

### Justification
A: The mechanism is plausible and was measured. Sliding verify_1 on master is FA3 main 7.9 + combine 3.9 + prepare_varlen 1.8 us + norm-gate (17.8 us shape). The s1 pseudo-decode kernel (grid p8, ticket) replaced that chain at 13.44 us in job 94523c33 on the exact master blob (d8769ef = BEST.submission_blob), so (A) is a re-run of measured code. Sliding decode_32 on master is _sliding_fused_kernel at 22.4 us (24.3 us shape, floor 3.3). Cluster C=4 measured 20.05 us in job 195dcc12; the Triton kernel it replaces is the same deep-splits kernel that is on master now. Each of the three rows has weight 46, and they sit in three different loads (decode 1, decode 32, verify 1), so the ~1.49x projection from substituting rows is arithmetically sound. The expected directions match the gates: full rows flat (byte-identical PSEUDO=false instantiations), sliding decode_16 flat (B=2..16 stays on Triton). Weakest link: the sliding decode_1 C=16 row. Its gain is 0.58 us (13.15 -> 12.57, 4.4%), and master's own sliding decode_1 has read 13.1-13.33 us across today's runs, so one run cannot confirm it. Worse, the hypothesis says gates route 'only when the printed interleaved graph-replay timing shows the new path winning by the stated margin' but never states (B)'s margin. If the builder reuses s1's >=1.05x, C=16 (about 1.046x) will usually fall back to Triton and the run silently measures master on that row. Second weakest: everything sits far above the 0.1-3.3 us floors, so there is headroom; the risk lies in the port, not the physics.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: the cluster C=16/C=4 wins were measured against this exact Triton _sliding_fused_kernel, which is unchanged on master, so the comparison carries over. No confound beyond drift on the near-noise decode_1 row.
- sliding-fused-triton-decode1-gate: the incumbent at sliding decode B=1 that C=16 replaces. Same code as in the stack run, so no interaction; the C=16 row stays near noise.
- full-decode1-streamk-s3-two-level-ticket: (A) adds a PSEUDO template flag and Lay<STAGES,256,true> to the same splitkv_attn_sk_kernel and builds with -DSK_PSEUDO=1. Full decode rows stay flat only if the landed instantiations are byte-identical (the sk_attrs IDENTICAL print must appear) and the landed-only rebuild fallback did not fire. A full-row move without IDENTICAL means the build changed, not the sliding work.
- fused-preattention-triton: (A) keeps _pre_attention_kernel ahead of the pseudo kernel at verify B=1. (B) folds norm/RoPE/FP8 store into the cluster kernel at decode B=1 and B=17..32, with a K-byte encode that only matches master within rel 5e-2 (byte match is info only). That is inside the 8% cache bound, but a row failure on a new-cache test would come from this encode, not the attention.

### Interference and A/B needs
No companion is warranted. The two sub-changes route disjoint shapes, (A) to sliding verify B=1 and (B) to sliding decode B=1 and B=17..32, and each run is a separate graph replay. Each row's gain or loss therefore belongs to exactly one sub-change, readable from the profile's top kernel (splitkv_attn_sk_kernel<...PSEUDO> vs sliding_cluster_kernel) and the PATH CHOSEN lines. An ab-revert of a landed hypothesis would not change that: neither route depends on a landed tuning other than the incumbent each replaces, and those incumbents are unchanged since the source measurements. The only ambiguous row is sliding decode_1, and the in-run interleaved timing that the gate prints (cluster C=16 vs Triton, cold L2, empty-graph subtracted) already serves as its control at no check cost. The A/B partner restack-sk-pseudo-cluster-plus-v3-fused-preattn is already minted and covers the full decode_16/32 question, so no duplicate is queued. With 8 checks left and 8 queued, spending one on a control would be waste.

### Open questions
- What margin gates (B)'s bands? Recommend keeping (B) config-only as it ran in 195dcc12 (C=16 at B=1, C=4 at B=17..32) and only printing the interleaved timing. Failing that, set C=16's margin at <=1.02x, since 1.05x would drop the B=1 band at its measured 1.046x.
- Compile budget: splitkv 49 s + sk 37 s + slide (fp8qk) 41 s + the -DSK_PSEUDO=1 rebuild, plus a possible landed-only rebuild fallback and an f16qk fallback build, plus all the gating timings. That is likely 3-5 min of the 15-min run. Print per-phase wall times. Is the torch extension cache warm on the judge?
- Does the s1 IDENTICAL check (sk_attrs for landed variants) still print IDENTICAL when (B)'s TU is also loaded in the same process? Register pressure and smem attributes should not change, but the line must be in the log to read the full rows as flat.
- The (B) self-check's cache tolerance is rel < 5e-2 with byte match as info only. Does the new-cache correctness bound (8% of the reference's) still pass on the kv_scale 0.015625 page-1 test? That test routes to master today (q_len 8 or page 64), so check that no cluster-gated test case has kv_scale != 1.


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn dispatch has no sync/.item()/prints/try; allocations are shape-only; all sync/prints are in import-time prepare code (diff checked).
- [x] Nothing derived from `data` is cached across calls: no module/static global holds a state pointer, TMA map or state-sized buffer (TU audited); only master's theta-keyed `_inv_freq` and self-resetting tickets.
- [x] Weights: n/a: attention problem, no weights.
- [x] Activations/KV: KV FP8, q rounded to E4M3 once in the cluster kernel (as in 41fcca0), fp32 softmax/merges, bf16 out.
- [x] Attention sees every key: cluster ranks cover [lo, seq) of the window, no skipping; checked against the fp32 reference in the self-check.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (grep for subprocess/fork/multiprocessing/Popen clean; lint network check passes).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a set around each load_inline and restored; arch flags printed; `-lineinfo`; cluster TU is sm_90 (no wgmma); PTX paths are those of the landed/ported TUs.

okc_lint: `OK with warnings` (36 host-sync warnings, all outside kernel_fn's call graph, prepare-time only); no errors. py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only emulation; no GPU). **Verdict: dispatch** (no fixes needed).

- (A) is byte-identical to s1 impl 6f95ce7 apart from the extended `_ps_replay_check` (A, A, refill B, refill A, refill B, each against its own fp32 reference, NaN-poisoned output); its 1.05x gate is unchanged and routed B=1 in 94523c33. The landed instantiations differ only in compile-time-folded code (`Lay<STAGES>` defaults, `plo = 0`, `R = B`), so full rows run master's kernels.
- (B) `_SLIDE_CPP`/`_SLIDE_CUDA` are byte-identical to 41fcca0, whose run 195dcc12 passed every gate (occupancy C=16 -> 14 >= 1, C=4 -> 62 >= 32, rel L2 ~1.6e-3 vs 2e-2, cache bytes 1.0000 match). The TU has no `__device__`/`__constant__` globals and no static pointers or maps (only `cudaFuncSetAttribute` in `slide_setup`). Python holds only extension handles, the theta-keyed `_inv_freq` and self-resetting tickets.
- Gates are config-only and strict: tp8, page 1, q_len 1, 0 < window <= 2048, RoPE, B=1 or 17..32, and they fall to `_sliding_forward` whenever the module is None. None of the 16 tests reaches the cluster route (page 64 / B 2..16 / verify), and the pseudo route reaches only verify bands whose replay check passed. The `_slide_two_state_check` setup (capture once on a third buffer set, then refill A / B / A / B with poisoned output, each against fp32) follows the two-state rule. Every check and the init are wrapped, so a failure falls to master with a PATH CHOSEN line instead of raising.
- kernel_fn's call graph has no host sync, prints, try/except, mutable state or allocation sized by input values (out/ws are shape-only). okc_lint gives OK with warnings (prepare-time syncs only) and py_compile passes.
- Budget: about 49 s splitkv + about 40 s sk(PSEUDO) + about 41 s slide fp8qk compiles, plus checks and gating, which is roughly 3-4 min cold, well inside the 15 min limit.

Residual risk: a sticky CUDA fault in a canary or self-check launch would poison the context, as in every landed TU. The sliding decode_1 C=16 gain (0.58 us) sits within drift, and the 16 KB log tail may cut off the earlier [ps] PATH CHOSEN lines; the phase and [slide] lines print last.
