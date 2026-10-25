# muse_attention_decode · 2026-10-25T034707Z · check · stack-streamk-ticket-cluster-c16-c4

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T014545Z-stack-streamk-ticket-cluster-c16-c4.md` (branch `hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4`).

- **Job:** `195dcc1209b6457b` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `41fcca0`

## Goal
Stack the two pending (user-held, not on master) decode wins on current master; they touch disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harvest', 'Do they stack?'). (1) full-decode-splitkv-s2-streamk-ticket-inproc (check e592b1f5d3db4f83, 1.4171x): as depends_on, claim restacks its patch (done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch) onto master automatically; keep it unchanged: TU muse_full_decode_splitkv_v2, splitkv_attn_sk_kernel with 4 stages + ticket combine, routed only for full layer, decode, 1<B<=64, page 1, with its canary and 8 self-checks. (2) sliding-cluster-s2-decode-c4-pdl-inproc (check 1990ce53e9d34073, 1.4001x): claim restacks only ONE parent patch, so the implementer must 3-way merge this second branch's diff BY HAND inside the worktree: git apply -3 --include=muse_attention_decode/submission.py on muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch (equivalently, the diff of branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc against its base 593859d3), resolving conflicts so both TUs, both prepare() blocks and both dispatch branches survive; each path keeps its own gate constant (LESSONS stack note). Then switch the cluster C=8 band OFF: sliding decode B=2..16 goes back to master's Triton path; keep C=16 at B=1 and C=4 at B=17..32 (fp8qk sliding_cluster_kernel); master keeps B>32, every verify and page 64. Do not build or self-check the C=8 instantiation. Keep both prepare() self-checks and canaries; print both PATH CHOSEN lines (streamk: full decode 1<B<=64; cluster: C=16 at B=1, C=4 at B=17..32, Triton at B=2..16) compactly and late (archived log is a ~15 KB tail). Run as an okc check. This check also RE-MEASURES streamk on top of current master (its 1.4171 CI lower bound 1.396 only just cleared master's 1.3949), so read the streamk kernel rows and the cluster kernel rows from the profile separately.

## What changed, and where
Merge (review recipe, which wins over the spec's `git apply -3` of the stage-2 delta): `git merge-file` in a scratch dir with base = `589a891^:submission.py`, ours = the restacked stream-K file in the worktree, theirs = branch `hyp/.../sliding-cluster-s2-decode-c4-pdl-inproc`. Exactly 4 conflicts, resolved as the review said:
1. Module docstring and imports: combined text (stream-K full decode + cluster sliding at C=16 B=1 and C=4 B=17..32); kept `import random` (stream-K needs it).
2. Head of kernel_fn: took the cluster side (`_slide_gate` -> `_slide_forward` inside try/except RuntimeError with master fallback; else master Triton `_sliding_forward`; verify falls to FA3).
3. Appended CUDA source and helpers: kept both (stream-K block, then the cluster TU and `_slide_*` helpers).
4. Import-time init: stream-K init block first (unchanged: tickets zeroed, `_sk_diagnostics`, tickets re-zeroed), then `try: _slide_init() except -> all _SLIDE_MOD None`.

Verified: AST-level comparison of every top-level function vs master blob c6854f0: `_sliding_forward` and its Triton kernels are byte-identical (deep-splits kept). Only `_full_decode_splitkv` (stream-K, expected) and `kernel_fn` differ.

C=8 off: `_slide_init` builds/checks only C in {16, 4} (`_SLIDE_MOD[8]` stays None; the C=8 template stays compiled-in, unused); `_slide_clusters` returns 0 for B=2..16 and B=1 no longer falls back to C=8; PATH CHOSEN for B=2..16 prints "sliding master Triton (deep-splits) ... (C=8 off by gate)"; the B=1/17..32 failure message `want` no longer references 8; stream-K's last line now reads "every other shape (verify, full batch 1, page 64, dp, sliding outside the cluster gate)"; docstring and stage comment updated.

Audit fixes kept: stream-K (gx = min(gridDim.x, T), tickets re-zeroed, binaryVersion == 90 gate) untouched; cluster try/except fallback around `_slide_forward` kept. No prepare(): all builds and self-checks run at import, as in both parents; the f16qk rebuild fallback remains (+~40 s worst case), diagnostics not trimmed.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only config-only gates and kernel launches; syncs/.item() are only in import-time self-checks (lint warnings below).
- [x] Nothing derived from `data` is cached across calls (tickets/workspaces are shape-independent buffers; gates read cfg only).
- [x] Weights: n/a: no weight repack touched.
- [x] Activations/KV/q precision unchanged from the two parents (cluster fp8qk is the parent's measured, self-checked path).
- [x] Attention sees every key: n/a for new logic; both kernels are the parents' unchanged, self-checked kernels.
- [x] No timing patches, harness access, reference import, child process or subprocess (grep for subprocess/fork/multiprocessing/Popen/os.system empty).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warns, all outside kernel_fn's call graph, import-time self-checks). py_compile ok; __pycache__ removed.

## Risks / what could fail
medium
Estimated headline ~1.440x (the read's load-level recompute; drift not removed) vs master 1.3949. Risks: the hand 3-way merge of the cluster patch over the restacked streamk patch (both edit prepare() and dispatch in submission.py); both parents are pending the user, so a win here is a landing candidate only with user approval; full decode_32 shows a known 242 vs 256-260 us bimodality on identical kernels, so judge streamk from kernel rows; the cluster kernel spills 32 B local.

## Result (after the run)
- job `195dcc1209b6457b` (check), status done, passed True, tests 16/16
- speedup 1.4266x, 95% CI [1.4052, 1.4537], score 2640.0993973842596 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_attention_decode/hypotheses/done/neutral/stack-streamk-ticket-cluster-c16-c4.result.json`, `muse_attention_decode/hypotheses/done/neutral/stack-streamk-ticket-cluster-c16-c4.patch`; branch `hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4`
