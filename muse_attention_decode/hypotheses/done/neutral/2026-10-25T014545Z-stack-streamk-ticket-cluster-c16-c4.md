---
problem: muse_attention_decode
slug: stack-streamk-ticket-cluster-c16-c4
created: 2026-10-25T014545Z
base: master
area: full + sliding decode attention (stack of two pending kernels)
change: >
  Stack the two pending (user-held, not on master) decode wins on current master; they touch
  disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harvest', 'Do they stack?'). (1)
  full-decode-splitkv-s2-streamk-ticket-inproc (check e592b1f5d3db4f83, 1.4171x): as depends_on,
  claim restacks its patch (done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch) onto
  master automatically; keep it unchanged: TU muse_full_decode_splitkv_v2, splitkv_attn_sk_kernel
  with 4 stages + ticket combine, routed only for full layer, decode, 1<B<=64, page 1, with its
  canary and 8 self-checks. (2) sliding-cluster-s2-decode-c4-pdl-inproc (check 1990ce53e9d34073,
  1.4001x): claim restacks only ONE parent patch, so the implementer must 3-way merge this second
  branch's diff BY HAND inside the worktree: git apply -3
  --include=muse_attention_decode/submission.py on
  muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch
  (equivalently, the diff of branch
  hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc against its base 593859d3),
  resolving conflicts so both TUs, both prepare() blocks and both dispatch branches survive; each
  path keeps its own gate constant (LESSONS stack note). Then switch the cluster C=8 band OFF:
  sliding decode B=2..16 goes back to master's Triton path; keep C=16 at B=1 and C=4 at B=17..32
  (fp8qk sliding_cluster_kernel); master keeps B>32, every verify and page 64. Do not build or
  self-check the C=8 instantiation. Keep both prepare() self-checks and canaries; print both PATH
  CHOSEN lines (streamk: full decode 1<B<=64; cluster: C=16 at B=1, C=4 at B=17..32, Triton at
  B=2..16) compactly and late (archived log is a ~15 KB tail). Run as an okc check. This check
  also RE-MEASURES streamk on top of current master (its 1.4171 CI lower bound 1.396 only just
  cleared master's 1.3949), so read the streamk kernel rows and the cluster kernel rows from the
  profile separately.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "streamk+ticket: splitkv A 111.8 + combine 8.2 = 120.0 us -> sk 108.9 us (-9.3%); shape 122.8 -> 110.9 us (-9.7%)"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "streamk+ticket: 254.3 -> 226.8 us kernel (-10.8%); shape 241.9 -> 226.8 us (-6.2%)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "cluster C=4: Triton 22.1 -> cluster 18.3 us kernel; shape 24.2 -> 20.2 us (-16.8%)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "cluster C=16: shape 13.2 -> 12.5 us (-5.0%), kernel flat"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: "C=8 band gated off: master Triton 12.9 us kernel (C=8 cluster was 13.8, shape +5.7%)"
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched FA3 control
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: B=1 full decode not routed (FA3)
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T034534Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T034535Z
base_sha: 63995118a6d87d5714181d318458a7595fe4bbac
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4
worktree: .worktrees/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4
mode: check
restack_of: full-decode-splitkv-s2-streamk-ticket-inproc
stale_base: null
implemented_at: 2026-10-25T034707Z
impl_commit: 41fcca02d61db12be673f401bba44d547d33fa13
submission_sha256: 8c8ef8bdee615f0949d6703101b48de03a1da47a1a89223fe4a64395f38250f9
norm_sha256: d38bad22f2e2fc66b2e82e27e9a1e3fb21010307aee25312ecd3b7cfa26ea734
submission_blob: edf13326bb5bcdae8c345dd5a780fcb948cb78d6
lint_warnings: 13
attempt_file: 2026-10-25T034707Z-check-stack-streamk-ticket-cluster-c16-c4.md
prepared_at: 2026-10-25T034707Z
job: 195dcc1209b6457b
dispatched_at: 2026-10-25T034716Z
outcome: neutral
verdict_detail: beats-base-not-best
speedup: 1.4265750765791432
speedup_ci: [1.405176450430456, 1.4537135820295741]
harvested_at: 2026-10-25T043354Z
best_at_harvest: 1.4321773019894537
promotion: null
---

# stack-streamk-ticket-cluster-c16-c4

## Change

Stack the two pending (user-held, not on master) decode wins on current master; they touch disjoint shapes and code paths (REWRITE_READ_2026-10-25 '01:36Z harvest', 'Do they stack?'). (1) full-decode-splitkv-s2-streamk-ticket-inproc (check e592b1f5d3db4f83, 1.4171x): as depends_on, claim restacks its patch (done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch) onto master automatically; keep it unchanged: TU muse_full_decode_splitkv_v2, splitkv_attn_sk_kernel with 4 stages + ticket combine, routed only for full layer, decode, 1<B<=64, page 1, with its canary and 8 self-checks. (2) sliding-cluster-s2-decode-c4-pdl-inproc (check 1990ce53e9d34073, 1.4001x): claim restacks only ONE parent patch, so the implementer must 3-way merge this second branch's diff BY HAND inside the worktree: git apply -3 --include=muse_attention_decode/submission.py on muse_attention_decode/hypotheses/done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch (equivalently, the diff of branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc against its base 593859d3), resolving conflicts so both TUs, both prepare() blocks and both dispatch branches survive; each path keeps its own gate constant (LESSONS stack note). Then switch the cluster C=8 band OFF: sliding decode B=2..16 goes back to master's Triton path; keep C=16 at B=1 and C=4 at B=17..32 (fp8qk sliding_cluster_kernel); master keeps B>32, every verify and page 64. Do not build or self-check the C=8 instantiation. Keep both prepare() self-checks and canaries; print both PATH CHOSEN lines (streamk: full decode 1<B<=64; cluster: C=16 at B=1, C=4 at B=17..32, Triton at B=2..16) compactly and late (archived log is a ~15 KB tail). Run as an okc check. This check also RE-MEASURES streamk on top of current master (its 1.4171 CI lower bound 1.396 only just cleared master's 1.3949), so read the streamk kernel rows and the cluster kernel rows from the profile separately.

## Expected

- `*layer=full load=decode_16 *` faster: streamk+ticket: splitkv A 111.8 + combine 8.2 = 120.0 us -> sk 108.9 us (-9.3%); shape 122.8 -> 110.9 us (-9.7%)
- `*layer=full load=decode_32 *` faster: streamk+ticket: 254.3 -> 226.8 us kernel (-10.8%); shape 241.9 -> 226.8 us (-6.2%)
- `*layer=sliding load=decode_32 *` faster: cluster C=4: Triton 22.1 -> cluster 18.3 us kernel; shape 24.2 -> 20.2 us (-16.8%)
- `*layer=sliding load=decode_1 *` faster: cluster C=16: shape 13.2 -> 12.5 us (-5.0%), kernel flat
- `*layer=sliding load=decode_16 *` flat: C=8 band gated off: master Triton 12.9 us kernel (C=8 cluster was 13.8, shape +5.7%)
- `*load=verify_*` flat: untouched FA3 control
- `*layer=full load=decode_1 *` flat: B=1 full decode not routed (FA3)

## Risk

Estimated headline ~1.440x (the read's load-level recompute; drift not removed) vs master 1.3949. Risks: the hand 3-way merge of the cluster patch over the restacked streamk patch (both edit prepare() and dispatch in submission.py); both parents are pending the user, so a win here is a landing candidate only with user approval; full decode_32 shows a known 242 vs 256-260 us bimodality on identical kernels, so judge streamk from kernel rows; the cluster kernel spills 32 B local.

## Adversarial review

Reviewed 2026-10-25T034534Z by an adversarial subagent (model opus).

**Verdict: proceed.** The two parents are disjoint in shapes, gates, module names and workspaces, so their rows can be read apart and no companion is needed. But the merge instruction as written is wrong: the cluster patch is a stage-1-to-stage-2 delta, not a diff against master. Three PATH CHOSEN/gate edits are also needed to switch C=8 off.

### Justification
A. Mechanism: both halves have already been measured on BEST's base (698d4cd3 lineage). Streamk is check e592b1f5: kernel rows splitkv A+combine 120.0 -> 108.9 us at full decode_16 and 254.3 -> 226.8 us at decode_32. Cluster is check 1990ce53: C=4 Triton 22.1 -> 18.3 us at sliding decode_32, and C=8 lost (12.9 -> 13.8) at decode_16. Disjointness holds in code. In kernel_fn the `if cfg["window"]:` block returns for every sliding decode (cluster, or master Triton _sliding_forward) before `_splitkv_gate`, which also requires window == 0. _slide_gate requires window > 0 and q_len 1 and does not read _EXT/_EXT2, so neither gate depends on a constant the other edits (LESSONS stack note satisfied). The builds are separate load_inline modules: muse_full_decode_splitkv_v1 (master), muse_full_decode_splitkv_v2 (streamk) and muse_sliding_cluster_v2 (+_f16qk only on fallback). Workspaces are separate too: streamk's _TICKETS (zeroed at import and re-zeroed after _sk_diagnostics) vs master Triton's _COUNTERS, and the cluster kernel uses DSMEM with no global counters. Budget: there is no prepare(). All builds and self-checks run at import, inside the 15-minute total (TASK.md:119). That is ~50 s v1 + ~37 s v2 + ~41 s cluster fp8qk (+~40 s if f16qk fallback) + both self-check ladders + streamk diagnostics, roughly 3-4 min of compile, which fits. Expected directions match the harvest rows. Weakest links: (1) the sliding decode_1 'faster' is a 0.7 us shape gain while the kernel got slower (10.4 -> 10.6 us), so it is outside the kernel and may be drift; do not count it. (2) The ~1.440x headline is a load-level recompute with drift left in, and full decode_32 is bimodal (242 vs 256-260 us on identical kernels); judge streamk from the sk kernel rows only. (3) BLOCKING FOR THE IMPLEMENTER: the cluster patch (done/neutral/sliding-cluster-s2-decode-c4-pdl-inproc.patch) has preimage blob bd96c94 = 593859d3:submission.py, the stage-1 commit sliding-cuda-cluster-dsmem-onelaunch, which already contains the whole cluster TU (9 sliding_cluster_kernel hits). Master (c6854f0) has none. So `git apply -3` of that patch, or 'the diff against 593859d3', carries only the stage-2 delta. In a scratch trial (git merge-file, outside the repo) the stage-1 TU then lands only inside a conflict block next to streamk's appended TU, and a quick 'take ours' resolution silently drops the cluster kernel. Correct recipe: 3-way merge with base = 589a891^ (= 3693fb8 lineage, blob 1c0f5fa, the true pre-cluster base), ours = the restacked streamk file, theirs = branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc. Equivalently, apply the cumulative diff 589a891^..that branch with -3. My trial gave 4 conflicts, all mechanical: module docstring, the kernel_fn sliding head (take the cluster side, which keeps master's Triton fallback), the appended TU block (keep BOTH), and the import-time init (run streamk init, then _slide_init). Master's sliding-deep-splits _sliding_forward stays untouched, so B=2..16 gets master's deep-splits Triton as intended.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: the cluster lineage branched from 3693fb8, before deep-splits. Its Triton fallback still says '16 splits at every batch'. If the merge takes the cluster side of _sliding_forward, B=2..16 regresses to pre-deep-splits Triton and sliding decode_16 is no longer the clean 'flat' control. Verify that _sliding_forward is byte-identical to master after the merge.
- full-decode-cuda-gather-splitkv: streamk replaces its kernel A + combine on 1<B<=64 but keeps v1 built and self-checked as the fallback. If v2 falls back, full decode rows silently become master. Check PATH CHOSEN: 'full decode streamk+ticket v2'.
- fused-preattention-triton / paired-heads-launch-cleanup(-ab-tune-batch-gate): streamk takes the unpaired q from master's pre-attention kernel (paired = ... and not splitkv). The cluster kernel does its own norm/rope/cache store for sliding decode. Neither half changes the other's pre-attention, so they do not confound.

### Interference and A/B needs
No companion. Each half was already measured alone against the same BEST, the shapes are disjoint (full decode 1<B<=64 vs sliding decode B=1 and 17..32), and the profile kernel rows (splitkv_attn_sk_kernel vs sliding_cluster_kernel vs _sliding_fused_kernel) attribute every moved shape to one half. The C=8-off decision is self-controlling: sliding decode_16 should return to master's Triton ~12.9 us kernel. Textual conflicts with the running checks: (a) sliding-decode-paired-kvhead-rows (branch 990710b..87a3093, base c6854f0) only edits _sliding_forward plus a new Triton kernel after master line 342. The stack does not touch that region, so the two merge textually clean. They do overlap semantically: the paired gate (length==1 and batch>1) would, in the stack, run only at B=2..16 and B>32, because cluster C=4 claims 17..32 earlier in kernel_fn. Its decode_32 result is therefore moot unless it beats cluster's 18.3 us, and if both land the B=17..32 owner must be chosen explicitly. (b) full-decode-splitkv-s3-fused-preattn (branch at 22918e4, base c6854f0) is a superset of streamk s2. It rewrites streamk lines (_build_sk signature, header docstring, the kernel_fn sk call around master 893-919, and the tail init block). It conflicts textually with this stack's unchanged s2 copy in those hunks and at the import-time init tail, where both stack and s3 append. If s3 lands first, re-derive this stack as s3 + cluster (the same 589a891^-based 3-way) rather than merging the two stacks.

### Open questions
- C=8 off, minimal safe edit: drop 8 from _slide_init's todo list, so _SLIDE_MOD[8] stays None. Make _slide_clusters return 0 for 2<=B<=16. Remove the B=1 fallback '(8 if _SLIDE_MOD[8] ...)'. Change the PATH CHOSEN loop so B=2..16 prints 'Triton by gate' instead of reading _SLIDE_WHY[8]. Leaving the C=8 template compiled is harmless; editing the CUDA switch to remove it adds risk for no gain.
- Does the import-time time budget hold if cluster fp8qk fails and the f16qk rebuild runs (+~40 s), on top of streamk's _sk_diagnostics? Keep the diagnostics or trim them to one B.
- Landing needs the user's approval for both parents, which are user-held.


## Implementation notes

Merge (review recipe, which wins over the spec's `git apply -3` of the stage-2 delta): `git merge-file` in a scratch dir with base = `589a891^:submission.py`, ours = the restacked stream-K file in the worktree, theirs = branch `hyp/.../sliding-cluster-s2-decode-c4-pdl-inproc`. Exactly 4 conflicts, resolved as the review said:
1. Module docstring and imports: combined text (stream-K full decode + cluster sliding at C=16 B=1 and C=4 B=17..32); kept `import random` (stream-K needs it).
2. Head of kernel_fn: took the cluster side (`_slide_gate` -> `_slide_forward` inside try/except RuntimeError with master fallback; else master Triton `_sliding_forward`; verify falls to FA3).
3. Appended CUDA source and helpers: kept both (stream-K block, then the cluster TU and `_slide_*` helpers).
4. Import-time init: stream-K init block first (unchanged: tickets zeroed, `_sk_diagnostics`, tickets re-zeroed), then `try: _slide_init() except -> all _SLIDE_MOD None`.

Verified: AST-level comparison of every top-level function vs master blob c6854f0: `_sliding_forward` and its Triton kernels are byte-identical (deep-splits kept). Only `_full_decode_splitkv` (stream-K, expected) and `kernel_fn` differ.

C=8 off: `_slide_init` builds/checks only C in {16, 4} (`_SLIDE_MOD[8]` stays None; the C=8 template stays compiled-in, unused); `_slide_clusters` returns 0 for B=2..16 and B=1 no longer falls back to C=8; PATH CHOSEN for B=2..16 prints "sliding master Triton (deep-splits) ... (C=8 off by gate)"; the B=1/17..32 failure message `want` no longer references 8; stream-K's last line now reads "every other shape (verify, full batch 1, page 64, dp, sliding outside the cluster gate)"; docstring and stage comment updated.

Audit fixes kept: stream-K (gx = min(gridDim.x, T), tickets re-zeroed, binaryVersion == 90 gate) untouched; cluster try/except fallback around `_slide_forward` kept. No prepare(): all builds and self-checks run at import, as in both parents; the f16qk rebuild fallback remains (+~40 s worst case), diagnostics not trimmed.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only config-only gates and kernel launches; syncs/.item() are only in import-time self-checks (lint warnings below).
- [x] Nothing derived from `data` is cached across calls (tickets/workspaces are shape-independent buffers; gates read cfg only).
- [x] Weights: n/a: no weight repack touched.
- [x] Activations/KV/q precision unchanged from the two parents (cluster fp8qk is the parent's measured, self-checked path).
- [x] Attention sees every key: n/a for new logic; both kernels are the parents' unchanged, self-checked kernels.
- [x] No timing patches, harness access, reference import, child process or subprocess (grep for subprocess/fork/multiprocessing/Popen/os.system empty).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warns, all outside kernel_fn's call graph, import-time self-checks). py_compile ok; __pycache__ removed.

## Result

- job `195dcc1209b6457b` (check), status done, passed True, tests 16/16
- speedup 1.4266x, 95% CI [1.4052, 1.4537], score 2640.0993973842596 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_attention_decode/hypotheses/done/neutral/stack-streamk-ticket-cluster-c16-c4.result.json`, `muse_attention_decode/hypotheses/done/neutral/stack-streamk-ticket-cluster-c16-c4.patch`; branch `hyp/muse_attention_decode/stack-streamk-ticket-cluster-c16-c4`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.17 | 12.57 | 1.0317 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.26 | 27.1 | 0.9969 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.34 | 15.26 | 0.998 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 122.79 | 111.64 | 1.1004 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.24 | 20.05 | 1.1809 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 241.9 | 226.98 | 1.0677 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 16.72 | 17.54 | 0.9417 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.03 | 30.32 | 0.9903 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.67 | 22.75 | 1.0041 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 161.98 | 162.52 | 1.0008 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.7 | 28.47 | 1.0095 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 300.83 | 301.81 | 0.9994 | flat |

gained on full decode_16, decode_32; sliding decode_32; lost on sliding verify_1. Expected hit: 6/7. Noise ±1.7%.

## Follow-ups

- beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-streamk-ticket-cluster-c16-c4`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

