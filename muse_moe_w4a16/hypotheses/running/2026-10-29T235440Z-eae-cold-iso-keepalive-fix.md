---
problem: muse_moe_w4a16
slug: eae-cold-iso-keepalive-fix
created: 2026-10-29T235440Z
base: master
mint_blob: "c6e41c9acc6169ac8a532684437955b1d560835b"
best_at_mint: 2.0201
area: "prepare-only correctness: [ea-enc] T=256 cold-half iso timing graphs"
change: >
  Correctness fix with no timed-path change: in _eae_class (muse_moe_w4a16/submission.py
  ~18447-18477), the iso[('c', a)] timing graphs are captured over h3/y3/l3 tensors allocated
  outside the capture and held only by the lambda's default args; the loop frees the previous set
  while its graph still writes to it (use-after-free during _rt_time; sticky illegal address on
  gpunode06 5/5 with the SGLang baseline in process, the judge's layout; written into freed memory
  without faulting on the judge so far). Keep a list of every (h3, y3, l3) until timing ends
  (keep.append after allocation; del keep with the graphs). Routes, picks and kernel_fn are
  unchanged; the moe board's final reruns (top entries rerun several times on held-out data) must
  not carry the fault.
expected:
  - shapes: "*"
    direction: same
    why: prepare-only memory-lifetime fix; timed path byte-identical in behavior
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "engine-a-a4b-a5b-encodings-race: the landed change that introduced the iso graphs; this only extends three tensors' lifetime"
target: check
value_gain: 0.1
value_p: 0.9
value_ev: 0.09
value_cost: 40k tokens, 1 judge run
evidence: >
  gpunode06 GPU 4 runs 20261029-16*-gpunode06 (takemyjob .takemyjob-wt/eaecrash): master moe 09b11a9d
  faulted 5/5 in [ea-enc] first replay of ('c', None) with the SGLang baseline in process; fixed
  bytes (sha 986a2bad, 948,330 B) passed 4/4; the moe board 24f183ff (2.0201) carries the bug
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T235601Z
review_companions: null
review_best: 2.020119462295186
review_blob: c6e41c9acc6169ac8a532684437955b1d560835b
stage: claimed
claimed_at: 2026-10-29T235601Z
base_sha: 2090c7b69f865761852a7f539d70bfb97f99f2c3
base_blob: c6e41c9acc6169ac8a532684437955b1d560835b
branch: hyp/muse_moe_w4a16/eae-cold-iso-keepalive-fix
worktree: .worktrees/muse_moe_w4a16/eae-cold-iso-keepalive-fix
mode: null
restack_of: null
stale_base: null
---

# eae-cold-iso-keepalive-fix

## Change

Correctness fix with no timed-path change: in _eae_class (muse_moe_w4a16/submission.py ~18447-18477), the iso[('c', a)] timing graphs are captured over h3/y3/l3 tensors allocated outside the capture and held only by the lambda's default args; the loop frees the previous set while its graph still writes to it (use-after-free during _rt_time; sticky illegal address on gpunode06 5/5 with the SGLang baseline in process, the judge's layout; written into freed memory without faulting on the judge so far). Keep a list `keep` of every (h3, y3, l3) until timing ends (keep.append after allocation; del keep with the graphs). Routes, picks and kernel_fn are unchanged; the moe board's final reruns (top entries rerun several times on held-out data) must not carry the fault.

## Expected

- `*` same: prepare-only memory-lifetime fix; timed path byte-identical in behavior

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** gpunode06 GPU 4 runs 20261029-16*-gpunode06 (takemyjob .takemyjob-wt/eaecrash): master moe 09b11a9d faulted 5/5 in [ea-enc] first replay of ('c', None) with the SGLang baseline in process; fixed bytes (sha 986a2bad, 948,330 B) passed 4/4; the moe board 24f183ff (2.0201) carries the bug
- **Gain if it works:** 0.1% headline; **p** 0.9; **EV** 0.09% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 0.1% < 2.5%; EV 0.09% < 1.5%), minted because the user described it.
- **Cost:** 40k tokens, 1 judge run.

## Risk

Root cause and node validation: research/h100-eae-crash-1029.md (branch hyp/muse_moe_w4a16/eae-t1-guard b743b4e4): with the fix, natural/forced v1/forced v6pe2 T=1 routes all pass with the baseline in process; every raced arm exact. Decode's stack is unaffected (graphs allocate inside the capture).

## Adversarial review

Reviewed 2026-10-29T235601Z by an adversarial subagent (model opus).

**Verdict: proceed.** The root cause is right and the 3-line keep-alive is enough. It changes only prepare's E race (_eae_class at T=256 cold), so kernel_fn's timed path and all routes and picks are byte-identical. Expected headline is flat; the run is a correctness/stability check, not a perf lead.

### Justification
The root cause checks out. _rt_graph (submission.py:4199) returns only (graph, out) and drops fn. torch.cuda.CUDAGraph keeps no reference to tensors allocated outside the capture. So once _rt_graph returns, the only reference to h3/y3/l3 (the lambda's default args) dies. The next arm iteration rebinds h3/y3/l3, which frees the previous set back to the caching allocator, while iso[('c', a)] still writes to those addresses during _rt_time's warm and timed replays. That is a use-after-free that depends on allocator layout, which matches the node evidence (fails 5/5 with the sglang baseline in process, passes once kept alive). On the last arm the tensors also outlive the loop only by accident of scoping. The fix appends (h3, y3, l3) to keep before capture and deletes keep only after _rt_time and the sync. Is it sufficient? Every other buffer the E-phase graphs touch is alive across replay. (a) graphs[a] = _rt_graph(lambda: kernel_fn(data, state)) allocates its intermediates inside the capture, so they live in the graph's private pool, which the graph keeps. Its inputs are data (alive for the whole draw and deleted only after graphs) and persistent state buffers. outs[a] is kept. (b) The iso graph's other operands are data['hidden_states'] and cold (locals alive through _rt_time) plus ea[...] persistent buffers (w13r/w2r/w13e*/w2e*, gs13/gs2, ws13x/ws2x, tk13x/tk2x, errx). The w13e*/w2e* pops in _eae_run happen after every _eae_class has returned and its graphs are deleted. (c) _eae_hy and the post-pick state-copy check run eagerly, with no graphs. (d) _rt_time's empty baseline graph has no operands. (e) Per the triage note, _wh_time's out and _wh_replay's outs stay referenced through _rt_time. kind 't1' and 'hot' never build iso graphs (cold is None). Weakest link: the cold tensor is a slice of _hm_align's output. If that is a view into a persistent hm buffer that kernel_fn's own T=256 replay rewrites, the iso graph's routing could change between replays. That affects only the values in the print-only crat, not memory safety or picks.

### Confounders on the current kernel
- engine-a-a4b-a5b-encodings-race (bbd1661f) introduced _eae_class and the iso cold graphs. This fix only repairs its prepare-time race and does not change what that race picks: crat is print-only, and picks come from rat on the kernel_fn graphs.
- v4x256 / v4x256c (T=256 hybrid cold half on v4cxs) is what makes _eae_base accept kind 'cold' at T=256 and so reach the buggy branch. Not a confounder for perf, since the timed path is unchanged.

### Interference and A/B needs
No companions. The diff changes no kernel, route, threshold or pick input: it only extends the lifetime of three prepare-only scratch tensors until after timing. Any headline delta is run-to-run noise. The only thing to read from the run is whether prepare completes and whether [ea-enc] PATH CHOSEN matches master's picks.

### Open questions
- Is cold (_hm_align(...)[1][6]) a view into a persistent hm buffer that kernel_fn's T=256 replays overwrite? If so, the printed 'cold iso' ratios time a drifting routing. This is harmless to picks but worth knowing before trusting crat.
- Optional hardening: have _rt_graph return or hold fn (e.g. attach it to the graph object) so that every future external-buffer capture is kept alive by construction.

