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
review: required
status: queued
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
