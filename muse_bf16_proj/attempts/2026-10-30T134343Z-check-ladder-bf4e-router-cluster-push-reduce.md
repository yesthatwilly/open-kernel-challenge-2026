# muse_bf16_proj · 2026-10-30T134343Z · check · ladder-bf4e-router-cluster-push-reduce


- **Job:** `ccf952b8789a4a6a` (`okc status <job>`)
Hypothesis: `muse_bf16_proj/hypotheses/queue/2026-10-30T133514Z-ladder-bf4e-router-cluster-push-reduce.md` (branch `ladder/bf4e`).

- **Job:** `<job>` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `f4805b7d` (408,074 B, sha256 bc4ac874f631919a...), on master a05ce26f (BF4d landed, 1.333; master has since moved by one attention CLAIMS commit, and the bf16 submission is unchanged)

## Goal
Campaign 2026-10-30-public-ladder rung BF4e (one mechanism; user-requested). Does BF4d's push cluster reduce, which transferred to the H200 on ouns (judge 9acab018: m1 0.870, m8 0.932), move the verify_1 router row (m = 8; judge 5.06 us, public 4.4) when it runs on today's router_kernel<8,8> load path? Pre-registered: -0.4 us on the H200, hit >= -0.13 us. The [rcp] line is the H200 in-prepare race.

## What changed, and where
- **Section (8):** router_kernel gets a `PUSH` template parameter, in the same unit (renamed `muse_bf16_router_cluster_m8_v2`).
  - PUSH = false is today's kernel unchanged.
  - PUSH = true: a non-blocking barrier.cluster.arrive.relaxed after the loads are issued, and its wait before any DSMEM access. Each CTA stores its 16 x 8 fp32 partial into rank 0's receive buffer (st.shared::cluster, +4 KB smem), then arrive.release. Ranks 1..7 exit; rank 0 alone waits (acquire) and sums each output over ranks 0..7 in order from local smem.
  - Same values and order: **bitwise** today's route.
  - `router_cluster(x, hi, lo, out, push)` and `router_info(push)`.
- **Section (18), flag `router_cluster_push_ok`:**
  - Prerequisites: router_cluster_ok, then a regs / spill / >= 16 co-resident clusters gate.
  - Bitwise equality with the pull reduce over 64 launches at each of m 1/2/5/8.
  - 3 read-flush `_cell_gate` rounds at m = 8 with section (16) masked off (it does not route on the judge), judged by `_bf1_rule`. The fallback is the pull reduce.
- **Routing:** `_router_cluster_forward` passes push = flag. It applies wherever router_kernel<8,8> serves (m = 2..8 on the judge; m = 1 stays on rv).
- **A/B self-check:** router m <= 8 maps to `router_cluster_push_ok` first; the second state inherits it.
- **Untouched:** every other section.

## Node validation (H100 gpunode06 GPU 5, takemyjob hopper/ladder-bf4 1ea60ac probes/bf4/py/bf4eab.py), run `runs/20261030-063708-gpunode06`
- **In-prepare race** [rcp] pull/push: 0.881 / 0.864 / 0.885, pooled **0.878**, ROUTE. The push instance has 166 regs, 0 spill, 86.5 KB smem and 30 clusters.
- **Bitwise:** 64/64 x 4 m in prepare, 200/200 x 4 m in the probe. check_implementation 12/12 (router m 1/2/5/8 x normal, normal, outliers).
- **Order-controlled A/B** (each arm captured twice; capture order and per-rep order shuffled; read flush; 3 x 41):

| m, seed | pull us | push us | push/pull | identity pull / push |
|---|---|---|---|---|
| 8, 3000 | 4.83 | 4.22 | 0.874 | 0.993 / 0.985 |
| 8, 9001 | 4.83 | 4.10 | 0.848 | 0.993 / 1.000 |
| 5, 3000 | 4.90 | 4.29 | 0.876 | 1.013 / 0.993 |
| 5, 9001 | 4.83 | 4.10 | 0.848 | 1.000 / 1.000 |
| 2, 3000 | 4.67 | 4.13 | 0.884 | 1.000 / 1.008 |
| 2, 9001 | 4.70 | 4.06 | 0.864 | 0.980 / 0.969 |

- **No-op gate** (byte-identical arms): 1.019 / 0.993 / 1.006, pooled 1.013.
- Import + prepare took 299 s on the H100; this rung adds no new unit.

## Rules check (against `git diff a05ce26f f4805b7d`)
- [x] CUDA-graph capturable: kernel_fn is unchanged; `_router_cluster_forward` passes a state flag (prepare constant) to the same single launch. Syncs and `.item()` are prepare-only.
- [x] Nothing derived from `data` is cached.
- [x] Router weight: the BF16 hi + lo pair with FP32 accumulation (today's route, rules :107-108). The output is bitwise today's.
- [x] Activations: x is BF16; output is fp32. Attention items n/a.
- [x] No timing patches, harness frames, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call: one kernel, no output caching.
- [x] First line kept. py_compile OK, __pycache__ removed. okc_lint: `OK with warnings` (all pre-existing prepare-side items).
- [x] Size 408,074 B < 1,000,000. Prepare-time cost is about 1 s.
- [x] Own code: a port of our BF4d push reduce. Structure idea cited as alygg77@6cae49a rsmall. No public or MIT code.

## Risks / what could fail
Low. This is the same mechanism and the same barrier semantics as BF4d, which passed on the H200 (bitwise checks and race ROUTE). The cluster kernel's load path differs (cp.async smem rather than ouns' path), but the reduce is identical in structure.

## Result (after the run)
<verify_1 router judge time vs 5.06; [rcp] ratio; delta vs -0.4 predicted; hit/miss>
- job `ccf952b8789a4a6a` (check), passed, 1.334x [1.314, 1.354] vs board 1.333: [rcp] router m8 pull/push 0.857/0.865/0.874 pooled 0.866 -> ROUTE; verify-1 router 4.9 us (was 5.4; public 4.4). Not submitted (user stop 10-30 ~14:00Z); scored-submit candidate.
