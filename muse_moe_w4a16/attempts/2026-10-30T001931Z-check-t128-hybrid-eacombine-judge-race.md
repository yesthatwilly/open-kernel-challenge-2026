# muse_moe_w4a16 · 2026-10-30T001931Z · check · t128-hybrid-eacombine-judge-race

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-30T000244Z-t128-hybrid-eacombine-judge-race.md` (branch `hyp/muse_moe_w4a16/t128-hybrid-eacombine-judge-race`).

- **Job:** `4c5032bc57be4cf5` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `3f8bfc17`

## Goal
Route moe T=128 to the forked H-16 hybrid (wh5ds hot half + E5-encoded cold half) with master's ea_combine as its tail (the 'gon' arm of takemyjob hopper/t128g-1029 c778f99, research/h100-t128g-1029.md), decided by an in-prepare race through kernel_fn on the JUDGE's own routing distribution: draw K >= 12 judge-like T=128 routings (the reference's generator with fresh seeds, as master's W6-M race does), time hybrid vs plain E5 interleaved per draw (>= 6 rounds), route the hybrid only if the MEAN over draws is <= 0.97 and no draw's median is > 1.03, and every draw is exact vs master. Otherwise T=128 stays plain E5. Print the per-draw ratios, mean, worst and PATH CHOSEN. Keep everything else unchanged (T=256 routes, E races, fork setup).

## What changed, and where
**Mechanism:** moe T=128 (verify_16) routes to the forked H=-16 hybrid 'gon' (wh5ds wg_kernel<G,14> hot half on the fork stream + E5 `{base}:e4` cold half + master's ea_combine tail) only when a [t128g] in-prepare race on 12 raw judge-like draws through kernel_fn gives mean gon/plain-E5 <= 0.97, no draw median > 1.03, all draws within the fp32 canary bound, and the copy check passes; else plain E5.

Worktree `muse_moe_w4a16/submission.py` only (+231/-1, 960,783 B < 1,000,000):
- `_hm_experts`: new tail branch on the prepare-set flag `state['hm']['eacmb'][T]`. It runs `ea['ext'].ea_combine(cache3, local_ids, tw, post_norm_w, out, eps, 1)` when `ea['cmb_t'][T]` is set, else `_post_norm_combine_kernel`, which is the same branch master's plain `_ea3_forward_core` takes. This is t128g c778f99's plain-E tail, keyed on its own flag instead of the gate.
- `_hm_setup`: records `hm['c128']` (the [hybrid-moe] canary verdict for T=128 H-16). **The old [hybrid-moe] T=128 gate is disabled** (`CAMPAIGN_T128G`): T=128 is skipped in the race loop with the reason "old [hybrid-moe] T=128 gate OFF (superseded by [t128g])". It shows in [hybrid-moe]'s PATH CHOSEN, in the [w6m-fork] Q1 line, and again in the [t128g] lines. T=256 is untouched.
- `_wh_phase`: keeps the `whk_x5` (wh5ds) bank when T=256 does not pick wh5ds. `_t128g_phase` pops it afterwards unless gon or T=256 uses it.
- `_prepare_main`: `_t128g_phase` runs last, after [ea-enc] (so plain = the final E5 + ea_combine route) and [hh-wh] (wh5ds bank built). This follows review ordering point 3.
- New `_t128g_phase` (end of file), which never raises. Any miss leaves T=128 on plain E5 and prints a reason. Steps:
  - **Preconditions:** the start deadline (230 s); the plain route is `<v4x s twin>:e4` (E5); the E module and E5 copies are present; the canary is clean; the wh5ds bank and ext are present; the fork stream exists (it is created if missing).
  - **Forced-cold control:** a synthetic routing with 16 routes per local expert, so none is hot. gon must equal plain **bitwise**. This is the review's goff == m check, done without a device gate.
  - **Race:** 12 raw `_eb_ref_ids(..., "normal")` draws, one fresh seed each (20261301+k, not order statistics). For each draw:
    - eager plain and gon outputs;
    - gon vs fp32 with `_rt_check` (half the judge's bounds) and <= 2x master's own rel-L2 (`_ea_canary`'s rule; gon is not bitwise vs plain, per the review);
    - wh5ds == master wg **bitwise** on the T=128 hot half (`_wh_exact`; hh-wh had proved it only at T=256);
    - graph == eager;
    - two graphs timed by `_rt_time`, 6 interleaved read-flushed rounds x 21 reps, with one draw's graphs alive at a time.
  - **Rule:** all draws within tolerance, err words clean (errx and whk), worst draw median <= 1.03, and mean of the medians <= 0.97. Then the copy check runs before 262 s (master's `_hm_copy_check`, which covers T=128 forked plus T=256).
  - **Cleanup:** gc and `empty_cache` before returning.
- **Prints:** one `[t128g]` tag, printed last in prepare (inside the 16 KB tail):
  - the old gate OFF, race start, the deadlines (230 / 255 / 262 / 300 s);
  - the arms: plain engine + tail, gon's hot kernel, cold engine, tail, and the fork stream pointer and priority;
  - the control result;
  - per draw: seed, touched experts, max routes, hot-expert count and hot routes, per-round ratios, median, and rel-L2 / worst row vs fp32 for gon and m;
  - the mean, the worst draw, tolerance, err words, the veto reason, the copy check;
  - `PATH CHOSEN T=128 = HYBRID gon (...) | PLAIN E5 (<route>; reason)`, with "old gate OFF" and the end time.
- **Deviations (the review wins):** "exact vs master" is replaced by the canary bound vs fp32 plus a bitwise forced-cold control. The prepare deadlines are 230 s start, 255 s mid-race, 262 s copy check.

**Prior (frontmatter `prior: none` is wrong):**
- CLAIMS:63 [refuted] "hybrid at T=128 on moe": H-16 and H-8 lost x0.94-0.95 on 10-28's judge runs.
- Master's own [hybrid-moe] T=128 gate on 702f1004 / 24f183ff: master 240.2 us vs H-16f 255.8 us on the judge, x1.065.

gon differs from both in four ways:
- the T=128 hot half is wh5ds, not the old wg;
- the cold half is E5, not v4cxs;
- the tail is ea_combine, not fused_post_expert_norm (about 2.3% on every node draw);
- the decision is a mean rule over 12 raw judge draws decided on the judge itself, not an every-round rule against a stale pre-E5 master.

The node-to-judge sign flip remains the main risk (review p ~0.25).

**Node validation** (gpunode06 GPU 5, nodejudge, bundle `njt128r` in `~/git/.takemyjob-wt/t128g`; configs `t128r-r1` / `t128rf-r1` = verify 16 + tests bs128-skewed, bs4):
- **`t128r` (the exact build, sha 7ffbac9947d8)** (`runs/20261029-171236-gpunode06`):
  - On the node, [ea-enc] kept T=128 on master (E/m best worst 1.001 > 0.98), so [t128g] printed `PATH CHOSEN T=128 = PLAIN E5 (v4cxs; plain T=128 route v4cxs is not E5 ...)` and correctly skipped.
  - Tests passed; prepare took 131 s.
- **`t128rf` (node-only copy, `EAE_BAR = 1.05`, which forces [ea-enc] to E5 so the race runs)** (`runs/20261029-171605-gpunode06`):
  - The control passed: gon == plain bitwise.
  - All 12 draws were within tolerance: gon rel-L2 equals master's, about 0.0039.
  - wh5ds == wg bitwise at T=128, graph == eager, and the err words were clean.
  - Mean gon/m was **0.913** (10/12 draws win); the worst was d6 at 1.016 (4 hot, 174 routes).
  - The copy check passed for T [128, 256], and the PATH CHOSEN was HYBRID gon.
  - The race ran from 137.8 s to 140.5 s (about 2.7 s). The verify 16 graph has 7 kernels (align, wg_gather, 2x wh_gemm, E5 K1/K2, ea_combine). Tests passed.
- **Judge caveat:** the route depends on [ea-enc] picking E5 at T=128 on the judge, as on 702f1004.

## Rules check (each must be true before the run)
Each box below was checked against `git -C .worktrees/muse_moe_w4a16/t128-hybrid-eacombine-judge-race diff`.

- [x] **CUDA-graph capturable.** The only kernel_fn-path change is the `_hm_experts` tail branch.
  - It reads the prepare-set flags `hm['eacmb'][T]` and `ea['cmb_t'][T]` (shape-keyed) and allocates `out` from the shapes.
  - It has no `.item()`, `.cpu()`, `synchronize()` or tensor print, and no control flow on input values.
  - Every sync and `.tolist()` in the diff is inside `_t128g_phase` / `_hm_setup` (prepare only).
- [x] **Nothing derived from `data` is cached across calls.**
  - The race uses synthetic in-prepare draws (`_eb_ref_ids` with fixed seeds). Only the T=128 route flags and the `c128` canary verdict stay in state.
  - Graphs and draw buffers are deleted per draw, followed by `gc` + `empty_cache` before prepare returns.
- [x] **Weights repacked losslessly in `prepare()` only.** No new repack. gon reads master's existing lossless copies:
  - the wh5ds `whk_x5` bank, now popped after the race instead of in `_wh_phase`;
  - the E5 `w13e4` / `w2e4` copies.
  - NVFP4 stays NVFP4.
- [x] **Activations BF16 where SGLang keeps BF16.** cache3 and out are bf16, and the combine accumulates in fp32 exactly as master's plain T=128 route does. KV cache / queries: n/a, since moe has no attention.
- [x] Attention sees every key: n/a (moe, no attention).
- [x] **No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.** The diff adds none. `_eb_ref_ids` is the submission's own generator.
- [x] **All work happens inside the timed call; no output caching or replay.** gon computes every route each call: hot on wgmma, cold on E5, then the combine.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
- Result: `OK with warnings`, rc 0, with 10 `warn dtype` and 202 `warn host-sync`.
- Master has 10 / 194. The 8 new host-sync warnings are all in `_t128g_phase`, each marked "outside kernel_fn's call graph; fine in prepare()".
- py_compile OK; __pycache__ removed.

## Risks / what could fail
high
The T=128 hybrid flipped sign node vs judge before (CLAIMS:63, H-16f x0.922-0.942 then refuted on the judge); the race on the judge's own draws is the safeguard. A worst draw > 1.03 vetoes.

## Result (after the run)
- job `4c5032bc57be4cf5` (check), status done, passed True, tests 5/5
- speedup 2.0421x, 95% CI [2.0114, 2.0727], score 6585.775939719688 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 2.035180868162415, "ci": [2.0046531551399784, 2.065708581184851], "job": "575820c989c8412b"}; base result: muse_moe_w4a16/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/t128-hybrid-eacombine-judge-race.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/t128-hybrid-eacombine-judge-race.patch`; branch `hyp/muse_moe_w4a16/t128-hybrid-eacombine-judge-race`
