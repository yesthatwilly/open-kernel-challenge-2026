---
problem: muse_moe_w4a16
slug: t128-hybrid-eacombine-judge-race
created: 2026-10-30T000244Z
base: master
mint_blob: "d80df5ecc71315466253dae86b28e4f6dfc41642"
best_at_mint: 2.0201
area: "moe T=128 expert block: forked hot/cold hybrid (wh5ds hot + E5 cold) with ea_combine tail vs plain Engine A E5"
change: >
  Route moe T=128 to the forked H-16 hybrid (wh5ds hot half + E5-encoded cold half) with master's
  ea_combine as its tail (the 'gon' arm of takemyjob hopper/t128g-1029 c778f99,
  research/h100-t128g-1029.md), decided by an in-prepare race through kernel_fn on the JUDGE's own
  routing distribution: draw K >= 12 judge-like T=128 routings (the reference's generator with
  fresh seeds, as master's W6-M race does), time hybrid vs plain E5 interleaved per draw (>= 6
  rounds), route the hybrid only if the MEAN over draws is <= 0.97 and no draw's median is > 1.03,
  and every draw is exact vs master. Otherwise T=128 stays plain E5. Print the per-draw ratios,
  mean, worst and PATH CHOSEN. Keep everything else unchanged (T=256 routes, E races, fork setup).
expected:
  - shapes: load=verify_16*
    direction: faster
    why: "node: gon mean 0.934 over 12 judge-shaped draws, 11/12 win; s2 p10/p50 lose x1.04-1.05"
  - shapes: load=decode_*
    direction: same
    why: T<=32 untouched
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: none
target: check
value_gain: 1.0
value_p: 0.35
value_ev: 0.35
value_cost: 300k tokens, 1 judge run
evidence: >
  H100 t128h/t128g (research/h100-t128h-1029.md, h100-t128g-1029.md; takemyjob hopper/t128g-1029
  c778f99): master plain E5 257-320 us per T=128 draw vs forked H-16 hybrid 242-296 us; hybrid +
  ea_combine tail mean 0.934 over 12 judge-shaped draws (11 win; worst 1.05x), all exact; moe
  board 2.0201 judged T=128 as plain E5 (702f1004 E5/m 0.955)
user_requested: true
review: proceed
status: running
review_at: 2026-10-30T000536Z
review_companions: null
review_best: 2.020119462295186
review_blob: d80df5ecc71315466253dae86b28e4f6dfc41642
stage: implemented
claimed_at: 2026-10-30T000536Z
base_sha: f03b88bbe6e9b548215edbe8af217b4a5c7b20da
base_blob: d80df5ecc71315466253dae86b28e4f6dfc41642
branch: hyp/muse_moe_w4a16/t128-hybrid-eacombine-judge-race
worktree: .worktrees/muse_moe_w4a16/t128-hybrid-eacombine-judge-race
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-30T001930Z
impl_commit: 3f8bfc172e2b894ac92bb59fcb4898a26ac6fbc1
submission_sha256: 7ffbac9947d86d1f5527b42a33e41c6a952dea97788beebd04beef69b7131179
norm_sha256: 6eb1945c7c6131a5c1194f79928736ce470463e777f5b714e205a2047277dd49
submission_blob: c49445a86f76dd17f8321c0f4b88d79ff7adfde2
lint_warnings: 212
---

# t128-hybrid-eacombine-judge-race

## Change

Route moe T=128 to the forked H-16 hybrid (wh5ds hot half + E5-encoded cold half) with master's ea_combine as its tail (the 'gon' arm of takemyjob hopper/t128g-1029 c778f99, research/h100-t128g-1029.md), decided by an in-prepare race through kernel_fn on the JUDGE's own routing distribution: draw K >= 12 judge-like T=128 routings (the reference's generator with fresh seeds, as master's W6-M race does), time hybrid vs plain E5 interleaved per draw (>= 6 rounds), route the hybrid only if the MEAN over draws is <= 0.97 and no draw's median is > 1.03, and every draw is exact vs master. Otherwise T=128 stays plain E5. Print the per-draw ratios, mean, worst and PATH CHOSEN. Keep everything else unchanged (T=256 routes, E races, fork setup).

## Expected

- `load=verify_16*` faster: node: gon mean 0.934 over 12 judge-shaped draws, 11/12 win; s2 p10/p50 lose x1.04-1.05
- `load=decode_*` same: T<=32 untouched

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 t128h/t128g (research/h100-t128h-1029.md, h100-t128g-1029.md; takemyjob hopper/t128g-1029 c778f99): master plain E5 257-320 us per T=128 draw vs forked H-16 hybrid 242-296 us; hybrid + ea_combine tail mean 0.934 over 12 judge-shaped draws (11 win; worst 1.05x), all exact; moe board 2.0201 judged T=128 as plain E5 (702f1004 E5/m 0.955)
- **Gain if it works:** 1.0% headline; **p** 0.35; **EV** 0.35% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.0% < 2.5%; EV 0.35% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge run.

## Risk

The T=128 hybrid flipped sign node vs judge before (CLAIMS:63, H-16f x0.922-0.942 then refuted on the judge); the race on the judge's own draws is the safeguard. A worst draw > 1.03 vetoes.

## Adversarial review

Reviewed 2026-10-30T000536Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with sharp build notes. The race is legal: it routes by shape class from synthetic in-prepare draws, and the hybrid with no gate never branches on input values. But the hypothesis requires 'exact vs master' on every draw, and gon is NOT bitwise vs plain E5 (its hot half runs on wgmma). Built as written, the race vetoes every time and the check reads as a false null. The gate must use the canary bound vs fp32.

### Justification
Mechanism. On the node (t128g, 2 replicates x 160 rounds), gon (wh5ds hot + E5 cold, forked, with the ea_combine tail) averages 0.934 vs plain E5, and 11 of the 12 judge-shaped draws win. The tail swap alone is worth about 2.3% on every draw (gon vs w16f). Losses are confined to draws with few hot routes: 0-28 routes, x1.00-1.05.

Weakest link: the node-to-judge transfer. The judge has already raced a T=128 forked hybrid in this exact prepare. On 702f1004 and 24f183ff, [hybrid-moe] measured master 240.2 us vs H-16f 255.8 us (x1.065 loss), while the node calls the comparable w16f a 0.956 win. That is CLAIMS:63's sign flip, reproduced on today's lineage. The judge's H-16f lacked all three of gon's changes (the T=128 hot half was still old wg, because hh-wh routes T=128 to master; the cold half was v4cxs, not E5; the tail was fused_post_expert_norm). Master's plain path has since gained E5 too (x0.955). So gon must recover about 6.5% plus E5's share on the judge just to break even. The 0.97 mean bar leaves only 3.6% headroom against the node's 0.934. p=0.35 is generous; about 0.25 is more honest. EV stays below the bar, but the mint is user-requested.

The draw rule is sound. Mean <= 0.97 with a per-draw veto at > 1.03 is exactly the rule t128g's reopen asks for ("the rule changes to a mean criterion"). One caveat: the node's worst judge-shaped draw (s2 p10) sat at 1.022-1.025, inside the A/A round range of 0.970-1.025, so the 1.03 veto can trip on noise. Use per-draw medians over >= 6 rounds, as written, not single rounds.

Prior: 'prior: none' is wrong. The closest attempts are CLAIMS:63 [refuted] 'hybrid at T=128 on moe' (the 10-28 runs lost x0.94-0.95) and master's own [hybrid-moe] T=128 gate. The differences are real (wh5ds hot, E5 cold, ea_combine tail, and a mean rule instead of the every-round rule), so this is not a retry. The md should cite both.

Bit-exactness of the tail. ea_combine is the same function master's plain T=128 route ends with, so the tail adds no numeric change. t128g proves goff == m bitwise: the gated route with every expert cold reproduces master bit for bit. gon itself is only within the canary bound (rel-L2 vs fp32 0.0039, the same as master's), because the hot experts' outputs come from wgmma, not Engine A's accumulation order. So 'every draw exact vs master' cannot hold for gon. The build must (a) check gon against the fp32 reference with _ea_canary's rule (finite, err word 0, rel-L2 and worst row within half the judge's bounds, no worse than 2x master's error) and (b) require bitwise equality only for a forced-cold control, if it keeps one.

Prepare cost. K = 12 draws x 6 rounds x 2 arms x ~21 reps is about 3k graph replays at ~0.3 ms each plus a 256 MB flush: roughly 1-3 s. Add 24 graph captures and 12 fp32 reference checks: roughly 3-8 s in total. That is cheap, but the window is tight. On the judge the [ea-enc] race finishes at about 153 s, and master's HM deadlines are start 250 / gate 255 / copy 262 s against the 300 s cap. The race must run AFTER the [ea-enc] and [hh-wh] decisions, so it compares against the real final plain route (E5 + ea_combine) and uses the wh5ds hot half at T=128. It needs a start deadline (around 230 s) that falls back to plain E5 and prints why. It must also keep one draw's graphs and data alive at a time.

### Confounders on the current kernel
- engine-a-a4b-a5b-encodings-race (bbd1661f; E5 at T=128): both arms carry E5, and the plain arm IS E5. If the race runs before [ea-enc] decides, it compares gon against a stale v4cxs master, which is exactly what 702f1004's [hybrid-moe] line did. Order it after.
- hothalf-wh-a5b-port (7da461db): currently routes wh5ds at T=256 only ('T=128 master'). gon needs wh5ds as the T=128 hot half, so the build must enable it for the hybrid's hot launches without changing the T=256 route. The gain depends on it.
- W6-M fork stream / v4x256c (_rt_fork_stream priority 0): gon is forked. Fork-stream behaviour on the judge was the suspect in CLAIMS:63's flip, and the T=256 hybrid shares the same stream object. Not separable in one run, but the per-draw print makes a judge-side loss readable.
- carrier CUDA combine at class 128 (81c9802e, EA_CMB_PIN 128): ea_combine is that combine. It is the same tail on both arms, so it is no confounder for the ratio, provided gon calls the same pinned combine variant as plain.

### Interference and A/B needs
No companion is warranted. The change touches only the T=128 route, and the decision is made and printed in prepare on the judge's own hardware. Every outcome can be read from one check: (a) PATH CHOSEN plain with the per-draw ratios printed is a clean transfer-failure reading, and the board is untouched; (b) PATH CHOSEN hybrid lets verify_16 be compared against 702f1004 / 24f183ff, and only T=128 changed. An ab-revert of E5 or wh5ds would answer a different question. The control the build needs is internal, not a separate run: keep master's plain E5 arm in the same interleaved race (and optionally the forced-cold goff for the bitwise and overhead check).

### Open questions
- The build must print the following, all under one tag (for example [t128g]):
- the prepare elapsed time at the race's start and end, and its deadline;
- what each arm actually runs: the plain arm's engine and tail (expect E5 + ea_combine, pinned), and gon's hot kernel (wh5ds wg_kernel<G,14>), cold engine (E5 variant), tail and fork stream (ptr, priority);
- per draw: its seed and label, touched experts, max routes, hot routes (experts with > 16 routes), the hot-expert count, the per-round gon/m ratios, the median, and rel-L2 and worst row vs fp32 for both gon and m;
- the mean over draws, the worst draw's median and which draw it was, and any veto reason;
- the copy-check result at T=128;
- 'PATH CHOSEN T=128 = HYBRID gon | PLAIN E5 (reason)';
- confirmation that the old [hybrid-moe] T=128 gate is either disabled or superseded, so two races cannot disagree.
- Does the hm gate's master arm in 702f1004 (240.2 us) predate E5 at T=128? If yes, the judge's last hybrid reading (x1.065) compared against a slower master, which makes the flip worse than it looks.
- Should K draws be EA_JUDGE_DRAWS-style order statistics (p10/p50/p90 of 15) or K independent reference draws? The node's 12 'judge-shaped' draws were p10/p50/p90 per seed. The judge's actual rows are single draws, so K raw draws estimate the expected headline better and include occasional low-hot draws.
- Is the T=128 hot half's wh5ds descriptor and split-commit setup valid at N tiles for T=128? hh-wh only proved and copy-checked it at T=256.


## Implementation notes

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

## Rules check

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
