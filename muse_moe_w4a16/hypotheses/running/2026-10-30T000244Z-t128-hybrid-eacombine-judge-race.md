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
stage: claimed
claimed_at: 2026-10-30T000536Z
base_sha: f03b88bbe6e9b548215edbe8af217b4a5c7b20da
base_blob: d80df5ecc71315466253dae86b28e4f6dfc41642
branch: hyp/muse_moe_w4a16/t128-hybrid-eacombine-judge-race
worktree: .worktrees/muse_moe_w4a16/t128-hybrid-eacombine-judge-race
mode: null
restack_of: null
stale_base: null
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

