---
problem: muse_moe_w4a16
slug: hothalf-wh-a5b-port
created: 2026-10-29T195507Z
base: master
mint_blob: "691018b5844d1a985d98f59021313b0c1a3ef3b7"
best_at_mint: 1.9521
area: moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
change: >
  Port decode's routed wh hot half (module muse_decode_wg_wh from branch
  hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control
  V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for
  N>=48; background-thread build; isolated-pair route gate <= 0.97 vs master and D0 every round,
  exact, 16-replay) into moe's hot half (muse_moe_w4a16/submission.py wg_kernel at :867-1700,
  dispatched at T=128/256 by the hybrid). Add one more arm wh5ds = wh + A5b+D+SC from takemyjob
  hopper/hh3-1029 6a33dbf (A4b code re-encoding of the same permuted bank, 7-bit packed (b-8)
  scale fields, per-stage wgmma descriptor, split commit; landed on prefill as
  experts-rega-a5b-desc-splitcommit, scored 9f934846), raced as a candidate against wh: route
  wh5ds only if it also beats wh by <= 0.98 every round, else wh. Scale bytes scanned in
  [0x08,0x7E] (else master); banks from weights only. Keep moe's fork, routing, cold half and PDL
  unchanged. Print the [hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK lines (<= 1.5 KB) at the end of
  prepare. Size: moe is 964,183 B; the cap is 1,000,000 B (build.py strips comments above 990 KB):
  measure and trim dead diag text first if needed.
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "H100 hh3 moe judge draw forked: wh5ds x0.982 vs wh, wh ~-15.8 us; moe +0.92% (wh) -> +1.29% (wh5ds)"
  - shapes: load=verify_16*
    direction: faster
    why: "T=128 hot half: wh isolated pair .90 on the judge (decode 44f30364)"
  - shapes: load=decode_*
    direction: same
    why: T<=32 do not use the hot half
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: none
target: check
value_gain: 1.29
value_p: 0.5
value_ev: 0.65
value_cost: 350k tokens, 1 judge run
evidence: >
  decode judge 44f30364: wh routed T=128/256, pair .898-.902 / .838-.842 vs master, exact 16/16,
  profile hot pair x0.80 at verify_32 sliding; H100 hh2/hh3 (research/h100-hothalf2-1029.md,
  h100-hothalf3-1029.md): moe forked every wh variant ~-15.8 us (~+0.9%), wh5ds x0.982 vs wh on
  the moe draw -> +1.29% moe headline; A5b landed on prefill (9f934846 1.5721)
user_requested: true
review: proceed
status: queued
review_at: 2026-10-29T200135Z
review_companions: null
review_best: 1.952050927455536
review_blob: 691018b5844d1a985d98f59021313b0c1a3ef3b7
---

# hothalf-wh-a5b-port

## Change

Port decode's routed wh hot half (module muse_decode_wg_wh from branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wg_kernel<G,V> with D0 control V=0 and wh V=7 = A3 dequant on a word-permuted whole bank + exact-N tiles + R4 ring wait<2> for N>=48; background-thread build; isolated-pair route gate <= 0.97 vs master and D0 every round, exact, 16-replay) into moe's hot half (muse_moe_w4a16/submission.py wg_kernel at :867-1700, dispatched at T=128/256 by the hybrid). Add one more arm wh5ds = wh + A5b+D+SC from takemyjob hopper/hh3-1029 6a33dbf (A4b code re-encoding of the same permuted bank, 7-bit packed (b-8) scale fields, per-stage wgmma descriptor, split commit; landed on prefill as experts-rega-a5b-desc-splitcommit, scored 9f934846), raced as a candidate against wh: route wh5ds only if it also beats wh by <= 0.98 every round, else wh. Scale bytes scanned in [0x08,0x7E] (else master); banks from weights only. Keep moe's fork, routing, cold half and PDL unchanged. Print the [hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK lines (<= 1.5 KB) at the end of prepare. Size: moe is 964,183 B; the cap is 1,000,000 B (build.py strips comments above 990 KB): measure and trim dead diag text first if needed.

## Expected

- `load=verify_32*` faster: H100 hh3 moe judge draw forked: wh5ds x0.982 vs wh, wh ~-15.8 us; moe +0.92% (wh) -> +1.29% (wh5ds)
- `load=verify_16*` faster: T=128 hot half: wh isolated pair .90 on the judge (decode 44f30364)
- `load=decode_*` same: T<=32 do not use the hot half

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** decode judge 44f30364: wh routed T=128/256, pair .898-.902 / .838-.842 vs master, exact 16/16, profile hot pair x0.80 at verify_32 sliding; H100 hh2/hh3 (research/h100-hothalf2-1029.md, h100-hothalf3-1029.md): moe forked every wh variant ~-15.8 us (~+0.9%), wh5ds x0.982 vs wh on the moe draw -> +1.29% moe headline; A5b landed on prefill (9f934846 1.5721)
- **Gain if it works:** 1.29% headline; **p** 0.5; **EV** 0.65% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.29% < 2.5%; EV 0.65% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Decode's wh check 44f30364 read flat on the headline (1.6794 vs 1.6904) but its verify_32 sliding hot pair fell 153.5 -> 122.8 us (x0.80) and the row 427.7 -> 403.2 us; a single run cannot resolve ~1%. Read this one on the profile's wg_kernel<.,7|8> rows and the [hh-wh] ROUTE/FORK lines, not the headline alone.

## Adversarial review

Reviewed 2026-10-29T200135Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. The board is fresh and no landed or refuted claim owns this mechanism; the user asked for this below-bar, H100-derived run. Fix three things before the build: verify_16 should be expected flat, not faster (moe routes T=128 to Engine A, CLAIMS:63); the prior is not 'none'; and the moe-specific W6-M fork transfer must be measured inside the run, because only verify_32 can move and that is about +0.9-1.3%, which a single headline cannot resolve.

### Justification
Mechanism, in short: this is a bit-exact cut in dequant instruction count and latency inside moe's wgmma hot half, applied only where the hybrid routes it. On H100 that is [cyc]-transferable. hh2: jd-T256 isolated pair x0.876-0.888 vs master for wh and x0.859-0.862 for a3. hh3: wh5ds x0.855 pair, x0.938 forked vs dq0, and x0.982 forked vs wh. The SASS facts transfer (168 regs, 0 spill, no C75xx, DEPBAR.LE not serialized), and so does the K-step table (w13 N32 837 -> 720 cyc/stage).

(1) The parts that win on decode are mostly dead weight on moe. moe's judge draw at T=256 has 8 hot experts, 6 of them N32 chunks (hh2 §1). wh runs a3's stage loop at N16/N32, so the R4 ring never fires there. The 8 exact-N instantiations cost 2-6% at an unchanged N (hh2 §4: N32 845 vs 796). That is why moe's isolated pair reads wh x0.876-0.882, worse than a3 at x0.859-0.862. Forked they are equal (wh +0.91%, a3 +0.94-0.95%). hh2's own verdict was 'moe: A3 only'. On moe the operative content is A3 + A5b code/scale re-encoding + D (D carries the moe gain per hh3 §2: wh5 -> wh5d moves jd-T256 fork /wh from .992 to .982; SC adds nothing). The ported module already carries an a3 attribution arm. Let it compete in the route (pick the best qualifying of a3 / wh / wh5ds) instead of hard-wiring wh as the only base for wh5ds.

(2) The verify_16 expectation is wrong. moe's T=128 does not run the hybrid. The board profile shows only ea4x + combine at bs=128 (232.3 us). Every 10-28/29 log reads '[w6m-fork] PATH T=128 = master', with H-16f at worst x0.922-0.942 vs master (a5aee751, 2864f0f4). CLAIMS:63 refutes the T=128 hybrid on moe. A x0.90 hot pair under a cold half twice its size will not close a 6-8% race deficit. The change also says to keep routing unchanged. So verify_16 must be expected 'same', and it then serves as an in-run noise control. The value case is unaffected: hh2/hh3's moe +0.92% / +1.29% already came from jd-T256 alone.

(3) Weakest link: the hot cut has to land through moe's own W6-M fork, and no judge run has measured that. Decode's 44f30364 is not evidence for it. Decode runs CAMPAIGN_HCF=False, so its hot and cold halves share one stream and overlap only through PDL. Its T=256 cold:hot ratio is about 1.01, so the hot w2 ends the graph. moe forks the hot half onto a dedicated priority-0 stream (CLAIMS:55, 88/89) with a cold half about 1.9x the hot. Per CLAIMS:81 the ea4 cold stream sets verify_32's makespan (~255 us of the 291 us row). CLAIMS:94 shows the half-to-half overlap swings with residency order. Per CLAIMS:70, the cold half blocks the hot path for ~170 of ~342 us. hh2/hh3 modelled this with a calibrated spin stand-in that has no ALU or L2 contention, and calibrated R before v4x256 made the cold half faster. hh3 calls its forked ratios 'a judge-shaped model, not a judge measurement'. Expect verify_32 291.1 -> ~272-276 us (-15..-20 us; the row's share is 16.7%, so +0.9-1.3% headline). The wg_kernel row should drop from 84.1 us to ~68-73 us.

(4) The decode profile shows a single-draw hazard. Decode's verify_16 sliding hot pair read 69.6 -> 92.8 us in the 44f30364 profile, while the in-run race read .90. Read per-kernel profile rows only together with the in-run [hh-wh] lines.

(5) The prior is wrong. 'prior: none' should cite: decode hothalf-a3-exactn-ring-wh (44f30364, neutral 1.6794, the parent being ported); CLAIMS:104 (A3 hot-half lead on moe; this is its first judge check); and CLAIMS:54 (W5 exact-N 80/96/112 tiles refuted on moe, <= x1.006). The difference from CLAIMS:54 is real: exact N only pays once A3 plus the ring lower the per-stage dequant cost (hh2 N80 1063 vs 1294 cyc). On moe's N32-dominated draw, though, nx is expected to contribute nothing. Say so instead of claiming it as part of the gain.

(6) The H100 lanes' own pre-registered rules argued against a moe check: hh2 said 'on moe ship A3 alone', and hh3's band > 0.98 meant no follow-up check. This run goes ahead on the user's request and today's steers, not on the H100 lanes' decision rules. The attempt file should say this.

### Confounders on the current kernel
- W6-M fork (CLAIMS:55) + v4x256c priority-0 fork stream (CLAIMS:88/89): verify_32's gain exists only as far as a shorter hot half lets the cold ea4x half finish earlier. A residency-order change (CLAIMS:94) can absorb the cut or invert it. The hypothesis does not touch the fork, so attribution needs an in-run forked kernel_fn A/B (master vs wh vs wh5ds hot half, same H-16f fork), not decode's evidence.
- v4x256 / v4cxs cold half (CLAIMS:85, 88): the cold stream sets the makespan (CLAIMS:81, ~255 us). The cold:hot ratio decides how much of the hot cut lands. hh2's stand-in R=1.91 predates v4x, and a faster cold half makes R smaller, so more of the cut should land. That direction is favourable, but nobody has measured it on the judge.
- X2 hybrid H-16 threshold (CLAIMS:44): it fixes which experts are hot and their N distribution (8 hot, mostly N32 chunks at T=256). That is why the ring and nx are idle on moe and only A3 + x5 + D act. A draw with more N>=64 hot chunks would favour wh. The [hh-wh] ROUTE line must print the per-routing N histogram.
- T=128 hybrid refuted (CLAIMS:63): verify_16 runs Engine A v4cxs + combine (stack2 / carrier, CLAIMS:69, 97), so wh cannot move it unless the wh phase runs before _hm_setup's race and flips H-16f past master. Run the wh phase after the W6-M decisions, as decode did after '[w6m-fork] end', so verify_16 stays an untouched control.
- S1a / H-116 arms (CLAIMS:62) and the H-8 arms race in the same hybrid race. If wh is installed before the race, every wg-hot arm's timing changes and the T=256 pick could move (e.g. H-8f, which reads x1.005-1.008 vs master). Installing wh after the race avoids this.
- master's non-hybrid _moe_wgmma path (_wg_gemm at :1993/:1998 for T in 128..512 that is not exactly 128/256, e.g. the bs=257 test): if wh replaces wg_kernel globally rather than at the hybrid call site (:14080), untimed tests run the new kernels on routings the canary never saw.

### Interference and A/B needs
No companion run. The change is bit-exact and routed in prepare against an in-run D0 control and master on the judge's H200. A second check would only add the ±1.2% identical-bytes headline spread, and it could not isolate the fork effect any better than an in-prepare A/B. Neither a revert of W6-M nor a serial-only arm needs its own judge run. Both questions are answered more precisely by the prepare-time races, provided the build times the production shape: kernel_fn at T=256 under the routed H-16f fork, master hot half vs wh vs wh5ds, forked and serial, per round on 2+ judge-like routings. Also time a3 / a3+D for attribution (free bytes; the module already has a3). The untouched rows (decode_*, verify_1, verify_16) are the run's own noise controls. Read the result in this order: the in-run FORK ratios first, then the verify_32 row (291.1 -> expect 272-276 us; anything above ~285 us is inside row noise), then the wg_kernel<G,V> profile rows, and the headline last.

### Open questions
- Phase order: does the wh phase run after _hm_setup's W6-M race and fork gate (recommended; T=128 stays master and verify_16 is a control) or before it (it then re-races every wg-hot H at T=128/256)? The build must say which, and print the resulting [w6m-fork] PATH T=128 / T=256 lines again inside the [hh-wh] block. The judge keeps only a ~16,000-char log tail, and the board's prepare log is already ~15,022 B.
- Route gate basis: decode gated wh on the isolated pair (<= 0.97 vs master and D0). For moe, add a no-loss condition on the forked kernel_fn T=256 graph (wh or wh5ds vs master <= 1.00 every round), so an arm that wins isolated but loses under the fork (residency order, CLAIMS:94) cannot route. wh5ds's '<= 0.98 vs wh every round' should be on the isolated pair (hh3 moe pair .970). On the fork (.982) it would almost never pass.
- Should the route consider a3 (and a3+x5+D) directly? On moe's draw a3 beats wh isolated (.859-.862 vs .876-.882) and ties it forked. The module already contains a3.
- Memory: master's wg bank plus a permuted copy plus an x5 copy (decode scanned 100,663,296 scale bytes, so each copy is roughly 0.9 GB). Are the copies shared across the judge's state copies, or deep-copied by the two-state copy check?
- What the build must print (<= 1.5 KB, at the end of prepare): [hh-wh] BUILD (s, bg start, join wait, -Werror/C75xx clean, prepare at); ATTRS (regs/spill/smem per arm D0/a3/wh/wh5ds, w13 and w2; 168/0 required); SCALES (count in range, permutation and x5 inverse self-test PASS); ROUTE T=256 (per-round pair ratios wh/m, wh/D0, a3/D0, wh5ds/wh, D0/m; hot-pair us; hot expert count + N-chunk histogram per routing; exact routings; replay 16/16); FORK T=256 (kernel_fn graph us master/wh/wh5ds forked and serial, per round, fork/serial; hot vs cold half anatomy us); the re-printed [w6m-fork] PATH T=128/256; PATH CHOSEN naming the profile symbol (wg_kernel<G,V>, V per arm). The attempt file must give the uploaded byte count.
- Read and kill rule for the single check: proceed to a scored run only if the headline beats 1.9521, an arm routed, and verify_32 <= ~280 us. If the in-run forked kernel_fn T=256 ratio vs master is > 0.98, the moe fork absorbs the hot cut: mark the mechanism [refuted] for moe and leave it to ride the next verify_32 candidate.

