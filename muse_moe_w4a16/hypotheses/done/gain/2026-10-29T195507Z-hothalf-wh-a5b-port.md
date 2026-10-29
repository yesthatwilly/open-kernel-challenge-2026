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
status: done
review_at: 2026-10-29T200135Z
review_companions: null
review_best: 1.952050927455536
review_blob: 691018b5844d1a985d98f59021313b0c1a3ef3b7
stage: harvested
claimed_at: 2026-10-29T200135Z
base_sha: db626b766d6eb2e9ef1ad995ca86f6aac6305898
base_blob: 691018b5844d1a985d98f59021313b0c1a3ef3b7
branch: hyp/muse_moe_w4a16/hothalf-wh-a5b-port
worktree: .worktrees/muse_moe_w4a16/hothalf-wh-a5b-port
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T202008Z
impl_commit: 9c58da7dd160f0918551ab3cc5e91500af556196
submission_sha256: 8c4b56a992bd04711d50092c01d617ff0f2530dedf594d4ec9dd17c79a63a148
norm_sha256: 1c704f3aef43f960f594a6296f8a2dd83b4863ae54b0cee53b5393ad84e0165c
submission_blob: 37ea90fc306c0ed8fb3d49094bf5f4acf6202307
lint_warnings: 202
attempt_file: 2026-10-29T202008Z-check-hothalf-wh-a5b-port.md
prepared_at: 2026-10-29T202008Z
job: 54b70249868f468b
dispatched_at: 2026-10-29T202018Z
outcome: gain
verdict_detail: gain
speedup: 1.9826789715234687
speedup_ci: [1.9529387869506167, 2.0124191560963207]
harvested_at: 2026-10-29T202409Z
best_at_harvest: 1.952050927455536
promotion: merged
mechanism: null
promotion_commit: 7da461db12b3c802a605fcdcaf2208314fe6dc60
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


## Implementation notes

**Mechanism:** moe's T=256 hybrid wgmma hot half (wg_kernel w13+w2 at _hm_experts' wg-hot call site, verify_32) re-routed in prepare to the best bit-exact arm of a new module (a3 = A3 dequant on a word-permuted bank, wh = A3 + exact-N + R4 ring, wh5ds = wh + A5b codes/packed scales + D + SC), gated on the isolated pair and the routed forked kernel_fn.

What changed (worktree `muse_moe_w4a16/submission.py` only, base db626b76, blob 691018b5):
- New section before the es_core include (`# ---- hothalf-wh-a5b-port`): module `muse_moe_wg_wh_<sha10>` derived from master's `_WG_SRC_HEAD/_WG_SRC_BODY` by decode's asserted edits (moe's wg text is byte-identical to decode's base, checked). `wg_kernel<G, V>`, V bits 1 = A3, 2 = exact N, 4 = R4 ring wait<2> for N >= 48, 8 = A5b+D+SC. Arms: D0 = 0 (master's text in this TU, the control), a3 = 1, wh = 7, wh5ds = 14 (exact N + ring with D for N >= 48, the SC+D stage loop below N 48, x5 dequant step everywhere; hh3's `hh_wh5ds.cuh` shape). The ring keeps decode's 64-bit descriptors for V & 8 == 0, so wh is the same code as decode's 44f30364 module.
- Banks, from the weights' wg bank only: the scale bytes are scanned in [0x08, 0x7E] (else OFF -> master); a3/wh read an A3 word-permuted copy (decode's `_wh_perm` map), wh5ds reads an A5b copy (ENC4 code map, prefill's `REGA_DQ_ENC4`, plus the packed 7-bit b-8 scale words). Each chunk is checked lossless by the inverse. A host self-test also checks, on 2^16 random words and all 119^2 scale pairs, that A5b's four code views of ENC4(q) equal A3's ext_prop of perm(q) and that the LOP3 scale equals A3's prmt*16 + 0x77807780 (the same model passed locally). Only the routed arm's copy is kept (~0.86 GB).
- Dispatch: only `_hm_experts`' wg-hot branch (serial and W6-M fork) calls `_wh_pair(state, wb, state['whk'].get(T), ...)`. `state['whk']` is a prepare-set {256: V or None}. `_moe_wgmma` (the bs=257 test path, T=128..512 non-hybrid) is untouched.
- Build: `prepare()` is now a wrapper. It starts a non-daemon thread (nice +10; ptxas `-Werror` probe first, C75xx grepped; plain rebuild if time allows) before es1a/ea6/ea4x/EPLF start, then `_prepare_main` runs master's prepare unchanged, and `finally` joins the thread. TORCH_CUDA_ARCH_LIST is set to 9.0a, as es1a/ea6 already leave it, and is not restored (master behaviour).
- Phase order (review open question 1): `_wh_phase` runs after `_hm_setup` (W6-M race, fork gate, v4x-t256, copy check) and `_eb_setup`. T=128 is never touched (`whk` has only key 256), so verify_16 is a noise control. The phase needs T=256 routed to a wg-hot H, else OFF.
- Route at T=256: exactness first (torch.equal of hs and output vs master's dispatched launches on the 3 judge-style draws `_ea_judge_routings` p10/p50/p90 + 4 synthetic all-N routings; err word [0, magic]). Then the isolated pair (gather once; graphs of w13+w2 per arm, `_rt_time`, 3 draws x 2 rounds = 6 rounds). An arm qualifies when it is exact, D0/m in 0.97..1.03, arm/m <= 0.97 and arm/D0 <= 0.97 in every round, and, for wh5ds only, wh5ds/wh <= 0.98 in every round (isolated, per the review). Among qualifiers the best mean pair goes first and needs replay 16/16 (two scratch sets, 8 refills alternating judge draws and synthetic routings, bitwise vs master's eager pair). Then the production-fork condition: kernel_fn T=256 graphs (routed H, fork as routed; forked and serial for m/a3/wh/wh5ds on the same 3 draws x 2 rounds) must give pick/m <= 1.00 in every round, else master. Last, `_hm_copy_check` with the pick routed; a FAIL goes to master. Every step starts before 240 s and the copy check before HM_COPY_DEADLINE_S, else master.
- Prints: at the very end of prepare (after the [ea6] lines), at most ~1.5 KB: `[hh-wh] BUILD` (s, bg start, join wait, -Werror/C75xx, phase start s), `ATTRS` (regs/spill per arm, w13/w2; 168 regs / 0 local required, else OFF), `SCALES` (count in range, copies lossless, self-test), `ROUTE T=256` (per-round wh/m, wh/D0, a3/D0, wh5ds/wh; D0/m range; pair us per arm; hot experts + N-chunk histogram per draw; exact; replay), `FORK T=256` (kfn us f/s per arm; per-round pick-mode ratios vs m; serial worst; m f/s; hot pair vs kfn serial anatomy), the re-printed `[w6m-fork] PATH T=128/256` lines, and `PATH CHOSEN T=256 <arm> wg_kernel<G,V>` (profile symbol) or `master wg_kernel<G> (why)`.

Deviations / review fixes:
- Review (1): a3 competes in the route. wh5ds is not hard-wired on wh; it is one of three candidates, plus the wh-relative bar.
- Review (2): verify_16 is expected `same` (in-run noise control). T=128 stays master by construction.
- Review open question 2: wh5ds's 0.98 is on the isolated pair; the fork gate is a no-loss gate (<= 1.00) on the routed kernel_fn.
- decode's a3nx attribution arm was dropped (size; no route use).
- The spec text says the A4b codes re-encode "the same permuted bank". hh3 (`k_recode_bank(bk.q13, ...)`) and prefill (`_rega_dq_a5b_bank`) both encode master's words, so wh5ds's bank is ENC4 of master's words, a second copy.
- SIZE: the base was 964,183 B. Trimmed dead diagnostics: `_ea_kernel_rates` and `_combo_f2_select_diag` (both only reachable under `not CAMPAIGN_LEAN_DIAG`, which is True) with their two call blocks, and the unreferenced `_ea_ref_block` (~5.2 KB). **Uploaded file: 993,167 B (`wc -c`) < 1,000,000.**
- Memory (review open question 4): the routed copy is a tensor in the state, so the judge's tensor-cloning copy (modelled by `_wg_copy_tree` in the copy check) clones it like master's other banks.

Prior (frontmatter `prior: none` is wrong; corrected here): decode hothalf-a3-exactn-ring-wh (check 44f30364, neutral 1.6794, the parent being ported); moe CLAIMS.md:104 (A3 hot-half [lead]; this is its first judge check); CLAIMS.md:54 (W5 exact-N 80/96/112 tiles refuted on moe, <= x1.006). On moe's N32-dominated T=256 draw, the exact-N and ring parts are expected to add nothing; the operative content is A3 (+ A5b + D).

Both H100 lanes' pre-registered rules argued against a moe check: hh2 said "on moe ship A3 alone", and hh3's band (> 0.98 forked vs wh) meant no follow-up check. This run goes ahead on the user's request (user_requested: true) and today's steers, not on the H100 lanes' decision rules.

Read/kill rule (review): read the in-run FORK ratios first, then verify_32 (291.1 -> expect 272-276 us; above ~285 us is inside row noise), then the `wg_kernel<G,V>` profile rows, then the headline. If the forked kernel_fn T=256 ratio vs master is > 0.98, the moe fork absorbs the hot cut: mark the mechanism [refuted] for moe.

Node validation (H100 gpunode06 GPU 1; nodejudge config `hhwh-moe`, 1 cold isolated prepare of these exact bytes, sha256 8c4b56a992bd, no tests, no timed shapes; takemyjob run `runs/20261029-131634-gpunode06`, 20:16-20:19Z). passed=True, prepare 125.7 s / 300 (peak 18.0 GiB alloc), `[cs-wh] whk {256: 14}`, hm on {128: 0, 256: -16} fon {256: True}. The [hh-wh] end block is 1,333 B:
```
[hh-wh] BUILD 42s bg from 0.0s, join waited 0s at 123s; ptxas -Werror clean, C75xx none
[hh-wh] ATTRS regs/spill w13 D0 r168/l0 a3 r168/l0 wh r168/l0 wh5ds r168/l0; w2 D0 r168/l0 a3 r168/l0 wh r168/l0 wh5ds r168/l0
[hh-wh] SCALES 100663296 in 0x08..0x7E; A3 perm + A5b ENC4/pack copies lossless, self-test PASS (0.7s)
[hh-wh] ROUTE T=256 pick wh5ds: wh/m .875/.874/.861/.859/.893/.893 wh/D0 .880/.877/.858/.859/.891/.891 a3/D0 .874/.874/.882/.884/.896/.896 wh5ds/wh .979/.979/.942/.944/.960/.961 D0/m 0.995..1.003; pair us m 115.6 D0 115.7 a3 102.5 wh 101.2 wh5ds 96.9; hot:N 4:32x2,48x2 7:32x5,80x1,96x1 7:32x5,64x1,128x2; exact 7/7; replay wh5ds 16/16 exact
[hh-wh] FORK T=256 kfn us f/s m 318.6/343.9 a3 308.0/331.0 wh 304.5/329.5 wh5ds 300.4/324.8; f/m a3 .974/.973/.966/.966/.960/.961 wh .975/.975/.933/.934/.957/.959 wh5ds .965/.964/.925/.925/.937/.940; s/m worst a3 0.977 wh 0.975 wh5ds 0.973; m f/s 0.889..0.955; hot pair m 115.6 vs kfn serial 343.9 (cold+rest 228.3)
[hh-wh] PATH CHOSEN T=256 wh5ds wg_kernel<G,14> (copy PASS), T=128 master (2.4s, prepare at 125.3s)
```
Read: wh5ds routed on H100 (isolated pair x0.84 vs master; forked kernel_fn x0.925-0.965, about -18 us on the routed T=256 graph). The forked ratio is below the review's 0.98 kill line on this node. On H100, wh beats a3 isolated on 4 of 6 rounds, unlike hh2's moe draw. The phase took 2.4 s and the build was hidden behind the other builds (join wait 0 s). The judge's H200 decides the transfer.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/hothalf-wh-a5b-port diff` (12 hunks: `_wg_wgmma_src(nbs)`, the prepare wrapper, the phase call, the final prints, 3 dead-diag removals + their 2 call blocks, the 2 `_hm_experts` call-site lines, the new section).
- [x] CUDA-graph capturable: kernel_fn's only new code is `_wh_pair(...)` at `_hm_experts`' wg-hot branch. It reads a prepare-set int from `state['whk']` keyed by `num_tokens` (shape), then makes two `wh_gemm` launches that encode tensor maps per call in C++, as master's `wg_gemm` does. No `.item()`/`.cpu()`/sync/print. No allocation or branch on input values. The routed graph was captured and replayed in prepare (replay check, fork race, copy check).
- [x] Nothing derived from `data` is cached across calls: `whk`, `whk_bank`/`whk_x5`, `whk_ext`, `whk_err` are set in prepare from the weights only; no input-derived tensor is stored.
- [x] Weights repacked losslessly in `prepare()` only: the A3 copy is a bit permutation of each 32-bit NVFP4 code word (scales untouched). The A5b copy is the ENC4 bit permutation plus the scale word packed as 7-bit b-8 fields, which is lossless for bytes in [0x08, 0x7E]: scanned first, and each chunk is checked by the inverse. Still NVFP4 codes and E4M3 scales, dequantized in-kernel.
- [x] Activations BF16 where SGLang keeps BF16: every arm is bit-exact to master's hot pair (torch.equal of h and y; required to route). The dequant arithmetic is the same bf16x2 fma per element.
- [x] Attention sees every key: n/a (MoE expert block, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The build is `torch.utils.cpp_extension.load_inline` in a thread, as master's es1a/ea6/ea4x builds are; `os.setpriority` lowers the build thread's nice. No subprocess/multiprocessing/os.fork in the diff.
- [x] All work happens inside the timed call; no output caching or replay: the hot half still computes w13+silu and w2 per call. Prepare-time graphs are dropped after timing.
- [x] Size: `wc -c` = 993,167 B < 1,000,000 (CUDA comments included; no build.py strip needed).
- [x] First line kept: `# okc: problem=muse_moe_w4a16`. `python3 -I -m py_compile` OK, `__pycache__` removed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
... (pre-existing prepare-side host-sync / dtype warnings unchanged) ...
warn  host-sync    L15495: loc = ids.long().flatten().cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15496: cnt = torch.bincount(loc[(loc >= 0) & (loc < 32)], minlength=32).tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15551: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15553: e = [int(t) for t in sb["whk_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15600: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15646: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15660: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15661: e = [int(v) for v in state["whk_err"].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L15733: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/hothalf-wh-a5b-port/muse_moe_w4a16/submission.py
```
(rc 0; every warning is prepare-side; the new ones are the `_wh_*` prepare functions above.)

## Result

- job `54b70249868f468b` (check), status done, passed True, tests 5/5
- speedup 1.9827x, 95% CI [1.9529, 2.0124], score 6782.994218003089 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.952050927455536, "ci": [1.922770163543703, 1.9813316913673689], "job": "8418677e467d4fb7"}; base result: muse_moe_w4a16/hypotheses/BEST.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/hothalf-wh-a5b-port.result.json`, `muse_moe_w4a16/hypotheses/done/gain/hothalf-wh-a5b-port.patch`; branch `hyp/muse_moe_w4a16/hothalf-wh-a5b-port`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 27.34 | 27.65 | 0.9874 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 111.48 | 104.06 | 1.011 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 140.17 | 140.41 | 1.0011 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 81.18 | 75.15 | 1.0131 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 232.32 | 230.0 | 1.0157 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 291.06 | 274.38 | 1.0676 | gained |

gained on verify_32 verify_32. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- candidate for promotion; stack further changes with `depends_on` once it lands.

