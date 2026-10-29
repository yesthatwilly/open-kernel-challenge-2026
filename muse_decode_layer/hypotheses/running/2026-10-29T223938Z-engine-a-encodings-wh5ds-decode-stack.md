---
problem: muse_decode_layer
slug: engine-a-encodings-wh5ds-decode-stack
created: 2026-10-29T223938Z
base: master
mint_blob: "315119e26cb18e755adf23c3336af2c15c72aafc"
best_at_mint: 1.6904
area: "decode expert block: Engine A dequant (ea4x/v4x/ea6) at T=1..32 and the cold half, plus the wh5ds hot half"
change: >
  Port moe's two landed expert-block changes into decode as one stack, each routed by its own
  in-prepare exact race: (1) Engine A A4b/A5b encodings exactly as landed in
  muse_moe_w4a16/submission.py (engine-a-a4b-a5b-encodings-race, scored 24f183ff 2.0201): E4/E5
  arms on the v4x twins and v6 K1/K2 (T=1 K1 x K2 independent), per-T race through kernel_fn
  (decode's T classes: 1, 8, 16, 32 and the cold half at 128/256), pick only if exact (h, y,
  output torch.equal, err 0, tickets 0) and <= 0.98 vs master every round; E5 only if every scale
  byte is in [0x08,0x7E]; (2) the wh5ds hot half from decode's own hothalf-wh5ds-moe-port branch
  (check 35cc220c: routed both classes, layer veto .933-.952) with its isolated-pair pick + layer
  veto. Also pin decode's T=16 ea4x pooled race to the board's draw 's' (v4cxs) as a deterministic
  route (it flipped to 'd' in 44f30364 and 35cc220c, ~-0.38% headline). Keep everything else (fork
  guard, routing, PDL) unchanged. Size: decode is 955,413 B on master; use build.py strip_comments
  (AST-identical, CUDA strings byte-identical) if needed; report wc -c. Prints: [ea-enc] and
  [hh-wh] summaries as in the moe/decode builds, the [ea4x-dl] route line, all at the end of
  prepare.
expected:
  - shapes: load=decode_*
    direction: faster
    why: "moe judge 702f1004: Engine A E4 T1 0.967, T8 0.960, T16 0.953, T32 0.951 per class"
  - shapes: load=verify_16*
    direction: faster
    why: cold half E5 T128 0.955 + wh5ds T=128 layer .933-.940
  - shapes: load=verify_32*
    direction: faster
    why: cold half E4 T256 0.974 (iso 0.960) + wh5ds T=256 layer .936-.952
  - shapes: load=verify_1*
    direction: faster
    why: T=8 class E4 0.960
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "hothalf-wh5ds-moe-port: that ported only the hot half (flat headline, +0.38%); this adds the Engine A encodings that moved moe +4.55% on the judge, across every decode T class, plus the T=16 pin"
target: check
value_gain: 2.8
value_p: 0.55
value_ev: 1.54
value_cost: 500k tokens, 1 judge run
evidence: >
  moe judge 702f1004 2.0646 / scored 24f183ff 2.0201: Engine A encodings E/m 0.951-0.974 per T
  class, exact; decode 35cc220c 1.6968: wh5ds routed both classes, layer gate .933-.952; decode
  board 7cc034f9 anatomy: expert block 57 us (S d1/v1), 98 us (S d16/d32), 253 us (S v16/v32) =
  48-68% of the rows
review: proceed
status: running
review_at: 2026-10-29T224707Z
review_companions: null
review_best: 1.6903955680487197
review_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
stage: claimed
claimed_at: 2026-10-29T224707Z
base_sha: 7c345f539f26d261c6ced014d2cf6d7f85a06d22
base_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
branch: hyp/muse_decode_layer/engine-a-encodings-wh5ds-decode-stack
worktree: .worktrees/muse_decode_layer/engine-a-encodings-wh5ds-decode-stack
mode: null
restack_of: null
stale_base: null
---

# engine-a-encodings-wh5ds-decode-stack

## Change

Port moe's two landed expert-block changes into decode as one stack, each routed by its own in-prepare exact race: (1) Engine A A4b/A5b encodings exactly as landed in muse_moe_w4a16/submission.py (engine-a-a4b-a5b-encodings-race, scored 24f183ff 2.0201): E4/E5 arms on the v4x twins and v6 K1/K2 (T=1 K1 x K2 independent), per-T race through kernel_fn (decode's T classes: 1, 8, 16, 32 and the cold half at 128/256), pick only if exact (h, y, output torch.equal, err 0, tickets 0) and <= 0.98 vs master every round; E5 only if every scale byte is in [0x08,0x7E]; (2) the wh5ds hot half from decode's own hothalf-wh5ds-moe-port branch (check 35cc220c: routed both classes, layer veto .933-.952) with its isolated-pair pick + layer veto. Also pin decode's T=16 ea4x pooled race to the board's draw 's' (v4cxs) as a deterministic route (it flipped to 'd' in 44f30364 and 35cc220c, ~-0.38% headline). Keep everything else (fork guard, routing, PDL) unchanged. Size: decode is 955,413 B on master; use build.py strip_comments (AST-identical, CUDA strings byte-identical) if needed; report wc -c. Prints: [ea-enc] and [hh-wh] summaries as in the moe/decode builds, the [ea4x-dl] route line, all at the end of prepare.

## Expected

- `load=decode_*` faster: moe judge 702f1004: Engine A E4 T1 0.967, T8 0.960, T16 0.953, T32 0.951 per class
- `load=verify_16*` faster: cold half E5 T128 0.955 + wh5ds T=128 layer .933-.940
- `load=verify_32*` faster: cold half E4 T256 0.974 (iso 0.960) + wh5ds T=256 layer .936-.952
- `load=verify_1*` faster: T=8 class E4 0.960

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** moe judge 702f1004 2.0646 / scored 24f183ff 2.0201: Engine A encodings E/m 0.951-0.974 per T class, exact; decode 35cc220c 1.6968: wh5ds routed both classes, layer gate .933-.952; decode board 7cc034f9 anatomy: expert block 57 us (S d1/v1), 98 us (S d16/d32), 253 us (S v16/v32) = 48-68% of the rows
- **Gain if it works:** 2.8% headline; **p** 0.55; **EV** 1.54% (bar: gain >= 2.5%, EV >= 1.5%).
- **Cost:** 500k tokens, 1 judge run.

## Risk

The expert block is 48-68% of every sliding decode row (1749Z anatomy); moe's per-class E/m 0.95-0.97 on the judge applied to that share is ~+2-3% decode, plus the wh5ds rider (~+0.4-0.8%) and the T=16 pin (~+0.38% recovered vs the last two runs). The user, 22:40Z: stay focused on the expert block.

## Adversarial review

Reviewed 2026-10-29T224707Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: CLAIMS.md:100 [shelf] wh5ds names this exact stack as its reopen, and board/blob are unmoved (1.6904, 315119e2). The kernels transfer: decode's v4x module hash okc_campaign_engine_a_v4x_5a6b0753a8 and its ea6 regs equal moe's. Proceed only with two fixes. (1) Size: decode is already stripped, so strip_comments frees ~2 KB against a ~1,011 KB build; ~13 KB of real Python trimming is needed. (2) Picks: moe's '<= 0.98 every round through kernel_fn' bar was set on an expert-only kernel_fn. Decode's kernel_fn is the whole layer, which dilutes the gain, so pick on the isolated Engine A pair and use the layer only as a veto.

### Justification
A. The mechanism holds. E4/E5 change only the dequant instruction form and are bit-exact, and moe's judge run measured E/m per class on the same kernel bytes (702f1004 log: T1 E4/E4 .967, T8 .960, T16 .953, T32 .951, T128 E5 .955, T256 E4 .974 with cold iso .960; regs 82/83 and 64-70, lmem 0). Decode runs the same classes: T=1 v6pe2 (FAST), T=8 v4xs, T=32 v4cxs, and the cold half v4cxs at both T=128 and T=256. The board's 7cc034f9 log reads 'cold half ... T128 v4cx ... S | T256 v4cx ... S', and its v4x module hash equals moe's. Every moe anchor that _eae_src4x/_eae_src6 str.replace targets occurs with the same count in decode's source, so the CUDA port is mechanical. Two transfer gaps remain. (a) moe's T=128 was a plain Engine A route, not a hybrid: moe's _eae_base returns 'hybrid route' for T != 256, and moe's cold-half key is state['hm']['cold_x'][T]. Decode's T=128 and T=256 are H-16 hybrids that key state['hc']['cold_x'] by _hc_class, so the T=128 cold-half E path, and the T=256 one through hc, is new decode code that no run has exercised. moe's T=128 E5 .955 is a whole-block ratio and does not carry over; the right prior is moe's T=256 pattern (cold iso .960 -> kernel_fn .974). (b) Dilution, the WEAKEST LINK. moe's kernel_fn is the expert block. Decode's is the whole layer, with the expert block at 41-64% of the row. Projected layer E/m: T=1 ~.979, T=8 (verify_1) ~.983, T=16 ~.971, T=32 ~.975, T=128/256 cold-only ~.985-.99. Under a copied '<= 0.98 in every round' layer bar, T=8, T=128, T=256 and probably T=1 fall back to master. The stack would then deliver ~1% rather than 2.8%. Value estimate with per-row shares: E ~+1.8% headline if every class routes, plus wh5ds ~+0.4-0.8%. 35cc220c read +0.38% with T=16 on d, against a board that drew s. Total ~2.2-2.8%, so the 2.5% bar is reachable but not assured. Note that moe's own check overstated: +4.55% on the check, +2.29% on the scored run. Prior is correct: the earlier decode run (35cc220c) carried only wh5ds, and this adds a different mechanism. There is no do-not-retry overlap. T=16 pin: in the board's scored 7cc034f9, T=16 drew s ('16:s>d S1.016 pooled near 6r f2'). Its check 26435509, 44f30364 and 35cc220c drew d, and the layer race is a tie (d 171.4 vs s 171.6 us). The pin is also a prerequisite: E mirrors only s twins (moe _eae_base rejects EA4X_ARMS & 256), so with d drawn T=16 gets no E. The pin is wise. Pin the exact arm the board routed (v4xs or v4cxs, from 7cc034f9's [ead4x] lines), and update EAD4X_BOARD[16] from the stale 'd' to 's' so the '(board s/d/s)' print stops misleading.

### Confounders on the current kernel
- hothalf-wh5ds (CLAIMS.md:100, branch hothalf-wh5ds-moe-port): at T=128 the hot-half cut 'hides under the cold half' (LOG 2582), so E-cold and wh5ds interact super-additively. Neither alone shows its true effect. Run the E race AFTER _wh_phase has routed wh5ds; that is moe's order, where [ea-enc] came after [hh-wh]. Otherwise the T=128/256 E ratios are measured on the wrong hot half. wh5ds's veto gate ran with master cold and must not be reused as evidence for E.
- H-16 hybrid pins (CLAIMS.md:69, HC_PIN {128:-16, 256:-16}): H was tuned at master cold speed. A faster cold half may favour another H, so any gain is understated, but the A/B stays valid.
- v6 fork guard and the side stream (CLAIMS.md:95): the cold half runs forked at priority 0. Time the E race at T=128/256 in fork mode, as judged, with serial as a second veto, as the wh5ds gate did.
- v4c pinned at T=9..16 (CLAIMS.md:83): 'the layer prefers v4c though v4's block is 1.8-2.5% faster in isolation'. Isolated-pair ratios do not map 1:1 to the layer at T=16, which is why an isolated pick still needs a layer veto.
- L3/L4 paired-scale decode (CLAIMS.md:74, :78): E4 keeps it and E5 replaces it. Both are raced against master in-run and are bit-exact, so this does not confound the result.
- v4x-routes pooled hysteresis race (CLAIMS.md:95): the T=16 pin overrides it at T=16 only. T=8 and T=32 keep the race, and E mirrors whatever s twin they draw (both drew s in all four logs read).

### Interference and A/B needs
No companion is warranted. Each sub-change has an in-run control. E is raced per class against master with exactness gates, and the [ea-enc] ratios attribute it. wh5ds has its own isolated-pair pick and layer veto, and 35cc220c (1.6968, T=16 on d) is a wh5ds-alone judged control on the same base. The T=16 pin is deterministic and its effect shows in the [ea4x-dl] route line. A gain decomposes by class from the prints. A flat result with E routed reads as 'the H200 layer does not surface it'. A flat result with E unrouted reads as the bar or dilution problem above, so the prints must carry both the iso and the layer ratios. A revert of any landed hypothesis would answer nothing this run needs.

### Open questions
- SIZE (blocking for the build). decode master 955,413 B carries 177 comment chars and no docstrings: it is already stripped. The wh5ds branch is 988,094 B (~986,000 after stripping). moe's E delta after stripping is ~23.7 KB (948,099 vs 924,399, tokenize estimate). The decode hc-cold plumbing adds ~1-2 KB. That projects ~1,011,000-1,013,000 B, so ~13-15 KB must go. The hypothesis's 'strip_comments if needed' frees ~2 KB. Trim Python only, keep CUDA strings byte-identical, and prefer prepare-only race and diagnostic code of settled lanes. Candidates: wh5ds race arms D0/a3/wh, racing wh5ds vs m only while the module text stays identical; the [ea6-dl] inc(v1)-vs-v6 bench lines; superseded W6-G confirm-race code under HC_PIN; profiler and canary-only paths of b12p/c320d; moe-only branches in the E port such as Engine B/es1a checks, since decode has no EB_*. Run AST string-constant equality against master and report wc -c.
- RACE BAR. Pick each class on the isolated Engine A pair: K1+K2 graph at T=1/8/16/32, and the cold pair at T=128/256 forked against the routed wh5ds hot half, at <= 0.98 every round. Then require a layer kernel_fn veto (<= 1.00 every round; fork and serial at T=128/256). A copied whole-layer 0.98 bar would reject most classes. At T=1 restrict the 8 K1xK2 combos to m/m, E4/E4, E5/E5 and E4/m, which cuts false picks and race time.
- BUILD/PREPARE BUDGET. In 35cc220c, ea6-dl took 38.6 s, ea4x-dl 39.1 s and hh-wh 45 s in the background, the routes finished at 68 s, and prepare ended ~125 s. moe's two E modules took ~51 s each from t=0 with contention, and its race took 1.8 s. Expected decode end ~140-155 s with three extra background nvcc builds (eaenc-4x, eaenc-6, hh-wh). Watch for contention pushing ea6-dl/ea4x-dl past their own windows (EAD4X_START_S 90, EAD4X_GATE_S 100, EAD4X_CHECK_S 108; the v4x routes finished at ~68 s in 35cc220c). Start the E threads at t=0 but at lower priority (os.nice in the thread is not available, so consider starting them after ea4x-dl joins at ~68 s; they would still finish by ~125 s). Keep a hard EAE join deadline (moe 220 s) and the 255 s end, with master on timeout.
- WHAT THE BUILD MUST PRINT, as the last ~2.5 KB of prepare. The log tail holds ~14 KB and 35cc220c's was already cut at the head (111 lines, 14,044 B). Required lines: [ea-enc] build s per module, join wait, phase-at; regs/lmem/occ per E instance; recode MB; scale scan and E5 eligibility; per T (1 with K1/K2, 8, 16, 32, 128, 256): exact y/n, iso E/m median[min,max], layer E/m per round (s and f at 128/256); PATH CHOSEN per T with the fallback reason. Reprint [hh-wh] ROUTE, GATE and PATH, stating that the E race ran after wh5ds was routed. Reprint [ea4x-dl] route with '16:s PIN <arm>' and the cold-half line. Reprint [hotcold] PATH T=128/256 condensed. End with the final 'prepare at X s'. Drop [hh-wh] ATTRS/SCALES (the moe check proved them) to make room.
- Which s arm did 7cc034f9 route at T=16, v4xs or v4cxs? The pooled race spans EAD4X_RACE (v4xd, v4cxd, v4xs, v4cxs), and CLAIMS.md:83 pinned v4c at T=9..16. Pin exactly the board's arm.
- Expected per-row reading: the largest E effect should show on decode_16 and decode_32 sliding (~2.4-2.9%), and the smallest on full verify rows (~1%). If verify_16/32 gain more than ~2%, credit the wh5ds+E-cold critical-path interaction, not E alone.

