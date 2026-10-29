# muse_decode_layer · 2026-10-29T231037Z · submit · engine-a-encodings-wh5ds-decode-stack

Hypothesis: `muse_decode_layer/hypotheses/done/gain/2026-10-29T223938Z-engine-a-encodings-wh5ds-decode-stack.md` (branch `hyp/muse_decode_layer/engine-a-encodings-wh5ds-decode-stack`).

- **Job:** `e42339989d554019` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `60932212`

## Goal
Promote check-proven gain engine-a-encodings-wh5ds-decode-stack (1.7197x [1.6939,1.7463], check job 824b60fac87c4ff4) to the board.

Port moe's two landed expert-block changes into decode as one stack, each routed by its own in-prepare exact race: (1) Engine A A4b/A5b encodings exactly as landed in muse_moe_w4a16/submission.py (engine-a-a4b-a5b-encodings-race, scored 24f183ff 2.0201): E4/E5 arms on the v4x twins and v6 K1/K2 (T=1 K1 x K2 independent), per-T race through kernel_fn (decode's T classes: 1, 8, 16, 32 and the cold half at 128/256), pick only if exact (h, y, output torch.equal, err 0, tickets 0) and <= 0.98 vs master every round; E5 only if every scale byte is in [0x08,0x7E]; (2) the wh5ds hot half from decode's own hothalf-wh5ds-moe-port branch (check 35cc220c: routed both classes, layer veto .933-.952) with its isolated-pair pick + layer veto. Also pin decode's T=16 ea4x pooled race to the board's draw 's' (v4cxs) as a deterministic route (it flipped to 'd' in 44f30364 and 35cc220c, ~-0.38% headline). Keep everything else (fork guard, routing, PDL) unchanged. Size: decode is 955,413 B on master; use build.py strip_comments (AST-identical, CUDA strings byte-identical) if needed; report wc -c. Prints: [ea-enc] and [hh-wh] summaries as in the moe/decode builds, the [ea4x-dl] route line, all at the end of prepare.

## What changed, and where
**Mechanism:** Engine A A4b/A5b dequant encodings (E4/E5) on decode's v4x s twins (T=8/16/32), FAST v6 K1/K2 (T=1) and the H-16 hybrid cold half (T=128/256), plus the wh5ds wgmma hot half (T=128/256) and a T=16 v4cxs pin. Each is picked by an in-prepare exact race.

- **Base.** The worktree file is decode's `hothalf-wh5ds-moe-port` (35cc220c) taken whole. Master's decode submission is unchanged since that branch's base. The moe E port follows on top.
- **E modules.** `_EAE_COMMON/_EAE_CONSUME/_EAE_M16`, `_eae_rep/_eae_dev/_eae_src4x/_eae_src6`, `_eae_recode`, `_eae_pair` and `_eae_xcore` are copied verbatim from muse_moe_w4a16 master, with docstrings and comments stripped. The generated E sources are byte-identical to moe's: e4x sha d78b67ee4b65, e6 sha ea2a55ed5ec3. The v4x module name is still okc_campaign_engine_a_v4x_5a6b0753a8, and the wh module hash equals the branch's.
- **Routing in the kernel path:**
  - `_ead_core` looks up `_EAE_CORES` first.
  - `_ead6_forward_core` gained `enc=(K1,K2)`, as in moe. K1/K2 come from `ea['ext6e']` on the E copies with flag 4096 = E5. The early combine stays on master's ext6.
  - The `_hc_experts` cold branch now calls `_eae_pair(..., hc['cold_e'][_hc_class(T)])`. With dq 0 it makes the same launches as master.
- **Keys.** T=1/8/16/32 are keyed by exact T (`ea['on'][T] = base:eN`), as in moe. The cold half is keyed by `_hc_class`, the same way as `cold_x`. Before routing, each class is canaried at 144/176 (class 128) and 200/240 (class 256): exact, err and tickets 0.
- **Order and race (review fixes 2 and 3):**
  - E module threads start after `_ead_setup` returns, so after the ea4x-dl race (~68 s). Join deadline 220 s. No class starts after 240 s, and the copy check is skipped after 248 s, which leaves master.
  - The E phase runs after `_wh_phase` and the fork gate.
  - Each class is picked on the isolated Engine A block: the core graph at T=1/8/16/32, and at T=128/256 the `_hc_experts` hybrid block (cold E, forked per state, against the routed wh5ds hot half). The bar is ≤0.98 in all 10 rounds (2 draws x 5) and exact: h/y on NaN-poisoned buffers, block output and kernel_fn's 3 outputs all `torch.equal`, err and tickets 0.
  - A layer `kernel_fn` veto follows at ≤1.00 in every round (fork + serial at 128/256).
  - T=1 arms are E4/E4, E5/E5 and E4/m. E5 runs only if every scale byte is in [0x08,0x7E].
  - Last, a state-copy check on the picks (`_ead_copy_one` / `_hc_copy_check`). Unrouted E copies are freed.
- **T=16 pin.** 7cc034f9 routed **v4cxs** at T=16 (anatomy line `T=16 (v4cxs)`). `EAD4X_PIN={16:'v4cxs'}` skips the pooled race at T=16. The layer and copy checks still run, and the route prints `16:s PIN v4cxs ...`. `EAD4X_BOARD[16]` is now 's'.
- **Trims (Python only).** These are needed to stay under 1,000,000 B:
  - The W6-G `_hcg_confirm` family, about 9.6 KB. It was a no-op when HC_PIN routes H-16; it now prints nothing.
  - The [ea6-dl] inc-vs-v6 block bench (`_ead6_block`, `_ea6_slopes`, `EA6_FLOOR`). v6 vs incumbent is now decided by the whole-layer guard alone.
  - The wh5ds race now races wh5ds against m only. Removed: the D0/a3/wh arms, the D0 fidelity check, the wh fallback, the A3 bank copy and `_wh_selftest`. `_WH_EDITS` and the CUDA text are unchanged. Kept: the attribute gate and the exact/replay/veto checks.
  - Comments in the wh section were stripped.
- **AST check.** All 54 of master's module-level strings are byte-identical, except the deleted `HCG_TAG`. Size: **993,828 B**.
- **Prints, at the end of prepare in this order:** [hh-wh] BUILD/ROUTE/GATE/PATH (PATH says E races after it), [ea4x-dl] route (with PIN) and the cold-half line, a condensed `[hotcold] PATH`, then the [ea-enc] lines: build/join/phase-at, regs/recode/scales, per-T iso and layer, copy check and PATH CHOSEN with the reason. The last line is `[campaign] prepare at X s`. [hh-wh] ATTRS/SCALES were dropped.
- **Deviations:**
  - The E threads are daemon threads, as decode's other builds are, where moe's were non-daemon.
  - The layer-exactness check is skipped (marked ` layer-nondet`) when kernel_fn m-vs-m itself differs.
  - The state-copy check uses decode's tolerance-based helpers.
  - After a veto, no second arm is tried.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the kernel path changes are dict lookups and `_eae_pair` launches only, with no sync, `.item()` or value-dependent branch. All syncs are in prepare().
- [x] Nothing derived from `data` is cached: routes are keyed by T or class, set in prepare.
- [x] Weights: E4/E5 and wh5ds copies are lossless bit-permutations and scale packings of the NVFP4 bank, made in prepare(). Bit-exactness is gated in-run.
- [x] Activations BF16, KV FP8, queries unchanged: no change to attention or activation dtypes.
- [x] Attention sees every key: attention is untouched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. load_inline only; the lint subprocess check is clean.
- [x] All work happens inside the timed call; no output caching.
- [x] Every CUDA string is byte-identical to master except the new E/wh module texts, checked by AST string-constant compare.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer`): `OK with warnings`, 0 errors, 197 host-sync warnings (all "outside kernel_fn's call graph; fine in prepare()"), and 8 dtype warnings (uint8 repack buffers, pre-existing). py_compile OK; __pycache__ removed.

## Risks / what could fail
medium

## Result (after the run)
(pending)
