# muse_decode_layer · 2026-10-27T1243Z · check · w6g-hcgate

Campaign `2026-10-26-expert-bandwidth`, wave 6, lever **W6-G**: the T=128 hybrid race re-run with W6-B's cells. The
orchestrator wrote the spec at about 12:35Z, on top of panel 2's verdict (`research/wave6-panel2-2026-10-27.md`).
There is one decision: the routed expert path of the T=128 hotcold class (128..191). Everything else is unchanged, T=256
included.


**Orchestrator changes (1254Z), after the audit:**
- (1) PROMOTE ONLY. If the cells-off race already routed a hybrid H, the confirm exits as a miss and that H, with
  W6-B's cells as W6-B routed them, is kept. Before, a margin miss of H+cells vs master+cells could demote a routed
  hybrid to master on noise, against the 'incumbent on any miss' rule. That is the board's draw (4a7a49d0).
- (2) The confirm runs AFTER _w6p_setup, so its timing can never push PERHEAD past its 230 s start deadline.
- Submission sha256 d6e5d459b808a9e5..., 975,200 bytes after the comment strip.

- **Job:** `41975d1ce6334756` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at commit 31fac70e on branch campaign/eb-w6g-hcgate (worktree
  `.worktrees/w6g-hcgate`). 31fac70e is 8e00ac9c plus the audit fix to `_hcg_cells_check` (see Verification).
  - Submission sha256 240c531fbdaf92ed....
  - build.py writes 975,076 bytes. The raw build was 1,061,045 bytes, so build.py stripped comments (AST identical).
    That is under 985,000 and under 990,000.
  - moe is byte-identical (641,707 bytes, md5 unchanged).
- **Base:** campaign/eb-fsilu a5e66cee, the campaign stack: W6-B T>=128 diet, PERHEAD, W6-E fork code (never routes),
  the c320t bar, the W6-M moe fork and fast silu. The base build is 971,620 bytes. The Engine D stub was already on the
  base, so it is not redone.

## Goal
The decode T=128 hotcold race (`_hc_setup` / `_hc_layer_gate`) runs before `_t128_setup`. So it times master against
the serial hybrids with all of W6-B's T=128 cells off. Two things that race cannot see:
- norm, pool, hilo and unpool apply to every candidate.
- The comb cell (ea_combine with PDL after cold K2) exists only under a hybrid. In 4a7a49d0 it measured x2.0-2.1
  isolated.

Today the race routed the hybrid in 4a7a49d0 (the board, 1.5697), fff7d3e7 and ca44d5c0. It routed master in 27a008fa,
a76e7cb5, ee9142eb, f2aece51 and 17e7915a, with H-16's worst round between x0.970 and x1.001. The v16 rows follow that
draw (x0.97 vs x1.06), and it explains most of the board's lead over the campaign stack.

W6-F's `_hcf_confirm` re-races with the cells, but it can only demote, and only after a fork routed. W6-G ports its
structure without the fork, and it can promote: when the cells-off race picked master or a serial H, the class is
re-raced with the cells as kernel_fn will run them, and the winner is routed.

Questions (each answered by a greppable `[w6g-hcgate]` line; these lines end FINAL PATHS, with the PATH CHOSEN lines
last):
1. **Does the confirm promote a hybrid at T=128 where the cells-off race said master? With what worst-round ratio?**
   - `[w6g-hcgate] Q1 T=128 worst round vs master+cells: H-16 x.. (cells-off x..), H-8 x.. (cells-off x..) (margin
     1.03/1.02) -> ..`
   - `[w6g-hcgate] Q1b promote at T=128: YES|no; cells-off pick <master|H..> kept|REPLACED -> <final> (<why kept>)`
   - `[w6g-hcgate] PATH CHOSEN T=128 (128..191): <final> with W6-B cells <cells> (cells-off race <pick>[, confirm
     decided]); N s`
2. **How do the judged v16s / v16f rows compare with 4a7a49d0 (hybrid) and f2aece51 (master)?** Prepare prints the
   confirm's layer delta in `[w6g-hcgate] Q2 v16s/v16f: layer T=128 <final> X us vs master+cells Y (+d)`. The answer
   itself comes from the judged rows, compared with each row's own SGLang time.
3. **When master was routed, does comb pass its own W6-B gate?** `[w6g-hcgate] comb T=128 (master was routed): canary
   ok; isolated r1/r2 x../.. (bar 1.15) -> on|off under H`.
4. **Canary:** `[w6g-hcgate] t128_hcg_canary: finite, err unchanged yes|no; id rows differing vs master H-16 n, H-8 n`.

Kill / expected (also printed as a `[w6g-hcgate] kill: ...; expected: ...` line):
- **Kill:** the original pick stays, which is today's behaviour, if any of these happens:
  - no H reaches >= 1.03 (or the tight 1.02 with a small IQR) against master+cells in every round on both inputs;
  - any canary, layer-check or copy-check miss;
  - a raise;
  - the start deadline (255 s) or the checks deadline (266 s) passes.
- **Expected:**
  - If H-16 is promoted where the race said master: v16s / v16f fall by about -10..-20 us each against f2aece51,
    matching 4a7a49d0's hybrid rows. All other rows stay at 0.
  - If the confirm keeps master: the rows match f2aece51, and only prepare time grows (about 2-4 s).

## What changed, and where
- `rewrite/lib/hotcold_decode.py`
  - **`_hc_setup`** records each class's cells-off race in `state['hc']`: its clean candidates and worst ratios
    (`race`), plus `why` and `times`. A class counts only when the race itself decided it: master because no H won
    every round, or an H that passed its layer check and the copy check. Deadline, check-miss and raise outcomes do
    not count, and the confirm skips them.
  - **New `_hcg_confirm`** runs in prepare() right after `_t128_setup` and never raises. Its structure is W6-F's
    `_hcf_confirm`, minus the fork.
    - **Candidates:** H-16 and H-8 if clean in the cells-off kernel check, plus the race's own pick if that was some
      other H.
    - **Cells:** W6-B's T=128 flags as routed. Under master, comb is inert, because only `_hc_tail` reads it.
    - **When master was routed (`_hcg_comb`):** `_t128_setup` never offered comb, so it is decided here first.
      - It runs the named, synced `t128_comb_canary` (`_t128_canary`'s comb rule).
      - It then checks comb's isolated W6-B bar (x1.15 in both read-flushed rounds, `_t128_ops` graphs).
      - Only if both pass do the H candidates run with comb.
    - **Skip:** if no cell applies to any candidate, the race already timed this configuration, so there is no
      re-race.
    - **Canary before any timing:** the named, synced `t128_hcg_canary`, an eager `kernel_fn` at T=128 per candidate.
      Outputs must be finite and the error words unchanged; differing id rows are printed for information.
    - **Race:** `_hc_layer_gate(state, device, 128, hs, 20261094 + 128)`. That is the race's own seed, so the same two
      judge-like inputs, 2 x 2 read-flushed rounds, and `_rt_select` against master with the race's default margin
      (1.03, or tight 1.02).
    - **Routing the winner:** an unchanged pick keeps its cells (comb goes back to its W6-B value). A changed pick is
      routed only after two checks:
      - H: `_hc_layer_check` (H + cells against master + cells, routing ids identical, strict bounds).
      - Master (a demotion: W6-B had gated the cells under the H): `_hcg_cells_check`, which compares master with the
        cells on against the cells off on two inputs under `_cb_cmp_layer`'s strict bounds. It sets
        `hc['on'][128] = 0` for both runs and restores the routed H in a `finally` (the confirm leaves the class at
        the cells-off pick's H).
      - Then for either: `_hc_copy_check(state, device, [128], ...)` on the final routing (two states, four
        alternating replays).
    - **Any miss:** T=128 goes back to the cells-off pick with comb's original flag.
    - **Output lines:** the stale `[hotcold] PATH CHOSEN T=128` and `[w6b-t128diet] PATH CHOSEN T=128:` lines are
      removed from FINAL PATHS and printed again after the confirm. The W6-B line is rewritten only when comb's flag
      changed.
  - **No fork:** `CAMPAIGN_HCF = False`. W6-E's dead post-W6-B fork gate (`_hcf_setup`, `_hcf_gate`, `_hcf_canary`,
    ...) is removed. `_hc_experts`' fork hooks stay, but `state['hc']['fk']` is never created, so they never engage.
  - `_hc_copy_check(..., off=...)` (ported from W6-F): the FAIL text names what turns off.
- `rewrite/src/muse_decode_layer.py` (prepare): the `_hcf_setup` call is replaced by
  `paths, hcg = _hcg_confirm(...)`. `hcg` is printed at the end of FINAL PATHS, each line at most 200 characters.
- **Not changed:** T=256, the race itself, W6-B's own cells and gates, CUDA (no new TU or module), moe.
- **Verification:**
  - `build.py`: decode 975,076 B. `--check` up to date for both problems; okc_lint exit 0 (only the usual prepare
    host-sync warnings).
  - `cuda_syntax.py` ok; py_compile ok.
  - An AST scan finds no undefined name in the `_hcg_*` functions.
  - A CPU emulation runs the real `_hcg_confirm`, `_hc_path_lines` and `_rt_select` against stubbed GPU helpers, over
    17 scenarios. All pass:
    - promote H-16;
    - promote, then copy FAIL;
    - master kept;
    - promote with comb off;
    - demote to master;
    - demote, then cells-check FAIL;
    - switch to H-8 with a layer FAIL;
    - switch to H-8 ok;
    - H kept;
    - no clean race;
    - gate raise;
    - deadline;
    - no cells with comb off, and no cells with comb on;
    - canary FAIL;
    - no W6-B state;
    - Marlin H48 pick.

    The emulation also checks four things: every line is at most 200 characters; PATH CHOSEN lines come last; each
    T=128 PATH line appears exactly once; and the copy check runs on the candidate routing with comb off under master.
  - **Audit fix (31fac70e).** The audit found that `_hcg_cells_check` never set `hc['on'][128]` to 0. `_hcg_confirm`
    puts the class back at the cells-off pick's H right after the confirm gate, so in a demotion both kernel_fn calls
    ran H (H + cells with comb off vs H bare), not master. Master + norm/pool/hilo/unpool would then have been routed
    with no strict-bound comparison against a reference. The 17-scenario emulation stubbed this helper without
    recording the routing, so it could not see the problem; re-run with the routing recorded, scenarios D and D2
    showed `('cells', -16)`. The helper now forces `on[128] = 0` for both runs and restores the H in a `finally`. A
    second CPU emulation runs the real `_hcg_cells_check` (h0 = -16, -8, 0; pass and bound miss). Every kernel_fn call
    sees `on[128] == 0` (ref with every cell off, got with the cells as kept), `on[128]` and `t1` are restored after
    each call, and a miss returns a problem after the first input. The 17 scenarios still pass.

## Rules check (each must be true before the run)
- [x] Weights lossless and unchanged: no weight path is touched. The confirm only picks among existing paths.
- [x] fp32 accumulation, no bf16 split-K partials: unchanged paths (Marlin / wgmma / Engine A / W6-B cells as before).
- [x] Activations bf16; KV FP8; queries <= FP8 E4M3; every key attended: attention is untouched.
- [x] CUDA-graph capturable: kernel_fn reads only shape plus the prepare-set flags (`hc.on[128]`, `t128.on[128]`). The
  copy check replays the final routing as graphs on two states.
- [x] Nothing derived from `data` is cached across calls. The confirm's draws are prepare-only synthetic inputs.
- [x] No timing patches, harness access, subprocess or network. The timing is prepare-only `_rt_time`, as in the race.
- [x] All work inside the timed call; no output caching or replay.
- [x] Read-flush gates (`_rt_time` default `flush="read"`); whole-layer gate with margin > 1.0; incumbent on any miss.
- [x] A named, synced canary before any timing (`t128_comb_canary` when needed, `t128_hcg_canary`).
- [x] Decision lines are printed again at the end of prepare; deadlines are 255 s (start) and 266 s (checks).
- [x] No new CUDA and no new load_inline module. No per-state workspace or ticket added.

## Risks / what could fail
- **The hilo router adds with float atomics.** `_hc_layer_check` requires bitwise-identical routing ids between H +
  cells and master + cells. A near-tie flip would fail it, and T=128 would keep the cells-off pick (no regression).
  W6-B's own layer checks with hilo on showed `ids_tokens_differ 0`. The canary prints differing id rows as a warning.
- **Deciding comb inside the confirm.** When master was routed, comb is decided by its canary and isolated bar alone.
  W6-B's whole-layer cells gate is replaced here by the confirm race itself (H + cells + comb against master + cells)
  plus the layer check.
- **The confirm can demote.** If the race picked an H that then loses to master with the cells, T=128 routes master
  after the cells check and the copy check. That changes today's behaviour in the H direction too. This is the spec's
  "route the winner".
- **A re-race is a second draw on the same inputs.** It is the same seed and margin rule, but a narrowly losing H gets
  a second chance under a different configuration. The skip rule stops an identical configuration from being re-rolled.
- **Prepare time:** about +2-4 s (canaries, 3 x 2 graphs x 2 rounds, checks). Prepare currently finishes the W6-B
  block at about 100-115 s, well inside both deadlines.

## Result (after the run)

- 2026-10-27T1303Z: **passed, 1.5749** (CI 1.551-1.599), above the 1.5697 board.
  - **Promoted:** cells-off race master; confirm H-16 x1.023 vs master+cells (cells-off x1.016), H-8 x0.977 -> H-16+comb
    (comb isolated x2.03 / x2.18).
  - Layer T=128 350.5 -> 336.3 us. Canary: id rows differing 0. v16s / v16f x1.000 / x0.997 vs 4a7a49d0 (its hybrid draw).
  - d1s x1.036, d1f x1.038. d32f x0.955, seen in 5 runs vs 4a7a49d0 (that run's lucky draw).
  - **Scored run c1daf58bf192471e.**
