---
name: hypothesize
description: Mint only the most promising kernel-optimization leads as markdown files in <problem>/hypotheses/queue/ (or h100/) for /churnsubs to execute. Every lead must clear a value bar (measured evidence, >= 2.5% headline if it works, gain x p >= 1.5%); minting nothing is a valid outcome. With a description, spec that one idea; without one, find the leads from the board's per-row triage, FINDINGS.md and the current best profile. Use for "/hypothesize <problem> [idea]", "/hypothesize all", "queue up ideas for <problem>", "what should we try next on <problem>".
user-invocable: true
allowed-tools:
  - Read
  - Bash(python3 -I */.claude/skills/hypothesize/bin/hyp.py *)
  - Bash(python3 -I */.claude/skills/churnsubs/bin/okc_lint.py *)
  - Bash(git log *)
  - Bash(git show *)
  - Bash(cat *)
  - Bash(ls *)
---

# hypothesize — write hypotheses for the queue

`/hypothesize <problem|alias|all> [description] [--n N]`

- **problem**: `muse_attention_decode` (`attn`, `attn_decode`), `muse_bf16_proj` (`bf16`),
  `muse_decode_layer` (`decode`), `muse_moe_w4a16` (`moe`), `muse_prefill_layer` (`prefill`),
  or `all` (one subagent per problem, in parallel, each minting N).
- **description**: optional. Given → one hypothesis specifying that idea (`--user-requested`:
  it may pass below the bar, and the file says so). Absent → mint **at most** N (default 2) of
  your own choosing, and only those that clear the value bar. Zero is a fine answer.
- Writes files only, via `hyp.py new` (and `hyp.py h100-result` for a node run's result); never commits, stages, or edits anything else. churnsubs
  commits the file at its next intake. Many /hypothesize runs may happen at once.

```
H="python3 -I $CLAUDE_PROJECT_DIR/.claude/skills/hypothesize/bin/hyp.py"
```

## The value bar

The user, 2026-10-28: "really really stringent on what it hypothesizes, because i cant get small
ROI for the tokens this skill takes. i need it only producing the most promising leads". So
`hyp.py new` refuses (exit 3) any lead that is not an A/B companion and is below the bar:

- `--target check` (should beat the board in one judge check): `--gain` >= 2.5% headline;
- `--target diag` (a prepare-only judge diagnostic) and `--target h100` (a node experiment for
  the next GPU window; written to `hypotheses/h100/` on hold, never dispatched by churn): the
  lever it sizes is worth >= 3%, and `--decision` pre-registers what each outcome leads to;
- every target: `--gain x --p` >= 1.5%, `--evidence` with numbers, units and a source (a job or
  run id, a profile line, a research/ file, a FINDINGS or LOG entry), and `--cost`.

On exit 3 do not retry with inflated numbers: report it as a rejected lead with its EV. Spend
the reading on finding one strong lead, not on writing several weak ones.

`hyp.py new` also exits 3 when the mechanism's newest `CLAIMS.md` line is `[refuted]` or
`[superseded]`: the claim names the slug, or shares 2+ distinctive words with the slug, area or an
identifier in the change (`s2b`, `mc2`, `c320`; dtypes such as `fp8` / `bf16` never count), or 1 that is
also in a slug the claim names. It prints the matching lines. Never re-mint such a mechanism
without a stated difference: `--prior "<slug>: <what differs, 40+ chars>"`, normally that the
claim's `reopen` condition is now met. Every file is stamped with `mint_blob` and `best_at_mint`;
churn holds it unrun once the board moves past the margin (`stale-board`) or master's submission.py
is no longer the blob it was minted or reviewed on (`stale-blob`).

## Procedure

0. `$H triage <problem>` (cheap, no model reads): the board run's rows ranked by how much the
   headline moves if each reached its practical floor, plus the queue's value-case count. A row
   already near its floor, or worth under ~1.03x even if halved, is not a lead. For `all`, run
   triage for every problem first and spawn subagents only for the problems with real headroom.
1. `$H context <problem>` → the files to read, the claims by tag, the active campaign, the
   diagnosis' standing, the current best, the queue, done slugs, area hints. Read in its
   `read_first` order. **`<p>/hypotheses/CLAIMS.md` first** (`$H claims <p>` prints the newest line per
   mechanism; `--match "<words>"` checks one mechanism): `[landed]` claims own their rows, `[lead]`
   is what to mint or check first, `[shelf]` mechanisms are minted only as a stack whose rows sum
   to >= 2.5% headline, `[unresolved]` only with the runs or pinned routes its reopen rule names,
   `[node-only]` only as a production-shape judge check of per-SM work (per-SM cycles transfer to
   the judge H200; HBM rates do not), and `[refuted]` / `[superseded]` are Do-not-try. Then the
   active campaign's FINDINGS, DISTILL §3/§4.4, the top 150 lines of its LOG and its H100 findings
   for this problem: most of master's recent code came from there, not from churn. `DIAGNOSIS.md`
   is in `read_first` only while the board is within the margin of the one it was written on; then
   its **Actions** come before your own ideas and its **Do not try** binds. Otherwise `context`
   marks it `historical` (date, board then and now): read it as background and check each action
   against CLAIMS and triage. Read
   `hypotheses/LESSONS.md` (cross-problem measured facts) and **FINDINGS.md** (learnings by area, open follow-ups, Do not try, where the
   best spends its time), then TASK.md (shapes, weights, floors, tests, rules), `baseline.py`
   (SGLang's pipeline: the named parts a hypothesis may touch), and master's `submission.py`
   (what the next hypothesis will be built on). For `all`, spawn one subagent only per problem
   that step 0 found real headroom in, with this same procedure and the list of slugs already
   queued, so the batch does not overlap.
2. **Choose ideas that pay and attribute.** Start from the `[lead]` claims, the `[shelf]`
   stacks that sum past the bar on one base, and the triage's top rows, and ask what
   measured gap explains them. Each hypothesis changes one named area, so a gain
   or loss can be pinned to it. Prefer, in order: (a) parity with SGLang's own path where the
   current best is below 1.0x (establishes the base), (b) the heaviest-weighted shapes with the
   biggest gap to the speed-of-light floor in FINDINGS, (c) launch-count reductions on sub-10 µs
   shapes, (d) stacked ideas with `depends_on` on a landed or pending gain. Never: top-k/sparse
   attention, precision below what the rules allow, caching anything derived from inputs, host
   syncs, work outside the timed call. Do not duplicate a queued/running/done slug, a
   Do-not-try entry, a `[refuted]`/`[superseded]` claim or rows a `[landed]` claim owns. **Size for the judge:** a gain must beat the best by ~1.2% on the headline
   (1.07% after a confirm rerun; churn.py `margin()`), so an idea whose weighted effect is
   smaller can only land stacked with others: say so, or stack it.
3. For each idea run `$H new` with: `--slug` (short, kebab), `--area` (one named part),
   `--change` (a paragraph a different engineer could implement without asking: which function,
   which call, what replaces what, what stays), `--expected` (JSON list of
   `{shapes: glob over shape keys like "layer=sliding load=decode_*", direction, why}`),
   `--risk`, `--priority` (1 = first), `--depends-on` when stacking, `--notes` for the risk text,
   the value case (`--target`, `--evidence`, `--gain` with its arithmetic in `--expected`, `--p`,
   `--cost`, and for diag / h100 `--decision` and `--instrument`), use `--dry-run` to check a lead
   before committing to the write-up,
   and **`--prior "<closest earlier slug>: what differs this time"`** (or `none`; required). Find
   the closest earlier attempt of the same mechanism before writing the change; when it was a
   regression, flat (`within-ci`) or rejected, `hyp.py` wants 40+ characters on what differs.
   The command lints the file and prints its path; on an error, fix the arguments and retry.
   **Rewrites.** When FINDINGS show a subsystem far from its floor (2x or more) and tuning the
   incumbent has stopped paying, mint `--kind rewrite --replaces "<kernels replaced>" --stage 1`:
   a new kernel that replaces the whole subsystem. Name the floor and the incumbent's per-shape
   time in `expected`, and the design choices that should close the gap (parallelism, bytes in
   flight, tensor-core path, dequant). A rewrite is judged on the replaced kernels' time against
   its Milestones, not on the headline; later stages use `--depends-on <previous stage>
   --stage N+1` and build on that code whatever its score. churnsubs runs rewrites ahead of
   primaries and keeps 40% of each day's checks for them, so a problem whose queue has no rewrite
   gets one when a rewrite lead clears the value bar (never one minted only to fill the share). Aim big: a whole-layer or whole-pipeline
   overhaul (one fused or persistent kernel across stages, a new parallel decomposition, a
   different data layout) is in scope when the profile says the gap to the leaders is
   structural. Plan it as stages, each with a readable milestone and a fallback to the incumbent.
4. Every new file starts with `review: required`. Before /churnsubs claims it, a fresh Opus
   subagent reviews it adversarially (justification, confounding by landed hypotheses, whether
   an A/B companion with another hypothesis reverted or retuned is needed) and may queue
   companions beside it with `kind: ab-revert|ab-tune|control` and `controls_for: <slug>`.
   Write the `change` and `expected` so that reviewer can check them: name the mechanism, the
   kernels it should move (from FINDINGS), and the shapes.
5. Reply with one line per file: path, target, EV (gain x p), the claims it builds on, and the
   evidence in a clause; then one line per rejected lead with its EV and why it fell short (a
   claim refusal counts). A finding you made while reading that no claim records yet (a dead end,
   a shelf row win, a node-only fact) goes in the reply as a proposed CLAIMS line in the
   `CLAIMS.md` format, for the main thread to append with `churn.py claim-add`; you write no
   file but hypotheses. Do not pad the queue to the
   day's quota: an unspent check is cheaper than a weak lead.

## H100 node experiments

`--target h100` writes to `<p>/hypotheses/h100/` on hold (churn never runs it; GPU windows need
the user's go). When its node run is read, file it with
`$H h100-result <p> <slug> --run <hopper-probe run id or queue job> --verdict held|flipped|untested|null
--note "<node numbers; judge numbers when checked>" [--effect <judge headline %>] [--board <x>]`
on master in the main checkout: it records the result in the frontmatter and a `## H100 result`
section, moves the file to `h100/done/`, and appends one CLAIMS line: `[lead]` when the transfer
held on the judge and `--effect` clears 2.5%, else `[node-only]` with the transfer caveat as its
reopen rule. `--dry-run` prints the line and changes nothing.

## Shape keys (for `expected.shapes`)

`k=v` pairs sorted by key, spaces inside values as `_`, e.g.
`batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic`
or for bf16_proj `layer=router load=verify_32 m=256`. Globs match the whole key.
