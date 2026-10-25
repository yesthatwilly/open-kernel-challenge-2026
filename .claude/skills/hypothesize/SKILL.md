---
name: hypothesize
description: Mint kernel-optimization hypotheses as markdown files in <problem>/hypotheses/queue/ for /churnsubs to execute. With a description, spec that one idea; without one, invent modest, single-area, attributable ideas from the problem's TASK.md, baseline pipeline, current best profile and FINDINGS.md. Use for "/hypothesize <problem> [idea]", "/hypothesize all", "queue up ideas for <problem>", "what should we try next on <problem>".
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
- **description**: optional. Given → one hypothesis specifying that idea. Absent → mint N
  (default 3) of your own choosing.
- Writes files only, via `hyp.py new`; never commits, stages, or edits anything else. churnsubs
  commits the file at its next intake. Many /hypothesize runs may happen at once.

```
H="python3 -I $CLAUDE_PROJECT_DIR/.claude/skills/hypothesize/bin/hyp.py"
```

## Procedure

1. `$H context <problem>` → the files to read, the current best, the queue, done slugs, area
   hints. If `DIAGNOSIS.md` is listed, read it first: it is today's clean-context diagnosis and
   its **Actions** are minted before your own ideas, its **Do not try** is binding. Read
   `hypotheses/LESSONS.md` (cross-problem measured facts) and **FINDINGS.md** (learnings by area, open follow-ups, Do not try, where the
   best spends its time), then TASK.md (shapes, weights, floors, tests, rules), `baseline.py`
   (SGLang's pipeline: the named parts a hypothesis may touch), and master's `submission.py`
   (what the next hypothesis will be built on). For `all`, spawn one subagent per problem with
   this same procedure and the list of slugs already queued, so the batch does not overlap.
2. **Choose ideas that attribute.** Each hypothesis changes one named area modestly, so a gain
   or loss can be pinned to it. Prefer, in order: (a) parity with SGLang's own path where the
   current best is below 1.0x (establishes the base), (b) the heaviest-weighted shapes with the
   biggest gap to the speed-of-light floor in FINDINGS, (c) launch-count reductions on sub-10 µs
   shapes, (d) stacked ideas with `depends_on` on a landed or pending gain. Never: top-k/sparse
   attention, precision below what the rules allow, caching anything derived from inputs, host
   syncs, work outside the timed call. Do not duplicate a queued/running/done slug or a
   Do-not-try entry. **Size for the judge:** a gain must beat the best by ~1.2% on the headline
   (1.07% after a confirm rerun; churn.py `margin()`), so an idea whose weighted effect is
   smaller can only land stacked with others: say so, or stack it.
3. For each idea run `$H new` with: `--slug` (short, kebab), `--area` (one named part),
   `--change` (a paragraph a different engineer could implement without asking: which function,
   which call, what replaces what, what stays), `--expected` (JSON list of
   `{shapes: glob over shape keys like "layer=sliding load=decode_*", direction, why}`),
   `--risk`, `--priority` (1 = first), `--depends-on` when stacking, `--notes` for the risk text,
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
   primaries and keeps 40% of each day's checks for them, so every `all` round mints at least
   one rewrite per problem whose queue has none. Aim big: a whole-layer or whole-pipeline
   overhaul (one fused or persistent kernel across stages, a new parallel decomposition, a
   different data layout) is in scope when the profile says the gap to the leaders is
   structural. Plan it as stages, each with a readable milestone and a fallback to the incumbent.
4. Every new file starts with `review: required`. Before /churnsubs claims it, a fresh Opus
   subagent reviews it adversarially (justification, confounding by landed hypotheses, whether
   an A/B companion with another hypothesis reverted or retuned is needed) and may queue
   companions beside it with `kind: ab-revert|ab-tune|control` and `controls_for: <slug>`.
   Write the `change` and `expected` so that reviewer can check them: name the mechanism, the
   kernels it should move (from FINDINGS), and the shapes.
5. Reply with one line per file: path, area, the expected shapes, and why it is worth a run. If
   the day's remaining quota exceeds the queue depth (see `hypotheses/INDEX.md`), say how many
   more the queue could absorb.

## Shape keys (for `expected.shapes`)

`k=v` pairs sorted by key, spaces inside values as `_`, e.g.
`batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic`
or for bf16_proj `layer=router load=verify_32 m=256`. Globs match the whole key.
