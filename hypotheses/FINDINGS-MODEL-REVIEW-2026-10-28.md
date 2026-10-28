# Findings model review (musespark, 2026-10-28)

This review was read-only; nothing changed in either repo. I re-checked every number below against master `d8296a17`. Where the design and the critique disagreed, I used the corrected figure.

Paths are relative to `/home/user/git/musespark`. Short names:
- `churn.py` is `.claude/skills/churnsubs/bin/churn.py`; `hyp.py` is `.claude/skills/hypothesize/bin/hyp.py`.
- Skill files are under `.claude/skills/`.
- `LOG` and `DISTILL` are in `campaigns/2026-10-26-expert-bandwidth/`.
- `DECISIONS`, `LESSONS` and `COSTS` are `hypotheses/*.md`.
- `hopper-probe/` is `/home/user/git/takemyjob/prototypes/hopper-probe/`.

Two corrections to the brief:
- The screen is DECISIONS:412-441: 3 STALE, 11 SUPERSEDED, 16 BELOW-BAR.
- Its 30 files were minted between 10-25 and 10-27, not on 10-24/25.

**Bottom line.** No single change would have prevented the 30/30. Each of the three verdicts needs its own gate:
- **STALE-type:** a freshness check when a file is claimed. 22 of the 30 were minted on boards that have since moved by 1.24% or more.
- **STALE and SUPERSEDED (14):** landings and kills recorded where the minter and the adversary read.
- **BELOW-BAR (16):** a value bar computed with the same formula that produces the score.

The biggest new finding: triage, which sizes every lead, uses the wrong row weights and the wrong time field. Today's attention lever choice was based on its numbers.

## 1. Verdicts

**Q1. Are hypotheses and campaigns informed by all the proper findings? No, and the gap runs both ways.**
- The minter and the adversary read churn's own results (through FINDINGS), LESSONS and five DIAGNOSIS files from 10-25 (hyp.py:343-346, hypothesize/SKILL.md:57).
- The words "campaign" and "DISTILL" appear 0 times in hypothesize/RESEARCH_PROMPT.md, churnsubs/ADVERSARY_PROMPT.md and hypothesize/SKILL.md.
- Only 1 of the 41 commits to `muse_*/submission.py` since 10-26 is a churn promote (9fbde72c). So `landed()` (churn.py:1570-1584), which is the lineage the adversary sees, tops out at:
  - decode: 1.3142, against a board of 1.6551;
  - moe: 1.3752, against a board of 1.8941.
- The campaign does not read churn either. LOG:55 lists "b12 unpooler m1" as a run-first pick, but churn already ran it: `muse_bf16_proj/hypotheses/done/neutral/2026-10-26T221215Z-b12-unpooler-slice-m1-r16-rowowner.md`, job a642424734f54365.

**Q2. Is there a better way to keep good findings apart from bad ones? Yes. The outcome folders cannot do it.**
- A folder only records whether a run beat the board at harvest:
  - 28 of the 50 churn items now on master sit in `done/neutral`;
  - 244 of the 257 gain and neutral patches no longer apply to master.
- `within-ci` sends real row gains to Do-not-try. Onepass is at `muse_attention_decode/hypotheses/FINDINGS.md:228`, while DECISIONS:400 says "keep it for stacking".
- Every `churn.py index` rewrites FINDINGS whole (churn.py:1859). Commit c3c14a4d erased two hand-written findings that way: the s2b duplicate (c18c73f7) and the V64 refutation (546f6386).

**Q3. Is there a better way to classify gains and losses? Yes, on three counts.**
- **Sizing.**
  - Triage weights each row by w/W and reads the raw time (hyp.py:291-298).
  - The score is the geometric mean over loads of load-time ratios, using `median_ns` (`muse_attention_decode/TASK.md:50`). Recomputed that way, all five boards match exactly.
  - Measured against that, the ratio of the true weight to triage's weight is:

    | problem | true ÷ triage weight |
    |---|---|
    | attention | 0.28-3.08 |
    | prefill | 0.51-2.43 |
    | bf16 | 0.42-1.98 |
    | decode | 0.81-1.54 |
    | moe | 1.00 |
- **Base.** With no base, rows are compared with SGLang (churn.py:568). 11 of the 12 no-base results show every row as "gained", including both onepass runs and 7284659f.
- **Verdict.** `classify` judges the headline alone (churn.py:613-641):
  - A gain hid a flat mechanism. `restack-sk-pseudo-cluster-plus-v3-fused-preattn` scored 1.4920, but its target rows read 1.0013 and 0.9973.
  - 29 neutrals with a base hit every predicted row. One of them is on Do-not-try (`muse_decode_layer/hypotheses/FINDINGS.md:172`, 5/5).
  - 31 neutral→gain and 7 regression→neutral reclassifications were never applied (`hypotheses/ticks/2026-10-28T145238Z.json`).

**Q4. Is the model reused for H100 and judge checks? Only by hand.**
- **Checks.** The queue and claim steps check only `review == proceed` (churn.py:1266-1301, 1476).
  - `sliding-verify1-pseudo-fused-preattn-rope` was claimed at 14:53Z (3c351ffe). Its review dated from 10-25, against a 1.5295 board.
  - That was 7 minutes before the screen's holds were committed (d6fb1c1c, 15:00Z).
  - It came back within-ci: 1.6191 against 1.6340 (66e589b9). Cost: at least 105k tokens plus one check (COSTS:270).
- **H100.** Its results have no channel into the hypothesis tools:
  - `hopper-probe/queue/HISTORY.tsv` (108 rows) has no problem or verdict column.
  - `research/h100-findings.md` (962 lines) is referenced by no skill.
  - "H100" appears 0 times in the five FINDINGS files, the five DIAGNOSIS files and LESSONS.
- H100 is the higher-yield source. H100-derived builds gave about 0.56% of board per judge run, against churn's 1 gain per about 14 checks (LOG:2130-2131).
- Its transfer results are what should gate the next checks:
  - M456's gain held on the judge (h100-findings.md:736-739).
  - X1's gain flipped on the judge: ×0.942 at T=8 (LOG:1853-1860).

## 2. What is broken, by severity

1. **Triage mis-sizes levers, and today's attention choice used it** (hyp.py:291, 293, 298).
   - LOG:2146-2148 says the sliding rows "carry 74% of the weight", that ~9 µs each would give ~3.0, and that every row at its floor would give ×3.5.
   - On the scoring formula, for the same run (4e386647edd845f7):

     | quantity | triage (LOG:2146-2148) | scoring formula |
     |---|---|---|
     | sliding rows' share of the weight | 74% | 37% |
     | all six sliding rows at their 5 µs floor | ~3.0 at ~9 µs each | 2.16 |
     | all six full rows at their floor | — | 2.21 |
     | every row at its floor | ×3.5 | ×2.06 |
   - The heaviest rows are full verify 16 and 32: true weights 0.124 and 0.132, against triage's 0.043. They run at 1.97× and 1.83× their floor.
   - The raw field adds 3.5-4.5 µs to 11 of the 12 rows.
   - The attn-fixed brief and the decision to drop attn-full-splitkv (LOG:2156-2160) were sized on the triage numbers.
2. **Campaign landings and kills never reach churn's lineage or prompts** (churn.py:1570-1584; ADVERSARY_PROMPT.md:11). Example: DISTILL.md:439 marked mc2 "retire" (1a7d0bd4, 10-26T23:59Z). It became a queue hold 39 hours later (DECISIONS:415-416).
3. **Nothing checks freshness at queue or claim time** (churn.py:1266-1301, 1476). The value bar runs only when a file is minted (hyp.py:127-156), and only since 1a8e3c68 (14:51Z today). So no queued file has a value case.
4. **FINDINGS is regenerated whole** (churn.py:1859). This is how c3c14a4d erased c18c73f7 and 546f6386.
5. **Today's screen verdicts disappear around 10-29T15:00Z.**
   - Holds expire 24 hours after their last commit (churn.py:51, 2032).
   - 29 of the 30 hold reasons do not match HOLD_PROTECT. Only `experts-rega-lpt-group-order` does.
   - FINDINGS shows expired items only as a count (churn.py:1903, 1945-1947).
6. **Most comparisons have no base.**
   - BEST.json `result` is null in all 5, so `base_result_for` finds nothing (churn.py:542).
   - "Where the time goes" (churn.py:1890) appears in 0 of the 5 FINDINGS files.
   - Without a base, row comparisons are SGLang ratios (churn.py:568).
7. **Verdicts read the headline only, and within-ci means Do-not-try** (churn.py:613-641, 1936).
8. **DIAGNOSIS is three days old but labelled "Today's" and binding** (churn.py:1900; hypothesize/SKILL.md:55-56).
   - It ran 10-25T2032-2034Z at boards of 1.5295/1.2566/1.3142/1.4021/1.4677. Today's boards are 1.6340/1.2794/1.6551/1.8941/1.5008.
   - Re-diagnosis is skipped on cost (DECISIONS:407-411).
9. **Quota pressure pushes toward /hypothesize:** churn.py:1389 (shown at `hypotheses/INDEX.md:20`) and 3to1/SKILL.md:56.
10. **Triage is blind on bf16 and prefill.**
    - For bf16, hyp.py:255 picks up `3e3ffc5afc57433b.profile.json.gz` first.
    - For prefill, hyp.py:272 rejects "1,036.3 µs" (`muse_prefill_layer/TASK.md:39`).
11. **hopper-probe tools read only campaign runs** (tools/records.py:29-30). HISTORY.tsv has only the columns utc/job/node/gpus/status/results.
12. **Two zombie claims count as in flight** (churn.py:1332): attention s7 (claimed 10-27T2128Z) and prefill s2 (claimed 10-25T2126Z).
13. **End-game findings have no owner.**
    - Memcheck reads VOID because it runs out of memory (LOG:1539).
    - DEC-02 failed (LOG:1726).
    - Two H200 jobs are held (LOG:1525-1527).
    - `hypotheses/PENDING_USER.md` mentions none of these.
14. **Review prompts are written to `/tmp/churn-prompts`** (churnsubs/bin/fill_prompt.py:9), on the nearly full tmpfs.

## 3. Recommended findings model

**Keep the raw evidence where it is:** `done/`, `campaigns/*/runs/` and `hopper-probe/runs/`.

**Add one curated file per problem: `<p>/hypotheses/CLAIMS.md`.** Every reader sees it and no regeneration touches it.
- It is written by agents, append-only, one line per finding:
  `- [tag] mechanism — rows — evidence (job/commit) — measured on <board> — reopen if …`
- The tag is the stash:

| tag | meaning | what readers do |
|---|---|---|
| `landed` | on master, via churn or a campaign | treat as lineage; it owns its rows |
| `lead` | moved its rows, ≥2.5% headline-equivalent, not on master | check it first |
| `shelf` | moved its rows, but below the bar | stack only when a set sums to ≥2.5% |
| `unresolved` | inside row noise, or a route flipped | re-test only with more runs or pinned routes |
| `node-only` | H100 only; transfer held, flipped or untested | a schedule change needs a production-shape check (LOG:1859-1860) |
| `refuted` | rows flat or lost, with enough runs | binding Do-not-try |
| `superseded` | a later landing owns those rows | queued items that depend on it are held |

**Two rules for the tags:**
- The FINDINGS render marks a line "re-measure" when its board is more than `margin()` below today's BEST.
- A row hit from a single run earns `shelf`, not `lead`:
  - rows drift 1.5-2.5% per shape between runs (LESSONS:12-15);
  - route flips alone made the 1.634 board a lucky draw against a ~1.623 level (LOG:1970-1973, 2081).

| # | change | files | effort | prevents |
|---|---|---|---|---|
| 1 | Freshness gate in `_eligible_queue` and `cmd_claim`. Hold with a typed `hold_kind` when BEST moved > `margin()` since review (or mint), the reviewed blob ≠ master's, or a new mint has no value case. Stamp `review_blob`/`review_best` and `mint_blob`/`best_at_mint`. Drop the quota hint and the "queues are shallower" rule | churn.py:1266-1301, 1389, 1476, `review-record`; hyp.py `cmd_new`; 3to1/SKILL.md:56 | ~35 lines | runs like 7284659f; 22 of the 30 screened |
| 2 | CLAIMS.md for all 5 problems, printed verbatim at the top of FINDINGS, and added to `read_first` and both prompts | churn.py `_findings`; hyp.py:346; ADVERSARY_PROMPT.md; RESEARCH_PROMPT.md | ~15 lines | erasures like c3c14a4d; kill reasons lost at expiry; the onepass back-and-forth |
| 3 | `churn.py register-landing <p> <job>`: writes BEST.json with `result`, BEST.result.json, a `[landed]` line and a DECISIONS line. `landed()` also reads `[landed]` lines | churn.py; 3to1/SKILL.md:29 | ~30 lines | lineage blind to the 40 non-churn landings; duplicates like s2b |
| 4 | Triage on the scoring formula: `median_ns`, with weight (1/L)·w·t / Σ_load(w·t) | hyp.py:291-298 | ~10 lines | leads and BELOW-BAR calls mis-sized 2-3× |
| 5 | Plumbing: `result` in all 5 BEST.json plus a fallback to BEST.result.json; `no-base` instead of SGLang ratios; `int()` sort; exact `{job}.json.gz` glob; `[\d,.]+` regex; release claims older than 24 h; move prompts off /tmp | churn.py:542, 568, 1332, 1583; hyp.py:255, 272; fill_prompt.py:9 | ~25 lines | 11 wrong attributions; "Where the time goes" missing in 5/5; triage blind on 2/5 |
| 6 | A mechanism verdict (`hit` / `flat` / `no-base`) from the existing attribution, next to the promotion verdict. within-ci + hit → `shelf`, not Do-not-try | churn.py:561-641, 1929-1945 | ~20 lines | row wins listed as dead ends; flat mechanisms hidden inside gains |
| 7 | Label DIAGNOSIS with its date and its board against today's; binding only within `margin()` | churn.py:1900; hypothesize/SKILL.md:56 | ~5 lines | 10-25 actions minted against 10-28 boards |

**Where claims get written.** Write them on master in the main checkout.
- churn resolves `REPO` per worktree (churn.py:184), and 3to1 commits on the campaign branch (3to1/SKILL.md:61). A line written inside a worktree lands on the wrong branch.
- Put claims where scored files already go: the same SKILL.md line says "land scored files on master immediately".

## 4. First step today (~2 h, 3 agents, ~0.3M tokens) and what to defer

**Today**, within the 3-agent cap (3to1/SKILL.md:75):
- **Agent 1 (code):** changes 1, 4 and 5, about 70 lines.
- **Agent 2 (code):** changes 2 and 3, plus the edits at 3to1/SKILL.md:29 and :56, about 45 lines.
- **Agent 3 (Sonnet distil, ~0.1-0.15M tokens):** seeds the five CLAIMS.md files with:
  - the 14 STALE/SUPERSEDED reasons from DECISIONS:412-441, before they expire;
  - the two erased entries (c18c73f7, 546f6386);
  - the DISTILL §3.1 and §3.4 kills (DISTILL.md:172-186, 261-287), as `refuted` with their reopen rules;
  - the 10-27/28 LOG verdicts:
    - S1a closed (LOG:1462);
    - X1 closed for v4 (LOG:1853-1860);
    - the lucky-draw board (LOG:1970-1973);
    - onepass sits inside the board's own row spread (LOG:2049-2052, 2081);
  - the H100 closures, as `node-only` with their transfer readings.

  It also adds the memcheck VOID, DEC-02 and the held H200 jobs to PENDING_USER.md.
- **Then:**
  - Re-run `hyp.py triage muse_attention_decode` with change 4.
  - Re-check the attn-fixed brief (LOG:2156) against full verify 16/32 before phase A spends more on it.

**Defer:**
- **The full ledger** (CLAIMS.jsonl, claims.py, FINDINGS as a render).
  - It rewrites `_findings`, which every tick uses.
  - It needs a writer that is safe inside worktrees, and a parser for submission.py.
  - CLAIMS.md gets most of the value now. Revisit after 11-05.
- **CELLS, per-cell MDE, route flags, and a base "level" over archived runs.**
  - One check per hypothesis has no statistical power.
  - The final reruns use pinned bytes (LOG:1973, 2077-2078).
  - Campaign records are keyed by content_hash, not git blob.
- **H100 registration through hyp.py, and problem/verdict columns in HISTORY.tsv.** The window closed at 10-28 13:37Z, so do this only if you reopen GPU work.
- **A claim check in okc-attempt-gate.py.** Warn-only at most: a parse bug there would block the 11-04 pinned reruns.
- **A `diagnose_due` rewrite.** The label in change 7 is enough while diagnosis is skipped on cost.