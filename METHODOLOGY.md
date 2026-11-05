# Methodology

How this run was organised, what the operating rules were, and how the philosophy changed between
2026-10-23 and the close on 2026-11-05. The rules below were set by the operator and carried
out by Claude Code sessions running autonomously for most of each day; dates are UTC and the quoted
rulings are verbatim from the session record.

## 1. The unit of work is the hypothesis

From the first day (10-23) a change to a kernel was never "an edit". It was a hypothesis: one named
mechanism, one file, one judge run, one record. The machinery:

- **`/hypothesize`** mints a hypothesis file with the mechanism, the rows it should move, the prior
  evidence and a predicted gain. From 10-28 it may only mint a lead that clears a value bar (measured
  evidence, at least 2.5% headline if it works, gain × probability at least 1.5%); minting nothing is
  a valid outcome.
- **`/churnsubs`** runs the queue: a worktree and branch per hypothesis, the code change, lint, the
  attempt note, the judge send, the harvest into `done/{gain,neutral,regression,failed,rejected}`,
  and the promotion of a proven gain to master. It is the only thing that commits kernel changes to
  master.
- **Adversarial review** before any claim: a fresh subagent reads the hypothesis and its result and
  asks whether the gain is confounded by something already landed, and whether an A/B companion
  (the change with another change reverted) is needed. It may queue the companion. A claim with
  `review != proceed` is refused.
- **The attempt note** (`<problem>/attempts/<stamp>-check|submit-<tag>.md`) is written before every
  send and names the goal, the exact change, the rules checklist, the risks and the pre-registered
  gate. A pre-tool hook refuses `okc check` and `okc submit` unless the note exists with the literal
  `<job id>` placeholder, which is filled in afterwards. 468 of these exist.
- **CLAIMS.md** per problem is the append-only ledger: one line per mechanism, tagged `landed`,
  `lead`, `shelf`, `unresolved`, `node-only`, `refuted` or `superseded`, with the rows, the evidence
  (job ids, commits, H100 run ids), the board at the time and a reopen condition. A later line about
  the same mechanism supersedes the earlier; nothing is edited or deleted. A `refuted` line is binding
  until its reopen condition holds. This is what stopped the same idea being tried twice across 13
  days and several hundred agent sessions.

## 2. Quota is spent, never hoarded, and never wasted

The judge allowed 10 scored runs and 20 checks per problem per day. The standing rule from 10-23 was
to spend both pools when prudent: checks first to answer questions, scored runs only to put a proven
read on the board. Identical bytes are deduplicated by the judge, so a repeat read needs a nonce
comment on line 3; "re-rolling" the same file for a luckier read was banned on 10-26, allowed again as
a deliberate choice in the final 24 hours when nothing else was left, and banned again by the operator
at 23:20Z on 11-04 ("dont do luck draws").

From 10-27 the day had a shape, the **3:1 rhythm**: from 00:00Z the biggest decode bets on
high-effort multi-agent workflows while there is quota and time to iterate; then the other problems'
smaller gains; then from 18:00Z every remaining check on every problem and the next day's plan by
23:00Z.

## 3. Measure on a proxy first, and know what the proxy can and cannot see

The judge is an H200 SXM. We had H100 nodes and an H200 NVL on the ASU Research Computing cluster
(Sol), used only inside the operator's own Slurm allocations through lane scripts a watcher picked up
per job. Harnesses:

- `smoke.py`: the twelve judge shapes, candidate against two copies of master (ref and dup), so the
  master-vs-master spread is the noise band for every row.
- `smoke_cold.py`: the same with L2 flushed between replays, because the judge's calls are cold.
- the robust set: 162 shapes beyond the timed twelve (odd batches, mixed page spreads, long contexts),
  pass/fail on correctness only.
- `judge_sim.py`: the reference check on every replay, including decode's routing check (a token
  that drops an expert over the 0.03 limit fails the run), the prepare time against the 280 s budget,
  and a determinism screen.

What transferred and what did not was learned the hard way and is written down in the ledgers:
structural changes (a different kernel, a fused launch, a different split) transferred from the H100
to the judge; bandwidth and scheduling knobs (pipeline depths, cluster modes, prefetch) often flipped
sign, so from 10-31 a config swap needed an H200 cold-L2 read before a judge check, and from 11-04 a
judge gate besides. Five screening designs on 11-04 (Plackett-Burman and 2^(k−p) resolution-IV
factorials, 7 to 10 knobs per problem, 12 to 16 points, replicated lanes, a pre-registered analysis)
showed every remaining knob on every problem inside noise.

## 4. Gates are pre-registered, and landing is by the scored sha

A candidate reached the judge only after passing its node gate (all tests, routing check, no row
below 0.985 to 0.99 of master-dup, headline over a bar set in the attempt note). A judge check that
beat its gate earned a scored run; a scored run that held on three reads landed on master as the
exact scored bytes (`cmp` against the judge's blob), registered with its job id. Rules that refined
this, each a dated operator ruling:

- **Robustness fixes land even at a speed cost inside noise** (10-31, after a real ring-slot race
  fix missed its speed guard by 0.0006: ".0006 is nothing, but i appreciate the real race it
  addresses").
- **Verified gains land even when the read is below the board** (11-02). The board keeps a team's
  best read; once gains were rider-sized (0.1% to 2%) and read noise was ±1% to 2.5% per row, a real
  gain rarely beat an old lucky read, but it still makes master faster as the base of every later
  carrier. Lucky compliant board entries are never withdrawn (the kernel boards carry no prizes and
  only top entries are rerun).
- **One mechanism per rung**, never bundled, so each claim is attributable; carriers that bundle
  proven riders are allowed only after each rider has its own record.

## 5. Compliance before speed

The operator's ruling on 10-30, when a probe found our decode router feeding a BF16 activation where
SGLang uses FP32: "always remain rules compliant. any findings that endanger that we can leave out of
our work ... STAY COMPLIANT." Every send since went through a rules checklist in its attempt note, and
`okc_lint.py` scans for host syncs, network use, caching of input-derived data and encoded payloads.

The one disqualification taught the rest: on 11-03 the judge removed a bf16 run for "compressed or
encoded source that's decoded at run time", a rule added to the live rules after our local mirror was
taken. The public bf16 lineage embedded its CUDA as zlib+base64 strings. Every payload was turned into
readable source with the compiled code byte-identical, the board row was rebuilt from that file, and a
grep for `zlib|base64|b64decode|decompress|lzma|bz2|fromhex|marshal|cubin` became part of every send.
The same update allowed BF16-accurate approximate math, FP8 P at power-of-two scales and FP16
accumulation in attention, which reopened the public files' lazy softmax that we had refused on 10-31.

## 6. The cornerstone change: openly adopting other people's open-source work

This is the philosophical turn the run is remembered for, and it happened in three steps.

**Step 1, 10-30: public files as targets, never as a base.** Public repositories (alygg77, vjeux)
were judge-proven faster on every board (attention 2.65× vs our 1.63×, decode 2.21× vs 1.72×, bf16
1.55× vs 1.28×, prefill 1.82× vs 1.57×, moe 2.40× vs 2.08×). The operator declined to adopt them:
"the point of not simply adopting their submission is a desire for the process to be fully data driven
... i only want to arrive at a more favorable position because i was able to do so through the
integration of systematically implemented knowledge". The plan was a "rank ladder": their per-row
times as known values, our own mechanisms tested one rung at a time toward them, our own findings
allowed to disagree.

**Step 2, 10-31: the honest inventory, and the pivot.** A check-only track ("public-plus") put the
public files and ours side by side on the same day's judge. Result: ours was faster on 0 of 68 rows;
116 of our findings agreed with their structure and 43 of our refutations were contradicted by it; of
48 candidate grafts from our work onto theirs, 1 survived review. The structural gap was ×1.13 to
×1.32 on every problem, and a day of trying to transfer findings had not moved it. At 05:10Z the
operator ruled:

> "from this point on, im embracing the MIT license as i rebase this systematic improvement system
> onto the public board. there is no hope for simply transferring the knowledge and findings on the
> initial repo; using this board and attempting to improve it with a good system is now the goal. i do
> not expect this to make the serving board, so this is strictly an academic exercise to see if a
> systematic approach can improve on a public board enough to be competetive. adopt any and all the
> problem solutions as the new foundation."

The licence turned out to be Apache-2.0 with an MIT portion; either way the obligation is attribution,
and it was met: upstream LICENSE, NOTICE and commit vendored under `third_party/`, line 2 of every
derived file naming the lineage, our own pre-adoption files tagged `ours-final-1031`. The public files
were not sent raw where they broke our compliance reading (rounded-P denominators, lazy max): they
were patched into "public-C" versions first, and the boards moved that morning from 2.005 to 2.434
(attention), 1.572 to 1.784 (prefill), 2.100 to 2.367 (moe) and 1.343 to 1.566 (bf16). Decode stayed
ours until a compliant public decode existed, then went 1.703 to 2.179 the same day.

**Step 3, 11-02 to 11-04: a study of the ecosystem, and adoption as a normal move.** The operator
asked for a study of trxphxle's repository ("welcoming a full addressing of the owners and their
license, and i expect this repo to show it too"), then of keell0renz's, whose judge-timed files beat
our masters by 1.6% to 5.2%. On 11-04 the ruling became general: the final entries are "the
culmination of many open source approaches ... fully expecting to attribute many repos, as much as is
necessary". A public base is adopted once it passes the same gate as any hypothesis; our riders are
ported onto it; its upstream notices travel with it. The last landings of the run were exactly that:
keell0renz's decode, prefill and bf16 files adopted with notices on 11-03, their attention merged with
our own batch-1 build on 11-04, and our MoE riders re-ported onto their MFIN kernels.

Two constraints came with the openness:

- **Grafts must generalise.** A faster kernel for a size class is a legitimate graft; a branch on
  the judge's exact tested shape is not, and the organizers' 11-03 update warned against judge-fitted
  variants. The rule "size-class routing only, never a table of the judge's shapes" was applied to
  every graft, and a candidate built with a judged-shape table (RBX) was rebuilt before it was tested.
- **Untrusted code is screened before it runs.** Public files were read for host syncs, network
  use and encoded payloads before any execution on the cluster, and agents that read public code were
  kept off the clean-room branches while a clean-room track still existed.

## 7. What the systematic approach did and did not buy

What it bought, measured on the judge: overall rank 480 of 1,163 on 10-26 to 147 of 2,014 at the
close (139 four hours before it, before other teams' last sends landed); every problem between
1.26× and 1.74× over its 10-26 start; a ledger of 792 mechanism claims with evidence; and about thirty
audited landings of our own worth 0.1% to 11% per row (the largest: cluster
push-merge and the one-CTA-per-split head layout on batch-1 attention, each about +10% on its row;
work-queue reorder on prefill +1.6% headline; the FRX ring reuse on moe +0.5%).

What it did not buy: a kernel of our own that beat the best public one on any problem. The final
day's attempt to graft the attention board's kernels into the decode layer ended at its bail-out
gate: a source diff showed decode already contained the same kernel. The closing judgment, in the
decode ledger, is that the public kernels are at their tuned values, the remaining riders are inside
judge noise, and the gap to the 100th rank (2.308× on decode at the close, 1.1% above our board read
and about 2.6% above master's true mean) needs a new mechanism of 1% or more that nobody on the public
boards has published.

## 8. Operating lessons worth keeping

- Write the plan so a cold session can run it: a `PLAN.md`, a `LOG.md` whose top block is the live
  state, and exact resume commands. Sessions were summarised and restarted dozens of times.
- Delegate execution to subagents with a narrow brief and a bail-out condition, and keep the main
  thread for decisions. The 11-04 graft study ran as three agents (interface, bytes, moe scout) and
  stopped in an hour because the interface agent could say "nothing to graft".
- Never block the loop on a decision; record it in `PENDING_USER.md` and continue with what does not
  depend on it.
- Always compare against two copies of master in the same run. A candidate that beats `ref` but not
  `dup` has measured noise.
- A single judge read has a standard deviation near 0.6% on these problems. Three reads before
  landing; two checks before a scored run; and a pre-registered bar before the first read.
- Keep attribution as a first-class artefact: a vendored notice directory, a header line per file,
  and a README index. It costs nothing and it is the condition under which open adoption is honest.
