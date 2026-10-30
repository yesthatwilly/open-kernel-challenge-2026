# Campaign `2026-10-30-rank-ladder` (Rank Ladder v3): close the gap to the public answer, optimizing rank

**Window:** 10-30 ~15Z → kernel close 11-04 23:59 PT = **11-05 06:59Z**.
**Home:** `campaigns/2026-10-30-rank-ladder/` on master (PLAN.md, LOG.md, VT.md, GAP-BUDGET.md, LANES.tsv, research/).
**Approving this plan approves:** lifting the stop for this campaign, the VT rule, D (their file on gpunode06, measurement
only), and free H100 use on **gpunode06 only** (any GPU there; H200 and every other node stay off-limits).

## Priority rule (user, 10-30 ~14:35Z): H100 first, now

**Spend tokens only on moving H100 trials forward**, starting now, whatever the hour.
- **Allowed spend:**
  - building and running trials;
  - reading results;
  - writing a verdict (one CLAIMS line plus a short research note);
  - minting the next trial.
- **Judge checks:** only for unions that held on the H100, dispatched with a minimal attempt file.
- **Not spent on:** reports, inventories beyond what the next trial needs, or long write-ups.
- **Keeping the GPUs full:**
  - no gpunode06 GPU sits idle while a lane has a trial ready;
  - while a lane builds, its GPU runs a replicate or identity arm for another lane from the queue.
- **Agents:**
  - at most 6 at a time;
  - brief = `brief-build.md` plus a one-paragraph lane brief;
  - reply ≤ 30 lines;
  - start a GPU run within 15 min.

### First hour: fill gpunode06

Takes: about 4 lane agents of ~250–400k tokens each, plus D-moe/D-decode run from the main thread with no agent.

| GPU | lane | trial | ready? |
|---|---|---|---|
| 1 | D-moe | nodejudge: their moe file vs ours, per-kernel times on pinned skewed routing, identity arm | now, no build (main thread runs nj-exp) |
| 2 | D-decode | the same for decode | now, no build |
| 3 | moe | MO3 stage 2: N32 dequant-once twin, register gate first, then A/B vs master (byte budget ≤ 17.9 KB) | agent build, ~1 h |
| 4 | attn-sliding | VT d16s: read AT8's grid at B=16, sweep CTAs/request at ours / midpoint / 16 per request / 1 step past | agent, ~30 min |
| 5 | attn D | weight-free driver: their attention file vs ours, per-kernel, all 12 rows | agent, ~0.5 day; first rows within the hour |
| 6 | bf16 | VT split-K/stages on m16/m32 pooler/unpooler via the existing probe pattern, then the weight-stream rung D names | agent |
| 0, 2, 7 | queue | replicates, identity arms, sweep points (filled through hopper-queue) | as lanes emit specs |

GPU numbers are illustrative: assign from `nvidia-smi` at start, skipping any GPU another user holds.

## Resume (cold start, any session)

1. `date -u`; read `campaigns/2026-10-30-rank-ladder/LOG.md` **Now** block first. It is the only live state: boards,
   ranks, quota left, each lane's current rung, and in-flight job ids (okc jobs, hopper-queue jobs, agent names).
2. `LANES.tsv` holds one row per lane: lane | rung | worktree | branch | status (minted / building / h100 / check /
   confirm / landed / killed) | next action | updated UTC.
3. Reconcile before acting: `okc jobs --json` and `okc problems` (quota); `hopper-queue` status (STOP file, queue);
   `git worktree list`; harvest anything finished and write its verdict (CLAIMS) before dispatching new work.
4. Continue from each lane's "next action". Every dispatch updates LANES.tsv + the LOG Now block in the same commit.
5. Memory pointer: `okc-campaign-rank-ladder.md` (index line in MEMORY.md) names this directory and the resume steps.

## Quota (okc problems, read 10-30 14:29Z)

| problem | checks left today | scored runs left today |
|---|---|---|
| attention | 9 / 20 | 2 / 10 |
| bf16 | 13 / 20 | 5 / 10 |
| decode | 12 / 20 | 4 / 10 |
| moe | 14 / 20 | 6 / 10 |
| prefill | 18 / 20 | 9 / 10 |
| **total** | **66 / 100** | **26 / 50** |

10-30 so far: 34 checks + 24 scored runs. Ahead: 5 full days (10-31 → 11-04) × 100 checks + 50 runs, plus 11-05
00:00–06:59Z if the quota resets at 00:00Z (to verify at that time). About 566 checks and 276 scored runs remain
in all. Quota is not the binding constraint.

*Revised after an adversarial Opus review (verdict: sound with fixes). Its load-bearing evidence was spot-checked:
attn `submission.py:5479-5480,6138` (AT2 = 32 plain CTAs × 64-key tile), CLAIMS:101-103, FACTS:57 (judge noise).*

## Context

Ladder v2 (10-30) ran R → I → C → J against alygg77@6cae49a's known answers: ~45 rung verdicts, ~14 landings (31% hit),
29 checks + 21 scored runs, about ×1.19 in total. Attention 1.634 → 1.805, bf16 1.284 → 1.343, decode compliant
1.671 → 1.699, moe compliant 2.100, prefill 1.572 unchanged. The user then stopped new work; the stop holds until lifted.

| problem | us | answer (would rank) | gap | rank now | rank at ×1.10 / ×1.15 |
|---|---|---|---|---|---|
| attention | 1.805 | 2.653 (#65) | ×1.464 | 525 / 1130 | 393 / 349 |
| decode | 1.699 | 2.206 (#86) | ×1.298 | 469 / 1039 | 338 / 304 |
| prefill | 1.572 | 1.817 (#45) | ×1.156 | 274 / 691 | 176 / 55 |
| bf16 | 1.343 | 1.550 (#118) | ×1.155 | 466 / 1471 | 323 / 132 |
| moe | 2.100 | 2.402 (#64) | ×1.144 | 340 / 1186 | 234 / 56 |

The product of the gaps is ×2.90, about 89 landings at 1.2% each. At 10-30's rate, with no decay, that is about 6.2 days;
5.3 days remain. And day 1 used up the pre-mapped rungs. **Odds of reaching the answer on all five are about 5%, so the
goal is rank, not parity.** moe, prefill and bf16 each have a rank cliff about ×1.10–1.15 above us.

**Binding constraints:** supply of mechanisms that clear the 1.2% margin, H100 → H200 transfer, and one drained node.
Quota is not one: 10-30 used 29 of 100 checks.

**Standing rules:** public code is never adopted or copied; one mechanism (or one value) per rung, each a controlled
test on our code; compliance before speed.

## What changes vs v2

1. **VT (value transfer), a new C-tier campaign.** Their evaluated values only **seed a sweep range** on our knob. The value
   we keep is our own measured optimum, from a bracketing sweep: ours, midpoint, theirs, one step past. We never set the
   literal without the sweep. **The user rules on this before any sweep.**
   - **A "same knob" mapping is valid only if all of these hold:**
     - same work unit per CTA;
     - same merge scheme;
     - same occupancy regime (CTAs per SM, smem and registers within about 10%);
     - our value at that shape actually differs;
     - no other code edit is needed (otherwise it is a mechanism rung).
   - **What transfers:** grid and CTA counts transfer (both GPUs have 132 SMs). Stage counts and latency values often do
     not (FINDINGS §1).
   - **Correction:** the "AT2 at 32 splits" example was wrong. Ours already runs 32 CTAs × one 64-key tile, the same as
     theirs, so d1s is a per-kernel gap (≈7.4 vs ≈3.7 µs est.): a D question.
   - **First real candidate:** d16s. Theirs runs 16 splits per request (256 CTAs, 2 per SM); our AT8 runs about 1 CTA/SM
     (CLAIMS:103). This is unverified until the inventory reads AT8's grid. d32s is already swept (AT8b refuted, CLAIMS:102).
2. **D (differential profile), a new R-tier probe.** Per-kernel timing of their file and ours on identical inputs on the
   H100. It is measurement only, and **needs the user's OK**.
   - **Order:** attention first, via a weight-free driver (≈0.5 day). Then moe and decode via nodejudge, which needs gpunode06
     drained and cleaned before resume. Prefill and bf16 only if a lane needs them (each needs a weights driver).
   - **What transfers:** per-SM cycles, instruction counts and grid shapes. End-to-end ratios on memory-bound kernels do not.
   - Their attention carries H200-tuned `_TUNED` values (`kernels/muse_attention_decode.py:2934-2937`).
   - Never get judge profiles by checking their file.
3. **Gap budget per problem.** Every remaining × is assigned to a named rung with an owner row. Rebuild it from the latest
   scored jobs, because the 10-30 ledger predates AT2/AT8/AT3b. Contributions are ratio^share on row times, never kernel
   times. Fix the decode ledger's unmatched v32-full row (it explains ×1.253 of ×1.298) and give v32f its own rung.
4. **Pre-screen every rung against the transfer rules** (FINDINGS §1) before building.
5. **Land on unions, confirmed twice.**
   - Riders that hold alone wait for a per-problem union check.
   - A landing needs 2 judge runs, or a CI that clears 1.2%. One unchanged file has swung up to about 6% per row between
     judge runs (FACTS:57).
   - Promote from the scored sha only.
6. **Kill a rung at its first H100 miss**; never tune on the judge.

## Steps

0. **Set up (on approval).**
   - Create `campaigns/2026-10-30-rank-ladder/` on master: PLAN.md (this plan), LOG.md (Now block), LANES.tsv. Commit.
   - Memory:
     - add `okc-campaign-rank-ladder.md` and its MEMORY.md index line;
     - update `okc-hopper-probing.md` (gpunode06, any GPU, user 10-30) and `okc-agent-concurrency.md` (stop lifted for
       this campaign).
   - H100:
     - `hopper-probe status gpunode06` (check `nvidia-smi` for anyone else's processes before taking a GPU);
     - clear queue/reserved of our own holds (keep a hold only if another user is on a GPU);
     - restart `hopper-queue` (rm queue/STOP).
   - Keep setup to a handful of tool calls; then the first-hour table above. Steps 1–3 (VT inventory, D, gap budget)
     are done only as far as the next H100 trial needs them.
   - Today's remaining quota is spent only on unions that held on the H100. Leaving most of it unspent is fine.
1. **VT inventory** (CPU, read-only) → `campaigns/2026-10-30-public-ladder/VT.md`.
   - **One row per value:** problem | rows | their symbol@file:line | their value evaluated at the judge shape |
     our symbol@file:line | our value at that shape (judge profile or grid math) | match test | status (same-knob / equal /
     structural / swept CLAIMS:n) | predicted row Δ and headline × | rung id.
   - The user sees the map before sweeps run.
2. **D profile** (on the user's OK), in the order above → `research/diffprof-<p>.md`.
3. **Gap budget** → `GAP-BUDGET.md`, from fresh scored-job rows + D + VT.
4. **Lanes, ranked by Δrank per expected landing.** The first three run first; the rest run when a slot frees up:
   1. **moe:** MO3 stage 2 (N32 twin, register gate first; ≈ +3%). It needs a byte budget first: 982,138 B, ≈17.9 KB free.
      Then VT crossovers.
   2. **attn-full:** verify 16/32 full and decode 16 full, the largest open headline × (FINDINGS §5). D names the cause;
      AT16 only if D finds nothing cheaper.
   3. **bf16:** the m16/m32 weight stream (×1.06, the named gap), plus VT split-K/stages. BF5 already matches public per SM.
   4. **attn-sliding:** the d1s/v1s per-kernel gap from D; VT d16s.
   5. **prefill:** VT tile/slab values; it rides the union (PF3 ≈ +0.4%). No lever above that has been found.
   6. **decode-ports:** byte-capped (996,666 B, 3.3 KB free). A trim rung comes before each port.
5. **J and landing.** Per problem per day: union check → 2-run confirm → scored submit when it beats the board → promote
   from the scored sha. This is under the existing submit + promote grant, once the stop is lifted.

## Cadence (until 11-05 06:59Z)

- **00–12Z:** H100 builds, VT sweeps, D.
- **12–18Z:** judge checks of what held.
- **18–24Z:** unions, confirms, scored submits.
- **At most 4 rounds a day** (AT2 alone took ≈1.5 h of build plus H100 time), with at most 6 agents.
- **Milestones are rank targets:**
  - moe/prefill past ×1.10 (rank ≈ 230 / 175) by 11-02, ×1.15 (≈ 55) by 11-04;
  - bf16 ×1.10 (≈ 320) by 11-03;
  - attention ×1.15 (≈ 350) by 11-04.
- A problem two days behind its milestone gets a lane moved to it.

## Reuse

- `hyp.py new --user-requested` and `hyp.py h100-result` (run from `.worktrees/public-ladder`).
- `churn.py claim-add` and `churn.py register-landing ... --source h100`.
- The build brief in the scratchpad: `brief-build.md` (new worktree per rung; never switch a branch).
- `hopper-probe`, nodejudge `nj-exp`, `hopper-queue` (`gpus=N,`).
- Public file, read-only: `~/git/okc-muse-spark-kernels`. Cite it as github.com/alygg77/okc-muse-spark-kernels@6cae49a.

## Verification and risks

- **Every rung:** one mechanism or value, a pre-registered delta, an identity arm, exact outputs, and `hyp.py lint`
  passing.
- **VT.md:** every value is evaluated at the judge shape.
- **D:** per-kernel sums reconcile to the judge row totals within the replay overhead.
- **Budgets:**
  - prepare ≤ 300 s (rule :98; we aim for 240);
  - attention import is 306.7 s against our own 480 s clock (`submission.py:7286`), so any new in-import race needs a
    time budget;
  - moe and decode byte budgets are checked before every check.
- **Before every check:** an attempt file plus the rules checklist.
- **Daily:** boards and ranks against the milestones in the LOG Now block, and a CLAIMS line for every verdict.

## Decisions

- Settled by the user (10-30): H100 on gpunode06 only, used freely.
- Settled by approving this plan: lift the stop for this campaign; the VT rule; D for measurement only.
- Still the user's: anything outside gpunode06; ending the campaign early.
