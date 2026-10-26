---
id: 2026-10-26-decode-first-rewrite
status: closed
objective: overall board rank (Muse Spark Open Kernel Challenge, kernel track)
kpi: muse_decode_layer headline (overall rank tracks decode rank, Spearman 0.996)
baseline: decode 1.3164 board (eaa8cc0f) / 1.3142 master (a903b9d0); overall rank 480 of 1163
branch: campaign/decode-first-rewrite
worktree: .worktrees/decode-first-rewrite
start: 2026-10-26T04:38Z
deadline: 2026-11-05T06:59Z (freeze 2026-11-04T18:00Z)
approved: 2026-10-26 by the user ("execute the plan")
---

# Campaign `2026-10-26-decode-first-rewrite`: a single-agent, decode-first plan

Durable home: `campaigns/2026-10-26-decode-first-rewrite/` on master. It holds the plan, findings, log, research and run archives. The code lives on branch `campaign/decode-first-rewrite` in worktree `.worktrees/decode-first-rewrite`.

## Context

- **Where the work stands.** About 250 churn hypotheses since 10-23, one modest change each, mostly neutral. Wins land at +0.5–2% while leaders are 1.3–1.9x ahead.
- **Time left.** The kernel track closes **2026-11-05 06:59Z** (Nov 4 23:59 PT), about 10 quota days away.
- **What you asked for.** A findings-informed plan for an ambitious, largely single-agent rewrite on a new branch, then its implementation.

**Your decisions (2026-10-26)**
- Objective: the **overall board**.
- **Churn paused.** All quota goes to the rewrite; unspent checks are accepted.
- One agent writes all code, plus a fresh **Opus audit** of each new CUDA module before its first check. Nothing else.
- Agent fan-out stays small (≤3 concurrent) to protect your token quota.

**Standing on 2026-10-26** (via `okc leaderboard`)

| problem | ours | rank | r200 | r100 | r50 | leader |
|---|---|---|---|---|---|---|
| decode_layer | 1.316 | 448/704 | 1.628 | 1.800 | 1.948 | 2.360 |
| moe_w4a16 | 1.402 | 490/797 | 1.861 | 2.050 | 2.174 | 2.554 |
| attention_decode | 1.535 | 373/755 | 1.825 | 2.036 | 2.279 | 2.963 |
| bf16_proj | 1.257 | 319/1019 | 1.307 | 1.421 | 1.509 | 1.637 |
| prefill (own board) | 1.468 | 144/433 | 1.368 | 1.558 | 1.644 | 1.963 |

Overall rank: 480 of 1163.

## 1. Findings that drive the plan

### 1.1 The overall board is effectively the decode_layer board
- Over 632 teams, ordering by decode rank alone gives Spearman 0.996 against the overall order, with the fewest order violations of any formula tried. Mean rank and summed time were both worse.
- alpinevm solved only moe and decode and is overall **#16**, exactly its decode rank.
- Holding our other three fixed, overall top-300/200/100/50 needs decode at about **1.45 / 1.61 / 1.78 / 1.93x**.
- **decode_layer is therefore the KPI.**
  - moe, bf16 and attention serve as testbeds with their own quota and isolated timing, and as feeders of kernels.
  - Prefill gets only ports.

### 1.2 Where decode_layer's time goes
Source: board profile `a903b9d0`, share of GPU-busy time per shape.

| shapes | experts (Marlin w13+w2, act, combine) | dense GEMMs | attention | small ops |
|---|---|---|---|---|
| T=1 (d1) | 36–39% | 38–39% | 11–16% | 10–11% |
| T=8..32 (v1, d16, d32) | 27–53% | 15–31% | 9–51% (full layers) | 5–14% |
| T=128/256 (v16, v32) | 45–70% | 12–17% | 3–35% | 9–10% |

- On active-expert bytes, Marlin runs at **1.2–1.9 TB/s**. It is worst at T=1 (about 2–3 local experts) and at T=128/256, where hot experts are re-streamed per m-block. At T=32 it uses bsm 16, the same as SGLang, so d32 sliding gets 1.087x.
- Scenario model (exact load math, current base):

| scenario | headline | ≈ rank |
|---|---|---|
| experts ×0.5 | 1.76 | ~150 |
| experts ×0.4 | 1.88 | ~75 |
| experts ×0.5 + dense | 1.91 | ~75 |
| + small ops ×0.5 | 2.07 | ~25 |
| dense alone | 1.40 | |
| full-layer attention alone | 1.35 | |
| small ops alone | 1.39 | |

- The decode-specific read caps the non-MoE program alone at about 1.36–1.42. The experts must change.

### 1.3 Root causes of the plateau (measured)
1. **NVFP4 consumer, not DRAM.** Every NVFP4 kernel in the repo shares master's `dequant8` + cvt scale path.
   - It costs about 3.5 issue slots per element; at 4 TB/s that needs **72–93% of the INT32 ALU pipe**.
   - The wg/rega consumer serialises dequant → 8 wgmma → `wait_group<0>` per stage, with both warpgroups in lockstep: about 1150–1200 clk per 9216 B stage per SM.
   - The rowowner ring kept only about 55 KB/SM in flight; bf16 streams need ≥96 KB.
   - Result: all of them sit at 1.5–2.5 TB/s.
2. **Measurement bias.**
   - Every prepare gate and timer flushes L2 with `flush.zero_()`, a write, while the judge reads. This leaves about 50 MB of dirty L2, inflating ≥37 MB streams by 2–5 µs and pushing ratios toward 1.
   - Many "kernel-only" TB/s numbers are eager single launches.
   - Prepare timings undercount the judge by 20–25%.
   - The verdicts "consumer is not the limiter" (wait<1>) and "bytes in flight is not the limiter" were measured in starved regimes.
3. **Self-inflicted stranded wins.**
   - Decode cells S1 A/B, SKW v1/b32 and S3 fell back on gates of the form "vs master ≤ 1e-2", even where the new kernel was closer to fp64 than master.
   - Master's own S3 c1-b16 gate flips run to run.
4. **Split-K plumbing, not mainloops, sank m≥128 dense GEMMs.**
   - The CUTLASS DataParallel mainloop runs at 85–92% of SM peak on the judge.
   - The Deterministic StreamK fixup plus an `initialize_workspace` memset node add 10–32 µs.
5. **Port lag and board/master divergence.**
   - Decode still runs Marlin at every T, nvjet at T≥8, and has stranded attention cells.
   - The attention and moe boards are branch code, not master.
   - Decode's board (eaa8cc0f, 1.3164) equals master a903b9d0 (1.3142) on the scored kernels.

### 1.4 Bad counterinteractions → guardrail
| interaction (evidence) | guardrail |
|---|---|
| SGLang kernels (Marlin, boundary) break PDL chains; PDL only pays between own kernels (P after boundary +0.3 µs, own chain −7.8 µs) | own the whole T≤32 chain before tuning launch gaps |
| Side streams contend: L2-prefetch side stream +21% at d16s; T>32 router races the pooler | no side streams in new code; at T>32 measure join slack first |
| Gates derived from shared constants (ALIGN_FUSED_MAX_TOKENS) | one bound constant per path |
| Module-global TMA maps across the judge's 2 state copies (g_maps) | per-state pointers and descriptors; 2-state alternation canary |
| Cross-CTA spins → bimodal judge loop (bs1 persistent) | last-arriver tickets only, never spins |
| moe verdicts do not transfer to decode (rega won in moe, gated OFF in decode B128/B256) | decode selectors time decode routings (soft, about 2/6/7/16 active experts) |
| Kernel-vs-kernel comparisons mislead (S1 14.6 vs FA3 13.7 µs while the whole FA3 chain was 25–29 µs) | selectors time the whole replaced path |
| K-halves "fp32 partials" actually routed **bf16 partials** | rules risk: never promote de95f817 / caaa76cc; fp32 accumulation everywhere SGLang has it |
| An illegal instruction in prepare poisons the context; no child processes | canary, -lineinfo, proven instruction forms only, first run of any new module is a check |

### 1.5 Unleveraged (ranked by decode ROI)
1. A NVFP4 engine with a cheaper **exact** dequant:
   - integer scale construction (`PRMT` + `IMAD`, no cvt), with the 2^126 bias split as code 2^-126, scale 2^119, fp32 epilogue 2^7;
   - a lossless bit permutation of codes so 3 of 4 pairs cost one shift + LOP3.
   - About half the INT ALU per element, zero conversions. Proven exact in the forensics report.
2. Read-once expert streaming at both ends: a T≤32 CTA-shared bulk ring with ≥128 KB/SM in flight, and a T≥128 hot expert read once (NMAX ≥ routes).
3. Judge-faithful, deterministic gates: fp64/fp32 reference with the candidate's **own** routing, read flush, graph replay.
4. An own-kernel T≤32 decode chain: R and boundaries folded into neighbours, and static weights (o, pooler, router, unpooler) prefetched before `griddepcontrol.wait`, recovering about 30 µs of idle DRAM at T=1.
5. Dense m=128/256: one M-tile, x multicast on a 2-CTA cluster, static split-K ≤132 CTAs, last-arriver reduce with no memset node. CUTLASS 4.5 builds on the judge in 49 s.
6. Glue at T>32: 26–38 µs of small hidden kernels; bench time exceeds the kernel sum by 15–20 µs on d16f, v16f and v32s.

### 1.6 Do not try (closed with evidence)
- Triton NVFP4 GEMV.
- Swap-AB persistent mma.sync with `__syncthreads` per stage.
- Grid-spin fused bs1.
- Split-I with fp32 y `red.add` (fsi midt 0.44–0.81x).
- Rega cluster multicast (MC2 1.3–1.9x slower).
- Ping-pong consumer warpgroups.
- Producer-side scale expansion.
- Deeper rings on the wg consumer.
- Lossless bf16 compression.
- Full-layer attention rewrites: sk is at 1.75x its KV floor, and every verify kernel lost to FA3.
- DeepGEMM: its API runs cuBLASLt here, and its raw kernel has no split-K.
- CUTLASS Deterministic StreamK.
- Marlin small-M knobs beyond the queued bsm/two-stream/hot-cold cells.
- Align and post-norm micro-work.
- Pairing on sliding layers.

## 2. Strategy and targets

Decode-first, fed by two NVFP4 engines and an own-kernel layer chain.
- Each engine is proven on moe (same expert shapes, separate quota, isolated timing), then switched on in decode behind decode-routing selectors.
- Small direct gains on the moe and bf16 boards come for free.

| milestone | content | decode target | ≈ overall rank |
|---|---|---|---|
| M0 (day 1) | W0 infra + W1 gate fix/restack | ≥ 1.335 | ~430 |
| M1 (day 3–4) | Engine A live at T≤32 in moe and decode | ≥ 1.48 | ~290 |
| M2 (day 5–6) | Engine B live at T=128/256 | ≥ 1.60 | ~210 |
| M3 (day 8) | own T≤32 chain + dense/glue ports | ≥ 1.72 | ~130 |
| Stretch (day 9) | upper-range engine numbers | 1.85–1.95 | 50–80 |

Side boards if the engines land: moe 1.40 → 1.8–2.1; bf16 1.257 → about 1.31–1.35.

## 3. Campaign home, branch and tooling (Phase 0, day 0)

**`campaigns/` is the durable home for programs larger than a hypothesis.** New, on master.
- `campaigns/README.md` defines the convention.
  - A campaign is a multi-day, multi-problem program with its own plan, findings, log, branch and run archive.
  - A hypothesis is one attributable change run by churn. A campaign may pause churn and may later spawn hypotheses.
  - Each campaign gets one directory, `campaigns/<YYYY-MM-DD>-<slug>/`.
- `campaigns/INDEX.md` lists campaigns with status (planned / active / paused / closed) and their best results.

**This campaign: `campaigns/2026-10-26-decode-first-rewrite/`**, committed on master.

| path | contents |
|---|---|
| `PLAN.md` | this plan, expanded with the evidence tables. Frontmatter: id, status, objective, KPI, branch, worktree, start, deadline |
| `FINDINGS.md` | the evidence base: overall-board fit, decode anatomy, root causes, counterinteractions, unleveraged areas, do-not-try list, user decisions |
| `research/` | the three agent reports verbatim (decode read, NVFP4 engine forensics, dense-GEMM forensics), extracted by script from this session's agent outputs; `board_fit.py` and `roi_model.py` with their printed outputs; the 2026-10-26 leaderboard JSON snapshots |
| `LOG.md` | append-only running state: every check, its job, rows that moved, decision. Every session resumes from here |
| `runs/<job>.*` | okc json, profile and log archive for the campaign's runs (as churn's `archive/`) |
| `RETRO.md` | written at close |

**Branch and churn pause**
- `git worktree add .worktrees/decode-first-rewrite -b campaign/decode-first-rewrite master`.
- The main tree stays on master; `.worktrees/` is gitignored. Campaign docs and run archives are committed on master from the main tree; churn is paused, so there are no conflicts.
- The okc attempt gate resolves `attempts/` per checkout, so `<p>/attempts/` inside the worktree just works.
- Pause churn:
  - Do not restart `/loop churnsubs`.
  - Run one harvest-only `churn.py tick` (no dispatch) for the in-flight prefill run.
  - Record the pause in `hypotheses/DECISIONS.md` and `PENDING_USER.md`, pointing at the campaign.

**Code layout on the branch.** It merges to master when milestones land.

| path | contents |
|---|---|
| `rewrite/lib/rt.py` | shared runtime (W0) |
| `rewrite/lib/cuda/*.cuh` | Engine A/B, dequant, PDL/bulk helpers |
| `rewrite/src/<p>.py` | per-problem sources with `#@include lib/...` markers |
| `rewrite/build.py` | expands includes into `<p>/submission.py` ("GENERATED" header), deterministic; then runs `.claude/skills/churnsubs/bin/okc_lint.py`. Starts as a copy of each master file, so the build is a no-op until engines are added |
| `rewrite/bin/report.py` | per-row paired ratios vs a base job, kernel names, PATH CHOSEN/GATE/DIAG lines in ≤15 lines |

## 4. Workstreams: named continuous-improvement paths, ROI order

Each workstream has a metric, a target and a kill criterion, and is measured on every relevant check.

### W0: Measurement and gate infrastructure (day 0; prerequisite for everything)
`rewrite/lib/rt.py` provides:
- **`read_flush()`**: 256 MB read reduction into one word, replacing every `flush.zero_()`.
- **`select()`**:
  - graph replay;
  - 41 interleaved reps;
  - empty-graph baseline subtracted;
  - 2 rounds ≥ 1.03 (or ≥ 1.02 with IQR ≤ 1.5× incumbent);
  - prints both medians;
  - asserts the launched kernel name via a one-call profile.
- **`check_vs_ref()`**:
  - fp64/fp32 reference computed with the candidate's own routing ids;
  - bounds equal to a fraction (≤0.5) of TASK.md tolerances, plus the 5× per-row rule and the near-zero-row rule;
  - never compares against master.
- **`canary()`**: per-launch prints, finiteness that names the first bad tensor, forced buckets (one-expert, all-local, skewed, zero-local, every N bucket), and 2-state alternation with refilled inputs.
- **Compact log helpers**: everything important within the 16 KB tail; PATH CHOSEN printed last.

Metric: prepare-vs-judge agreement per selected cell (target within 5%).

### W1: decode gate fix and stranded restack (day 0–1)
- Move decode's combo/S1/S3/SKW gates to W0 `check_vs_ref` and `select`.
- Restack, from `hyp/muse_decode_layer/sliding-sk-window-port-decode32-verify1`:
  - S1 A/B (sliding B=2..32);
  - SKW v1/b32;
  - S3 cells.
- Add cells: w13 atomic at T=1, Marlin bsm 8 at T=17..32, hot/cold bsm at T=128/256.
- Estimate +1.5–3.5% (d32s 239.5 → ~231, v1s 181 → ~172, v16f/v32f −8 to −10 µs).
- Kill a cell if its row does not move past drift in a check with all cells on.
- **Scored submit at M0.**

### W2: Engine A, NVFP4 at T≤32 routes/expert (day 1–4): the #1 lever
**Step 1: the ladder.** One moe check whose prepare prints rungs R0–R6, for an early verdict on dequant/consumer feasibility:

| rung | measures |
|---|---|
| R0 | read vs write flush |
| R1 | pure 1-D bulk stream at 2–3 CTAs/SM |
| R2 | + proposed dequant vs master's |
| R3 | + mma.sync / wgmma `wait<1>` at full grid |
| R4 | + real B operand |
| R5 | + epilogue |
| R6 | real kernels vs Marlin on judge-like and random routings |

- `dq_exact` must be 0 mismatches (16 codes × 119 scale bytes).
- The exact print format is in the forensics report.

**Step 2: the kernels.** Three kernels chained with PDL:
- **K1, w13 + SiLU.**
  - Builds the device work map from `topk_ids` and zeros `ss`.
  - Gate and up rows are interleaved per 16-row tile.
- **K2, w2 + per-route sum of squares.** Its weight ring is prefetched before `griddepcontrol.wait`.
- **K3, post-norm combine.**

Shared design of K1/K2:
- A CTA-shared 18 KB 1-D bulk ring, ≥128 KB/SM in flight.
- mma.sync m16n8k16 swap-AB; each warp owns 16 rows over the full K.
- Stream-K balance with a no-spin last-arriver ticket.
- Prepare-time lossless repack into fragment order plus the bit layout; scale bytes are only permuted.
- h and y stay bf16; fp32 accumulation; global scale applied in the fp32 epilogue.

**Step 3: rollout.** Enable per class with W0 selectors: moe bs 1/8/16/32 first, then decode T=1, 8, 16, 32 on decode routings.

| | value |
|---|---|
| Metric | kernel-only TB/s (R6) and expert-block µs per T |
| Target (moe) | d1 46 → 26–29, v1 104 → 73–84, d16 151 → 101–117, d32 199 → 134–156 µs |
| Decode effect | about +13–15% headline (expert block ×0.55–0.6 at T≤32) |
| Kill a class | R2 (dequant) < 3.0 TB/s with read flush after the proposed dequant, or R6 worse-of-two-routings ≥ 0.98 of Marlin after 3 checks; that class keeps Marlin |

### W3: Engine B, NVFP4 at T=128/256 (day 3–6)
- Start from moe's judge-proven wg/wg2 code.
- **Consumer:**
  - BK=64 stages;
  - double-buffered register A with `wgmma.wait_group<1>` (measured at full grid, not on the starved canary);
  - the W2 dequant.
- **Cells:**
  - N≤64 at 2 CTAs/SM (setmaxnreg 32/104);
  - N=128/256 at 1 CTA/SM with NMAX ≥ max routes, so hot experts are read and dequantized once;
  - LPT order (hot experts first) and a stream-K tail.
- Cold experts may route to Engine A.
- No multicast, no ping-pong.

| | value |
|---|---|
| Target (moe) | v16 323 → 215–245, v32 358 → 235–265 µs |
| Decode effect | v16/v32 expert block ×0.65–0.75 → about +7–9% headline |
| Kill | B256 not ≤ 0.85× of the incumbent after 3 checks; fall back to Marlin hot/cold cells |

### W4: own-kernel decode chain at T≤32 (day 5–8)
- Fold R (routing/align) and both boundaries (`muse_spark_norm_residual`, 3.4–8 µs each) into neighbouring own kernels.
- Build an O kernel that scales to T=32 (today `dl_o` runs at 0.41 TB/s at T=32).
- **PDL on every link**, with static-weight prefetch before the wait.
- T=1 ports:
  - bf16 bulk-A2 qkv (13.6 → 10.5 µs row);
  - swapab o;
  - bulk-style P (pooler + router);
  - ouns unpool;
  - attention per-kvhead cluster / c320 inside piece A where its selector wins.
- Estimate +4–6%.
- Metric: shape µs minus the expert block, per T. Each piece has its own selector; drop any piece that loses.

### W5: dense m=128/256 GEMM and T>32 glue (day 6–8; bf16 is the testbed)
- **One-check diagnostic:** CUTLASS with Nondeterministic fixup and no memset. If qkv m=256 ≤ 17 µs, the reduction diagnosis is confirmed.
- **Kernel:**
  - all tokens in one M-tile × 128 weight rows, BK 64;
  - TMA with x multicast across a 2-CTA cluster;
  - static split-K ≤132 CTAs;
  - last-arriver fp32 reduce with self-resetting per-state counters, no memset;
  - tile-contiguous weight repack.
- **Effects:** bf16 +3.5–4.5%; decode about +0.9%.
- **Decode glue:** router + pooler concatenated GEMM at T≥128, pre-expert norm folded, side stream removed. +0.5–1%, after a join-slack measurement.

### W6: attention (no new kernels)
Only the W1 restacks and W4's T=1 ports. Full-layer attention and verify rewrites stay closed.

## 5. Schedule (UTC days; quota resets 00:00Z)

| day | work | checks (decode / moe / bf16) | scored |
|---|---|---|---|
| 10-26 | Phase 0, W0, W1 build; moe ladder check | 2 / 1 / 0 | — |
| 10-27 | W1 restack; Engine A K1/K2/K3 build + audit; first moe checks | 4 / 4 / 0 | decode M0 |
| 10-28 | Engine A iterate in moe; begin decode T=1/8 integration | 4 / 6 / 0 | moe if it gains |
| 10-29 | Engine A decode T≤32 rollout; Engine B build | 6 / 4 / 0 | decode M1 |
| 10-30 | Engine B moe | 4 / 6 / 1 (W5 diagnostic) | moe |
| 10-31 | Engine B decode T=128/256 | 6 / 3 / 0 | decode M2 |
| 11-01 | W4 chain stage 1 (R and boundary folding, O at T=32) | 6 / 1 / 2 | — |
| 11-02 | W4 PDL prefetch + T=1 ports; W5 kernel | 6 / 1 / 3 | decode |
| 11-03 | W5 into decode; T>32 glue; nearby-shape sweep | 6 / 2 / 2 | decode M3, bf16 |
| 11-04 | Freeze by 18:00Z; final rules audit; final scored runs per problem; land on master | 2 / 1 / 1 | all |

Planned spend is well under the quota (20 checks / 10 scored per problem per day); unspent checks are accepted. Any slip moves W5 out first, then W4 pieces; W2/W3 are protected.

## 6. Single-agent operating protocol
- **Loop per change:**
  1. Write the design note in the campaign `LOG.md`.
  2. Edit `rewrite/src` or `rewrite/lib`.
  3. `build.py` + `okc_lint`.
  4. CPU dry run: `py_compile`, import of the Python paths, every PATH CHOSEN branch exercised where possible, grep for bare names in f-strings.
  5. For a new or changed CUDA module: one Opus audit subagent, read-only, using the churn AUDIT_PROMPT focus list (gate always-fallback, sm_90a/c++20 flags, per-state pointers, canary coverage, spins).
  6. Attempt file from ATTEMPT_TEMPLATE with the rules checklist ticked against the diff.
  7. `okc check` from the worktree.
  8. Archive, then `report.py` against the base job.
  9. Log the outcome and decide.
- **Every check must answer a stated question and carry a DIAG/ladder.** Never spend a check that only re-measures master.
- **Judging:**
  - Per-row paired ratios plus kernel names plus PATH CHOSEN lines.
  - Headline gain margin 1.24% for one run.
  - Decode d16s drifts ±5%: read it only with its kernel rows.
  - Runs on stale bases are read only for the changed rows.
- **Scored submits:** when a check beats the board by the margin and the rows confirm, submit the same bytes. Check-vs-scored is not deduplicated. Log each in DECISIONS.md.
- **Landing:** at M0–M3, commit the generated submission files to master with a DECISIONS note. Master stays the best known code, and BEST.json is updated through `churn.py` where its commands apply.
- **Token discipline:**
  - Never read whole 300–400 KB submissions; use grep and line ranges.
  - Keep state in the campaign `LOG.md` and summarise each check in ≤10 lines.
  - At most one audit agent at a time; no other subagents.
- **Rules gate (unchanged):**
  - CUDA-graph capturable; no data-derived caching.
  - NVFP4/BF16 weights only losslessly repacked; codes and scale bytes only permuted.
  - bf16 activations; FP8 KV.
  - Every key attended; no processes or harness access.
  - Withdraw any run that later proves rule-breaking.

## 7. Verification
- **W0:**
  - `dq_exact` 0/1904 mismatches.
  - The R0 rung quantifies the write-flush penalty.
  - Selector predictions within 5% of judge rows on 2+ cells.
- **Each engine stage:**
  - okc check passes all tests (decode 13/13, moe 5/5).
  - PATH CHOSEN plus kernel-name assertion show the new kernel live.
  - Canary passes forced buckets and 2-state alternation.
  - Per-row improvement on target rows; no untouched row worse beyond drift.
  - Nearby shapes (T=2..7, 9..15, 17..31, 33, 129..255) correct and without cliffs, checked in a prepare self-check sweep.
- **Milestones:** scored submit → `okc leaderboard muse_decode_layer --json` and the overall rank recorded in the campaign `LOG.md`.
- **Final day:** re-run the full rules checklist on each final file; `okc_lint` clean; rules version still 2026-10-22.1.

## 8. Risks and mitigations

| risk | mitigation |
|---|---|
| Engine A misses 3 TB/s | the ladder localises the limiter on day 0–1; class-level fallback to Marlin |
| Context poisoning on a new instruction | proven forms only, canary first, check before any scored run |
| Compile budget (15 min including compile) | ≤4 load_inline modules of about 45 s each; ladder module deadline-gated |
| Decode drift masks gains | judge on kernel rows and paired ratios; pooled confirm rerun only at milestones |
| Token quota | single agent, lean reads, ≤1 audit agent |
| Late-schedule slip | W2/W3 first; W5/W4 are the shock absorbers; freeze 11-04 18:00Z |

## 9. First actions after approval
Execution starts on the user's "execute the plan" (2026-10-26 04:38Z, quota day 10-26 already 4.6 h old). State at start: master d89b52b1 clean, one in-flight churn run (prefill `attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload`), no `campaigns/` dir yet. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

1. **Save it durably first**, before anything that spends quota, on master.
   - Write `campaigns/README.md` and `campaigns/INDEX.md`.
   - Fill `campaigns/2026-10-26-decode-first-rewrite/`:
     - `PLAN.md` and `FINDINGS.md`;
     - `research/`: the agent reports extracted verbatim by script before this session's temp files expire, the board-fit and ROI scripts re-run with outputs saved, and the leaderboard snapshots;
     - `LOG.md`.
   - Commit.
2. **Memory.**
   - Record that `campaigns/` is the destination for campaign-level plans.
   - Record objective = overall board (decode KPI), churn paused, and the single-agent + audit protocol, pointing at the campaign directory.
   - Record the fan-out feedback: ≤3 concurrent agents, your token quota.
   - Mark okc-standing-autonomy as paused and okc-plan-before-minting as resolved.
3. **Pause churn.** One harvest-only `churn.py tick`; DECISIONS/PENDING_USER notes.
4. **Create the branch.** Create the worktree and branch `campaign/decode-first-rewrite`.
5. **Build Phase 0 and W0.** Then dispatch the first decode W1 check and the moe ladder check (Engine A rungs) on today's quota.
