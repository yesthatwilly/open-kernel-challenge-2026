---
id: 2026-10-31-public-plus
status: active (approved by the user 2026-10-31T00:29Z; written ~00:30Z)
objective: find out, on the judge, whether anything this repo learned makes the proven public files (alygg77@6cae49a) faster, under our compliance rules
kpi: per-row and headline ratio of each arm against the same-day control of the same public file (RESULTS.tsv)
branch: campaign/public-plus (code, variant files, attempt files) from master d6a0d421; docs on master
worktree: .worktrees/public-plus
start: 2026-10-31
deadline: 2026-11-04 12:00Z (last check); kernel boards close 2026-11-05 06:59Z
approved: user decisions 2026-10-30 ~23:45Z (base, MIT, check split, agents); plan approved 2026-10-31T00:29Z (ExitPlanMode, after the user asked for a durable cold-runnable MD)
---

# Public-plus: our findings on top of the proven public files (check-only, never scored)

**Save location (step 0):** this file is the campaign's PLAN.md. On approval, copy it verbatim to
`campaigns/2026-10-31-public-plus/PLAN.md` on master before anything else runs.

## 0. Read order (cold start)

1. `date -u`. Then this file, then `campaigns/2026-10-31-public-plus/LOG.md` (the Now block at the top is the only live
   state).
2. `LANES.tsv` (one row per arm: arm | problem | file | build state | checks | verdict | next action | updated UTC).
3. `RESULTS.tsv` (one row per check). `FINDINGS.md` (the inventory and verdicts below, plus results as they land).
4. Steering banners: `git -C ~/git/musespark diff HEAD -- campaigns/INDEX.md campaigns/2026-10-31-public-plus/` and
   `git status`. Act on any new banner and acknowledge it in LOG.md (memory `okc-loop-operating-lessons`).
5. Reconcile before acting:
   - `okc problems` (quota) and `okc jobs` (in-flight jobs);
   - `git worktree list`;
   - union-confirm's LOG Now: `git show campaign/union-confirm:campaigns/2026-10-31-union-confirm/LOG.md | head -40`
     (its main line has check priority);
   - gpunode06: `~/git/takemyjob/prototypes/hopper-probe/bin/hopper-probe status gpunode06`.
6. Harvest anything finished (§6) before dispatching anything new.

## 1. Why this campaign exists

On 2026-10-30 ~23:25Z the user asked three things:
1. Inventory every finding we have against the proven repo.
2. Explain why a full day of testing toward a known answer did not close the gap.
3. After the 00:00Z reset, run an aggressive track, not limited to controllable gains. Its one job is to find out
   whether anything this repo learned improves the proven record. Its submissions stay separate, never score, and
   closely mimic whatever provides the gains we have not matched.

**User decisions (AskUserQuestion, ~23:45Z):**
- Base = the public file plus our grafts; the first check of the day per problem = the public file as a control.
- bf16: include Nathen Fernandes' MIT routes, with the MIT notice kept.
- Checks: union-confirm's main line first; this track takes every remaining check.
- Agents: up to 6 at a time.

**For this track only, these reverse two standing rules:** "never a base" (memory `okc-data-driven-integration`) and
the never-send guard (rank-ladder PLAN:119, arch-mirror PLAN:113-115).

**Still binding:** `okc-stay-compliant`, the attempt gate, and the rules checklist.

**Never, from this track:** an `okc submit`, a public-derived file on master, or a merge of this branch into master.

## 2. State at writing (2026-10-31 00:24Z)

- **Boards:**

| problem | ours | public | gap |
|---|---:|---:|---:|
| attention | 2.005 (scored 4db5f139, master d6a0d421, landed by union-confirm at 00:13Z) | 2.6530 | x1.32 |
| decode | 1.6989 (master file restored to the scored bytes, 3549aee6) | 2.2058 | x1.30 |
| moe | 2.1000 | 2.4015 | x1.14 |
| bf16 | 1.3425 | 1.5502 | x1.15 |
| prefill | 1.5721 | 1.8167 | x1.16 |

- **Quota 10-31:**
  - 20/20 checks on every problem;
  - scored runs 10/10, except attention 9/10.
- **In flight:** nothing from this campaign. union-confirm is idle; its next item is the decode swap-AB root cause on
  the H100, on the user's go.
- **Public clone:** `~/git/okc-muse-spark-kernels` at 6cae49a. File sha256s are in its `results.json`.
- **Public file sizes:**

| problem | bytes | first line |
|---|---:|---|
| attention | 154,807 | `# okc: problem=` |
| decode | 839,695 | `# okc: problem=` |
| moe | 326,062 | `# okc: problem=` |
| bf16 | 382,614 | `# okc: problem=` |
| prefill | 292,650 | `# okc: problem=` |

- **Evidence:** read-only workflow `wf_1949358b-dd0`, 11 agents (5 inventories, 5 adversarial verifiers, 1 "why").
  - Journal: `~/.claude/projects/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/subagents/workflows/wf_1949358b-dd0/journal.jsonl`.
  - Step 0 exports it into the repo, because the session directory is not durable.

## 3. Inventory: our 357 findings against the proven repo

The unit is a distinct mechanism: the newest CLAIMS line per mechanism, from master plus the arch-mirror and
union-confirm branches. The verifiers flagged about 39 classifications as wrong (about 11%), so read every count as
±10%.

| relation to the public file | attn | decode | moe | bf16 | prefill | total |
|---|---:|---:|---:|---:|---:|---:|
| **agrees**: public does the same thing (we re-derived it) | 15 | 16 | 25 | 7 | 11 | **74** |
| **agrees**: both of us found it dead (their NEGATIVE_RESULTS) | 12 | 8 | 9 | 5 | 8 | **42** |
| **refuted by them**: we refuted it, but they use it successfully inside their structure | 18 | 7 | 7 | 6 | 5 | **43** |
| conflicting negatives | 0 | 2 | 1 | 1 | 0 | 4 |
| only we have it | 2 | 2 | 5 | 4 | 8 | 21 |
| different design; outcome not comparable | 28 | 25 | 17 | 25 | 10 | 105 |
| not comparable (FA3 call sites, Triton tweaks) | 26 | 15 | 20 | 4 | 3 | 68 |
| **mechanisms** | 101 | 75 | 84 | 52 | 45 | **357** |

- **Rows where ours beats theirs: 0 of 68.** The best we do is a tie on 3 bf16 rows: unpooler v16 19.09/19.07,
  router d1 3.71/3.69, o v32 9.28/9.26 µs.
- **Validated (116).** Most 10-30 landings re-derived mechanisms their file already has:
  - AT2, AT8, AT5b and S1 are their atomic merge and single launch;
  - BF4 is their rgemv;
  - BF5 is their pingpong.
- **Our refutations that their file refutes (43).** Each was measured alone on our base and works inside theirs:
  - prefill: PDL, the LPT atomic queue, the register ring, the bulk epilogue, N=256 (prefill CLAIMS:47-83);
  - attention: the 4-warp CTA, fixed-reference atomics, the refilled ring (attention CLAIMS:89,118; AM-CLAIMS:131,133);
  - moe: persistent single launch (moe CLAIMS:110,124,135);
  - bf16: lossless 12-bit weights (CLAIMS:78,86; LESSONS:151).

**Graft-onto-public verdicts.** Every candidate was checked by an adversarial verifier reading the public source.

| problem | proposed | survives (speed) | survives (compliance, a cost) | main reasons the rest died |
|---|---:|---|---|---|
| attention | 10 + 3 missed | **G1 W2: 2 workspace copies at v1s, +0.14..+0.58%** (am/s1:research/am-s1-s2.md:97-114) | G0a (offset at max-8), G0b (FP32 unrounded row sum); **M1** (cheaper compliant offset) proposed by the verifier; G0c R11 guard (robustness, 0 µs) | already present (QMAX = their PCHK path, gate preload); their H200 sweep (d1f nc); our evidence doesn't transfer (w256, SEGK) |
| decode | 8 + 2 | none | G7 (rounded-P sum + lazy max); also a **tvm_ffi load_inline** gray area (technical-rules:96) | lean dequant already in (DQ7, PRMTSC); PDLKV vestigial; races already pinned |
| moe | 11 + 2 | none (all below a third of noise or refuted) | none needed (clean) | NOFENCE measured by them at +0.23%; finisher/y-bf16/spins ≤0.1% each; 2 CTA/SM refuted by our own X2 |
| bf16 | 10 + 2 | none | none needed (clean) | their sweeps cover the knobs (ring, Lt, rk); MT=4 rsmall exists and both authors left it off |
| prefill | 12 + 2 | none | C1 (FA3 everywhere) or G0 (FP32 denominator); verifier adds **C1 + our fa3c contiguous gather** and a per-class FA3/G0 race | PF3 can't fit N=256 registers; EXPMODE variants rejected by their pin; G1 moot under C1 |

**Compliance (load-bearing).** Under the user's FA3-precision ruling (AT7b/AT7f, memory `okc-stay-compliant`), the
public attention, decode and prefill files **cannot be sent as published**:
- **Rounded-P row sum:** the softmax denominator is the sum of E4M3-rounded P, built from ones rows in the PV wgmma.
  Sites: attention PUB:1838-1843, 2116-2119; decode :704-705, 2238-2240; prefill :3625-3628, 3664, 3919.
- **Lazy max offset** (P_max in [1,256], against FA3's 256) in attention and decode. Sites: attention PUB:1700-1711,
  1921-1988; decode :2328-2359.

moe and bf16 are clean.

So "the public file as control" holds literally for moe and bf16. For the other three the control is a compliance-patched
**public-C**. Our own findings set its cost:
- AT7c's FP32 denominator (`research/at7c-1030.md`);
- the M1 offset (`at7-1030.md:46`, `at7fg-1030.md:21-22`);
- our fa3c contiguous gather (prefill);
- R11 (`REPLICATION.md:12`).

This is the one place our knowledge reliably applies to their structure.

## 4. Why the right answer is still far away

**The proven answer is a whole file, not a list of moves.**
- **The speed is vjeux's architecture.** Their README:48-52 says most of the speed comes from that base. Attention and
  prefill are vjeux's files unchanged.
- **alygg77's own increments are small:** KPERM3 +1.25% on moe, about +2.4% on bf16, nothing above noise on attention or
  prefill. Their FACTS and NEGATIVE docs explain those small deltas, not why vjeux is fast, and our facts already agree
  with most of them.

**How much of each gap is structure rather than single knobs:** attention about 100%, prefill about 100%, decode about
100% (a composite), moe about 91%, bf16 about 84%.

**What their structure looks like next to ours:**
- Attention: one templated kernel on all 12 rows, against our 23 kernels with 2-3 launches on 9 rows.
- moe: one persistent kernel with one grid barrier and no cooperative launch.
- bf16: a 12-bit codec inside a consumer built to absorb its decode cost.

**Moving parts of their answer onto our code mostly failed. Whole-component replacements worked.**
- **Values:** 2 of 2 failed (attention CLAIMS:112, bf16 CLAIMS:88).
- **Single decisions failed:** T1 +1.9 µs, AT1 +18.6 µs, S3 +5.4 µs, MO3-S2 +80 µs.
- **Clean-room bundles lost the details that set the cost.** The empty L3 moe skeleton cost 17-19 µs, against their
  whole 24.6 µs d1 row. F1 built an exp-phase alternation that their file never executes.
- **What worked was replacing whole components:** 16 landings for x1.25 on the day. S1 alone gave +4.2%.

**Other drags:**
- **Transfer.** At v32f the full-attention gap is +7 µs on the H100 and +59 µs on the judge.
- **Noise.** The riders are 0.2-1% each, under a 1.2-2% noise floor.
- **The arithmetic.** About 85 more landings at ~1.4% each would be needed.
- **Lost time and quota.** About 4.9 h were idle under stops, and 61 of 100 checks went unused.
- **Precision.** Part of their speed is precision we refuse. public-C measures how much of 2.653, 2.206 and 1.817 is
  reachable under our rules.

**Expected yield of this track, stated up front:**
- **Speed:** at most about +0.6% on attention and about +0.3..0.5% on a moe stack; no speed graft on the other
  problems.
- **The main yield is measurement:**
  - compliant public-C targets;
  - the first judge-side per-kernel profiles of the public moe and bf16 files;
  - whether our compliance knowledge lowers the tax on their structure.
- **A null across the board is the likely outcome.** It would confirm that the gap is architecture.

## 5. Layout and guards

| what | where | branch |
|---|---|---|
| PLAN.md, FINDINGS.md, LOG.md, LANES.tsv, RESULTS.tsv, `runs/<job>.*`, `research/public-exposed-wf_1949358b-dd0/` | `campaigns/2026-10-31-public-plus/` | master (main tree) |
| variant files `<arm>.py` | `campaigns/2026-10-31-public-plus/files/<problem>/` | `campaign/public-plus` only |
| attempt files `<ts>-check-pp-<arm>.md` | `<problem>/attempts/` in `.worktrees/public-plus` (the gate looks in the checkout that owns the file, `.claude/hooks/okc-attempt-gate.py`) | `campaign/public-plus` |

- **Commit discipline.** Master commits touch only `campaigns/2026-10-31-public-plus/`, `campaigns/INDEX.md` and
  memory. Commit by path, so another session's staged files are never swept in: `git commit -m "..." -- <paths>`.
  union-confirm commits to master from the same tree.
- **Header on every variant:**
  - keep line 1 (`# okc: problem=...`) and every public header line;
  - insert one line 2: `# Check-only variant <arm> of alygg77@6cae49a kernels/<file> (Apache-2.0), modified by yesthatwilly <UTC date>: <one-line change>. pp-nonce <arm>-<n>`.
  - This covers Apache-2.0 §4(b) and defeats okc dedup.
  - bf16 keeps its MIT block (PUB lines 5-28).
- **Contamination.** Every agent or session that reads or edits a public file is public-exposed. It never writes
  master kernels or arch-mirror clean-room stages. arch-mirror writers must count reads of `files/` and
  `research/public-exposed-*` as public reads (arch-mirror PLAN.md:65-73).
- **Never scored.** No `okc submit` of anything under `files/`. The end-of-day check: `okc jobs` shows only `check`
  rows for pp jobs.

## 6. Operating rules

**Before every check:**
1. Copy `ATTEMPT_TEMPLATE.md` to `.worktrees/public-plus/<problem>/attempts/$(date -u +%Y-%m-%dT%H%M%SZ)-check-pp-<arm>.md`.
2. Fill in:
   - Goal: the arm and its target rows;
   - What changed: line ranges in the public file;
   - Rules check: every box;
   - Risks;
   - the audit verdict;
   - for arms that touch something we refuted, a "Reopen: public-base track, different structure" line, which
     silences the gate's claims warning.

**Dispatch:**
- `cd ~/git/musespark/.worktrees/public-plus && okc check <problem> campaigns/2026-10-31-public-plus/files/<p>/<arm>.py --json`
- Fill the job id into the attempt file immediately, then set the LANES row to `check`.

**Harvest** (`okc status <job> --json` once done), with `C=~/git/musespark/campaigns/2026-10-31-public-plus`:
- `okc status $JOB --json > $C/runs/$JOB.okc.json`
- `python3 -I -c "import json,sys,shutil,os;sj=json.load(open(sys.argv[1]));[shutil.copy(p,os.path.join(sys.argv[2],sys.argv[3]+'.'+os.path.basename(p))) for p in list((sj.get('profile') or {}).values())+[(sj.get('output') or {}).get('log')] if p and os.path.exists(p)]" $C/runs/$JOB.okc.json $C/runs $JOB`
- RESULTS.tsv row: `utc | job | problem | arm | headline [ci] | per-row us (shape keys) | ratio vs control mean per row | verdict`.
- Fill the attempt file's Result. Update LOG Now and LANES. Commit by path.

**Reading and decision rules** (aggressive: no H100 gate, no value bar, bundles allowed):

| situation | rule |
|---|---|
| control (A/A, 2 nonce'd reads) | Control mean = geomean of the 2 headlines. Per-row spread s_r = abs(r1-r2)/mean. Headline spread s_h likewise. If s_h > 2%, take a 3rd control read before judging any arm |
| arm, first read | Headline at or above control mean -1%, or any target row better by more than max(s_r, 1.5%): take a 2nd read. Otherwise **kill** (one read only) |
| arm **win** | Over 2 reads: headline geomean ≥ control mean × (1 + max(s_h, 0.5%)), **and** every target row better than control by more than s_r in both reads |
| arm **row-only hold** | Target rows move as above, but the headline is within noise. Record it; it is a stacking candidate |
| compliance bases (A0/A0'/P0/P1/P2/D0) | Always 2 reads. They are targets, not wins or losses. The cheaper compliant base becomes the control for that problem's later arms |
| correctness failure | Record it, fix once, re-check once. A second failure kills the arm |
| two or more wins on one problem | One stacked arm of the winners, 2 reads, on the next quota day |
| a win at all | Report it to the user. Porting it to our scored line is a separate rung and needs the user's decision |

**Check budget:**
- Before each wave, read union-confirm's Now. Reserve whatever its next items need; the default reserve is 6 checks
  for attention and decode and 2 for the others.
- This track uses the rest, up to about 25 checks in total on 10-31.
- 18:00-24:00Z: spend this track's leftover checks on second reads, then third control reads.

**H100 (gpunode06), for correctness and triage only:**
- GPUs 1/3/4/5/6; never 0/2/7.
- Check `nvidia-smi --query-compute-apps` first, because union-confirm may be using it.
- Run through `hopper-probe` with its guard, or the queue (memory `okc-hopper-probing`).
- Node-local `/tmp/USER-hopper` only.

## 7. Arms (10-31)

| # | problem | arm | change | build | checks |
|---|---|---|---|---|---|
| 1 | bf16 | **B0-aa1, B0-aa2** | the raw public file plus the notice/nonce line (gives the first judge per-kernel profile of their bf16) | main thread | 2 |
| 2 | moe | **M0-aa1, M0-aa2** | raw public plus the notice/nonce line | main thread | 2 |
| 3 | moe | **M1-stack** | NOFENCE (drop the fence at PUB:2151); finisher skips its own red.add (:2263-2272); y stored BF16 with (1+w) kept in FP32 (:2097-2110, :2649-2709); `__nanosleep` 64/32 → 16/8 (:2634, :2011, :2014, :2137); `OKC_WGSDFIN` 2 → 3 (:1855) | builder + audit (nodejudge inproc correctness) | 2 |
| 4 | attention | **A0** public-C full | G0a: offset at max-8, FA3 range (:1700-1711, :1916-1988, :2007-2053); G0b: FP32 unrounded row sum, m64n72 → m64n64 (:1838-1843, :2002-2003, :2072, :2116-2119); R11 guard (:2172, :2204) | builder + audit (H100 correctness incl. M2 adversarial coherent inputs) | 2 |
| 5 | attention | **A0'** public-C cheap | M1 (offset starts at the first-tile max minus 8; repack only at the satfinite byte 0x7E) + G0b + R11 | builder | 2 |
| 6 | attention | **A1** | best of A0/A0' + `_TUNED[(1,8,True,1)] = dict(nc=2, coop=1)` (PUB:2940-2943); then an nc=4 arm | knob; H100 v1s triage (sliding rows transfer 1:1) | 2 + 1 |
| 7 | attention | **A2** (speculative) | + 8-warp d1s ONE+SPEC build: warp = one KV head x 16 keys, 4-way key merge (PUB:2126-2154, 2808-2811) | builder; H100 d1s triage | 1-2 |
| 8 | prefill | **P0** C1 | all attention through the public's FA3 path (`_attention_norm_gate` :648-654, `_choose_attention` :811-833); add `assert T > 8` | builder (1-3 lines) | 2 |
| 9 | prefill | **P1** | C1 + our contiguous fa3c gather on the full classes (port our Triton `_kv_gather_contig` from `muse_prefill_layer/submission.py`, or wire their unused CUDA `kv_gather_kernel` :3601/:1387) | builder | 2 |
| 10 | prefill | **P2** | G0: FP32 unrounded denominator in their attention (:3625-3628, :3643, :3664, :3825, :3919, :4919, plus attn_prep2 :3738/:3770 and the combine :1585-1593); `ATTN_FIXED=None` so `_tune_attn` races FA3 against G0 per class; pin the winners | builder | 2 |
| 11 | decode | **D0** public-C | the A0'/A0 fixes ported into ATT7 (:704-705, :1970, :2238-2240, :2328-2359) + the 4 `tvm_ffi.cpp.load_inline` modules (:11769, :11778, :12633, :13842) moved to `torch.utils.cpp_extension.load_inline`. **Kill decode** if the port is not mechanical (more than ~200 changed lines, or numerics change); no speed graft survives there anyway | builder (sizing first) | 0-2 |
| 12 | attention | **M2** diagnostic (no check) | run the unmodified public attention through `reference.check_implementation` on coherent full-verify inputs that put each segment's max in its first tile (LESSONS:145). Tells us whether their lazy offset holds on held-out data | H100, correctness only | 0 |

## 8. First hour (exact)

```bash
date -u
cd ~/git/musespark
# step 0: durable copies on master (docs only)
C=campaigns/2026-10-31-public-plus; mkdir -p $C/runs $C/research/public-exposed-wf_1949358b-dd0
cp ~/.claude/plans/inventory-the-hypothesis-findings-you-starry-hamster.md $C/PLAN.md
J=~/.claude/projects/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/subagents/workflows/wf_1949358b-dd0/journal.jsonl
cp $J $C/research/public-exposed-wf_1949358b-dd0/journal.jsonl
cp ~/.claude/projects/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/workflows/scripts/okc-findings-vs-public-wf_1949358b-dd0.js $C/research/public-exposed-wf_1949358b-dd0/
# render FINDINGS.md from the journal (sections 3-4 of this PLAN plus per-problem inventory/verdict tables),
# LOG.md (Now block), LANES.tsv (header + arms 1-12), RESULTS.tsv (header); INDEX.md row (status active)
git commit -m "public-plus: PLAN, FINDINGS, inventory export (wf_1949358b-dd0)" -- $C campaigns/INDEX.md
# branch + worktree for code
git worktree add .worktrees/public-plus -b campaign/public-plus master
PUB=~/git/okc-muse-spark-kernels; git -C $PUB rev-parse --short HEAD   # expect 6cae49a
sha256sum $PUB/kernels/*.py   # compare with $PUB/results.json kernel_sha256
WT=~/git/musespark/.worktrees/public-plus; F=$WT/$C/files; mkdir -p $F/{moe,bf16,attention,prefill,decode}
mk() { python3 -I -c "import sys;s,d,n=sys.argv[1:4];a,b=open(s).read().split('\n',1);open(d,'w').write(a+'\n'+n+'\n'+b)" "$@"; }
for n in 1 2; do
  mk $PUB/kernels/muse_bf16_proj.py  $F/bf16/B0-aa$n.py "# Check-only copy of alygg77@6cae49a kernels/muse_bf16_proj.py (Apache-2.0; MIT parts keep their notice), unmodified except this line. yesthatwilly $(date -u +%F) pp-nonce B0-aa$n"
  mk $PUB/kernels/muse_moe_w4a16.py $F/moe/M0-aa$n.py  "# Check-only copy of alygg77@6cae49a kernels/muse_moe_w4a16.py (Apache-2.0), unmodified except this line. yesthatwilly $(date -u +%F) pp-nonce M0-aa$n"
done
head -3 $F/moe/M0-aa1.py; wc -c $F/*/*.py
# attempt file per check (gate: one fresh unconsumed file per run, in $WT/<problem>/attempts/), then:
cd $WT && okc check muse_bf16_proj $C/files/bf16/B0-aa1.py --json   # fill job id; repeat for B0-aa2, M0-aa1, M0-aa2
```

Then launch the build Workflow (§9) and record its run id in LOG Now.

## 9. Build workflow `public-plus-build` (ultracode, ≤ 6 agents)

- A `pipeline()` over [attention (arms 4-7, 12), prefill (8-10), moe (3), decode (11)], with two stages per item.
- **Stage 1, build** (effort high, `isolation: 'worktree'`, based on `campaign/public-plus`). The brief is that item's
  rows in §7 plus §5. The builder:
  - writes each arm file under `files/<p>/`;
  - keeps every public header line plus the notice line;
  - does no `okc` runs;
  - may use gpunode06 GPUs 1/3/4/5/6 for correctness and triage per §6;
  - returns arm paths, a diff summary with public line ranges, bytes, measured prepare/compile time, and H100 results.
- **Stage 2, audit** (effort high). It checks:
  - the rules checklist against the diff;
  - compliance of the patched softmax against `at7-1030.md`, `at7c-1030.md` and `at7fg-1030.md` (FA3 P range, FP32
    unrounded denominator, no approximate max that can clip);
  - correctness via `reference.check_implementation` on the H100 (drivers: dattn for attention; nodejudge inproc for
    moe; `pfcmp.py` for prefill; public `timing_kit/paired.py` in-process for decode, which avoids the nodejudge shims);
  - headers, bytes under 1,000,000, prepare under 300 s, compile within 15 min.
  - It returns PASS/FAIL per arm and a filled attempt-file draft. FAIL goes back to its builder once.
- The main thread dispatches checks as audits pass, in §7 order.

## 10. Open decisions for the user

None block the start. These are defaults, which the user can override:
- Raw-file controls for attention, decode and prefill are **not sent**, under the standing compliance rule. Their
  public-C reads stand in for the controls.
- Porting any track win to our scored line is the user's call, per win.
- M2 (a correctness-only H100 diagnostic of their lazy offset) runs by default.

## 11. Verification

- **Before each check:**
  - the attempt file and rules checklist are filled;
  - the audit is PASS;
  - the file path is under `files/`;
  - the notice and nonce line are present;
  - bytes are under 1,000,000.
- **After each check:** a RESULTS.tsv row, the archived `runs/<job>.*`, the attempt Result filled in, LOG Now and
  LANES updated, and a commit by path.
- **End of each day:**
  - every arm is read twice or explicitly left at one read;
  - FINDINGS.md states the public-C numbers and any wins;
  - `okc jobs` shows no scored run from this track;
  - `git log master -- muse_*/submission.py` shows nothing from this campaign;
  - write the next day's arms into LANES.tsv.
- **Memory (step 0):**
  - new `okc-campaign-public-plus.md` and its MEMORY.md index line;
  - an exception note appended to `okc-data-driven-integration.md` and `okc-campaign-arch-mirror.md` (this track
    only: public base, check-only, contamination rule).
