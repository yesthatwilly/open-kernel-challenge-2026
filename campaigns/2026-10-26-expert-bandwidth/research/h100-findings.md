# H100 node ledger (gpunode06): X5 / x5-h100-program

This is the H100 node ledger of the X5 campaign and its x5-h100-program. Every record carries the MANIFEST sha256 of
the bytes it ran, the GPU, cold or warm caches, and the SM clock under load. A run below 1800 MHz under load, or one
that throttled, is void.

Conventions for the records below:
- Node: gpunode06, 8x H100 80GB HBM3 SXM, driver 615.71.09. Root drained it on 2026-10-26 15:36 for "hardware
  troubleshooting". Runs go through takemyjob `prototypes/hopper-probe` (`bin/hopper-queue` and the on-node guard), as
  USER, outside Slurm.
- Run dirs live under `~/git/takemyjob/prototypes/hopper-probe/runs/` and are named in local time (MST = UTC-7).
  `queue/HISTORY.tsv` uses UTC.
- Caches: `warm` means the env's build cache (`fresh_build` false), so prepare time and the deadline-driven prepare
  phases are not judge-like. `cold` means empty per-run build dirs. Every nodejudge timed replay runs after a 256 MiB L2
  read flush, so L2 is cold for every replay.
- SM clock: nodejudge takes an nvidia-smi snapshot before and after a run (1830 MHz both times on every record below).
  It does not sample during the timed loop, so the 1800 MHz rule is **unverified** for nodejudge timings: they are not
  void and not proven valid either. Each esl KV line records the clock it measured under load (`mhz`); lines below
  1800 MHz are marked void.
- Harness: the 10-27 nodejudge runs used the committed harness, takemyjob HEAD 0c0bc56. The tracebacks in the 24-* logs
  match its 1009-line `nodejudge.py` line for line.
- **Bytes: every nodejudge run below ran moe `board.py` sha256 ab46fbb722c2 (641,473 B).** That is the board from before
  a4d1c81d (the W6-M fork, scored 1.7187). It is **not** the current board, master `muse_moe_w4a16/submission.py`
  498afbd8e73d (906,398 B, 1.7225, landed 10-27 21:57Z).

## Node state, 2026-10-28 00:48Z (read-only check before the redeploy)

- Slurm `IDLE+DRAIN`, 0 jobs, Reason `hardware troubleshooting [root@2026-10-26T15:36:35]`. No compute process on any
  GPU. nv-hostengine and dcgm-exporter run as services; no DCGM diag (`nvvs`) was running.
- Idle temperatures: GPU0 86 C (229 W at 0% utilisation, 1815 MHz, **SW thermal slowdown active**, hotter than the
  78-79 C seen on 10-27), GPU2 65 C, GPUs 1/3/4/5/6/7 29-37 C at about 120 W. The runner's 55 C idle limit already
  excludes GPU0 and GPU2. GPU 7 is reserved (`queue/reserved`).
- `/tmp/USER-hopper` is absent: `clean gpunode06 nodejudge` removed the env on 10-27 at 21:25Z.
- The redeploy was **held at about 01:00Z**. `setup` would ship the uncommitted nodejudge edits (E1/E2, mid-build) and
  run them. The runner would ship `build/nodejudge.tgz`, which was rebuilt at 21:23:42Z with an earlier partial E1
  `nodejudge.py` (sha256 c2caa74f50ce, neither HEAD nor the working tree) and still packs board ab46fbb722c2. No
  hopper-probe GPU run has happened since 21:23Z (`runs/INDEX.tsv`).

## Harvest: queue jobs that finished 2026-10-27 19:38Z-21:23Z

The 23 jobs in `queue/done/`, plus the 8 fuzz-filler runs (2 of them finished but were never reaped). Fault counts
cover test failures, plus miscompares on timed replays (each timed replay is checked against the fp32 reference),
plus sanitizer errors. In every nodejudge run, bytes = board ab46fbb722c2 and GPU = H100 SXM. Clock "snap" = 1830 MHz
before and after, with nothing sampled under load.

| job | run dir | GPU | caches | clock | verdict | faults | key numbers |
|---|---|---|---|---|---|---|---|
| 30-h100-nj-moe-board-flips-1 | 20261027-123454-gpunode06 | 1 | cold | snap | PASS | 0: tests 5/5, 0/600 replays | headline x1.3929 [1.3720, 1.4138] vs SGLang; prepare 164.4 s; the one gate flip (Engine A v1 at T=1..8) |
| 30-h100-nj-moe-board-flips-2 | 20261027-123457-gpunode06 | 3 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3893 [1.3685, 1.4101]; prepare 163.0 s |
| 30-h100-nj-moe-board-flips-3 | 20261027-123459-gpunode06 | 4 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3908 [1.3699, 1.4117]; prepare 163.0 s |
| 30-h100-nj-moe-board-flips-4 | 20261027-123502-gpunode06 | 5 | cold | snap | PASS | 0: 5/5, 0/600 | x1.3921 [1.3712, 1.4130]; prepare 161.5 s |
| 25-h100-nj-moe-sweep | 20261027-125905-gpunode06 | 1 | warm | snap | PASS | 0: tests 37/37, 0/120 | 32 held-out shapes (27 bs in 2..255, 5 skewed) plus the judge 5, all within tolerance; x1.3835 |
| 26-h100-nj-moe-soak-1 | 20261027-125907-gpunode06 | 3 | warm | snap | PASS | 0: 5/5, 0/6000 | 1000 checked reps per shape x 6 shapes; x1.3900 [1.3691, 1.4108] |
| 26-h100-nj-moe-soak-2 | 20261027-125910-gpunode06 | 4 | warm | snap | PASS | 0: 5/5, 0/6000 | x1.3862 [1.3654, 1.4070] |
| 22-h100-nj-moe-bigt-1 | 20261027-141313-gpunode06 | 1 | warm | snap | PASS | 0: 5/5, 0/2000 | T=128 / T=256 raw median 297.9 / 349.8 us; 8.61 / 8.59 us per distinct local expert (R2 0.84 / 0.93) |
| 22-h100-nj-moe-bigt-2 | 20261027-141316-gpunode06 | 3 | warm | snap | PASS | 0: 5/5, 0/2000 | 299.5 / 348.4 us; 8.67 / 8.51 us per expert (R2 0.84 / 0.92) |
| 22-h100-nj-moe-k9-1 | 20261027-141318-gpunode06 | 4 | warm | snap | PASS | 0: 5/5, 0/6000 | per-expert OLS slope T=1/8/16: 11.34 / 9.89 / 9.56 us (K9 section) |
| 22-h100-nj-moe-k9-2 | 20261027-141321-gpunode06 | 5 | warm | snap | PASS | 0: 5/5, 0/6000 | 11.63 / 10.12 / 9.63 us |
| 24-h100-nj-moe-san-ea4 | 20261027-141326-gpunode06 | 6 | warm | before 1830 | **VOID** (OOM fallbacks) | memcheck 0 errors; harness OOM: tests 3/5, no timing | kernel filter `kns=ea4_`; prepare fell back under memcheck (79.2 GiB of 80) |
| 24-h100-nj-moe-san-ea_combine | 20261027-141530-gpunode06 | 1 | warm | - | **VOID** (OOM) | memcheck 0 errors; prepare aborted (OOM), 0 tests | `kns=ea_combine` |
| 24-h100-nj-moe-san-ea_w | 20261027-141533-gpunode06 | 3 | warm | - | **VOID** (OOM) | memcheck 0 errors; prepare aborted (OOM), 0 tests | `kns=ea_w` |
| 24-h100-nj-moe-san-wg | 20261027-141535-gpunode06 | 4 | warm | before 1830 | **VOID** (OOM fallbacks) | memcheck 0 errors; tests 2/5 (3 OOM), no timing | `kns=wg_` |
| 20-h100-esl-cold-rep1 | 20261027-133349-gpunode06 | 1,3 | n/a | 1810-1830 | PASS | 0: KV 200/200 ok, 4 sanitizer runs 0 errors | P1: es streamed N=16 23.2-24.6 el/clk/SM; e devres ceilings N=16 27.2-30.4 |
| 20-h100-esl-cold-rep2 | 20261027-133352-gpunode06 | 4,5 | n/a | 1802-1830 | PASS | 0: 200/200, 0 | matches rep1: clk_k16 median diff 0.04%, p90 0.86%, max 5.9% |
| 20-h100-esl-k4-rep1 | 20261027-133552-gpunode06 | 1,3 | n/a | 1679-1830; **30/192 lines void** | PASS | 0: 192/192, 6 sanitizer runs 0 errors | cta r2 N=64: S2 14.2-14.4, D2 14.5-14.6, S4 18.4 (target 25.6); N=128 S2 12.5-12.6 (target 12.8); N=256 7.9 |
| 20-h100-esl-k4-rep2 | 20261027-133555-gpunode06 | 4,5 | n/a | 1665-1830; **24/192 void** | PASS | 0: 192/192, 0 | matches rep1: median 0.03%, p90 0.38%, max 5.6% |
| 21-h100-esl-k1-rep1 | 20261027-133757-gpunode06 | 1,3 | n/a | 1803-1830 | PASS | 0: 240/240, 10 sanitizer runs 0 errors | ring-free K1 (cta, tc130) cad8: N=8/16 78.7, N=32 64.0, N=64 32.0, N=128 16.0, N=256 8.0 el/clk/SM |
| 21-h100-esl-k2-rep1 | 20261027-133800-gpunode06 | 4,5 | n/a | 1830 | PASS | 0: 280/280; K2CHECK 8 x 65,536 cases, 0 mismatches; 4 sanitizer runs 0 errors | dq2 best 34.6-35.5 el/clk/SM (imad / sts sinks, 3-4 WGs) vs the ~39.4 ALU ceiling; ar fold4 about 30 |
| 21-h100-esl-k6-rep1 | 20261027-133959-gpunode06 | 1,3 | n/a | 1830 (timed lines) | PASS, **K6 NO-GO** | 0: 64/64, 2 sanitizer runs 0 errors | K6GO 0/16: mixed/(hot+cold) 0.83-0.93, but hot slowdown 1.66-1.87 and cold 1.14-1.52 against the x1.25 limit |
| 21-h100-esl-k7-rep1 | 20261027-134002-gpunode06 | 4,5 | n/a | 1830 | PASS | 0: 96/96, 2 sanitizer runs 0 errors | ingest N=128 hbm-own 90-101 B/clk/SM, i.e. 39.5-45.0 el/clk/SM at BM=128; l2-shared 105.7-107.4 B/clk/SM; N=256 hbm-own 83-89, i.e. 19.4-20.9 el/clk/SM |
| 90-h100-nj-moe-fuzz-odd x4 | 20261027-141634 / -141819 / -142000 / -142210 (the last unreaped) | 5 / 1 / 1 / 1 | warm | snap | PASS | 0: 53/53 tests in each run (212 cases), 0/40 replays | 48 odd/prime bs plus the judge 5, a fresh seed each run |
| 90-h100-nj-moe-fuzz-skew x4 | 20261027-141637 / -141822 / -142002 / -142213 (the last unreaped) | 6 / 3 / 3 / 3 | warm | snap | PASS | 0: 27/27 in each run (108 cases), 0/40 | 22 skewed-routing bs plus the judge 5 |

The esl bundle stamp is 20261027T202132Z-src9081729ad153, with binary hashes esl_cold 130/133 b480f8b7f55250cf /
e1c34525c7dc3aba, esl_hot (k4/k6/k7) 2c4defd247220ed3 / 3d5c466d4971232a, esl_k1 9fc5d14e04b08777 / 76cbf6f3b1a3c45f,
and esl_k2 146a70f383a3eb18 / 2517440ff1035e84. The 20:23-20:29Z validation runs used the same stamp, and the
replicates' SASS `text=` hashes match them line for line.

### S3 / K9: replay time against distinct local experts at T=1/8/16 (22-h100-nj-moe-k9-1/2)

2 x 2000 reps per shape, raw replay medians in us. The empty-graph fixed cost (3.6-3.9 us) is not subtracted and does
not affect the slopes. Each cell gives the median and n, pooled over both runs, for cells with n >= 10.

| distinct local experts | T=1 (decode 1) | T=8 (verify 1) | T=16 (decode 16) |
|---|---|---|---|
| 1 | 28.1 (16) | | |
| 2 | 37.2 (2097) | | |
| 3 | 50.5 (1566) | | |
| 4 | 57.2 (300) | 58.8 (55) | |
| 5 | 72.4 (19) | 75.0 (154) | |
| 6 | | 83.2 (336) | 86.6 (32) |
| 7 | | 92.6 (630) | 95.7 (92) |
| 8 | | 101.4 (825) | 103.8 (186) |
| 9 | | 114.0 (851) | 117.1 (312) |
| 10 | | 123.6 (636) | 126.6 (476) |
| 11 | | 131.3 (320) | 133.3 (605) |
| 12 | | 142.1 (124) | 143.7 (648) |
| 13 | | 151.8 (42) | 153.1 (585) |
| 14 | | | 163.3 (472) |
| 15 | | | 172.8 (304) |
| 16 | | | 182.4 (155) |
| 17 | | | 191.9 (87) |
| 18 | | | 202.8 (30) |

Per-expert slope, in us per additional distinct local expert. Each run's OLS fit has a 95% bootstrap interval over reps:

| T | k9-1 (GPU 4) | k9-2 (GPU 5) | with local routes as a 2nd regressor | adjacent-step medians (n >= 30) |
|---|---|---|---|---|
| 1 | 11.34 [11.20, 11.52] | 11.63 [11.47, 11.79] | routes = distinct at T=1 | 2->3 +13.3, 3->4 +6.8 (both runs); pooled 1->2 +9.1, 4->5 +15.2 |
| 8 | 9.89 [9.84, 9.93] | 10.12 [10.07, 10.17] | 9.89 / 10.07 per expert, -0.01 / +0.15 per route | median 9.6, range 6.8-15.4; 4->5 +15.4 (pooled +16.2); 8->9 +11.2 / +14.1 |
| 16 | 9.56 [9.51, 9.61] | 9.63 [9.58, 9.68] | 9.51 / 9.55 per expert, +0.37 / +0.39 per route | median 9.5, range 6.4-13.4; 8->9 +13.4 / +13.2; 10->11 +6.4 / +6.9 |

Reading:
- The spread between replicates (0.29 / 0.23 / 0.07 us at T=1/8/16) is wider than each run's bootstrap interval, so
  quote the replicate range: **T=8 9.9-10.1, T=16 9.56-9.63 us per expert.** At T=1 a single slope does not describe
  the data: the steps alternate 13.3 / 6.8 us. The two-step mean from d=2 to d=4 is 10.0 us per expert.
- The staircase is not uniform. The largest steps are into d=3 and d=5 at T=1 (+13.3, +15.2), into d=5 at T=8
  (+15.4 / +16.2) and into d=9 at T=8 and T=16 (+11.2 to +14.1). The smallest are into d=4 at T=1 (+6.8) and into d=11
  at T=8 and T=16 (+6.4 to +8.1). That looks like a grouping or boundary cost in the board's expert loop, and it is
  worth a look in S3's design.
- Time follows distinct experts, not routes: at most 0.4 us per local route at T=16, about 0 at T=8.
- Fit of t = a + b*k (k = distinct local experts, natural routing), with a_net = the intercept minus the 3.6-3.9 us
  replay fixed cost:

  | T | b, OLS (us per expert) | b, fit to group medians (n >= 30) | a_net, OLS | a_net, group medians |
  |---|---|---|---|---|
  | 1 | 11.34 / 11.63 | 10.00 / 10.10 (k=2..4) | 11.2 / 10.8 | 14.3 / 14.3 |
  | 8 | 9.89 / 10.12 | 10.09 / 9.74 | 19.9 / 18.0 | 17.7 / 21.4 |
  | 16 | 9.56 / 9.63 | 9.56 / 9.62 | 26.0 / 25.5 | 25.4 / 24.9 |

- Against the program's M08 rule (`research/h100-program-2026-10-27.md` §3):
  - b is above the 9.28 us "streaming engine" bar at every T (1.03-1.25x). The rule also needs that time concentrated
    in one kernel, and these runs do not profile.
  - a(T=1) straddles the 12 us "fixed-cost work" bar (10.8-14.3 us). The estimate depends on the estimator because of
    the sawtooth and the extrapolation to k=0.
  - Both questions stay unresolved here. M08's forced k in {0..8} (E1b) settles them.
- Bandwidth, as an assumption: the reference NVFP4 layout is 28.3 MB per expert (w13 16 MiB + 2 MiB scales, w2 8 MiB +
  1 MiB scales). At that size, b = 9.6-10.1 us is 2.8-2.95 TB/s, against the node's 3.11 TB/s read rate (HANDOFF §6).
  The board's packed bytes may differ; the program's 9.28 us bar implies a smaller per-expert byte count.
- Transfer: R8 (program §1) scales b by x0.69-0.97 on the judge, i.e. about 6.6-9.8 us per expert there. The judge's
  own T=1 mode gap was 10.3 us (record 2a7b5d20), against 13.3 us for d=2->3 here (x0.77). Settle any absolute number
  with a judge check.

### S1a draws: T=128/256 against routing features (22-h100-nj-moe-bigt-1/2)

| T | run | raw median (p10 / p90) | corr with distinct / routes / busiest expert | OLS fit (us) | R2 |
|---|---|---|---|---|---|
| 128 | bigt-1 (GPU 1) | 297.9 (264.1 / 330.3) | +0.75 / +0.41 / +0.01 | 20.2 + 0.399 routes + 8.61 distinct + 0.105 max | 0.84 |
| 128 | bigt-2 (GPU 3) | 299.5 (264.3 / 335.2) | +0.76 / +0.40 / +0.01 | 23.0 + 0.391 routes + 8.67 distinct + 0.069 max | 0.84 |
| 256 | bigt-1 | 349.8 (317.3 / 378.9) | +0.87 / +0.25 / -0.17 | 72.4 + 0.174 routes + 8.59 distinct - 0.023 max | 0.93 |
| 256 | bigt-2 | 348.4 (317.9 / 377.9) | +0.87 / +0.24 / -0.14 | 77.4 + 0.159 routes + 8.51 distinct - 0.015 max | 0.92 |

- **The slow draws are the wide ones, not the hot ones.** The slowest 10% of draws average 25.7 distinct local experts
  at T=128 (fastest 10%: 18.1-18.4) and 28.0 at T=256 (19.9-20.2). Their busiest expert is the same or less loaded
  (T=128: 67-71 rows in both deciles; T=256: 113-122 rows in the slow decile vs 149-169 in the fast one).
- So the worst-draw class S1a must beat is many distinct experts: T=128 with d >= 26 runs 318.9-332.6 us, and T=256 with
  d >= 28 runs 370.0-382.2 us.
- 8.5-8.7 us per distinct expert at T >= 128 corresponds to more than the node's read rate (3.3 TB/s on 28.3 MB). At
  these sizes the per-expert term is partly overlapped with other work, or the fit splits it with the route term.

### Memcheck per kernel family (24-h100-nj-moe-san-{ea4, ea_combine, ea_w, wg})

- compute-sanitizer memcheck reported **0 errors** in all four runs (`--kernel-name kns=<family>`). All four are still
  **VOID, not a pass**: the program counts an out-of-memory fallback as voiding the run (program §2, E2).
- Under memcheck, the board's prepare() filled the 80 GB H100 (79.2 GiB in use), even with the kernel filter. As a
  result:
  - The ladder, the combo phases (F1, F2, C1a, C1b), Engine A ("v3 not built") and the wgmma bucket selector raised OOM
    and fell back to master or Marlin.
  - ea4 then failed 2 of its 5 tests on OOM, and wg failed 3 of 5. Neither got a timed replay.
  - ea_combine and ea_w aborted in the harness's weight checksum after prepare, so they ran no tests.
- The filtered families therefore never ran on their judged paths. This repeats the 10-27 18:3xZ finding in the
  nodejudge README.
- Ways to make memcheck conclusive: a lighter prepare under the sanitizer (diagnostics off, fewer state copies), or
  running on the 141 GB H200 when it is available.

### Board gate flips across 4 cold prepares (30-h100-nj-moe-board-flips-1..4)

- Each run logged 25 `PATH CHOSEN` lines, and 21 are identical across all four runs. Two F2 B256 lines differ only in
  which routing was worse; their decision (align2 chunk 64) is the same.
- **One gate flipped, in 1 of 4 runs: Engine A's v1/v4 split (the remaining two lines).** Run 1 chose v1 for T=1..8 and
  v4 for T=9..16; runs 2-4 chose v1 for T=1 only and v4 for T=2..16. Program §0 records run 1's pick as a tie (1.059 vs
  1.059) and counts class 8 going v1 in 3 of 7 board prepares, warm runs included.
- Cost of the flip: verify 1 (T=8) paired x1.2010 in run 1 against x1.2132 / 1.2179 / 1.2155, about -1.2%. decode 16
  is identical (x1.1804-1.1818).
- The other rows vary 0.5-1% between runs whose decisions matched (verify 16 x1.5573-1.5824, verify 32 x1.7999-1.8157).
- Headline spread across the 4 runs: 0.26% (x1.3893-1.3929). Cold prepare took 161.5-164.4 s (judge record 2a7b5d20:
  163.8 s).

### Held-out sweep and soak (25-h100-nj-moe-sweep, 26-h100-nj-moe-soak-1/2)

- Sweep: 37/37 tests pass, i.e. the judge 5 plus 32 held-out shapes (27 bs in 2..255 and 5 skewed). No held-out shape
  in that range failed on board ab46fbb7.
- Soak: 12,000 timed replays across two seeds, every one checked against the fp32 reference on fresh inputs: 0
  miscompares, and no fault or hang. Every row's paired ratio lies within 0.5% of the four cold runs' range.

### Stage-0 ladder replicates (20-h100-esl-*, 21-h100-esl-*)

- All 8 runs are correct: every KV line has ok=1, SUMMARY bad=0, every memcheck and synccheck run reports 0 errors,
  and K2CHECK has 0 mismatches.
- Replicates on different GPUs agree closely. For the cold and k4 replicate pairs, and for k1 and k2 against their
  validation runs, the median clk_k16 difference is 0.03-0.07% and the p90 at most 0.9%. Single lines differ by up to
  3.8-5.9%, and one k2 line by 25.7%. k7 matches its validation within 0.9% median and 3.2% max (B/clk), and k6 gives
  the same verdict.
- Against the 20:2xZ "quick" validation runs, esl-cold differs more (median 0.23%, max 12.5%), and k4 differs by a 1.3%
  median over its 32 shared lines. The quick forms are noisier, and their clocks dipped to 1696-1788 MHz.
- **Void lines: esl-k4 had 30 (rep1) and 24 (rep2) of 192 lines measured at 1665-1795 MHz.** All of them are mode=dev
  device-wide streamed lines at 1.0-3.0 TB/s. The cta lines that carry the K4 targets were at 1830 MHz and stand.
- K4 against its targets, from the cta lines: the ring K-step reaches 12.5-12.6 el/clk/SM at N=128 (target 12.8, 98%)
  but only 14.2-18.4 at N=64 (target 25.6).
- K6 is a NO-GO on both toolchains, the same verdict as the validation run.

### Not part of the harvest

- `runs/20261027-142513-gpunode06` (nodejudge-e1-validate, GPU 7) started at 21:25:27Z, after the 21:22Z hold. It died at
  once with `python: command not found`, because the concurrent `clean` was removing the env. `INDEX.tsv` lists it as
  DONE rc=0, but no GPU work ran.
