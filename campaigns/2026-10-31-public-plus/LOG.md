# Public-plus: LOG

## Now (2026-10-31T11:20Z) — /3to1 on the public foundation; phase B: PP LANDED 2.4729; S6M checks + DPP audit in flight

- **~06:05Z, the user (verbatim): "resume /3to1 on the new foundation. free and open use of h100".** This lifts the
  "stop after foundation" hold. The 3:1 rhythm (campaigns/DAILY-RHYTHM.md) now runs on master's foundation files,
  and this campaign is the active one. H100: free use on gpunode06 GPUs 1/3/4/5/6, under the hopper-probe guard.
- Boards (BEST, 10-31): attention 2.4337, prefill 1.7836, moe 2.3670, bf16 1.5657, decode 1.6989 (ours).
- Quota at 06:05Z: decode 20 checks/10 runs; moe 18/9; bf16 18/9; attention 18/8; prefill 14/9.
- Phase A (06:06Z): an ultracode workflow with three items.
  - **D0**: a compliant public decode. It ports the A0 softmax fix into ATT7 and moves tvm_ffi load_inline to torch;
    the kill rule is PLAN §7 #11.
  - **M1-stack**: the moe testbed (PLAN §7 #3).
  - **A1**: attention W2 on A0, the phase-B rider.
  - Workflow wf_1c6314b7-81b (task wzlr8upqf), launched 06:10Z. Design runs at effort max; build and audit at high. GPUs: D0 1,3; M1 4,5; A1 6. Variant files go in `files/<p>/` on campaign/public-plus,
    rebased on master.
- **07:33Z: phase-A workflow done** (6 agents, about 1.6M tokens).
  - **D0 build and audit: PASS.**
    - Independent checks on the H100: 24/24 scored rows, 13/13 TASK tests, 100/100 adversarial cases, 36/36 R11 cases.
    - Speed: x0.997 against the public decode on the H100, so the judge estimate is about 2.16-2.20.
    - The four tvm_ffi glue modules now use torch load_inline; that move costs nothing.
    - Gray-area flags resolved in the attempt file: ctypes into cuBLASLt, `--use_fast_math` and the GLE T=1 pooler.
  - **M1:** only M1a NOFENCE helps. d32 is x1.007 on the H100 and the headline about +0.1%, so it gets one read. The
    M1s stack measured -0.5% and is not sent.
  - **A1 / A1n4 (W2) KILLED:** v1s is 2-4% slower on the H100 (A0 already differs from our old base).
- **07:34Z checks dispatched:**
  - D0-r1 and D0-r2 (decode): de38cec7 and 4b2adae2, each mapped to its file in the attempt files on harvest.
  - M1a-r1 (moe): eaf35636.
- **Concurrent session:** union-confirm landed decode **1.7030** (63d07851, swap-AB root-cause fix, master eb66b591)
  at 07:20Z on our old decode line. D0 is now measured against 1.7030. If D0 wins, it replaces that line on master
  (foundation ruling); union-confirm's LOG must be told.
- **~07:43Z LANDED on master (scored, register-landing done):**
  - **decode D0 public-C: 2.1793** (201db0b8; checks 2.2060 4b2adae2 and 2.1708 de38cec7). Was 1.7030, so +28.0%.
    Our old decode line is at tag `ours-decode-1703-1031`; **all five problems now sit on the public foundation.**
  - **moe M1a NOFENCE: 2.3867** (f4e16629; check eaf35636 2.3867). Was 2.3670, so +0.83%, borderline (under 1.2%).
  - union-confirm: its decode line is superseded by the foundation ruling; its swap-AB lesson stays in LESSONS.
- **Next (phase B, since decode's phase-A item is landed):**
  - 07:46Z: lever-finding workflow wf_d49ebbab-bc8 launched: 5 effort-max finders (GPUs 1/3/4/5/6) and a max judge; its ranking picks the next build;
  - then the cheap riders on moe, bf16 and prefill, at most 3 agents;
  - decode checks left: 18.
- **09:14Z lever workflow wf_d49ebbab-bc8 done** (6 agents, 3.3M tokens). The ranking is in LEVERS-1031.md:
  1. attention pp/ppstsm (softmax ping-pong; about +1%);
  2. moe S6, a 6-bit scale code (about +1.2-1.5%, then a decode port);
  3. decode DL2H, the TINY L2 hints (about +0.5% d1);
  4. the prefill knob bundle;
  5. the bf16 nvjet race;
  6. moe HF;
  7. decode cgemv;
  8. compact R11 (parked).
- **09:20Z decode DL2H checks:** 5b0f8dae (r1) and cc512a50 (r2). GO if d1s moe is at least 1.5 us faster than D0 in
  both reads, with no other row more than 0.5% worse.
- **09:19Z DL2H KILLED:** reads 2.1912 and 2.2029 (mean 2.197) against D0 checks of 2.188. That is noise, and d1s is flat. The H100's L2-hint gain does not transfer to the H200. Not scored; a CLAIMS refuted line is added. Decode checks left: 16.
- **09:22Z build workflow wf_2144e5b0-bff:** S6 (design max; GPUs 3,4), ATT pp/ppstsm (GPU 5), PF knobs (GPU 6), BF
  race (GPU 1). Each is built, then audited.
- **10:28Z build workflow wf_2144e5b0-bff done** (8 agents, 1.8M tokens; branch commits f489056f, db79fa99, c7bd8479,
  f509fbf9).
  - **S6 (moe): audit PASS.** H100 x1.020 / x1.021 on every row; prepare 144.7 s.
  - **ATT PP / PPS (attention): audit PASS.** L2-resident v16f/v32f x0.951-0.959; non-flat SASS identical.
  - **BF BFR (bf16): audit PASS.** v16 pooler picks non-coop nvjet in 10/10 prepares, x0.986-0.990.
  - **PF PFK / PFK0 (prefill): KILLED on the H100 builder gate** (+0.16% / +0.09%, below +0.35%). Recorded as a
    node-only CLAIMS line; the reopen is a spare phase-C check.
- **10:29-10:30Z, 9 checks dispatched** (attempt files 2026-10-31T1029*Z on campaign/public-plus 45106f85; gates
  pre-registered in each):
  - moe S6-r1 / r2 / r3: 7748ec1f, 3bd96942, 1aaf3cc1 (r3 = mk_s6.py n=3);
  - attention PPS-r1 / r2: a8425ac5, e11914e5;
  - attention PP-r1 / r2: ac0335f9, d933848f;
  - bf16 BFR-r1 / r2: 7d43d14a, a16e53de.
  - The claims hook warned about T1 AT2W4 on ATT. That is a false match (a different mechanism), noted as Reopen in
    the attempt files.
- **Open note for the user, before any S6 scored submit:** the S6 gain depends on the generator's narrow scale range
  (bytes 0x5C..0x7E). Wide-scale weights fall back to the 8-bit layout, with no gain and no breakage. Default: score
  it if the gate passes, and say so in the landing report.
- **10:36Z, 9 checks harvested** (results are in the attempt files, b9d5cba5):
  - **ATT PP: GO.** Reads 2.5236 and 2.4515 (mean 2.487, +2.2% over 2.4337). v16f 111.6/112.8 us, v32f 204.8/206.2 us.
    - PPS also passed its gate rows (2.4529 / 2.4274), but PP is kept.
    - Gate note: d1s mean 5.515 is +0.315 us over the stale 0ef88b07 base (5.20). Against the same-day A0 reads it is
      +0.155, with identical SASS, so I treated it as noise. This is disclosed in the submit attempt.
    - **Scored submit 14cec684** (PP-score.py, body = PP-r1).
  - **S6: HOLD, not scored.** d32 is -2..-3% in 3/3 reads, but the headline mean is 2.3548 (-1.3%). v1 is +3..10% and
    d16 +2..4%. The H100's uniform x1.020 did not transfer. CLAIMS line: unresolved.
  - **BFR: KILL.** The H200 race kept coopB on the v16 pooler in both reads, a no-op. Headline 1.5533/1.5523 is under
    1.5657. CLAIMS line: node-only.
- **10:45Z build workflow wf_514e7d34-141** (task w8s1bc9rk; build and audit at high):
  - DPP, the decode port of PP: re-review decode's #error-guarded OKC_TURN, or port PP's 3 edits (GPUs 5, 1);
  - S6M, S6 only in the build(s) serving d32 (GPUs 3, 4);
  - HF, the moe helper fence drop (GPU 6).
  - Decode cgemv is left for phase C.
- **10:43Z LANDED on master: attention PP 2.4729** (scored 14cec684 [2.4358, 2.5233]; was 2.4337, so +1.6%).
  register-landing is done. Rows: v16f 113.71, v32f 207.28.
- **11:18Z wf_514e7d34-141 done** (4 agents, 0.8M tokens):
  - **S6M: audit PASS.** S6 in the WGS copy and MAIN build only, which serve d32, v16 and v32 (d32 cannot be
    isolated). TINY, TINY2 and MID keep master's SASS. **Checks 83c362cf and bbcf6b9c** (11:19Z). The GO gate is in the
    attempt files. The predicted outcome is close to a coin flip, about +0.2%.
  - **HF: KILLED on the H100.** No effect after A/A correction. CLAIMS line: node-only.
  - **DPP (decode port of PP): built** (208760ec); correct and bit-identical to D0 under fixed plans. It missed the H100
    size bar: v16f -1.6..-1.95 us, v32f -1.9..-4.2 us. PP itself read -1.1/-1.4 us on the H100 and -5.4/-9 us on the
    judge, so the default is a judge read after an independent audit. Audit agent launched 11:20Z (GPU 5); 2 decode
    checks follow if it passes.
- **11:22Z S6M KILLED:** 2.3640 / 2.3822 (mean 2.3731, under 2.3867); d32 128.67 / 124.53 against the 124.0 bar. The d16
  row runs code identical to master, yet read 94.9/96.8 against 91.3/91.9, so judge row noise is about 5% today. The
  S6 family is closed for moe (CLAIMS refuted). moe checks left: 12.
- **11:25Z lever round 2, wf_c7b20f1e-b31** (task wt39bo22e): 3 effort-max finders (decode GPU 1, moe GPUs 3/4,
  attention GPU 6) and a max judge. Briefed on today's transfer record: favour compute, issue, latency and launch
  levers, with judge profiles as the primary evidence. Output: LEVERS-1031b.md, which feeds the rest of phase B and
  tomorrow's phase A.
- **Next:** the DPP audit, then 2 decode checks. Phase C (18Z): spend the remaining checks.
- Compliance stays first (the user: "compliance is of absolute utter importance"): FA3 precision, the attempt gate
  and the rules checklist.

## Earlier Now (2026-10-31T03:45Z)

- **~03:50Z, the user (verbatim): "stop lifted for the 10 public-C checks, go ahead".** This lifts the 00:32Z stop
  only for the following (PLAN §14):
  - A0 and A0′ (attention), P0, P1 and P2 (prefill), 2 reads each, 10 check-only runs;
  - one builder plus one audit agent;
  - H100 correctness on gpunode06 GPUs 1/3/4/5/6.
  - Everything else stays stopped: M1-stack, A1/A2, D0 and the launch diet.
- **05:0xZ: public-C HARVESTED** (8/8 passed; `publicC-*-1031.md`, `runs/`, RESULTS.tsv):

  | arm | reads | mean | A/A | vs our board | vs published |
  |---|---|---:|---:|---:|---:|
  | A0 attention (exact max, P<=256, FP32 denom, R11) | 2.4361, 2.4377 | 2.4369 | 0.07% | +21.5% (2.005) | -8.1% (2.653) |
  | P0 prefill (FA3 everywhere) | 1.7448, 1.7398 | 1.7423 | 0.28% | +10.8% (1.5721) | -4.1% (1.8167) |
  | P1 prefill (P0 + contiguous gather) | 1.7353, 1.7464 | 1.7408 | 0.64% | +10.7% | -4.2% |
  | P2 prefill (own kernel, FP32 denom) | 1.7848, 1.7634 | 1.7741 | 1.21% | +12.9% | -2.3% |

  - A0 and P2 are faster than our BEST on every row (A0 ours/theirs 1.06-1.62; P2 1.11-1.20).
  - A0's compliance cost is concentrated in full v16/v32 (+17.5% vs their 10-29 row).
  - P1's gather adds nothing.
  - Adoption decision (PLAN §14) now waits on the user. The stop holds for everything else.
- 04:29Z: dispatched the 8 public-C checks (check-only).
  - A0 attention: 5b751676 (r1), 3ba805db (r2).
  - P0 prefill: 04b9092b, cb644182.
  - P1: b8b44f36, 253ff43c.
  - P2: d114d2e8, c456c4b9.
  - Builder: campaign/public-plus de7ef03f. Independent audit: PASS on A0, P0, P1 and P2.
  - A0′ dropped: P reaches 432, above the user's P_max 256.
  - R11 kept: robustness; the raw public file returns NaN on large logits.
  - 2 of the 10 allowed checks are unused.
- §13 done: KERNEL-COUNTS.md and the FINDINGS "Answer 10-31".
- Adopting vjeux's attention and prefill as our base is the user's call, after public-C reads (PLAN §14).

## Earlier Now (2026-10-31T00:48Z)

- **USER STOP, ~00:32Z (verbatim):** "do not launch any new work. get the four control runs back, harvest, process,
  plan out. but do not launch more agents or checks submissions or h100 work". It still holds. Nothing is running
  from this campaign.
- **Controls HARVESTED** (FINDINGS "Results so far"; `controls-*-1031.md`; `runs/`):

| problem | mean | A/A | published | our board | gap today |
|---|---:|---:|---:|---:|---:|
| bf16 | 1.5403 | 0.50% | 1.5502 | 1.3425 | x1.147 |
| moe | 2.3704 | 0.47% | 2.4015 | 2.1000 | x1.129 |

  Ours is slower on every row against today's public reads.
- **Plan out:** PLAN §12.
  - Option B (recommended): 3 builders; about 10 checks; compliant attention/prefill targets + W2 + the moe stack.
  - Option A: the full build.
  - Option C: stop the track.
  - The choice is waiting on the user.
- **Quota 10-31:** bf16 18/20 checks, moe 18/20, the others 20/20. Scored runs unchanged.
- **Build workflow wf_b31b245a-546:** stopped at 00:32Z before any edit; worktrees and branches cleaned.
- union-confirm: IDLE at master d6a0d421 (attention 2.005).

## History
- 2026-10-31T00:48Z: harvested B0/M0 (bf16 1.5403, moe 2.3704); attempt Results on campaign/public-plus 98be5aac; PLAN §12 written.
- 2026-10-31 00:28-00:31Z: step 0 committed (master e241fdfd); branch campaign/public-plus + worktree; B0/M0 controls dispatched 00:30Z (e18ff591).

- 2026-10-30 23:28Z - 10-31 00:14Z: read-only inventory workflow wf_1949358b-dd0 (11 agents, ~3.8M subagent tokens,
  45 min). The result is in FINDINGS.md and PLAN §3-4.
  - Ours is faster on 0 of 68 rows; the public file is faster on every one.
  - Of 48 graft candidates, 1 speed graft survived (attention W2).
  - The public attention, decode and prefill files fail our FA3-precision ruling; moe and bf16 are clean.
- 2026-10-30 ~23:45Z: user decisions (AskUserQuestion):
  - public base plus grafts;
  - MIT routes kept;
  - the main line gets checks first;
  - up to 6 agents.
