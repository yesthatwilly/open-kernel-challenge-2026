# Public-plus: LOG

## Now (2026-10-31T09:22Z) — /3to1 on the public foundation; phase A landed D0 + M1a

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
