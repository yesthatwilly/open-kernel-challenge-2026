# Public-plus: LOG

## Now (2026-10-31T03:45Z)

- **~03:50Z, the user (verbatim): "stop lifted for the 10 public-C checks, go ahead".** This lifts the 00:32Z stop
  only for the following (PLAN §14):
  - A0 and A0′ (attention), P0, P1 and P2 (prefill), 2 reads each, 10 check-only runs;
  - one builder plus one audit agent;
  - H100 correctness on gpunode06 GPUs 1/3/4/5/6.
  - Everything else stays stopped: M1-stack, A1/A2, D0 and the launch diet.
- 04:29Z: 8 public-C checks in flight (check-only).
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
