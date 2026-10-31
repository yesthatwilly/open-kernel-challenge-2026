
> **USER RULING 2026-10-31 ~05:10Z (verbatim):** "from this point on, im embracing the MIT license as i rebase this
> systematic improvement system onto the public board. there is no hope for simply transferring the knowledge and
> findings on the initial repo; using this board and attempting to improve it with a good system is now the goal. i do
> not expect this to make the serving board, so this is strictly an academic exercise to see if a systematic approach
> can improve on a public board enough to be competetive. adopt any and all the problem solutions as the new
> foundation."
>
> What this means for the repo:
> - **Master `submission.py` now comes from alygg77@6cae49a** (notices in `third_party/okc-muse-spark-kernels/`):
>   - attention = public-C A0 (2.4369 judge check mean);
>   - prefill = public-C P2 (1.7741);
>   - moe = public raw (2.3704);
>   - bf16 = public raw (1.5403).
> - **decode = public-C D0 since 10-31 ~07:43Z (scored 2.1793, was 1.7030; old line at tag `ours-decode-1703-1031`).**
>   moe = public + M1a NOFENCE (scored 2.3867); attention = A0 + PP softmax turns (scored 2.4729, 10:43Z) + C5L exact-max bookkeeping (scored 2.4880, 19:45Z); decode = D0 + DPP (scored 2.2220, 11:52Z). /3to1 resumed 06:05Z (user: "resume /3to1 on the new foundation. free
>   and open use of h100").
> - (historical) decode stayed ours until a compliant public-C decode (D0) existed. The raw public decode fails the FA3-precision
>   ruling.
> - Our pre-adoption files are at tag `ours-final-1031`.
> - **Superseded:** "public = target, never base", the arch-mirror clean-room, and public-plus "never on master".
> - **Still binding:** okc-stay-compliant, the attempt gate and the rules checklist.
> - **Scored, 10-31 ~05:10Z, now BEST:**
>
>   | problem | new board | old board |
>   |---|---:|---:|
>   | attention | **2.4337** | 2.005 |
>   | prefill | **1.7836** | 1.5721 |
>   | moe | **2.3670** | 2.100 |
>   | bf16 | **1.5657** | 1.3425 |
>
>   Everything is idle after the foundation (user: "stop after foundation"). Decode D0 is "explore soon"; compliance
>   is paramount.

> **NEWEST USER STEER, 2026-10-27 ~00:07Z (standing until the challenge closes): THE 3:1 DAILY RHYTHM.** Full text:
> `campaigns/DAILY-RHYTHM.md`; invoke `/3to1` (one tick) or `/loop /3to1`. It retires the 10-26 ~23:00Z STOP banner
> (distil done; the user resumed by hand at 00:03Z) and the 19:05Z "using every check is not required" rule.
> - **00:00Z until decode's reasonable checks are spent (phase A):** plan, build, deploy and check the biggest, most
>   promising decode rewrites/overhauls (moe as decode's testbed), on Opus 5.5 **ultracode** workflows.
> - **Then until 18:00Z (phase B):** the other problems' minor gains (moe > bf16 > attention > prefill).
> - **18:00Z-24:00Z (phase C):** heartbeats use up **every remaining check** on every problem; `/hypothesize` and other
>   work advancing the solutions; write the next day's phase-A plan by ~23:00Z.
> - **Always:** a scored submit whenever a check beats that board (unused scored runs are fine); attempt file and
>   rules checklist first; never re-roll identical bytes.

> **SUPERSEDED 2026-10-27 by `campaigns/DAILY-RHYTHM.md` (its phase order still holds) — USER STEER, 2026-10-26 ~19:05Z, relayed from the user's openclaw session. Not yet acknowledged. Read before acting on this tick.**
>
> **Do not idle until 00:00Z, and do not pre-build hours ahead of a reset.** Decode and moe quota is spent for 10-26. Their
> 00:00Z plan is already written (submit 1.5347, moe dq A/B + H=-8 + v4c, decode H=-8 + v4c + anatomy). Stop adding to it.
> **Now:** move to the highest-gain problem that still has checks today and work there with immediate check feedback.
> Priority by gain for decode: decode > moe (~53% of weighted decode time, x1.19-1.27) > **bf16_proj** (~26%, x1.03-1.04)
> > **attention_decode** (~15%, x1.01-1.06) > prefill (no decode value; last). So bf16_proj now, then attention_decode,
> favouring kernels that port into decode (campaign #2/#3: CUDA qkv/o/P on bf16, B=1 attention ports). Submit when a
> check beats that board. Return to decode/moe at 00:00Z. Apply the same rule every day: when a problem's checks run
> out, write its reset plan, then switch to a problem with checks left.
> Using every check is not required; never spend checks on low-value questions. Unchanged: every check answers a new
> question, no identical bytes re-rolled, challenge rules apply. Full text: PLAN.md section 6 (top) and section 11.
> **Acknowledge in LOG.md with a dated entry and keep this policy in the Now block.** Your Now rewrites at 18:49Z and
> 19:05:50Z overwrote earlier copies of this note unread.

# Campaign index

| campaign | status | objective / KPI | branch | start | deadline | best so far |
|---|---|---|---|---|---|---|
| [2026-10-26-decode-first-rewrite](2026-10-26-decode-first-rewrite/PLAN.md) | closed ([RETRO](2026-10-26-decode-first-rewrite/RETRO.md)) | overall board; KPI muse_decode_layer headline | `campaign/decode-first-rewrite` | 2026-10-26 | 2026-11-05 06:59Z | decode 1.3164 -> 1.4422, moe 1.402 -> 1.5762, overall 480 -> 362 (at close) |
| [2026-10-26-expert-bandwidth](2026-10-26-expert-bandwidth/PLAN.md) | **superseded 10-31** by public-plus (foundation); was: the 3:1 rhythm since 2026-10-27 00:07Z; live state at the top of [LOG.md](2026-10-26-expert-bandwidth/LOG.md); X5 Hopper work on hold since 2026-10-27 21:22Z | overall board via decode; KPI muse_decode_layer headline, testbed moe | `campaign/expert-bandwidth` | 2026-10-26 | 2026-11-05 06:59Z | decode 1.4422 -> 1.4956 (check 1.5347 to score), moe 1.5762 -> 1.6851, bf16 1.2566 -> 1.2664; overall 370/1265 |
| [2026-10-31-public-plus](2026-10-31-public-plus/PLAN.md) | **active**: /3to1 on the public foundation since 2026-10-31 06:05Z (user: "resume /3to1 on the new foundation. free and open use of h100"); earlier: check-only public-C track; live state at the top of [LOG.md](2026-10-31-public-plus/LOG.md) | does anything this repo learned make the proven public files (alygg77@6cae49a) faster under our compliance rules; KPI per-row ratio vs same-day control | `campaign/public-plus` (variant files + attempts); docs on master | 2026-10-31 | 2026-11-04 12:00Z | inventory: ours faster on 0/68 rows; 1 of 48 graft candidates survived review (attention W2). Controls 10-31: public bf16 1.5403, moe 2.3704 (A/A 0.5%) |
