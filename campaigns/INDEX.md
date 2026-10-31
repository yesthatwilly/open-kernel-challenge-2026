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
| [2026-10-26-expert-bandwidth](2026-10-26-expert-bandwidth/PLAN.md) | **active**: the 3:1 rhythm ([DAILY-RHYTHM.md](DAILY-RHYTHM.md)) since 2026-10-27 00:07Z; live state at the top of [LOG.md](2026-10-26-expert-bandwidth/LOG.md); X5 Hopper work on hold since 2026-10-27 21:22Z | overall board via decode; KPI muse_decode_layer headline, testbed moe | `campaign/expert-bandwidth` | 2026-10-26 | 2026-11-05 06:59Z | decode 1.4422 -> 1.4956 (check 1.5347 to score), moe 1.5762 -> 1.6851, bf16 1.2566 -> 1.2664; overall 370/1265 |
| [2026-10-31-public-plus](2026-10-31-public-plus/PLAN.md) | **active** since 2026-10-31T00:29Z (check-only, never scored); live state at the top of [LOG.md](2026-10-31-public-plus/LOG.md) | does anything this repo learned make the proven public files (alygg77@6cae49a) faster under our compliance rules; KPI per-row ratio vs same-day control | `campaign/public-plus` (variant files + attempts); docs on master | 2026-10-31 | 2026-11-04 12:00Z | inventory: ours faster on 0/68 rows; 1 of 48 graft candidates survived review (attention W2) |
