# Campaign instructions (from the user)

> **STOP — NEWEST USER STEER, 2026-10-26 ~23:00Z, relayed from the user's openclaw session: DISTIL YOUR CONTEXT, THEN STOP THE LOOP. DO NOT RUN THE 00:00Z RESET PLAN. Act on it this tick.**
>
> The user's goals, in order: (1) capture the value in this session's context; (2) the user resumes the campaign
> manually in a fresh session after clearing context. This replaces the ~22:00Z "sprint ended" banner.
> - **No judge actions at all:** no okc check, submit or withdraw; the 00:00Z reset plan is on hold for the new
>   session. No builds, no new hypothesis work. Let the three minting agents finish and queue their output.
> - **Write `campaigns/2026-10-26-expert-bandwidth/DISTILL.md`** on master. Spend your context here: quality over
>   brevity, but no padding. A cold session with none of your context must be able to resume from it. Include:
>   1. **Resume block:** exact state, boards and ranks, quota, branches/worktrees/HEADs, and the first actions in order (the
>      full 00:00Z reset plan: `submission.edoff-27ef83df.py` hash 613b6c2d / check f59cb301; the moe dq A/B + H=-8 + v4c
>      check at 731d149e; the decode H=-8 + v4c + anatomy check; then the bulk12 decode port at T=1).
>   2. **Copy every scratchpad draft into the repo** (`campaigns/2026-10-26-expert-bandwidth/scratchpad/`: submit_attempt.md,
>      dqab_attempt.md, anat_attempt.md, now.md, log_now.py, bfcmp.py, b12_section.py, anything else still needed). /tmp
>      does not survive and a new session cannot find it.
>   3. **Judgment that exists only in your context:** why each killed line died, and what would make it worth reopening;
>      where the judge disagrees with the prepare-time gates (b12r m=8, Engine D, the tie-break); levers ranked by
>      expected gain, effort and kill rule; traps and dead ends not to retry; what you would do differently.
>   4. **A hardened plan for the next work:** the rest of expert-bandwidth (dq3/v4c/H=-8 outcomes as branches), the
>      bulk12 decode port, then `decode-chain`; the queued sprint hypotheses (attention sliding s3 "proceed", the bf16
>      combo, the prefill items) with your verdict on each.
>   5. **Operating lessons for the next loop:** `date -u` stamps; read the Now block before rewriting it; the user's
>      steers arrive as banners in INDEX.md, PLAN.md and the campaign CLAUDE.md; heartbeats vs prompt-cache expiry;
>      keep the main context lean by delegating execution to subagents.
> - Read and fold in `research/openclaw-relay-2026-10-26.md` (the judge-weighted time shares by area, steering-channel
>   findings, cost data). Update FINDINGS.md with anything that is only in your context, and update your okc-* memories
>   where facts changed (mark this campaign paused for a manual resume).
> - **Then:** set the LOG Now block to "PAUSED for a manual resume by the user; read DISTILL.md first", commit, stop the
>   loop (`ScheduleWakeup` with stop), and give a summary of 15 lines or fewer. Do not restart the loop.

> **STOP — USER STEER, 2026-10-26 ~19:05Z, relayed from the user's openclaw session. Not yet acknowledged. Read before acting on this tick.**
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
