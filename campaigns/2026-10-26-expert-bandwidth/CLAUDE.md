# Campaign instructions (from the user)

> **STOP — NEWEST USER STEER, 2026-10-26 ~22:00Z, relayed from the user's openclaw session: THE SPRINT IS ENDED. No new work. Act on it this tick.**
>
> The user said: "stop any more new sprints. Keep what's in flight." This replaces the ~21:40Z expiring-quota sprint banner.
> - **Start nothing new:** no new /hypothesize minting, no new adversarial reviews, no new implementations, no new
>   worktrees, no new subagents of any kind. Minted hypotheses not yet implemented stay queued; do not implement them.
> - **Keep what is in flight:** let agents already running finish; harvest checks already dispatched; submit the exact
>   bytes of any harvested check that beats its board (rules checklist as usual). A hypothesis already implemented
>   and committed when you read this may be dispatched; anything earlier than that stops.
> - Your own bf16 line likewise: finish and check what is already built/audited (b12r); do not start another build.
> - **Then go quiet** until the 00:00Z reset plan (decode scored run, moe + decode checks). Long heartbeats while idle.
> - Log what the sprint did (checks used, wins, landed, what was left queued) in LOG.md.

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
