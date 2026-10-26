# Campaign instructions (from the user)

> **STOP — NEW USER STEER, 2026-10-26 ~21:40Z, relayed from the user's openclaw session: EXPIRING-QUOTA SPRINT until 00:00Z. Act on it this tick.**
>
> The user explicitly chose (over a one-agent option and a bf16+attention-only option): **use as much of today's remaining
> checks and scored runs as possible before 00:00Z, ~30 hypotheses, across all pools with quota: bf16_proj, attention_decode
> AND prefill** (and moe's 4 scored if a moe check ever beats its board; none expected). This overrides, for this window only:
> "churn paused", "prefill stays idle", "<=3 concurrent agents" and "one Opus audit per module".
> - **Run it in parallel:** revive the hypothesis pipeline (/hypothesize + /churnsubs, one worktree per hypothesis) with
>   **about 8-10 concurrent agents**. One check answers each hypothesis; submit the exact bytes of any check that beats its board.
> - **Priority inside the sprint:** bf16_proj first (decode-portable ideas, bulk12 variants), then attention_decode, then prefill.
>   Draw ideas from each problem's hypotheses/ backlog and FINDINGS, not only new ones.
> - **Keep:** the rules checklist on every check and submit; no identical bytes re-rolled; fresh hypotheses only.
>   Lighter audits are acceptable for small deltas; a new CUDA module still gets a quick read-only audit.
> - **Do not touch** decode or moe code or quota, or the 00:00Z reset plan. Your own bulk12 line keeps going alongside.
> - **Hard stop:** dispatch nothing after ~23:45Z; harvest and land winners; at 00:00Z return to the reset plan and the
>   19:05Z policy below (single agent, gain-priority order). Log the sprint's results (checks used, wins, landed) in LOG.md.

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
