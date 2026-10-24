You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_attention_decode/hypotheses/queue/2026-10-23T223550Z-fused-preattention-triton.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_attention_decode/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_attention_decode/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   /tmp/claude-1000/churn-reviews/landed-muse_attention_decode.json
5. /tmp/claude-1000/churn-reviews/attn-decode-dirty-submission.py.bak (the exact code this hypothesis will run: an uncommitted fused pre-attention Triton kernel that already passed 16/16 as check job dfee6f2939974bad at 1.127x [1.111,1.144]; its profile may be under muse_attention_decode/profiles/). Master's muse_attention_decode/submission.py is the starter (BEST 0.0023x), so this is the first real base for this problem. — the kernel the hypothesis will be built on (its base).
6. muse_attention_decode/baseline.py — SGLang's own pipeline (the named parts), muse_attention_decode/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): -fa3-batched-q-verify -fa3-drop-cu-seqlens-decode -fa3-hdim128-head-pairing -fa3-num-splits-full-decode -fa3-num-splits-full-verify -fa3-num-splits-sliding -fa3-scheduler-metadata-full-verify -fa3-single-split-sliding -fa3-window-trimmed-page-table -fused-full-combine-normgate -fused-preattention-triton -fused-sliding-normgate-epilogue -fused-sliding-preattn-into-window-kernel -paired-heads-launch-cleanup -sglang-path-parity -stream-fork-preattention -triton-flash-decode-deep-pipeline -triton-flash-decode-full-both-heads -triton-flash-decode-full-verify -triton-window-attention-sliding

Answer these, adversarially:
A. **Justification.** Is the stated mechanism plausible on this hardware for these shapes? Does
   the profile data in FINDINGS support it (which kernels, how many µs, how far from the
   speed-of-light floor)? Are the `expected` shapes and directions consistent with the shape
   weights and floors in TASK.md? Name the weakest link in the argument.
B. **Confounding by what already landed.** For each landed hypothesis that touches the same
   area, the same launch sequence, the same memory traffic, or the same numerical path: could a
   gain or loss measured for this hypothesis be caused by the interaction rather than by the
   change itself? Would the change behave differently if that landed hypothesis were absent or
   tuned differently?
C. **Interference and A/B.** Decide specifically whether an additional run is warranted to make
   *this* hypothesis's single result meaningful: (i) this change with landed hypothesis X
   reverted (`ab-revert`), (ii) this change with X tuned differently (`ab-tune`), or (iii) a
   control that isolates one of the hypothesis's own sub-changes (`control`). A companion is
   warranted only when, without it, both a gain and a loss would be ambiguous. At most two.
   Each must be implementable from the parent's branch: say exactly what to revert (the patch
   path from the landed list, applied in reverse) or what to tune, and what stays.
D. **Rules.** Flag anything in the change that could break the competition rules (precision,
   caching derived from inputs, every-key attention, host syncs, work outside the timed call).

Return ONLY a JSON object, no prose around it, with this schema:
{
 "model": "opus",
 "verdict": "proceed" | "hold" | "proceed-with-companions",
 "summary": "one or two sentences",
 "justification": "your assessment of A, including the weakest link",
 "confounders": ["one line per landed hypothesis that confounds, with why", ...],
 "interference": "your reasoning for C: why companions are or are not needed",
 "rules_flags": ["...", ...],
 "companions": [
   {"slug": "<parent-slug>-ab-revert-<x>", "kind": "ab-revert" | "ab-tune" | "control",
    "area": "...", "change": "a paragraph implementable from the parent's branch without asking",
    "expected": [{"shapes": "<glob over shape keys>", "direction": "faster|slower|flat", "why": "..."}],
    "reason": "what this companion disambiguates", "priority": 1-5, "risk": "low|medium|high"}
 ],
 "open_questions": ["...", ...]
}
`hold` only when the hypothesis is unjustifiable, duplicates a landed/do-not-retry item, or would
break a rule. Prefer `proceed` with sharp notes over a timid `hold`.


Additional context: This hypothesis is now the base for 12 queued dependents (re-pointed from the held sglang-path-parity at the user's direction). Its result already exists (job dfee6f2939974bad); identical code resubmitted returns that job. Judge mainly the rules compliance and whether the code is a sound, upstreamable base; flag anything that would make every dependent's result suspect. Per-shape numbers are in result.json files under hypotheses/done/.

When done, ALSO write your JSON object to /tmp/claude-1000/churn-reviews/review-fused-preattention-triton.json (that one file only), then return it.
