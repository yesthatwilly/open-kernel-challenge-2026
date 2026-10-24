You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_prefill_layer/hypotheses/queue/2026-10-23T230250Z-moe-align-side-stream.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_prefill_layer/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_prefill_layer/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   /tmp/claude-1000/churn-reviews/landed-muse_prefill_layer.json
5. muse_prefill_layer/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_prefill_layer/baseline.py — SGLang's own pipeline (the named parts), muse_prefill_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): -experts-dequant-grouped-mm-cutlass -experts-dequant-overlap-attention -experts-dequant-threshold-turn4 -experts-dequant-triton-moe-16k -experts-triton-w4a16-inkernel-fp4 -fa3-batched-mode -fa3-contiguous-kv-gather -moe-align-side-stream -qk-norm-fp8-cache-store-fusion

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


Additional context: Master just landed sglang-pipeline-parity (check 1.0039x [0.9888,1.0189], job 62025c368c3c4e8e; all 8 shapes within 0.987-1.009x); its per-shape result and profile are in muse_prefill_layer/hypotheses/done/gain/ and the worktree .worktrees/muse_prefill_layer/sglang-pipeline-parity/muse_prefill_layer/profiles/. This is the first SGLang-path profile at prefill shapes: use it to check where time goes. moe_w4a16 (sister problem) found: Triton W4A16 GEMV with in-kernel FP4 dequant was 2-2.6x SLOWER than Marlin at small M (triton-w13-grouped-gemv-small-m, 0.751x); next-larger Marlin block_size_m lost 8%.

When done, ALSO write your JSON object to /tmp/claude-1000/churn-reviews/review-moe-align-side-stream.json (that one file only), then return it.
