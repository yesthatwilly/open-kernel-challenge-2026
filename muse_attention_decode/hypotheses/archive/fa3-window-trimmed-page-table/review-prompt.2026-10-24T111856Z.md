You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_attention_decode/hypotheses/queue/2026-10-23T225858Z-fa3-window-trimmed-page-table.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_attention_decode/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_attention_decode/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_attention_decode",
 "landed": [
  {
   "slug": "fused-preattention-triton",
   "area": "pre-attention norm+rope",
   "outcome": "gain",
   "speedup": 1.1274459340435283,
   "promotion_commit": "3e98bd00dcb717e6e3670b7819e7d1c2331f52ba",
   "patch": "muse_attention_decode/hypotheses/done/gain/fused-preattention-triton.patch",
   "md": "muse_attention_decode/hypotheses/done/gain/2026-10-23T223550Z-fused-preattention-triton.md",
   "change": "Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at offset h*64 "
  },
  {
   "slug": "fa3-hdim128-head-pairing",
   "area": "FA3 call (head packing)",
   "outcome": "gain",
   "speedup": 1.1705932092473075,
   "promotion_commit": "86dc7133cb6594c6738fddf31911079e71700bfc",
   "patch": "muse_attention_decode/hypotheses/done/gain/fa3-hdim128-head-pairing.patch",
   "md": "muse_attention_decode/hypotheses/done/gain/2026-10-23T225840Z-fa3-hdim128-head-pairing.md",
   "change": "On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2 (Hkv is even: 2 under tp8, 16 under dp; "
  }
 ]
}

5. muse_attention_decode/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_attention_decode/baseline.py — SGLang's own pipeline (the named parts), muse_attention_decode/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): -fa3-batched-q-verify, -fa3-drop-cu-seqlens-decode, -fa3-num-splits-full-decode, -fa3-num-splits-full-verify, -fa3-num-splits-sliding, -fa3-scheduler-metadata-full-verify, -fa3-single-split-sliding, -fa3-window-trimmed-page-table, -fused-full-combine-normgate, -fused-sliding-normgate-epilogue, -fused-sliding-preattn-into-window-kernel, -paired-heads-launch-cleanup, -paired-heads-launch-cleanup-ab-tune-batch-gate, -sglang-path-parity, -stream-fork-preattention, -triton-flash-decode-deep-pipeline, -triton-flash-decode-full-both-heads, -triton-flash-decode-full-verify, -triton-window-attention-sliding

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

If the hypothesis has `kind: rewrite`, it replaces a whole subsystem over several staged runs and
is judged on the replaced kernels' own time against the incumbent and the speed-of-light floor (its
`## Milestones`), not on the headline. For a rewrite, weigh instead: is the design sound for this
hardware and these shapes (parallelism, bytes in flight, dequant cost, tensor-core path), is the
correctness plan adequate, is the stage's milestone readable from the profile, and what is the
single biggest risk to fix in the next stage. Do not hold a rewrite because one run cannot
attribute a headline change or because the first stage may be slower than the incumbent; hold it
only for a rules problem or a design that cannot reach its floor. Companions are rarely useful for
a rewrite: the next stage is the follow-up.

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


Also read /home/user/git/musespark/hypotheses/LESSONS.md and /home/user/git/musespark/hypotheses/DECISIONS.md. No GPU on this node; judge on design.
