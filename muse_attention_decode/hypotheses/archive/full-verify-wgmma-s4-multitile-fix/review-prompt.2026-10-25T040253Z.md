You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_attention_decode/hypotheses/queue/2026-10-25T035310Z-full-verify-wgmma-s4-multitile-fix.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_attention_decode/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_attention_decode/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_attention_decode",
 "landed": [
  {
   "slug": "sliding-deep-splits-on-cuda-splitkv",
   "area": "sliding attention, batched decode, split depth",
   "outcome": "gain",
   "speedup": 1.3948738389144513,
   "promotion_commit": "0a56d22c1f1e824cf006d1c3c939fb7fa0d0e962",
   "patch": "muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.patch",
   "md": "muse_attention_decode/hypotheses/done/gain/2026-10-24T145005Z-sliding-deep-splits-on-cuda-splitkv.md",
   "change": "Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no conflicts): the fused sliding kernel'"
  },
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
  },
  {
   "slug": "paired-heads-launch-cleanup",
   "area": "attention boundary",
   "outcome": "gain",
   "speedup": 1.2436140703098866,
   "promotion_commit": "fa95f6d9e5bbaa7ab5978727425c8247ebbaa5c1",
   "patch": "muse_attention_decode/hypotheses/done/gain/paired-heads-launch-cleanup.patch",
   "md": "muse_attention_decode/hypotheses/done/gain/2026-10-23T225840Z-paired-heads-launch-cleanup.md",
   "change": "Stacked on fa3-hdim128-head-pairing (and using fused-preattention-triton's kernel, which must have landed too): remove the five pad/un-pad launches the pairing added on full layers. (1) Pad at the source: the fused pre-attention Triton kernel's query programs (one per (token, head)) write their 64 n"
  },
  {
   "slug": "paired-heads-launch-cleanup-ab-tune-batch-gate",
   "area": "FA3 call (head packing) gate",
   "outcome": "neutral",
   "speedup": 1.2686474414890054,
   "promotion_commit": "bfb7bac32c93cb1602f4273c5d2c65f69e1c7d6f",
   "patch": "muse_attention_decode/hypotheses/done/neutral/paired-heads-launch-cleanup-ab-tune-batch-gate.patch",
   "md": "muse_attention_decode/hypotheses/done/neutral/2026-10-24T104027Z-paired-heads-launch-cleanup-ab-tune-batch-gate.md",
   "change": "From the parent's branch (paired-heads-launch-cleanup built on master), change only the pairing predicate to paired = (window == 0 and hkv % 2 == 0 and batch > 1), where batch = cfg['batch'], and use this single predicate both for the pre-attention kernel's q layout (pass it as a constexpr, e.g. PAI"
  },
  {
   "slug": "sliding-fused-triton-decode1-gate",
   "area": "sliding-layer attention (batch-1 decode gate)",
   "outcome": "neutral",
   "speedup": 1.2930643047593426,
   "promotion_commit": "b8a0e4db5c9dfb55e627e0feb577fbd51ca8959d",
   "patch": "muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-gate.patch",
   "md": "muse_attention_decode/hypotheses/done/neutral/2026-10-24T120440Z-sliding-fused-triton-decode1-gate.md",
   "change": "Port _sliding_fused_kernel and its helpers (_counters, _num_splits, _norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _sliding_forward) verbatim from muse_attention_decode/hypotheses/done/regression/sliding-single-launch-fused-attention.patch (impl commit db5fa4d87590) onto current mast"
  },
  {
   "slug": "full-decode-cuda-gather-splitkv",
   "area": "FA3 call (full-layer decode gather)",
   "outcome": "gain",
   "speedup": 1.3467624733171553,
   "promotion_commit": "3693fb89b6ce898de437c29d4859b90c33277309",
   "patch": "muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.patch",
   "md": "muse_attention_decode/hypotheses/done/gain/2026-10-24T140133Z-full-decode-cuda-gather-splitkv.md",
   "change": "Stage 1 of a rewrite of full-attention decode, built on master (sliding-fused-triton-decode1-gate). Gate (config/shape only): cfg window == 0 and q_len == 1 and batch > 1 and num_q_heads == 16 and num_kv_heads == 2 and head_dim == 64 and page_size == 1; every other shape (all sliding, full decode_1,"
  }
 ]
}
5. muse_attention_decode/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_attention_decode/baseline.py — SGLang's own pipeline (the named parts), muse_attention_decode/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): a3-batched-q-verify, a3-drop-cu-seqlens-decode, a3-num-splits-full-decode, a3-num-splits-full-verify, a3-num-splits-sliding, a3-scheduler-metadata-full-verify, a3-single-split-sliding, glang-path-parity, liding-cluster-s2-decode-c4-pdl, liding-decode-paired-kvhead-rows-control-unpaired-s8, liding-verify-cuda-streamk-window, plitkv-l2-128b-prefetch-hint, riton-flash-decode-deep-pipeline, riton-flash-decode-full-both-heads, riton-flash-decode-full-verify, riton-window-attention-sliding, tack-streamk-ticket-cluster-c16-c4, tream-fork-preattention, ull-decode-splitkv-s2-streamk-ticket, ull-decode-splitkv-s3-fused-preattn, ull-verify-wgmma-s3-warpgroup-per-kvhead, ull-verify-wgmma-s4-multitile-fix, ull-verify-wgmma-s4-sliding-window, ull-verify1-cuda-streamk-batch1, ull-verify1-wgmma-s4-batch1-grid-sm, used-full-combine-normgate, used-sliding-normgate-epilogue, used-sliding-preattn-into-window-kernel

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
only for a rules problem or a design that cannot reach its floor. Ambition is not a defect: the
user keeps 40% of the day's checks for rewrites, so a large overhaul is judged on whether its
target is physically reachable and its plan sound, not on how much it changes. Companions are rarely useful for
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
