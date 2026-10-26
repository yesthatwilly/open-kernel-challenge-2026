You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_prefill_layer/hypotheses/queue/2026-10-25T211857Z-experts-rega-mc4-cluster-high-ft.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_prefill_layer/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_prefill_layer/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_prefill_layer",
 "landed": [
  {
   "slug": "stack-tonight-five-prefill",
   "area": "router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul",
   "outcome": "neutral",
   "speedup": 1.0705080897445196,
   "promotion_commit": "a81d19c4982ee781eb5faf78329a9ddc63e58206",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-24T233200Z-stack-tonight-five-prefill.md",
   "change": "Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted local-rows silu-mul on the Marlin path, prepare() self-checked), e"
  },
  {
   "slug": "sglang-pipeline-parity",
   "area": "whole-layer parity (SGLang pipeline) + routing remap",
   "outcome": "gain",
   "speedup": 1.0038664293875537,
   "promotion_commit": "6e616b8a059e028919a29923a63abca52d6f180d",
   "patch": "muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.patch",
   "md": "muse_prefill_layer/hypotheses/done/gain/2026-10-23T230250Z-sglang-pipeline-parity.md",
   "change": "Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler; FusedQKNorm with the [2^21, 128] fp32 cos|sin table and interleaved RoPE on sliding layers; k.to(FP8) a"
  },
  {
   "slug": "experts-turn4-ssdq-s1b-dataptr-symbol-fix",
   "area": "w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a dequant-producer warpgroup + SS wgmma grouped GEMM, stage 1b: rebuild with the undefined-symbol fix",
   "outcome": "neutral",
   "speedup": 1.1004181185941053,
   "promotion_commit": "ca1d41b1dc9f1cca7f8d13c3e0e5b2027055e6fa",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/experts-turn4-ssdq-s1b-dataptr-symbol-fix.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-25T054848Z-experts-turn4-ssdq-s1b-dataptr-symbol-fix.md",
   "change": "Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714), which NEVER RAN: the log reads '[ssdq] load_inline failed after 30.8s: ... muse_prefill_ssdq_v1.so: undefined symbol: _ZNK2at10TensorBase8data_ptrIxEEPT_v' then 'OFF: build failed (PATH CHOSEN = marlin at "
  },
  {
   "slug": "experts-turn-rega-s4-port-t4096",
   "area": "w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets",
   "outcome": "gain",
   "speedup": 1.3274587526558561,
   "promotion_commit": "23c1d5fa90e63b93b142d0f2461c4647a1e9d68c",
   "patch": "muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-port-t4096.patch",
   "md": "muse_prefill_layer/hypotheses/done/gain/2026-10-25T054847Z-experts-turn-rega-s4-port-t4096.md",
   "change": "Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_4 (64%/43%), against byte/math floors of 189 us and 306 us, so the two turn loads sit at 1.01x SGLang while first_turn is 1.19x. Every Trito"
  },
  {
   "slug": "experts-turn-rega-s4-first-turn-n128",
   "area": "w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM",
   "outcome": "gain",
   "speedup": 1.4060131381907617,
   "promotion_commit": "a1a5497d7a46aeb3c1aa4a18e70871a23c1b6be3",
   "patch": "muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-first-turn-n128.patch",
   "md": "muse_prefill_layer/hypotheses/done/gain/2026-10-25T054848Z-experts-turn-rega-s4-first-turn-n128.md",
   "change": "Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is the biggest absolute gap: the Triton engine runs 8.08-9.17 ms per layer (37-57% of every first_turn shape; 57% of the weight-138 sliding ct"
  },
  {
   "slug": "align-two-phase-all-t-rega",
   "area": "MoE align/dispatch on the rega path (200 < T <= 16384): _align_parallel replaced by the two-phase chunk-count + prefix-place align, including the first_turn class",
   "outcome": "neutral",
   "speedup": 1.4457544543391616,
   "promotion_commit": "1932a9c4fb3d8dc735bb1a513853637c0ada892c",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-25T125453Z-align-two-phase-all-t-rega.md",
   "change": "Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft master; patch muse_prefill_layer/hypotheses/done/neutral/turn-align-two-phase-chunk-prefix.patch) onto master a1a5497d, extended to the first_turn class. WHY NOW: the new master's profile (job 07a069bc69"
  },
  {
   "slug": "fa3-full-contiguous-kv-triton-gather-gated",
   "area": "FA3 call on full-attention layers (KV source: contiguous TMA copy of the paged page-size-1 rows, per-class timing gate)",
   "outcome": "neutral",
   "speedup": 1.4586709689454935,
   "promotion_commit": "020566aa96573ce235f3b70405c66115316f0fbf",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/fa3-full-contiguous-kv-triton-gather-gated.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-25T152235Z-fa3-full-contiguous-kv-triton-gather-gated.md",
   "change": "Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 changes. Sliding layers, the page-64 and kv-scaled cases, qkv GEMM, FusedQKNorm, the q/k/v FP8 casts, store_cache, o_norm_gate and everything "
  },
  {
   "slug": "combo-a-master-normgather-naive-stack",
   "area": "restack: pre-expert norm + rega gather fused (all rega classes) on master 1.4587; naive per-shape winner stack",
   "outcome": "neutral",
   "speedup": 1.4604901800994607,
   "promotion_commit": "33aa4083ab571529b6fa2ede89d066994e37370a",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/combo-a-master-normgather-naive-stack.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-25T161233Z-combo-a-master-normgather-naive-stack.md",
   "change": "COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py (master 020566aa / blob of muse_prefill_layer/submission.py wi"
  },
  {
   "slug": "experts-triton-w4a16-first-turn-blockt256",
   "area": "w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)",
   "outcome": "neutral",
   "speedup": 1.0228712349305686,
   "promotion_commit": "a1f93c294889a5676375f229f5a889a1419686fe",
   "patch": "muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.patch",
   "md": "muse_prefill_layer/hypotheses/done/neutral/2026-10-24T111900Z-experts-triton-w4a16-first-turn-blockt256.md",
   "change": "Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_CONFIGS in muse_prefill_layer/submission.py). One change: a wider token tile at first_turn only, so each in-register dequantized weight tile"
  },
  {
   "slug": "experts-hybrid-marlin-turn-triton-first-turn",
   "area": "w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)",
   "outcome": "gain",
   "speedup": 1.0599319720006266,
   "promotion_commit": "b92a0f47192b31b8618c3c53aab281b72204f3c4",
   "patch": "muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.patch",
   "md": "muse_prefill_layer/hypotheses/done/gain/2026-10-24T124504Z-experts-hybrid-marlin-turn-triton-first-turn.md",
   "change": "Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3 experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in muse_prefill_layer/submission.py change; kernel_fn is byte-identical between parity and master and stays so. (1) Restore the parity Marlin path v"
  }
 ]
}

5. muse_prefill_layer/submission.py — the kernel the hypothesis will be built on (its base).
6. muse_prefill_layer/baseline.py — SGLang's own pipeline (the named parts), muse_prefill_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): fa3-num-splits-cap-gated-turn, experts-rega-s4-bm256-wide-tile-high-ft-cf, experts-rega-mc2-bm256-stack-high-ft, experts-rega-low-2cta-fullslab-mc2, attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload

Answer these, adversarially:
A. **Justification.** Is the stated mechanism plausible on this hardware for these shapes? Does
   the profile data in FINDINGS support it (which kernels, how many µs, how far from the
   speed-of-light floor)? Are the `expected` shapes and directions consistent with the shape
   weights and floors in TASK.md? Name the weakest link in the argument. Check the `prior`
   field (the closest earlier attempt and what differs) against that slug's file and outcome,
   and FINDINGS' "Do not try" list: is the difference real, or is this the same mechanism again?
   A `prior: none` that has an obvious earlier attempt is grounds for hold.
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


Context for this review: an expiring-quota sprint (the user's 21:40Z steer). The run must be dispatched before ~23:45Z, so prefer proceed with sharp notes; companions only if essential. Return ONLY the JSON object.