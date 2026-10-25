You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_prefill_layer/hypotheses/queue/2026-10-25T110612Z-experts-turn4-ssdq-s2-tma-raw-weight-ring.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
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
5. muse_prefill_layer/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_prefill_layer/baseline.py — SGLang's own pipeline (the named parts), muse_prefill_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): a3-batched-mode, arlin-w2-atomic-add-turn, ense-gemm-cublaslt-algo-pick-prefill-t, irst-turn-moe-dead-work-stack, orm-residual-boundaries-triton-row-kernel, outer-pooler-concat-gemm-turn, outer-triton-hilo-splitk-turn, ttn-full-prefill-cuda-fp8-wgmma-hard-gate, urn-align-two-phase-chunk-prefix, urn-moe-local-align-sorted-silu, xperts-cuda-wgmma-grouped-gemm-remeasure-sm90a, xperts-cuda-wgmma-s2-setmaxnreg-pingpong-first-turn, xperts-dequant-grouped-mm-cutlass, xperts-dequant-overlap-attention, xperts-dequant-threshold-turn4, xperts-dequant-triton-moe-16k, xperts-first-turn-ssdq-s2-large-m, xperts-marlin-block-m-48-turn4, xperts-triton-first-turn-per-gemm-config-pick, xperts-triton-turn4-persistent-local-align, xperts-turn-rega-s4-first-turn-n128, xperts-turn-rega-s4-port-t4096, xperts-turn-rega-s4-w2-only-t4096, xperts-turn1-cuda-wgmma-rega-port-bucket32, xperts-turn1-ssdq-s2-m64-tile, xperts-turn4-ssdq-s2-tma-raw-weight-ring

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
