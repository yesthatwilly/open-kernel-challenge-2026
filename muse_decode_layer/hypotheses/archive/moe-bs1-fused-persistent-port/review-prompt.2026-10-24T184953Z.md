You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_decode_layer/hypotheses/queue/2026-10-24T184355Z-moe-bs1-fused-persistent-port.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_decode_layer/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_decode_layer/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_decode_layer",
 "landed": [
  {
   "slug": "small-t-launch-minimal-dense-chain-pdl",
   "area": "whole-layer fusion",
   "outcome": "gain",
   "speedup": 1.228328123558382,
   "promotion_commit": "f1f2f35cf76890e924a0d0ae1576b1050ed9020f",
   "patch": "muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-24T172138Z-small-t-launch-minimal-dense-chain-pdl.md",
   "change": "WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.7 us, the dense GEMMs ~55 us against a 28.8 us weight floor (qkv 37.7 + o 16.8 + router 8.4 + pooler 67.1 + unpooler slice 8.4 MB), FA3 ~14 "
  },
  {
   "slug": "marlin-block-m-cap-16",
   "area": "w13 marlin",
   "outcome": "gain",
   "speedup": 1.0620874943175542,
   "promotion_commit": "ef7dea6cac43678c09ae91b20ece2c40e0670c4b",
   "patch": "muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-23T225545Z-marlin-block-m-cap-16.md",
   "change": "Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9, which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as chosen and clamps the"
  },
  {
   "slug": "fa3-hdim128-head-pairing-batched",
   "area": "FA3 call",
   "outcome": "neutral",
   "speedup": 1.0873503247934666,
   "promotion_commit": "09da1e0d71f2fca87bd222e2507f638ddb81d38e",
   "patch": "muse_decode_layer/hypotheses/done/neutral/fa3-hdim128-head-pairing-batched.patch",
   "md": "muse_decode_layer/hypotheses/done/neutral/2026-10-24T103909Z-fa3-hdim128-head-pairing-batched.md",
   "change": "Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attention layers with batch > 1. Gate, config only: paired = cfg[\"window\"] == 0 and cfg[\"batch\"] > 1 and cfg[\"num_kv_heads\"] % 2 == 0 (never a tenso"
  },
  {
   "slug": "sglang-pipeline-parity-router-hilo-m8",
   "area": "router GEMM",
   "outcome": "gain",
   "speedup": 1.021632132299037,
   "promotion_commit": "164c1d27648cffdd4d7f5a04f0d6f582e1f9ac56",
   "patch": "muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-23T225318Z-sglang-pipeline-parity-router-hilo-m8.md",
   "change": "Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules, the cos|sin table, the "
  },
  {
   "slug": "post-expert-norm-triton-small-t",
   "area": "post-expert norm+combine",
   "outcome": "gain",
   "speedup": 1.1635238875350442,
   "promotion_commit": "ea896254423db20bfb9c584010c4b8ca73228a23",
   "patch": "muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.patch",
   "md": "muse_decode_layer/hypotheses/done/gain/2026-10-24T140057Z-post-expert-norm-triton-small-t.md",
   "change": "Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch) into muse_decode_layer/submission.py _experts. Today _experts ends with SGLang's fused_post_expert_norm_weighted_sum(y, t"
  }
 ]
}
5. muse_decode_layer/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_decode_layer/baseline.py — SGLang's own pipeline (the named parts), muse_decode_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): 2-prefetch-o-pooler-during-sliding-attention, a3-full-single-split-probe, a3-num-splits-full-no-metadata, a3-num-splits-sliding, arlin-two-stream-halves-port, f16-m32-swapab-wgmma-port, liding-fused-batched-decode-port, oe-align-local-expert-count, oe-align-parallel-large-t-port, oe-align-single-kernel-small-t-control-capacity-256, oe-align-small-t-restack-on-master, oe-bs1-fused-persistent-port, oe-dispatch-prep-on-side-stream, oe-small-t-cuda-swapab-port, outer-triton-splitk-side-stream, re-expert-rmsnorm-triton, refork-qkv-o-swapab-t16-t32, riton-flash-decode-full-verify-paired-kv, rtllm-decode-full-attention, ull-verify-cuda-splitkv-stage2, used-preattention-paired-pad-restack

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


TODAY (2026-10-24 ~18:50Z) HARVEST, relevant to dependencies: moe cuda-nvfp4-swapab-mma-persistent-small-m FAILED its tests (8f198afc3bf14f6f); moe cuda-fused-moe-bs1-persistent-w13-silu-w2 neutral 1.307 within CI (1105a02aa3fa4e56); moe marlin-two-stream-expert-halves-small-m neutral 1.302 (5d630369b1634f77); decode moe-chain-cuda-swapab-expert-once-large-t FAILED tests; decode small-t-launch-minimal-dense-chain-pdl GAIN 1.228 and is now on master (the base). Any hypothesis that is a port of a failed or neutral run should be held unless its own case stands without the parent winning.
