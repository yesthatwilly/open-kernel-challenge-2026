You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_prefill_layer/hypotheses/queue/2026-10-29T150952Z-experts-rega-stage-anatomy-diag-r2.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_prefill_layer/hypotheses/CLAIMS.md — the curated claims, one line per finding, tagged [landed] [lead] [shelf] [unresolved]
   [node-only] [refuted] [superseded]; the newest line per mechanism is the current one. Claims that share a
   distinctive word with this hypothesis (a pre-filter; read the whole file anyway):
   - muse_prefill_layer/hypotheses/CLAIMS.md:48 (81x): - [refuted] slab multicast across a 2-CTA cluster in rega (mc4 / mc2 / bm256): the ingest-cap premise is false (mc2 cut issued L2 bytes 24-39% and ran 1.31-1.81x slower) | rows: load=turn_1, load=turn_4, load=first_turn | evidence: 0ca60bb7, a6f60630 (DISTILL:185, 439); LOG:599, 645 | board: 1.4677 | blob: - | reopen: a stage anatomy (experts-rega-stage-anatomy-diag) shows ingest-bound stages | 2026-10-26 campaign
   - muse_prefill_layer/hypotheses/CLAIMS.md:56 (fa3c, w2st): - [landed] final near-ties pinned: fa3c contig-varlen at turn_4 / turn_1, w2st arm a at high / ft, the never-routed LO phase skipped (~40 s shorter prepare); deterministic final-rerun bytes | rows: load=turn_1, load=turn_4, load=first_turn | evidence: check 7db6fa1a 1.5055 (vs same-code checks 1.4986 / 1.4994), scored b2941ea23dc743a3 1.5008 (LOG:2039-2042, 2056-2070); land 350343bc | board: 1.5008 | blob: c63801e7 | reopen: - | 2026-10-28 campaign
   - muse_prefill_layer/hypotheses/CLAIMS.md:63 (anatomy, diag): - [shelf] prepare-only rega stage anatomy (experts-rega-stage-anatomy-diag), 0% headline by itself; it is the revival test for mc2 / bm256 | rows: all | evidence: DECISIONS:421; DISTILL:185, 431 | board: 1.5008 | blob: - | reopen: - | 2026-10-28 screen
3. muse_prefill_layer/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
4. The active campaign's findings: campaigns/2026-10-26-expert-bandwidth/FINDINGS.md, campaigns/2026-10-26-expert-bandwidth/DISTILL.md, and the top 150 lines of campaigns/2026-10-26-expert-bandwidth/LOG.md (the Now block). Its landings, kills and H100 transfer readings are not in
   churn's own files (most of master's recent submission.py commits are campaign landings).
5. muse_prefill_layer/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI). The hypothesis's freshness:
   mint 2026-10-29T150952Z on board 1.5008, blob c63801e7; BEST now 1.5008 (job b2941ea23dc743a3), master blob c63801e7; moved 0.00% against a 1.24% margin; freshness gate: fresh; value case: EV 0.6%.
6. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_prefill_layer",
 "landed": [
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
  },
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
   "slug": "experts-rega-w13-fast-silu-epilogue",
   "area": "silu-mul (rega w13 epilogue)",
   "outcome": "gain",
   "speedup": 1.4949064012876574,
   "promotion_commit": "9fbde72cc980ca534c8db9600bf6f101c82d5a29",
   "patch": "muse_prefill_layer/hypotheses/done/gain/experts-rega-w13-fast-silu-epilogue.patch",
   "md": "muse_prefill_layer/hypotheses/done/gain/2026-10-26T224003Z-experts-rega-w13-fast-silu-epilogue.md",
   "change": "Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a full-accura"
  },
  {
   "slug": "rega w2 early gs / sid loads (arm a) routed at the low class (pop / random x0.975; rega-lo guard x0.966-0.968)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.4897,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_prefill_layer/hypotheses/CLAIMS.md:54",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=turn_1; evidence: 59eb665f162a4ac9 1.4986, confirm 8716d98c405f417d 1.4994, scored 99767cf374bb4688 1.4897 (LOG:1266-1285; FINDINGS:237 is this line); land 95af1c5f"
  },
  {
   "slug": "final near-ties pinned: fa3c contig-varlen at turn_4 / turn_1, w2st arm a at high / ft, the never-routed LO phase skipped (~40 s shorter prepare); deterministic final-rerun bytes",
   "area": null,
   "outcome": "landed",
   "speedup": 1.5008,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_prefill_layer/hypotheses/CLAIMS.md:56",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: load=turn_1, load=turn_4, load=first_turn; evidence: check 7db6fa1a 1.5055 (vs same-code checks 1.4986 / 1.4994), scored b2941ea23dc743a3 1.5008 (LOG:2039-2042, 2056-2070); land 350343bc"
  }
 ]
}
7. muse_prefill_layer/submission.py (master) — the kernel the hypothesis will be built on (its base).
8. muse_prefill_layer/baseline.py — SGLang's own pipeline (the named parts), muse_prefill_layer/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
9. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): a3-num-splits-cap-gated-turn, outing-triton-hilo-sum-topk-localids, ront-qknorm-rope-fp8-store-fusion-restack, ront-routing-pdl-launch-diet-stack, ttn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload, xperts-rega-dynamic-item-scheduler, xperts-rega-epilogue-overlap-next-stage, xperts-rega-epilogue-wg-local-bulk-store, xperts-rega-fine-n-high-plus-fast-silu, xperts-rega-low-2cta-fullslab-mc2, xperts-rega-lpt-group-order, xperts-rega-mc2-bm256-stack-high-ft, xperts-rega-mc4-cluster-high-ft, xperts-rega-pdl-chain-regate-iqr-slack, xperts-rega-rotate-finebucket-fastsilu-combo, xperts-rega-rotation-fastsilu-combo, xperts-rega-stage-anatomy-diag, xperts-rega-stage-anatomy-diag-r2, xperts-rega-w2-epilogue-prefetch-bulk-store-cf

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
E. **Claims and freshness (a hold, not a note).** Hold when (i) the mechanism's newest claim is
   [refuted] or [superseded] and the hypothesis does not show that claim's `reopen` condition is met;
   (ii) a [landed] claim (or a campaign landing in step 4) already owns the rows it targets with a
   different mechanism, so it is built on a base that no longer exists; (iii) the board moved past the
   margin since `best_at_mint` / the last review (the freshness line in step 5) and the value case
   (`value_gain`, `expected`) was not re-derived on today's rows. A [shelf] mechanism proceeds only as part
   of a stack whose rows sum to >= 2.5% headline; an [unresolved] one only with the extra runs or pinned
   routes its `reopen` names; a [node-only] one only when the change is per-SM work (per-SM cycles transfer
   to the judge H200; HBM rates do not). Cite the CLAIMS.md line numbers you relied on.

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
 "freshness": "E: the claims (CLAIMS.md:N) and board/blob movement you checked, and what they decided",
 "companions": [
   {"slug": "<parent-slug>-ab-revert-<x>", "kind": "ab-revert" | "ab-tune" | "control",
    "area": "...", "change": "a paragraph implementable from the parent's branch without asking",
    "expected": [{"shapes": "<glob over shape keys>", "direction": "faster|slower|flat", "why": "..."}],
    "reason": "what this companion disambiguates", "priority": 1-5, "risk": "low|medium|high"}
 ],
 "open_questions": ["...", ...]
}
`hold` only when the hypothesis is unjustifiable, duplicates a landed/do-not-retry item, fails E, or
would break a rule. Prefer `proceed` with sharp notes over a timid `hold`. Put a one-line reason in
`summary` that starts with the E clause when E decided it (e.g. "E(i): CLAIMS.md:23 refuted ...").
