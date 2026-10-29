You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_bf16_proj/hypotheses/queue/2026-10-29T150920Z-b12mc-multicast-x-pooler-qkv-m8-readflush-diag.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_bf16_proj/hypotheses/CLAIMS.md — the curated claims, one line per finding, tagged [landed] [lead] [shelf] [unresolved]
   [node-only] [refuted] [superseded]; the newest line per mechanism is the current one. Claims that share a
   distinctive word with this hypothesis (a pre-filter; read the whole file anyway):
   - muse_bf16_proj/hypotheses/CLAIMS.md:50 (b12x): - [refuted] b12x at m>=8: x streamed with each 12-bit chunk (per-row TMA copies, then L2 hot-spotting) | rows: load=verify_1, load=decode_16, load=decode_32 | evidence: 22ddaf4a, 0acdca48, 2980f57a; rotation took qkv m=8 only to gate 0.983 (DISTILL:179; LOG:529-560); many small cp.async.bulk copies cost 30-50 ns each, serialized (DISTILL:263) | board: 1.2664 | blob: - | reopen: x read once per cluster (TMA multicast over 2-4 CTAs, one box per chunk) and a read-flush gate at <= 0.95 | 2026-10-26 campaign
3. muse_bf16_proj/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
4. The active campaign's findings: campaigns/2026-10-26-expert-bandwidth/FINDINGS.md, campaigns/2026-10-26-expert-bandwidth/DISTILL.md, and the top 150 lines of campaigns/2026-10-26-expert-bandwidth/LOG.md (the Now block). Its landings, kills and H100 transfer readings are not in
   churn's own files (most of master's recent submission.py commits are campaign landings).
5. muse_bf16_proj/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI). The hypothesis's freshness:
   mint 2026-10-29T150920Z on board 1.2794, blob a36f3392; BEST now 1.2794 (job 3e3ffc5afc57433b), master blob a36f3392; moved 0.00% against a 1.24% margin; freshness gate: fresh; value case: EV 1.05%.
6. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_bf16_proj",
 "landed": [
  {
   "slug": "sglang-gemm-dispatch-parity",
   "area": "bf16 GEMM dispatch",
   "outcome": "gain",
   "speedup": 1.0022291437677175,
   "promotion_commit": "965b04691a75690b8c34d2cffcab24da88f291e6",
   "patch": "muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.patch",
   "md": "muse_bf16_proj/hypotheses/done/gain/2026-10-23T221829Z-sglang-gemm-dispatch-parity.md",
   "change": "Replace the starter's plain x @ W[:n].T with the exact dispatch SGLang's kernel uses: sglang.srt.layers.quantization.unquant._bf16_gemm_dispatch_impl(x, W[:n], None) for the four bf16 layers (qkv_gate, o, pooler, unpooler), and for the fp32 router keep F.linear(x.float(), router) at m <= 8 but build"
  },
  {
   "slug": "router-hilo-at-m8",
   "area": "router GEMM",
   "outcome": "gain",
   "speedup": 1.0218879905574376,
   "promotion_commit": "f9dc05da5a1e98751f4fbb5a8ebe1b246a4b46b4",
   "patch": "muse_bf16_proj/hypotheses/done/gain/router-hilo-at-m8.patch",
   "md": "muse_bf16_proj/hypotheses/done/gain/2026-10-23T225128Z-router-hilo-at-m8.md",
   "change": "In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX_TOKENS from 8 to 1, so m=8 (and the m=7/m=9 tests) take the two torch.mm(x, router_hi_t/router_lo_t, out_dtype=torch.float32) GEMMs that th"
  },
  {
   "slug": "router-triton-hilo-fused-control-splitk-all-m",
   "area": "router GEMM",
   "outcome": "gain",
   "speedup": 1.1510295903072447,
   "promotion_commit": "dba4ffd5945248c0299bb6953ab4b561d0b3eeeb",
   "patch": "muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.patch",
   "md": "muse_bf16_proj/hypotheses/done/gain/2026-10-23T230544Z-router-triton-hilo-fused-control-splitk-all-m.md",
   "change": "Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile, tl.trans(lo_tile"
  },
  {
   "slug": "bf16-shape-gated-triton-winners-combo",
   "area": "bf16 GEMM dispatch (shape-gated Triton routing, m<=16)",
   "outcome": "neutral",
   "speedup": 1.1760470299345995,
   "promotion_commit": "658d63bcdf2a330b0c3093873d5f04a9ee92ff9f",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/bf16-shape-gated-triton-winners-combo.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-24T112147Z-bf16-shape-gated-triton-winners-combo.md",
   "change": "Built on master (dba4ffd: parity dispatch + 3-D split-K Triton router; router untouched). Combine, behind one shape gate in kernel_fn, the four individually-neutral Triton kernels ONLY on the shapes where each one's own kernel beat cuBLAS in its own run (paired ratio vs SGLang > 1 and kernel time in"
  },
  {
   "slug": "bf16-m32-swapab-wgmma-triton",
   "area": "bf16 GEMM dispatch (m=17..32, swap-AB Triton)",
   "outcome": "neutral",
   "speedup": 1.1984474737274622,
   "promotion_commit": "8b9ad430b2ed965f732443cb8de2069b58f43366",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-24T140147Z-bf16-m32-swapab-wgmma-triton.md",
   "change": "Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-AB Triton kernel, _bf16_swapab_kernel, and route the four bf16 layers to it at 17 <= cfg m <= 32 only (the scored m=32 rows of load decode_3"
  },
  {
   "slug": "bf16-restack-m16-swapab-router-swapab-m32",
   "area": "restack (m<=16 swap-AB GEMMs + router swap-AB)",
   "outcome": "neutral",
   "speedup": 1.2215006063849365,
   "promotion_commit": "b99b1dd3508b24bbdf847b51d2fb53de219ca957",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-24T174421Z-bf16-restack-m16-swapab-router-swapab-m32.md",
   "change": "Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16 kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3. Its patch is in "
  },
  {
   "slug": "restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128",
   "area": "shape-gated restack: router GEMM m<=8 + unpooler-slice GEMM m<=8 + pooler/unpooler GEMM m=65..128",
   "outcome": "neutral",
   "speedup": 1.236629518166488,
   "promotion_commit": "040fc7e18f4d9c17ea9fc4a4b9a4cae9216a2335",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-25T054102Z-restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.md",
   "change": "Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check 1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked alone on b99b1dd, each gated to exactly the (layer, m) cells it won; nothing else changes and no constant is retuned. Patches live in muse_bf1"
  },
  {
   "slug": "pooler-nn-layout-widen-m256",
   "area": "pooler GEMM (m=129..256, cuBLAS NN layout)",
   "outcome": "neutral",
   "speedup": 1.2393674362823563,
   "promotion_commit": "67b46a57e69d86ab79f476a19502b06e05d4a471",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/pooler-nn-layout-widen-m256.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-25T145403Z-pooler-nn-layout-widen-m256.md",
   "change": "Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at NN_GATED_MIN_M=65 <= cfg m <= NN_G"
  },
  {
   "slug": "combo-rowowner-reduce-free-m1-qkv-o-plus-naive",
   "area": "combination (naive stack + bulk row-owner widened to qkv_gate/o m=1)",
   "outcome": "neutral",
   "speedup": 1.2566457115954903,
   "promotion_commit": "36c26199f24a1b0d49cb3f63d518a67cfd388f04",
   "patch": "muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.patch",
   "md": "muse_bf16_proj/hypotheses/done/neutral/2026-10-25T161307Z-combo-rowowner-reduce-free-m1-qkv-o-plus-naive.md",
   "change": "Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS a conjectural widening of source A's bulk row-owner kernel to the other two m=1 weight streams that master runs with a split-K redu"
  },
  {
   "slug": "bulk12: m=1 row-owner kernel streaming a lossless 12-bit weight code (pooler 17.72 -> 14.8 us, ~4.8 TB/s raw-equivalent)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.2664,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_bf16_proj/hypotheses/CLAIMS.md:56",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=pooler load=decode_1, layer=o load=decode_1, layer=qkv_gate load=decode_1; evidence: check 474490c8258d4521 1.2699, scored 3834ceeca7c0410d 1.2664 (LOG:500-517); land 5325ea40"
  },
  {
   "slug": "read-flush gates for every bulk12 cell + pooler m=128 K-halves with fp32 partials (khalf b2: pooler m=128 22.05 -> 21.12 us)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.2735,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_bf16_proj/hypotheses/CLAIMS.md:59",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=pooler load=verify_16; evidence: check 670ad7c6b0da41ec 1.2726, scored bc147c5b87bc465d 1.2735 (LOG:939-942); land 8f233e88; pooler-k-halves-fp32-partials-restack de95f81782374124 (FINDINGS:216) is this line"
  },
  {
   "slug": "router zero-fill PDL overlap at m=9..256 (all four classes) + the pooled khalf gate rule (3 rounds, 2.0x pooled IQR)",
   "area": null,
   "outcome": "landed",
   "speedup": 1.274,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_bf16_proj/hypotheses/CLAIMS.md:62",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=router load=decode_16, layer=router load=decode_32, layer=router load=verify_16, layer=router load=verify_32, layer=pooler load=verify_16; evidence: a20f80486c5b4bc1 (router PDL x1.031-1.101 routed, FINDINGS:218), check 023f4260e86e423e 1.2723, scored 254c83ae55834807 1.2740 (LOG:1283-13"
  },
  {
   "slug": "qkv_gate m=128 cuBLASLt plan ranked and gated under the judge's read flush",
   "area": null,
   "outcome": "landed",
   "speedup": 1.2794,
   "promotion_commit": null,
   "patch": null,
   "md": "muse_bf16_proj/hypotheses/CLAIMS.md:65",
   "source": "campaign",
   "claim_now": "landed",
   "change": "rows: layer=qkv_gate load=verify_16; evidence: check 0c9095e2 1.2844, scored 3e3ffc5afc57433b 1.2794 (LOG:2011-2012, 2034); land 62da2d9e"
  }
 ]
}
7. muse_bf16_proj/submission.py (master) — the kernel the hypothesis will be built on (its base).
8. muse_bf16_proj/baseline.py — SGLang's own pipeline (the named parts), muse_bf16_proj/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
9. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): 12-qkv-m1-r32-one-cta-per-sm-readflush-gate, 12mc-multicast-x-pooler-qkv-m8-readflush-diag, half-gate-three-rounds-pooled, half-gate-three-rounds-pooled-control-gate-stats-pooled-iqr, ombo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16, ooler-khalf-lt-batched-plan-m128-m256, ooler-khalf-m128-fp32-partials-triton-castadd, outer-hilo-m128-m256-readflush-config-sweep, t-config-sweep-m256-beyond-heuristic, t-nn-layout-cells-o-m128-qkv-m128

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
