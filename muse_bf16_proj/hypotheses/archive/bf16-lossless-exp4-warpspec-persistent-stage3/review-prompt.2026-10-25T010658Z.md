You are the adversarial reviewer for one kernel-optimization hypothesis in the Muse Spark Open
Kernel Challenge workspace at /home/user/git/musespark. You are a fresh pair of eyes; nobody has told you the
hypothesis is good. Your job is to find out whether running it once would produce a result that
can be interpreted, and if not, what extra run(s) would make it interpretable.

Read, in this order (all paths relative to /home/user/git/musespark; read-only, change nothing):
1. muse_bf16_proj/hypotheses/queue/2026-10-25T003715Z-bf16-lossless-exp4-warpspec-persistent-stage3.md — the hypothesis (frontmatter + body). The `change` paragraph is what will be built.
2. muse_bf16_proj/hypotheses/FINDINGS.md — learnings by area, where the current best spends its time, do-not-retry list.
3. muse_bf16_proj/hypotheses/BEST.json — the current best result on master's lineage (job, speedup, CI).
4. The hypotheses already landed on master's submission.py, oldest first, with their patches:
   {
 "ok": true,
 "problem": "muse_bf16_proj",
 "landed": [
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
  }
 ]
}
5. muse_bf16_proj/submission.py (master) — the kernel the hypothesis will be built on (its base).
6. muse_bf16_proj/baseline.py — SGLang's own pipeline (the named parts), muse_bf16_proj/TASK.md — shapes,
   weights, floors, correctness bounds, timing method, rules.
7. Other hypotheses in flight or queued for this problem (slugs; do not duplicate them): -swapab-m32-splitk2-two-ctas-per-sm, -unpooler-slice-cuda-cluster-dsmem-rewrite, f16-lossless-exp4-warpspec-persistent-stage3, kv-gate-o-m128-wgmma-splitk-semaphore, kv-swapab-m32-splitk8-stages3, npooler-full-m16-swapab-route, ooler-m32-tma-splitk-extend, ooler-skinny-tiled-weight-layout, outer-cuda-cluster-dsmem-splitk-rewrite, outer-hilo-addmm-epilogue-control-inplace-out, outer-triton-hilo-fused-control-hybrid-cublas-above-m32

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
