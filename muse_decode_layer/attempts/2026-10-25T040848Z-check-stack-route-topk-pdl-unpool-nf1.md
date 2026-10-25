# muse_decode_layer · 2026-10-25T040848Z · check · stack-route-topk-pdl-unpool-nf1

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T035400Z-stack-route-topk-pdl-unpool-nf1.md` (branch `hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1`).

- **Job:** `87211e1362b64c4c` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `2fe4d35`

## Goal
Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim restacks dl-route-warp-topk-pdl onto master from its patch (depends_on). (A) dl-route-warp-topk-pdl as is: per-warp top-8 in dl_route_norm_align_kernel (no block barriers), PDL between P (dl_pooler_router_kernel launch_dependents) and R (cudaLaunchKernelEx, norm_w/bias loaded before griddepcontrol.wait), binary-search pad loop. Run b471a950: R 7.01 -> 5.79 us at T=1, sliding decode_1 -3.4 us and full decode_1 -1.8 vs BEST. (B) From boundaries-unpooler-pdl-small-t (muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.patch), take ONLY dl_unpool_kernel at NF=1 (T<=8: rows 0..1023 = UNPOOL_SLICE, 16 tiles x SPLIT 8 = 128 CTAs, zero columns 1024..8191 written by the CTAs after griddepcontrol.wait, PDL launch, weight ring before the wait), replacing _gemm(expert_output, unpooler[:1024]) + F.pad; it ran 6.5 vs 9.2 us. Do NOT take dl_boundary_kernel: it lost its own timing (3.91/4.09/4.20 vs SGLang FusedPostNormResidualPreNorm 2.79/2.92/3.07 us) and was shipped anyway; both boundaries stay on master SGLang modules, and T=9..32 (NF=2/4) keep master _gemm(expert_output, unpooler). Since the unpool kernel no longer feeds a PDL boundary, it launches with PDL after Marlin/combine and the following SGLang boundary launches normally. THE TWO EDIT THE SAME _DL_CUDA TU (dense-chain module) and both rename the module, so the restack patch will likely conflict or the unpooler half must be ported by hand: the implementer hand-merges (B) into the restacked (A), uses one new module name, and keeps a single state flag per piece (state["dl"] route as in A; a new state["dlu1_ok"] for the NF=1 unpool). Self-check: the existing _dl_self_check cases (routing, near-tie, no-local-slot, align, Marlin y, graph replay) plus the unpool cases at T=1,4,8 (ffn_output vs master unpooler+pad, rel L2 <= 2e-3, zero columns exactly 0) and a captured graph replay of P -> R -> experts -> unpool vs eager; failure turns off only the failing piece. Print PATH CHOSEN for route and unpool per T, and CUDA-event times of R (pdl 0/1) and unpool NF=1 vs master. T>32 master byte for byte. First measurement is an okc check.

## What changed, and where
- Worktree state: claim had already restacked dl-route-warp-topk-pdl.patch (staged, 390+/22-). Applied boundaries-unpooler-pdl-small-t.patch on top with `git apply -3`; clean, no hand-port (separate _DLB_CUDA TU, as the review said; the review wins over the spec's "same TU" warning).
- kernel_fn: both boundaries are SGLang again (attn_boundary / ffn_boundary; no _dlb_attn/_dlb_ffn/use_dlb early return). `_dlb_unpool` (dl_unpool_kernel, PDL launch) runs only when `tokens <= SLICED_UNPOOL_MAX_TOKENS` (8) and `dlb["unpool"][1]`; T 9..32 and T>32 stay on master _gemm, T>32 untouched.
- `_dlb_setup` rewritten unpool-only: per-piece flag `dlb["unpool"][1]` (route keeps state["dl"]); self-check = unpool at T=1,4,8 (poisoned output, twice, rel L2 <= 2e-3, zero columns exactly 0), then chain graph replay (SGLang boundary -> P -> R -> experts -> unpool -> SGLang boundary) vs eager vs master at T=1,8,16,31,32. Failure turns off only the unpool (dl route untouched).
- Timing selectors assert the winner and print `WINNER = ...` / `PATH CHOSEN = ...`: unpool NF=1 custom vs master slice+pad (memset-subtracted, summed T=1+8) must win else master; R selector (`_dl_time_r` now returns a winner): new R at the chosen pdl vs master R (variant 2, no PDL) on summed P->R graph pairs T=1/8/32 must win, else `dl["r_variant"]=2` and `_dl_forward` launches master's R.
- Dead code left: dl_boundary kernel still compiles in the _DLB TU (adds compile time) and the _dlb_attn/_dlb_ffn/_dlb_coeffs/_dlb_pick_round helpers are unused; boundary build/self-check/timing is no longer run.
- Not run (no GPU on this node).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn has no sync/.item(); selectors run in prepare() only (checked diff).
- [x] Nothing derived from `data` is cached: dlb holds only weight-derived/ws buffers; r_variant is a prepare-time choice.
- [x] Weights: unchanged (unpooler used in place; no repack added).
- [x] Activations BF16/FP8 as before: n/a, no precision change; boundaries are SGLang.
- [x] Attention untouched: n/a.
- [x] No timing patches/harness/module edits/subprocess/fork (grep clean).
- [x] All work inside the timed call; no output caching.
- lint: `OK with warnings` (only host-sync warnings in prepare()); py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Small gain near one check resolution; read the decode_1 rows (err ~0.6 us) and the R / dl_unpool_kernel profile rows, not the headline. A submit-grade A/B would be needed to land it.

## Result (after the run)
- job `87211e1362b64c4c` (check), status done, passed True, tests 13/13
- speedup 1.2671x, 95% CI [1.2481, 1.2861], score 17851.61371363095 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2680313318618024, "ci": [1.2490108618838753, 1.2870518018397292], "job": "c243a29f26dc4725"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-route-topk-pdl-unpool-nf1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-route-topk-pdl-unpool-nf1.patch`; branch `hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1`
