# muse_moe_w4a16 · 2026-10-24T184142Z · submit · post-norm-combine-hoisted-slot-loads

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T183618Z-post-norm-combine-hoisted-slot-loads.md` (branch `hyp/muse_moe_w4a16/post-norm-combine-hoisted-slot-loads`).

- **Job:** `d7501477ce2748d5` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `fcb99a3`

## Goal
num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other kernel stay master byte for byte. Rewrite master's _post_norm_combine_kernel body so all local-slot row loads are issued before any reduction. Today each of the 8 slots sits under 'if eid >= 0:', so the compiler emits 8 serial (load ids -> branch -> load 8 KB row -> sum-of-squares) chains and the program pays ~8 DRAM round trips. New body: load the 8 ids at once (ids = tl.load(ids_ptr + t*TOP_K_C + tl.arange(0, TOP_K_C))) and the 8 weights g likewise; then in a tl.static_range(TOP_K_C) loop, extract eid_s and g_s with tl.sum(tl.where(tl.arange(0,TOP_K_C)==s, ids, 0)) (same for g), and load each row as v_s = tl.load(y_ptr + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no branch, so the 8 loads are independent; then acc += tl.where(eid_s >= 0, g_s, 0.0) * v_s * nw * tl.rsqrt(tl.sum((v_s*nw)^2)/H + eps). Remote slots are still never read (masked load), so the uninitialised y rows of remote slots are never touched and a fully masked row contributes exactly 0 (g forced to 0, so rsqrt(eps) of a zero row times 0 is 0, no NaN). fp32 math, eps, nw = w + 1 in fp32, output bf16 and the launch (grid (num_tokens,), BLOCK_H = 4096) are unchanged except num_warps 8 -> 16 so each thread holds 8 elements per row (64 live fp32 values for 8 rows). Keep the function name so the profile row stays comparable.

## What changed, and where
Edited only `_post_norm_combine_kernel` body and its launch `num_warps` 8 -> 16 in the worktree submission.py. Deviation from the md's per-slot static_range form, following the adversarial review: a single [TOP_K_C, BLOCK_H] masked 2D tile load (mask = (ids>=0)[:,None] & mask[None,:]) issues all 8 row loads textually before any reduction; then ss = tl.sum(v*v, axis=1), r = rsqrt(ss/H+eps), acc = tl.sum((where(valid,g,0)*r)[:,None]*v, axis=0). This also removes the 16 scalar-extraction reductions. Remote rows stay masked (never read); fully masked row gives coef 0 so no NaN. fp32 math, nw = w+1, eps, bf16 output, grid and BLOCK_H unchanged. Not run locally (no GPU).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff touches only a Triton kernel body and a num_warps constant; no host syncs or input-dependent control flow.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: untouched (prepare() not in diff).
- [x] Activations BF16 where SGLang keeps BF16: unchanged; fp32 internal math, bf16 output as before.
- [x] Attention: n/a, not touched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune added; first-line marker kept.

Lint (okc_lint.py): `OK with warnings` — 8 host-sync warnings, all in prepare() outside kernel_fn's call graph, pre-existing, none in the diff. py_compile clean, __pycache__ removed.

## Risks / what could fail
low
Judge on the _post_norm_combine_kernel row in top_kernels (Triton times reproduce within ~1%), not the headline: +0.8-1.0% geomean is under one run's resolution. Overlap: cuda-fused-moe-bs1-persistent-w13-silu-w2 (running) replaces this kernel at bs=1 and the queued cuda-fused-moe-midm-single-launch-routed replaces it at 2..32; cuda-nvfp4-swapab-mma-persistent-small-m and marlin-two-stream keep it, so a gain stacks on them. Same mechanism as the running cuda-dispatch-combine-pdl-verify combine (issue all slot loads first) but on disjoint shapes (<32 vs >32). Risk: register pressure at 16 warps is low (64 fp32 per thread); a correctness slip would show as a failed test, not a slow run.

## Result (after the run)
- job `d7501477ce2748d5` (submit), status done, passed True, tests 5/5
- speedup 1.3282x, 95% CI [1.3083, 1.3481], score 10125.377124588194 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/post-norm-combine-hoisted-slot-loads.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/post-norm-combine-hoisted-slot-loads.patch`; branch `hyp/muse_moe_w4a16/post-norm-combine-hoisted-slot-loads`
