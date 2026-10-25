# muse_decode_layer · 2026-10-25T011431Z · check · dl-route-warp-topk-pdl

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T224452Z-dl-route-warp-topk-pdl.md` (branch `hyp/muse_decode_layer/dl-route-warp-topk-pdl`).

- **Job:** `b471a950805c4053` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `91e7a28`

## Goal
Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32 (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes, nothing else: (a) top-8 without block barriers: replace the 8-round loop (warp_max_u64 + s_kred + 2 __syncthreads per round) with a per-warp top-8: each warp runs 8 rounds of warp_max_u64 over its 32 keys with a lane-local taken flag (no block sync), lane r writes its round-r winner to s_cand[warp*8+r]; one __syncthreads; warp 0 then picks the global top-8 of the 64 candidates (each lane holds 2, 8 rounds of warp_max_u64) and writes s_sel[0..7]; one __syncthreads. route_key, tie order (lower id wins), weights s/(sum+1e-15), topk_ids, topk_w and local_ids writes unchanged. (b) PDL between P and R: in dl_pooler_router_kernel add griddepcontrol.launch_dependents after the main K loop of every CTA (before the split-K partial store), and launch R with cudaLaunchKernelEx and the programmatic serialization attribute when dl["pdl"] is 1; in R, load norm_w (its 16 values per thread) and bias[e] into registers BEFORE griddepcontrol.wait and read ei/logits only after it. Also hoist the pad loop: compute each padding slot from s_seg0/s_seg1/s_count by a binary search over 32 experts instead of the linear scan. Self-check: the existing _dl_self_check cases (routing, near-tie, no-local-slot, align structure, Marlin y, graph replay) must pass unchanged; print R CUDA-event time with pdl 0/1 for T 1, 8, 32.

## What changed, and where
All edits in the worktree's muse_decode_layer/submission.py (`_DL_CUDA`, `_build_dense_chain`, `_dl_route*`, `_dl_self_check`, `_dl_setup`). Not compiled or run (no GPU/nvcc on this node).

- Kernel P: `griddep_launch()` after the main K loop of every CTA, before the split-K partial store. Always executed; harmless without the attribute.
- Kernel R: master's kernel kept verbatim as `dl_route_norm_align_v1_kernel` (variant 2, prepare() timing and comparison only; kernel_fn never launches it). New `dl_route_norm_align_kernel<BSEARCH>`: per-warp top-8 (lane-local taken flag, lane r keeps round-r winner in s_cand[warp*8+r]), one barrier, warp 0 takes the top-8 of 64 candidates (2 per lane), one barrier. route_key, score, tie order, weights, topk_ids/topk_w/local_ids writes, NaN handling (same keys, same `taken ? 0 : key` rule) unchanged.
- Review fix 4: before `griddepcontrol.wait` only norm_w (16 floats/thread as 4 float4) and bias[e] are read; logits, ei and local_start come after it.
- PDL launch of R via `cudaLaunchKernelEx` + programmatic serialization when `pdl` (a prepare() decision, `dl["pdl"]`); the wrapper takes `pdl` and `variant` (0 new/binary-search pad, 1 new/linear pad, 2 v1; v1 is never launched with the attribute).
- Binary-search pad (5 steps over s_seg0) for padding slots and block_experts; the linear scan stays as BSEARCH=false for the A/B.
- Review fix 1-3 (prepare() prints, CUDA-event, 20 launches per graph): `R T=1/8/32 bsm=.. old(v1) / new / new-linear-pad`, and `P->R pair T=1/8/32 new pdl=0 / new pdl=1 / old-R pdl=0` (10 pairs per graph). bsm is printed.
- Review fix 6: the self-check adds a bit-exact comparison of both new variants against v1 (xn, topk_ids, topk_w, local_ids, sorted_ids, block_experts, n_padded) at T=1/8/32; a mismatch fails the self-check, so the run falls to master's path.
- Review fix 7: `-lineinfo` added; self-check prints before and sync+print after P, R (via `_dl_forward_traced`, self-check only), R alone and each R variant. Module name unchanged. No sm_90a-only PTX beyond griddepcontrol (existing -arch=sm_90a), so no TORCH_CUDA_ARCH_LIST change.
- Review fix 5: DL_MAX_TOKENS and every bound constant untouched. The review wins over the spec on: keeping the old R for timing, the P->R graph A/B, and norm_w/bias-only pre-wait reads (same as the spec's intent). No new mbarrier/spin waits.
- Deviation: trace syncs live in `_dl_forward_traced` (not `_dl_forward`) because okc_lint flagged syncs in a function reachable from kernel_fn.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path is `_dl_forward` (P launch + R launch, no sync/.item/.cpu); verified in `git diff` that syncs only appear in prepare()-only functions.
- [x] Nothing derived from `data` is cached across calls (only prepare constants norm_w/bias/ws/cnt).
- [x] Weights: no change to weight repacking.
- [x] Activations: BF16 xn as before, same arithmetic as v1; KV/query paths untouched.
- [x] Attention: n/a: not touched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (no subprocess/fork anywhere in the diff).
- [x] All work inside the timed call; no output caching or replay (v1 R and timing run only in prepare()).
- [x] Gate: DL_MAX_TOKENS unchanged; build/self-check failure falls back to master, never raises.

okc_lint: `OK with warnings` (only prepare()-scope host-sync warnings); py_compile ok; __pycache__ removed.

## Risks / what could fail
medium
Medium: a wrong tie order fails routing checks; the near-tie self-check case covers it. The profiler row for R will include its PDL wait when pdl=1, so judge on the printed CUDA-event times and the shape totals.

## Result (after the run)
- job `b471a950805c4053` (check), status done, passed True, tests 13/13
- speedup 1.2599x, 95% CI [1.2410, 1.2788], score 17954.09489968111 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/dl-route-warp-topk-pdl.result.json`, `muse_decode_layer/hypotheses/done/neutral/dl-route-warp-topk-pdl.patch`; branch `hyp/muse_decode_layer/dl-route-warp-topk-pdl`
