# muse_decode_layer · 2026-10-24T231259Z · check · l2-prefetch-o-pooler-during-sliding-attention

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T184430Z-l2-prefetch-o-pooler-during-sliding-attention.md` (branch `hyp/muse_decode_layer/l2-prefetch-o-pooler-during-sliding-attention`).

- **Job:** `96ee63e15edb438c` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `9ba13fe`

## Goal
On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.e. sliding decode_16, decode_32 and verify_1), overlap an L2 warm-up of the two dense weights the main stream reads right after attention with the attention phase itself. On these shapes the attention phase (FusedQKNorm .. FusedRMSNormGate, ~25-30 us) moves little DRAM traffic (a 2048-token FP8 window, <= 8 MB), while the o GEMM and the pooler after it run near the HBM limit (pooler 67 MB in 20 us at T=16). New Triton kernel _l2_warm_kernel(ptr, n_words, sink_ptr, WORDS_PER_PROG, BLOCK: tl.constexpr = 4096): ptr is the weight viewed as int32 words; program p walks its contiguous WORDS_PER_PROG range in BLOCK steps with tl.load(ptr + offs, mask=offs < n_words, other=0, eviction_policy="evict_last"), XOR-accumulates into a register vector, and at the end stores the XOR-reduced scalar to sink_ptr + p only under the mask (acc == 0x7F3A5C1D) so the loads stay live; grid 32 programs (a quarter of the SMs, so attention keeps the rest). prepare(): state["prefetch_stream"] = torch.cuda.Stream(device); state["l2_sink"] = torch.zeros(64, dtype=torch.int32) on device (never read); flat int32 views of dense["o"] (16.8 MB, whole) and of dense["pooler"][:1536] (rows 0..1535 = 25.2 MB), ~42 MB total, under the 50 MB L2. kernel_fn on the gated shapes: right after qkv = _gemm(...) (so the warm-up does not compete with the qkv weight stream), pf = state["prefetch_stream"]; pf.wait_stream(main); with torch.cuda.stream(pf): launch _l2_warm_kernel on the o view, then on the pooler slice; master attention continues on main unchanged. Join with main.wait_stream(pf) immediately before the return (the warm-up is long finished by then; this only closes the fork for graph capture). Only constant prepare-time weights are read; nothing derived from inputs is read or kept. Every other shape: master byte for byte.

## What changed, and where
Edited only the worktree's muse_decode_layer/submission.py (+43 lines, no deletions).
- New `_l2_warm_kernel(ptr, n_words, sink_ptr, WORDS_PER_PROG, BLOCK)` as specified (evict_last int32 loads, XOR fold, sink store only under mask acc == 0x7F3A5C1D). Constants L2_WARM_* near ROPE_POSITIONS: gate 2 <= T <= 32, 32 programs, BLOCK 4096, 1536 pooler rows.
- prepare(), after _dl_setup: state["prefetch_stream"], state["l2_sink"] (int32, device), state["l2_warm"] = ((int32 view, n_words) for dense["o"], pooler slice). Prints which buffer is warmed.
- RETARGET (required by review): pooler slice is `state["dl"]["w_pr"][:1536]` when the dense chain is ON (what kernel P reads at T<=32), else `dense["pooler"][:1536]` (master's non-dl path, which reads dense["pooler"]). o: `dense["o"]` is read by master at every T, so unchanged. Decided at prepare() from state["dl"] (a prepare-time fact, not input-derived).
- kernel_fn: gate is config-only (window != 0 and 2 <= tokens <= 32); fork right after the qkv GEMM, join (main.wait_stream) just before return. Warm kernels write only the sink; outputs untouched. Other shapes: master path.
- Deviation: none beyond the retarget.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no host sync/alloc in kernel_fn; stream fork/join via wait_stream, same pattern as the existing side stream; Triton kernel precompiled-on-first-launch pattern same as _post_norm_combine_kernel (first launch happens in warmup/capture as for other Triton kernels).
- [x] Nothing derived from data cached: only constant weights read; sink is scratch.
- [x] Weights: no repack/precision change; int32 views alias existing bf16 buffers.
- [x] Activations/KV/queries: untouched (diff only adds warm-up).
- [x] Attention sees every key: attention code unchanged.
- [x] No timing patches, harness access, module edits, reference import, subprocess, network.
- [x] All work in the timed call; no output caching or replay (warm-up is extra loads only).
- [x] n/a: no triton.autotune used.

Lint (okc_lint.py): `OK with warnings` — 18 host-sync warnings, all pre-existing in prepare()/self-check helpers (outside kernel_fn), none from this change. py_compile OK, __pycache__ removed.

## Risks / what could fail
high
Bold probe of a lever nothing has tried: each timed replay starts with a cold L2, and on sliding shapes the attention phase leaves HBM nearly idle for ~25 us. Counter-evidence (LESSONS): moe w2-l2-prefetch-decode1 cost 13.8 us and Marlin did not speed up, but that warm-up was serialized before Marlin and Marlin at bs=1 is not bandwidth-bound; here the warm-up is overlapped and the targets are cuBLAS GEMMs at 3.2-3.4 TB/s. Read the o and pooler nvjet rows and the 2 x warm rows; if the GEMM rows do not move, the knob is dead. Overlap: sliding-fused-batched-decode-port shortens the decode_16/32 attention window (less time to hide 42 MB) and small-t-launch-minimal-dense-chain-pdl (in judge) replaces the pooler kernel at T<=32 when ON (then only the o part applies). All three are separate checks on master, so nothing is confounded in the judge today.

## Result (after the run)
- job `96ee63e15edb438c` (check), status done, passed True, tests 13/13
- speedup 1.1670x, 95% CI [1.1495, 1.1845], score 19382.948327269925 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/l2-prefetch-o-pooler-during-sliding-attention.result.json`, `muse_decode_layer/hypotheses/done/regression/l2-prefetch-o-pooler-during-sliding-attention.patch`; branch `hyp/muse_decode_layer/l2-prefetch-o-pooler-during-sliding-attention`
