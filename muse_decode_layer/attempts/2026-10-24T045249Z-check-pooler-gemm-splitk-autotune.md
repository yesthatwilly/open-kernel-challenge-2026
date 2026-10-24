# muse_decode_layer · 2026-10-24T045249Z · check · pooler-gemm-splitk-autotune

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-23T225545Z-pooler-gemm-splitk-autotune.md` (branch `hyp/muse_decode_layer/pooler-gemm-splitk-autotune`).

- **Job:** `2e1a43ab85324776` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `3212dc6`

## Goal
For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in prepare(): candidates are (a) SGLang's _bf16_gemm_dispatch_impl as today, (b) F.linear, (c) a 2-way and (d) a 4-way split-K done as torch.bmm over K-chunks with out_dtype=torch.float32 partials (x viewed as [S, T, 8192/S] against a prepared, lossless [S, 8192/S, 4096] transposed-chunk copy of the same bf16 weight) followed by one sum over S and a cast to bf16, so the GEMM output is bf16 exactly where SGLang keeps it and the only extra precision is fp32 partials. prepare() times each candidate at T in {1, 8, 16, 32, 128, 256} by capturing a CUDA graph of the candidate and replaying it (not eagerly; AGENTS.md warns eager winners lose under graphs), ~50 replays each with a cold-L2 flush, and records the winner per T; kernel_fn branches on cfg['tokens'] (a shape, allowed) to call the winner. The default stays (a) for any T not in the table. qkv, o and unpooler GEMMs are untouched. Attribution: whichever shapes change are those whose T picked a non-default form; prepare() prints the table so the log shows the choice.

## What changed, and where
All in the worktree's `muse_decode_layer/submission.py` (uncommitted; 136 +, 2 -):

- New `_pooler(x, state, tokens)` replaces `_gemm(ffn_input, dense["pooler"])` in kernel_fn (main stream, same spot, still overlapping the side-stream router). It branches on `cfg['tokens']` only via `state["pooler_form"].get(tokens, 0)`; form 0, the default for any T not in the table, is the unchanged `_bf16_gemm_dispatch_impl` call.
- `_pooler_split`: `x.view(T, S, 8192 // S).transpose(0, 1)` (strided batch view, no copy), `torch.bmm(..., out_dtype=torch.float32)` partials, `.sum(0)` in fp32, `.to(bfloat16)` once. Output is bf16 exactly where SGLang has it.
- Candidates: 0 = dispatch (a); 1/2 = 2-/4-way split-K, "nn" weight operand `[S, K/S, N]` = views of one contiguous `pooler.t()` copy (one buffer serves both S); 3/4 = 2-/4-way split-K, "tn" operand: contiguous `[S, N, K/S]` chunk copy used through `.transpose(1, 2)`. All are lossless relayouts of the same bf16 values.
- `_tune_pooler` (called at the end of prepare()): guards `torch.bmm(out_dtype=float32)` with a try on a small strided bf16 test (also checks dtype and allclose to fp32 bmm); on failure it prints a line and DROPS every split-K candidate (table empty, kernel_fn = base). For each T in {1, 8, 16, 32, 128, 256} on synthetic randn x (local seeded generator, global RNG untouched): eagerly checks each split form against dispatch (rel L2 <= 1e-2, else dropped), warms up 3x on a side stream, captures a CUDA graph per candidate (private pools), then 5 warmup + 50 timed rounds, interleaving candidates, each replay preceded by a 256 MB `flush.zero_()` cold-L2 flush and bracketed by CUDA events; medians in us. A split form wins only if the best one beats dispatch by >= 3% or >= 0.5 us; otherwise 0. It prints per T every candidate median, the pick, and both the dispatch and the pick medians, then the final table. Graphs, events, x, ref, flush, w_t and unused chunk copies are freed (sync, then empty_cache), and the current stream is restored in a `finally`. Only `pooler_form` (dict T -> int) and `pooler_chunks` (only the chunk tensors the table actually uses) persist in state.
- Deviations: (b) F.linear dropped. Per the review it resolves to the same cuBLAS TN kernel as `_bf16_gemm_dispatch_impl` / `x @ w.T` on H200, so it would just re-time (a). Added the review's open-question TN-layout variant (`[S, N, K/S]` + transpose) as candidates 3/4, beside the spec's NN layout, because layout can change the cuBLAS kernel family. The spec's "x viewed as [S, T, 8192/S]" is done as `x.view(T, S, K/S).transpose(0, 1)`, as required (a plain `x.view(S, T, ...)` would be wrong).
- Memory: at most +67 MB per persisted layout (NN shared across S; TN one copy per S picked). Transient during prepare: about 3 x 67 MB of chunk copies + 256 MB flush.
- Uncertainties: whether bmm with this strided x (batch stride K/S, ld K) goes to cuBLAS strided-batched without an internal copy (x is at most 4 MB, so even a copy is cheap and gets timed by the autotune anyway); whether the judge's torch has `bmm(out_dtype=)` (if not, the run is the base). Nothing was run: this node has no GPU.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/pooler-gemm-splitk-autotune diff`:

- [x] CUDA-graph capturable: kernel_fn's new path is a dict lookup on `cfg['tokens']` (a shape), a view, bmm, sum and cast. No `.item()`/`.cpu()`/`synchronize()`/prints/events in kernel_fn; every sync, event and print is in `_tune_pooler`, called only from prepare().
- [x] Nothing derived from `data` is cached across calls: the table and chunks come from synthetic randn inputs and the weights in prepare().
- [x] Weights: the pooler chunk copies are lossless bf16 relayouts (transpose/permute + contiguous) made in prepare(). BF16 stays BF16 and no other weight is touched.
- [x] Activations: the pooler output is bf16, as in SGLang. The split-K partials are fp32 (finer than cuBLAS's bf16 output) and are rounded to bf16 once. KV cache and queries are unchanged.
- [x] n/a: attention untouched (every key still attended; no top-k/sparse/skip).
- [x] No timing patches (torch.cuda.Event is only *used* in prepare(), never reassigned), no harness frames/I/O/secrets, no module edits, no reference import, no subprocess/network. The only new import is `statistics` (stdlib).
- [x] All work happens inside the timed call: the per-call pooler GEMM runs in kernel_fn. prepare() only chooses which GEMM form to run for a T and relays the weight; it caches no outputs.
- [x] First line `# okc: problem=muse_decode_layer` kept; no triton.autotune.

okc_lint output:

```
warn  dtype        L242: flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)  (allowed only where SGLang does the same)
warn  host-sync    L274: torch.cuda.synchronize(device)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L286: torch.cuda.synchronize(device)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L379: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

The dtype warn is the prepare-only uint8 L2-flush buffer, not an activation. py_compile passed and the resulting __pycache__ was removed.

## Risks / what could fail
medium
Risk: the gain is bounded by ~7 us per shape on shapes of 260-700 us (1-3%), close to the +-1.5% CI, so the adversarial reviewer may ask for a control run with the autotune forced to (a) if the result is ambiguous; prepare() must print the chosen table so a flat result can be read as 'autotune picked (a) everywhere' versus 'split-K did not help'. The weight chunk copy is a lossless relayout of the same bf16 values (allowed repacking); it doubles pooler weight memory (134 MB), fine. prepare() timing with graph capture/replay for 4 candidates x 6 T values must stay well under the 300 s budget (it is seconds). The fp32-partials-then-sum path must not be mistaken for a precision reduction: it is strictly finer than cuBLAS's bf16 output rounding. Graph-safe: the branch is on T only.

## Result (after the run)
- job `2e1a43ab85324776` (check), status done, passed True, tests 13/13
- speedup 1.0189x, 95% CI [1.0036, 1.0342], score 22200.45481505138 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.020038644953596, "ci": [1.004738065279292, 1.0353392246278998], "job": "5755a412dda244a1"}; base result: muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/pooler-gemm-splitk-autotune.result.json`, `muse_decode_layer/hypotheses/done/neutral/pooler-gemm-splitk-autotune.patch`; branch `hyp/muse_decode_layer/pooler-gemm-splitk-autotune`
