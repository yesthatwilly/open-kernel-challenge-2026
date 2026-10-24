---
problem: muse_decode_layer
slug: pooler-gemm-splitk-autotune
created: 2026-10-23T225545Z
base: master
area: pooler GEMM
change: >
  For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight
  read whose floor is 14 us), pick the GEMM form per scored T in prepare(): candidates are (a)
  SGLang's _bf16_gemm_dispatch_impl as today, (b) F.linear, (c) a 2-way and (d) a 4-way split-K
  done as torch.bmm over K-chunks with out_dtype=torch.float32 partials (x viewed as [S, T,
  8192/S] against a prepared, lossless [S, 8192/S, 4096] transposed-chunk copy of the same bf16
  weight) followed by one sum over S and a cast to bf16, so the GEMM output is bf16 exactly where
  SGLang keeps it and the only extra precision is fp32 partials. prepare() times each candidate at
  T in {1, 8, 16, 32, 128, 256} by capturing a CUDA graph of the candidate and replaying it (not
  eagerly; AGENTS.md warns eager winners lose under graphs), ~50 replays each with a cold-L2
  flush, and records the winner per T; kernel_fn branches on cfg['tokens'] (a shape, allowed) to
  call the winner. The default stays (a) for any T not in the table. qkv, o and unpooler GEMMs are
  untouched. Attribution: whichever shapes change are those whose T picked a non-default form;
  prepare() prints the table so the log shows the choice.
expected:
  - shapes: "*load=decode_16*"
    direction: faster
    why: "m=16: the muse_bf16_proj review measured pooler/unpooler at ~1.5x their 14 us floor on one nvjet kernel; split-K partials over 4x more CTAs should close part of the ~7 us gap (~2-3% of the sliding shape)"
  - shapes: "*load=decode_32*"
    direction: faster
    why: m=32, same regime
  - shapes: "*load=decode_1*"
    direction: flat
    why: "m=1: cuBLAS gemv already near bandwidth; a split-K adds a sum launch on a launch-bound shape, so the autotune should keep (a) here; control"
  - shapes: "*load=verify_32*"
    direction: flat
    why: m=256 is compute-shaped and cuBLAS picks a proper tile; expect (a) to stay
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T044925Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T044925Z
base_sha: 952188da0ff9551ff95f13844604e502f412b70f
base_blob: 6884d26a928d1de1b07bf8d28e1be4e53dfdcfc5
branch: hyp/muse_decode_layer/pooler-gemm-splitk-autotune
worktree: .worktrees/muse_decode_layer/pooler-gemm-splitk-autotune
mode: null
implemented_at: 2026-10-24T045248Z
impl_commit: 3212dc6f2523fcc1167d7d7cde69207b1aa90e31
submission_sha256: 0083fdfae7be1d98338e77af5438f30aa94fd4b7139cf4156152b3a2bcb52acf
norm_sha256: a8ed9cbc2ee7518fa36313fed4f197106f5a5eb7a48d97b84cb33c40ddf44354
submission_blob: f0a682c9218ad58ca95c95bb33cc0dddc9f9fffc
lint_warnings: 4
---

# pooler-gemm-splitk-autotune

## Change

For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in prepare(): candidates are (a) SGLang's _bf16_gemm_dispatch_impl as today, (b) F.linear, (c) a 2-way and (d) a 4-way split-K done as torch.bmm over K-chunks with out_dtype=torch.float32 partials (x viewed as [S, T, 8192/S] against a prepared, lossless [S, 8192/S, 4096] transposed-chunk copy of the same bf16 weight) followed by one sum over S and a cast to bf16, so the GEMM output is bf16 exactly where SGLang keeps it and the only extra precision is fp32 partials. prepare() times each candidate at T in {1, 8, 16, 32, 128, 256} by capturing a CUDA graph of the candidate and replaying it (not eagerly; AGENTS.md warns eager winners lose under graphs), ~50 replays each with a cold-L2 flush, and records the winner per T; kernel_fn branches on cfg['tokens'] (a shape, allowed) to call the winner. The default stays (a) for any T not in the table. qkv, o and unpooler GEMMs are untouched. Attribution: whichever shapes change are those whose T picked a non-default form; prepare() prints the table so the log shows the choice.

## Expected

- `*load=decode_16*` faster: m=16: the muse_bf16_proj review measured pooler/unpooler at ~1.5x their 14 us floor on one nvjet kernel; split-K partials over 4x more CTAs should close part of the ~7 us gap (~2-3% of the sliding shape)
- `*load=decode_32*` faster: m=32, same regime
- `*load=decode_1*` flat: m=1: cuBLAS gemv already near bandwidth; a split-K adds a sum launch on a launch-bound shape, so the autotune should keep (a) here; control
- `*load=verify_32*` flat: m=256 is compute-shaped and cuBLAS picks a proper tile; expect (a) to stay

## Risk

Risk: the gain is bounded by ~7 us per shape on shapes of 260-700 us (1-3%), close to the +-1.5% CI, so the adversarial reviewer may ask for a control run with the autotune forced to (a) if the result is ambiguous; prepare() must print the chosen table so a flat result can be read as 'autotune picked (a) everywhere' versus 'split-K did not help'. The weight chunk copy is a lossless relayout of the same bf16 values (allowed repacking); it doubles pooler weight memory (134 MB), fine. prepare() timing with graph capture/replay for 4 candidates x 6 T values must stay well under the 300 s budget (it is seconds). The fp32-partials-then-sum path must not be mistaken for a precision reduction: it is strictly finer than cuBLAS's bf16 output rounding. Graph-safe: the branch is on T only.

## Adversarial review

Reviewed 2026-10-24T044925Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Runnable and interpretable from one check run because only the pooler path changes, the branch is on T only, and prepare() prints the per-T winner table; but the mechanism is weaker than stated (nvjet 128xT '4x1 splitK' already runs a 4-way split-K over ~128 CTAs, the practical H200 skinny-GEMM ceiling measured on bf16_proj is 3.3-3.6 TB/s, and the layer-level gain is capped by the side-stream router slack of ~5-7 us), so expect neutral-to-small, with the honest readout in the per-shape sliding decode_16/32 rows and the printed table rather than the headline.

### Justification
Profile support: in the best run (job 5755a412dda244a1) the pooler is nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 20.3 us at T=16 and 128x32 20.8 us at T=32, plus a cublasLt splitKreduce ~2 us, against a 14.0 us floor for the 67 MB weight stream (3.3-3.5 TB/s achieved), so the ~7 us gap per shape is real. But three things shrink it. (1) The '4x more CTAs' claim is not a difference from today: the nvjet kernel's name says splitK with a 4x1 layout, i.e. cuBLAS already splits K four ways and reduces in a second kernel; a torch.bmm over S=4 K-chunks with fp32 partials plus a torch.sum reproduces the same CTA count, the same weight traffic and the same reduce round-trip, just with cuBLAS's batched heuristic choosing the tile instead of its split-K heuristic. The only genuinely different candidate is S=2. (2) bf16_proj's pooler-triton-skinny-m32 (job 9f0ad00bbc874d31) measured a plain full-K stream at 20.2 us = 3.3 TB/s and nvjet at 3.4-3.6 TB/s; nothing yet has pushed this shape past ~3.6 TB/s on H200, so the realistic saving is 0-3 us, not 7. (3) In the layer, the pooler overlaps the router hi/lo pair (nvjet tss 64x16 splitK 9.3 us for 2 calls) + add + FusedMuseSparkRouting on the side stream, joined before the experts; the critical path is max(pooler ~22, side ~14-17), so even a perfect pooler can only recover the ~5-7 us slack, and extra CTAs contend with the side stream's kernels. Net: ~1-2% on sliding decode_16/32 (262-265 us), <1% on the full decode_16/32 shapes (440-615 us) which the 'faster' globs also cover but which will read flat regardless, and ~0.3% on the geomean. Shapes/directions vs TASK.md are otherwise consistent: T table {1,8,16,32,128,256} covers all six loads; decode_1 and verify_32 flat are right; the expected list omits verify_1 (T=8, same regime as T=16, should be faster-or-flat) and verify_16 (T=128, should be flat), both of which the table does tune. Candidate (b) F.linear is dead weight: bf16_proj's parity review established that _bf16_gemm_dispatch_impl resolves to the same cuBLAS TN kernel as x @ w.T on H200. Weakest link: the hypothesis assumes split-K adds parallelism cuBLAS lacks, when cuBLAS is already split-K here; the run really tests whether cuBLAS's batched-GEMM kernel choice for [T,2048]x[2048,4096] streams faster than its split-K choice for [T,8192]x[8192,4096], which is an empirical coin flip with a 2-3 us ceiling.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8 (router GEMM, landed): it moved T=8's side-stream router from the 26 us fp32 SIMT path (which overran the 21 us pooler, making the pooler irrelevant to verify_1's critical path) to the hi/lo bf16 pair (~15 us + routing, now hidden under the pooler with ~3-5 us slack). So any verify_1 gain from a faster pooler exists only because of that landed change, and is capped at that slack; at T=16/32 the hi/lo path is SGLang's own and unchanged by the landing, but the same max(pooler, side-branch) ceiling and SM contention with the side stream apply. This conditions the size of the effect, not its sign or attribution, and reverting it would only hide the pooler behind the slower router at T=8, so no A/B is needed.

### Interference and A/B needs
No companion is warranted. A gain is unambiguous: the only changed code path is the pooler GEMM at the tabled T values, the branch is shape-only, and prepare() prints which form each T picked, so a per-shape gain pairs directly with a non-default pick. A loss is attributable too: a loss with (c)/(d) chosen means the isolated-winner split-K is worse in situ (side-stream contention or the extra sum launch), which is the mechanism-level answer. A flat result with (a) chosen everywhere means split-K never beat nvjet even in isolation; a flat result with (c)/(d) chosen on the sliding decode shapes means the isolated saving did not survive the max(pooler, router) critical path or was below ~1%, and either reading closes the question for this area. ab-revert of the landed hypothesis is pointless (it only makes the T=8 pooler invisible); ab-tune has no tunable that touches the pooler; a forced-(a) control is byte-for-byte the base kernel_fn and would be a wasted run. The one refinement worth noting as a follow-up, not a companion, is an in-situ autotune (time each candidate with the router hi/lo + routing running on the side stream as kernel_fn does) if the table picks (c)/(d) but the rows stay flat.

### Open questions
- Does torch.bmm accept out_dtype=torch.float32 in the judge's PyTorch build? master's torch.mm(out_dtype=) works there; if bmm does not, candidates (c)/(d) vanish and the run degenerates to the base.
- Which cuBLAS kernel does the batched [T, 2048] x [2048, 4096] GEMM pick, and with what layout? Storing chunks as [S, 2048, 4096] (K-major rows) gives cuBLAS an NN operand instead of the TN layout nvjet currently uses; consider adding the [S, 4096, 2048] + transpose(1, 2) layout as a candidate, since layout can change the kernel family as much as S does.
- Expected list omits verify_1 (T=8) and verify_16 (T=128) although both are tabled; add them (faster-or-flat and flat) so the attribution grid is complete, and expect the full decode_16/32 shapes (440/615 us) to read flat even on success since the pooler is 3-5% of them.
- Record in FINDINGS, from the profile, the achieved TB/s of whatever bmm kernel the autotune selects (67 MB / kernel us): the skinny-GEMM family on this problem (and bf16_proj's pooler-skinny-tiled-weight-layout) hinges on whether anything beats ~3.6 TB/s on H200 for this weight stream.
- If the table picks (c)/(d) for T=16/32 but the sliding decode rows are flat, the next step is an in-situ autotune (candidates timed with the router hi/lo + routing on the side stream) rather than more split factors; if the table picks (a) everywhere, the area should go on the do-not-retry list for cuBLAS-level tricks and any further pooler work needs a different mechanism (e.g. overlapping the pooler with the unpooler/attention tail, not a faster stream).


## Implementation notes

All in the worktree's `muse_decode_layer/submission.py` (uncommitted; 136 +, 2 -):

- New `_pooler(x, state, tokens)` replaces `_gemm(ffn_input, dense["pooler"])` in kernel_fn (main stream, same spot, still overlapping the side-stream router). It branches on `cfg['tokens']` only via `state["pooler_form"].get(tokens, 0)`; form 0, the default for any T not in the table, is the unchanged `_bf16_gemm_dispatch_impl` call.
- `_pooler_split`: `x.view(T, S, 8192 // S).transpose(0, 1)` (strided batch view, no copy), `torch.bmm(..., out_dtype=torch.float32)` partials, `.sum(0)` in fp32, `.to(bfloat16)` once. Output is bf16 exactly where SGLang has it.
- Candidates: 0 = dispatch (a); 1/2 = 2-/4-way split-K, "nn" weight operand `[S, K/S, N]` = views of one contiguous `pooler.t()` copy (one buffer serves both S); 3/4 = 2-/4-way split-K, "tn" operand: contiguous `[S, N, K/S]` chunk copy used through `.transpose(1, 2)`. All are lossless relayouts of the same bf16 values.
- `_tune_pooler` (called at the end of prepare()): guards `torch.bmm(out_dtype=float32)` with a try on a small strided bf16 test (also checks dtype and allclose to fp32 bmm); on failure it prints a line and DROPS every split-K candidate (table empty, kernel_fn = base). For each T in {1, 8, 16, 32, 128, 256} on synthetic randn x (local seeded generator, global RNG untouched): eagerly checks each split form against dispatch (rel L2 <= 1e-2, else dropped), warms up 3x on a side stream, captures a CUDA graph per candidate (private pools), then 5 warmup + 50 timed rounds, interleaving candidates, each replay preceded by a 256 MB `flush.zero_()` cold-L2 flush and bracketed by CUDA events; medians in us. A split form wins only if the best one beats dispatch by >= 3% or >= 0.5 us; otherwise 0. It prints per T every candidate median, the pick, and both the dispatch and the pick medians, then the final table. Graphs, events, x, ref, flush, w_t and unused chunk copies are freed (sync, then empty_cache), and the current stream is restored in a `finally`. Only `pooler_form` (dict T -> int) and `pooler_chunks` (only the chunk tensors the table actually uses) persist in state.
- Deviations: (b) F.linear dropped. Per the review it resolves to the same cuBLAS TN kernel as `_bf16_gemm_dispatch_impl` / `x @ w.T` on H200, so it would just re-time (a). Added the review's open-question TN-layout variant (`[S, N, K/S]` + transpose) as candidates 3/4, beside the spec's NN layout, because layout can change the cuBLAS kernel family. The spec's "x viewed as [S, T, 8192/S]" is done as `x.view(T, S, K/S).transpose(0, 1)`, as required (a plain `x.view(S, T, ...)` would be wrong).
- Memory: at most +67 MB per persisted layout (NN shared across S; TN one copy per S picked). Transient during prepare: about 3 x 67 MB of chunk copies + 256 MB flush.
- Uncertainties: whether bmm with this strided x (batch stride K/S, ld K) goes to cuBLAS strided-batched without an internal copy (x is at most 4 MB, so even a copy is cheap and gets timed by the autotune anyway); whether the judge's torch has `bmm(out_dtype=)` (if not, the run is the base). Nothing was run: this node has no GPU.

## Rules check

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
