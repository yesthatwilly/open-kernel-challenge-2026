# muse_attention_decode · 2026-10-25T150712Z · submit · epilogue-gate-l2-prefetch-cuda

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T150036Z-epilogue-gate-l2-prefetch-cuda.md` (branch `hyp/muse_attention_decode/epilogue-gate-l2-prefetch-cuda`).

- **Job:** `917a9689ba4645bd` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `de94b09f`

## Goal
Built on current master eb1b68b (no depends_on). Mechanism: both landed CUDA kernels read the output gate logits (qkv row r, bf16 columns (HQ + 2*HKV)*D .. +HQ*D, i.e. 2 KB = 16 aligned 128-byte lines per row) only in their final epilogue, after the last merge, and the judge flushes L2 before every replay, so that load is a cold HBM round trip (~0.7-1 us) on the critical path of the last CTA. The q / new-k / v loads at kernel start touch other lines of the same row, so the gate lines are never warm. Fix: issue L2 prefetches for exactly those gate lines at kernel entry, so the epilogue load hits L2. No other change; no math, layout, rounding, grid or route changes. Use one helper in each CUDA source: __device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); } (plain prefetch.global.L2, no evict_* / createpolicy cache-hint form). (1) sliding_cluster_kernel (_SLIDE_CUDA; master routes sliding decode B=1 at C=16 and B=17..32 at C=4): immediately after b / rank are computed and before the prologue page-index loads, let NR = HQ / C; threads tid < NR prefetch the line qkv + b*qkv_stride + (HQ + 2*HKV)*D + (rank*NR + tid)*D (the rows this rank finishes in the DSMEM reduce-scatter), and thread tid == NR prefetches v_scale_p (read with __ldg in the same epilogue). (2) splitkv_attn_sk_kernel (_SK_CUDA, every template instantiation; master routes full decode B=1 two-level, full decode 1<B<=64 single-level ticket, and sliding verify B=1 PSEUDO): as the first statements of the kernel, before load_prefix / load_prefix_pseudo, with R = PSEUDO ? B*8 : B (the same R the kernel already uses), for (int i = blockIdx.x*THREADS + tid; i < R*16; i += gridDim.x*THREADS) prefetch_l2(qkv + (int64_t)(i >> 4)*qkv_stride + (HQ + 2*HKV)*D + (i & 15)*64). Every CTA participates before the early returns, so the lines are in L2 whichever CTA wins the ticket; at most one prefetch per thread at the gated shapes (R*16 <= 512 lines vs grid*THREADS). Leave splitkv_attn_sk_fused_kernel (skF, not routed on master) and the stage-1 kernels untouched; _SKF_HEAD slices _SK_CUDA up to 'template <int STAGES, bool TICKET', so put the helper above that point or in both sources and confirm the skF build line still prints. Every prepare-time self-check, replay check and timing gate stays as is, so a broken build falls back exactly as master's would. In the log, confirm before trusting the run: the [slide] fp8qk regs/local/spill line (master: 124 regs, 32 B local, 103424 B smem; an extra spill is a red flag), [slide-gate] C=16 ROUTE and C=4 ROUTE, PATH CHOSEN sliding verify B=1 pseudo grid p8, and PATH CHOSEN full decode_1 streamk two-level, all as master. Rules: prefetch is a hint with no data dependence, so the kernel stays graph-capturable, adds no host sync, allocation or input-dependent Python branch, caches nothing across calls, and every key is still attended with FP8 KV and E4M3 q.

## What changed, and where
Worktree base 5e7ce21f; master (9716bda4) has an identical muse_attention_decode/submission.py, so no merge was needed. +16 lines, only P/submission.py.
- _SK_CUDA: `prefetch_l2` helper defined once, above load_prefix (before the `template <int STAGES, bool TICKET` slice point, so _SKF_HEAD carries it and skF still compiles; skF kernel itself untouched). In splitkv_attn_sk_kernel, right after `const int R`, before load_prefix: grid-stride loop over R*16 gate lines.
- _SLIDE_CUDA: helper defined once above sliding_cluster_kernel; after b/rank, tid < HQ/C prefetch the gate line of row rank*NR+tid, tid == NR prefetches v_scale_p.
- Review fix 1 (applied): loop bound uses the kernel's own R = PSEUDO ? B*TOK : B, not B*8, so TOK=1 (sliding decode B=17..32 sk window) never reads past the tensor. Overrides the spec.
- Review fix 2 (applied): helper exactly once in _SK_CUDA above the slice point, once in _SLIDE_CUDA.
- Review fix 3 (applied): no PATH CHOSEN / logging line touched, including the sliding decode B=17..32 sk-window line.
- Open questions (SASS position, regs/spill, L2 survival at full decode_1): cannot be answered here (no GPU); the existing [slide]/[sk]/[sd]/[ps]/[skF] regs/local/spill lines print at prepare time and must be read from the run log.
- Deviation: none beyond the R correction.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: only a prefetch hint added, no sync/alloc/.item() in kernel_fn.
- [x] Nothing derived from data is cached across calls (no globals added; pointers come from kernel args).
- [x] Weights unchanged.
- [x] Activations/KV/q precision unchanged (FP8 KV, E4M3 q).
- [x] Attention sees every key: no math, grid or route change.
- [x] No timing patches, harness edits, reference import, subprocess/fork.
- [x] All work inside the timed call; no output caching.

okc_lint: "OK with warnings" (only existing host-sync warnings in prepare(), outside kernel_fn). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Medium risk only because it edits two landed CUDA sources (one 40 s rebuild each); the added instruction is the plain prefetch.global.L2 hint on in-bounds addresses, which cannot fault and changes no data, and every route keeps its own self-check and timing gate. If the build fails the run measures the fallbacks (Triton sliding decode, FA3 sliding verify_1) and loses ~2-5% on those rows, so the auditor must grep the build and PATH CHOSEN lines listed in the change. Expected headline ~+0.6-1.0% if the epilogue load is as exposed as the cold-L2 model says; a flat result says the compiler already overlapped it or the line was warm, and closes this knob for both kernels. Weakest link: sector-level L2 behaviour; if a gate line was already pulled in with the q row the effect is zero, which the per-row deltas will show (sliding decode_1 and sliding verify_1 are 0.05 us-resolution rows). Independent of preattention-tiled-every-verify-block (Triton pre-attention on verify) and of the queued sliding-decode-sk-s3 line (which edits splitkv_attn_sk_fused_kernel, not these two kernels).

## Result (after the run)
- job `917a9689ba4645bd` (submit), status done, passed True, tests 16/16
- speedup 1.5020x, 95% CI [1.4795, 1.5245], score 2507.51511990477 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5016522435853301, "ci": [1.47912745993155, 1.530423257736894], "job": "252abb7f11ec48c3"}; base result: muse_attention_decode/hypotheses/done/neutral/restack-sliding-decode-sk-window-on-cluster-master.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/epilogue-gate-l2-prefetch-cuda.result.json`, `muse_attention_decode/hypotheses/done/neutral/epilogue-gate-l2-prefetch-cuda.patch`; branch `hyp/muse_attention_decode/epilogue-gate-l2-prefetch-cuda`
