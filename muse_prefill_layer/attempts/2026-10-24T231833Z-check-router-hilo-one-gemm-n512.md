# muse_prefill_layer · 2026-10-24T231833Z · check · router-hilo-one-gemm-n512

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T185248Z-router-hilo-one-gemm-n512.md` (branch `hyp/muse_prefill_layer/router-hilo-one-gemm-n512`).

- **Job:** `f4e3c5f4c306427b` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `dcffe36`

## Goal
Built on master (635cfee). prepare(): add state['router_hilo_t'] = torch.cat((hi.t(), lo.t()), dim=1).contiguous(), bf16 [8192, 512] (the same BF16 hi/lo split master stores as router_hi_t and router_lo_t, side by side: a lossless repack of the pair the rules allow). kernel_fn, inside the existing `with torch.cuda.stream(side_stream):` block, replace the two lines `logits = torch.mm(ffn_input, router_hi_t, out_dtype=fp32); logits += torch.mm(ffn_input, router_lo_t, out_dtype=fp32)` with `both = torch.mm(ffn_input, state['router_hilo_t'], out_dtype=torch.float32)` ([T, 512] fp32) and `logits = both[:, :256] + both[:, 256:]` (one elementwise add, fresh contiguous [T, 256] fp32). Same math: each logit is still x.hi + x.lo, each dot accumulated in fp32 by cuBLAS, summed in fp32. FusedMuseSparkRouting, the stream fork/join, the pooler and everything else are unchanged. MECHANISM: today the pair is two cuBLAS GEMMs with N=256 plus an in-place add; at T=753 cuBLAS runs them as split-K nvjet tss_64x96 (32.7 us sliding, 81.8 us full turn_1, plus cublasLt splitKreduce kernels) and at T=3012 as tss_64x104 at ~190 TFLOP/s (131.9-139.6 us), all on the side stream sharing SMs with the pooler. One GEMM with N=512 gives cuBLAS twice the N tiles to fill the grid without split-K at these M, one launch and one reduce fewer. prepare(): one check that logits from the new path equal master's two-GEMM logits within 1e-5 relative on synthetic bf16 [753, 8192] and [3012, 8192] inputs, and that the routing kernel picks identical top-8 on them; print both paths' kernel names via a short torch.profiler pass if cheap, else skip.

## What changed, and where
Built in the main thread 2026-10-24 ~23:35Z with self-review (user approved; no new agents before the reset). prepare(): state["router_hilo_t"] = cat(hi.t(), lo.t(), dim=1) bf16 [8192, 512]. kernel_fn side stream: one torch.mm(ffn_input, router_hilo_t, out_dtype=fp32) then logits = both[:, :256] + both[:, 256:]. router_hi_t/lo_t kept in state but unused. Same math per logit (x.hi + x.lo in fp32); the cuBLAS reduction order for the N=512 call may differ by fp32 rounding only.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no host sync, allocation on input values or value-dependent control flow added to kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 and BF16 unchanged; the router hi/lo split is the one the rules allow (hi+lo, fp32 accumulate).
- [x] Activations BF16 as in SGLang; KV FP8; queries unchanged.
- [x] Every key attended (attention untouched).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (prepare() only builds a weight repack).
- [x] okc_lint: OK with warnings (pre-existing prepare-only host syncs).

## Risks / what could fail
low
Same area as the held router-triton-hilo-splitk-turn (Triton split-K port) and router-pooler-concat-gemm-turn, but a different, smaller mechanism (stay on cuBLAS, one N=512 call); judge it on the router rows and the turn_1 full shape, where the side branch is most likely on the critical path. The extra [T,256] add kernel replaces master's in-place add, so the launch count drops by one GEMM. Risk: cuBLAS may still pick split-K at M=753, N=512 (then the gain is one launch); expected headline move is small, ~0.3-0.8%.

## Result (after the run)
- job `f4e3c5f4c306427b` (check), status done, passed True, tests 5/5
- speedup 1.0607x, 95% CI [1.0448, 1.0767], score 365218.55816558946 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/router-hilo-one-gemm-n512.result.json`, `muse_prefill_layer/hypotheses/done/neutral/router-hilo-one-gemm-n512.patch`; branch `hyp/muse_prefill_layer/router-hilo-one-gemm-n512`
