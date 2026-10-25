# muse_decode_layer · 2026-10-25T011821Z · check · boundaries-unpooler-pdl-small-t

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T224452Z-boundaries-unpooler-pdl-small-t.md` (branch `hyp/muse_decode_layer/boundaries-unpooler-pdl-small-t`).

- **Job:** `1dfe83fdbeea458a` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `fadc183`

## Goal
Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte for byte). Three additions to the existing _DL_CUDA TU (new module name, so it recompiles): (1) dl_boundary_kernel: one CTA per token, 256 threads, exact math of FusedPostNormResidualPreNorm as master constructs it (read sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py and muse_spark_layers MuseSpark1xRMSNorm / MuseSpark1xRMSNormWithInputScale / MuseSpark1xNormPreservingResidualConnection: post-norm eps POST_NORM_EPS with gain center one, the residual coefficients from precompute_coefficients read once in prepare into fp32 device scalars, pre-norm with the effective 1+w weight in fp32, outputs branch bf16 [T,8192] and new stream fp32 [T,8192]). It executes griddepcontrol.launch_dependents as its FIRST instruction, then griddepcontrol.wait, then reads its inputs, so the PDL-launched kernel after it (P after the attention boundary) issues its pre-wait weight ring while the boundary runs. Launched itself with the PDL attribute. It replaces state["attn_boundary"](...) and state["ffn_boundary"](...) on the gate (router_dtype is already None there). (2) dl_unpool_kernel<NF>: the P body over dense unpooler bf16 (K = 4096): at T <= 8 only rows 0..1023 (UNPOOL_SLICE; 16 tiles x SPLIT 8 = 128 CTAs) and its CTAs also store bf16 zeros to columns 1024..8191 of the [T,8192] ffn_output (each CTA a disjoint 56-column-per-token slice, after griddepcontrol.wait), replacing _gemm(expert_output, unpooler[:1024]) and F.pad; at 9..32 tokens all 8192 rows (128 tiles x SPLIT 2 = 256 CTAs) replacing _gemm(expert_output, unpooler). Weight ring prefetch before the wait, PDL launch, fixed-order ticket split-K reduce as P. (3) the ffn boundary then runs dl_boundary_kernel (PDL, it waits on the unpooler). Prepare self-check on synthetic data at T in 1, 4, 8, 9, 16, 32: both boundary outputs vs the SGLang modules (branch rel L2 <= 2e-3, stream rel L2 <= 1e-5 relative to |stream|), ffn_output vs master unpooler+pad (rel L2 <= 2e-3, zero columns exactly 0), a captured CUDA graph of boundary -> P -> R -> experts -> unpool -> boundary replayed vs eager; on failure dlb_ok False. Print [dl-boundary-unpool] ON|OFF and CUDA-event times of each new kernel. Attention side, P, R, Marlin, combine unchanged.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (+743/-3). Nothing could be run here (no GPU), so every kernel is unexecuted.

- New TU `_DLB_CUDA` (module `muse_decode_boundary_unpool_v1`, flags `-O3 -arch=sm_90a -lineinfo`, printed), built at import only when the dense-chain TU built. DEVIATION from the spec ("same TU"): a separate TU, so a compile error here cannot take down the landed dense chain; it also avoids textual conflicts with the other TU-editing hypotheses. No sm_90a-only PTX (griddepcontrol, mma.sync, plain cp.async.cg only), so no feature guard. No mbarriers or spin waits (ticket reduce only), so nothing to bound.
- `dl_boundary_kernel`: one CTA per token, 256 threads. `griddepcontrol.launch_dependents` is the first instruction, then L2 prefetch of the constant vectors, then `griddepcontrol.wait`, then the reads. Attn: s1 = alpha1*s + beta1*rms(a, 1e-8) (no weight), out = bf16(rms(s1, 1e-5)*(1+pre_ffn)). FFN: s2 = alpha2*s1 + beta2*rms(f*(1+post_ffn_norm), 1e-8) (the (1+w) is an INPUT scale before the RMS), out = bf16(rms(s2, 1e-5)*(1+next_pre_attn)). Stream stays fp32, activations bf16. At T <= 8 the FFN boundary reads only columns 0..1023 (review optional item 3).
- `dl_unpool_kernel<NF,SPLIT,NROWS>`: P's pipeline over K=4096. T<=8: <1,8,1024> (16 tiles x 8 = 128 CTAs), CTAs also store bf16 zeros to columns 1024..8191 (56 per token per CTA) AFTER `griddepcontrol.wait`; T 9..16: <2,2,8192>; T 17..32: <4,2,8192> (256 CTAs). Fixed-order ticket split-K reduce; ws/cnt self-reset.
- kernel_fn: gate `use_dlb = use_dl and state["dlb"] is not None and tokens <= DLB_MAX_TOKENS` (new constant DLB_MAX_TOKENS = 32, distinct from DL_MAX_TOKENS and DL2_MAX_TOKENS; state["dlb_ok"] mirrors it). T > 32 and any failure: master byte for byte (the edits are an if-branch plus an early return).
- Review corrections applied (the review wins over the spec): (1) alpha/beta are per-channel fp32 [8192] vectors, read from the residual module's own tensors (buffers/params/attrs whose name contains alpha/beta, numel 8192) when present, otherwise reference.py's formula with w = gate.float()/0.3 and clamp 1e-3; the source is printed. I could not read sglang here, so the buffer names are searched, not known. (2) (1+w) of post_ffn_norm is an input scale before the RMS; the attn post-norm has no weight; effective weights come from the modules' `_get_effective_weight()` (fallback 1+w).
- Review items applied: self-check graph rerun with the new boundary (chain T=1, 8, 16, 31, 32: T=32 through the SGLang combine, T<32 through the Triton one, via the real `_experts`), graph vs eager <= 1e-3 and vs SGLang boundaries + master unpooler (next <= 1e-2, stream <= 2e-3). prepare() times `dl_unpool_kernel` against the master `_gemm` path per NF (T=1, 16, 32; 200 MB memset before each launch, its time subtracted, PDL off, so conservative) and keeps master's path for any NF where custom is not faster, printing the pick per NF. Zero stores after the wait kept. `dl_boundary_kernel` CUDA-event time printed at T=1, 8, 32 next to the SGLang module's.
- Self-check per the spec: T in 1,4,8,9,16,32; branch rel L2 <= 2e-3, stream rel L2 <= 1e-5; unpooler vs master <= 2e-3 with the zero columns exactly 0 (output poisoned with 7.0 first); each launch is announced with a flushed print before and `torch.cuda.synchronize()` plus a print after. Failure policy: a boundary or chain failure turns the whole path off; an unpooler failure only turns that NF to master (finer than the spec's single flag); a chain failure with the custom unpooler retries with the master unpooler.
- Biggest risks: the 1e-5 stream tolerance rests on SGLang keeping the post-norm output in fp32 before the residual add (if it rounds to bf16 the check fails and the run silently measures master); unexecuted cp.async/ldmatrix code copied from P with K=4096; the early launch_dependents PDL edge.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only `torch.empty` of shape-sized buffers and kernel launches (cudaLaunchKernelEx); no `.item()`/`.cpu()`/synchronize/prints; control flow on `tokens` and state only. The self-check captures the full chain in a graph and replays it.
- [x] Nothing derived from `data` is cached across calls: alpha/beta/weights/ws/cnt are built from `weights` in prepare(); `data["stream"]` and activations are read per call (git diff checked).
- [x] Weights: only prepare()-time float32 vectors derived from the norm/gate weights (losslessly computed by the modules' own formulas); the unpooler stays the BF16 `state["dense"]["unpooler"]` untouched.
- [x] Activations BF16 where SGLang keeps BF16 (ffn_input, ffn_output, next input), residual stream fp32; no FP8/KV changes.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: load_inline only (no child process in the source; the existing build path is reused).
- [x] All work happens inside the timed call; no output caching or replay (new kernels run every call).
- [x] Gate constant DLB_MAX_TOKENS is separate from DL_MAX_TOKENS and DL2_MAX_TOKENS.

okc_lint output (non-host-sync lines; host-sync warnings are all prepare()-only, "outside kernel_fn's call graph"):

```
OK with warnings: muse_decode_layer/submission.py
```
py_compile OK; __pycache__ removed.

## Risks / what could fail
high
High: the boundary must reproduce SGLang residual-gate numerics exactly (stream output is checked against fp32 reference); the self-check gates it. An early launch_dependents is legal because the dependent still griddepcontrol.waits before reading anything this kernel writes; only constant weights are read before the wait. Disjoint from dense-chain-stage2-attn-side-pdl (different kernels), so the two can stack by merge. bf16-m32-swapab-wgmma-port (held) also targets the T=32 unpooler.

## Result (after the run)
- job `1dfe83fdbeea458a` (check), status done, passed True, tests 13/13
- speedup 1.2559x, 95% CI [1.2371, 1.2748], score 18010.581858384812 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.patch`; branch `hyp/muse_decode_layer/boundaries-unpooler-pdl-small-t`
