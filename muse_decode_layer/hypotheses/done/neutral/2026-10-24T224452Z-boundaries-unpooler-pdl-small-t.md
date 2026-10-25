---
problem: muse_decode_layer
slug: boundaries-unpooler-pdl-small-t
created: 2026-10-24T224452Z
base: master
area: post-MoE unpooler + residual boundaries (T<=32)
change: >
  Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate
  plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte for byte). Three additions to
  the existing _DL_CUDA TU (new module name, so it recompiles): (1) dl_boundary_kernel: one CTA
  per token, 256 threads, exact math of FusedPostNormResidualPreNorm as master constructs it (read
  sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py and muse_spark_layers
  MuseSpark1xRMSNorm / MuseSpark1xRMSNormWithInputScale /
  MuseSpark1xNormPreservingResidualConnection: post-norm eps POST_NORM_EPS with gain center one,
  the residual coefficients from precompute_coefficients read once in prepare into fp32 device
  scalars, pre-norm with the effective 1+w weight in fp32, outputs branch bf16 [T,8192] and new
  stream fp32 [T,8192]). It executes griddepcontrol.launch_dependents as its FIRST instruction,
  then griddepcontrol.wait, then reads its inputs, so the PDL-launched kernel after it (P after
  the attention boundary) issues its pre-wait weight ring while the boundary runs. Launched itself
  with the PDL attribute. It replaces state["attn_boundary"](...) and state["ffn_boundary"](...)
  on the gate (router_dtype is already None there). (2) dl_unpool_kernel<NF>: the P body over
  dense unpooler bf16 (K = 4096): at T <= 8 only rows 0..1023 (UNPOOL_SLICE; 16 tiles x SPLIT 8 =
  128 CTAs) and its CTAs also store bf16 zeros to columns 1024..8191 of the [T,8192] ffn_output
  (each CTA a disjoint 56-column-per-token slice, after griddepcontrol.wait), replacing
  _gemm(expert_output, unpooler[:1024]) and F.pad; at 9..32 tokens all 8192 rows (128 tiles x
  SPLIT 2 = 256 CTAs) replacing _gemm(expert_output, unpooler). Weight ring prefetch before the
  wait, PDL launch, fixed-order ticket split-K reduce as P. (3) the ffn boundary then runs
  dl_boundary_kernel (PDL, it waits on the unpooler). Prepare self-check on synthetic data at T in
  1, 4, 8, 9, 16, 32: both boundary outputs vs the SGLang modules (branch rel L2 <= 2e-3, stream
  rel L2 <= 1e-5 relative to |stream|), ffn_output vs master unpooler+pad (rel L2 <= 2e-3, zero
  columns exactly 0), a captured CUDA graph of boundary -> P -> R -> experts -> unpool -> boundary
  replayed vs eager; on failure dlb_ok False. Print [dl-boundary-unpool] ON|OFF and CUDA-event
  times of each new kernel. Attention side, P, R, Marlin, combine unchanged.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "incumbent 132.7/142.6 us: unpooler 5.3 + splitKreduce ~1.7 + F.pad ~2 -> ~3 us, two SGLang boundaries 3.6+3.7 -> ~2+2, plus P ring fill overlapped with the boundary ~2 us: ~8-10 us (+6-7%)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "same at T=8: ~8-9 us of 186.5/195.9 (+4-5%)"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "unpooler 67 MB ~19.5 us nvjet -> ~18 at 3.7 TB/s, boundaries and overlap ~4 us: ~5-6 us (+2-2.5% sliding)"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: ~5-7 us of 247/535 (+1-2.5%); headline +2.5-3.5%
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "outside gate: control"
risk: high
depends_on: small-t-launch-minimal-dense-chain-pdl
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T010851Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T010851Z
base_sha: 63b7f3a3bc19c847c204f5d5298872f66d86505a
base_blob: 4da94677234faaa06b42b5ced541db8961fd6f34
branch: hyp/muse_decode_layer/boundaries-unpooler-pdl-small-t
worktree: .worktrees/muse_decode_layer/boundaries-unpooler-pdl-small-t
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T011821Z
impl_commit: fadc1836cb3258ecd4aa35c99d7b6731c61ebe6e
submission_sha256: ed0b3be71ffdce937b892bcf4d72b3fc8a43a56f873535f170c9ade758ff640d
norm_sha256: 49f6255c1af271dde88fdaeb641cab353db592d2757d4e4cb9de0aa7e2b5de0f
submission_blob: 7781304a6fb5d86a719e72d889f8db724b7f310f
lint_warnings: 33
attempt_file: 2026-10-25T011821Z-check-boundaries-unpooler-pdl-small-t.md
prepared_at: 2026-10-25T011821Z
job: 1dfe83fdbeea458a
dispatched_at: 2026-10-25T011829Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2559449871115151
speedup_ci: [1.2371058123048424, 1.2747841619181877]
harvested_at: 2026-10-25T034135Z
best_at_harvest: 1.2521147631671776
promotion: null
---

# boundaries-unpooler-pdl-small-t

## Change

Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte for byte). Three additions to the existing _DL_CUDA TU (new module name, so it recompiles): (1) dl_boundary_kernel: one CTA per token, 256 threads, exact math of FusedPostNormResidualPreNorm as master constructs it (read sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py and muse_spark_layers MuseSpark1xRMSNorm / MuseSpark1xRMSNormWithInputScale / MuseSpark1xNormPreservingResidualConnection: post-norm eps POST_NORM_EPS with gain center one, the residual coefficients from precompute_coefficients read once in prepare into fp32 device scalars, pre-norm with the effective 1+w weight in fp32, outputs branch bf16 [T,8192] and new stream fp32 [T,8192]). It executes griddepcontrol.launch_dependents as its FIRST instruction, then griddepcontrol.wait, then reads its inputs, so the PDL-launched kernel after it (P after the attention boundary) issues its pre-wait weight ring while the boundary runs. Launched itself with the PDL attribute. It replaces state["attn_boundary"](...) and state["ffn_boundary"](...) on the gate (router_dtype is already None there). (2) dl_unpool_kernel<NF>: the P body over dense unpooler bf16 (K = 4096): at T <= 8 only rows 0..1023 (UNPOOL_SLICE; 16 tiles x SPLIT 8 = 128 CTAs) and its CTAs also store bf16 zeros to columns 1024..8191 of the [T,8192] ffn_output (each CTA a disjoint 56-column-per-token slice, after griddepcontrol.wait), replacing _gemm(expert_output, unpooler[:1024]) and F.pad; at 9..32 tokens all 8192 rows (128 tiles x SPLIT 2 = 256 CTAs) replacing _gemm(expert_output, unpooler). Weight ring prefetch before the wait, PDL launch, fixed-order ticket split-K reduce as P. (3) the ffn boundary then runs dl_boundary_kernel (PDL, it waits on the unpooler). Prepare self-check on synthetic data at T in 1, 4, 8, 9, 16, 32: both boundary outputs vs the SGLang modules (branch rel L2 <= 2e-3, stream rel L2 <= 1e-5 relative to |stream|), ffn_output vs master unpooler+pad (rel L2 <= 2e-3, zero columns exactly 0), a captured CUDA graph of boundary -> P -> R -> experts -> unpool -> boundary replayed vs eager; on failure dlb_ok False. Print [dl-boundary-unpool] ON|OFF and CUDA-event times of each new kernel. Attention side, P, R, Marlin, combine unchanged.

## Expected

- `*load=decode_1 *` faster: incumbent 132.7/142.6 us: unpooler 5.3 + splitKreduce ~1.7 + F.pad ~2 -> ~3 us, two SGLang boundaries 3.6+3.7 -> ~2+2, plus P ring fill overlapped with the boundary ~2 us: ~8-10 us (+6-7%)
- `*load=verify_1 *` faster: same at T=8: ~8-9 us of 186.5/195.9 (+4-5%)
- `*load=decode_16 *` faster: unpooler 67 MB ~19.5 us nvjet -> ~18 at 3.7 TB/s, boundaries and overlap ~4 us: ~5-6 us (+2-2.5% sliding)
- `*load=decode_32 *` faster: ~5-7 us of 247/535 (+1-2.5%); headline +2.5-3.5%
- `*load=verify_16 *` flat: outside gate: control
- `*load=verify_32 *` flat: outside gate: control

## Risk

High: the boundary must reproduce SGLang residual-gate numerics exactly (stream output is checked against fp32 reference); the self-check gates it. An early launch_dependents is legal because the dependent still griddepcontrol.waits before reading anything this kernel writes; only constant weights are read before the wait. Disjoint from dense-chain-stage2-attn-side-pdl (different kernels), so the two can stack by merge. bf16-m32-swapab-wgmma-port (held) also targets the T=32 unpooler.

## Adversarial review

Reviewed 2026-10-25T010851Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible; duplicates nothing landed, running or do-not-retry. Two spec errors to fix before the build: residual coefficients are per-channel [8192] vectors, not scalars, and post_ffn applies (1+w) as an input scale before the RMS. The T=9..32 full-width unpooler will probably be flat or slower than nvjet, so decode_16/32 'faster' rests almost entirely on the boundaries.

### Justification
At T<=8 the replaced chain (nvjet sliced unpooler 8.4 MB + splitKreduce + F.pad, ~9 us vs a ~3 us P-body GEMV) is real overhead; the P-body GEMV already runs fault-free at ~3.4-3.6 TB/s for NF=1. Fusing both boundaries with an immediate launch_dependents makes attn boundary -> P the first adjacent hand-written pair, so P's W_pr prefetch overlaps for real: decode_1/verify_1 ~5-8 us. Incumbents are stale (master decode_1 127.5/142.9). Weakest link: the T=9..32 full-width unpooler (NF=2/4) inherits P's x L2 re-read (2.85 TB/s at T=32, ~3.1 at T=16) vs master nvjet ~3.1-3.2 TB/s: parity at T=16, up to ~2 us slower at T=32. At T=1 the one-CTA boundary runs two dependent block reductions over ~100 KB, ~2-3 us vs SGLang 3.6. Spec errors: (1) alpha = sqrt(clamp(sigmoid(-w/0.3)*(1+sigmoid(w/0.3)), 1e-3)) and beta = sigmoid(w/0.3) per channel; attn/ffn_residual_gate are [8192] (reference.py:109,123,339-344), so alpha/beta must be fp32 [8192] device vectors from the module's precomputed buffers, not scalars, else dlb_ok goes False and the check measures master. (2) ffn boundary (reference.py:472-476): stream2 = alpha2*stream1 + beta2*rms(f*(1+post_ffn_norm), 1e-8); (1+w) is an INPUT scale before the RMS; only the attn post-norm has no weight; next attention input rms(stream2)*(1+next_pre_attn_norm) eps 1e-5. (3) Optional: at T<=8, f columns 1024..8191 are zero, so the ffn boundary can read 1024 columns. Headline ~+1.5-2.5%, inside CI; read profile rows.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (base): overlap exists only because P is PDL-launched; re-run _dl_self_check graph (attn_boundary -> P -> R -> experts) with the new boundary; part of any gain shows as a shorter P row.
- post-expert-norm-triton-small-t: the unpooler's predecessor (Triton at T<32, SGLang at T=32); self-check must cover both layouts through the real _experts.
- stack-align-par-sidestream-sliding1: no interaction at T<=32; stale incumbents only.
- marlin-block-m-cap-16, fa3-hdim128-head-pairing-batched, sglang-pipeline-parity-router-hilo-m8: no confound.

### Interference and A/B needs
No companion. Rows dl_boundary_kernel, dl_unpool_kernel<1> (T<=8) and dl_unpool_kernel<2/4> (T=9..32) are separable from master's rows; a shorter P row isolates the PDL overlap; verify_16/32 are the drift control. Build note: in prepare time dl_unpool_kernel vs _gemm per NF on synthetic data and keep nvjet for any NF where the custom kernel is slower; print the pick per NF. Overlaps: dense-chain-stage2 (kernel-disjoint, same TU: textual conflict, complementary); split-KV checks (no overlap); dl-route-warp-topk-pdl edits P and R in the same TU; dl-p-wide-tile-t16-t32 fixes the x re-read this inherits.

### Open questions
- dl_boundary_kernel CUDA-event time at T=1 with one CTA vs SGLang 3.6/3.7 us? If >= 3 us, several CTAs per token?
- Does SGLang FusedPostNormResidualPreNorm read alpha/beta as precomputed fp32 buffers or recompute from bf16 gates? Match its rounding.
- Is dl_unpool_kernel<2/4> faster than nvjet at T=9..32 in prepare timing? If not, gate it off.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (+743/-3). Nothing could be run here (no GPU), so every kernel is unexecuted.

- New TU `_DLB_CUDA` (module `muse_decode_boundary_unpool_v1`, flags `-O3 -arch=sm_90a -lineinfo`, printed), built at import only when the dense-chain TU built. DEVIATION from the spec ("same TU"): a separate TU, so a compile error here cannot take down the landed dense chain; it also avoids textual conflicts with the other TU-editing hypotheses. No sm_90a-only PTX (griddepcontrol, mma.sync, plain cp.async.cg only), so no feature guard. No mbarriers or spin waits (ticket reduce only), so nothing to bound.
- `dl_boundary_kernel`: one CTA per token, 256 threads. `griddepcontrol.launch_dependents` is the first instruction, then L2 prefetch of the constant vectors, then `griddepcontrol.wait`, then the reads. Attn: s1 = alpha1*s + beta1*rms(a, 1e-8) (no weight), out = bf16(rms(s1, 1e-5)*(1+pre_ffn)). FFN: s2 = alpha2*s1 + beta2*rms(f*(1+post_ffn_norm), 1e-8) (the (1+w) is an INPUT scale before the RMS), out = bf16(rms(s2, 1e-5)*(1+next_pre_attn)). Stream stays fp32, activations bf16. At T <= 8 the FFN boundary reads only columns 0..1023 (review optional item 3).
- `dl_unpool_kernel<NF,SPLIT,NROWS>`: P's pipeline over K=4096. T<=8: <1,8,1024> (16 tiles x 8 = 128 CTAs), CTAs also store bf16 zeros to columns 1024..8191 (56 per token per CTA) AFTER `griddepcontrol.wait`; T 9..16: <2,2,8192>; T 17..32: <4,2,8192> (256 CTAs). Fixed-order ticket split-K reduce; ws/cnt self-reset.
- kernel_fn: gate `use_dlb = use_dl and state["dlb"] is not None and tokens <= DLB_MAX_TOKENS` (new constant DLB_MAX_TOKENS = 32, distinct from DL_MAX_TOKENS and DL2_MAX_TOKENS; state["dlb_ok"] mirrors it). T > 32 and any failure: master byte for byte (the edits are an if-branch plus an early return).
- Review corrections applied (the review wins over the spec): (1) alpha/beta are per-channel fp32 [8192] vectors, read from the residual module's own tensors (buffers/params/attrs whose name contains alpha/beta, numel 8192) when present, otherwise reference.py's formula with w = gate.float()/0.3 and clamp 1e-3; the source is printed. I could not read sglang here, so the buffer names are searched, not known. (2) (1+w) of post_ffn_norm is an input scale before the RMS; the attn post-norm has no weight; effective weights come from the modules' `_get_effective_weight()` (fallback 1+w).
- Review items applied: self-check graph rerun with the new boundary (chain T=1, 8, 16, 31, 32: T=32 through the SGLang combine, T<32 through the Triton one, via the real `_experts`), graph vs eager <= 1e-3 and vs SGLang boundaries + master unpooler (next <= 1e-2, stream <= 2e-3). prepare() times `dl_unpool_kernel` against the master `_gemm` path per NF (T=1, 16, 32; 200 MB memset before each launch, its time subtracted, PDL off, so conservative) and keeps master's path for any NF where custom is not faster, printing the pick per NF. Zero stores after the wait kept. `dl_boundary_kernel` CUDA-event time printed at T=1, 8, 32 next to the SGLang module's.
- Self-check per the spec: T in 1,4,8,9,16,32; branch rel L2 <= 2e-3, stream rel L2 <= 1e-5; unpooler vs master <= 2e-3 with the zero columns exactly 0 (output poisoned with 7.0 first); each launch is announced with a flushed print before and `torch.cuda.synchronize()` plus a print after. Failure policy: a boundary or chain failure turns the whole path off; an unpooler failure only turns that NF to master (finer than the spec's single flag); a chain failure with the custom unpooler retries with the master unpooler.
- Biggest risks: the 1e-5 stream tolerance rests on SGLang keeping the post-norm output in fp32 before the residual add (if it rounds to bf16 the check fails and the run silently measures master); unexecuted cp.async/ldmatrix code copied from P with K=4096; the early launch_dependents PDL edge.

## Rules check

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

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only review plus minimal fixes). This node has no GPU, and there is no sglang source on it or in the repo, so risks A and B are now settled at prepare() time by probing the module itself instead of by reading its source.

**Verdict: dispatch-with-fixes.**

Fixes (worktree `muse_decode_layer/submission.py`):
1. (B) `_dlb_coeffs` no longer searches attribute names for alpha/beta. It builds a probe `FusedPostNormResidualPreNorm(post_attn (weightless), _residual(gate), pre_ffn)` and reads alpha = new_stream(stream=1, branch=0) and beta = new_stream(stream=0, branch=1). These are exactly the per-channel values the module applies, whatever its buffer names or storage dtype. They are cross-checked against reference.py:339-344 (a mismatch over 1e-2 raises, so the path turns OFF and prints PATH CHOSEN) and the source is printed.
2. (A) Bit 1 of the kernel's `has_scale` argument, now `flags` (module renamed to `_v2`), rounds the post-norm output to bf16 before the residual add. New `_dlb_pick_round` runs both variants against the attn and ffn modules on synthetic data, keeps the closer one per boundary in `dlb["rnd1"]` / `dlb["rnd2"]`, and prints the errors.
3. The stream self-check tolerance is now relative to the residual update, `|s - ref| / |ref - alpha*s_in| <= 4e-3`, which is bf16-level and is the quantity the judge checks, instead of `1e-5 * |stream|`. The old check would have failed on any bf16 rounding inside SGLang, whose contribution is about 3e-4 of |stream|. Formula bugs such as a wrong alpha, a missing scale or a wrong channel still give >= 1e-2.
4. If `_dlb_setup` raises, it now re-zeroes the P/R tickets (`cnt_p` / `cnt_r`) before falling back.

Checked OK:
- Formulas against reference.py:452-453, 472-476 and 339-344: the attn post-norm has no weight; the ffn (1+w) is an input scale before the RMS with eps 1e-8; the pre-norm uses eps 1e-5 times (1+w).
- PDL: the boundary prefetches only the prepare constants before `griddepcontrol.wait`. The unpooler issues only the weight ring before the wait and stores its zeros after it. Both kernels wait unconditionally, so the chain is transitive.
- The unpooler is a faithful copy of P's pipeline and indexing: K=4096, workspace and tickets in range, zero slices 16-byte aligned.
- The per-NF timing uses synthetic data only.
- kernel_fn has no host sync, only shape-sized `torch.empty` calls and launches, and is graph-capturable.
- No child process beyond load_inline, nothing cached from `data`, and no ptxVersion gate.

Residual risk: all of the CUDA code is still unexecuted; a fault in the new kernels or a module probe that misbehaves turns the path OFF, so the run measures the landed dense chain.

## Result

- job `1dfe83fdbeea458a` (check), status done, passed True, tests 13/13
- speedup 1.2559x, 95% CI [1.2371, 1.2748], score 18010.581858384812 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/boundaries-unpooler-pdl-small-t.patch`; branch `hyp/muse_decode_layer/boundaries-unpooler-pdl-small-t`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 127.48 | 123.87 | 1.025 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.93 | 140.45 | 1.0162 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 204.98 | 208.88 | 0.9859 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 349.47 | 357.68 | 0.9965 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 245.77 | 247.46 | 0.9971 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 529.71 | 520.57 | 0.9954 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 190.8 | 181.58 | 1.0146 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 192.44 | 191.12 | 1.0003 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 429.61 | 427.43 | 1.0084 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 563.65 | 575.79 | 1.0088 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 592.52 | 604.9 | 1.0083 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 861.3 | 889.52 | 0.9952 | flat |

no shape moved outside noise. Expected hit: 2/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: boundaries-unpooler-pdl-small-t`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

