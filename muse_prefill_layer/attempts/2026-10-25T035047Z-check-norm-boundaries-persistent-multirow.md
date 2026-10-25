# muse_prefill_layer · 2026-10-25T035047Z · check · norm-boundaries-persistent-multirow

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T224410Z-norm-boundaries-persistent-multirow.md` (branch `hyp/muse_prefill_layer/norm-boundaries-persistent-multirow`).

- **Job:** `46afe489e2754c83` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `b2254b6`

## Goal
Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program structure as the incumbent ... re-mint as a multi-row/persistent variant if wanted'). Replace both state['attn_boundary'](...) and state['ffn_boundary'](...) calls in kernel_fn with one new Triton kernel _boundary_persistent_kernel launched twice, at every T. Math per row, fp32 throughout, exactly as the held hypothesis spells it out (read its change field): b = branch[t].float(); FFN boundary only: b *= post_ffn effective weight (1 + w); b *= rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b; out = s_new * rsqrt(mean(s_new^2) + 1e-5) * pre-norm effective weight (pre_ffn for the attention boundary, next_pre_attn for the FFN boundary); store s_new fp32 and out bf16 into fresh torch.empty tensors. Take alpha, beta and the effective weights from the SGLang modules master already builds in prepare() (_residual(...) after precompute_coefficients(), _norm(...) effective weights), as fp32 [8192] tensors, rather than re-deriving constants. STRUCTURE (the difference from the incumbent and from the held row kernel): grid = NUM_SMS * 2 programs (host int from torch.cuda.get_device_properties), each walks rows t = pid, pid + nprog, ... < T; the row is processed in two passes over 8192 columns in BLOCK=2048 chunks with 16-byte vector loads (pass 1 accumulates sum(b^2) while keeping the branch chunk and stream chunk... if 8192 fp32 + 8192 bf16 per row do not fit in registers at num_warps 8, re-read them in pass 2 from L2, which is where they will be), and the next row's first chunk is loaded before the current row's stores (software prefetch), so each SM keeps two rows in flight and there are no CTA launch waves (753 rows was ~3-6 waves of one-row CTAs). prepare(): compare against the SGLang modules on synthetic inputs at T = 753 and 3012 for both boundaries (stream rel L2 <= 1e-6, bf16 output max diff <= 1 ulp); on mismatch log '[boundary] OFF: <reason>' and keep master's modules, else 'ON'. Nothing else changes.

## What changed, and where
Only `muse_prefill_layer/submission.py` changed (+242/-3). Added before `prepare()`: `_boundary_rows_kernel` (one row per program, whole 8192 row in registers, grid T), `_boundary_persistent_kernel` (grid min(T, 2 x SMs), row loop with the next row's branch/stream loads issued before the current row's math and stores), a shared `_boundary_row` math helper (fp32 throughout: optional post-FFN (1+w) multiply, rsqrt(mean+1e-8), alpha/beta combine, rsqrt(mean+1e-5) x pre-norm weight; s_new fp32 and out bf16 into torch.empty), `_boundary_launch`, and the prepare-time parity gate and sweep. `prepare()` ends with `state["bnd"] = _boundary_prepare(...)`; `kernel_fn` calls `_boundary(state, "attn"|"ffn", ...)`, which uses the T-class choice (T<1536 -> 753, <8192 -> 3012, else 16384; a shape-only lookup) or master's SGLang modules unchanged.

Per the review (which wins):
- Sweep per T class (753, 3012, 16384) on synthetic data: master vs {rows nw4 (single wave, every row resident), rows nw8, persistent nw8}. Each is CUDA-graph timed (attn + ffn boundary per graph, 256 MB L2 flush before each replay, median of 25). Prints `[boundary] variant ... us` and `PATH CHOSEN` per class and overall. A variant is picked only if <= 0.98x master; otherwise master's modules. Any exception means `[boundary] OFF: <reason>`.
- Parity gate relaxed: stream rel L2 <= 1e-6, out max <= 2 bf16 ulp AND out rel L2 <= 3e-3, both boundaries, at every swept T, against the SGLang modules; a failing variant is skipped (logged), not silently chosen.

Deviations: (1) persistent kernel keeps the whole row as one 8192 block rather than 4 x 2048 two-pass chunks, because the review showed chunking cuts bytes in flight and gives no win; prefetch is of the next whole row. (2) alpha/beta are computed in prepare() from the gate weights with the reference's formula (temperature 0.3, eps 1e-3) because the SGLang module's coefficient attribute names could not be inspected (no sglang on this node); the parity gate catches any mismatch. Effective weights come from the modules' `_get_effective_weight()`. (3) Not rebased onto a81d19c by me; the worktree is the current master. Nothing could be run here (no GPU/Triton): the JIT's bf16-rounding question is left to the parity gate (OFF if the stream rel L2 fails).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn only launches Triton kernels and torch.empty on shapes; branching is on shape/contiguity only; the sweep's syncs are in prepare().
- [x] Nothing derived from `data` is cached across calls: state holds only weight-derived alpha/beta/weights and the sweep choice from synthetic data.
- [x] Weights: n/a: no weights repacked or stored at another precision (alpha/beta/effective weights are fp32 [8192] derived coefficients, as SGLang's modules hold them).
- [x] Activations: branch bf16 in, next attention input bf16 out, stream fp32 as master; math fp32.
- [x] Attention sees every key: n/a: attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay (fresh torch.empty outputs each call).

okc_lint: OK with warnings (host-sync warnings all in prepare(), outside kernel_fn; one dtype warn for the uint8 L2-flush scratch buffer in prepare()); py_compile OK; __pycache__ removed.

### Implementation notes, revision (coordinator fixes)
- Stream parity is now |s - ref| / |ref - alpha*s_in| <= 4e-3 (SGLang rounds the post-norm output to bf16); output gate is <= 2 ulp OR out rel L2 <= 3e-3. Supersedes the 1e-6 / AND gate above.
- alpha/beta are probed from the SGLang modules in prepare() (stream=1, branch=0 gives alpha; stream=0, branch=1 gives beta divided by the normalized branch), cross-checked against the reference formula; > 1e-2 apart prints `[boundary] OFF: <reason>`. The cited `_dlb_coeffs` file did not exist, so the probe is written from the description. Supersedes deviation (2). Lint OK (warnings only), py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Small: the review of the held row kernel put the incumbent at ~84% of floor at first_turn; this variant targets the turn shapes' wave and tail loss. Judge on the two muse_spark_norm_residual_kernel rows vs the new kernel's rows (Triton rows reproduce within ~1%), not the headline. No overlap with in-flight work (the attention rewrites end at the o projection's input; the expert work ends before the unpooler).

## Result (after the run)
- job `46afe489e2754c83` (check), status done, passed True, tests 5/5
- speedup 1.0716x, 95% CI [1.0556, 1.0877], score 361505.54086162685 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.result.json`, `muse_prefill_layer/hypotheses/done/neutral/norm-boundaries-persistent-multirow.patch`; branch `hyp/muse_prefill_layer/norm-boundaries-persistent-multirow`
