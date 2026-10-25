---
problem: muse_prefill_layer
slug: norm-boundaries-persistent-multirow
created: 2026-10-24T224410Z
base: master
area: attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)
change: >
  Built on master (635cfee). The multi-row/persistent re-mint the review of the held
  norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program structure as the
  incumbent ... re-mint as a multi-row/persistent variant if wanted'). Replace both
  state['attn_boundary'](...) and state['ffn_boundary'](...) calls in kernel_fn with one new
  Triton kernel _boundary_persistent_kernel launched twice, at every T. Math per row, fp32
  throughout, exactly as the held hypothesis spells it out (read its change field): b =
  branch[t].float(); FFN boundary only: b *= post_ffn effective weight (1 + w); b *=
  rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b; out = s_new * rsqrt(mean(s_new^2)
  + 1e-5) * pre-norm effective weight (pre_ffn for the attention boundary, next_pre_attn for the
  FFN boundary); store s_new fp32 and out bf16 into fresh torch.empty tensors. Take alpha, beta
  and the effective weights from the SGLang modules master already builds in prepare()
  (_residual(...) after precompute_coefficients(), _norm(...) effective weights), as fp32 [8192]
  tensors, rather than re-deriving constants. STRUCTURE (the difference from the incumbent and
  from the held row kernel): grid = NUM_SMS * 2 programs (host int from
  torch.cuda.get_device_properties), each walks rows t = pid, pid + nprog, ... < T; the row is
  processed in two passes over 8192 columns in BLOCK=2048 chunks with 16-byte vector loads (pass 1
  accumulates sum(b^2) while keeping the branch chunk and stream chunk... if 8192 fp32 + 8192 bf16
  per row do not fit in registers at num_warps 8, re-read them in pass 2 from L2, which is where
  they will be), and the next row's first chunk is loaded before the current row's stores
  (software prefetch), so each SM keeps two rows in flight and there are no CTA launch waves (753
  rows was ~3-6 waves of one-row CTAs). prepare(): compare against the SGLang modules on synthetic
  inputs at T = 753 and 3012 for both boundaries (stream rel L2 <= 1e-6, bf16 output max diff <= 1
  ulp); on mismatch log '[boundary] OFF: <reason>' and keep master's modules, else 'ON'. Nothing
  else changes.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "boundary rows 20.5-23.0 us each vs a 15.4 us byte floor (67-75%): target ~17-18 us each, ~-0.8% on turn_1 shapes"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "74.7-79.6 us each vs 61.7 us floor: target ~66 us, ~-0.7%"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "~396-400 us each vs 335 us floor at ~4.0 TB/s achieved: 0-5% on the rows, ~0-0.4% per shape"
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T034755Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T034755Z
base_sha: dc91f8e5bab764e046dc86798ba9f2cc8afa6ee9
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/norm-boundaries-persistent-multirow
worktree: .worktrees/muse_prefill_layer/norm-boundaries-persistent-multirow
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T035046Z
impl_commit: b2254b60488e1eb7ff8ddec70dc5732faf43f795
submission_sha256: 099456e4ff1371e9b8d2494e7d85536ff61c9e1c299e1e6c3b62ee1490d07f7b
norm_sha256: 981b8d34f7cb8de7b232b56a65feb66bb3b80ea8d27dcaa1f0d48986a89585ad
submission_blob: 64df734f661474162d248733c3783591c8b005df
lint_warnings: 22
---

# norm-boundaries-persistent-multirow

## Change

Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program structure as the incumbent ... re-mint as a multi-row/persistent variant if wanted'). Replace both state['attn_boundary'](...) and state['ffn_boundary'](...) calls in kernel_fn with one new Triton kernel _boundary_persistent_kernel launched twice, at every T. Math per row, fp32 throughout, exactly as the held hypothesis spells it out (read its change field): b = branch[t].float(); FFN boundary only: b *= post_ffn effective weight (1 + w); b *= rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b; out = s_new * rsqrt(mean(s_new^2) + 1e-5) * pre-norm effective weight (pre_ffn for the attention boundary, next_pre_attn for the FFN boundary); store s_new fp32 and out bf16 into fresh torch.empty tensors. Take alpha, beta and the effective weights from the SGLang modules master already builds in prepare() (_residual(...) after precompute_coefficients(), _norm(...) effective weights), as fp32 [8192] tensors, rather than re-deriving constants. STRUCTURE (the difference from the incumbent and from the held row kernel): grid = NUM_SMS * 2 programs (host int from torch.cuda.get_device_properties), each walks rows t = pid, pid + nprog, ... < T; the row is processed in two passes over 8192 columns in BLOCK=2048 chunks with 16-byte vector loads (pass 1 accumulates sum(b^2) while keeping the branch chunk and stream chunk... if 8192 fp32 + 8192 bf16 per row do not fit in registers at num_warps 8, re-read them in pass 2 from L2, which is where they will be), and the next row's first chunk is loaded before the current row's stores (software prefetch), so each SM keeps two rows in flight and there are no CTA launch waves (753 rows was ~3-6 waves of one-row CTAs). prepare(): compare against the SGLang modules on synthetic inputs at T = 753 and 3012 for both boundaries (stream rel L2 <= 1e-6, bf16 output max diff <= 1 ulp); on mismatch log '[boundary] OFF: <reason>' and keep master's modules, else 'ON'. Nothing else changes.

## Expected

- `*load=turn_1*` faster: boundary rows 20.5-23.0 us each vs a 15.4 us byte floor (67-75%): target ~17-18 us each, ~-0.8% on turn_1 shapes
- `*load=turn_4*` faster: 74.7-79.6 us each vs 61.7 us floor: target ~66 us, ~-0.7%
- `*load=first_turn*` faster: ~396-400 us each vs 335 us floor at ~4.0 TB/s achieved: 0-5% on the rows, ~0-0.4% per shape

## Risk

Small: the review of the held row kernel put the incumbent at ~84% of floor at first_turn; this variant targets the turn shapes' wave and tail loss. Judge on the two muse_spark_norm_residual_kernel rows vs the new kernel's rows (Triton rows reproduce within ~1%), not the headline. No overlap with in-flight work (the attention rewrites end at the o projection's input; the expert work ends before the unpooler).

## Adversarial review

Reviewed 2026-10-25T034755Z by an adversarial subagent (model opus).

**Verdict: proceed.** Rules-clean and readable from the two boundary rows, but the persistent design does not fix the problem it names. At T=753 with 264 programs, every program still walks ceil(753/264)=3 rows, so the tail is the same three row-times as the incumbent's waves. Chunked two-pass rows also keep fewer bytes in flight than a whole row in registers. Proceed, with a prepare-time sweep that also includes a single-wave one-row-per-program variant (num_warps 4, every row resident at once), with the parity gate relaxed to max 2 ulp plus rel L2, and on a81d19c.

### Justification
The floors and incumbent rows are those the held row-kernel's review verified: 12 B/elem x 8192 x T at 4.8 TB/s gives 15.4/61.7/335 us, against incumbent 20.5-23.0 / 74.7-79.6 / 396-400 us per boundary. The first_turn incumbent is already about 84% of the floor (about 4.0 TB/s, near practical HBM streaming), so the realistic gains are at turn_1 and turn_4: a few us per row and about 0.2-0.8% per shape, below headline drift. Judge on the rows (muse_spark_norm_residual_kernel x2 replaced by _boundary_persistent_kernel x2). The mechanism claim is 'no CTA launch waves (753 rows was ~3-6 waves)'. But the work unit is still one 96 KB row and the grid is 2 x SM = 264. Rows per program are 2 or 3, so the critical path is 3 row-times, the same quantization as 753 one-row CTAs at 2 per SM (2.85 waves). On H100, CTA launch costs well under 1 us per wave, so persistence removes almost nothing at T=753. What would fix the turn_1 tail is putting all 753 rows in flight at once: about 72 MB in about 15 us needs roughly 36+ KB in flight per SM, which a single wave of small one-row programs (num_warps 4, 6+ CTAs/SM) gives. Bytes in flight: chunking each row into 4 x 2048 columns with two passes means each program has about 12 KB (bf16 branch + fp32 stream chunk) outstanding, plus one prefetched chunk, about 24-48 KB per SM at 2 programs. The incumbent issues the whole 96 KB row at once. The persistent variant could therefore be slower at first_turn, where the incumbent is already near the floor. The alpha/beta/effective-weight vectors (fp32 [8192] x 3, 96 KB) cannot stay in registers across rows when rows are chunked, so the multi-row 'weights loaded once' advantage the earlier review pointed to does not happen; they are re-read from L2 every row, as in the incumbent. WEAKEST LINK: equal-sized row units mean a persistent grid cannot beat wave quantization at T=753. The win, if any, comes from the software prefetch overlapping one row's stores with the next row's loads, which is worth at most about one row-time out of three.

### Confounders on the current kernel
- stack-tonight-five-prefill (landed, current master a81d19c): the spec says 'master (635cfee)', which is b92a0f4's submission blob. The stack does not touch either boundary (router GEMM, align, w2 zero-store, Marlin silu-mul), but it changes what runs just before ffn_boundary (expert output path) and so the L2 state of the branch input. Rebase on a81d19c and compare the rows against a81d19c's profile (c4d14d62f11e4aaf / 502f2eab95164781), not b92a0f4's.
- sglang-pipeline-parity: it defines both boundaries as SGLang's FusedPostNormResidualPreNorm. The prepare() parity gate is against those same modules, which is the baseline, not a confound.
- norm-residual-boundaries-triton-row-kernel (held, not run): the same math in a one-row-per-program structure. Since it never ran, this run's rows are compared directly with the JIT incumbent, and no stacking ambiguity arises.

### Interference and A/B needs
No companion run; the replaced rows are directly readable and nothing landed shares this launch or numerical path. The one-row-vs-persistent question belongs inside the run as a prepare-time sweep, not as a second check: {persistent 2 x SM, num_warps 8} vs {one row per program, num_warps 4, grid T} vs {one row per program, num_warps 8}. Time each with graph replays at T=753 and 3012 (and once at 16384), print '[boundary] variant ... us', pick per T band from shape only, and fall back to master's modules if nothing beats them by 2%. Prepare timings rank correctly even though they undercount (LESSONS 2026-10-25).

### Open questions
- At T=753, does a single-wave one-row-per-program variant (all rows in flight) beat both the persistent kernel and the JIT incumbent? That is the actual tail fix.
- Does the chunked two-pass persistent kernel lose at first_turn, where the incumbent streams at about 4.0 TB/s? If it does, gate it to T <= 4096.
- Does the JIT kernel round the normalized branch to bf16 before the residual combine? If so, the fp32-throughout math fails the stream rel L2 1e-6 gate. Check the sglang fused_norm_residual source before building.


## Implementation notes

Only `muse_prefill_layer/submission.py` changed (+242/-3). Added before `prepare()`: `_boundary_rows_kernel` (one row per program, whole 8192 row in registers, grid T), `_boundary_persistent_kernel` (grid min(T, 2 x SMs), row loop with the next row's branch/stream loads issued before the current row's math and stores), a shared `_boundary_row` math helper (fp32 throughout: optional post-FFN (1+w) multiply, rsqrt(mean+1e-8), alpha/beta combine, rsqrt(mean+1e-5) x pre-norm weight; s_new fp32 and out bf16 into torch.empty), `_boundary_launch`, and the prepare-time parity gate and sweep. `prepare()` ends with `state["bnd"] = _boundary_prepare(...)`; `kernel_fn` calls `_boundary(state, "attn"|"ffn", ...)`, which uses the T-class choice (T<1536 -> 753, <8192 -> 3012, else 16384; a shape-only lookup) or master's SGLang modules unchanged.

Per the review (which wins):
- Sweep per T class (753, 3012, 16384) on synthetic data: master vs {rows nw4 (single wave, every row resident), rows nw8, persistent nw8}. Each is CUDA-graph timed (attn + ffn boundary per graph, 256 MB L2 flush before each replay, median of 25). Prints `[boundary] variant ... us` and `PATH CHOSEN` per class and overall. A variant is picked only if <= 0.98x master; otherwise master's modules. Any exception means `[boundary] OFF: <reason>`.
- Parity gate relaxed: stream rel L2 <= 1e-6, out max <= 2 bf16 ulp AND out rel L2 <= 3e-3, both boundaries, at every swept T, against the SGLang modules; a failing variant is skipped (logged), not silently chosen.

Deviations: (1) persistent kernel keeps the whole row as one 8192 block rather than 4 x 2048 two-pass chunks, because the review showed chunking cuts bytes in flight and gives no win; prefetch is of the next whole row. (2) alpha/beta are computed in prepare() from the gate weights with the reference's formula (temperature 0.3, eps 1e-3) because the SGLang module's coefficient attribute names could not be inspected (no sglang on this node); the parity gate catches any mismatch. Effective weights come from the modules' `_get_effective_weight()`. (3) Not rebased onto a81d19c by me; the worktree is the current master. Nothing could be run here (no GPU/Triton): the JIT's bf16-rounding question is left to the parity gate (OFF if the stream rel L2 fails).

## Rules check

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
