---
problem: muse_prefill_layer
slug: norm-residual-boundaries-triton-row-kernel
created: 2026-10-24T165356Z
base: master
area: attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2)
change: >
  Built on master (635cfee). Replace both FusedPostNormResidualPreNorm calls in kernel_fn
  (state['attn_boundary'] after the o projection, state['ffn_boundary'] after the unpooler;
  SGLang's JIT muse_spark_norm_residual_kernel<8192, 512, ...>) with one new Triton kernel
  _norm_residual_boundary_kernel, launched twice, at every T. Per row t (one program per row,
  BLOCK = 8192 = the whole row in registers, num_warps 16; also try 8 in prepare and keep the
  faster by graph-replay timing on synthetic [3012, 8192] inputs), with fp32 math throughout: b =
  branch[t].float() (bf16 [T, 8192]: attn_output or ffn_output); for the FFN boundary first b = b
  * (1 + post_ffn_norm) (MuseSpark1xRMSNormWithInputScale's input scale; the attention boundary's
  post-norm has no weight); b = b * rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b
  (stream fp32 [T, 8192]); out = (s_new * rsqrt(mean(s_new^2) + 1e-5)) * (1 + w_pre) with w_pre =
  pre_ffn_norm (attention boundary) or next_pre_attn_norm (FFN boundary), stored bf16; s_new
  stored fp32. alpha, beta (fp32 [8192]) are precomputed in prepare() per
  reference.residual_coefficients: w = gate / 0.3, beta = sigmoid(w), alpha =
  sqrt(clamp(sigmoid(-w) * (1 + beta), min=RESIDUAL_GATE_EPS)); take them, and the effective (1 +
  w) norm weights and eps values, from the SGLang modules master already builds
  (_residual(...).precompute_coefficients(), _norm(...)._get_effective_weight() / effective weight
  attributes) so the arithmetic matches SGLang's rather than re-deriving constants; prepare()
  asserts the Triton outputs match the JIT kernel's on synthetic inputs at T=753 (stream rel L2 <=
  1e-6, bf16 output max ulp diff <= 1) for both boundaries before kernel_fn may use it, and falls
  back to master's modules if not. Outputs are fresh tensors (torch.empty) of the same
  dtypes/shapes; the router_dtype=None / branch_dtype=bf16 options master passes are the only ones
  reproduced. Everything else (GEMMs, attention, experts) is unchanged. Mechanism: each boundary
  moves 12 B per element (fp32 stream in + out, bf16 branch in, bf16 normed out), floor 15.4 us at
  T=753, 61.7 us at 3012, 335 us at 16384 at 4.8 TB/s; master measures 23.0/20.5 us at turn_1
  (67-75%), 78.0/74.7 at turn_4 (~80%), 398/396 at first_turn (84%), and the branch input was just
  written by the GEMM and is partly L2-resident, so the DRAM floor is lower still. A single-pass
  whole-row-in-registers kernel with 16 B vector loads should run at ~90% of bandwidth on all
  three T.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: two boundary kernels 23.0 + 20.5 us vs 2 x 15.4 floor; target ~2 x 17 us, ~8-10 us of 1080-1420 (~0.7%)
  - shapes: "*load=turn_4*"
    direction: faster
    why: 78.0 + 74.7 us vs 2 x 61.7 floor; target ~2 x 66, ~20 us of 3335-4938 (~0.5%)
  - shapes: "*load=first_turn*"
    direction: faster
    why: 398 + 396 us vs 2 x 335 floor; target ~2 x 365, ~60 us of 14275-22333 (~0.3-0.4%)
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T165634Z
review_companions: null
hold_reason: "reviewer: same one-row-per-program structure as the incumbent; expect 0.2-0.4% headline, below run noise; re-mint as a multi-row/persistent variant if wanted"
outcome: expired
verdict_detail: "hold expired after 28 h: reviewer: same one-row-per-program structure as the incumbent; expect 0.2-0.4% headline, below run noise; re-mint as a m"
expired_at: 2026-10-25T211329Z
---

# norm-residual-boundaries-triton-row-kernel

## Change

Built on master (635cfee). Replace both FusedPostNormResidualPreNorm calls in kernel_fn (state['attn_boundary'] after the o projection, state['ffn_boundary'] after the unpooler; SGLang's JIT muse_spark_norm_residual_kernel<8192, 512, ...>) with one new Triton kernel _norm_residual_boundary_kernel, launched twice, at every T. Per row t (one program per row, BLOCK = 8192 = the whole row in registers, num_warps 16; also try 8 in prepare and keep the faster by graph-replay timing on synthetic [3012, 8192] inputs), with fp32 math throughout: b = branch[t].float() (bf16 [T, 8192]: attn_output or ffn_output); for the FFN boundary first b = b * (1 + post_ffn_norm) (MuseSpark1xRMSNormWithInputScale's input scale; the attention boundary's post-norm has no weight); b = b * rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b (stream fp32 [T, 8192]); out = (s_new * rsqrt(mean(s_new^2) + 1e-5)) * (1 + w_pre) with w_pre = pre_ffn_norm (attention boundary) or next_pre_attn_norm (FFN boundary), stored bf16; s_new stored fp32. alpha, beta (fp32 [8192]) are precomputed in prepare() per reference.residual_coefficients: w = gate / 0.3, beta = sigmoid(w), alpha = sqrt(clamp(sigmoid(-w) * (1 + beta), min=RESIDUAL_GATE_EPS)); take them, and the effective (1 + w) norm weights and eps values, from the SGLang modules master already builds (_residual(...).precompute_coefficients(), _norm(...)._get_effective_weight() / effective weight attributes) so the arithmetic matches SGLang's rather than re-deriving constants; prepare() asserts the Triton outputs match the JIT kernel's on synthetic inputs at T=753 (stream rel L2 <= 1e-6, bf16 output max ulp diff <= 1) for both boundaries before kernel_fn may use it, and falls back to master's modules if not. Outputs are fresh tensors (torch.empty) of the same dtypes/shapes; the router_dtype=None / branch_dtype=bf16 options master passes are the only ones reproduced. Everything else (GEMMs, attention, experts) is unchanged. Mechanism: each boundary moves 12 B per element (fp32 stream in + out, bf16 branch in, bf16 normed out), floor 15.4 us at T=753, 61.7 us at 3012, 335 us at 16384 at 4.8 TB/s; master measures 23.0/20.5 us at turn_1 (67-75%), 78.0/74.7 at turn_4 (~80%), 398/396 at first_turn (84%), and the branch input was just written by the GEMM and is partly L2-resident, so the DRAM floor is lower still. A single-pass whole-row-in-registers kernel with 16 B vector loads should run at ~90% of bandwidth on all three T.

## Expected

- `*load=turn_1*` faster: two boundary kernels 23.0 + 20.5 us vs 2 x 15.4 floor; target ~2 x 17 us, ~8-10 us of 1080-1420 (~0.7%)
- `*load=turn_4*` faster: 78.0 + 74.7 us vs 2 x 61.7 floor; target ~2 x 66, ~20 us of 3335-4938 (~0.5%)
- `*load=first_turn*` faster: 398 + 396 us vs 2 x 335 floor; target ~2 x 365, ~60 us of 14275-22333 (~0.3-0.4%)

## Risk

Small (~0.4-0.7% per load), so judge it on the two muse_spark_norm_residual_kernel rows being replaced by faster _norm_residual_boundary_kernel rows, not the headline. Risks: (1) the JIT kernel may already be near a practical bandwidth ceiling at T=16384, leaving only turn_1 to move; (2) exact semantics of MuseSpark1xRMSNormWithInputScale / NormPreservingResidualConnection (order of the input scale vs the post-norm, eps placement) must match SGLang, which is why the coefficients come from the SGLang modules and prepare() checks against the JIT kernel before use; (3) whole-row registers at 8192 fp32 per row with 16 warps is 16 values per thread for each of stream and branch, fine for occupancy. Correctness bounds are generous (3-15% rel L2) but the residual stream feeds the next layer, so the prepare() parity check is strict. No caching, no syncs in kernel_fn, bf16 where SGLang keeps bf16 and fp32 for the stream.

## Adversarial review

Reviewed 2026-10-24T165634Z by an adversarial subagent (model opus).

**Verdict: proceed.** Rules-clean, correctly sized and well gated: the two muse_spark_norm_residual_kernel rows are measured (23.0/20.5, 78.0/74.7, ~398/396 us) and the byte floors check out. But the proposed kernel is structurally the incumbent (one 8192-wide row per CTA, 512 threads), so the realistic win is a few us per boundary, concentrated at turn_1 and turn_4, and close to nothing at first_turn. Judge it only on the two replaced rows. The headline move (~0.2-0.4%) is far below the 1.5-2.5% drift floor.

### Justification
Floors verified: 12 B/elem x 8192 x T at 4.8 TB/s gives 15.4 us (T=753), 61.7 us (3012) and 335 us (16384). Master's profile fb1e52d2119d4175 matches the hypothesis's numbers exactly: turn_1 23.0/20.5 (22.1/20.5 full), turn_4 78.0/74.7 (77.1/74.9), first_turn 395.8-398.6 us per boundary. Expected directions (all faster, small) agree with TASK.md weights. Shape-level savings: turn_1 ~5-9 us of 1080-1420, turn_4 ~10-20 us of 3335-4938, first_turn ~0-40 us of 14-22 ms. Each load's gain is <=0.7%, so the geomean moves ~0.2-0.4%, which is unreadable on the headline (LESSONS: 1.5-2.5% drift per shape). Triton row times reproduce within ~1%, so the replaced rows are readable. Mechanism caveats: (1) At first_turn the JIT kernel already sustains ~4.04 TB/s effective, and H200 practical streaming is ~4.0-4.3 TB/s, so the stated 90%-of-4.8 (~365 us) target is optimistic. Expect 0-5% on those rows. (2) The first boundary is slower than the second at every T (23.0 vs 20.5, 78 vs 75) because attn_boundary reads data['stream'] from a cold L2 while the branch input is partly L2-resident after the GEMM. That residency is the same for both kernels, so it is not a lever for the new one. (3) Where is the turn_1 gap (67-75% of floor) actually spent? Probably wave quantization and tail: 753 one-row CTAs on 132 SMs at 1-2 CTAs/SM of 512 threads is ~3-6 waves, plus launch ramp. The proposed design copies that structure. WEAKEST LINK: nothing in the change differs structurally from the incumbent JIT kernel. The same grid (one row per program), the same thread count (16 warps = 512 threads) and the same whole-row-in-registers approach give no a-priori reason to beat a hand-written CUDA kernel. A further cost: each program reloads alpha, beta, (1+w_pre) and (1+post_ffn) as fp32 [8192], 96-128 KB of L2 reads per row, against 96 KB of DRAM traffic per row. The levers that could actually beat the incumbent are absent: several rows per program or a persistent grid, so the weight vectors load once into registers and the tail shrinks at T=753, and the num_warps 8/16 sweep with 1-2 CTAs/SM. Recommend the implementer add a ROWS_PER_PROGRAM (1, 2, 4) or persistent-grid option to the prepare-time sweep, at no extra run cost.

### Confounders on the current kernel
- sglang-pipeline-parity: established both boundaries as SGLang's FusedPostNormResidualPreNorm with router_dtype=None/branch_dtype=bf16. The change replaces exactly those calls, and the prepare() parity check is against those same modules. That is the baseline being compared, not a confound.
- experts-triton-w4a16-first-turn-blockt256 and experts-hybrid-marlin-turn-triton-first-turn: they change only the expert GEMM path. The unpooler GEMM still sits between the experts and ffn_boundary, so the branch tensor and its L2 residency are unchanged. Neither touches the stream or the attention boundary. No plausible interaction.

### Interference and A/B needs
No companion is warranted. The change is self-contained, and the result is read from the two replaced rows in this run's profile: muse_spark_norm_residual_kernel x2 disappears and _norm_residual_boundary_kernel x2 appears, per shape, against master's known numbers. Triton kernel times reproduce within ~1%, so a 2-5 us move per row is resolvable at turn_1 and turn_4. No landed hypothesis overlaps this launch sequence or numerical path, so an ab-revert would re-measure unrelated code. The JIT-kernel control is master itself (fb1e52d2119d4175). Checks are spent and runs go out as scored submits, so spending a second run on an expected ~0.3% headline is not justified. One requirement makes the single run interpretable: prepare() must print which path it chose (Triton or the fallback to master's modules) and the chosen num_warps (and ROWS_PER_PROGRAM if added). The profile kernel names also show it, but a silent fallback would otherwise read as a null result.

### Open questions
- The bf16 max-ulp <= 1 parity gate on synthetic T=753 inputs: a different reduction order can flip the rounding of a few elements by 1 ulp. If any row's sum-of-squares ordering yields a 2-ulp difference, the gate silently falls back to master, which is a wasted scored submit. Consider gating on max ulp <= 2 plus a rel-L2 bound (e.g. <= 1e-5) on the bf16 output, and print the observed max ulp and rel L2 in the log.
- Does the JIT kernel compute the post-norm of the branch fully in fp32, or round any intermediate (e.g. the normalized branch) to bf16 before the residual combine? The stream rel-L2 <= 1e-6 gate would catch a mismatch (bf16 rounding gives ~1e-3 on the beta*b term), but only after the change was built. Check the JIT source (sglang muse_spark_v12/fused_norm_residual) first.
- MuseSpark1xRMSNormWithInputScale with is_post_norm=True and post_norm_gain_center_type='one': confirm that the input scale (1+post_ffn) is the only weight applied, and that there is no post-normalization gain. The hypothesis asserts this, and the parity check will catch it if it is wrong.
- Would multi-row programs (2-4 rows per CTA, weights loaded once) or a persistent grid be the actual win at T=753? If the single-row variant lands within 1 us of the JIT kernel at turn_1, that is the next stacked hypothesis.

