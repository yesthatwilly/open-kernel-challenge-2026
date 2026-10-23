---
problem: muse_decode_layer
slug: sglang-pipeline-parity-router-hilo-m8
created: 2026-10-23T225318Z
base: master
area: router GEMM
change: >
  Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py
  composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the
  FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules,
  the cos|sin table, the router hi/lo split and the side stream; kernel_fn runs
  _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler, the FP8 k/v casts plus
  store_cache, flash_attn_with_kvcache (FA3, scheduler_metadata passed through on full decode),
  the two boundary JIT kernels, the router on the side stream, moe_align_block_size plus the two
  moe_wna16_marlin_gemm calls, silu_and_mul and fused_post_expert_norm_weighted_sum. ONE
  deliberate structural change on top of parity, in the router GEMM: lower ROUTER_FP32_MAX_TOKENS
  from 8 to 1, so the fp32 F.linear router (a SIMT sm80_xmma f32 GEMM at m=8, measured at 26.2 us
  in the muse_bf16_proj review versus ~15 us for the hi/lo bf16 pair) only serves m=1 (where it is
  a gemv) and every m>=8 shape runs the two torch.mm(..., out_dtype=float32) hi/lo GEMMs;
  consequently attn_boundary's router_dtype is None at m=8 (one fewer fp32 output written). The
  rules name the hi/lo pair with fp32 accumulation as allowed at any batch size. Everything else
  stays byte-for-byte SGLang's path. Attribution: load=verify_1 shapes (T=8) carry the router
  change; all other shapes are pure parity and must land at ~1.0x, which becomes the base every
  later muse_decode_layer hypothesis stacks on.
expected:
  - shapes: "*load=verify_1*"
    direction: faster
    why: "T=8: the fp32 SIMT router GEMM (~26 us on the side stream, longer than the ~15-20 us pooler GEMM it overlaps) becomes two bf16 tensor-core GEMMs (~15 us) that hide under the pooler; expect 2-4% on these two shapes"
  - shapes: "*load=decode_1*"
    direction: flat
    why: T=1 stays on the fp32 F.linear gemv exactly as SGLang; pure parity, control
  - shapes: "*load=decode_16*"
    direction: flat
    why: T=16 was already on the hi/lo path in SGLang; pure parity
  - shapes: "*load=decode_32*"
    direction: flat
    why: T=32 already hi/lo in SGLang; pure parity
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 already hi/lo; pure parity
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 already hi/lo; pure parity
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-23T230349Z
review_companions: null
stage: claimed
claimed_at: 2026-10-23T230349Z
base_sha: c33a496b7fcb0965469bd5d5a5abc1b4f6da76ab
base_blob: 0a0fc36b7854c07f5047c1df4e08ee984a830651
branch: hyp/muse_decode_layer/sglang-pipeline-parity-router-hilo-m8
worktree: .worktrees/muse_decode_layer/sglang-pipeline-parity-router-hilo-m8
mode: null
---

# sglang-pipeline-parity-router-hilo-m8

## Change

Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_layer_for_marlin, the FusedQKNorm / FusedRMSNormGate / FusedPostNormResidualPreNorm / FusedMuseSparkRouting modules, the cos|sin table, the router hi/lo split and the side stream; kernel_fn runs _bf16_gemm_dispatch_impl for qkv_gate, o, pooler and unpooler, the FP8 k/v casts plus store_cache, flash_attn_with_kvcache (FA3, scheduler_metadata passed through on full decode), the two boundary JIT kernels, the router on the side stream, moe_align_block_size plus the two moe_wna16_marlin_gemm calls, silu_and_mul and fused_post_expert_norm_weighted_sum. ONE deliberate structural change on top of parity, in the router GEMM: lower ROUTER_FP32_MAX_TOKENS from 8 to 1, so the fp32 F.linear router (a SIMT sm80_xmma f32 GEMM at m=8, measured at 26.2 us in the muse_bf16_proj review versus ~15 us for the hi/lo bf16 pair) only serves m=1 (where it is a gemv) and every m>=8 shape runs the two torch.mm(..., out_dtype=float32) hi/lo GEMMs; consequently attn_boundary's router_dtype is None at m=8 (one fewer fp32 output written). The rules name the hi/lo pair with fp32 accumulation as allowed at any batch size. Everything else stays byte-for-byte SGLang's path. Attribution: load=verify_1 shapes (T=8) carry the router change; all other shapes are pure parity and must land at ~1.0x, which becomes the base every later muse_decode_layer hypothesis stacks on.

## Expected

- `*load=verify_1*` faster: T=8: the fp32 SIMT router GEMM (~26 us on the side stream, longer than the ~15-20 us pooler GEMM it overlaps) becomes two bf16 tensor-core GEMMs (~15 us) that hide under the pooler; expect 2-4% on these two shapes
- `*load=decode_1*` flat: T=1 stays on the fp32 F.linear gemv exactly as SGLang; pure parity, control
- `*load=decode_16*` flat: T=16 was already on the hi/lo path in SGLang; pure parity
- `*load=decode_32*` flat: T=32 already hi/lo in SGLang; pure parity
- `*load=verify_16*` flat: T=128 already hi/lo; pure parity
- `*load=verify_32*` flat: T=256 already hi/lo; pure parity

## Risk

Risk: this is SGLang's own code path plus a one-constant dispatch change the rules explicitly permit, so correctness risk is near zero (routing ties are judged with 0.03-0.05 margins, far above the hi/lo split's ~2^-16 relative error). The expected gain on verify_1 is small (2-4%) and could be inside the +-1.5% CI if the router already hid under the pooler at m=8; a flat verify_1 result still leaves the parity base intact. The commit step's normalized-AST dedup must see a difference from baseline.py: the threshold constant and the resulting router_dtype branch are that difference; do not drop them. Implementation must keep the file self-contained (no import of reference.py), keep the side stream join (main_stream.wait_stream) before local_ids, and pass scheduler_metadata through unchanged on full decode. If the judge image's import paths differ from baseline.py's, the run fails at import, which is a rejection without timing cost.

## Adversarial review

Reviewed 2026-10-23T230349Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** SGLang parity plus a single rules-sanctioned dispatch change (fp32 router only at T=1) is a genuine, quantitatively grounded mechanism whose only signal is on the two verify_1 (T=8) shapes; nothing has landed to confound it and the judge's same-run per-shape baseline timing is the parity control, so no companion run is warranted.

### Justification
A. Mechanism: in baseline.py the router runs on a side stream forked before the pooler GEMM and joined right after it, so at T=8 the critical path is max(pooler, fp32 router + routing JIT). Measured in isolation on muse_bf16_proj (job 017804b34fed40b3, cold L2): fp32 F.linear router at m=8 = 26.1 us (two sm80_xmma f32 SIMT kernels 9.3 + 8.2 us plus a 2.3 us elementwise), pooler at m=8 = 21.3 us (nvjet 128x8 splitK 18.9 + reduce 1.6), hi/lo bf16 router = 14.5 us at m=16, 15.1 at m=32, 16.6 at m=128, 17.8 at m=256 (nvjet splitK + splitKreduce + one add). So today the side branch (~26 + ~3 us routing) overruns the pooler by ~8-10 us; after the change the side branch (~14-15 + ~3) hides under the pooler. Expected saving ~8-10 us on shapes SGLang runs at 267.9/270.7 us, i.e. ~3%, matching the stated 2-4%. verify_1 is one of six equally weighted loads, so the overall geomean moves only ~0.5%, inside the +-1.5% CI; the attribution must come from the per-shape rows (per-shape err was ~0.3% on bf16_proj), which the result JSON provides. The expected table is consistent with TASK.md: only load=verify_1 has T=8; decode_1 (T=1) stays on the fp32 gemv (10.4 us, already hidden under the 21 us pooler, so correctly 'flat'); decode_16/32 and verify_16/32 (T=16..256) were already hi/lo, correctly 'flat'. The profile in FINDINGS is of the untouched starter (0.0045x) and says nothing about SGLang's pipeline; all the supporting numbers are borrowed from bf16_proj, measured in isolation, not concurrently. Numerics: the reference's router input is the bf16-rounded h (starter: h.to(bfloat16) then h.float() @ router.T), which is exactly the ffn_input the hi/lo path consumes, so the only deviation is the weight split (~2^-17 relative) and fp32 accumulation order, far below the 0.03-0.05 routing margins. Dedup judgement: the change alters which kernels execute at T=8 (SIMT fp32 GEMM -> tensor-core bf16 pair) with a falsifiable per-shape prediction; it is a genuine mechanism, not a cosmetic bypass, although it is the smallest possible one. Weakest link: the assumption that the two streams overlap in the captured graph as cleanly as the isolated numbers imply. The 26 us SIMT GEMM uses few CTAs (n=256, 32x32 tiles) and should co-run with the pooler, but if the fp32 router is already partly hidden or the pooler stretches under contention, the gain shrinks toward zero; a flat verify_1 would be uninformative about whether the router was on the critical path at all, though it still leaves the parity base intact.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
Nothing has landed on master (landed list empty; master is the plain-PyTorch starter), so there is no landed hypothesis to revert or re-tune. The natural control, pure SGLang parity with ROUTER_FP32_MAX_TOKENS=8, is exactly what the judge already times in the same run as baseline_ns for every shape, with paired, order-swapped ratios; the ten non-T=8 shapes landing at ~1.0x validate that the parity port is faithful, and the two verify_1 shapes' ratio against that same-run baseline isolates the router change. A gain on verify_1 with the other ten at ~1.0x is unambiguous; a loss on verify_1 with the other ten at ~1.0x is also attributable to the router change alone (only one constant differs). The only ambiguous outcome is the ten parity shapes not landing at ~1.0x, and a control run could not disambiguate that any better than the run's own per-shape rows (a `control` companion would be AST-identical to baseline.py and measure what baseline_ns already reports). No companion.

### Open questions
- Does the m=8 hi/lo torch.mm(..., out_dtype=float32) pick a kernel pair comparable to m=16's ~14.5 us? There is no m=8 hi/lo measurement anywhere; if cuBLAS picks a poor shape at m=8 the side branch could still overrun the pooler.
- Does the fp32 SIMT router actually extend the critical path when co-scheduled with the pooler's splitK kernel inside a replayed graph, or is it already mostly hidden? Only the verify_1 per-shape rows of this run will tell; a flat result cannot distinguish 'already hidden' from 'hi/lo no faster at m=8'.
- Will the ten pure-parity shapes land at ~1.00x? Any systematic offset there (harness-side differences in state construction, stream creation or graph capture) would undermine the base every queued dependent stacks on and should be investigated before dispatching dependents.
- Follow-up, not a companion: at T=1 the fp32 gemv path costs 10.4 us (three kernels) on the side stream but is hidden under the 21 us pooler, so extending hi/lo to T=1 is not expected to pay; the hypothesis is right to leave decode_1 untouched.

