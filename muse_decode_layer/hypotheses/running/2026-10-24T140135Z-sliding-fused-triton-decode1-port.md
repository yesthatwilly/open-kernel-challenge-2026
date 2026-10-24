---
problem: muse_decode_layer
slug: sliding-fused-triton-decode1-port
created: 2026-10-24T140135Z
base: master
area: FA3 call
change: >
  Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding
  decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from
  muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers
  (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its
  _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In
  kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if
  cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch
  (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE =
  bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed),
  data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'],
  data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated
  attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That
  one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way
  split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x
  sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q
  FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape
  (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path
  byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula,
  as in the attention problem (decode's cos_sin table stays for the other shapes).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=decode_32*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: gated out
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T164533Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T164533Z
base_sha: d170c1fd1af3e71b7de6f970d623da8b2a86af59
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/sliding-fused-triton-decode1-port
worktree: .worktrees/muse_decode_layer/sliding-fused-triton-decode1-port
mode: null
---

# sliding-fused-triton-decode1-port

## Change

Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE = bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed), data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'], data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula, as in the attention problem (decode's cos_sin table stays for the other shapes).

## Expected

- `*layer=sliding load=decode_1 *` faster: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
- `*layer=full*` flat: untouched
- `*layer=sliding load=decode_16*` flat: gated out
- `*layer=sliding load=decode_32*` flat: gated out
- `*layer=sliding load=verify_*` flat: gated out

## Risk

Single-shape gain (~+0.4-0.6% geomean) but on the heaviest-weighted shape of a launch-bound load, and the kernel is proven correct and faster on identical inputs in the sibling. Caveat: decode's fused-preattention-triton (neutral) showed that fusing pre-attention launches alone did not move sliding decode_1 here, so the gain may be only the attention+norm-gate part (~3.5 us, +2%). Correctness risk is in the I/O mapping: decode's qkv is [T, 2304] contiguous from the GEMM (attention's was [B, L, 2304]); the fused kernel's RoPE must match FusedQKNorm's interleaved convention (it passed the attention problem's tests with the same SGLang reference). The 13 tests include batch 1 ctx 10 sliding (q_len 1), which takes this path at small ctx and its page size. Composes with post-expert-norm-triton-small-t and the bf16 GEMM port (disjoint kernels on the same shape).

## Adversarial review

Reviewed 2026-10-24T164533Z by an adversarial subagent (model opus).

**Verdict: proceed.** A single-shape, config-gated port of a kernel that is proven correct and faster on the same inputs in muse_attention_decode. It does not interact with anything that has landed. The gain is real but small: about 2-4% on sliding decode_1 and roughly +0.3-0.5% on the geomean, which falls inside the CI. So judge it by the sliding decode_1 row and top_kernels, not by the headline.

### Justification
Mechanism: on sliding decode_1 master runs, after the qkv GEMM, FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3 (no scheduler_metadata on sliding, so FA3 schedules internally, then runs main and combine) and FusedRMSNormGate. That is about 8-9 serialized graph nodes, and the change collapses them into one 32-CTA launch (1 x 2 x 16). FA3 main alone is 10.2 us of the 145.8 us shape (FINDINGS). The attention problem measured exactly these replaced launches going from 16.56 to 13.12 us, although its baseline already had the fused pre-attention kernel. The I/O mapping checks out against muse_attention_decode/submission.py on master. The qkv row layout (q 16x64, k 2x64, v 2x64, gate 16x64) is identical. decode's qkv is the [1, 2304] contiguous GEMM output, so passing stride(0)=2304 is correct. The positions, cache_loc, seq_lens, page_table, k_scale and v_scale fields have the same names, dtypes and semantics. The caches are [pages, page_size, 2, 64] in both problems. The window is lo = seq - 2047 old keys plus the new key, which equals FA3's (2047, 0). The output is the [T, 1024] bf16 that _gemm(o, dense['o']) expects. inv_freq is built from the same formula as decode's prepare() cos_sin table (theta 500000, arange(0,64,2)/64). Numerically the fused path is finer than master: the query is bf16 rather than E4M3, and the norm-gate is computed on fp32 o before any bf16 rounding. Numbers: the hypothesis quotes 172.8 us for the shape, but that is stale. Master now runs sliding decode_1 at 145.8 us, so 4-8 us would be +2.7-5.5% on the shape. The decode_1 load is 46*145.8 + 16*157.0 = 9219 us-weight, and 4 us there is about 2% of that load, or about 0.33% on the 6-load geomean. That is consistent with the stated direction but below the stated +0.4-0.6%, and well below the ±1.5% CI and the 1.5-2.5% per-shape run drift (LESSONS). Weakest link: the decode_layer finding fused-preattention-triton (neutral) showed that removing the five pre-attention launches did not move sliding decode_1 here. So the claim that the extra pre-attention launches are worth 4-8 us is unsupported. The defensible gain is only the attention-plus-norm-gate part, about 3-3.5 us (+2-2.4% on the shape), which is barely above drift on one row. The run is still readable from the profile: the cutlass flash_fwd row (10.2 us), the FA3 combine, FusedQKNorm, store_cache, the cast kernels and FusedRMSNormGate should disappear from sliding decode_1's top_kernels, replaced by one _sliding_fused_kernel row. That row should be at or below the attention problem's ~9-10 us kernel time.

### Confounders on the current kernel
- None of the four landed hypotheses confounds this change.
- fa3-hdim128-head-pairing-batched: gated to window == 0 and batch > 1, so it never runs on sliding decode_1. The new gate is checked before the paired/unpaired branch and returns o before reaching it.
- post-expert-norm-triton-small-t: active at T=1 on the same shape, but it acts on downstream, disjoint kernels (the MoE combine). Every replay starts with a cold L2, so no cache interaction carries across. The effects add.
- marlin-block-m-cap-16: inert at T=1 (block_size_m 8 is chosen either way).
- sglang-pipeline-parity-router-hilo-m8: T=1 still takes the fp32 F.linear router path, so it is inert for this shape.

### Interference and A/B needs
No companion is warranted. The change is a config-only branch touching exactly one scored shape, and no landed hypothesis acts on that shape's attention block. A gain or a loss is therefore attributable from the sliding decode_1 row's paired ratio together with its top_kernels. Every other row is byte-identical code and works as a drift control: if they move uniformly by +/-2%, that is run drift. A pre-attention-only control (fused attention kernel plus master's FusedQKNorm/store_cache) would not change the decision, because fused-preattention-triton already measured that half as flat here. An ab-revert of any landed item would be uninformative, since none of them overlaps. The headline will be within CI no matter what, so the decision to merge should rest on the sliding decode_1 row being above about +2.5% and the replaced kernels being gone from the profile, not on CI separation.

### Open questions
- Scored-submit correctness. The only test that takes the new path is 'batch 1, ctx 10, layer sliding', at the test default page size (probably 64, which the PAGE constexpr handles), plus the scored batch-1 ctx-80000 sliding decode on every timed call. Splits 1..15 are empty at ctx 10 and are masked to -inf partials; the combine gives them weight 0. Can the implementer confirm the attention problem's test list contains the same batch-1 ctx-10 sliding case, so that this exact path is already proven at small ctx?
- The hypothesis's 172.8 us baseline and +0.4-0.6% geomean estimate are stale against master's 145.8 us row. Expect about +0.3% on the geomean. Should the harvest classify on the per-shape row rather than the headline? A neutral classification here would be expected even if the change is a real win.
- bf16-shape-gated-triton-gemms-small-t (queued) replaces the qkv GEMM at m=1. If it lands first, its output must stay a [1, 2304] bf16 tensor with stride(1)=1, and the gate must pass its actual stride(0). The two changes compose because they touch disjoint kernels on the same shape.
- The gate is batch == 1 only. muse_attention_decode later measured the fused kernel at 16 splits on decode_16 as +33% (sliding-fused-triton-decode-batched-deep-splits). Sliding decode_16 and decode_32 are a larger lever here (weight 46 each), and that should be a follow-up hypothesis once this one is proven.

