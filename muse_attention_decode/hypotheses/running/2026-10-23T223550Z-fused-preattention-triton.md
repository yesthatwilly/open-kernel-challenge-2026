---
problem: muse_attention_decode
slug: fused-preattention-triton
created: 2026-10-23T223550Z
base: master
area: pre-attention norm+rope
change: >
  Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8),
  v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per
  (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at
  offset h*64 as even/odd halves, apply scaleless RMSNorm (rsqrt(mean(x^2)+1e-5)), apply
  interleaved RoPE with pos=positions[t] and inv_freq[i] = theta^(-2i/64) (a per-(theta, device)
  module-level table, built from config, not data) when rope_theta != 0, and store the result as
  float8e4nv into a fresh [B*L, Hq, 64] FP8 query buffer (the 5.473 factor stays folded into FA3's
  softmax_scale). Programs h >= Hq do the same norm+RoPE on key head h-Hq, divide by k_scale, and
  store FP8 straight into k_cache.view(-1, Hkv*64) at row cache_loc[t]; the same program loads the
  matching value head (offset (Hq+Hkv+kh)*64), divides by v_scale and stores it FP8 into v_cache
  at the same row. FA3 (with the FP8 query buffer) and rms_norm_gate stay exactly as in the parity
  pipeline. The exact kernel is already written in the uncommitted working-tree submission.py
  (check job dfee6f2939974bad, queued, no result yet); churnsubs should take that code verbatim
  onto the parity base.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "launch-bound: SGLang spends ~21 us on a shape whose floor is 0.1 us; four of its seven launches disappear (about 8-10 us of ~2.5 us launches)"
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same launch count, 8 tokens; still launch-bound (23 us vs 0.1 us floor)
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "27 us vs 1.6 us floor: still dominated by fixed launch cost, same four launches removed"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 31.5 us vs 3.3 us floor, same mechanism
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same mechanism at 8 tokens
  - shapes: "*layer=full load=decode_1*"
    direction: faster
    why: 30.8 us vs 4.3 us floor; the same four launches go, FA3 unchanged
  - shapes: "*layer=full load=decode_16*"
    direction: flat
    why: 201 us is FA3 over 16 x 80K tokens; the saved launches are under 5% of it
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: 388 us, FA3-bound
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T075244Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T075244Z
base_sha: ee19eef05eb85988ab83008de621ef92e267203b
base_blob: 7cafe07522219f5d70a1d23f29e3ee2fbdb497b7
branch: hyp/muse_attention_decode/fused-preattention-triton
worktree: .worktrees/muse_attention_decode/fused-preattention-triton
mode: check
implemented_at: 2026-10-24T075304Z
impl_commit: 5b9490d351677736d89780f3e73b0f331a8332d4
submission_sha256: fde179d5b9e42bf50f8403b1b014c9e5d15141e65445669a5bddfceb09b64f27
norm_sha256: cf6d25c1a6f133b50185228070b3bdbd2052abf2bf501fb0bf38c080d06b32f8
submission_blob: f14211b4b0e88f8b3f683d9f9275b76a64b66e02
lint_warnings: 0
attempt_file: 2026-10-24T075304Z-check-fused-preattention-triton.md
prepared_at: 2026-10-24T075304Z
---

# fused-preattention-triton

## Change

Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at offset h*64 as even/odd halves, apply scaleless RMSNorm (rsqrt(mean(x^2)+1e-5)), apply interleaved RoPE with pos=positions[t] and inv_freq[i] = theta^(-2i/64) (a per-(theta, device) module-level table, built from config, not data) when rope_theta != 0, and store the result as float8e4nv into a fresh [B*L, Hq, 64] FP8 query buffer (the 5.473 factor stays folded into FA3's softmax_scale). Programs h >= Hq do the same norm+RoPE on key head h-Hq, divide by k_scale, and store FP8 straight into k_cache.view(-1, Hkv*64) at row cache_loc[t]; the same program loads the matching value head (offset (Hq+Hkv+kh)*64), divides by v_scale and stores it FP8 into v_cache at the same row. FA3 (with the FP8 query buffer) and rms_norm_gate stay exactly as in the parity pipeline. The exact kernel is already written in the uncommitted working-tree submission.py (check job dfee6f2939974bad, queued, no result yet); churnsubs should take that code verbatim onto the parity base.

## Expected

- `*layer=sliding load=decode_1*` faster: launch-bound: SGLang spends ~21 us on a shape whose floor is 0.1 us; four of its seven launches disappear (about 8-10 us of ~2.5 us launches)
- `*layer=sliding load=verify_1*` faster: same launch count, 8 tokens; still launch-bound (23 us vs 0.1 us floor)
- `*layer=sliding load=decode_16*` faster: 27 us vs 1.6 us floor: still dominated by fixed launch cost, same four launches removed
- `*layer=sliding load=decode_32*` faster: 31.5 us vs 3.3 us floor, same mechanism
- `*layer=sliding load=verify_*` faster: same mechanism at 8 tokens
- `*layer=full load=decode_1*` faster: 30.8 us vs 4.3 us floor; the same four launches go, FA3 unchanged
- `*layer=full load=decode_16*` flat: 201 us is FA3 over 16 x 80K tokens; the saved launches are under 5% of it
- `*layer=full load=decode_32*` flat: 388 us, FA3-bound

## Risk

Risks: the per-(theta, device) inv_freq table is built lazily on the first sliding call; build it at import or in the first eager call so nothing allocates under graph capture. num_warps=1 with 32 lanes per half-head is sized for head_dim 64 only. New FP8 cache entries must stay within 8% of the reference: norm and RoPE are computed in fp32 exactly as reference._rms/_rope, so the only difference is the fp32-vs-bf16 rounding before the FP8 cast, which SGLang's own path shares. The rms check for the output is against the submission's own cache entries, so the k/v write path is checked separately.

## Adversarial review

Reviewed 2026-10-24T075244Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, rules-compliant base. The measured result (dfee6f2939974bad, 1.127x [1.111,1.144], 16/16) matches every expected direction, and nothing has landed that could confound it. Land it as a verbatim whole-file replacement of master's submission.py, not as a patch onto the held parity base. Fix two latent fragilities (contiguity assumptions, the lazy inv_freq table) before 12 dependents inherit them.

### Justification
Mechanism: SGLang's sliding and small full shapes are launch-bound (floors 0.1-4.3 us vs 21-36 us measured), and the kernel collapses FusedQKNorm(+RoPE), the k and v FP8 casts, store_cache and the q cast into one launch. The profile of dfee6f2939974bad confirms it: _pre_attention_kernel takes 1.3-2.7 us per shape and the remaining time is FA3 main, FA3 combine, prepare_varlen_num_blocks and norm_gate. Per-shape paired ratios: sliding decode_1 1.246, decode_16 1.298, decode_32 1.249, verify_1 1.334, verify_16 1.304, verify_32 1.194; full decode_1 1.127, verify_1 1.213 (both faster, as expected); full decode_16 1.029, decode_32 1.015, verify_16 1.036, verify_32 1.015 (expected flat; they come out slightly faster, as expected from saving a few us on 200-390 us). The expected list matches the TASK weights: the sliding shapes (weight 46) carry the gain. Weakest link: the RoPE angle precision at large positions. ang = float(pos) * inv_freq goes through tl.cos/tl.sin. If Triton lowers these to the approximate MUFU path (cos.approx.f32), the error grows with |angle| (up to about 8e4 rad at 80K context for i=0), whereas the reference uses accurate torch cos/sin on the same fp32 product. All 16 tests passed and every timed 80K sliding call was checked, so it is within bounds today. The margin against the 8% k-entry bound and the output bound is not known, and a dependent that changes rounding elsewhere (an extra bf16 hop, a different FP8 cast) could take the slack. A second, minor point: k and q go fp32 -> FP8 directly, while SGLang goes fp32 -> bf16 -> FP8. This is finer, not coarser, so it is allowed, but the bits differ from SGLang.

### Confounders on the current kernel
- None: landed-muse_attention_decode.json is empty, and master is the plain-PyTorch starter (BEST 0.0023x). This hypothesis is itself the first base, so no landed change can share its launch sequence, memory traffic or numerics.

### Interference and A/B needs
No companion is warranted. The result already exists, and resubmitting identical code returns the same job. Nothing has landed to revert or tune. Splitting the sub-changes (q fused vs k/v store fused) would only re-attribute a gain that all comes from one mechanism (removing launches), and it would not change any dependent's interpretation. Each dependent measures against this base, so its own A/B is internal to its own change. The real risk to dependents is procedural, not statistical. The kernel lives only in the dirty working tree, and the hypothesis text says 'take that code verbatim onto the parity base', but sglang-path-parity is held and was never landed. Churnsubs must commit the .bak file as a whole-file replacement of master's submission.py (blob identical to the dfee6f2939974bad submission), so that every dependent's parent commit is that exact code and their baseline-relative numbers are comparable to 1.127x.

### Open questions
- Does tl.cos/tl.sin on fp32 lower to libdevice cosf/sinf or to cos.approx/sin.approx on this Triton version? If approx, how much of the 8% k-entry margin is left at position ~80K, and should the kernel compute angles with explicit range reduction (or a precomputed fp32 table via libdevice) to protect dependents?
- What are the actual check margins (k-entry rel-L2 and output error vs bound) in submission.dfee6f2939974bad.log? Recording them in FINDINGS lets dependents see whether they ate into the precision slack.
- The hypothesis body is stale ('check job ... queued, no result yet', 'onto the parity base'). Update it to the measured result, and set base to master with whole-file replacement, before churnsubs dispatches dependents.


## Implementation notes

Whole-file replacement of master's starter `muse_attention_decode/submission.py` with the exact
file behind check job dfee6f2939974bad (16/16 passed, 1.127x [1.111, 1.144]), copied verbatim
from the main tree's uncommitted working copy (sha256 prefix fde179d5b9e42bf5). It is not a patch
onto sglang-path-parity, which was held as identical to baseline.py. At the user's direction this
hypothesis has no parent and is the base for the 12 re-pointed dependents. Nothing was changed
from the measured code, so the measured result stands.

`_pre_attention_kernel` fuses FusedQKNorm (+ interleaved RoPE on sliding layers), the k/v FP8
casts, store_cache and the q FP8 cast into one launch. FA3 and rms_norm_gate are as in baseline.py.

Follow-ups from the adversarial review, left out on purpose because they would change the measured
code: add stride/contiguity asserts on qkv, k_cache, v_cache and the q buffer; build `_INV_FREQ`
for theta=500000 at import (or warm it in prepare) so a capture before the first eager call cannot
allocate inside the graph; check whether tl.cos/tl.sin lower to approximate MUFU at ~80K positions
and record the k-entry check margin from the dfee6f2939974bad log. q and k go fp32 -> FP8 directly
(SGLang goes fp32 -> bf16 -> FP8): finer, allowed, but not bit-identical to SGLang.

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; q buffer allocated from shapes only; branching on config only.
- [x] Nothing derived from `data` is cached across calls (`_INV_FREQ` derives from config theta/d only).
- [x] Weights: n/a, this problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 E4M3; queries FP8 E4M3.
- [x] Attention sees every key: FA3 called with the full window (window-1, 0) / full context, unchanged from baseline.py.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `CLEAN: .../muse_attention_decode/submission.py`
