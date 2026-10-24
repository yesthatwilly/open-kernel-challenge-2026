# muse_attention_decode · 2026-10-24T083412Z · submit · fused-preattention-triton

Hypothesis: `muse_attention_decode/hypotheses/done/gain/2026-10-23T223550Z-fused-preattention-triton.md` (branch `hyp/muse_attention_decode/fused-preattention-triton`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `c5eb032`

## Goal
Promote check-proven gain fused-preattention-triton (1.1274x [1.1105,1.1444], check job 9e8fcb3fe4b04458) to the board.

Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at offset h*64 as even/odd halves, apply scaleless RMSNorm (rsqrt(mean(x^2)+1e-5)), apply interleaved RoPE with pos=positions[t] and inv_freq[i] = theta^(-2i/64) (a per-(theta, device) module-level table, built from config, not data) when rope_theta != 0, and store the result as float8e4nv into a fresh [B*L, Hq, 64] FP8 query buffer (the 5.473 factor stays folded into FA3's softmax_scale). Programs h >= Hq do the same norm+RoPE on key head h-Hq, divide by k_scale, and store FP8 straight into k_cache.view(-1, Hkv*64) at row cache_loc[t]; the same program loads the matching value head (offset (Hq+Hkv+kh)*64), divides by v_scale and stores it FP8 into v_cache at the same row. FA3 (with the FP8 query buffer) and rms_norm_gate stay exactly as in the parity pipeline. The exact kernel is already written in the uncommitted working-tree submission.py (check job dfee6f2939974bad, queued, no result yet); churnsubs should take that code verbatim onto the parity base.

## What changed, and where
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

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; q buffer allocated from shapes only; branching on config only.
- [x] Nothing derived from `data` is cached across calls (`_INV_FREQ` derives from config theta/d only).
- [x] Weights: n/a, this problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 E4M3; queries FP8 E4M3.
- [x] Attention sees every key: FA3 called with the full window (window-1, 0) / full context, unchanged from baseline.py.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `CLEAN: .../muse_attention_decode/submission.py`

## Risks / what could fail
low

## Result (after the run)
(pending)
