---
problem: muse_decode_layer
slug: fused-preattention-paired-pad-restack
created: 2026-10-24T165619Z
base: master
area: pre-attention norm+rope
change: >
  Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch
  muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch, _pre_attention_kernel
  reading RoPE from state['cos_sin']) onto master once paired-heads-pad-unpad-triton has landed
  (build on master if it landed, else on its branch), and finish the port of
  muse_attention_decode's landed chain fused-preattention-triton + paired-heads-launch-cleanup
  part (1) 'pad at the source', which decode_layer never got. 3-way merge the patch's kernel, its
  prepare() warm-up and its kernel_fn call onto master, then add attention master's PAIRED and G
  constexprs to the kernel verbatim (muse_attention_decode/submission.py on master,
  _pre_attention_kernel lines 117-124: when PAIRED, query program h writes its 64 FP8 values into
  half (h // G) % 2 of a [tokens, Hq, 128] FP8 row and 64 FP8 zeros into the other half). In
  kernel_fn: one launch, grid (tokens, Hq + Hkv), num_warps 1, replaces state['qk_norm'](...), the
  k/v divide by the scales, k.to(FP8), v.to(FP8), store_cache(...) and either
  q.view(T,16,64).to(FP8) (unpaired) or the _pair_pad_q_kernel launch that
  paired-heads-pad-unpad-triton added (paired: window == 0 and batch > 1 and hkv % 2 == 0,
  unchanged predicate, passed as PAIRED). q buffer is torch.empty((tokens, hq, 2*d if paired else
  d), FP8). gate and v stay slices of qkv; FA3, _paired_norm_gate_kernel / o_norm_gate and
  everything after the o GEMM are byte-identical to the base. If sliding-fused-triton-decode1-port
  has landed, its sliding batch-1 decode branch keeps running before this code and is untouched;
  if full-decode-cuda-splitkv-port has landed, pass PAIRED=False on its gated shapes exactly as
  attention master does (paired = ... and not splitkv). Delete _pair_pad_q_kernel only if nothing
  else calls it.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: FusedQKNorm + 2 FP8 casts + store_cache + _pair_pad_q_kernel (5 launches) become 1; attention saved 3.5-6.4 us for this 5->1 on 17-36 us shapes, ~1% of ~400 us
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: same 5->1 at T=128 on the paired path, ~4-6 us of ~580 us
  - shapes: "*load=decode_1*"
    direction: faster
    why: unpaired 5->1 on latency-bound 167-177 us shapes; the neutral parent moved these +2.3% (full decode_1 also loses its q cast); sliding decode_1 flat if sliding-fused-triton-decode1-port landed (it bypasses this code)
  - shapes: "*load=verify_1*"
    direction: faster
    why: T=8 unpaired 5->1 on ~230-245 us shapes, ~1-2%
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: the parent gained 3.3% here, the one shape outside drift
  - shapes: "*load=verify_32*"
    direction: flat
    why: few us on ~620-910 us shapes; Marlin dominates
risk: low
depends_on: paired-heads-pad-unpad-triton
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# fused-preattention-paired-pad-restack

## Change

Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.patch, _pre_attention_kernel reading RoPE from state['cos_sin']) onto master once paired-heads-pad-unpad-triton has landed (build on master if it landed, else on its branch), and finish the port of muse_attention_decode's landed chain fused-preattention-triton + paired-heads-launch-cleanup part (1) 'pad at the source', which decode_layer never got. 3-way merge the patch's kernel, its prepare() warm-up and its kernel_fn call onto master, then add attention master's PAIRED and G constexprs to the kernel verbatim (muse_attention_decode/submission.py on master, _pre_attention_kernel lines 117-124: when PAIRED, query program h writes its 64 FP8 values into half (h // G) % 2 of a [tokens, Hq, 128] FP8 row and 64 FP8 zeros into the other half). In kernel_fn: one launch, grid (tokens, Hq + Hkv), num_warps 1, replaces state['qk_norm'](...), the k/v divide by the scales, k.to(FP8), v.to(FP8), store_cache(...) and either q.view(T,16,64).to(FP8) (unpaired) or the _pair_pad_q_kernel launch that paired-heads-pad-unpad-triton added (paired: window == 0 and batch > 1 and hkv % 2 == 0, unchanged predicate, passed as PAIRED). q buffer is torch.empty((tokens, hq, 2*d if paired else d), FP8). gate and v stay slices of qkv; FA3, _paired_norm_gate_kernel / o_norm_gate and everything after the o GEMM are byte-identical to the base. If sliding-fused-triton-decode1-port has landed, its sliding batch-1 decode branch keeps running before this code and is untouched; if full-decode-cuda-splitkv-port has landed, pass PAIRED=False on its gated shapes exactly as attention master does (paired = ... and not splitkv). Delete _pair_pad_q_kernel only if nothing else calls it.

## Expected

- `*layer=full load=decode_16*` faster: FusedQKNorm + 2 FP8 casts + store_cache + _pair_pad_q_kernel (5 launches) become 1; attention saved 3.5-6.4 us for this 5->1 on 17-36 us shapes, ~1% of ~400 us
- `*layer=full load=verify_16*` faster: same 5->1 at T=128 on the paired path, ~4-6 us of ~580 us
- `*load=decode_1*` faster: unpaired 5->1 on latency-bound 167-177 us shapes; the neutral parent moved these +2.3% (full decode_1 also loses its q cast); sliding decode_1 flat if sliding-fused-triton-decode1-port landed (it bypasses this code)
- `*load=verify_1*` faster: T=8 unpaired 5->1 on ~230-245 us shapes, ~1-2%
- `*layer=sliding load=decode_32*` faster: the parent gained 3.3% here, the one shape outside drift
- `*load=verify_32*` flat: few us on ~620-910 us shapes; Marlin dominates

## Risk

Low expected value (~+1% headline, near the 1.5% drift), so judge it per shape: the paired rows against paired-heads-pad-unpad-triton's rows, the unpaired batch-1 rows against the neutral parent's. It is the only attention-side landed piece not yet ported or queued for decode_layer (port audit 2026-10-24). Confounders: full-decode-cuda-splitkv-port replaces the paired path on full decode_16/32 if it lands first (then only verify_16/32 keep the paired pad), and sliding-fused-triton-decode1-port removes sliding decode_1 from this code path. Correctness: the k/v cache store moves into Triton, so the cache contents are checked by the judge's later reads; keep the parent's kv_scale handling (divide before the cast) and cache_loc int64 indexing.
