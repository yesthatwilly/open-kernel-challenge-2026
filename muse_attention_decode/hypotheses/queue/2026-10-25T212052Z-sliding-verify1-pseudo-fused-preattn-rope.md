---
problem: muse_attention_decode
slug: sliding-verify1-pseudo-fused-preattn-rope
created: 2026-10-25T212052Z
base: master
area: sliding-layer attention (batch-1 verify, PSEUDO stream-K kernel, pre-attention fold)
change: >
  DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch,
  whatever its score; the change touches only the sliding verify B=1 band). Sliding verify_1
  (weight 46, ~0.70%/us, the second-highest rate on the board) is 13.2 us of shape for
  splitkv_attn_sk_kernel PSEUDO 9.9 us + _pre_attention_kernel 1.5 us: about 1.8 us is the second
  graph node and its gap. Fold the pre-attention into the PSEUDO kernel so B=1 sliding verify is
  ONE launch, as combo-b1's V3 did for full decode_1 (FUSED prologue; 18.98 vs V1 21.41 us at its
  gate) and as skF does for full decode B>1. Put it in a NEW load_inline module muse_sv1_fused_v1
  (helper head + a copy of the PSEUDO kernel template with one extra template flag FUSED; the
  landed _EXT2 PSEUDO instantiations stay untouched, as skF was kept apart). FUSED=true at TOK=8:
  (a) q: when a CTA starts a segment of pseudo-row r = 8b + j, it loads token j's raw bf16 q row
  (16 heads x 64, 2 KB) from qkv (riding the page-index cp.async group), applies exactly what
  _pre_attention_kernel does with ROPE=True, PAIRED=False (its RMSNorm and eps, interleaved GPT-J
  RoPE pair (2i, 2i+1) at angle positions[8b+j] * inv_freq[i] from master _inv_freq, then the e4m3
  cast with the same scale handling) and builds the q fragments the non-fused path reads from
  smem. Use accurate sincosf. (b) new K/V rows: the 8 new keys (token j's own key = the last key
  of pseudo-row j, cache slot cache_loc[8b+j], in the PSEUDO kernel's own sp_r convention) are
  built by every CTA whose tile range contains any of them, from qkv (k: RMSNorm then RoPE then /
  k_scale -> e4m3; v: / v_scale -> e4m3), and written over the landed ring stage in smem after
  cp_wait and before that tile's S MMA (one __syncthreads), never read back through cp.async;
  exactly one CTA (the owner of pseudo-row 7's last tile, checked on the host like
  _skf_owner_check) stores the 8 rows to k_cache/v_cache. Other CTAs may read the stale slots by
  cp.async; they overwrite them in smem first. kc/vc become non-const in this module only. Route:
  sliding verify, window 2048, q_len 8, B == 1 only, config-only predicate; bands 2..32 unchanged.
  GATE at B=1 x 80000: interleaved [FUSED PSEUDO, routed variant and grid spec] vs [pre-attn +
  master's routed PSEUDO variant/grid] (and print every grid spec p4/p8/sm for FUSED); route only
  at >= 1.03x. SELF-CHECK vs an fp32 windowed causal reference WITH RoPE: B=1 lens 1, 7, 8, 63,
  2047, 2048, 2055, 80000, kv 0.25, grids 3/p4/p8/sm, coherent data; compare the stored new cache
  rows with _pre_attention_kernel's rows by rel L2 (< 2e-2; never bytewise: LESSONS, Triton
  sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad) and check other rows unchanged; two-state
  alternating capture-once replay with refilled inputs and NaN-poisoned outputs. Any miss: B=1
  stays on [pre-attn + PSEUDO] with a PATH CHOSEN line. New CUDA: first measurement is an okc
  check. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no
  timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may
  hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting
  int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo,
  in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM
  and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2
  graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path
  WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: 13.2 us shape = PSEUDO 9.9 + pre-attn 1.5 + ~1.8 node gap; one launch with a ~0.5 us prologue should give <= 11.2 us (floor 0.1); ~-2 us at 0.70%/us is about +1.4% headline
  - shapes: "*layer=sliding load=verify_16 *"
    direction: flat
    why: B == 1 only
  - shapes: "*layer=sliding load=verify_32 *"
    direction: flat
    why: B == 1 only (SV32 tiled from the base)
  - shapes: "*layer=full *"
    direction: flat
    why: untouched beyond the base
  - shapes: "*load=decode_* *"
    direction: flat
    why: untouched beyond the base
risk: medium
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 1
kind: primary
controls_for: null
prior: "full-decode1-sk-two-level-fused-preattn: that fold was on full decode_1 (weight 16, no RoPE, one new row, gained 22.4->21.6 within CI); this folds RoPE pre-attention for 8 query tokens and 8 new K/V rows into the TOK=8 PSEUDO kernel on sliding verify_1 (weight 46, 0.70%/us). Queued sliding-decode-sk-s3-fused-preattn-rope does the TOK=1 decode case only"
review: required
status: queued
---

# sliding-verify1-pseudo-fused-preattn-rope

## Change

DIAGNOSIS action 3(ii). Stacked on merge-perkvhead-v3chain-sv32-tiled (build on its branch, whatever its score; the change touches only the sliding verify B=1 band). Sliding verify_1 (weight 46, ~0.70%/us, the second-highest rate on the board) is 13.2 us of shape for splitkv_attn_sk_kernel PSEUDO 9.9 us + _pre_attention_kernel 1.5 us: about 1.8 us is the second graph node and its gap. Fold the pre-attention into the PSEUDO kernel so B=1 sliding verify is ONE launch, as combo-b1's V3 did for full decode_1 (FUSED prologue; 18.98 vs V1 21.41 us at its gate) and as skF does for full decode B>1. Put it in a NEW load_inline module muse_sv1_fused_v1 (helper head + a copy of the PSEUDO kernel template with one extra template flag FUSED; the landed _EXT2 PSEUDO instantiations stay untouched, as skF was kept apart). FUSED=true at TOK=8: (a) q: when a CTA starts a segment of pseudo-row r = 8b + j, it loads token j's raw bf16 q row (16 heads x 64, 2 KB) from qkv (riding the page-index cp.async group), applies exactly what _pre_attention_kernel does with ROPE=True, PAIRED=False (its RMSNorm and eps, interleaved GPT-J RoPE pair (2i, 2i+1) at angle positions[8b+j] * inv_freq[i] from master _inv_freq, then the e4m3 cast with the same scale handling) and builds the q fragments the non-fused path reads from smem. Use accurate sincosf. (b) new K/V rows: the 8 new keys (token j's own key = the last key of pseudo-row j, cache slot cache_loc[8b+j], in the PSEUDO kernel's own sp_r convention) are built by every CTA whose tile range contains any of them, from qkv (k: RMSNorm then RoPE then / k_scale -> e4m3; v: / v_scale -> e4m3), and written over the landed ring stage in smem after cp_wait and before that tile's S MMA (one __syncthreads), never read back through cp.async; exactly one CTA (the owner of pseudo-row 7's last tile, checked on the host like _skf_owner_check) stores the 8 rows to k_cache/v_cache. Other CTAs may read the stale slots by cp.async; they overwrite them in smem first. kc/vc become non-const in this module only. Route: sliding verify, window 2048, q_len 8, B == 1 only, config-only predicate; bands 2..32 unchanged. GATE at B=1 x 80000: interleaved [FUSED PSEUDO, routed variant and grid spec] vs [pre-attn + master's routed PSEUDO variant/grid] (and print every grid spec p4/p8/sm for FUSED); route only at >= 1.03x. SELF-CHECK vs an fp32 windowed causal reference WITH RoPE: B=1 lens 1, 7, 8, 63, 2047, 2048, 2055, 80000, kv 0.25, grids 3/p4/p8/sm, coherent data; compare the stored new cache rows with _pre_attention_kernel's rows by rel L2 (< 2e-2; never bytewise: LESSONS, Triton sin/cos differ in 1-6% of E4M3 bytes at ~5000 rad) and check other rows unchanged; two-state alternating capture-once replay with refilled inputs and NaN-poisoned outputs. Any miss: B=1 stays on [pre-attn + PSEUDO] with a PATH CHOSEN line. New CUDA: first measurement is an okc check. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=verify_1 *` faster: 13.2 us shape = PSEUDO 9.9 + pre-attn 1.5 + ~1.8 node gap; one launch with a ~0.5 us prologue should give <= 11.2 us (floor 0.1); ~-2 us at 0.70%/us is about +1.4% headline
- `*layer=sliding load=verify_16 *` flat: B == 1 only
- `*layer=sliding load=verify_32 *` flat: B == 1 only (SV32 tiled from the base)
- `*layer=full *` flat: untouched beyond the base
- `*load=decode_* *` flat: untouched beyond the base

## Risk

Weighted effect ~+1.4% alone, above the margin; stacked on the merge for a submit-worthy pair (~+2.5% over the best). Risk: the 8-row ownership across CTAs (several CTAs hold new keys); RoPE accuracy (compare by rel L2).
