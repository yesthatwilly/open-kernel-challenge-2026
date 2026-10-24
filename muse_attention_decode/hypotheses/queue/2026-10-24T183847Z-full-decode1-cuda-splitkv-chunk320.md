---
problem: muse_attention_decode
slug: full-decode1-cuda-splitkv-chunk320
created: 2026-10-24T183847Z
base: master
area: full-layer attention (batch-1 decode gate)
change: >
  Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's
  existing CUDA split-KV path with a smaller work chunk, instead of unpaired FA3 +
  FlashAttnFwdCombine + sglang norm_gate. (1) In _SPLITKV_CUDA make the chunk a template
  parameter: template <int CHUNK> on splitkv_attn_kernel and on splitkv_combine_kernel (TILES =
  CHUNK / TILE, IDX_OFF/Q_OFF/... offsets and SMEM_A become constexpr functions of CHUNK; the idx
  double buffer is 2 * CHUNK ints). Replace issue_meta's loop `for j < CHUNK / THREADS` with `for
  (int e = tid; e < CHUNK; e += THREADS)` so CHUNK need not be a multiple of 256. Everything else
  in the kernels (ring, swizzle, mma, softmax, item-end merge, partial slot b * max_chunks +
  chunk) stays byte for byte. (2) Give splitkv_combine_kernel a second template parameter NW
  (warps per block): red_m/red_l/red_acc sized [NW], loop 1 strides by NW * 32, loop 2 strides by
  NW, the final warp-0 merge sums NW entries, __launch_bounds__(NW * 32). (3) Host functions
  splitkv_attn and splitkv_combine take a new int64 chunk argument (update _SPLITKV_CPP too) and
  dispatch: chunk 1024 -> <1024> (combine <1024, 4>, i.e. today's code), chunk 320 -> <320>
  (combine <320, 16>, 512 threads), anything else TORCH_CHECK fails. cudaFuncSetAttribute is set
  once per instantiation. Rename the module to muse_full_decode_splitkv_v2 so no stale build is
  reused. (4) Python: _splitkv_gate widens 1 < batch to 1 <= batch; _full_decode_splitkv picks
  chunk = 320 if batch == 1 else 1024 (config only), max_chunks = ceil(page_table.shape[1] /
  chunk), and passes chunk to both calls; grid stays 2 x SM count. At 80K keys that is 250 items
  of 5 tiles, one per CTA, versus 79 items at 1024. kernel_fn is unchanged otherwise: batch 1 is
  already unpaired, so _pre_attention_kernel writes the [1, 16, 64] FP8 q the split-KV path takes.
  (5) Self-check: add a batch-1 pass to _splitkv_self_check with lens [80000] and [5000] at
  kv_scale 1 and 0.25, grid device and grid 3, same rel L2 < 2e-2 bound; batch > 1 cases
  unchanged. Print '[splitkv] batch-1 chunk 320 path ON/OFF' once at import.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "incumbent 27.3 us = FA3 main 14.6 + FlashAttnFwdCombine 7.1 + pre 1.3 + norm_gate 1.1 + gaps. 20.5 MB of KV over 250 CTAs at the kernel's ~2.5-2.9 TB/s is ~7-9 us, a 250-partial combine with 16 warps ~2-3 us, pre 1.3: ~14-17 us shape, -35-45%; full is 42% of the decode_1 load, so ~+15-18% on that load, ~+2.5-3% geomean"
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside the gate
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: batch > 1 keeps chunk 1024 and the 4-warp combine, same code via the <1024, 4> instantiation
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len 8 is outside the gate
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T184132Z
review_companions: null
---

# full-decode1-cuda-splitkv-chunk320

## Change

Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's existing CUDA split-KV path with a smaller work chunk, instead of unpaired FA3 + FlashAttnFwdCombine + sglang norm_gate. (1) In _SPLITKV_CUDA make the chunk a template parameter: template <int CHUNK> on splitkv_attn_kernel and on splitkv_combine_kernel (TILES = CHUNK / TILE, IDX_OFF/Q_OFF/... offsets and SMEM_A become constexpr functions of CHUNK; the idx double buffer is 2 * CHUNK ints). Replace issue_meta's loop `for j < CHUNK / THREADS` with `for (int e = tid; e < CHUNK; e += THREADS)` so CHUNK need not be a multiple of 256. Everything else in the kernels (ring, swizzle, mma, softmax, item-end merge, partial slot b * max_chunks + chunk) stays byte for byte. (2) Give splitkv_combine_kernel a second template parameter NW (warps per block): red_m/red_l/red_acc sized [NW], loop 1 strides by NW * 32, loop 2 strides by NW, the final warp-0 merge sums NW entries, __launch_bounds__(NW * 32). (3) Host functions splitkv_attn and splitkv_combine take a new int64 chunk argument (update _SPLITKV_CPP too) and dispatch: chunk 1024 -> <1024> (combine <1024, 4>, i.e. today's code), chunk 320 -> <320> (combine <320, 16>, 512 threads), anything else TORCH_CHECK fails. cudaFuncSetAttribute is set once per instantiation. Rename the module to muse_full_decode_splitkv_v2 so no stale build is reused. (4) Python: _splitkv_gate widens 1 < batch to 1 <= batch; _full_decode_splitkv picks chunk = 320 if batch == 1 else 1024 (config only), max_chunks = ceil(page_table.shape[1] / chunk), and passes chunk to both calls; grid stays 2 x SM count. At 80K keys that is 250 items of 5 tiles, one per CTA, versus 79 items at 1024. kernel_fn is unchanged otherwise: batch 1 is already unpaired, so _pre_attention_kernel writes the [1, 16, 64] FP8 q the split-KV path takes. (5) Self-check: add a batch-1 pass to _splitkv_self_check with lens [80000] and [5000] at kv_scale 1 and 0.25, grid device and grid 3, same rel L2 < 2e-2 bound; batch > 1 cases unchanged. Print '[splitkv] batch-1 chunk 320 path ON/OFF' once at import.

## Expected

- `*layer=full load=decode_1 *` faster: incumbent 27.3 us = FA3 main 14.6 + FlashAttnFwdCombine 7.1 + pre 1.3 + norm_gate 1.1 + gaps. 20.5 MB of KV over 250 CTAs at the kernel's ~2.5-2.9 TB/s is ~7-9 us, a 250-partial combine with 16 warps ~2-3 us, pre 1.3: ~14-17 us shape, -35-45%; full is 42% of the decode_1 load, so ~+15-18% on that load, ~+2.5-3% geomean
- `*layer=sliding*` flat: outside the gate
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: batch > 1 keeps chunk 1024 and the 4-warp combine, same code via the <1024, 4> instantiation
- `*load=verify_*` flat: q_len 8 is outside the gate

## Risk

Full decode_1 is the one scored shape none of the three CUDA rewrites in the judge touches (full-cuda-onelaunch-warpspec-bulk-gather gates 1 < batch, its batch 1 is stage 3). Precedent: the Triton gather at batch 1 lost 5% to FA3 at 0.89 TB/s (full-fused-triton-decode1-gate); the CUDA kernel streams 2.94 TB/s at decode_16, so the batch-1 risk is parallelism, which the 320-token chunk addresses. Weakest link: the combine merging 250 partials per head from L2 (the 4-warp combine already costs 8 us at B=16); read its own row. Sibling: full-decode1-cuda-splitkv-chunk640 is the A/B on chunk size; run whichever the reviewer prefers first.

## Adversarial review

Reviewed 2026-10-24T184132Z by an adversarial subagent (model opus).

**Verdict: proceed.** Moving full decode_1 (27.3 us, 6.3x its 4.3 us floor, the only scored decode shape still on FA3) onto master's proven CUDA split-KV kernel with 250 CTA-sized 320-token items is sound and its effect will show up clearly in its own profile row. No rules problem and no companion is needed: the queued chunk640 sibling already serves as the chunk-size A/B.

### Justification
Plausible. FINDINGS shows full decode_1 at 27.3 us (FA3 main 14.6 + FlashAttnFwdCombine 7.1 + pre 1.3 + norm_gate + gaps) against a 4.3 us floor (20.5 MB of KV). That is the worst ratio to floor of any full shape. At batch 1 the existing kernel with CHUNK 1024 would put only 79 items on the 264 CTA slots, so 60% of SMs would hold one CTA each. At 320 it gets 250 items, one per CTA on about 125 SMs with 2 CTAs each, so roughly 96 KB of ring in flight per SM, comfortably above what Little's law needs at 4.8 TB/s. The template change is safe for TILES = 5: the cross-item meta prefetch needs TILES >= STAGES - 1 = 3 (item k+1's idx/q group is issued at tile k*TILES and must complete before tile (k+1)*TILES is issued at iteration (k+1)*TILES - 3), and that holds. The strided issue_meta loop handles 320 = 256 + 64 correctly. The arithmetic for the expected gain checks out: decode_1 load = 46*13.2 + 16*27.3 µs, so full is 42% of it; -11 to -13 us on full gives about +17% on that load and about +2.7% geomean over 6 loads, which is above the ±1.5% CI floor. The other expected-flat directions follow from the gate (batch > 1 keeps the <1024,4> instantiation byte for byte; verify and sliding sit outside the gate). WEAKEST LINK: the attn-kernel estimate of 7-9 us assumes 2.5-2.9 TB/s, a rate measured at decode_16 where each CTA streams 256 KB+ across items. At batch 1 each CTA streams only 80 KB (5 tiles) behind three dependent cold-DRAM round trips: seqlens read, then cp_wait<0> on page indices, then the first K/V tile. Fixed latency is therefore about 25-35% of the item, and 9-11 us is more realistic. Close behind is the combine: only 16 CTAs, two serial passes over 250 partials (the ml pass is a 128-byte-strided gather), and a second launch gap. The current 4-warp combine already costs 8.2 us at B=16, so 2-3 us here is optimistic and 3-5 us is plausible. Even in that pessimistic case (pre 1.3 + attn 11 + combine 5 + gaps 1.5, about 19 us) the shape still beats 27.3 us, so the direction is robust and only the magnitude is uncertain.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: not a confounder but the substrate. This hypothesis re-instantiates that kernel at a new CHUNK, so any batch-1 weakness of its design (serialized meta prologue, item-end smem merge) is measured together with the chunk choice. The chunk640 sibling, the same code at another chunk, separates the two.
- paired-heads-launch-cleanup-ab-tune-batch-gate: defines the incumbent being replaced (batch 1 unpaired FA3 + sglang rms_norm_gate). The gain therefore also includes dropping FA3's FlashAttnFwdCombine (7.1 us) and the separate norm_gate launch (the epilogue is fused into splitkv_combine), not only the chunk size. That is fine because the hypothesis claims the whole path swap, but the result should not be read as 'chunk 320 is optimal'.
- fused-preattention-triton / paired-heads-launch-cleanup: pre-attention kernel and q layout unchanged at batch 1 (PAIRED already false), so they do not confound; the pre kernel's 1.3 us should stay flat and is a sanity check.
- sliding-deep-splits-on-cuda-splitkv: shares the decode_1 load but a different shape and kernel; cold L2 per replay means no cross-shape interaction. Sliding decode_1 must stay at ~13.2 us.

### Interference and A/B needs
No companion warranted. The readout is per-shape and per-kernel: the profile will show splitkv_attn_kernel<320> and splitkv_combine_kernel<320,16> times directly against FA3 14.6 + combine 7.1. A loss is therefore attributable from the profile (an attn row above about 14 us means the batch-1 prologue/latency design is the problem; a combine row above about 6 us means partial count or combine design is). A gain is unambiguous because the gate isolates one shape and the other 11 shapes act as built-in controls (the batch>1 path is the <1024,4> instantiation of identical code, so decode_16/32 full must be flat within noise or the template refactor itself changed codegen). An ab-revert of any landed hypothesis would not change the interpretation. A chunk-1024-at-batch-1 control would only split path-swap from chunk effect, and the queued sibling full-decode1-cuda-splitkv-chunk640 already gives a second point on that curve. Run chunk320 first: it saturates CTA slots, and the combine cost difference between 250 and 125 partials is small compared with the latency-hiding gain.

### Open questions
- Does the combine's first pass (ml max over 250 partials, 128-byte strided reads) plus the 512-thread single-CTA-per-head design actually come in under 3 us? If not, the follow-up is a combine that reads ml once (online merge in one pass) or a one-launch tail merge in the attn kernel.
- Would prefetching item 0's page indices before the seqlens prefix sum (B=1 needs no prefix) remove one dependent DRAM round trip, worth roughly 0.5-1 us at batch 1?
- Merge risk: full-verify-cuda-streamk-splitkv and full-verify1-cuda-streamk-batch1 (running/queued) likely edit _SPLITKV_CUDA too; the CHUNK template refactor will conflict textually with them if both land.
- At 250 items on 264 slots, 14 CTAs exit immediately and ~7 SMs host one CTA; does SM-level imbalance (some SMs 2 CTAs, some 1) show up as a tail? Chunk 304-ish (264 items) is a possible later tune.

