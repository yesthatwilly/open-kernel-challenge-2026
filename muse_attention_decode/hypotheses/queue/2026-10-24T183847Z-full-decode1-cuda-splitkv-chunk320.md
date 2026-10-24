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
review: required
status: queued
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
