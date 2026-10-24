---
problem: muse_attention_decode
slug: full-decode1-cuda-splitkv-chunk640
created: 2026-10-24T183847Z
base: master
area: full-layer attention (batch-1 decode gate)
change: >
  Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer
  decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's CUDA split-KV path
  instead of FA3. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel
  (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e <
  CHUNK; e += THREADS)`); kernel bodies otherwise byte for byte. (2) splitkv_combine_kernel gets
  template <int CHUNK, int NW> (red arrays [NW], loop strides NW * 32 and NW, __launch_bounds__(NW
  * 32)). (3) Host functions take an int64 chunk argument (update _SPLITKV_CPP): 1024 -> <1024>
  attn and <1024, 4> combine (today's code); 640 -> <640> attn and <640, 8> combine; anything else
  TORCH_CHECK fails. Module renamed muse_full_decode_splitkv_v2. (4) Python: _splitkv_gate allows
  1 <= batch; _full_decode_splitkv uses chunk = 640 if batch == 1 else 1024, max_chunks =
  ceil(page_table.shape[1] / chunk), grid unchanged (2 x SM count; at 80K keys only 125 items
  exist, so 125 CTAs get one 10-tile item each and the rest exit at the `bx >= total` check). (5)
  _splitkv_self_check adds batch-1 cases lens [80000] and [5000], kv_scale 1 and 0.25, grid device
  and 3, rel L2 < 2e-2. Print '[splitkv] batch-1 chunk 640 path ON/OFF' once at import.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "incumbent 27.3 us (FA3 14.6 + combine 7.1 + pre 1.3 + norm_gate 1.1). 125 CTAs x 160 KB each with 3 tiles in flight per CTA; ~8-10 us for kernel A and only 125 partials for an 8-warp combine (~1.5-2.5 us): ~14-17 us shape"
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside the gate
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: batch > 1 keeps chunk 1024
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# full-decode1-cuda-splitkv-chunk640

## Change

Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's CUDA split-KV path instead of FA3. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e < CHUNK; e += THREADS)`); kernel bodies otherwise byte for byte. (2) splitkv_combine_kernel gets template <int CHUNK, int NW> (red arrays [NW], loop strides NW * 32 and NW, __launch_bounds__(NW * 32)). (3) Host functions take an int64 chunk argument (update _SPLITKV_CPP): 1024 -> <1024> attn and <1024, 4> combine (today's code); 640 -> <640> attn and <640, 8> combine; anything else TORCH_CHECK fails. Module renamed muse_full_decode_splitkv_v2. (4) Python: _splitkv_gate allows 1 <= batch; _full_decode_splitkv uses chunk = 640 if batch == 1 else 1024, max_chunks = ceil(page_table.shape[1] / chunk), grid unchanged (2 x SM count; at 80K keys only 125 items exist, so 125 CTAs get one 10-tile item each and the rest exit at the `bx >= total` check). (5) _splitkv_self_check adds batch-1 cases lens [80000] and [5000], kv_scale 1 and 0.25, grid device and 3, rel L2 < 2e-2. Print '[splitkv] batch-1 chunk 640 path ON/OFF' once at import.

## Expected

- `*layer=full load=decode_1 *` faster: incumbent 27.3 us (FA3 14.6 + combine 7.1 + pre 1.3 + norm_gate 1.1). 125 CTAs x 160 KB each with 3 tiles in flight per CTA; ~8-10 us for kernel A and only 125 partials for an 8-warp combine (~1.5-2.5 us): ~14-17 us shape
- `*layer=sliding*` flat: outside the gate
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: batch > 1 keeps chunk 1024
- `*load=verify_*` flat: outside the gate

## Risk

A/B sibling of full-decode1-cuda-splitkv-chunk320: 320 maximizes CTAs (250, ~2 per SM) at the cost of 250 partials; 640 halves the partials and keeps ~1 CTA per SM. The two differ only in the chunk/NW constants, so comparing their kernel A and combine rows attributes the trade. Untouched by every rewrite in the judge.
