---
problem: muse_attention_decode
slug: full-verify1-sk-pseudo-decode
created: 2026-10-25T042335Z
base: master
area: full-layer attention (batch-1 verify, stream-K pseudo-decode)
change: >
  Built on master (full-decode stream-K kernels landed, c3c1b01). Full verify_1 is 30.0 us (FA3
  main 15.0 + combine 7.6 + prepare 2.7 + pre-attn + norm-gate) against a 4.3 us floor and is 38%
  of the verify_1 load. At full verify batch 1 only (window == 0, q_len 8, batch 1, tp8, page 1;
  config-only) route through the landed single-level stream-K decode kernel
  (splitkv_attn_sk_kernel <4 stages, ticket>, grid 2xSM) over 8 pseudo-requests, one per verify
  token: new template flag PSEUDO (false = landed code, byte-identical); pseudo-request j: seq_j =
  cache_seqlens[0] - 7 + j, keys [0, seq_j), page_table row 0, FP8 q = token j of master
  _pre_attention_kernel output (PAIRED=False, already writes the 8 new rows), norm-gate epilogue
  with qkv row j into out row j; tile prefix over 8 pseudo-requests from cache_seqlens; workspace
  grid + 8. Each 80K-key window is shared by all 8 pseudo-requests and stream-K places equal key
  offsets of different pseudo-requests on concurrently running CTAs, so DRAM traffic stays ~20 MB
  while L2 serves 8x (160 MB). This keeps the gather-bound 16-row mma.sync design (LESSONS: the
  64-row verify tiling is MMA-bound on mma.sync). Prepare: time it vs master FA3 verify_1 path
  (prepare+main+combine+norm-gate) cold L2 median of 10, print both, route only if >= 5% faster
  (PATH CHOSEN with timings). Self-check vs fp32 causal reference: 80000, 4097, 63 and 8 keys, kv
  1.0 and 0.25, grids 3/7/device, two replays. B > 1 verify and all decode unchanged. BUILD/SAFETY
  as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First
  measurement is an okc check.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: replaced ~26.5 us (FA3 15.0 + combine 7.6 + prepare 2.7 + norm-gate) of 30.0; 160 MB L2 at ~9 TB/s ~18 us + merge, target shape <= 24 us
  - shapes: "*layer=full load=verify_16* *layer=full load=verify_32* *load=decode_* *layer=sliding*"
    direction: flat
    why: outside the gate; PSEUDO=false is landed code
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# full-verify1-sk-pseudo-decode

## Change

Built on master (full-decode stream-K kernels landed, c3c1b01). Full verify_1 is 30.0 us (FA3 main 15.0 + combine 7.6 + prepare 2.7 + pre-attn + norm-gate) against a 4.3 us floor and is 38% of the verify_1 load. At full verify batch 1 only (window == 0, q_len 8, batch 1, tp8, page 1; config-only) route through the landed single-level stream-K decode kernel (splitkv_attn_sk_kernel <4 stages, ticket>, grid 2xSM) over 8 pseudo-requests, one per verify token: new template flag PSEUDO (false = landed code, byte-identical); pseudo-request j: seq_j = cache_seqlens[0] - 7 + j, keys [0, seq_j), page_table row 0, FP8 q = token j of master _pre_attention_kernel output (PAIRED=False, already writes the 8 new rows), norm-gate epilogue with qkv row j into out row j; tile prefix over 8 pseudo-requests from cache_seqlens; workspace grid + 8. Each 80K-key window is shared by all 8 pseudo-requests and stream-K places equal key offsets of different pseudo-requests on concurrently running CTAs, so DRAM traffic stays ~20 MB while L2 serves 8x (160 MB). This keeps the gather-bound 16-row mma.sync design (LESSONS: the 64-row verify tiling is MMA-bound on mma.sync). Prepare: time it vs master FA3 verify_1 path (prepare+main+combine+norm-gate) cold L2 median of 10, print both, route only if >= 5% faster (PATH CHOSEN with timings). Self-check vs fp32 causal reference: 80000, 4097, 63 and 8 keys, kv 1.0 and 0.25, grids 3/7/device, two replays. B > 1 verify and all decode unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## Expected

- `*layer=full load=verify_1 *` faster: replaced ~26.5 us (FA3 15.0 + combine 7.6 + prepare 2.7 + norm-gate) of 30.0; 160 MB L2 at ~9 TB/s ~18 us + merge, target shape <= 24 us
- `*layer=full load=verify_16* *layer=full load=verify_32* *load=decode_* *layer=sliding*` flat: outside the gate; PSEUDO=false is landed code

## Risk

Kill: prepare timing not >= 5% below FA3 (gate keeps FA3, flat). Risk: L2 thrash if CTAs for different pseudo-requests drift apart in key offset; a pseudo-request-interleaved tile order (tile index = key_tile*8 + j) is the fallback variant to time in the same prepare.
