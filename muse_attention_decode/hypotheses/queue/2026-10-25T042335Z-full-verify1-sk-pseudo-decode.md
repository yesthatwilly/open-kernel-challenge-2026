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
review: proceed
status: hold
review_at: 2026-10-25T043143Z
review_companions: null
hold_reason: "Reviewer: low priority; dispatch after full-verify-wgmma-s4-multitile-fix (51b0d5c110f3459c) and after sliding-verify-sk-pseudo-decode-s1 harvest; build on s1 PSEUDO code with window=0; if the wgmma fix works, full-verify1-wgmma-s4-batch1-grid-sm owns this row."
---

# full-verify1-sk-pseudo-decode

## Change

Built on master (full-decode stream-K kernels landed, c3c1b01). Full verify_1 is 30.0 us (FA3 main 15.0 + combine 7.6 + prepare 2.7 + pre-attn + norm-gate) against a 4.3 us floor and is 38% of the verify_1 load. At full verify batch 1 only (window == 0, q_len 8, batch 1, tp8, page 1; config-only) route through the landed single-level stream-K decode kernel (splitkv_attn_sk_kernel <4 stages, ticket>, grid 2xSM) over 8 pseudo-requests, one per verify token: new template flag PSEUDO (false = landed code, byte-identical); pseudo-request j: seq_j = cache_seqlens[0] - 7 + j, keys [0, seq_j), page_table row 0, FP8 q = token j of master _pre_attention_kernel output (PAIRED=False, already writes the 8 new rows), norm-gate epilogue with qkv row j into out row j; tile prefix over 8 pseudo-requests from cache_seqlens; workspace grid + 8. Each 80K-key window is shared by all 8 pseudo-requests and stream-K places equal key offsets of different pseudo-requests on concurrently running CTAs, so DRAM traffic stays ~20 MB while L2 serves 8x (160 MB). This keeps the gather-bound 16-row mma.sync design (LESSONS: the 64-row verify tiling is MMA-bound on mma.sync). Prepare: time it vs master FA3 verify_1 path (prepare+main+combine+norm-gate) cold L2 median of 10, print both, route only if >= 5% faster (PATH CHOSEN with timings). Self-check vs fp32 causal reference: 80000, 4097, 63 and 8 keys, kv 1.0 and 0.25, grids 3/7/device, two replays. B > 1 verify and all decode unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## Expected

- `*layer=full load=verify_1 *` faster: replaced ~26.5 us (FA3 15.0 + combine 7.6 + prepare 2.7 + norm-gate) of 30.0; 160 MB L2 at ~9 TB/s ~18 us + merge, target shape <= 24 us
- `*layer=full load=verify_16* *layer=full load=verify_32* *load=decode_* *layer=sliding*` flat: outside the gate; PSEUDO=false is landed code

## Risk

Kill: prepare timing not >= 5% below FA3 (gate keeps FA3, flat). Risk: L2 thrash if CTAs for different pseudo-requests drift apart in key offset; a pseudo-request-interleaved tile order (tile index = key_tile*8 + j) is the fallback variant to time in the same prepare.

## Adversarial review

Reviewed 2026-10-25T043143Z by an adversarial subagent (model opus).

**Verdict: proceed.** The key-range algebra is correct and the change is rules-safe, but its physical floor is weak. 8 pseudo-requests x 80K keys is ~164 MB of L2-served cp.async, which at a realistic 4-7 TB/s is ~23-40 us against the ~26.5 us it replaces, so the 5% gate will likely keep FA3. Dispatch it after sliding-verify-sk-pseudo-decode-s1 (it shares the PSEUDO template, so build on that code) and after the in-flight full-verify-wgmma-s4-multitile-fix reports. If the wgmma verify kernel works, the queued full-verify1-wgmma-s4-batch1-grid-sm owns this row.

### Justification
Mechanism: keeping the gather-bound 16-row tile and moving each token's reuse to L2 avoids the 64-row MMA-bound verify tile. Request-major stream-K over 8 near-equal pseudo-requests (~16.5 CTAs each) naturally co-schedules equal key offsets, so DRAM traffic plausibly stays ~20 MB. WEAKEST LINK: the claim '160 MB L2 at ~9 TB/s ~18 us'. The landed kernel moves ~3 TB/s at B=16 and 0.89 TB/s at B=1 grid SM. Its rate is bounded by cp.async bytes in flight per SM, not by where the bytes come from, so L2 hits raise it by the latency ratio (~1.5-2x), not to 9 TB/s. At 2xSM and ~5 TB/s effective that is ~33 us before the merge, above the 26.5 us replaced. So the likely outcome is the gate keeping FA3, flat, which costs a check for a kill. Target <=24 us shape needs >=7 TB/s of sustained L2 gather. The wgmma verify line (stage 3 fixing in flight as full-verify-wgmma-s4-multitile-fix, then full-verify1-wgmma-s4-batch1-grid-sm) reads 20.5 MB once and has a ~8.5 us + ~2 us merge plan. It is the physically better route to this row. Not a duplicate, but strictly the backup.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (landed): the kernel being templated. PSEUDO=false must be byte-identical (print and compare sk_attrs). The change chooses the SINGLE-level ticket at grid 2xSM: ~33 partials per pseudo-request merged by one CTA, the same serial-merge shape that cost 34.8 vs 23.1 us at B=1 decode. Include the two-level variant (gticket sized 8*32) and grid SM in the prepare.
- sliding-verify-sk-pseudo-decode-s1 (queued, p1): the same PSEUDO template flag (it is this change plus a window lo_p). Two independent implementations of one flag will conflict at landing. Build this one on that branch's PSEUDO code with window=0, or fold it in as a fourth band there. The full verify_1 row stays separately attributable either way, since sliding rows and full rows are disjoint.
- full-verify1-cuda-streamk-batch1 (queued, built on the regressed A8 stage 2) and full-verify1-wgmma-s4-batch1-grid-sm (held on wgmma s3): same row, same B=1 gate. Never stack them; whichever runs, the others' code must be off this row.

### Interference and A/B needs
No companion. The prepare prints this kernel against FA3 (prepare+main+combine+norm-gate, pre-attn excluded) on the same input, routes only at >=5%, and PATH CHOSEN names the variant. The full verify_1 row is either this kernel or byte-identical FA3, so it is attributable alone.

### Open questions
- BUILDER/DISPATCH: wait for full-verify-wgmma-s4-multitile-fix (51b0d5c110f3459c). If its wgmma kernel passes and reaches >=2.4 TB/s at verify_16, prefer full-verify1-wgmma-s4-batch1-grid-sm for this row and drop this. Otherwise run this one, on sliding-verify-sk-pseudo-decode-s1's PSEUDO code.
- BUILDER: in the prepare, also time the pseudo-request-interleaved tile order (tile = key_tile*8 + j) named in Risk, and two-level vs single-level at grid SM and 2xSM. Print achieved effective bytes/us (164 MB / t) so the L2-bandwidth assumption is measured, not argued.

