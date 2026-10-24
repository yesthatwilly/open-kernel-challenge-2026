---
problem: muse_attention_decode
slug: stream-fork-preattention
created: 2026-10-23T225858Z
base: master
area: launch overhead
change: >
  Keep every launch of the parity pipeline but put the independent pre-attention launches on side
  streams so the captured CUDA graph has parallel branches and a shorter critical path. Two
  module-level side streams s_v and s_q are created at import (torch.cuda.Stream(), from nothing
  in data). In kernel_fn: cur = torch.cuda.current_stream(); s_v.wait_stream(cur); under
  torch.cuda.stream(s_v): v8 = (v / v_scale if scaled else v).to(float8_e4m3fn).view(uint8),
  reading its slice of qkv (it does not depend on FusedQKNorm). On cur: q, k = FusedQKNorm(...) as
  before. s_q.wait_stream(cur); under torch.cuda.stream(s_q): q8 = q.view(tokens, Hq,
  64).to(float8_e4m3fn). On cur: k8 = k.reshape(tokens,
  Hkv*64)(/k_scale).to(float8_e4m3fn).view(uint8); cur.wait_stream(s_v); store_cache(k8, v8, ...);
  cur.wait_stream(s_q); flash_attn_with_kvcache(q=q8, ...) and rms_norm_gate exactly as before.
  Call v8.record_stream(cur) and q8.record_stream(cur) right after producing them so the caching
  allocator does not reuse their memory early. The serial chain goes from 9 launches (FusedQKNorm,
  k cast, v cast, store_cache, q cast, FA3 prepare, FA3 main, FA3 combine, rms_norm_gate) to 7 on
  the critical path (the v cast overlaps FusedQKNorm, the q cast overlaps the k cast and
  store_cache). No kernel changes; the test is whether the judge's graph replay honours branch
  parallelism and what two launches of overlap are worth. Full and sliding layers both get it (the
  pre-attention stages are the same on both).
expected:
  - shapes: "*layer=sliding load=decode_1 page*"
    direction: faster
    why: "21 us launch-bound shape: two ~2.3 us launches leave the critical path"
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: faster
    why: same, 23 us
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 27 us, same two launches overlapped
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 31.5 us, same
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same mechanism at q_len 8
  - shapes: "*layer=full load=decode_1 page*"
    direction: faster
    why: 30.8 us, same two launches overlapped; FA3 unchanged
  - shapes: "*layer=full load=decode_16*"
    direction: flat
    why: 201 us FA3-bound; 4-5 us of overlap is inside the CI
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: 388 us FA3-bound
  - shapes: "*layer=full load=verify_16*"
    direction: flat
    why: FA3-bound
  - shapes: "*layer=full load=verify_32*"
    direction: flat
    why: FA3-bound
risk: medium
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: required
status: hold
hold_reason: "moot under master 1.269x (sweep by the attention minter): targets the parity pipeline's five pre-attention launches, which master fused into one kernel"
---

# stream-fork-preattention

## Change

Keep every launch of the parity pipeline but put the independent pre-attention launches on side streams so the captured CUDA graph has parallel branches and a shorter critical path. Two module-level side streams s_v and s_q are created at import (torch.cuda.Stream(), from nothing in data). In kernel_fn: cur = torch.cuda.current_stream(); s_v.wait_stream(cur); under torch.cuda.stream(s_v): v8 = (v / v_scale if scaled else v).to(float8_e4m3fn).view(uint8), reading its slice of qkv (it does not depend on FusedQKNorm). On cur: q, k = FusedQKNorm(...) as before. s_q.wait_stream(cur); under torch.cuda.stream(s_q): q8 = q.view(tokens, Hq, 64).to(float8_e4m3fn). On cur: k8 = k.reshape(tokens, Hkv*64)(/k_scale).to(float8_e4m3fn).view(uint8); cur.wait_stream(s_v); store_cache(k8, v8, ...); cur.wait_stream(s_q); flash_attn_with_kvcache(q=q8, ...) and rms_norm_gate exactly as before. Call v8.record_stream(cur) and q8.record_stream(cur) right after producing them so the caching allocator does not reuse their memory early. The serial chain goes from 9 launches (FusedQKNorm, k cast, v cast, store_cache, q cast, FA3 prepare, FA3 main, FA3 combine, rms_norm_gate) to 7 on the critical path (the v cast overlaps FusedQKNorm, the q cast overlaps the k cast and store_cache). No kernel changes; the test is whether the judge's graph replay honours branch parallelism and what two launches of overlap are worth. Full and sliding layers both get it (the pre-attention stages are the same on both).

## Expected

- `*layer=sliding load=decode_1 page*` faster: 21 us launch-bound shape: two ~2.3 us launches leave the critical path
- `*layer=sliding load=verify_1 page*` faster: same, 23 us
- `*layer=sliding load=decode_16*` faster: 27 us, same two launches overlapped
- `*layer=sliding load=decode_32*` faster: 31.5 us, same
- `*layer=sliding load=verify_*` faster: same mechanism at q_len 8
- `*layer=full load=decode_1 page*` faster: 30.8 us, same two launches overlapped; FA3 unchanged
- `*layer=full load=decode_16*` flat: 201 us FA3-bound; 4-5 us of overlap is inside the CI
- `*layer=full load=decode_32*` flat: 388 us FA3-bound
- `*layer=full load=verify_16*` flat: FA3-bound
- `*layer=full load=verify_32*` flat: FA3-bound

## Risk

Medium risk, entirely about the harness: multi-stream work inside torch.cuda.graph capture is supported (the capture follows event dependencies and all side streams join cur before the end), but if the judge captures with a mode that forbids side streams, or replays the graph serially, the result is flat and the hypothesis is rejected cleanly. Correctness is unaffected: the same kernels on the same data; the only hazard is a missing wait_stream, which would read an FP8 buffer before its cast (store_cache before v8, FA3 before q8) and would show as a cache or output mismatch in the tests. Subsumed by fused-preattention-triton if that lands (one pre-attention launch leaves nothing to overlap), so this is only worth running before or beside it; a gain here also tells whether the FA3 prepare/combine launches could later be overlapped with the q-side work. Must create the streams at import, never under capture.
