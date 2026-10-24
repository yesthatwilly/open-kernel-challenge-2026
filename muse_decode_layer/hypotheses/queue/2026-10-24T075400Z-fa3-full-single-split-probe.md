---
problem: muse_decode_layer
slug: fa3-full-single-split-probe
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  Sensitivity probe re-minted from the held fa3-num-splits-full-no-metadata review: an explicit
  num_splits can only CAP FA3's device-side dynamic split (min(dynamic, static)), never raise it,
  so this caps BELOW the dynamic pick. On full-attention layers only (cfg['window'] == 0, both
  q_len=1 decode and q_len=8 verify), call flash_attn_with_kvcache with num_splits=1 and
  scheduler_metadata=None (the judge's metadata encodes its own split schedule built with
  num_splits=0, so it must be dropped for the cap to take effect; verify blocks already pass
  None). Every other argument of the call, the sliding branch and everything outside the FA3 call
  stay exactly as on master. Purpose: learn whether full decode attention at page size 1 is
  gather-pattern-bound or parallelism (memory-level-parallelism) bound before spending runs on a
  custom kernel (triton-flash-decode-full-paired-kv) or trtllm-decode-full-attention. Grids become
  64 CTAs at B=32, 32 at B=16, 2 at B=1 (no partials, no FlashAttnFwdCombine). Reading: if
  decode_32 full FA3 stays near its 491.6 us (1.33 TB/s) with half the CTAs, per-CTA gather
  latency/pattern is the limiter and the fix must change the access pattern (both heads' 128-byte
  row per token, deeper pipelining); if it roughly doubles, in-flight parallelism matters and a
  kernel that oversubscribes the SMs (more splits) is the lever. This is a check-run diagnostic
  and is expected to lose on the B=1 full shapes.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: "the diagnostic: 64 CTAs instead of ~128; flat means gather-pattern-bound, ~2x FA3 time (~+490 us) means MLP-bound; either is a large readable move vs a 491.6 us FA3"
  - shapes: "*layer=full load=decode_16*"
    direction: slower
    why: "32 CTAs on 132 SMs: under a quarter wave, slower unless the per-CTA path can pull 4x its current bandwidth"
  - shapes: "*layer=full load=verify_32*"
    direction: flat
    why: same probe with 8 queries per request (metadata already None); compare its move with decode_32 to see whether the q_len=8 tile changes the bound
  - shapes: "*layer=full load=verify_16*"
    direction: slower
    why: 32 CTAs, as decode_16
  - shapes: "*layer=full load=decode_1*"
    direction: slower
    why: "2 CTAs over 625 key tiles of 128: FA3 18 us -> roughly 10x, minus the 6.9 us combine"
  - shapes: "*layer=full load=verify_1*"
    direction: slower
    why: 2 CTAs, as decode_1
  - shapes: "*layer=sliding*"
    direction: flat
    why: sliding branch untouched; control
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-full-single-split-probe

## Change

Sensitivity probe re-minted from the held fa3-num-splits-full-no-metadata review: an explicit num_splits can only CAP FA3's device-side dynamic split (min(dynamic, static)), never raise it, so this caps BELOW the dynamic pick. On full-attention layers only (cfg['window'] == 0, both q_len=1 decode and q_len=8 verify), call flash_attn_with_kvcache with num_splits=1 and scheduler_metadata=None (the judge's metadata encodes its own split schedule built with num_splits=0, so it must be dropped for the cap to take effect; verify blocks already pass None). Every other argument of the call, the sliding branch and everything outside the FA3 call stay exactly as on master. Purpose: learn whether full decode attention at page size 1 is gather-pattern-bound or parallelism (memory-level-parallelism) bound before spending runs on a custom kernel (triton-flash-decode-full-paired-kv) or trtllm-decode-full-attention. Grids become 64 CTAs at B=32, 32 at B=16, 2 at B=1 (no partials, no FlashAttnFwdCombine). Reading: if decode_32 full FA3 stays near its 491.6 us (1.33 TB/s) with half the CTAs, per-CTA gather latency/pattern is the limiter and the fix must change the access pattern (both heads' 128-byte row per token, deeper pipelining); if it roughly doubles, in-flight parallelism matters and a kernel that oversubscribes the SMs (more splits) is the lever. This is a check-run diagnostic and is expected to lose on the B=1 full shapes.

## Expected

- `*layer=full load=decode_32*` flat: the diagnostic: 64 CTAs instead of ~128; flat means gather-pattern-bound, ~2x FA3 time (~+490 us) means MLP-bound; either is a large readable move vs a 491.6 us FA3
- `*layer=full load=decode_16*` slower: 32 CTAs on 132 SMs: under a quarter wave, slower unless the per-CTA path can pull 4x its current bandwidth
- `*layer=full load=verify_32*` flat: same probe with 8 queries per request (metadata already None); compare its move with decode_32 to see whether the q_len=8 tile changes the bound
- `*layer=full load=verify_16*` slower: 32 CTAs, as decode_16
- `*layer=full load=decode_1*` slower: 2 CTAs over 625 key tiles of 128: FA3 18 us -> roughly 10x, minus the 6.9 us combine
- `*layer=full load=verify_1*` slower: 2 CTAs, as decode_1
- `*layer=sliding*` flat: sliding branch untouched; control

## Risk

This run is expected to lower the overall speedup (it is information, not a candidate for master: never promote it). The length draws differ per shape (spread=agentic, per-shape seeds), so compare each full shape with its own base row, not decode with verify. Confirm in prepare() via a print of sgl_kernel.__version__ that the judge's FA3 accepts num_splits with cache_seqlens (varlen dynamic split); if num_splits=1 with scheduler_metadata=None is rejected, the run fails fast in the tests. The 1M-token full verify test (batch 1, ctx 1048576, q_len 8) runs at 2 CTAs: 8192 key tiles serially per CTA, a few ms, well inside the 15-minute budget. Correctness: no split means no fp32 combine; numerics only improve.
