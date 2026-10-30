# muse_attention_decode · 2026-10-30T094755Z · submit · ladder-at8-atomic-merge-d16s

- **Job:** `69342c7ddbae4257`
- **File:** muse_attention_decode/submission.py at git `ladder/at8` 20b5de80 (on 43c32389 = AT2, byte-identical to
  master e2590b3c, board 1.757), 602,000 B

## Goal
Ladder rung AT8 (campaign 2026-10-30-public-ladder, user-requested). Target is sliding decode_16 (d16s): ours 15.4 us
on the judge (master Triton `_sliding_fused_kernel`, 12.7 us in the profile); public 7.9 us.
- **Pre-registered:** -4.0 us on H100 with a 2-CTA/SM grid. That grid **missed** at -0.8 us.
- **Amended, post hoc:** a 1-CTA/SM grid, confirmed on fresh seeds at -1.60 / -1.68 us (x1.09). The in-import race
  measured x1.133.
- **Expected on the judge:** d16s about -1.0 to -1.5 us (15.4 -> ~14.0-14.4), headline about +0.6-0.9%, under one run's
  margin. Read this check per row.

## What changed, and where
- `sliding_at2_kernel` became `template <bool WIDE>` in the AT2 module (renamed `muse_sliding_at2_v2`).
  - WIDE=false is AT2's B=1 code, unchanged.
  - WIDE=true takes `tile` old keys per CTA: ceil(tile/128) ring slots in one cp.async group, each warp walks tiles
    sub, sub+4, ..., and the gate lines are prefetched across the splits.
  - Host `slide_at2_attn` takes `tile`: 64 runs AT2's instance, any other value runs WIDE. `slide_at2_setup` reports
    the attributes of both instances.
- AT2's merge, R11 guard, ticket and self-reset finalize are applied per request: own workspace rows, slots and ticket.
- `_slide_at8_*` Python. It runs after the AT2 phase and reuses that module, with no second build:
  - Grid rule `_slide_at8_tile`: the smallest tile in 64/128/192/256/384 with B x ceil(2047/tile) <= SMs.
  - Self-check over 8 cases at B=2..16 (at8, forced and mixed guard, tile-128 arms), against FP32 and against master
    Triton. Cache rows must equal master's, and the workspace must be clean after every launch.
  - Two-state graph-replay check through kernel_fn at B=16.
  - Interleaved cold-L2 race against master Triton at B=16 x 80000, routed at >= 1.03x.
  - Starts only within the 480 s import-clock rule (deadline 60 s; it takes about 1 s).
- `_slide_at2_clean` handles B=1 and B<=16 buffers.
- `kernel_fn`: sliding decode B=2..16 takes AT8 when the frozen import flag is set. Otherwise master Triton runs,
  unchanged.

## Rules check
- [x] CUDA-graph capturable: one launch with a fixed grid (B x ceil(2047/tile), tile from config only). No host sync,
      no value-dependent allocation or control flow. The workspace is allocated once at import.
- [x] Nothing derived from `data` is cached across calls. The workspace, slots and tickets return to 0 / -inf / 0 at
      the end of every launch (self-reset, checked after every self-check launch, after the replay check and after
      the race).
- [x] Weights: not touched.
- [x] KV cache stays FP8 E4M3 and queries E4M3 (AT2's code, unchanged), output BF16. The merge is FP32, and the softmax
      denominator is FP32 from unrounded P.
- [x] Every window key is attended exactly once: the splits cover [lo, seq) disjointly, plus the new key. No skipping.
      The R11 guard reroutes a row to an exact slot and never drops mass.
- [x] No timing patches, harness access, reference import, subprocesses or network.
- [x] All work happens inside the timed call. No output caching.

## Risks
- **The judge's H200 race could fall below 1.03x.** In that case master Triton keeps B=2..16, at no loss.
- **The gain is small** (about 1 us on one row); expect the headline inside run noise.
- **Import:** the module build grows by at most about 1.5 s (one extra instance) and the AT8 phase takes about 1 s.
  The node import was 316 s.

## Expected PATH lines
- `[slide-gate] AT8-vs-master B=16 x 80000: ... (ROUTE AT8)`
- `[slide] PATH CHOSEN: sliding B=2..16 AT8 ON: split-KV atomic merge, keys/CTA {2: 64, ..., 16: 256}`
- `[slide] PATH CHOSEN: sliding B=1 AT2 ON: ...` (unchanged)


## Why a scored run
Check 91664bcc 1.776x beats the board 1.757, same bytes 20b5de80. Standing rule: scored submit + promotion when a check beats the board.

## Result (after the run)
- job `69342c7ddbae4257` (submit), passed, **1.760x [1.733, 1.786]** vs board 1.757 -> landed on master (bytes 20b5de80).
