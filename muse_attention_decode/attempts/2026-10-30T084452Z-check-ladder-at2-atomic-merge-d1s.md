# muse_attention_decode · 2026-10-30T084452Z · check · ladder-at2-atomic-merge-d1s

- **Job:** `6aa73de650b144c8`
- **File:** muse_attention_decode/submission.py at git `ladder/at2` 43c32389 (rebased on master 860ca13d, board 1.715), 589,247 B

## Goal
Ladder rung AT2 (campaign 2026-10-30-public-ladder, user-requested). Target is sliding decode_1 (d1s): ours 10.8 us on
the judge, of which sliding_cluster_kernel<PERHEAD> is 9.6 us; the public kernel takes 5.2 us.
- Pre-registered: -2.4 us on H100 (judge-scaled about -2.2 us), headline about +1.9%.
- H100 measured: -2.29 us (x1.17) kernel A/B, and x1.181 in the in-import race.

## What changed, and where
- New CUDA kernel `sliding_at2_kernel` in its own module `muse_sliding_at2_v1`. The source is the PERHEAD module's with
  the kernel inserted; the master and PERHEAD module sources are byte-identical.
  - Plain split-KV: 32 CTAs x one 64-key tile, both kv heads, no cluster launch.
  - Queries, gather, attend_tile, intra-CTA merge and the new K/V store are master's code.
  - Fixed-reference (R = 16) FP32 red.global.add merge into a persistent per-request workspace, with the R11 guard
    (exact per-split fallback slots plus a fallback-row counter).
  - fence.acq_rel and one atom.acq_rel ticket per CTA.
  - Last-CTA finalize: v_scale, RMSNorm, sigmoid(gate), bf16 out. It self-resets the workspace, slots and ticket.
  - The gate row is L2-prefetched by the first 16 CTAs.
- `_slide_at2_*` Python:
  - Build, canary, self-check at 10 lengths vs FP32 and vs the incumbent; the forced and mixed guard arms; workspace
    clean after every launch.
  - Two-state graph-replay check through kernel_fn.
  - Interleaved cold-L2 race vs the routed incumbent (PERHEAD, else C=16) at B=1 x 80000, routed at >= 1.03x.
  - Starts only within the 480 s import-clock rule (deadline 120 s).
- `_slide_forward`: B=1 takes AT2 when the import-time flag is set; else the incumbent, unchanged.

## Rules check
- [x] CUDA-graph capturable: one kernel launch with fixed grid (B x 32). No host sync, no value-dependent allocation or
      control flow. The workspace is allocated once at import.
- [x] Nothing derived from `data` is cached across calls. The workspace, slots and ticket return to zero / -inf / 0 at
      the end of every launch (self-reset, checked after every self-check launch and after the race).
- [x] Weights: not touched (attention has none).
- [x] KV cache stays FP8 E4M3, queries E4M3 (unchanged code), output BF16. The merge is FP32.
- [x] Every key in the window is attended exactly once (32 x 64 tiles cover [lo, seq), plus the new key). No skipping;
      the softmax and PV run over every key. The R11 guard never drops mass: it reroutes a row to an exact slot.
- [x] No timing patches, harness access, reference import, subprocesses or network.
- [x] All work happens inside the timed call. No output caching.

## Risks
- **Prepare/import time.** Import grows by about 40-60 s on the node (306.7 s cold there). The AT2 phase starts only
  within the 480 s import-clock budget; otherwise PERHEAD stays.
- **The judge's H200 race might fall below 1.03x.** In that case PERHEAD keeps the row, at no loss.
- **Gated to B=1 only.** Every other row is unchanged.

## Expected PATH lines
- `[slide-gate] AT2-vs-incumbent B=1 x 80000: ... (ROUTE AT2)`
- `[slide] PATH CHOSEN: sliding B=1 AT2 ON: split-KV 32 x 64 keys, fixed-reference atomic merge`

## Result
- job `6aa73de650b144c8` (check), passed 16/16, **1.743x [1.717, 1.770]** vs board 1.715 (+1.6%): d1s 8.9 us (was 10.8; public 5.2); judge log: PATH CHOSEN sliding B=1 AT2 ON.
