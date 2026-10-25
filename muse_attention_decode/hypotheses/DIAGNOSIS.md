---
diagnosed_at: 2026-10-25T203356Z
problem: muse_attention_decode
best: 1.5295x [1.5066,1.5525], job 8713665f3064462c (submit, combo-b1-chain-c320-fused-pre-pdl, branch code, not master). merge-perkvhead-c320-sv32-tiled scored 1.5347 (a78f29aeab9b47cb), which is inside the margin.
---

# muse_attention_decode diagnosis

## Diagnosis
**Wrong bottleneck (4).** The two verify loads hold the geomean down. Speedup per load from the BEST rows: verify_16 1.270x and verify_32 1.269x, against 1.71-1.79x for decode_16/32. Full verify_16/32 has run paired FA3 since 10-24, at 148.7/288.7 µs (about 2.2 TB/s). That is 2.3x its floor, while the CUDA stream-K decode kernel moves the same bytes at 3.05 TB/s. Most of the day's runs instead went to rows that already score well:
- **Repetition loop (1):** about 10 runs on full decode_1 (`full-decode1-cuda-splitkv-chunk320`/`-chunk640`, `full-decode1-streamk-s3-two-level-ticket`, `full-decode1-sk4-cluster-dsmem-merge`, `full-decode1-sk-two-level-fused-preattn`, `full-decode1-sk-pdl-from-preattn`, three `combo-*`, `merge-perkvhead-c320-sv32-tiled`, with `full-decode1-sk-two-level-deep-ring` queued).
- **Repetition loop (1), pre-attention:** five pre-attention regrids (`preattention-group-tiled-programs`, `preattention-tiled-gate-large-tokens`, `combo-naive-*`, `combo-tiled-*`, `merge-*`), with `preattention-tiled-every-verify-block` queued.
- **Local minimum (2):** the last seven headlines all fall between 1.5017 and 1.5347 on one design: FA3 verify plus CUDA decode.

**Build-churn (3)** also applies. Of about 15 CUDA load_inline runs, five measured the fallback or failed:
- wgmma s3 and s4 failed their self-checks.
- s5 was correct but gated off.
- `full-decode-sk-s4-deferred-ticket-arrive` hit an illegal address.
- `sliding-verify-sk-pseudo-s2-token-group` raised a NameError.

The import clock is already 222 s across 4 modules.

**Unstacked wins (7):** the B=1 V3 chain (0.791) is not yet on per-kvhead master.

## Strategy
**pivot**: move the check budget from the decode-side combos, which sit within the margin of one another, to a wgmma verify kernel. Its ceiling is decode-level bandwidth on the verify rows, worth about +5-6.5% headline. The best decode-side combo might add about 1%.

## Actions (priority ordered; /hypothesize mints these first)
1. **What:** full-layer verify for 1<B≤64 (q_len 8, page 1, tp8) gets a wgmma kernel with an overlapped pipeline, replacing paired FA3 + prepare + combine + norm-gate. Dispatch the queued `full-verify-wgmma-ws-s1-producer-split` with two additions:
   - (a) A gather-only roofline DIAG for each design (no MMA, same ring) at B=16 x 80K, so the producer's issue rate is measured before the compute is blamed.
   - (b) A second variant that keeps s5's proven layout (2 CTAs/SM, all 256 threads issue cp.async) but splits the ring per warpgroup, one per kv head, with no CTA-wide EMPTY rendezvous. It issues S(n+1) before softmax(n), with `wgmma.wait_group 1`.

   Gate both variants at ≥1.05x against paired FA3, as s5 did.

   **Why:** `full-verify-wgmma-s5-qfrag-reload` (71e40388) passed all 19 bisect cases and the two-state replays. It never routed because it was slow:
   - B=16: 288.7 vs FA3 164.4 µs.
   - B=32: 429.5 vs 302.4 µs.
   - Twice the bytes cost only 1.49x the time, so the kernel is latency-bound in its serial per-tile chain, not bandwidth-bound.
   - The design risk in WS-s1 is its producer width. A 1-CTA/SM single-producer-warp gather capped at 1.77 TB/s (`full-cuda-onelaunch-warpspec-bulk-gather` BULK=0), and WS-s1 has 128 issuers per SM against master decode's 512. Hence (a) and (b).
   - The compute fits: about 42 GFLOP at B=16 is about 21 µs of FP8 wgmma, and about 164M exps is about 40 µs of MUFU. Both hide under a 108 µs gather once they overlap.

   **Impact:** shape 162.5→≈115 µs at full verify_16 (0.073%/µs) and 302→≈225 µs at verify_32 (0.043%/µs). That is about +3.4% plus +3.3% at decode parity, or about +5% at the M2 targets.

   **Prior:** `full-verify-wgmma-s5-qfrag-reload`. It had the same math with a serial S→transpose→softmax→PV chain and a per-tile CTA rendezvous.

2. **What:** the queued `sliding-verify-wgmma-window-s1`: the s5 kernel plus a `window` argument for sliding verify B=2..32, with pre-attention kept and a per-band gate. **Why:** the master sliding verify_16 path is prepare 1.9 + FA3 10.7 + combine 4.0 + norm-gate 1.4 (= 18.0 µs replaced). B=16 has only 528 tiles, about 2 per CTA, so s5's slow per-tile chain costs little here. The window rows carry weight 46 (0.21%/µs at verify_16, 0.124%/µs at verify_32). **Impact:** at M2, verify_16 22.6→≈17 µs and verify_32 29→≈21.6 µs, about +2.1% headline. **Prior:** `sliding-verify-sk-pseudo-decode-s1` used mma.sync and lost at B=16/32 (0.653x/0.482x); wgmma removes that compute wall.

3. **What:** the B=1 launch chain, which is cheap and uses a proven mechanism:
   - (i) Hand-merge the `combo-b1-chain-c320-fused-pre-pdl` V3 chain (c320 + fused pre-attention + PDL) onto per-kvhead master, together with SV32 tiled. This is the PENDING_USER offer.
   - (ii) Apply the same fused-pre-attention + PDL chain to the sliding verify_1 PSEUDO route. Today that route is sk 9.9 + `_pre_attention_kernel` 1.5 µs plus a node gap, 13.2 µs shape.

   **Why:** V3 measured 18.98 vs V1 21.41 µs, and the gain compounded beyond its parts (LESSONS). Sliding verify_1 is worth 0.70%/µs, the second-highest rate on the board. **Impact:** full decode_1 −2.4 µs (+0.8%), sliding verify_1 ≈−2 µs (+1.4%). **Prior:** `merge-perkvhead-c320-sv32-tiled` carries only c320 V1, and `sliding-decode-sk-s3-fused-preattn-rope` folds pre-attention on decode, not on verify_1.

4. **What:** fix `full-decode-sk-s4-deferred-ticket-arrive`, which faulted with an illegal address (0/16 tests, first failure batch 1 ctx 10 sliding). That points to a fault at import, not a design loss. Rerun it as a check with a sync and print after each self-check launch. **Why:** the sk-diag shows the ticket epilogue costs 112.7 vs 106.9 µs no-ticket at B=16 and 221.4 vs 218.6 µs at B=32. It is the only untested lever on full decode B≥2, where ring and chunk knobs are exhausted. **Impact:** about −5 µs at full decode_16 (0.107%/µs) and −3 µs at decode_32, about +0.7%. **Prior:** same slug; the mechanism has never been measured.

5. **What:** once action 1 routes, add a B=1 grid-SM two-level-ticket variant of the verify kernel for full verify_1, which is 30.3 µs against a 4.3 µs floor. Until then, the queued `full-verify-fa3-num-splits-band-gate` is the cheap stopgap for its 7.2 µs combine. **Why:** mma.sync token-group lost at 0.596x (`full-verify1-sk-two-level-token-group`). The decode_1 two-level ticket took 27.3→22.6 µs. **Impact:** ≈−8 µs at full verify_1 (0.24%/µs), about +2%. **Prior:** `full-verify1-wgmma-s4-batch1-grid-sm`, held; unhold it on top of action 1.

## Do not try
- `sliding-decode-paired-kvhead-rows`, `sliding-verify-fa3-head-pairing`, and the queued `sliding-decode-paired-kvhead-rows-control-unpaired-s8`: pairing kv-head rows on SLIDING layers doubled the rows per kv head. Kernel 12.9→16.8 µs, plus about 11 µs of overhead, and FA3 combine 4.1→7.6 µs. Retire the queued control.
- mma.sync kernels for q_len-8 verify at B≥16 (`full-verify-cuda-streamk-splitkv` 1.44 TB/s, `sliding-verify-sk-pseudo-decode-s1` B=16/32 0.653x/0.482x, `sliding-verify-sk-pseudo-s2-token-group` 0.82x/0.70x, `full-verify1-sk-two-level-token-group` 0.596x): 64 query rows per kv head make mma.sync plus softmax compute-serialised. The wgmma line replaces this approach; only B=1 sliding verify won.
- `full-cuda-onelaunch-warpspec-bulk-gather`: one `cp.async.bulk` per 128-B row, issued by a single producer warp, is issue-bound at 0.49 TB/s (BULK=0 1.77). Sm_90 has no TMA gather for page-1 rows.
- Full decode B≥2 ring depth, stages and chunk size (`splitkv-chunk2048-batch32`, 5-stage 0.982-0.995x, `splitkv-pdl-attn-to-combine`, `splitkv-combine-wide-mlp`): each moved ±0.3 µs. The kernel sits at 3.0 TB/s, near the random-128-B gather limit.
- FUSED_PRE for full decode B≥2 (`full-decode-splitkv-s3-fused-preattn` re-gated in `restack-sk-pseudo-cluster-plus-v3-fused-preattn`: 0.988x/1.014x): the fold costs as much as the launch it saves.
- `fa3-window-trimmed-page-table`, `fa3-scheduler-metadata-side-stream`: an extra launch or side stream costs more than the FA3 work it trims.
- Pre-attention regrids beyond SV32 (`preattention-group-tiled-programs`, `preattention-tiled-gate-large-tokens`; the combo regate kept every class except SV32 on master): at most 1 µs per row. Deprioritise the queued `preattention-tiled-every-verify-block` below actions 1-3.
- `epilogue-gate-l2-prefetch-cuda`: flat. The gate operand is not on the critical path.

## LESSONS edits (proposed; churnsubs applies them)
- correct: `Full-attention decode reaches only 1.3-1.7 TB/s: the page-size-1 row gather is the limit.` → This is FA3/Triton only. The CUDA stream-K kernel reaches 3.0-3.07 TB/s on the same gather (`full-decode-splitkv-s2-streamk-ticket-inproc` sk-diag), and FA3 verify runs at about 2.2 TB/s.
- correct: `... Head pairing is dead on this problem.` (2026-10-25 03:41Z harvest) → "kv-head row pairing is dead on SLIDING layers". Full-layer FA3 pairing at B>1 is master (`fa3-hdim128-head-pairing`, `paired-heads-launch-cleanup`), so the line as written contradicts the FA3 head-pairing lesson.
- correct: `In the B=2..16 band, neither the cluster ... nor a batched Triton fused kernel ... beats FA3. Leave B=2..16 on FA3.` → On attention, B=2..16 sliding decode runs the Triton deep-splits kernel (`_sliding_fused_kernel` 12.7 µs at decode_16, board profile), not FA3. "FA3" holds for decode_layer only. Say "leave B=2..16 on the incumbent".
- merge: `the mma.sync split-KV design ... is 1.7-1.8x SLOWER than FA3 on q_len-8 verify` (decode `full-verify-cuda-splitkv-stage2`) with `attention: full batch-1 decode through the fused Triton kernel ...` → one verify line that adds the attention evidence (pseudo-decode 0.653x/0.482x at B=16/32, token-group 0.596x at full verify_1) and the s5 fact: a correct FP8 wgmma verify kernel with a serial per-tile chain ran at 0.570x/0.704x of FA3, with time growing 1.49x for 2x the bytes, so it is latency-bound, not bandwidth-bound (`full-verify-wgmma-s5-qfrag-reload` 71e40388).
