---
problem: muse_attention_decode
slug: fa3-single-split-sliding
created: 2026-10-23T225858Z
base: master
area: FA3 call
change: >
  On sliding-window layers only (config window != 0) pass num_splits=1 to flash_attn_with_kvcache;
  everything else in the call (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
  softmax_scale, causal, window_size=(window-1, 0), scheduler_metadata=None, descale) stays as in
  the parity pipeline, and full layers are untouched. Mechanism (launch count, not balance): with
  num_splits left to FA3 on a sliding layer, FA3 runs its dynamic split-KV path, which is up to
  three launches (prepare_varlen_num_blocks, the split main kernel writing fp32 partials and LSE,
  flash_fwd_combine); with num_splits=1 the main kernel writes the bf16 output directly and the
  combine is skipped, and with an explicit split count the dynamic prepare pass has nothing to
  decide. On a 2,048-token window each (request, kv head) tile streams at most 16 blocks of 128
  rows x 128 B = 256 KB, so a single CTA per tile finishes in a few microseconds; the trade is
  that single-CTA streaming time against two launches of ~2.3 us each on shapes that are 21-35 us
  and launch-bound. This differs from fa3-num-splits-sliding, which keeps several splits per tile
  (16/9/5) and tests balance; this tests whether any split at all is worth its combine on sliding
  layers.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 32 tiles x 256 KB on 32 SMs is ~3-4 us of streaming; the combine and prepare launches (~4-5 us) go
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 64 tiles, same mechanism
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same tiles at q_len 8; the FA3 per-tile compute at 64 packed query rows is still memory-trivial
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=decode_1 page*"
    direction: flat
    why: "only 2 CTAs stream 512 KB in total: each pulls 256 KB alone, ~3-5 us, about what the two launches cost; a loss here attributes to single-CTA bandwidth"
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: flat
    why: same as decode_1
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: low
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T132543Z
review_companions: null
hold_reason: "reviewer expects an overall loss: one CTA per (request, KV head) streams 16-17 serial blocks, likely losing 8-12 us on sliding verify_1 and slowing decode_16/verify; only decode_32/verify_32 might gain; attention has 2 checks"
outcome: expired
verdict_detail: "hold expired after 32 h: reviewer expects an overall loss: one CTA per (request, KV head) streams 16-17 serial blocks, likely losing 8-12 us on s"
expired_at: 2026-10-25T211327Z
---

# fa3-single-split-sliding

## Change

On sliding-window layers only (config window != 0) pass num_splits=1 to flash_attn_with_kvcache; everything else in the call (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, causal, window_size=(window-1, 0), scheduler_metadata=None, descale) stays as in the parity pipeline, and full layers are untouched. Mechanism (launch count, not balance): with num_splits left to FA3 on a sliding layer, FA3 runs its dynamic split-KV path, which is up to three launches (prepare_varlen_num_blocks, the split main kernel writing fp32 partials and LSE, flash_fwd_combine); with num_splits=1 the main kernel writes the bf16 output directly and the combine is skipped, and with an explicit split count the dynamic prepare pass has nothing to decide. On a 2,048-token window each (request, kv head) tile streams at most 16 blocks of 128 rows x 128 B = 256 KB, so a single CTA per tile finishes in a few microseconds; the trade is that single-CTA streaming time against two launches of ~2.3 us each on shapes that are 21-35 us and launch-bound. This differs from fa3-num-splits-sliding, which keeps several splits per tile (16/9/5) and tests balance; this tests whether any split at all is worth its combine on sliding layers.

## Expected

- `*layer=sliding load=decode_16*` faster: 32 tiles x 256 KB on 32 SMs is ~3-4 us of streaming; the combine and prepare launches (~4-5 us) go
- `*layer=sliding load=decode_32*` faster: 64 tiles, same mechanism
- `*layer=sliding load=verify_16*` faster: same tiles at q_len 8; the FA3 per-tile compute at 64 packed query rows is still memory-trivial
- `*layer=sliding load=verify_32*` faster: same
- `*layer=sliding load=decode_1 page*` flat: only 2 CTAs stream 512 KB in total: each pulls 256 KB alone, ~3-5 us, about what the two launches cost; a loss here attributes to single-CTA bandwidth
- `*layer=sliding load=verify_1 page*` flat: same as decode_1
- `*layer=full*` flat: untouched; control

## Risk

Low risk, one keyword argument; correctness unaffected (reduction order only). If FA3 still launches the prepare kernel with an explicit num_splits (it may, for the varlen tile scheduler's semaphore), only the combine launch is saved and the gain halves. If FA3's dynamic path already picked 1 split on sliding layers at batch 32, decode_32/verify_32 come out flat and the gain is confined to the smaller batches. Interacts with fa3-window-trimmed-page-table and fa3-num-splits-sliding (all three touch the sliding split schedule); each is a one-area change on the parity base so they attribute separately, and the reviewer may want them run in sequence rather than stacked.

## Adversarial review

Reviewed 2026-10-24T132543Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-keyword, rules-clean probe that still targets 5 live sliding FA3 shapes (decode_16/32, verify_1/16/32; sliding decode_1 now bypasses FA3). Shapes are config-branched and independent, so one run is readable per shape without companions. The headline 'faster' expectations are probably wrong: the profile suggests the split is worth more than its combine. Run it as a cheap, readable test of whether splitting pays for its combine on sliding layers, not as a likely gain.

### Justification
Mechanism, partly right: under varlen dynamic split an explicit num_splits is only a cap (LESSONS; the decode_layer fa3-full-single-split-probe review). But a cap of 1 does bind. Split=false, so the main kernel writes bf16 directly and FlashAttnFwdCombine is skipped. That is the one case where the 'only a cap' hold reasoning that parked fa3-num-splits-sliding does not apply. prepare_varlen_num_blocks will almost certainly still launch, because cu_seqlens_q (varlen) plus a local window needs the tile scheduler's semaphore and the dynamic path. So the most that can be saved is the combine: 3.1-4.2 us per shape in the current-best profile (decode_16 3.2, decode_32 3.3, verify_1 3.7, verify_16 4.2, verify_32 4.1 us), not the 4-5 us the hypothesis claims. The hypothesis's '~2.3 us per launch' also does not match the measured combine. Cost side, the weakest link: the claim that one CTA streams a 2,048-token window (16-17 blocks x 128 rows x 128 B = 256 KB at page size 1) 'in a few microseconds' is not supported. In the same-run profiles the sliding FA3 main kernel grows as the tile count rises and FA3's dynamic pick has fewer splits per tile: about 8.2 us at decode_1 (2 tiles, many splits), 9.9 us at decode_16 (32 tiles), 13.3 us at decode_32 (64 tiles). That fits roughly 7 us fixed plus ~0.75 us per serial 128-row block. At 16-17 serial blocks per CTA that predicts a ~19-20 us main kernel. Against 9.9+3.2 = 13.1 us at decode_16 that is a ~+5-7 us loss, and against 13.3+3.3 = 16.6 us at decode_32 a ~+2-4 us loss. Independent evidence: the Triton fused sliding kernel took 17.4 us at S=4 and 28.8 us at S=2 (sliding-single-launch-fused-attention), so long single-CTA page-size-1 gathers are latency-bound, not bandwidth-bound. The grid falls to 32/64 CTAs (B*Hkv, one packed-GQA m-block) on 132 SMs, which is far from the 1.6-3.7 us floors, so this cannot be a bandwidth win. Expected table vs TASK: (1) 'sliding decode_1 flat' is moot. Master routes batch-1 decode to _sliding_fused_kernel, so that shape never reaches FA3 and is a byte-identical control. (2) 'sliding verify_1 flat' is the biggest risk. It runs 2 CTAs x ~17 serial blocks against a 7.7 us split main kernel plus 3.7 us combine, so expect perhaps +8-12 us on a ~17 us shape. At weight 46 the sliding shape dominates the verify_1 load (46x17 vs 16x30 us), so this alone could cost several percent of geomean. (3) decode_32/verify_32 are the only rows where a gain is plausible, if FA3's dynamic pick there is already ~1-2 splits. (4) Full rows flat is correct: the change is gated on window != 0. Net: likely below 1.293x overall, possibly with a small win at batch 32.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate: sliding decode_1 no longer calls FA3 (kernel_fn returns _sliding_forward for window && batch==1 && q_len==1), so the hypothesis's decode_1 expectation is void. That row is a pure drift control. Sliding verify_1 still goes through FA3 and becomes the only batch-1 sliding row affected.
- fused-preattention-triton: it precedes FA3 on the same stream and writes q/k/v, but it does not change FA3's split decision or the window traffic. It adds no interaction beyond a fixed ~1.7-4 us in front, which is identical in both arms.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup / paired-heads-launch-cleanup-ab-tune-batch-gate: they touch full layers only (paired = window == 0 and ...), and the sliding call stays unpaired with metadata None. No interaction. The full rows are controls.
- Not landed but relevant: fa3-scheduler-metadata-side-stream (running) touches the same sliding FA3 call's schedule. If it lands first, its metadata must be built with num_splits=1 (or this change must drop it), or the cap and the precomputed schedule disagree. Do not stack the two without rebuilding.

### Interference and A/B needs
No companion is warranted. The change branches only on config, and every scored shape is an independent kernel_fn call, so one run already gives a separate same-run answer for each of the five affected sliding shapes, with the full rows and sliding decode_1 as unchanged drift controls. Read each shape's FA3 main-kernel time and the absence of FlashAttnFwdCombine/prepare in top_kernels, not the headline. No landed hypothesis changes the sliding FA3 split path, so an ab-revert has nothing to isolate. A batch-gated variant (num_splits=1 only at batch>=32) can be decided from this run's per-shape rows without another run. Spending a second check on it beforehand would duplicate information. If the run shows wins only at batch 32, mint that gate as a stacked follow-up.

### Open questions
- Does the judge's sgl_kernel FA3 still launch prepare_varlen_num_blocks with num_splits=1 + cu_seqlens_q + local window? Expected yes (semaphore), which caps the saving at the 3.1-4.2 us combine. Confirm from top_kernels.
- Sliding verify_1 (2 CTAs, ~17 serial blocks at q_len 8) is the likely large loss the hypothesis marks 'flat'. If it regresses heavily, treat the run as information and never promote it whole.
- What split count does FA3 pick dynamically today at sliding decode_32/verify_32? If it is already 1-2, those rows should be ~flat-to-faster by ~3 us. That is the only place this could become a gated keeper.
- Attention has 2 checks left today. Given the likely net loss, churn may reasonably rank this below any candidate with a positive expected headline. That is a scheduling call, not a reason to hold.

