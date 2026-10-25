---
problem: muse_attention_decode
slug: full-decode1-sk-two-level-deep-ring
created: 2026-10-25T153027Z
base: master
area: full-layer attention (batch-1 decode, stream-K ring depth)
change: >
  Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages,
  two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4 in a 22.4 us shape, 0.9 TB/s
  on 20 MB (s5 log [sk3-diag]: 22.9 us eager two-level grid SM). At grid SM there is ONE CTA per
  SM holding ~9.5 tiles of 64 tokens, and the 4-stage ring keeps only 3 tiles (48 KB) in flight,
  so the CTA needs ~3 dependent DRAM round trips for its range. Grid 2xSM (more in flight) was
  slower (24.4 us) because it doubles the partials the two-level merge reads, so ring depth at
  grid SM is the untested lever that adds bytes in flight without adding partials. CHANGE:
  instantiate the existing kernel template at STAGES = 8 (and 12 if it fits) with TWO = true (the
  Lay<STAGES> layout grows by 16 KB per stage: ~160 KB at 8, ~225 KB at 12; 1 CTA/SM at grid SM is
  already the occupancy, so nothing is lost), add the variants to the B=1 [sk3-diag]-style prepare
  timing (interleaved cold-L2 graph replay of pre-attn + kernel, empty graph subtracted, median of
  31, lens 80000), and route B=1 to the fastest only at >= 1.03x vs master's <4, two-level> at
  grid SM, with rel L2 < 1e-2 vs master; else master. Self-check the routed variant on master's
  B=1 two-level cases (80000 / 4097 / 63 / 1 keys, kv 1 and 0.25, grids 3/7/SM) with tickets
  summing to 0, plus the two-state alternating replay. Print regs/local/smem per variant. Batch >
  1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine).
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: kernel 19.0 us at 0.9 TB/s vs floor 4.3 us; 7 tiles in flight per CTA instead of 3 should remove 1-2 of ~3 round trips (-2 to -4 us)
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: B>1 untouched
  - shapes: "*layer=full load=decode_32 *"
    direction: flat
    why: B>1 untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T203307Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T203307Z
base_sha: 6d8fce0bdeed57b76fcd060aaec81554c9798b15
base_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
branch: hyp/muse_attention_decode/full-decode1-sk-two-level-deep-ring
worktree: .worktrees/muse_attention_decode/full-decode1-sk-two-level-deep-ring
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T203814Z
impl_commit: 8945b6ab3c60b478f652f6bbb24ac98d3abc5eae
submission_sha256: b769d39e3819acc9f5bf7b89c4011b04922a2d57f97460aa24293485263dccc8
norm_sha256: b4054cd3ff8db0745897065c1643d86de8b908ee5371aba2f5e34ff527c79aec
submission_blob: 246f5494a75eb62a330900ae938f598bee8341b8
lint_warnings: 64
attempt_file: 2026-10-25T203814Z-submit-full-decode1-sk-two-level-deep-ring.md
prepared_at: 2026-10-25T203814Z
job: 26c049edfdd842b3
dispatched_at: 2026-10-25T203823Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.5108895860706448
speedup_ci: [1.488226242279585, 1.5335529298617043]
harvested_at: 2026-10-25T204820Z
best_at_harvest: 1.5295495462227864
promotion: null
---

# full-decode1-sk-two-level-deep-ring

## Change

Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages, two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4 in a 22.4 us shape, 0.9 TB/s on 20 MB (s5 log [sk3-diag]: 22.9 us eager two-level grid SM). At grid SM there is ONE CTA per SM holding ~9.5 tiles of 64 tokens, and the 4-stage ring keeps only 3 tiles (48 KB) in flight, so the CTA needs ~3 dependent DRAM round trips for its range. Grid 2xSM (more in flight) was slower (24.4 us) because it doubles the partials the two-level merge reads, so ring depth at grid SM is the untested lever that adds bytes in flight without adding partials. CHANGE: instantiate the existing kernel template at STAGES = 8 (and 12 if it fits) with TWO = true (the Lay<STAGES> layout grows by 16 KB per stage: ~160 KB at 8, ~225 KB at 12; 1 CTA/SM at grid SM is already the occupancy, so nothing is lost), add the variants to the B=1 [sk3-diag]-style prepare timing (interleaved cold-L2 graph replay of pre-attn + kernel, empty graph subtracted, median of 31, lens 80000), and route B=1 to the fastest only at >= 1.03x vs master's <4, two-level> at grid SM, with rel L2 < 1e-2 vs master; else master. Self-check the routed variant on master's B=1 two-level cases (80000 / 4097 / 63 / 1 keys, kv 1 and 0.25, grids 3/7/SM) with tickets summing to 0, plus the two-state alternating replay. Print regs/local/smem per variant. Batch > 1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=full load=decode_1 *` faster: kernel 19.0 us at 0.9 TB/s vs floor 4.3 us; 7 tiles in flight per CTA instead of 3 should remove 1-2 of ~3 round trips (-2 to -4 us)
- `*layer=full load=decode_16 *` flat: B>1 untouched
- `*layer=full load=decode_32 *` flat: B>1 untouched
- `*layer=sliding *` flat: untouched
- `*load=verify_* *` flat: untouched

## Risk

5 stages gave nothing at B=16 (bandwidth-bound there); B=1 is latency-bound, which is why depth may pay only here. Full decode_1 is 40% of the decode_1 load.

## Adversarial review

Reviewed 2026-10-25T203307Z by an adversarial subagent (model opus).

**Verdict: proceed.** Deepening the ring of the existing <STAGES, two-level> stream-K kernel at grid = SM count for full decode_1 is untried, rules-clean and self-measuring via in-prepare variant timing. Likely gain is small (~+0.5-1% headline, inside CI), and B=1 stays slower than the unlanded c320 path (17.4 us). Read it on the per-variant kernel times. 12 stages does not fit shared memory; use 8 and 10 (or 11).

### Justification
At grid 132, ~9.5 tiles of 16 KB per CTA; 4 stages = 3 in flight (~1 meta round trip + ~3 dependent waves); 8 stages ~1 + 1.4: 1-2 fewer round trips (1-2 us each). Weakest link: most of the 19.0 us is prologue, per-tile compute and the two-level merge/finalize tail, which depth does not touch; steady-state bytes in flight (6.3 MB) already exceed Little's law. Smem: Lay<8> 164,896 B fits; Lay<12> 234,528 B > 232,448 B limit; Lay<10> 199,712 B, Lay<11> 217,120 B. Full decode_1 weight 16 at 22.4 us; a 3 us cut ~+0.9% headline. Prior (missing field): full-decode1-sk4-cluster-dsmem-merge, -sk-two-level-fused-preattn, -sk-pdl-from-preattn, -cuda-splitkv-chunk320/640; none changed ring depth.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket: defines the kernel and merge tail; flat means the tail dominates; [sk3-diag] per-variant times separate it.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn / restack-sliding-decode-sk-window-on-cluster-master: PSEUDO instantiations (variants 4-7) of the same template; do not renumber them or touch Lay<4,PS_MAXR,true> / sk_smem / sk_prepare for v>=4; extra instantiations add compile time.
- Unlanded c320 path (17.4 us full decode_1): confounds the value, not the measurement.

### Interference and A/B needs
No companion: prepare times <4,two-level> vs 8/10 stages under the same cold-L2 graph replay and routes at >= 1.03x; the log plus the profile's full decode_1 top kernel read it.

### Open questions
- Is attention's only scored run best spent here (best case ~1.52-1.53 vs board 1.5347)?
- Was STAGES=5 or deeper timed at B=1 grid SM in any [sk3-diag] log?
- Phase split of the 19.0 us (merge/tail vs streaming) via clock64 in prepare.


## Implementation notes

Edited only the worktree's muse_attention_decode/submission.py (+157/-11).
- CUDA: new variants 8 = splitkv_attn_sk_kernel<8, true, true> and 9 = <10, true, true> (non-PSEUDO, two-level), present in both the SK_PSEUDO and landed builds. SK_MAX_VARIANT = 9 in both; sk_smem(8) = Lay<8>, sk_smem(9) = Lay<10> with static_asserts <= 232,448 B; sk_prepare done[] grown to 16; sk_attn/sk_attrs: `pseudo` is now variants 4-7 only, two-level checks include >= 8, and the landed (non-pseudo) build still rejects 4-7. PSEUDO variants 4-7, Lay<4,PS_MAXR,true> and their sk_smem/sk_prepare behaviour are unchanged.
- Python: `_B1_VARIANT` (default 3) used by `_full_decode_sk` at B=1; `_sk_launch` allocates group slots for 3/8/9; `_SK_LABELS` extended; `_deep_ring_setup()` runs in prepare after `_sk_validate_b1` passes: per-variant attrs print, 10 B=1 self-check cases and ticket sum, eager rel L2 vs master variant 3, interleaved cold-L2 graph-replay timing of [pre-attn + kernel] (empty graph subtracted, median of 31, lens 80000), routes the fastest only at >= 1.03x and rel L2 < 1e-2, then runs a two-state alternating replay (`_alt_check`); any miss/exception resets `_B1_VARIANT = 3`. Prints per-variant us and PATH CHOSEN.
- Deviation (review wins): 12 stages dropped (Lay<12> = 234,528 B > limit); 8 and 10 stages used. Variants numbered 8/9, not remapped.
- Not run (no GPU): compile and smem sizes unverified beyond static_assert.

## Rules check

- [x] CUDA-graph capturable: kernel_fn change is a module-global int read selecting a launch; no sync/.item()/prints added to kernel_fn; all syncs are in prepare-time `_deep_ring_setup`.
- [x] Nothing derived from `data` is cached across calls: `_B1_VARIANT` is chosen from synthetic lens in prepare only; tickets are self-resetting int32 arrays.
- [x] n/a: no weights touched.
- [x] KV cache FP8, q FP8 E4M3, fp32 softmax/merges unchanged (same kernel template).
- [x] Attention sees every key: same kernel, only ring depth differs; self-checks include 80000/4097/63/1 keys.
- [x] No timing patches, harness access, subprocess or child processes added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: OK with warnings (only host-sync warnings outside kernel_fn's call graph, prepare-time). py_compile clean, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only emulation; no GPU). **Verdict: dispatch-with-fixes.**

Fixes (worktree submission.py only):
1. **Gate that always failed (two-state replay).** `_deep_ring_setup` built the `_alt_check` states with `_splitkv_synthetic(l, 1.0, gen, dev)` without `max_pages`, so each length set got its own page-table width (80038 for [80000], 1988 for the refill [1950]); the in-place `copy_` refill at step 2 raises on the shape mismatch, so the routed variant was always reset to 3 and the deep ring could never run. Now `mp = max(max(s) for s in sets) + 1 + 37` is passed as `max_pages`, as master's `_skf_setup` replay does.
2. **Fallback cascade on a deep-ring compile failure.** Variants 8/9 were instantiated in both the PSEUDO and the landed builds, so a compile error in `<8|10, true, true>` would fail both and drop `_EXT2` (stage 2/3 lost at every full-decode batch). The deep variants are now guarded by `#ifndef SK_NO_DEEP` (SK_MAX_VARIANT 9 / 7 / 3); `_build_sk` tries PSEUDO+deep (`muse_decode_sk_v4pd`), then PSEUDO without deep (`muse_decode_sk_v4p`, master's source behaviour), then landed without deep. `_DEEP_BUILT` records it, and `_deep_ring_setup` prints PATH CHOSEN master when it is false.

Checked and fine: Lay<8> 164,896 B, Lay<10> 199,712 B <= 232,448 B; `cudaFuncSetAttribute` runs in `sk_prepare` before every launch/attr query; the ring math (`nn % STAGES`, `cp_wait<STAGES-2>`, meta double buffer, prologue meta refills) is valid for STAGES 8 and 10 (meta block k+1 lands by tile 16k+16-STAGES+1 >= 16k); no co-residency needed (last-arriver tickets, no spin); tickets reset by the last arriver at both levels and re-zeroed in `finally`; `_B1_VARIANT` is set only after build/attrs/10 self-checks/ticket sum 0/rel L2/1.03x gate and reset to 3 on any replay miss or exception; kernel_fn reads only the module int (graph-capturable, no sync/alloc by values). FUSED_PRE routes only B >= 2, so the B=1 path is the one selected. okc_lint OK (prepare-time host-sync warnings only), py_compile clean.

Residual risk: compile of the new instantiations unverified (now isolated), +~20 s compile and ~20-30 s setup against ~470 s of a 900 s budget; likely gain small (within CI).

## Result

- job `26c049edfdd842b3` (submit), status done, passed True, tests 16/16
- speedup 1.5109x, 95% CI [1.4882, 1.5336], score 2492.7698454755905 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-deep-ring.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-deep-ring.patch`; branch `hyp/muse_attention_decode/full-decode1-sk-two-level-deep-ring`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 12.1 | 11.25 | 1.076 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 17.39 | 22.46 | 0.771 | lost |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.46 | 15.68 | 0.9892 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.77 | 112.35 | 0.9896 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 17.46 | 17.27 | 0.997 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.89 | 227.82 | 0.9991 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.23 | 13.32 | 0.9855 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.27 | 30.51 | 1.0008 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.62 | 22.64 | 0.9933 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.46 | 163.22 | 0.9967 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.01 | 28.66 | 1.0025 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 301.99 | 302.1 | 1.0018 | flat |

gained on sliding decode_1; lost on full decode_1. Expected hit: 3/5. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-sk-two-level-deep-ring`) that adds a further change in the same area is the only way to learn more.

