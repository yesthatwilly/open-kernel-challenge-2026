# muse_attention_decode · 2026-10-25T203814Z · submit · full-decode1-sk-two-level-deep-ring

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T153027Z-full-decode1-sk-two-level-deep-ring.md` (branch `hyp/muse_attention_decode/full-decode1-sk-two-level-deep-ring`).

- **Job:** `26c049edfdd842b3` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `8945b6ab`

## Goal
Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages, two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4 in a 22.4 us shape, 0.9 TB/s on 20 MB (s5 log [sk3-diag]: 22.9 us eager two-level grid SM). At grid SM there is ONE CTA per SM holding ~9.5 tiles of 64 tokens, and the 4-stage ring keeps only 3 tiles (48 KB) in flight, so the CTA needs ~3 dependent DRAM round trips for its range. Grid 2xSM (more in flight) was slower (24.4 us) because it doubles the partials the two-level merge reads, so ring depth at grid SM is the untested lever that adds bytes in flight without adding partials. CHANGE: instantiate the existing kernel template at STAGES = 8 (and 12 if it fits) with TWO = true (the Lay<STAGES> layout grows by 16 KB per stage: ~160 KB at 8, ~225 KB at 12; 1 CTA/SM at grid SM is already the occupancy, so nothing is lost), add the variants to the B=1 [sk3-diag]-style prepare timing (interleaved cold-L2 graph replay of pre-attn + kernel, empty graph subtracted, median of 31, lens 80000), and route B=1 to the fastest only at >= 1.03x vs master's <4, two-level> at grid SM, with rel L2 < 1e-2 vs master; else master. Self-check the routed variant on master's B=1 two-level cases (80000 / 4097 / 63 / 1 keys, kv 1 and 0.25, grids 3/7/SM) with tickets summing to 0, plus the two-state alternating replay. Print regs/local/smem per variant. Batch > 1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## What changed, and where
Edited only the worktree's muse_attention_decode/submission.py (+157/-11).
- CUDA: new variants 8 = splitkv_attn_sk_kernel<8, true, true> and 9 = <10, true, true> (non-PSEUDO, two-level), present in both the SK_PSEUDO and landed builds. SK_MAX_VARIANT = 9 in both; sk_smem(8) = Lay<8>, sk_smem(9) = Lay<10> with static_asserts <= 232,448 B; sk_prepare done[] grown to 16; sk_attn/sk_attrs: `pseudo` is now variants 4-7 only, two-level checks include >= 8, and the landed (non-pseudo) build still rejects 4-7. PSEUDO variants 4-7, Lay<4,PS_MAXR,true> and their sk_smem/sk_prepare behaviour are unchanged.
- Python: `_B1_VARIANT` (default 3) used by `_full_decode_sk` at B=1; `_sk_launch` allocates group slots for 3/8/9; `_SK_LABELS` extended; `_deep_ring_setup()` runs in prepare after `_sk_validate_b1` passes: per-variant attrs print, 10 B=1 self-check cases and ticket sum, eager rel L2 vs master variant 3, interleaved cold-L2 graph-replay timing of [pre-attn + kernel] (empty graph subtracted, median of 31, lens 80000), routes the fastest only at >= 1.03x and rel L2 < 1e-2, then runs a two-state alternating replay (`_alt_check`); any miss/exception resets `_B1_VARIANT = 3`. Prints per-variant us and PATH CHOSEN.
- Deviation (review wins): 12 stages dropped (Lay<12> = 234,528 B > limit); 8 and 10 stages used. Variants numbered 8/9, not remapped.
- Not run (no GPU): compile and smem sizes unverified beyond static_assert.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn change is a module-global int read selecting a launch; no sync/.item()/prints added to kernel_fn; all syncs are in prepare-time `_deep_ring_setup`.
- [x] Nothing derived from `data` is cached across calls: `_B1_VARIANT` is chosen from synthetic lens in prepare only; tickets are self-resetting int32 arrays.
- [x] n/a: no weights touched.
- [x] KV cache FP8, q FP8 E4M3, fp32 softmax/merges unchanged (same kernel template).
- [x] Attention sees every key: same kernel, only ring depth differs; self-checks include 80000/4097/63/1 keys.
- [x] No timing patches, harness access, subprocess or child processes added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: OK with warnings (only host-sync warnings outside kernel_fn's call graph, prepare-time). py_compile clean, __pycache__ removed.

## Risks / what could fail
medium
5 stages gave nothing at B=16 (bandwidth-bound there); B=1 is latency-bound, which is why depth may pay only here. Full decode_1 is 40% of the decode_1 load.

## Result (after the run)
- job `26c049edfdd842b3` (submit), status done, passed True, tests 16/16
- speedup 1.5109x, 95% CI [1.4882, 1.5336], score 2492.7698454755905 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5295495462227864, "ci": [1.5066063030294445, 1.552492789416128], "job": "8713665f3064462c"}; base result: muse_attention_decode/hypotheses/done/neutral/combo-b1-chain-c320-fused-pre-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-deep-ring.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-deep-ring.patch`; branch `hyp/muse_attention_decode/full-decode1-sk-two-level-deep-ring`
