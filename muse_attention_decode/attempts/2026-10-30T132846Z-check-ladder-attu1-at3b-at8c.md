# muse_attention_decode · 2026-10-30T132846Z · check · ladder-attu1-at3b-at8c


- **Job:** `1cd27f58344c4b9b` (`okc status <job>`)
- **File:** muse_attention_decode/submission.py at `ladder/at3b` 0c170baa (= master 7f587e02, board 1.803, + AT3 7e324ff2
  cherry-picked from 67c239fd + AT3b), 643,338 B.

## Goal
Ladder rungs AT3 + AT3b (campaign 2026-10-30-public-ladder, user-requested). Target is sliding verify_1 (v1s): ours
13.8 us on the judge (Triton pre-attn 1.6 + stream-K PSEUDO 9.8 in the profile); public 6.8.
- AT3 alone, with one CTA finalizing 128 rows, missed: +0.6/+0.9 us on the H100 ([refuted]).
- AT3b splits that finalize over the last 8 CTAs to arrive. Pre-registered -0.4 us; measured -0.32 / -0.34 us
  (fresh seeds, H100). In-import race x1.054.
- **Expected on the judge:** v1s about -0.2..-0.5 us, headline about +0.1-0.3%, well inside one run's margin. Read
  this check per row and by the PATH line.

## What changed, and where
- `sliding_at3_kernel` added to the AT2 module (renamed `muse_sliding_at2_v3`, function `slide_at3_attn`;
  `slide_at2_setup` returns 12 attrs: [8..11] = AT3, 101 regs, 0 B local, 2 CTAs/SM). AT2/AT8 code is unchanged.
- The kernel runs after master's Triton pre-attention (unchanged: e4m3 q, the 8 new cache rows):
  - 33 CTAs x 64 keys over the key union [S-2047, S+8).
  - Each warp handles one kv head x token pair as a full m16 tile (FP8 QK, f16 PV), with each row masked to its own
    window.
  - Merge: fixed-reference R=16 FP32 red.global into a [128][64] + l workspace, with the R11 guard to exact slots.
  - Finalize: fence plus one atom ticket. The last 8 arrivals each finalize one token's 16 rows after a bounded
    acquire spin to all-arrived (error bit on timeout). They self-reset, and a done counter resets the ticket.
- `_slide_at3_init` (in `_slide_init`, after AT8; no build of its own; deadline 60 s under the 480 s rule):
  - Self-check at 17 lengths with the at3, forced-guard and mixed-guard arms, against FP32 and the incumbent.
  - Two-state replay through kernel_fn.
  - Cold-L2 race of [pre-attn + PSEUDO p8] vs [pre-attn + AT3], routed at >= 1.03x.
- `kernel_fn`: after the pre-attention launch, `_SLIDE_AT3 and _slide_at3_gate(cfg)` (sliding, q_len 8, B=1, window
  2048, page 1, tp8) sends the step to AT3. Otherwise PSEUDO/FA3 runs, unchanged.

## Rules check (against rules/technical-rules.llms.txt and the diff)
- [x] **CUDA graphs:** one fixed-grid launch (33 CTAs from config); no host sync and no value-dependent
      allocation/branching in kernel_fn. Workspace allocated once at import.
- [x] **Fresh inputs / no caching:** workspace, slots and tickets return to 0 / -inf / 0 in every launch (self-reset,
      checked after every self-check launch, after the replay and after the race). Nothing derived from data persists.
- [x] **Precision:**
  - The KV cache stays FP8 E4M3.
  - Queries are the pre-attention's E4M3 (as FA3), and P is f16 for PV (finer than E4M3).
  - The softmax denominator is FP32 from unrounded P, and the merge is FP32.
  - The output is BF16.
  - The epilogue uses hardware rcp/rsqrt/exp (~2 ulp fp32).
- [x] **Every key:** each token attends every key of its own window exactly once (the CTAs cover [S-2047, S+8)
      disjointly). No skipping. The R11 guard reroutes rows to exact slots and never drops mass.
- [x] No timing patches, harness access, reference import, subprocesses or network.
- [x] All work happens inside the timed call.
- [x] Size 643,338 B < 1,000,000. Import 300 s on the H100 node, cold caches (the AT3 phase adds about 1 s).

## Risks
- **Spin:** the finalizers spin on co-residency of 33 CTAs (one wave; same pattern as master's AT5b). The spin is
  bounded and sets an error bit, but a timeout would give a wrong output. It never happened on the node.
- **Race margin:** x1.054 against the 1.03 gate on the H100. If the H200 race lands below 1.03, PSEUDO stays, at no loss.
- **Small gain:** the headline will likely sit inside noise. The v1s judge row itself spreads 12.7-13.9 us between runs
  (CLAIMS :65).

## Expected PATH lines
- `[slide-gate] AT3-vs-incumbent verify B=1 x 80000: ... (ROUTE AT3)`
- `[slide] PATH CHOSEN: sliding verify B=1 AT3 ON: pre-attn + split-KV 33 x 64 keys, fixed-reference atomic merge, finalize over the last 8 CTAs (AT3b)`

## Union note (main thread)
File = AT3b 0c170baa + AT8c 810f9651 cherry-picked (502bf2bb): two shelved riders on different rows (v1s, d16s), both bitwise/judge-metric, both raced at import with fallback. AT8c rules: as in its own attempt (gate logits preloaded in registers; no precision change). Node-validated separately, not together: the check is the union validation.

## Result
- job `1cd27f58344c4b9b` (check), passed, **1.829x [1.802, 1.862]** vs board 1.803 (+1.4%): AT3b routed (PATH CHOSEN sliding verify B=1 AT3 ON ... AT3b), v1s 13.4 us; d16s 12.1 us.
