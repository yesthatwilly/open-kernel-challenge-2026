# muse_attention_decode · 2026-10-25T084628Z · check · sliding-decode-sk-window-route

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T054449Z-sliding-decode-sk-window-route.md` (branch `hyp/muse_attention_decode/sliding-decode-sk-window-route`).

- **Job:** `118c6de9704146f3` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `376be02c`

## Goal
On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b], token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6 us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes; the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >= 1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv 0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Base: worktree base 6f95ce7 (s1 impl), master is ba17a47; no stale_base flag in the md, merge of master not done (master's P/submission.py was not touched by this build). Edits only in the worktree's `muse_attention_decode/submission.py` (+442/-18). Nothing could be run on this node (no GPU/torch): compile and gate numbers are unverified.

Kernel (SK_PSEUDO source): `splitkv_attn_sk_kernel` gets a trailing template parameter `int TOK = 8`; `load_prefix_pseudo<TOK>`. TOK = 8 keeps the shift/mask spellings (`>> 3`, `& 7`, `B * 8` constants), so variants 4/5 are the same code as s1; the prepare prints their regs / local / smem / CTAs ("must equal s1's line"). New variants 6 = `<4, ticket, single, PSEUDO, TOK=1>` and 7 = `<4, ticket, two-level, PSEUDO, TOK=1>`: R = B rows, sp = cache_seqlens[b], lo = max(0, sp - 2048), page-table row b. Host side: tok = 1 for variant >= 6, ticket/group checks cover variant 7, `done[8]`, max variant 7. The PSEUDO=false instantiations are untouched.

Python: `_ps_synthetic(..., L=8)` parameter (decode uses L=1). New block "Sliding DECODE": `_sd_gate` (config-only), `_sd_grid` (2R/3R/4R/8R/SM/2xSM, capped at 2xSM), `_sd_sk_launch`, `_sd_run` = `_ps_prepare_q` (PAIRED=False, ROPE on) + sk launch, `_sliding_decode_sk`, `_sd_validate`, `_sd_gate_band`, `_sd_replay_check`, `_sd_setup`. kernel_fn: sliding decode takes `_sliding_decode_sk` when `_sd_gate(cfg)`, else `_sliding_forward` (Triton) byte-identical. Tickets are the existing `_ps_ticket` / `_ps_gticket` self-resetting zero arrays (R <= 32 fits); no other module-level device state.

Review items:
- Empty-graph baseline: `_sd_interleaved` captures an empty graph (fallback: one-element kernel, printed), replays it round-robin with the others and every number is printed raw and net of it; the ratio uses net numbers (floor 0.5 us).
- Pair as two nodes in ONE graph: each candidate is `_sd_run` (pre-attn + sk) captured as a single graph, timed interleaved (start rotated per rep, cold L2 via a 256 MB read, median of 21) against `_sliding_forward` (Triton) in its own graph.
- 2R-4R at B=16: specs p2, p3, p4, p8, sm, 2sm x single/two-level (B=16: grids 32, 48, 64, 128, 132, 264); same at B=1 and B=32.
- B=17..32: the best variant's sk kernel alone, pre-attention alone, pair and Triton are re-timed in a second interleaved pass and the sk row is printed next to the cluster C=4 figure (18.0 us).
- TOK=8 byte-identical: see Kernel above; attrs printed for 4/5 and 6/7.
- Route is an import-time constant: `_SD_ROUTE` is filled once by `_sd_setup` at import and replaced by a `MappingProxyType` before kernel_fn runs; kernel_fn only reads it.
- Tickets back to 0: asserted after the self-check and after the replay checks (sum of both arrays must be 0, else every band drops to Triton).
- lens-1 case kept: the canary is B=1 lens 1 (cache_seqlens 2), plus lens 0, 63, 2047, 2048, 2049, 80000, ragged B=16/32 (with 1, 63, 2047, 2048, 2049, 0 entries), kv 0.25.
- Two-state replay check (`_sd_replay_check`): kernel_fn captured on state A and state B, 4 rounds alternating A, B with in-place refill of each state (other seq lens, fresh K/V/routing) and NaN-poisoned output before each replay; reference from each state's own cache after the replay.
- Deviation: B=1 is timed and printed for the record but never routed (expected row says flat); the 1.05x gate applies to bands 2..16 (gated at B=16) and 17..32 (gated at B=32). Bound for correctness is rel L2 < 2e-2 as in s1's checks (vs fp32 windowed reference).
- Expected timing caveat from the review: B=16 is near the gate; the review's fold-pre-attention-into-prologue next step applies if B=2..16 falls back.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn (decode route = `_pre_attention_kernel` + one `sk_attn` launch, torch.empty workspaces by shape/config only); no .item()/sync/prints in kernel_fn, no try/except; syncs are only in prepare-time self-check/gate code (lint: host-sync warnings only outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: tickets are zero-after-launch arrays, `_inv_freq` is the existing theta-keyed constant, `_SD_ROUTE` is config-decided at import; no device pointers, TMA maps or state-sized buffers in globals.
- [x] n/a: weights (no weights in this problem).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries e4m3 from the pre-attention kernel, fp32 softmax/merges, bf16 out.
- [x] Attention sees every key: each request's whole [max(0, sp - 2048), sp) window incl. the new row; no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, child processes or network (no subprocess/fork anywhere; one docstring reworded to pass the lint's network word check).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a: the module still builds with TORCH_CUDA_ARCH_LIST=9.0a restored after, arch flags printed; no new wgmma/sm_90a-only PTX added (unchanged cp.async/mma.sync kernel); bounded waits: no mbarrier/spin added; the ticket merge has no spin.

okc_lint: `OK with warnings: muse_attention_decode/submission.py` (only host-sync warnings in prepare()-side code, outside kernel_fn); `python3 -I -m py_compile` ok; __pycache__ removed.

## Risks / what could fail
medium
Kill: if B=16 [pre-attn + sk] is not >= 1.05x faster than Triton in the printed gate, sliding decode stays Triton/cluster. If B=17..32 wins, compare its kernel row with the cluster C=4 row (18.0 us) before a restack picks one. Weight 46 rows: worth ~+1-2% headline per band if it reaches M2.

## Result (after the run)
- job `118c6de9704146f3` (check), status done, passed True, tests 16/16
- speedup 1.4973x, 95% CI [1.4748, 1.5197], score 2515.4561714330553 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/sliding-verify-sk-pseudo-decode-s1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode-sk-window-route.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode-sk-window-route.patch`; branch `hyp/muse_attention_decode/sliding-decode-sk-window-route`
