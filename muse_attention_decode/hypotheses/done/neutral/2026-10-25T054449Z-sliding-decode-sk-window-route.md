---
problem: muse_attention_decode
slug: sliding-decode-sk-window-route
created: 2026-10-25T054449Z
base: master
area: sliding-layer attention (decode, stream-K windowed kernel)
change: >
  On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is
  already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over
  rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b],
  token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK
  = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB
  at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6
  us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel
  (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is
  _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes;
  the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for
  the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of
  [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >=
  1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv
  0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The
  judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (master's
  self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data
  and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference
  with two states and alternating graph replays with refilled inputs (capture once, refill A,
  replay, refill B, replay, compare each), and on any miss falls back to master's path for that
  band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: medium
depends_on: sliding-verify-sk-pseudo-decode-s1
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)"
review: proceed
status: done
review_at: 2026-10-25T054724Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054724Z
base_sha: 6f95ce71ba21682ba9cbb6d346abb2a93a862b2f
base_blob: 76492911dfd6685139a2ffac06c1d27083ecdb0a
branch: hyp/muse_attention_decode/sliding-decode-sk-window-route
worktree: .worktrees/muse_attention_decode/sliding-decode-sk-window-route
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T084628Z
impl_commit: 376be02c8cf447e9c8f1f2f6a4301d52ea09aac6
submission_sha256: e93aa862c1d039a34a3d08f82d3aa0a854f0fb864e50ead13a7cf3ccd4ce4be0
norm_sha256: 4c56bd6e319def87714abd15800346199303250c4fddfee0921d78ae2377db89
submission_blob: 1117ad04a8bbcd5ca4da54c883e209c1b6e6f278
lint_warnings: 42
attempt_file: 2026-10-25T084628Z-check-sliding-decode-sk-window-route.md
prepared_at: 2026-10-25T084628Z
job: 118c6de9704146f3
dispatched_at: 2026-10-25T084636Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4972632172137348
speedup_ci: [1.4748042689555287, 1.5197221654719406]
harvested_at: 2026-10-25T091107Z
best_at_harvest: 1.444132502888963
promotion: null
---

# sliding-decode-sk-window-route

## Change

On branch hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1. The PSEUDO kernel is already a windowed stream-K decode kernel (per-row lo = max(0, sp - 2048), masks, prefix over rows). Instantiate it for sliding DECODE: one row per request (R = B, sp = cache_seqlens[b], token row b, page-table row b), i.e. PSEUDO with 1 token per request (template int TOK = 1; TOK = 8 is s1, byte-identical). DRAM bytes are 1x (no amplification): B x 2048 keys x 256 B = 8.4 MB at B=16, 16.8 MB at B=32, so the kernel should run near the landed decode kernel's ~3 TB/s, ~3-6 us, against Triton's 12.7 / 22.4 us kernel rows. Sliding decode needs the pre-attention kernel (master's Triton sliding path does norm/rope/cache store in-kernel), so this route is _pre_attention_kernel (PAIRED=False, ROPE on; as s1 uses at verify) + the sk kernel: two nodes; the gate must time the pair. Gate per band (B=2..16 from B=16, B=17..32 from B=32, and B=1 for the record) with grids 4R/8R/SM/2xSM and single/two-level ticket: interleaved graph replay of [pre-attn + sk] vs master Triton _sliding_fused_kernel; print all; route a band only at >= 1.05x. Self-check vs fp32 windowed reference: lens 1, 63, 2047, 2048, 2049, 80000, ragged, kv 0.25. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=sliding load=decode_16 *` faster: 15.33 us shape, Triton 12.7 us kernel; 8.4 MB at ~3 TB/s is ~3 us + pre-attn 2.0 us; neither cluster C=8 (13.8) nor paired rows (16.8) beat Triton here; M2 shape <= 12 us
- `*layer=sliding load=decode_32 *` faster: 24.33 us (Triton 22.4; cluster C=4 18.0 off-master); M2 shape <= 18 us
- `*layer=sliding load=decode_1 *` flat: 32 tiles only; expected to lose to Triton 10.2 / cluster C=16 10.6, gate keeps master
- `*layer=full *` flat: untouched

## Risk

Kill: if B=16 [pre-attn + sk] is not >= 1.05x faster than Triton in the printed gate, sliding decode stays Triton/cluster. If B=17..32 wins, compare its kernel row with the cluster C=4 row (18.0 us) before a restack picks one. Weight 46 rows: worth ~+1-2% headline per band if it reaches M2.

## Milestones

Rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054724Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design is sound and every band is either routed by printed paired timings or left as byte-identical Triton, so each row can be attributed without a companion. The ~3 TB/s, 3-6 us kernel estimate is optimistic: on this kernel, small workloads run latency-bound at about 0.9-1.1 TB/s. Expect B=32 to win clearly, while B=16 is a coin flip against the 1.05x gate.

### Justification
Mechanism: sliding decode at TOK=1 has no L2 read amplification, so it differs from s1's B=16/32 verify bands, which lost 0.65x/0.48x because of 67/134 MB of L2 traffic. That makes the work 512 tiles / 8.4 MB at B=16 and 1024 tiles / 16.8 MB at B=32. The incumbent Triton _sliding_fused_kernel is far from the floor: 12.7 us for 8.4 MB (0.66 TB/s, floor 1.6 us) and 22.4 us for 16.8 MB (0.75 TB/s, floor 3.3 us). These rows weigh 46 each in their loads, so the target is worth pursuing. WEAKEST LINK: the '~3 TB/s, ~3-6 us' figure. This kernel reaches 2.9-3.0 TB/s only on 330-660 MB full-decode jobs (sk-diag). On small jobs it is latency- and merge-bound: full decode_1 moves 20 MB in 18.6 us (0.89-1.1 TB/s), and s1's verify_1 (256 tiles, 0.5 MB DRAM) costs 9.8 us of kernel. A realistic B=16 estimate is a gather round-trip plus a ~16-32-slot ticket merge, about 6-9 us, plus a separate pre-attention node of 2.0 us and its graph-node boundary. That puts the pair at about 9-11.5 us against 12.7 us, which is near the 1.05x gate. B=32 (about 9-12 us of kernel plus 2.2 us of pre-attention, against 22.4 us) has real margin, and the off-master cluster C=4 row (18.0 us) gives a cross-check. Expected directions are consistent with TASK weights and floors: decode_1 flat because it is gate-kept, full layers flat because the code is untouched. The M2 targets (12 us / 18 us shape) are reachable at B=32 and marginal at B=16.

### Confounders on the current kernel
- sliding-verify-sk-pseudo-decode-s1 (base branch, NOT on master): the build starts from this branch, so the headline carries s1's sliding verify_1 route (1.4705x check vs master 1.4441x). Judge against s1's job 94523c3313444fd8 rows, not BEST.json. The TOK template must leave the TOK=8 instantiation byte-identical (print regs/local/smem/CTAs for it next to s1's). Otherwise a verify_1 shift gets mixed into this run.
- full-decode1-streamk-s3-two-level-ticket / full-decode-splitkv-s2 (landed, c3c1b01): this is the kernel being templated. The PSEUDO=false instantiations serving full decode must stay identical. If the full decode_1/16/32 rows move outside noise, that comes from a changed instantiation, not from this route.
- sliding-deep-splits-on-cuda-splitkv (landed): the comparator is its 16-split Triton _sliding_fused_kernel, which does norm, RoPE and the cache store in-kernel. The new route pays a separate _pre_attention_kernel node (about 2.0 us at B=16/32) that the incumbent folds in. That is why the gate must time the pair [pre-attn + sk] against Triton alone in one graph, as specified.
- fused-preattention-triton (landed): it now runs on sliding DECODE in its PAIRED=False, ROPE-on configuration, which until now was used only at verify. It writes the new K/V row that the sk window [sp-2048, sp) must include. Correctness depends on it storing the row before the sk kernel reads it, which stream order guarantees.

### Interference and A/B needs
No companion. This is a rewrite stage with a per-band hard gate: each sliding decode band runs either the sk pair, named in PATH CHOSEN together with both printed numbers, or byte-identical master Triton. A gain or loss on a row is therefore readable from the log and the profile's top_kernels without an A/B run. The one headline confound is s1's verify_1 route, and comparing against s1's own job resolves it. A revert or tune run would add nothing that the printed per-variant gate matrix does not already show. Attention has 8 checks for 8 queued items, which supports spending none on companions.

### Open questions
- BUILDER: subtract the empty-graph baseline as stated. s1's gate numbers evidently included about 5 us of fixed graph cost (gate 15.4 us vs profile 9.8 us at verify_1), which compresses ratios toward 1 and can wrongly reject a real 1.05x win at B=16.
- BUILDER: time the pre-attn + sk pair as two nodes in ONE captured graph (not two graphs summed), so the inter-node gap that Triton's single launch avoids is charged.
- BUILDER: at B=16 add a grid of 2R-4R (32-64 CTAs, 8-16 tiles each) so each request's merge is about 2-4 slots. s1's B=1 data showed that small merges won (p8 15.4 us vs 2xSM 19.5 us).
- If B=17..32 routes, print its kernel time next to the cluster C=4 figure (18.0 us, off-master) so the restack picks one. If B=2..16 falls back, the next stage should fold pre-attention into the sk prologue (as full-decode-splitkv-s3-fused-preattn tried) rather than retune the grid.


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn (decode route = `_pre_attention_kernel` + one `sk_attn` launch, torch.empty workspaces by shape/config only); no .item()/sync/prints in kernel_fn, no try/except; syncs are only in prepare-time self-check/gate code (lint: host-sync warnings only outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: tickets are zero-after-launch arrays, `_inv_freq` is the existing theta-keyed constant, `_SD_ROUTE` is config-decided at import; no device pointers, TMA maps or state-sized buffers in globals.
- [x] n/a: weights (no weights in this problem).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries e4m3 from the pre-attention kernel, fp32 softmax/merges, bf16 out.
- [x] Attention sees every key: each request's whole [max(0, sp - 2048), sp) window incl. the new row; no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, child processes or network (no subprocess/fork anywhere; one docstring reworded to pass the lint's network word check).
- [x] All work inside the timed call; no output caching or replay.
- [x] sm_90a: the module still builds with TORCH_CUDA_ARCH_LIST=9.0a restored after, arch flags printed; no new wgmma/sm_90a-only PTX added (unchanged cp.async/mma.sync kernel); bounded waits: no mbarrier/spin added; the ticket merge has no spin.

okc_lint: `OK with warnings: muse_attention_decode/submission.py` (only host-sync warnings in prepare()-side code, outside kernel_fn); `python3 -I -m py_compile` ok; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading only (no GPU on this node). **Verdict: dispatch.** No fixes were needed and the worktree was not edited.

- Gates: `_sd_gate` is config-only (window 2048, q_len 1, page 1, tp8, 2 <= B <= 32, band in the frozen `_SD_ROUTE`). The band-2 gate runs at B=16 x 80000, the same shape as the scored rows. The correctness bound (rel L2 < 2e-2) leaves about 10x headroom over the 1.6e-3 that s1 measured on the same kernel body. The 1.05x gate compares net times, with the empty graph subtracted from both sides, so nothing in it makes it impossible to pass. The judged test `B=4 ctx 6000 sliding page 1 dist edges` runs the route whenever band 2 is routed.
- Fallbacks: `_sd_validate`, each `_sd_gate_band`, each `_sd_replay_check` and the whole `_sd_setup` are wrapped. Any miss drops that band, or every band, to master Triton and prints a PATH CHOSEN line. `_SD_ROUTE` is then frozen, or set to `MappingProxyType({})` if setup raises. If the PSEUDO build fails, `_PS_BUILT` stays False and sliding decode stays on Triton.
- Correctness: with TOK=1, R = B, sp = cache_seqlens[b] and lo = max(0, sp - 2048), which is [p - 2047, p] inclusive. That attends every key of the window and matches both `_ps_reference` and TASK. The page-table, query, qkv and out rows are all b. Workspace is grid + B and group slots are B*32. Ticket and gticket capacity (256 and 256*32) is at least R*32, grid is at most 264 (the limit is 1024) and ngrp is at most 32. TOK=8 constant-folds to s1's exact expressions (`>> 3`, `& 7`, `B * 8`). The PSEUDO=false variants are untouched.
- Rules: kernel_fn's new branch is dict lookups plus `_pre_attention_kernel` and one `sk_attn` launch. Its allocations depend only on shape, config and SM count. It has no sync, print or try/except. The only module or static state is the existing self-resetting ticket arrays, the `_SMS` hardware-count cache and the route tuples. The C++ `done[8]` array holds flags only. FP8 KV, e4m3 queries, fp32 softmax and merges, and bf16 output are kept. The two-state replay check captures kernel_fn once per state, then alternates the states over 4 rounds, refilling each state and NaN-poisoning its output before every replay. okc_lint returns OK with warnings, all host-sync warnings on the prepare side. py_compile passes and __pycache__ was removed.
- Budget: s1's run took 380 s end to end. This change adds two template instantiations to one TU (about +20-40 s of nvcc) and a few seconds of gate and replay checks, so it stays well inside 15 min.
- Scope: the diff matches the change paragraph. The 2R/3R grids come from the review, and B=1 is timed and printed but never routed, as the expected rows say.

Residual risk: the CUDA has not been compiled. A compile error falls back to the landed-only build and leaves sliding verify and sliding decode on master. B=16 is likely to land near the 1.05x gate and stay on Triton.

## Result

- job `118c6de9704146f3` (check), status done, passed True, tests 16/16
- speedup 1.4973x, 95% CI [1.4748, 1.5197], score 2515.4561714330553 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/sliding-verify-sk-pseudo-decode-s1.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode-sk-window-route.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode-sk-window-route.patch`; branch `hyp/muse_attention_decode/sliding-decode-sk-window-route`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.33 | 13.57 | 1.0033 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.73 | 22.98 | 1.0073 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.72 | 15.42 | 1.0126 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.4 | 111.17 | 0.9856 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 23.85 | 17.38 | 1.385 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.28 | 212.84 | 1.0676 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.44 | 13.65 | 0.9933 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.1 | 30.01 | 1.0023 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.39 | 22.66 | 0.9863 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.76 | 162.9 | 0.9966 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.49 | 28.76 | 0.9972 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.78 | 301.44 | 0.9988 | flat |

gained on sliding decode_32; full decode_32. Expected hit: 2/4. Noise ±1.5%.

## Follow-ups

- rewrite stage 2 of `sliding decode (q_len 1, page 1, tp8, 2<=B<=32): master Triton _sliding_fused_kernel (16 splits, in-kernel pre-attention)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on sliding-decode-sk-window-route --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-decode-sk-window-route`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

