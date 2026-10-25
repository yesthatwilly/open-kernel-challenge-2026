---
problem: muse_prefill_layer
slug: fa3-full-contiguous-kv-triton-gather-gated
created: 2026-10-25T152235Z
base: master
area: "FA3 call on full-attention layers (KV source: contiguous TMA copy of the paged page-size-1 rows, per-class timing gate)"
change: >
  Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention
  layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 changes. Sliding layers, the
  page-64 and kv-scaled cases, qkv GEMM, FusedQKNorm, the q/k/v FP8 casts, store_cache,
  o_norm_gate and everything after stay master's. MECHANISM: at page_size 1 FA3 runs its paged
  non-TMA path: producer warps gather KV rows with cp.async, PackGQA is forced and the key tile is
  smaller. fa3-contiguous-kv-gather (job 75bbf278dd9e430e, regression 0.988x) measured FA3's own
  kernel on a contiguous cache at 857 / 4300 / 7747 us on first_turn full ctx 0 / 32768 / 65536,
  against 907 / 4545 / 8164 paged in the same-era parity run (-5 to -6%). It also measured 1563 vs
  1635 us at turn_4 full and 405 vs 419 us at turn_1 full. It lost for two reasons. Its torch
  advanced-index gather (k_cache.view(-1,2,64)[page_table.clamp(min=0)],
  at::native::vectorized_gather_kernel) cost 100.5 us at turn_1 and 391.6 us at turn_4, about 0.2
  TB/s. And it was applied to every shape unconditionally, sliding included. This hypothesis keeps
  the TMA kernel win and removes both losses. (1) GATHER: a new Triton kernel
  _kv_gather_contig(k_cache, v_cache, page_table, cache_seqlens, k_buf, v_buf, W) with grid (B,
  cdiv(W, 128)). Each program loads 128 page_table entries of request b and widens them to int64
  rows. For each position p < cache_seqlens[b] it copies the 128-byte K row and the 128-byte V row
  (both kv heads, viewed as uint8 [rows, 128], 16-byte vector loads) into k_buf[b, p] and v_buf[b,
  p]. For p >= cache_seqlens[b] it stores zeros, so a masked tail tile never feeds FP8 NaN bytes
  into P x V. W = page_table.shape[1] * page_size, a shape. k_buf and v_buf are torch.empty((B, W,
  2, 64), float8_e4m3fn) per call: shape-derived, graph-safe, no module-global buffers. The kernel
  runs after store_cache, so the step's own rows are read back exactly as FA3 reads them today.
  (2) CALL: flash_attn_with_kvcache(q=q_fp8, k_cache=k_buf, v_cache=v_buf, page_table=None,
  cache_seqlens=data['cache_seqlens'], causal=True, the same softmax_scale, window_size (-1, -1),
  scheduler_metadata=None), in one of two modes. VARLEN keeps cu_seqlens_q and max_seqlen_q as
  master does. BATCHED passes q.view(B, L, 16, 64) with no cu_seqlens_q. That is legal because
  every prefill config has the same q_len for each request. BATCHED is the form the
  fa3-batched-mode review asked to re-mint on a contiguous base. The output is reshaped to [T,
  1024] for o_norm_gate exactly as now. (3) PREPARE-TIME HARD GATE: for each full-layer class the
  judge scores (turn_1 b1 L753 ctx~80000; turn_4 b4 L753 ctx~80000; first_turn b1 L16384 at ctx 0,
  32768 and 65536), build synthetic FP8 caches with a randomly permuted page-size-1 page table at
  the scored widths. CUDA-graph capture three candidates on the same pre-cast FP8 q: A = master's
  paged call; B = gather + contiguous VARLEN; C = gather + contiguous BATCHED. Before every replay
  write a flush buffer of at least 2x L2. Alternate the order, and take the median of at least 25
  replays. Route the class to the faster of B and C only if it is <= 0.97 x A; otherwise A. Key
  the decision by config-only fields, as attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc's
  _full_attn_route did: (batch, q_len, round((page_table width - q_len) / 16384)). Any key not
  timed runs A. Log '[fa3 contig] GATE <class>: paged X us, contig-varlen Y us, contig-batched Z
  us -> A|B|C', with the gather's own us from a separate capture, and one 'PATH CHOSEN' line per
  routed shape the first time kernel_fn sees it (a de-dup set used for printing only, never for
  routing). (4) SELF-CHECK in prepare(), before the gate: B and C against A on coherent synthetic
  data (keys sharing a direction, so softmax is peaked, per LESSONS 2026-10-25 on FP8 P) for b1
  ctx3000 L753, b4 seq_lens {0, 777, 5000, 80000} L753, b1 ctx0 L16384 and b2 {37, 1000} L211.
  Pass: rel L2 <= 1e-2, worst row <= 5e-2, all finite. A failure turns that variant off. Any
  exception anywhere in (1)-(4) leaves every class on A, so master's path runs. kernel_fn gains
  only two torch.empty calls, one Triton launch and the FA3 call: no host sync, no branching on
  values, and nothing derived from data is cached.
expected:
  - shapes: "*ctx=65536*layer=full*load=first_turn*"
    direction: faster
    why: FA3 kernel 8352.6 us of 18349.9 us (45%). The contiguous run measured -5.1% on FA3 (8164 -> 7747); the gather moves ~42 MB (~15-20 us). Net ~-400 to -600 us, 2-3% of the shape. Read the FA3 row and the GATE line.
  - shapes: "*ctx=32768*layer=full*load=first_turn*"
    direction: faster
    why: FA3 4542.4 us of 14825.7 (30%). -5.4% measured on contiguous (4545 -> 4300); the gather is ~25 MB. Net ~-220 us, ~1.5% of the shape.
  - shapes: "*ctx=0*layer=full*load=first_turn*"
    direction: flat
    why: FA3 911.7 us of 11599.6; -5% is ~45 us, 0.4% of the shape, inside noise.
  - shapes: "*layer=full*load=turn_4*"
    direction: flat
    why: FA3 1643.1 us. The contiguous kernel saved ~72 us, but a ~165 MB gather (~50 us at ~3.3 TB/s) eats most of it. The gate decides; at best ~-25 us (0.7%).
  - shapes: "*layer=full*load=turn_1*"
    direction: flat
    why: FA3 417.5 + 6.9 combine. A -14 us kernel saving against a ~12 us gather; the gate most likely keeps the paged path.
  - shapes: "*layer=sliding*"
    direction: flat
    why: "Sliding layers keep master's paged FA3 call unchanged."
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T152551Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T152552Z
base_sha: 7faa35ba9069ec4a6c1942887916e7e77d381e1c
base_blob: a2fe2a75893c60d4f68cf94fcce8ce8f9b708a71
branch: hyp/muse_prefill_layer/fa3-full-contiguous-kv-triton-gather-gated
worktree: .worktrees/muse_prefill_layer/fa3-full-contiguous-kv-triton-gather-gated
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T153230Z
impl_commit: 8367d46cf7527dea6fd3a2644c7b4e46c6378fc4
submission_sha256: 6a87e63c212c463fdb8d647d539d387ca03a3e3ddf4b0f8d788394977a7e30ef
norm_sha256: 3b7c506364e79a82fc669458246dd4d5a53b9de437a9819e4c3b103c1de893b1
submission_blob: 9fb1f7c4b4282dd7461c6cdba926fe3ebf12128e
lint_warnings: 74
attempt_file: 2026-10-25T153230Z-submit-fa3-full-contiguous-kv-triton-gather-gated.md
prepared_at: 2026-10-25T153230Z
job: 1cc92a43b8be40f8
dispatched_at: 2026-10-25T153238Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4586709689454935
speedup_ci: [1.4367909044113112, 1.4805510334796759]
harvested_at: 2026-10-25T153923Z
best_at_harvest: 1.4480354973438967
promotion: merged
hand_landed: true
promotion_commit: 020566aa96573ce235f3b70405c66115316f0fbf
---

# fa3-full-contiguous-kv-triton-gather-gated

## Change

Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 changes. Sliding layers, the page-64 and kv-scaled cases, qkv GEMM, FusedQKNorm, the q/k/v FP8 casts, store_cache, o_norm_gate and everything after stay master's. MECHANISM: at page_size 1 FA3 runs its paged non-TMA path: producer warps gather KV rows with cp.async, PackGQA is forced and the key tile is smaller. fa3-contiguous-kv-gather (job 75bbf278dd9e430e, regression 0.988x) measured FA3's own kernel on a contiguous cache at 857 / 4300 / 7747 us on first_turn full ctx 0 / 32768 / 65536, against 907 / 4545 / 8164 paged in the same-era parity run (-5 to -6%). It also measured 1563 vs 1635 us at turn_4 full and 405 vs 419 us at turn_1 full. It lost for two reasons. Its torch advanced-index gather (k_cache.view(-1,2,64)[page_table.clamp(min=0)], at::native::vectorized_gather_kernel) cost 100.5 us at turn_1 and 391.6 us at turn_4, about 0.2 TB/s. And it was applied to every shape unconditionally, sliding included. This hypothesis keeps the TMA kernel win and removes both losses. (1) GATHER: a new Triton kernel _kv_gather_contig(k_cache, v_cache, page_table, cache_seqlens, k_buf, v_buf, W) with grid (B, cdiv(W, 128)). Each program loads 128 page_table entries of request b and widens them to int64 rows. For each position p < cache_seqlens[b] it copies the 128-byte K row and the 128-byte V row (both kv heads, viewed as uint8 [rows, 128], 16-byte vector loads) into k_buf[b, p] and v_buf[b, p]. For p >= cache_seqlens[b] it stores zeros, so a masked tail tile never feeds FP8 NaN bytes into P x V. W = page_table.shape[1] * page_size, a shape. k_buf and v_buf are torch.empty((B, W, 2, 64), float8_e4m3fn) per call: shape-derived, graph-safe, no module-global buffers. The kernel runs after store_cache, so the step's own rows are read back exactly as FA3 reads them today. (2) CALL: flash_attn_with_kvcache(q=q_fp8, k_cache=k_buf, v_cache=v_buf, page_table=None, cache_seqlens=data['cache_seqlens'], causal=True, the same softmax_scale, window_size (-1, -1), scheduler_metadata=None), in one of two modes. VARLEN keeps cu_seqlens_q and max_seqlen_q as master does. BATCHED passes q.view(B, L, 16, 64) with no cu_seqlens_q. That is legal because every prefill config has the same q_len for each request. BATCHED is the form the fa3-batched-mode review asked to re-mint on a contiguous base. The output is reshaped to [T, 1024] for o_norm_gate exactly as now. (3) PREPARE-TIME HARD GATE: for each full-layer class the judge scores (turn_1 b1 L753 ctx~80000; turn_4 b4 L753 ctx~80000; first_turn b1 L16384 at ctx 0, 32768 and 65536), build synthetic FP8 caches with a randomly permuted page-size-1 page table at the scored widths. CUDA-graph capture three candidates on the same pre-cast FP8 q: A = master's paged call; B = gather + contiguous VARLEN; C = gather + contiguous BATCHED. Before every replay write a flush buffer of at least 2x L2. Alternate the order, and take the median of at least 25 replays. Route the class to the faster of B and C only if it is <= 0.97 x A; otherwise A. Key the decision by config-only fields, as attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc's _full_attn_route did: (batch, q_len, round((page_table width - q_len) / 16384)). Any key not timed runs A. Log '[fa3 contig] GATE <class>: paged X us, contig-varlen Y us, contig-batched Z us -> A|B|C', with the gather's own us from a separate capture, and one 'PATH CHOSEN' line per routed shape the first time kernel_fn sees it (a de-dup set used for printing only, never for routing). (4) SELF-CHECK in prepare(), before the gate: B and C against A on coherent synthetic data (keys sharing a direction, so softmax is peaked, per LESSONS 2026-10-25 on FP8 P) for b1 ctx3000 L753, b4 seq_lens {0, 777, 5000, 80000} L753, b1 ctx0 L16384 and b2 {37, 1000} L211. Pass: rel L2 <= 1e-2, worst row <= 5e-2, all finite. A failure turns that variant off. Any exception anywhere in (1)-(4) leaves every class on A, so master's path runs. kernel_fn gains only two torch.empty calls, one Triton launch and the FA3 call: no host sync, no branching on values, and nothing derived from data is cached.

## Expected

- `*ctx=65536*layer=full*load=first_turn*` faster: FA3 kernel 8352.6 us of 18349.9 us (45%). The contiguous run measured -5.1% on FA3 (8164 -> 7747); the gather moves ~42 MB (~15-20 us). Net ~-400 to -600 us, 2-3% of the shape. Read the FA3 row and the GATE line.
- `*ctx=32768*layer=full*load=first_turn*` faster: FA3 4542.4 us of 14825.7 (30%). -5.4% measured on contiguous (4545 -> 4300); the gather is ~25 MB. Net ~-220 us, ~1.5% of the shape.
- `*ctx=0*layer=full*load=first_turn*` flat: FA3 911.7 us of 11599.6; -5% is ~45 us, 0.4% of the shape, inside noise.
- `*layer=full*load=turn_4*` flat: FA3 1643.1 us. The contiguous kernel saved ~72 us, but a ~165 MB gather (~50 us at ~3.3 TB/s) eats most of it. The gate decides; at best ~-25 us (0.7%).
- `*layer=full*load=turn_1*` flat: FA3 417.5 + 6.9 combine. A -14 us kernel saving against a ~12 us gather; the gate most likely keeps the paged path.
- `*layer=sliding*` flat: Sliding layers keep master's paged FA3 call unchanged.

## Risk

Low risk: the same FA3 binary, the same FP8 q and KV and P precision, and every key attended; only the KV source changes from paged rows to a contiguous copy of the same rows. Headline stake is small (~+0.3%; first_turn load +0.6%, turn_4 at most +0.3%), so judge it on the FA3 row per routed shape and the GATE lines. The scored effect may sit inside the +-1.5% interval. Distinct from the regressed fa3-contiguous-kv-gather in form, which its follow-up forbade repeating: a bandwidth-sane Triton gather instead of torch indexing, full layers only, a per-class cold-L2 timing gate instead of unconditional routing, and a zeroed tail. Failure modes: (a) FA3's contiguous path may pick a tile config whose advantage is smaller at the judge's widths than at the parity era's: the gate makes this flat, not a loss; (b) the batched-mode variant may fail to accept FP8 q without cu_seqlens_q in sgl_kernel's wrapper: caught by the self-check, which turns C off. The step's rows must be gathered after store_cache (same stream). If it lands, it is the base a later stage can use to put the prefix gather on a side stream under the qkv GEMM (prefix rows do not depend on this call's qkv), which would let turn_4 route too.

## Adversarial review

Reviewed 2026-10-25T152551Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. The prepare-time GATE lines time paged and contiguous FA3 against each other in the same process, which is a better A/B than any scored companion. But the expected headline move is about +0.15% geomean, roughly 10x smaller than the +-1.5% interval, so judge this scored submit on the GATE lines and the full first_turn FA3 rows, never on the headline.

### Justification
Mechanism is real and already measured once. At page_size 1, FA3 takes the non-TMA paged path: cp.async row gathers and forced PackGQA. The regression run 75bbf278 (5/5 tests passed, so the contiguous VARLEN API works with an FP8 cache) measured FA3 alone at 856.7 / 4299.7 / 7746.9 us on first_turn full ctx 0 / 32768 / 65536, plus 405.4 us on turn_1 and 1563.0 us on turn_4. Its losses came from the torch vectorized_gather_kernel: 100.5 us on turn_1 and 391.6 us on turn_4, about 0.4 TB/s on 165 MB. That run also gathered on the sliding layers, which no longer happens. The judge's page_table is a torch.randperm scatter (reference.py:261), so the gate's randomly permuted synthetic page table matches what the judge serves, and the judge's cold-L2 replays match the gate's flush. Arithmetic: turn_4 needs (1563 + ~55-65 us gather of 165 MB of scattered 128 B rows) / 1647 = 0.98-0.99, and turn_1 needs (405 + ~14) / 422 = 0.99. Both miss the 0.97 bar, so the gate keeps A there, as the md predicts. first_turn ctx0 at (857 + ~3) / 907-912 = 0.94-0.95 and ctx32768 at (4300 + ~10) / 4525-4545 = 0.95 should route. Converting to the score: about 16 x (390 + 225 + 45) us = ~10.5 ms of a 2.297 s first_turn load (0.46%), which is ~0.15% of the geomean. The md's '+0.6% first_turn load' is about 30% optimistic. WEAKEST LINK: the 5% FA3 advantage compares one contiguous sample against paged rows from other runs, and those rows themselves vary: ctx65536 paged FA3 is 8164.1 (parity 62025c), 8004.1 (check 4e6f815a) and 8352.6 us (scored bff8e367), about +-2%. Against the 8004 row, the ctx65536 advantage is only 3.2%, right at the 0.97 bar. So the class with the most at stake may route differently from run to run (in prepare, in the judge, and in the final-review reruns). That does no correctness harm, because both paths are exact, but each run's attribution must use the GATE line it printed. The md's cited numbers (8352.6 / 18349.9, 4542.4, 911.7, 1643.1, 417.5) are from bff8e367, which is consistent.

### Confounders on the current kernel
- sglang-pipeline-parity (muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.patch): it set the exact FA3 call being swapped (paged, VARLEN, scheduler_metadata from data) and the store_cache that writes the step's rows. It does not confound the measurement, because the change lives entirely inside that call. The one hard requirement it imposes is ordering: the Triton gather must run on the main stream after store_cache, or the step's own 753/16384 keys are stale.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128 / align-two-phase-all-t-rega: no shared kernels, traffic or numerics with attention. The only indirect coupling is the 300 s prepare budget: master already uses 124.4 s of its internal 200 s (bff8e log). The new self-check, Triton compile and 5-class x 3-candidate x >=25-replay gate (~5-10 s) must run inside that budget accounting and fall back to A on overrun, or a slow prepare could starve the rega gates.
- No landed item changes the L2 state FA3 sees (the judge flushes before each replay), so the gather-then-FA3 L2 residency is the same in the gate and the judge. Remaining landed items (stack-tonight-five, hybrid/triton expert stages, ssdq s1b) touch only the MoE half: no interaction.

### Interference and A/B needs
No companion. Every landed item outside parity sits in the MoE half. Parity is the base and cannot sensibly be reverted. A revert or tune A/B therefore has nothing to isolate. The two sub-effects the md separates (gather cost vs. FA3 TMA kernel gain, and VARLEN vs. BATCHED) come for free: they are separate rows in the profile (Triton gather kernel vs the flash::enable_sm90_or_later row), and the GATE line reports A/B/C and the gather's own time from the same process, with cold L2 and alternated order. Every outcome is already readable. If the GATE routes ctx32768/65536 to B or C and the FA3 row drops by more than the gather row adds, the mechanism is confirmed. If the GATE keeps A everywhere, the result is flat, nothing is lost, and the contiguous advantage at these widths is below 3% (the idea closes). If the GATE routes but the scored rows lose, the gate's synthetic timing disagrees with the judge, and that is visible by comparing the GATE us against the profile rows. A scored companion would only add a second noisy headline, ~10x inside its own CI, and the in-process gate already answers the question it would ask.

### Open questions
- The ctx65536 class is the one with the most at stake, and its contiguous advantage may sit at 3% against the 0.97 bar. Should the bar be 0.98 for first_turn classes only? There the gather is <1% of FA3 time and a 2% win is still ~160 us x 16. As written, a real 3% win can be gated off by gate noise.
- BATCHED mode (C) with FP8 q and no cu_seqlens_q in sgl_kernel's wrapper is unverified. The self-check turns it off on failure, but an exception in the wrapper during graph capture must also be caught per variant, so that a C failure does not knock B off too (the md says 'any exception anywhere leaves every class on A', which would discard a working B).
- Turn_1 paged FA3 already runs a split plus FlashAttnFwdCombine (6.9 us). Contiguous VARLEN may choose a different num_splits. If the gate ever routes turn_1, read both the FA3 row and the combine row. Expected: the gate keeps A.
- Since this is a scored submit with an expected headline of about +0.15%, the board entry will most likely land within noise of 1.448x, above or below it. The harvest should classify it from the GATE lines and the full first_turn FA3 rows, not from the speedup.


## Implementation notes

Base 7faa35ba equals master 47eaa27e for submission.py (master moved only md claims), so no merge was needed. All edits are in `muse_prefill_layer/submission.py` of the worktree: new block `FA3C_*` / `_kv_gather_contig_kernel` / `_fa3c_*` before `kernel_fn`, one call in `prepare()`, and the routed branch in `kernel_fn`.

What changed: Triton kernel `_kv_gather_contig_kernel` (grid (B, cdiv(W,128)), 128 B rows as uint8, zeros for p >= cache_seqlens[b], `do_not_specialize=["width","pt_stride"]`, one compile serves every shape). `_fa3c_attend` runs A (master's paged call), B (gather + contiguous varlen) or C (gather + contiguous batched, q viewed [B,L,16,64], no cu_seqlens_q). `_fa3c_setup` runs a coherent-data self-check then a cold-L2 CUDA-graph gate (256 MB flush, 25 alternations, order rotated and flipped, median) per class, and fills `state['fa3c']['route']` keyed by `(batch, q_len, round((W - q_len)/16384))`. kernel_fn routes only when window == 0, page_size == 1, kv_scale == 1 and the key is in the dict; otherwise master's call is untouched.

Review items applied:
1. Gate placed AFTER `_rega_setup` (after master's whole setup); own deadlines `FA3C_START_DEADLINE_S` 235 s (skip all, master's paged path) and `FA3C_CLASS_DEADLINE_S` 265 s (skip remaining classes). First_turn classes are timed first.
2. Exceptions are caught per variant (selfcheck, capture and replay): B failing does not switch off C and vice versa. Only an unexpected outer exception clears all routes.
3. Bar 0.98 for the three first_turn classes, 0.97 for turn_1 and turn_4.
4. Triton gather is launched (compiled) in prepare()'s self-check and in every class's eager warm-up before any capture, with `do_not_specialize` on the integer args. Buffers are shaped from page_table width and batch, held in the state's route dict, never module-global. If the live width differs from the gated width the call uses shape-derived `torch.empty` (graph-safe) instead.
5. One `GATE` line per class (paged / varlen / batched / gather alone / bar / choice) and one `PATH CHOSEN` line in prepare(), reprinted after the align2 lines so the log tail keeps them; kernel_fn prints a `PATH CHOSEN` line the first time it sees each (key, width) (print-only set in state).
Other review points: the gather runs after store_cache on the same stream; turn_1 combine row to be read if it ever routes; harvest should classify from the GATE lines and FA3 rows, not the headline.

Deviations: route dict keeps the key spec'd, plus an exact-width check (above). Gate widths are my reading of reference.py: fixed spread W = ctx + L, turn_4 agentic spread W = ceil(1.075 ctx) + L with lens at ctx*(1 +- 0.075). Per-state buffers allocated at gate time (about 0.4 GB per state at most), instead of per-call torch.empty. The self-check does not replay across two states (buffers are per state and there are no globals).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds a dict lookup on shapes, one Triton launch and the FA3 call; no sync, no value branching (diff checked).
- [x] Nothing derived from `data` is cached across calls: routes and buffers come from synthetic prepare() cases; the seen-set is print-only.
- [x] Weights: n/a, untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (copied bytes); queries FP8 E4M3 as master.
- [x] Attention sees every key: contiguous copy of every cached row, zeros only beyond cache_seqlens; causal, window (-1,-1) as master on full layers.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, processes or network.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: OK with warnings (only host-sync warnings in prepare() and uint8 byte-buffer dtype warnings, no errors). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only plus minimal fixes; no GPU on this node). **Verdict: dispatch-with-fixes.**

Fixes in the worktree submission.py:
1. kernel_fn no longer prints a per-shape `PATH CHOSEN` line, and `state["fa3c"]["seen"]` is gone. That was per-call mutable state plus a print inside the graph-captured call. The prepare-time GATE and PATH CHOSEN lines remain and are enough.
2. The gate could pick a variant that failed mid-replay. `med` kept any variant with at least 12 samples, even one dropped from `graphs` after a replay error. It now keeps only variants still in `graphs`.
3. New byte-exact gather check `_fa3c_gather_exact` in each self-check case. It fills the buffers with 0xFF (FP8 NaN) first, then requires every live row to equal `cache[page_table[b, :n]]` byte for byte and every position at or past cache_seqlens to be 0. A miss or a raise turns both B and C off. Each self-check line also logs `bitwise-equal to paged`, for information only. The FA3 output itself is not required to be bit-identical: contiguous TMA and paged cp.async use different tiles and PackGQA, so the accumulation order differs. A bit-exact gate would never pass. The output check stays rel L2 <= 1e-2, worst row <= 5e-2, vs the paged call.

Verified with no change needed:
- W and keys come from shapes and config only: W = page_table.shape[1], which reference.py sizes from the config as `_longest(ctx, spread) + q_len` (turn_4 = 86753, matching the gate). batch and q_len come from cfg. cache_seqlens is read only on device, as the Triton mask. The fallback `torch.empty` is sized (batch, width, 2, 64).
- Per-state buffers are not module-global and are fully rewritten on every call (all W positions, zeros past cache_seqlens), so sharing them between two states would also be safe.
- `_fa3c_setup` runs after `_rega_setup`, so the 150 s rega first_turn start deadline is untouched. fa3c has its own 235 s and 265 s deadlines.
- Variant C fails closed. Exceptions are caught per variant in the self-check, in capture and in replay, and a wrong output fails the tolerance. Sliding layers, page 64, kv-scaled and every untimed key stay on master's call. No test shape hits a routed key.

Residual risk: a sticky CUDA fault (illegal address) inside FA3 batched mode C during the self-check could not be contained by try/except, and the ctx65536 class sits near its 0.98 bar, so its routing may differ from run to run.

## Result

- job `1cc92a43b8be40f8` (submit), status done, passed True, tests 5/5
- speedup 1.4587x, 95% CI [1.4368, 1.4806], score 265586.8994774491 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4480354973438967, "ci": [1.4263149648837383, 1.469756029804055], "job": "bff8e36788a04917"}; base result: muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/fa3-full-contiguous-kv-triton-gather-gated.result.json`, `muse_prefill_layer/hypotheses/done/neutral/fa3-full-contiguous-kv-triton-gather-gated.patch`; branch `hyp/muse_prefill_layer/fa3-full-contiguous-kv-triton-gather-gated`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 704.38 | 720.18 | 0.9946 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1097.89 | 1132.2 | 1.0055 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2098.36 | 2152.68 | 0.9906 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3644.6 | 3701.15 | 1.0191 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 12078.49 | 11852.94 | 1.0342 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 11285.6 | 10947.18 | 1.0234 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15310.56 | 15058.23 | 1.0423 | gained |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18819.51 | 18490.87 | 1.0471 | gained |

gained on full first_turn. Expected hit: 5/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: fa3-full-contiguous-kv-triton-gather-gated`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

