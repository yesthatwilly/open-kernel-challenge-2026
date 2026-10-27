---
problem: muse_prefill_layer
slug: experts-rega-round-rotated-schedule
created: 2026-10-26T223951Z
base: master
area: w13 + w2 expert GEMMs (rega persistent item->CTA schedule)
change: >
  Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk
  items changes; every item is computed by master's code, so outputs are bit-identical. MECHANISM:
  the producer thread and the consumer warpgroups both walk for (it = blockIdx.x; it < total; it
  += gridDim.x) with gridDim = 132, and decode_item takes chunk c = f + (it - f*T) % k of an
  expert with k chunks (T = n_tiles = 64 for w13, 32 for w2; every group starts at a multiple of
  T). 132 and f*T are multiples of 4, so for every expert with k = 2 or 4 chunks CTA b always gets
  chunk b % k: at k = 2 the even CTAs take all the full 128-route chunks and the odd CTAs all the
  short tail chunks (N bucket 16-64, roughly 0.6x the cost), at k = 4 the CTAs with b % 4 == 3
  take every tail. In the low class (k = 1, 64-item groups, 132 = 2*64 + 4) CTAs below 64 see
  mostly even expert groups and CTAs 64-127 mostly odd ones. A launch lasts as long as its busiest
  CTA. CHANGE: derive a module muse_prefill_rega_rot from master's source (master's module and
  strings untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch:
  asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays
  over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last
  prepare lines) in which BOTH loops become: for (int r = 0;; ++r) { const int it = r * gridDim.x
  + (blockIdx.x + r * (n_tiles + 1)) % gridDim.x; if (it >= total) break; <master's loop body> }.
  Each round still covers items [r*G, r*G + G) (same set, same chunk-minor neighbours, same L2
  working set, a bijection inside the round), but the odd shift n_tiles + 1 moves every CTA by one
  group plus one item per round, so its chunk index and group parity change every round. The
  producer and the consumers must use the identical expression (the stage counter n still advances
  KS per item). GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y
  with master's rega on random, skewed, one and pop routings, plus the two-state replay check;
  then 2 interleaved rounds x 11 cold-L2 graph alternations of master's routed rega pair vs the
  rot pair on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick
  give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random routing
  (random routing at T=3012 gives k = 1 for every expert, so it cannot show the effect; the pop
  routing gives k = 1-6). ON for the class only if pop <= 0.98 in both rounds and random <= 1.00.
  state['rega_rot_ok'][cls] is a plain bool per state; _experts_rega launches the rot module's
  rega_gemm when it is set. Print per class: experts by k (1, 2, 3, 4, >4), KERNELS w13 / w2 us
  master vs rot on both routings, PATH CHOSEN. A build or compare failure keeps master's kernel
  for that class.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class: the judge-like ~1.8x-skewed T=3012 routing has many k=2 experts whose full chunks all land on the even CTAs; rebalancing should take ~4-10% off rega w13+w2 (1098.5 us sliding, 1133.9 us full), ~2-5% of each turn_4 shape"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low class: each CTA alternates between even and odd expert groups instead of mostly one parity; ~2-4% off rega 418.5 / 437.9 us, ~1-2% of each turn_1 shape"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "ft: k ~8 per expert and 132 = 4 mod 8, so CTAs with b % 8 in {3, 7} hold the tails; ~2-4% off rega 5067-5739 us, ~1-2% of each first_turn shape"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-triton-persistent-grid-first-turn: that changed the old Triton engine's grid size at T > 4096; this keeps rega's 132-CTA persistent grid and only rotates the item->CTA map so CTAs stop aliasing onto chunk (b % k) of k=2/4 experts and onto one expert-group parity"
review: proceed
status: running
review_at: 2026-10-27T095330Z
review_companions: null
stage: implemented
claimed_at: 2026-10-27T095330Z
base_sha: bb27243410f70166648b1b7ac353fa74b08cb1ad
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-round-rotated-schedule
worktree: .worktrees/muse_prefill_layer/experts-rega-round-rotated-schedule
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-27T100624Z
impl_commit: 2afa5f06c097ab4bec808546012418dfef871045
submission_sha256: eec14c3fa4814dcbfe9670626fd4bee922c1328f58617b278f8689d2b8794f81
norm_sha256: c66260217c69fc4bfe58a65278fe9fe6f3028a825eaecd25a9fa50180bafcdd3
submission_blob: 7625a6d5a0b9ae4ddfb8f79af713ae09d20385f5
lint_warnings: 107
attempt_file: 2026-10-27T100624Z-check-experts-rega-round-rotated-schedule.md
prepared_at: 2026-10-27T100624Z
---

# experts-rega-round-rotated-schedule

## Change

Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk items changes; every item is computed by master's code, so outputs are bit-identical. MECHANISM: the producer thread and the consumer warpgroups both walk for (it = blockIdx.x; it < total; it += gridDim.x) with gridDim = 132, and decode_item takes chunk c = f + (it - f*T) % k of an expert with k chunks (T = n_tiles = 64 for w13, 32 for w2; every group starts at a multiple of T). 132 and f*T are multiples of 4, so for every expert with k = 2 or 4 chunks CTA b always gets chunk b % k: at k = 2 the even CTAs take all the full 128-route chunks and the odd CTAs all the short tail chunks (N bucket 16-64, roughly 0.6x the cost), at k = 4 the CTAs with b % 4 == 3 take every tail. In the low class (k = 1, 64-item groups, 132 = 2*64 + 4) CTAs below 64 see mostly even expert groups and CTAs 64-127 mostly odd ones. A launch lasts as long as its busiest CTA. CHANGE: derive a module muse_prefill_rega_rot from master's source (master's module and strings untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines) in which BOTH loops become: for (int r = 0;; ++r) { const int it = r * gridDim.x + (blockIdx.x + r * (n_tiles + 1)) % gridDim.x; if (it >= total) break; <master's loop body> }. Each round still covers items [r*G, r*G + G) (same set, same chunk-minor neighbours, same L2 working set, a bijection inside the round), but the odd shift n_tiles + 1 moves every CTA by one group plus one item per round, so its chunk index and group parity change every round. The producer and the consumers must use the identical expression (the stage counter n still advances KS per item). GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y with master's rega on random, skewed, one and pop routings, plus the two-state replay check; then 2 interleaved rounds x 11 cold-L2 graph alternations of master's routed rega pair vs the rot pair on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random routing (random routing at T=3012 gives k = 1 for every expert, so it cannot show the effect; the pop routing gives k = 1-6). ON for the class only if pop <= 0.98 in both rounds and random <= 1.00. state['rega_rot_ok'][cls] is a plain bool per state; _experts_rega launches the rot module's rega_gemm when it is set. Print per class: experts by k (1, 2, 3, 4, >4), KERNELS w13 / w2 us master vs rot on both routings, PATH CHOSEN. A build or compare failure keeps master's kernel for that class.

## Expected

- `*load=turn_4*` faster: high class: the judge-like ~1.8x-skewed T=3012 routing has many k=2 experts whose full chunks all land on the even CTAs; rebalancing should take ~4-10% off rega w13+w2 (1098.5 us sliding, 1133.9 us full), ~2-5% of each turn_4 shape
- `*load=turn_1*` faster: low class: each CTA alternates between even and odd expert groups instead of mostly one parity; ~2-4% off rega 418.5 / 437.9 us, ~1-2% of each turn_1 shape
- `*load=first_turn*` faster: ft: k ~8 per expert and 132 = 4 mod 8, so CTAs with b % 8 in {3, 7} hold the tails; ~2-4% off rega 5067-5739 us, ~1-2% of each first_turn shape

## Risk

Bit-exact by construction. Size: ~+1-2% headline if the modeled imbalance holds, enough to clear the margin alone; if the pop ratio is above 0.98 the class stays OFF and the run is master re-measured, so read the KERNELS lines, not the headline. Adds one ~35 s module build to prepare (140 s today). This is not the consumer dequant/MMA overlap DIAGNOSIS bans: the per-item code is master's.

## Adversarial review

Reviewed 2026-10-27T095330Z by an adversarial subagent (model opus).

**Verdict: proceed.** The aliasing mechanism is real: decode_item gives CTA b chunk (b mod k) of every k=2/4 expert and a simple model says rotation removes about half the imbalance at turn_4. The size claims are optimistic for first_turn and turn_1, and the 'random <= 1.00' gate rule will turn the high class OFF about half the time on noise alone. Proceed with the gate fixed and the low-class precedence made explicit; no companions.

### Justification
Mechanism: decode_item uses c = f + local % k; grid 132 and f*T a multiple of 32, so for k in {2,4} CTA b always gets chunk b mod k and tails pile onto the same CTAs. rega is L2->SM ingest-capped (5.8-6.4 TB/s), so tail items are cheaper. A read-only max/mean model (bytes per stage, independent CTAs): turn_4 pop w13 1.129 -> 1.074, w2 1.147 -> 1.115 (upper bound ~5%/3%); turn_4 random flat (k=1); turn_1 pop modest; ft pop ~1%; ft random rotation WORSE. Weakest link: if the ingest cap is a shared L2 limit, imbalance costs less than max/mean. Expect ~1-3% on turn_4 rega, roughly +0.3-0.7% headline, inside the CI; read the KERNELS lines. Prior (experts-triton-persistent-grid-first-turn) differs. Run before the rotation combos.

### Confounders on the current kernel
- combo-b-normgather-two-cta-low-regate (on master): rega_low_variant 0/1/2 is chosen per prepare and _experts_rega launches rega_lo_ext when variant != 0; rot replaces only master's rega_gemm, so the low-class rot result is real only when variant 0 is chosen, and _rega_lo_phase gates against master's rega_gemm. Required: use rot only when variant == 0, run the rot phase after _rega_lo_phase, print rega_low_variant beside the rot low verdict.
- combo-b prepare budget: the ~35 s rot build plus gating must run last in prepare so _rega_lo_phase / _fa3c_setup deadlines are unaffected; log total prepare seconds.
- experts-turn-rega-s4-* / align-two-phase-all-t-rega define the permuted item space (base, not confounders).
- combo-a / fa3-full-contiguous-kv: no interaction.

### Interference and A/B needs
No companion: bit-exact and self-gated per class with in-process interleaved master-vs-rot KERNELS timings. Two fixes inside the parent: (1) GATE: at the high class random routing gives k=1 for every expert and zero effect, so use random <= 1.01 (or the gate's IQR slack) as a no-regression guard; at ft keep the strict rule. (2) SHIFT (optional): n_tiles+1 = 33 divides 132 for w2; shifts coprime to 132 (e.g. 37 for w2, 67 for w13) model slightly better; report the shift used.

### Open questions
- Is the ingest cap per-SM or shared L2? Per-CTA clock64 spans (min/median/max) from the gate's master run would answer it.
- Which rega_low_variant does master pick on the judge? If V64/V32 the low-class half cannot act.
- If rotation shows only ~2%, should experts-rega-dynamic-item-scheduler or lpt-group-order move ahead of the rotation combos?
- Should ft use the judge-like pop routing only?


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-rega-round-rotated-schedule`, file `muse_prefill_layer/submission.py` only (+699 / -1). Master's `_REGA_HEAD` / `_REGA_BODY` / `_REGA_CPP`, module `muse_prefill_rega_v2`, its v1 fallback and their sha256 checks are untouched (the rot phase requires `_rega_lo_identity()[0]`, i.e. master's v2 text still hashes to `REGA_V2_SHA256`; checked offline: True).

- **New module `muse_prefill_rega_rot`** (`_rega_rot_src`): derived from master's head + wgmma wrappers + body by exact-once asserted edits (`_REGA_ROT_EDITS`; any miss -> no build, class OFF). Kernel renamed `rega_kernel_rot<G, BIG, DIAG>`, two new kernel params `int rot_shift, long long* spans`. BOTH loop headers (asserted exactly two) `for (int it = blockIdx.x; it < total; it += gridDim.x) {` become `for (int rnd = 0;; ++rnd) { const int it = rot_item(rnd, rot_shift); if (it >= total) break;` with ONE shared `__device__ rot_item(rnd, shift) = rnd * G + (blockIdx.x + rnd * shift) % G`, so producer and consumers use the identical expression by construction (review fix 3). The loop bodies (decode_item, the stage counter n advancing KS per item, consume_item, epilogue), master's bounded `mbar_wait` (error word 1 on timeout, `dead` flag) and all error words are untouched, so a mismatch would end in error word 1 -> class OFF, never a hang. `break` is correct: if round rnd's item is past total, every later round's is too (pos < G). Host: `rega_gemm` -> `rega_gemm_rot_impl(..., rot_shift, spans)` with its own launch block (asserts 0 <= shift < grid), exported entries `rega_init_rot` (attributes + dynamic-smem attribute for all 6 instantiations), `rega_gemm_rot` (timed; DIAG = false only) and `rega_gemm_rot_diag` (prepare-only). Offline: derived source audits clean (`_rega_audit_cpp`), `diff` vs master's source shows only the edits above. Not compiled here (no nvcc / GPU on this node).
- **Bit-identical by construction:** each round still covers items [rnd*G, rnd*G+G) (bijection inside the round), every item is computed by master's unchanged per-item code into the same output rows; only which CTA computes it and when changes. No cross-item reduction exists (each item owns a disjoint output tile), so the result cannot depend on order. Verified at prepare time anyway (step 2 below).
- **SHIFT (review item 4):** `REGA_ROT_SHIFT_PREF = (67, 37)` (w13, w2), passed per launch as prepare-time ints `state['rega_rot_shift']`; `_rega_rot_shift` bumps to the next value coprime to the actual grid (132 on H100 -> 67 / 37 unchanged). Deviation from the md's `n_tiles + 1` (65 / 33): 33 divides 132 so w2 CTAs would cycle through only 4 positions; used the review's suggested coprime values for both. Both are odd, so for k = 2 / 4 experts the chunk index b+rnd*s mod k still changes every round as the md intends. Shifts are printed in the GATE and PATH CHOSEN lines.
- **Dispatch (`_experts_rega`):** `rot = (not variant) and state['rega_rot_ext'] is not None and state['rega_rot_ok'][_rega_class(T)]`; branch order `variant` (rega_gemm_lo) -> `rot` -> master's `rega_gemm`, for both w13 and w2 (review fix 2: the rot path only when rega_low_variant == 0). Shapes + prepare-time bools / ints only; no syncs or allocation changes in the timed path.
- **Prepare placement (review fix 2):** `_rega_rot_phase` runs after `_rega_lo_phase` (so after `_rega_setup`, `_fa3c_setup`, `_norm_gather_setup` too), last in `prepare()`; its summary lines are printed after the align2 / fa3c / rega-lo summaries, so they are the last prepare lines (review item 5). Own deadlines: start (build) 200 s, class start high/low 262 s, ft 248 s, replay check 276 s, span diag 280 s. Earlier phases' deadlines are unaffected. The last lines print `phase took Xs; prepare total Ys` and the flags (incl. rega_low_variant).
- **Gate per class (`_rega_rot_class`, order high T=3012, low T=753, ft T=16384; deviation: the md lists low first, I gate the class with the largest expected effect first so a deadline squeeze costs ft, not high):** pop routing alpha calibrated per T with `_rega_pop_counts` / `_rega_pop_pick` to 1.83x the uniform local routes (alpha ~2.2 as `_rega_lo_phase`). 1) eager traced canary of the whole `_experts_rega` call with rot ON (norm-gather prenorm as kernel_fn), error word / markers / finite. 2) Exact: per routing random, skewed, one, pop: one align2 + gather, master w13/w2 and rot w13/w2 (rot w2 fed master's hs) into zeroed buffers: `torch.equal` on hs and y, error words 0, both rot launch markers. Prints experts by k (1, 2, 3, 4, >4), total chunks, max k per routing. 3) Timing (`_rega_rot_time`): graphs pair:master, pair:rot (w13 then w2), w13 alone, w2 alone (each module), empty; 2 rounds x 11 alternations, order rotated and flipped, 256 MB flush before every replay, empty-graph baseline subtracted; on the pop routing and the random routing. KERNELS lines: master/rot us and rot/master ratio for pair, w13, w2 per round and pooled, per routing. 4) Only if the timing passes and before 276 s: `_rega_replay_check` (A / deepcopy B / tensor-clone C states, refilled inputs, fp32 reference bound, launch markers) with rot ON for this class, keys `REGA_ROT_STATE_KEYS`.
- **Gate rule (review fix 1):** ON iff pop pair ratio <= 0.98 in BOTH rounds AND random pair ratio (median over both rounds' 22 samples per side) <= 1.01 at high (random T=3012 is k=1 everywhere: a no-regression guard), <= 1.00 at ft (strict; random k~5-8 where rotation is expected to be worse) and <= 1.00 at low (the md's rule; the review's relaxation names only high). Any exception stops later classes (their verdicts OFF); any miss keeps master's kernel for that class. If no class is ON, `state['rega_rot_ext']` is None.
- **Span diagnostic (review open question, item 6):** done WITHOUT touching any timed kernel: master's module is untouched, and the rot module's timed instantiations are `DIAG = false` where all clock code sits behind `if constexpr (DIAG)` (only two extra unused kernel params remain). Two prepare-only `DIAG = true` instantiations (stage-1 tables: w13, w2) record per CTA the clock64 span of the consumer item loop, stages and bytes (weights + activation slabs). Run print-only after every verdict, on the high and low pop routings, with shift 0 (= master's item order exactly) and with the rot shift, eager, warm-up + 256 MB flush. SPANS lines: kcycles min/med/max, cycles max/mean, bytes max/mean, corr(cycles, bytes), stages max/min. Reading: per-SM ingest cap -> cycles max/mean ~ bytes max/mean and corr ~ 1; shared L2 cap -> cycles flatter than bytes. Costs two extra instantiations in the build (estimate +10-15 s nvcc on top of the ~33 s the pdl module took).
- **Log tail:** summary lines truncated to 900 chars; estimated ~6-8 KB for three classes + spans + PATH CHOSEN, printed last.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-round-rotated-schedule diff`:

- [x] CUDA-graph capturable: the only timed-path change is in `_experts_rega` (an `elif rot:` branch calling `rot_ext.rega_gemm_rot` with the same tensors and two prepare-time ints); no `.item()` / `.cpu()` / `synchronize()` / prints there; the branch is on shapes (`_rega_class(T)`) and prepare-time flags, never input values. All new syncs / `.cpu()` / `.tolist()` are in prepare-only functions (`_rega_rot_*`).
- [x] Nothing derived from `data` is cached across calls: state gains `rega_rot_ext` (module), `rega_rot_ok` (bools), `rega_rot_shift` (ints) only; per-call buffers are allocated per call as in master.
- [x] Weights: unchanged; the rot kernel reads master's same `rega_bank` (NVFP4 codes + e4m3 scales, lossless retile from master's prepare); no new weight storage.
- [x] Activations BF16 where SGLang keeps BF16: per-item math is master's (bf16 xs / hs / y, fp32 accumulate); output bit-identical to master's rega kernel (asserted at prepare time). KV cache / queries untouched.
- [x] Attention sees every key: attention code untouched (n/a to this change beyond that).
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network: load_inline of one more extension in prepare() as master already does for rega / ssdq / rega_lo; no harness access.
- [x] All work happens inside the timed call; no output caching or replay: prepare only builds / gates; kernel_fn recomputes everything per call.
- [x] First line `# okc: problem=muse_prefill_layer` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_prefill_layer`): `OK with warnings`, 0 errors. 107 warnings (master: 93); the 14 new ones are all `host-sync ... (outside kernel_fn's call graph; fine in prepare())` in the new `_rega_rot_*` prepare functions, e.g.:

```
warn  host-sync    L6100: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6101: be = block_experts[:used].cpu().long()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6131: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6333: sp = spans.view(grid, 3).cpu().double()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6338: srt = sorted(cyc.tolist())  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` OK; no `__pycache__` left in the worktree.
