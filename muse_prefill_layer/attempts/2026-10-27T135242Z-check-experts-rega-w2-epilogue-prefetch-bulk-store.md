# muse_prefill_layer · 2026-10-27T135242Z · check · experts-rega-w2-epilogue-prefetch-bulk-store

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-27T133154Z-experts-rega-w2-epilogue-prefetch-bulk-store.md` (branch `hyp/muse_prefill_layer/experts-rega-w2-epilogue-prefetch-bulk-store`).

- **Job:** `665c32dd8c794c0b` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `1f1f74ed`

## Goal
Built on master (blob 42ded471, experts-rega-w13-fast-silu-epilogue landed). NO NEW nvcc BUILD: master's muse_prefill_rega_fsilu module already compiles rega_kernel<false, *> (w2) but never dispatches it (_experts_rega always launches w2 from state['rega_ext'], and the fsilu gate's pair graphs use master's w2 too). MECHANISM: rega's epilogue is serialized (tensor core idle), and the fast-silu run showed it costs 5-10x what instruction counts predict. In master's w2 epilogue (consume_item, the !G branch) (1) gsc = __ldg(gs + m.e) is issued only after the last wgmma_wait (the asm memory clobbers keep it there), and (2) the scatter loop 'for (v = ctid; v < nvalid * 16; v += 256)' issues, per 16-byte chunk, an __ldg of sorted_ids[slot0 + t] that is an L2 round trip on fresh slots, then the shared load and the global store: 8 iterations per thread at N=128. CHANGE, as further asserted text edits in _rega_fsilu_src() (each target must occur exactly once; every edit sits inside 'if constexpr (!G)' so the w13 instantiation the fsilu gate measures is textually unchanged): (a) at the top of consume_item, before the k loop, w2 only: load gsc and the thread's route id sid_r = (ctid < nvalid) ? __ldg(sorted_ids + m.slot0 + ctid) : -1 into registers; (b) after the stmatrix staging every consumer thread executes fence.proxy.async.shared::cta, then the existing named_bar(1, 256); then each thread with ctid < nvalid and 0 <= sid_r < num_routes issues ONE cp.async.bulk.global.shared::cta.bulk_group of 256 B from ep + ctid * PITCH2 (272 = 17 x 16, aligned) to out + sid_r * out_ld + m.j * 128, then cp.async.bulk.commit_group; (c) before the FIRST named_bar(1, 256) of the next epilogue (and once after the item loop) the issuing threads run cp.async.bulk.wait_group.read 0 (staging reuse), and cp.async.bulk.wait_group 0 before the consumer exits. Values, addresses and rounding are master's: y is bit-identical. Rename the module (e.g. muse_prefill_rega_fsilu_w2st) so the cache cannot serve the old build. _rega_fsilu_attrs: fsilu's ON criteria read only the two w13 rows; the w2 rows (regs, local 0, smem, occ >= 1) gate only the new w2 path. GATE per class (low 753 when rega_low_variant == 0, high 3012, ft 16384), after the fsilu gates, reusing their scaffolding (_rega_fsilu_tables, _rega_fsilu_bench, the calibrated judge-like pop routing and random): exactness first, y from master's w2 vs the module's w2 on the same hs (from fsilu's w13) torch.equal on random, skewed, one and pop, error word 0 and the w2 launch marker; then w2-alone one-launch graphs, master vs new, 2 rounds x 11 alternations, cold-L2 (256 MB read flush) AND warm-L2, both routings; ON only when new/master <= 0.99 in every round, mode and routing; then the two-state replay check with the path ON. state['rega_w2st_ok'][cls]; _experts_rega launches w2 from state['rega_fsilu_ext'] when that class is ON (variant 0 and fsilu ext present), else master's. Print per class the w2 us both ways and '[rega-w2st] PATH CHOSEN' lines after fsilu's (16 KB tail). Budget: no build; ~3-8 s of gating after fsilu (prepare ~216 -> ~224 s); a step deadline of 270 s leaves master's w2.

## What changed, and where
Implemented 2026-10-27T135059Z in the worktree's `muse_prefill_layer/submission.py` only (uncommitted; `git diff --stat`: 1 file, +518 / -25).

What changed:
- **CUDA, fsilu module only** (`REGA_W2ST_EDITS`, 15 asserted (old, new) edits that `_rega_fsilu_src()` applies after the silu edit; any target not occurring exactly once means no source, so no build). Master's `_REGA_HEAD` / `_REGA_BODY`, `muse_prefill_rega_v2`, `REGA_V2_SHA256` and the v1 fallback are untouched. I checked offline that the derivation applies by extracting the strings via `ast` and diffing master's source against the derived one (36347 -> 38841 chars).
  - `consume_item<NB, G, SB = false>` and `rega_kernel<G, BIG, SB = false>`. Every kernel-side change sits inside `if constexpr (!G ...)`. The w13 path changes text only: `float gsc; if constexpr (G) gsc = __ldg(gs + m.e);` is the same computation as before.
  - Both w2 arms (the early loads): before the k loop, `gsc_w2 = __ldg(gs + m.e)` and `sid_r = ctid < nbk*16 ? __ldg(sorted_ids + slot0 + ctid) : -1`.
  - Arm (a), SB=false, `rega_gemm(which=1)` of this module. After the first epilogue `named_bar` and before the staging barrier, rows ctid < nvalid write `sid_r` into a 128-int shared table. Master's 16-B st.global scatter loop then reads `sid` from that table instead of `__ldg(sorted_ids + slot0 + t)`. The skip predicate `sid >= 0 && sid < num_routes` is unchanged.
  - Arm (b), SB=true, `which=2`:
    - `fence.proxy.async.shared::cta` after the stmatrix staging, before the existing staging `named_bar`.
    - Then each thread with ctid < nvalid and `0 <= sid_r < num_routes` issues one 256-B `cp.async.bulk.global.shared::cta.bulk_group` from `ep + ctid*PITCH2` to `out + sid_r*out_ld + j*128`, then `commit_group`.
    - `cp.async.bulk.wait_group.read 0` runs before the next epilogue's first `named_bar`.
    - `cp.async.bulk.wait_group 0` runs after the consumer's item loop.
  - Host side:
    - `rega_init` reports 6 rows: master's 4, then `<w2, s1/ft, SB>` with flag +4.
    - `rega_gemm` accepts `which` 0..2, and `WG_LAUNCH(G_, B_, S_)` dispatches 2 to `rega_kernel<false, big, true>`.
  - The module is renamed `muse_prefill_rega_fsilu_w2st`.
- **Python**:
  - `_rega_fsilu_attrs` handles 6 rows. fsilu's ON criteria now read only the two w13 rows. The w2 rows (master's criteria plus local 0) come back as `{(which, big): ok}` and gate only w2st. A row count other than 6 turns everything OFF.
  - `_rega_fsilu_phase` / `_rega_fsilu_class` take an optional prepare-local `info` dict and record the build seconds, the w2 attrs and each class's calibrated alpha in it. No behaviour change for fsilu otherwise.
  - New section `_rega_w2st_*` sits before `kernel_fn`. `_rega_w2st_phase` runs in `prepare()` right after the fsilu phase, and its lines print after fsilu's summary, with PATH CHOSEN last.
  - `_experts_rega` launches w2 from `state['rega_fsilu_ext'].rega_gemm(state['rega_w2st_which'][cls], ...)` when `state['rega_w2st_ok'][cls]` is set (variant 0 and fsilu ext present). Otherwise it uses master's `rega_ext`. Only the shape class and prepare-time plain bools/ints are read.
- **Gate (per review)**: the control is folded into the same gate.
  - Class order is high, ft, then low (low is print-only, never routed, and skipped when `rega_low_variant != 0`).
  - The judge-like pop routing reuses fsilu's calibrated alpha for the class, or calibrates via `_rega_fsilu_calibrate` if fsilu never reached that step.
  - Exactness: on random, skewed, one and pop, hs comes from the w13 the class dispatches. Master's w2, arm (a) and arm (b) each write a y prefilled with the same sentinel (-3.0, so unwritten rows are compared too). Each launch must leave error word 0 and the w2 marker, and each arm's y must be `torch.equal` to master's. An arm that misses is dropped and its reason is printed.
  - Timing: one-launch w2-alone graphs `w2:m` / `w2:a` / `w2:b` plus an empty graph, all in the same `_rega_fsilu_bench`: 2 rounds x 11 alternations, cold-L2 (256 MB read flush) and warm-L2, pop and random. Each line prints the round-mean us for master and both arms, plus every per-round ratio.
  - Pass: arm/master <= 0.99 in every round, mode and routing (8 ratios).
  - The passing arm with the lower geomean is routed for high/ft after `_rega_replay_check` passes with it ON (plan `REGA_FSILU_REPLAY_PLAN[cls]`, keys `REGA_W2ST_STATE_KEYS`). Otherwise master's w2 runs.
  - State: `state['rega_w2st_ok'][cls]` (bool) and `state['rega_w2st_which'][cls]` (0/1/2).
  - If `rega_fsilu_ext` is None, the phase prints `[rega-w2st] OFF: rega_fsilu_ext is None ...`.
  - Every `[rega-w2st] PATH CHOSEN <cls>` line carries `fsilu=ON/OFF` for that class.
  - Step deadline: `REGA_W2ST_STEP_DEADLINE_S` = fsilu's 265 s. Summary lines are capped at 600 chars.

Deviations / judgement calls:
- **The fsilu module is rebuilt (said explicitly, per review item 5).** Its source changes, so the existing fsilu build now compiles 6 `rega_kernel` instantiations instead of 4. There is no additional nvcc module and no second build. Expect the fsilu build to grow from ~40 s to maybe ~55-60 s; this is an estimate, untested. The build seconds are printed in `[rega-fsilu] phase ... (build Xs ...)` and again in the first `[rega-w2st]` line. fsilu's 200 s start deadline and 265 s step deadline are unchanged. On ec3bd018 timings the build starts at ~168 s, so fsilu's ft step should still start well before 265 s. Projected prepare ~245-255 s, which leaves less headroom than the md's "~224 s".
- The arm-(a) "128-int __shared__ table" is the w2 staging tile's unused 16-B pad column (bytes 256..271 of each 272-B row; stmatrix writes only bytes 0..255) rather than a new `__shared__` array. This means no static-smem change for either w13 or w2, and no new kernel parameter. The cost is a one-time ~8-way bank conflict on the 128 int writes per item.
- Arm (b) uses a single `cp.async.bulk.wait_group 0` at consumer exit, which subsumes the md's extra `.read` wait after the item loop.
- `rega_w2st_ok` is a per-class bool as the md says. The routed arm goes in the separate per-class int `rega_w2st_which`.

Uncertainties: nothing could be compiled or run here (no GPU / nvcc / torch on this node).
- PTX syntax and ptxas acceptance of the `cp.async.bulk` S2G / `wait_group` / `fence.proxy.async` asm are unverified. A build failure turns fsilu OFF as well (one module), losing the landed fsilu gain for that run.
- The build-time growth is unmeasured.
- The log tail will still lose the earliest fsilu live lines; the summaries survive.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-w2-epilogue-prefetch-bulk-store diff`:

- [x] CUDA-graph capturable: `kernel_fn`'s path (`_experts_rega`) gains only a dict lookup on the shape class and a plain-int `which` read from prepare-time state, then one extension launch. There is no `.item()`, `.cpu()`, sync or allocation on values. All syncs, `.tolist()` and `torch.equal` are in prepare-only `_rega_w2st_*` and fsilu functions (lint: outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls. The new state keys are `rega_w2st_ok` (bools) and `rega_w2st_which` (ints), set in `prepare()` from synthetic gate cases. `fsilu_info` is prepare-local and deleted.
- [x] Weights: untouched. The same `bank["q2"]` / `bank["gs2"]` NVFP4 bank feeds master's and the new w2. No repack and no precision change.
- [x] Activations BF16 where SGLang keeps BF16: y is the same bf16 values (the gate asserts bit-identical to master's). KV cache and queries are untouched.
- [x] Attention sees every key: n/a, attention is untouched (experts w2 epilogue only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only build is the existing `load_inline` of the fsilu module (renamed). The printed text contains no tensor values.
- [x] All work happens inside the timed call; no output caching or replay. The w2 GEMM still runs every call; only its epilogue's load placement and store instruction change.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept. No `triton.autotune` added.
- [x] Lint and py_compile pass (output below); `__pycache__` was removed from the worktree.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`), rc 0. 107 host-sync warnings (master at 2750f9d6 has 102; the 5 new ones are `torch.cuda.synchronize()` in the prepare-only `_rega_w2st_*` functions) and the 5 dtype warnings master already has; none is in kernel_fn's call graph:

```
warn  host-sync    (107 lines: torch.cuda.synchronize() / .cpu() / .tolist() / .item(), all "outside kernel_fn's call graph; fine in prepare()")
warn  dtype        L3470, L3471, L3761, L4308, L6889: torch.empty(..., dtype=torch.uint8)  (allowed only where SGLang does the same; unchanged master lines)
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile muse_prefill_layer/submission.py`: ok (its `__pycache__` deleted).

## Risks / what could fail
low
Headline +0.3-1.0% if the w2 rows move as expected: likely below the 1.2% judge margin alone, so it is sized to stack with another epilogue gain (experts-rega-epilogue-wg-local-bulk-store or experts-rega-epilogue-overlap-next-stage); read the scored rega_kernel<false,*> rows against ec3bd018, not only the gate (LESSONS 2026-10-27: cold-L2 gates can show gains the warm judged graph does not). Risks: async-proxy ordering (missing fence.proxy.async gives stale rows; the torch.equal check catches it), and the fsilu attrs rule (today a bad w2 row would turn fsilu OFF; the split in (attrs) prevents that). If nvcc already unrolls the scatter loop and hoists its loads, expect the low end.

## Result (after the run)
- job `665c32dd8c794c0b` (check), status done, passed True, tests 5/5
- speedup 1.5010x, 95% CI [1.4785, 1.5235], score 258091.27588325107 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.488912488847924, "ci": [1.466578801515205, 1.5112461761806426], "job": "ec3bd01859574bf1"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-w13-fast-silu-epilogue.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-epilogue-prefetch-bulk-store.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-epilogue-prefetch-bulk-store.patch`; branch `hyp/muse_prefill_layer/experts-rega-w2-epilogue-prefetch-bulk-store`
