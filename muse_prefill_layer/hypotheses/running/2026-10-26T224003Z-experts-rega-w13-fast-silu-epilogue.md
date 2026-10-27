---
problem: muse_prefill_layer
slug: experts-rega-w13-fast-silu-epilogue
created: 2026-10-26T224003Z
base: master
area: silu-mul (rega w13 epilogue)
change: >
  Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32
  accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module
  is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a
  full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread
  evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an
  estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's
  source by one asserted edit (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns
  hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in
  pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is
  -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class
  (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings:
  finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference
  (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2
  graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99
  in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs
  difference, PATH CHOSEN.
expected:
  - shapes: "*"
    direction: faster
    why: "per the adversarial review: both silu forms issue the same two MUFU ops (EX2, RCP), so the fast form saves only FFMA/branch overhead, ~0.3-0.8% of w13 at ft and less at turn_1: first_turn ~-10 to -30 us, turn_4 ~-2 to -6 us, turn_1 ~-1 to -2 us, ~0.1-0.3% of a shape (below the judge margin). This run's purpose is to close or open the silu-epilogue line from the prepare-time w13-alone ratios (cold + warm L2, judge-like + random routings) plus the scored rega_kernel<true,*> rows and the w13/w2 ratio vs 73b40cd1 and d00770b6; a near-1.00 w13 ratio everywhere is the expected null and closes it"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-triton-w13-fused-silu-epilogue: that added silu to the old Triton engine's w13 epilogue (more epilogue work; lost 1-3% at first_turn); rega already fuses silu, and this only swaps the accurate expf and IEEE division for __expf and __frcp_rn"
review: proceed
status: running
review_at: 2026-10-27T112554Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-27T112554Z
base_sha: c984c14ea32ac08522ee284ee7e6b46826ad0312
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue
worktree: .worktrees/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-27T113829Z
impl_commit: 490497570e2c5e063189505738b81a068a0cd319
submission_sha256: 8e25e1bb376f2e439851c673ac5510cd2c6505978501246d3acc98e51b137a94
norm_sha256: f8abb0a041d8f65dc728fe11aedb4b127abe982ea5e792ac9bdb826e45e52e45
submission_blob: 42ded47196d5f0417c17f38574876508f0d822f1
lint_warnings: 107
attempt_file: 2026-10-27T113829Z-check-experts-rega-w13-fast-silu-epilogue.md
prepared_at: 2026-10-27T113829Z
job: 21911eb694ce467c
dispatched_at: 2026-10-27T113841Z
---

# experts-rega-w13-fast-silu-epilogue

## Change

Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's source by one asserted edit (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings: finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2 graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99 in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs difference, PATH CHOSEN.

## Expected

- `*` faster (resized 2026-10-27 to the adversarial review's sizes): both silu forms issue the same two MUFU ops (EX2, RCP), so the fast form saves only FFMA/branch overhead, ~0.3-0.8% of w13 at ft and less at turn_1: first_turn ~-10 to -30 us, turn_4 ~-2 to -6 us, turn_1 ~-1 to -2 us, ~0.1-0.3% of a shape (below the judge margin).
- Purpose of this run: close or open the silu-epilogue line. Read the prepare-time w13-alone fsilu/master ratios (`[rega-fsilu] ... TIME` lines: cold and warm L2, judge-like pop and random routings, per class), plus the scored `rega_kernel<true,*>` rows and the w13/w2 ratio against 73b40cd1 and d00770b6. A near-1.00 w13 ratio everywhere is the expected null and closes the line; a w13 ratio <= 0.99 in every mode opens it.

## Risk

Not bit-exact (rare 1-ulp bf16 differences in hs), bounded by the gate against master and the fp64 reference. Size ~+0.5-0.8% headline: below the judge margin alone, so it lands through experts-rega-rotate-finebucket-fastsilu-combo or experts-rega-rotation-fastsilu-combo.

## Adversarial review

Reviewed 2026-10-27T112554Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with sharp notes. The mechanism is plausible and rule-clean, and the predecessor's review asked for this run first and alone. Its size is overstated ~3-10x: per-item arithmetic gives at most ~1% of w13 (~0.1-0.3% of a shape), so a pair gate at <= 0.99 will very likely keep every class OFF. Gate on w13 alone, use a genuinely approximate divide, and read the result as 'does the silu epilogue cost anything at all'.

### Justification
silu_mul runs in the serialized consumer epilogue of rega_kernel<true,*> (NB/4 silus per thread per item, no fast math). An item is ~130K clk at ft (64 stages); the accurate silu costs ~1.2-1.8K issue slots per item (~1-1.4%), and both forms have 2 MUFU ops (EX2, RCP), so the fast form saves only FFMA/branch overhead: ~0.3-0.8% of w13 at ft, less at turn_1. Expected scored deltas: ft ~-10 to -30 us, turn_4 ~-2 to -6 us, turn_1 ~-1 to -2 us. __frcp_rn is not a fast reciprocal (rcp.rn.f32); use __fdividef(hg, 1.f + __expf(-hg)) * hu or rcp.approx.ftz. Closer unlisted prior: experts-rega-producer-scale-expand-all-t (6be2a8d3), a consumer-instruction diet that gated OFF everywhere; per-shape rows cannot read a <=1% effect. Landing route via the rotation combos and fine-N-high-plus-fast-silu is gone (held/dead).

### Confounders on the current kernel
- combo-b low class: muse_prefill_rega_lo has its own silu_mul copy; fsilu must be OFF at low unless rega_low_variant == 0, the variant printed, and the gate must time the kernel that would be dispatched.
- combo-a normgather: gate and replay-check on the same prenorm dispatch as the judge path.
- experts-turn-rega-s4-*: silu_mul lives in _REGA_HEAD, hashed by REGA_V2_SHA256; edit only a derived copy; assert the target occurs exactly once in _rega_src().
- align-two-phase-all-t-rega: part of the ~143-167 s of prepare before this phase.

### Interference and A/B needs
No companion. Make the single result mean something: (a) gate on w13 ALONE (one-launch graphs): w13 <= 0.99 in both rounds and the pair <= 1.00; (b) per LESSONS 2026-10-27, time each class on both cold-L2 and warm-L2 back-to-back replays and on a judge-like routing whose master w13 time brackets the judged row (turn_1 ~255-293 us, turn_4 ~707-763 us, ft ~3400-3850 us), choosing the pop alpha per class or requiring both pop and random; ON only if every condition passes; (c) read the scored rega_kernel<true,*> rows and the w13/w2 ratio vs 73b40cd1 and d00770b6; a near-1.00 gate everywhere is the expected null and closes the silu line.

### Open questions
- Use __fdividef or rcp.approx.ftz, not __frcp_rn.
- If the build stays under ~45 s, add a print-only ceiling variant (silu_mul returns hg*hu, never dispatched) to bound any silu speedup.
- Canary the fast vs accurate form on a synthetic hg sweep {-100, -88.7, -87, -20, 0, 20, 100}.
- Landing route if it gates ON: its own result or a re-mint.
- Reconcile FINDINGS' queued rotation+fastsilu combos (held).


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue`, base master c984c14e (blob f4751fc0). Only `muse_prefill_layer/submission.py` changed (+594/-1). The diff has three hunks.

- **prepare()**: `_rega_fsilu_phase` runs LAST, after `_rega_lo_phase` (it needs `rega_low_variant`) and after norm-gather's setup. Its compact lines print after `lo_lines`, so its `[rega-fsilu] PATH CHOSEN` lines are the last prepare lines. Each line is capped at 700 chars.
- **_experts_rega**: a bool `fsilu = not variant and state['rega_fsilu_ext'] is not None and state['rega_fsilu_ok'][_rega_class(T)]` picks `fsilu_ext.rega_gemm(0, ...)` for the **w13 launch only**. It uses the same arguments, grid and error word as master's call. It never applies on the `rega_gemm_lo` path. The gather / norm-gather, w2 and the weighted sum stay master's. The bool depends only on shape and prepare-time values.
- **Derived module** `muse_prefill_rega_fsilu`: `_rega_fsilu_src()` applies ONE asserted edit to master's `_rega_src()`, `'hg / (1.f + expf(-hg)) * hu'` -> `'__fdividef(hg, 1.f + __expf(-hg)) * hu'`. The target must occur exactly once. The result must contain the new text exactly once, no old text, and the expected length; otherwise there is no build and master stays. It is built through master's own `_rega_build(name, src)`, with master's `_REGA_CPP`, flags (`-O3 -std=c++20 -lineinfo`, sm_90a, no fast math) and function names. Before deriving, `_rega_lo_identity()` checks that master's v2 source still hashes to `REGA_V2_SHA256`. A python-only check here, on the real strings, confirmed: sha256 ok, target count 1, and the derived source differs by exactly that one line. `_REGA_HEAD`, master's module, `REGA_V2_SHA256` and the v1 fallback are untouched. A pre-build audit (templated accessors, `_rega_audit_cpp` globals) and the attribute / register-pool check of all 4 instantiations (master's criteria) both run before any launch.
- **Deviation from the md's `change` (per the review)**: the md's `__frcp_rn` (rcp.rn, not fast) is replaced by `__fdividef(hg, 1.f + __expf(-hg)) * hu`. That is div.approx (`x * rcp.approx(y)`) and ex2.approx, in fp32, with master's single bf16 rounding in pack_bf16. For hg <= ~-88.7, `1 + __expf(-hg)` is >= 2^126 or inf, so div.approx gives -0. The accurate form gives -0 at inf and a ~1e-37 value at -88.7. bf16 can hold that value, but it is irrelevant.
- **Low class**: gated only when `rega_low_variant == 0`. Otherwise it is OFF with the reason. The variant is printed on the low GATE and PATH CHOSEN lines. Every check and every timing table uses the prenorm dispatch the judge path uses: raw x into `_experts_rega(prenorm=norm_gather_ok[cls])`, and in the timing tables the fused `_norm_gather` (or module norm + `rega_gather` when it is off) into xs.
- **Per class** (low T=753, high T=3012, ft T=16384), in `_rega_fsilu_class`:
  1. A traced canary through `_experts_rega` with fsilu ON for that class.
  2. **Judge-like routing**: bisect the pop alpha grid 1.0..3.0 (master's `_rega_pop_counts`), starting at 1.83x the uniform routes, with at most 4 probes of master's **w13 alone (warm-L2)**. Stop inside the band (low 255-293, high 707-763, ft 3400-3850 us), else take the nearest. The chosen alpha, master w13 us, routes, in/OUT and all probes are printed.
  3. **Accuracy** on random, skewed and the chosen pop routing:
     - master's vs fsilu's w13 alone on the same xs: error word 0 and the w13 marker; fsilu hs finite and within rel L2 <= 2e-3 of master's (max |d| and ndiff printed);
     - the routed `_experts_rega` output, fsilu ON vs master: fsilu's error <= 1.05x master's against an **fp64** reference. `_rega_fsilu_reference_out` is master's `_rega_reference_y` with the matmuls and silu in float64, then master's weighted sum;
     - fsilu hs must differ from master's somewhere (summed ndiff > 0). Otherwise the fast form did not run (a guard against symbol interposition, since the module keeps master's function names).
  4. **Timing**, on the judge-like pop routing and on random: 2 rounds x 11 alternations of four graphs (`w13:m`, `w13:f` as one-launch graphs, and `pair:m`, `pair:f` = that w13 + master's w2), plus an empty graph subtracted. Each round runs both modes:
     - **cold-L2**: a 256 MB READ flush before each replay, as the judge does, so no dirty lines are left (fine-n-buckets used a `zero_()` flush);
     - **warm-L2**: one unmeasured replay, then 3 back-to-back replays timed.
  5. **GATE**: ON only if w13 f/m <= 0.99 AND pair f/m <= 1.00 in **every round of both L2 modes on both routings**, the hg sweep passed, and then master's two-state replay check (fsilu ON, plan per class as in fine-n-buckets) passed.
  - The printed TIME lines show master->fsilu us, the per-round ratios for w13 and the pair, and w2 ~= pair - w13 (for the w13/w2 read).
- **hg sweep canary** (once per phase): hg in {-100, -88.7, -87, -20, 0, 20, 100} x hu in {1, -2.5}, fast vs accurate form elementwise. It is a tiny prepare-only NVRTC kernel via `torch.cuda.jiterator._create_jit_fn` (the cheapest route: no second nvcc build, no edit to the module). Checks:
  - both forms finite;
  - an exact 0 must match in sign (-0 at hg = -100, and the hg = 0 cells);
  - where |acc| <= 1e-30 (hg <= -87) the fast result must be just as tiny;
  - elsewhere rel <= 1e-5.
  It prints the hu = 1 cells as `hg:fast/acc` and the fast form's worst error vs fp64. If jiterator cannot run, the result is None: timing still runs and prints, but no class may turn ON. The sweep tests the same intrinsics through NVRTC, not the nvcc-built module itself. The module's own fast path is checked by the hs compare, ndiff and the fp64 output bound.
- **Ceiling variant (review open question 6)**: SKIPPED. It would need either a second nvcc build (~40 s; master's build was 38.6 s, fine-n-buckets' 41.7 s, so above the ~45 s budget) or a second edit. A one-line note is printed instead.
- **Deadlines and logging**:
  - no build after 200 s of prepare (OFF line `SKIPPED`);
  - no class step (canary, calibration, each accuracy routing, each timing routing, replay) after 265 s;
  - logged: build seconds, per-class seconds, phase seconds and prepare total;
  - every miss or exception leaves master (`rega_fsilu_ext` = None when every class is OFF), and an exception in one class stops the later classes.
  - Prepare reached ~167 s before this phase in c951d916. Expect the build to finish at ~210 s and the phase at ~230-240 s.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-w13-fast-silu-epilogue diff`:

- [x] CUDA-graph capturable: the only kernel_fn-reachable change is a Python branch in `_experts_rega` on the token count and the prepare-time bools / module handle. The launch takes the same arguments as master's (`hs` allocated as before; no `.item()` / `.cpu()` / synchronize / tensor print). All syncs and `.cpu()` in the diff are in prepare-only `_rega_fsilu_*` functions (lint: outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls: state gains only `rega_fsilu_ext` (the module) and `rega_fsilu_ok` (plain bools per class). Every gate input is synthetic (`_rega_case` / `_rega_pop_case`), made in prepare and freed.
- [x] Weights: no new repack. fsilu reads master's lossless retiled `rega_bank` (NVFP4 codes + e4m3 scales unchanged). The fp64 dequant exists only in the prepare-time reference and is freed per class.
- [x] Activations: the w13 output hs stays bf16, with one rounding of the fp32 silu*up product exactly as master's. Only the fp32 silu's approximation changes (div.approx / ex2.approx, ~1-2 ulp in fp32, bounded vs master and the fp64 reference). KV cache and queries are untouched.
- [x] Attention sees every key: n/a, attention is untouched (the diff only touches the expert w13 launch).
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network. The nvcc build goes through master's existing `_rega_build` / `load_inline` path, as master's own modules do. The hg sweep uses jiterator (in-process NVRTC) in prepare only. Master's `_REGA_HEAD` / `_REGA_BODY` / sha256 constants are only read.
- [x] All work happens inside the timed call; no output caching or replay: kernel_fn still computes the whole layer each call. The prepare-time graphs are freed after timing.
- [x] Prepare budget: build start deadline 200 s, class-step deadline 265 s. The phase never raises, and any miss keeps master's w13. Its PATH CHOSEN lines are the last prepare lines, each <= 700 chars.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept. No triton.autotune.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`): exit 0, `OK with warnings: muse_prefill_layer/submission.py`. There are 107 warnings (102 host-sync, 5 dtype), all host-sync in prepare()-only code or the pre-existing dtype warnings. The new lines' warnings are all of the form `warn host-sync L<n>: torch.cuda.synchronize() / .cpu().tolist() (outside kernel_fn's call graph; fine in prepare())`, at L5892-5894 (sweep), L5985, L6000, L6003, L6031, L6040, L6046, L6103, L6187, L6221, L6339. No errors. `python3 -I -m py_compile` ok; no `__pycache__` left in the worktree.
