---
problem: muse_prefill_layer
slug: experts-rega-pdl-chain-gather-w13-w2
created: 2026-10-25T211833Z
base: master
area: "launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)"
change: >
  DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no
  programmatic dependent launch (griddepcontrol appears 0 times in submission.py). Port
  muse_moe_w4a16's helpers (submission.py ~line 1029: pdl_wait() = griddepcontrol.wait,
  pdl_launch() = griddepcontrol.launch_dependents; host side cudaLaunchKernelEx with
  cudaLaunchAttributeProgrammaticStreamSerialization = 1, ~line 1524) into the rega module.
  rega_gemm gains an int pdl argument; when 1 it launches with the attribute (graph capture keeps
  the programmatic edge, as moe's PDL chain does on the judge). In rega_kernel: everything that
  reads only align outputs (sorted_ids, block_experts, n_padded, chunk tables in smem: produced
  two or more grids back, complete before the previous grid launched) and weights may run BEFORE
  pdl_wait: barrier init, tensormap prefetch, the table decode, and the producer's first STAGES
  weight bulk copies (bulk_g2s of wsrc for the first items; weight addresses come from decode_item
  over align tables). pdl_wait() is called by the producer before its FIRST activation tma_2d and
  by every consumer warp before its first epilogue store, so no read of xs/hs and no write of out
  happens before the previous grid's memory is visible. pdl_launch() is issued by each CTA's
  producer after it has issued its last item's last stage. Apply pdl=1 to w13 (primary:
  _norm_gather_kernel or ext.rega_gather, launched normally; implicit trigger at its exit) and to
  w2 (primary: w13, explicit trigger). Do not touch fused_post_expert_norm_weighted_sum (SGLang
  Triton, launched normally after w2). All three rega classes. GATE per class in _rega_setup:
  exact-compare pdl=1 vs pdl=0 (bit-identical expected), two-state replay check, cold-L2
  interleaved graph timing of the whole _experts_rega call; ON at <= 0.99x (a small, low-risk
  win); state['rega_pdl'][cls] plain int. Print per class both medians and PATH CHOSEN.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "Persistent rega grid with a +-1-item tail at ~16-19 items per CTA, plus two launch gaps and two prologues (barrier init, table decode, first weight stages) on 756.7 / 1150.3 us shapes: ~10-25 us per shape, ~2-3% of the turn_1 load."
  - shapes: "*load=turn_4*"
    direction: faster
    why: same ~10-25 us on 2197.8 / 3744.8 us shapes, < 1%.
  - shapes: "*load=first_turn*"
    direction: flat
    why: "~15 us on 11-18.6 ms shapes, ~0.1%. Headline ~+0.8-1.0%: below the 1.24% margin, so this must stack (with mc2 or the low-class full slab)."
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
prior: none
review: proceed
status: running
review_at: 2026-10-25T212524Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212551Z
base_sha: 30f67a9af087bff842ec0bd3a8a3ea252c962f52
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2
worktree: .worktrees/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T011532Z
impl_commit: 6dc70581856a124be8b2b3db7bd12bd061be2cb9
submission_sha256: bdd042627638da883bc5506b818839cc05b278586a6d354861818193b48c7b1d
norm_sha256: d71d3354792137b1b4de1fcf682140da76da6fcb33ef84286a766afbfed133b0
submission_blob: 1834ae36e06f81546b302addcb09c9a912b7e381
lint_warnings: 98
---

# experts-rega-pdl-chain-gather-w13-w2

## Change

DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no programmatic dependent launch (griddepcontrol appears 0 times in submission.py). Port muse_moe_w4a16's helpers (submission.py ~line 1029: pdl_wait() = griddepcontrol.wait, pdl_launch() = griddepcontrol.launch_dependents; host side cudaLaunchKernelEx with cudaLaunchAttributeProgrammaticStreamSerialization = 1, ~line 1524) into the rega module. rega_gemm gains an int pdl argument; when 1 it launches with the attribute (graph capture keeps the programmatic edge, as moe's PDL chain does on the judge). In rega_kernel: everything that reads only align outputs (sorted_ids, block_experts, n_padded, chunk tables in smem: produced two or more grids back, complete before the previous grid launched) and weights may run BEFORE pdl_wait: barrier init, tensormap prefetch, the table decode, and the producer's first STAGES weight bulk copies (bulk_g2s of wsrc for the first items; weight addresses come from decode_item over align tables). pdl_wait() is called by the producer before its FIRST activation tma_2d and by every consumer warp before its first epilogue store, so no read of xs/hs and no write of out happens before the previous grid's memory is visible. pdl_launch() is issued by each CTA's producer after it has issued its last item's last stage. Apply pdl=1 to w13 (primary: _norm_gather_kernel or ext.rega_gather, launched normally; implicit trigger at its exit) and to w2 (primary: w13, explicit trigger). Do not touch fused_post_expert_norm_weighted_sum (SGLang Triton, launched normally after w2). All three rega classes. GATE per class in _rega_setup: exact-compare pdl=1 vs pdl=0 (bit-identical expected), two-state replay check, cold-L2 interleaved graph timing of the whole _experts_rega call; ON at <= 0.99x (a small, low-risk win); state['rega_pdl'][cls] plain int. Print per class both medians and PATH CHOSEN.

## Expected

- `*load=turn_1*` faster: Persistent rega grid with a +-1-item tail at ~16-19 items per CTA, plus two launch gaps and two prologues (barrier init, table decode, first weight stages) on 756.7 / 1150.3 us shapes: ~10-25 us per shape, ~2-3% of the turn_1 load.
- `*load=turn_4*` faster: same ~10-25 us on 2197.8 / 3744.8 us shapes, < 1%.
- `*load=first_turn*` flat: ~15 us on 11-18.6 ms shapes, ~0.1%. Headline ~+0.8-1.0%: below the 1.24% margin, so this must stack (with mc2 or the low-class full slab).

## Risk

Prior art outside prefill: muse_moe_w4a16 combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 ran a PDL w13->w2 chain on the judge and front-end/epilogue wins did not interfere (LESSONS 183); attention combo-b1-chain-c320-fused-pre-pdl showed launch-gap wins compound with kernel wins. A read of xs/hs or a write of out before griddepcontrol.wait is a silent race that random routings may not expose: the exact-compare must run under graph replay with pdl=1. First measurement is an okc check.

## Adversarial review

Reviewed 2026-10-25T212524Z by an adversarial subagent (model opus).

**Verdict: proceed.** PDL on norm_gather -> rega w13 -> rega w2 is a sound, bit-identical, low-risk change with no prior in prefill. The stated mechanism overclaims, though: griddepcontrol.wait cannot hide the ±1-item tail. Only launch latency and the prologue overlap (barrier init, table decode, first weight stages), which realistically saves about 4-12 µs per shape, mostly at turn_1. Read the result from the per-class gate medians, not the headline or the per-kernel profile rows.

### Justification
Mechanism: plausible but smaller than claimed. rega_kernel is __launch_bounds__(THREADS,1) with near-full smem, so one CTA fits per SM. Early-launched w2 CTAs can only become resident on SMs where a w13 CTA has already exited, and they block at pdl_wait before any activation TMA or MMA. So w13's ±1-item tail is not recovered. What overlaps is (a) the graph kernel-to-kernel launch gap, about 1-3 µs per boundary, and (b) the prologue (mbar init, s_first/chunk table decode over nb blocks, tensormap prefetch, the first STAGES weight bulk copies, which are cold-L2 HBM latency because the judge flushes L2). That is about 2-6 µs per boundary times two boundaries, about 4-12 µs per shape. At turn_1 that is 0.5-1.5% of 756.7/1150.3 µs (the hypothesis says 10-25 µs, 2-3%). At turn_4 it is under 0.5%, and first_turn is flat. Expected directions are consistent with TASK weights; only the magnitude is optimistic. Correctness argument checks out against the code. sorted_ids, block_experts and n_padded come from align, which precedes the normally-launched gather, so they are complete before w13 can start. Weights and gs are static. The err magic and atomic writes before the wait are harmless. Every read of xs/hs and write of out/y sits behind pdl_wait, which also covers allocator-aliasing WAR hazards inside the captured graph. No deadlock: the rega grid is persistent and all CTAs are co-resident, so the explicit trigger fires only after every w13 CTA has issued or exited. Prior: 'none' is correct for prefill (grep finds no PDL in done/ or running/; DIAGNOSIS says griddepcontrol appears 0 times). The cited moe precedent (LESSONS 188, combo-toprow on PDL master) and attention combo-b1 (LESSONS 191) are real. It is not on any Do-not-try list. WEAKEST LINK: the gate. ON at <= 0.99x of the whole _experts_rega call (align + gather + w13 + w2 + post-norm) needs about 4.5-5 µs saved at turn_1, about 11 µs at turn_4 and about 55 µs at first_turn. turn_4 and first_turn will almost certainly gate OFF, so 'all three classes' is moot. turn_1 sits right at the threshold, which is exactly the single-round marginal-selector misroute the new LESSONS line warns about. Because the output is bit-identical and the change cannot plausibly cost much, a better gate is 'not slower' (ratio <= 1.00-1.01 over two interleaved cold-L2 rounds with an IQR bound), or an absolute-µs criterion. That keeps the small win on every class instead of a coin flip at 0.99.

### Confounders on the current kernel
- combo-a-master-normgather-naive-stack (landed): the w13 primary is the Triton _norm_gather_kernel with a persistent grid of SMs x resident x NG_MULTS (1 or 2), chosen per class. With an implicit trigger at its exit, w13 overlap with the gather's tail depends on that grid choice. In classes where the gather is OFF, the primary is ext.rega_gather (one CTA per slot, a very different tail). Per-class savings at the gather->w13 boundary differ by which gather won, so this is not a confound of the sign, only of the magnitude.
- w2-nonlocal-zero-store-off-first-turn (in stack-tonight-five-prefill, landed): when state['zero_nonlocal_w2'] is True, _experts_rega issues torch.zeros(y), a memset/fill node, between w13 and w2. w2's programmatic edge would then be against that node, not w13 (an invalid or ignored primary under capture). On the live rega path it is False (the ft class refuses otherwise), but the build must force pdl=0 for w2 whenever zero_nonlocal_w2 is True.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128 (landed engine): the prologue PDL hides is this kernel's table decode. It is cheap in the small class (s_be in smem, about 50-100 blocks at T=753) and heavier in the BIG class (block_experts read through __ldg over thousands of blocks), so the overlap is per class. That is a property of the engine, not an interaction that could flip the sign.
- combo-b-normgather-two-cta-low-regate (best's job; low variant gated OFF): if the low-class V64/V32 variant were ON, w13/w2 would run through state['rega_lo_ext'].rega_gemm_lo, which this change does not touch. Today variant=0, so the change does apply at turn_1. If a later restack turns V64 ON, PDL silently drops out of the low class.

### Interference and A/B needs
No companion is warranted. The change is bit-identical and self-gated. The prepare gate already times pdl=1 against pdl=0 in-process, interleaved under cold-L2 graph replay, and prints both medians per class, so that log line is the A/B. No landed hypothesis needs reverting to read it: the only real coupling (the zero_nonlocal_w2 memset node) is a correctness guard to build in, not something to measure. A control separating 'launch attribute only' from 'pre-wait prologue + weight prefetch' would split 4-12 µs into two pieces under okc noise and could not be read. Two reading rules make the single run interpretable. (1) Do NOT read the per-kernel profile rows: with PDL, rega_kernel<false,*> (and <true,*>, behind the gather) start earlier and spin in griddepcontrol.wait, so their reported durations GROW even when the shape gets faster. Read whole-shape µs and the gate medians. (2) The headline gain (about 0.3-0.8%) is below the 1.24% margin by the hypothesis's own admission, so score the run on the turn_1 gate ratio and the turn_1 shape rows, and carry it as a stack component (with mc2 / full-slab low) rather than judging it as a standalone submit.

### Open questions
- Gate threshold: should the ON criterion be 'not slower' (<= 1.00-1.01, two interleaved rounds, IQR bound), given the output is bit-identical and the expected saving (about 1% at turn_1, under 0.5% elsewhere) is at or below the proposed 0.99x? As written, turn_4 and ft will almost surely gate OFF, and turn_1 is a marginal single-threshold selector (the 2026-10-25 LESSONS misroute pattern).
- Producer restructure: issuing the first STAGES weight bulk_g2s before pdl_wait and that stage's activation tma_2d after it needs the expect_tx (WST + slab bytes) armed once per stage, before the weight copy, with the activation TMA completing the same transaction. Confirm the builder does not double-arm or under-count the full barrier.
- Does the Triton _norm_gather_kernel launch interact with the programmatic attribute on w13 under torch graph capture (the primary is a Triton launch, not cudaLaunchKernelEx)? The moe precedent chained two CUDA-extension kernels, not Triton -> CUDA. Check that capture records a programmatic edge and not a full edge (a full edge would make the gather->w13 half a silent no-op).
- Stacking with the queued mc2 / cluster line (experts-rega-mc2-*, rega_mc_kernel): the PDL attribute and the pdl_wait/pdl_launch placement have to be re-ported into the cluster kernel's cudaLaunchKernelEx (the cluster-dim attribute plus the programmatic attribute). Plan that port so a later mc2 win does not silently drop the PDL saving.


## Implementation notes

All changes are in `muse_prefill_layer/submission.py` of the worktree (+496 lines, no deletions).

- **Separate module, master untouched.** `_REGA_HEAD`, `_REGA_BODY`, `_REGA_CPP`, the v2/v1 sha256 checks and the low-class module are unchanged. The PDL CUDA source is derived at prepare time from master's strings by exact-once text edits (`_REGA_PDL_EDITS`, `_rega_pdl_src`; any anchor miss gives None and no PDL build). It builds as `muse_prefill_rega_pdl` with distinct symbols: `rega_kernel_pdl<G,BIG>`, `rega_gemm_pdl(..., int64 pdl)`, `rega_init_pdl`, `rega_gather_pdl[_kernel]`. Offline check: master's hash still equals REGA_V2_SHA256, the derived source passes `_rega_audit_cpp`.
- **Kernel.** `pdl_wait()` / `pdl_launch()` (griddepcontrol, sm_90). With `pdl=1` the producer first issues, for the first STAGES stages, `expect_tx` (armed once, whole-stage bytes) plus the weight `bulk_g2s`, then `pdl_wait()`, then the main loop (stages n < STAGES skip their expect_tx and weight copy, issue only the two activation TMAs, so the barrier is neither double-armed nor under-counted). Every consumer thread calls `pdl_wait()` right after `setmaxnreg.inc`, before any smem read of activations and before any store to `out`. The producer calls `pdl_launch()` after its last item's last stage. Barrier init, table decode and the align-table reads stay before the wait (align precedes the normally launched gather). `pdl=0` skips every PDL instruction.
- **Host.** `rega_gemm_pdl` launches via `cudaLaunchKernelEx`; `ProgrammaticStreamSerialization=1` only when `pdl=1`, otherwise a plain launch. A launch error raises inside prepare only (caught by the phase).
- **Phase.** `_rega_pdl_phase` runs after `_rega_lo_phase` (it needs the low variant int). Its own deadlines: start 195 s ('skipped: deadline' printed), per class 255/255/235 s, replay check before REGA_COPY_DEADLINE_S. It builds the module (prints arch flags, build time), prints regs/local bytes per instantiation (`_rega_pdl_attrs`), and per routed class (`rega_ok`) runs `_rega_pdl_class`: traced eager canary; graphs of the whole `_experts_rega` call for master's ACTUALLY ROUTED path (flag off, incl. the norm-gather path and the low variant) vs PDL over the same static inputs; cold-L2 (flush before each replay) bit-for-bit compare over several routings x 3 PDL replays plus error word / launch markers; two interleaved cold-L2 rounds x 11 alternations with empty-graph baseline; the two-state (A/B/C) replay check with the flag on (`REGA_PDL_STATE_KEYS`). Both gate lines and every `PATH CHOSEN` line are printed again as the last lines of prepare().
- **Flags.** `state['rega_pdl'][cls]` plain ints, `state['rega_pdl_ext']` the module (None when no class is ON; keys are read with `.get`, so master's mini states and the older phases are unaffected). `_experts_rega` uses `_rega_pdl_flags(state, T, variant)`.

### Review fixes applied
1. Gate is 'not slower' (REGA_PDL_RATIO = 1.005, inside the user's 1.00-1.01 band) in BOTH interleaved rounds with candidate IQR <= 1.5x master's, instead of 0.99 (user instruction, matches the review's open question 1).
2. w2 is forced to pdl=0 whenever `state['zero_nonlocal_w2']` is True (memset node between w13 and w2); w13 may still chain to the gather.
3. Both launches are pdl=0 on the `rega_gemm_lo` path (variant != 0), and the low class is not PDL-gated when the variant is not master's.
4. Open question 2 (expect_tx): armed once per stage before the weight copy, activation TMAs complete the same transaction; stages n >= STAGES unchanged.
5. Open question 3 (Triton primary): not assumable from the moe precedent; the exact compare and the timing run on CUDA-graph captures of the real prenorm path, so a missing programmatic edge shows up as a flat gate (no correctness effect: pdl_wait is a no-op when there is no programmatic edge). Cannot be verified without a GPU here.
6. Open question 4 (mc2 stacking): the PDL kernel/launch edits are isolated as `_REGA_PDL_EDITS` text edits over master's strings; a later cluster kernel needs the same three insertions (prefetch pass, consumer wait, trigger) plus the programmatic attribute next to its cluster-dim attribute.

### Deviations
- No `prefetch.tensormap` (master's rega kernel has none; the maps are __grid_constant__).
- The consumers' wait is at the start of the consumer branch instead of just before the first epilogue store: they cannot do useful work before the first full barrier anyway, and it is the simpler, stricter placement.
- The mc2 coupling, review reading rules (do not read per-kernel rows: rega_kernel_pdl durations grow with PDL) are for the judge-side reading, nothing to build.
- Nothing could be compiled or run here (no nvcc/GPU/torch). Review the new CUDA as untested.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-pdl-chain-gather-w13-w2 diff` (one file, +496 / -0).

- [x] CUDA-graph capturable: kernel_fn's added code is `_rega_pdl_flags` (shape + prepare-time ints) and one `rega_gemm_pdl` launch via `cudaLaunchKernelEx` on the current stream; no `.item()`/`.cpu()`/`synchronize()`/tensor prints, no allocation or control flow on input values. All syncs and prints are in the prepare-only phase. Every wait is bounded (existing `mbar_wait`; `griddepcontrol.wait` returns when the prerequisite grid completes).
- [x] Nothing derived from `data` is cached across calls: the new state is the module (`rega_pdl_ext`) and per-class ints picked in prepare from synthetic cases; tensor maps are still encoded per call in C++.
- [x] Weights: untouched (same retiled bank, NVFP4 stays NVFP4).
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries FP8: no numeric change (output bit-identical by construction and compared bit for bit in prepare).
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the new build is `load_inline` in-process like master's; no subprocess/fork.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Standing build rules: two interleaved rounds (REGA_PDL_ROUNDS), master's actually routed path timed flag-off, medians/IQRs/kernel names/PATH CHOSEN printed and repeated as the last lines of prepare; per-state flags (`rega_pdl`), a miss falls back to master's exact path and touches no other flag; build time and regs/spills printed; new symbol names; own start deadlines with 'skipped: deadline'; no module-global device state.

okc_lint: `OK with warnings` (0 errors; only the usual host-sync/dtype warnings outside kernel_fn's graph). `python3 -I -m py_compile` ok, __pycache__ deleted.

## Pre-flight audit

Verdict: dispatch-with-fixes. Nothing could be compiled or run (no GPU/nvcc); audit by reading plus emulation.

- Anchors: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/pdl_edit_check.py imports the worktree submission with stubbed torch/triton and confirms all 11 _REGA_PDL_EDITS anchors occur exactly once in master's strings, _rega_pdl_src() returns source (saved to /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/pdl_derived.cu), _rega_pdl_audit is clean, and master's v2 sha256 is unchanged. A miss returns None -> OFF line + PATH CHOSEN, build never attempted; any raise is caught by the phase and falls to master's path.
- Pre-wait reads (concern 1): only wbank weights (bulk_g2s) and align outputs (n_padded, block_experts, sorted_ids; align precedes the normally launched gather, so complete). Every read of xs/hs (producer TMA) is after the producer's pdl_wait; every consumer thread waits before any smem/activation use or store to out. WAR via allocator aliasing is covered because writes to hs/y happen after the wait. expect_tx: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/pdl_stage_emul.py emulates prefetch + main loop over grid/total/KS/block grids: every stage armed once and weight-copied once (0 violations); tx bytes equal the full stage (WST + two slabs) in both places.
- Triton primary (concern 2): without launch_dependents the implicit trigger fires at block exit, and griddepcontrol.wait waits for full grid completion + memory flush, so semantics stay correct (just less overlap). Capture of ProgrammaticStreamSerialization on a kernel following a kernel node yields a programmatic edge (CUDA >= 12.3); if not honoured it is a full edge and pdl_wait is a no-op (flat gate). Unverifiable without GPU; gate decides.
- Forced pdl=0 (concern 3): _rega_pdl_flags returns (0,0) when variant != 0 (rega_gemm_lo path) and w2=0 when zero_nonlocal_w2; the w2 then goes through master's ext.rega_gemm. Correct.
- Deadlock: trigger needs all w13 CTAs to have issued/exited, so no early w2 CTA can starve an unscheduled w13 CTA; 1 CTA/SM by registers.
- Deadlines: master prepare total at end of rega-lo is 125-165 s in recent judge logs (archive/combo-b-normgather-two-cta-low-regate log: 139.9 s); one nvcc build 39 s. PDL starts <= 195 s, class deadlines 255/255/235; typical finish ~210-250 s, worst ~285 s < 300.

Fix: the candidate-IQR veto (pdl IQR <= 1.5 x master IQR) had no absolute floor; added REGA_PDL_IQR_FLOOR_US = 1.0 on master's IQR so a timer-resolution spread cannot veto a real win. okc_lint OK (0 errors), py_compile ok, __pycache__ removed.

Residual risk: untested CUDA (never compiled); the PDL win (~4-12 us per shape) is near the 1.005 gate and may only turn ON at turn_1, and late-prepare classes (ft) may be skipped by deadline.
