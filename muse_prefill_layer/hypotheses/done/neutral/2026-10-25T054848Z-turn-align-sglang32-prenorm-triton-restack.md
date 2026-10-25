---
problem: muse_prefill_layer
slug: turn-align-sglang32-prenorm-triton-restack
created: 2026-10-25T054848Z
base: master
area: "MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins"
change: >
  Built on master (a81d19c). Two row-level wins already measured in the archive, applied together,
  shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: master's _align_parallel_kernel scales
  with T because every program streams all n route ids: 16.7 us at T=753, 68.3 us at T=3012
  (stack-tonight-five profile) and 394 us at T=16384 (align-local-parallel-first-turn profile),
  while SGLang's moe_align_block_size over the 32 local experts (master's first_turn call) costs
  count_and_sort 87 + align 51 = 139 us at T=16384, and the pre-stack master's 256-expert version
  cost ~17+~10 us at T=3012. In _experts_marlin replace _align_parallel(local_ids, 0,
  LOCAL_EXPERTS, bsm) with moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS) when T > 1024 (and
  at T <= 1024 only if prepare's timing says so), keeping local-routes-only blocks (ids are 0-31 /
  -1). Its sorted_ids/block_experts/n_padded layout is the one _silu_mul_sorted and Marlin already
  accept (the silu setup self-checks against moe_align_block_size output); extend
  _align_par_setup's 15-case check to compare the Marlin pair's output on the new align vs
  master's on the same routing (bit-exact expected) and fall back to _align_parallel on any
  mismatch. Prepare timing in CUDA graphs with a cold-L2 flush at T=753 and T=3012 picks per T
  class, ON only at <= 0.9x of _align_parallel. (2) PRE-EXPERT NORM: take
  moe-pre-post-norm-triton-multitoken's Triton _pre_expert_rmsnorm_kernel (its patch in
  done/neutral) for the pre-expert norm only (its post-expert combine lost and stays master's),
  with its existing per-T prepare A/B (measured 13.0 -> 9.7 us at T=753, 24.8 -> 20.1 us at
  T=3012, ~77 -> 69 us at T=16384; vs fp32 ref error identical to master's). No CUDA extension;
  graph-safe; nothing cached from inputs; no prints in kernel_fn.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "align 68 -> ~25 us and pre-norm 24.8 -> 20.1 us: about -45 us per turn_4 shape (1.4% sliding, 0.9% full)"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "pre-norm -3 us, align -5 to -8 us if its gate picks SGLang at T=753: ~1% of turn_1 sliding"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: pre-norm -8 us of 14-22 ms; align unchanged there
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T085209Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T085209Z
base_sha: e6cf5ce8f79493b9904b7c6d74e6a6c25f14f820
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/turn-align-sglang32-prenorm-triton-restack
worktree: .worktrees/muse_prefill_layer/turn-align-sglang32-prenorm-triton-restack
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T085911Z
impl_commit: fb2bd51078e513f04f4de0f9b066ee834cc6e0c9
submission_sha256: 7c16998b2c1a958d783f2bb1bdbc6e90cc9931d22b61989a27a6b9c4b530159b
norm_sha256: 33b2313415f5a9bfdfed0b6118ab1539bda1a5900b29c07c2b4e5120e1293589
submission_blob: 8507f9e93acf5b0c3acb3b0dd45789b556d1344a
lint_warnings: 31
attempt_file: 2026-10-25T085911Z-check-turn-align-sglang32-prenorm-triton-restack.md
prepared_at: 2026-10-25T085911Z
job: 9cfd4f58d24044b2
dispatched_at: 2026-10-25T085923Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.069365828013475
speedup_ci: [1.053325340593273, 1.085406315433677]
harvested_at: 2026-10-25T091110Z
best_at_harvest: 1.074386115190733
promotion: null
---

# turn-align-sglang32-prenorm-triton-restack

## Change

Built on master (a81d19c). Two row-level wins already measured in the archive, applied together, shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: master's _align_parallel_kernel scales with T because every program streams all n route ids: 16.7 us at T=753, 68.3 us at T=3012 (stack-tonight-five profile) and 394 us at T=16384 (align-local-parallel-first-turn profile), while SGLang's moe_align_block_size over the 32 local experts (master's first_turn call) costs count_and_sort 87 + align 51 = 139 us at T=16384, and the pre-stack master's 256-expert version cost ~17+~10 us at T=3012. In _experts_marlin replace _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm) with moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS) when T > 1024 (and at T <= 1024 only if prepare's timing says so), keeping local-routes-only blocks (ids are 0-31 / -1). Its sorted_ids/block_experts/n_padded layout is the one _silu_mul_sorted and Marlin already accept (the silu setup self-checks against moe_align_block_size output); extend _align_par_setup's 15-case check to compare the Marlin pair's output on the new align vs master's on the same routing (bit-exact expected) and fall back to _align_parallel on any mismatch. Prepare timing in CUDA graphs with a cold-L2 flush at T=753 and T=3012 picks per T class, ON only at <= 0.9x of _align_parallel. (2) PRE-EXPERT NORM: take moe-pre-post-norm-triton-multitoken's Triton _pre_expert_rmsnorm_kernel (its patch in done/neutral) for the pre-expert norm only (its post-expert combine lost and stays master's), with its existing per-T prepare A/B (measured 13.0 -> 9.7 us at T=753, 24.8 -> 20.1 us at T=3012, ~77 -> 69 us at T=16384; vs fp32 ref error identical to master's). No CUDA extension; graph-safe; nothing cached from inputs; no prints in kernel_fn.

## Expected

- `*load=turn_4*` faster: align 68 -> ~25 us and pre-norm 24.8 -> 20.1 us: about -45 us per turn_4 shape (1.4% sliding, 0.9% full)
- `*load=turn_1*` faster: pre-norm -3 us, align -5 to -8 us if its gate picks SGLang at T=753: ~1% of turn_1 sliding
- `*load=first_turn*` unchanged: pre-norm -8 us of 14-22 ms; align unchanged there

## Risk

Expected headline +0.3-0.5%, inside the CI: read it on the _align_* and pre-norm kernel rows, not the headline. Cheap insurance: the rewrites' Marlin fallback at turn_4 inherits the faster align.

## Adversarial review

Reviewed 2026-10-25T085209Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, row-readable restack. The align swap partly undoes the landed parallel align at turn_4, and the pre-norm half reuses an already-checked kernel. One real gap to fix before building: SGLang's 32-expert align does NOT give local-only blocks (remote routes go into -1 blocks), so prepare must time the align together with the Marlin pair and the sorted silu, not the align alone.

### Justification
The mechanism holds up. _align_parallel_kernel runs 33 CTAs on 132 SMs, and each one streams all n = 8T route ids twice in 512-id chunks, so its cost grows linearly with T: 16.7 us at T=753, 68.3 us at T=3012, 394 us at T=16384. SGLang's moe_align_block_size over 32 experts costs 139 us at T=16384, which scales linearly to about 25 us at T=3012, matching the pre-port master's ~27 us. That makes about -40 us at turn_4 plausible. The pre-norm numbers (24.8 -> 20.1 us at T=3012, 13.0 -> 9.7 at T=753) come from moe-pre-post-norm-triton-multitoken's own prepare A/B, which passed correctness, and the -45 us turn_4 estimate is consistent (~1.4% of 3319 us sliding, ~0.9% of 4849 us full). The headline effect, about +0.3-0.5%, sits well inside the ~±1.5% CI. The Marlin rows move ±70 us between identical-code runs (turn_4 sliding: 2093 / 2118 / 2148 / 2223 us across four recent runs), so the result can only be read on the align and pre-norm kernel rows and the prepare logs, as the hypothesis says. WEAKEST LINK: the change says 'keeping local-routes-only blocks', but moe_align_block_size(local_ids, bsm, 32) puts the ~7/8 remote routes into -1 blocks (moe-align-32-experts-first-turn's md says so explicitly). At T=3012 with bsm 64 that is roughly 330 extra -1 m-blocks next to ~80 local ones. Marlin (is_ep) and _silu_mul_sorted then walk a list about 5x longer up to n_padded. The archive suggests Marlin does not care: turn_4 Marlin was 2093/2173 us before the local-only port and 2148/2115 us after, which is within noise. But a prepare gate that times only the align could pick an align that is faster on its own and slower end to end. Gate on the whole _experts_marlin call (align + w13 Marlin + silu + w2 Marlin) in the graph with a cold-L2 flush at T=753 and T=3012, and switch it ON only when the end-to-end time drops by more than the timing noise. Also, 'bit-exact' needs care: Marlin output per route is order-independent with use_atomic_add=False, so bit-exact is a fair expectation. Compare only local rows, since remote cache3 rows are unwritten or garbage in both versions.

### Confounders on the current kernel
- stack-tonight-five-prefill (landed moe-align-local-routes-parallel-port inside it): this hypothesis partly reverts that sub-change at T > 1024, so any turn_4 align gain is that port's own T-scaling being undone. It is still attributable, because the align kernel rows (_align_parallel_kernel vs moe_align/count_and_sort) appear in the profile and prepare logs both timings.
- stack-tonight-five-prefill (landed silu-mul-triton-local-rows-marlin-turn inside it): _silu_mul_sorted consumes the align's sorted_ids/n_padded. With SGLang's layout it walks the -1 blocks too (masked, so correctness is unaffected but there is extra work), so the silu row may grow slightly at turn_4 and offset some of the align saving. Time it inside the end-to-end gate.
- experts-hybrid-marlin-turn-triton-first-turn: it fixes MARLIN_MAX_TOKENS = 4096, which decides that turn_4 (T=3012) is on the Marlin path where this align change applies. If that threshold changes, for example because a queued turn_4 rewrite lands, the align half becomes dead code at turn_4. No confound for this run.
- sglang-pipeline-parity: the pre-norm replaces state['pre_expert_norm'] (MuseSpark1xRMSNorm) from the parity pipeline. The Triton kernel uses the module's _get_effective_weight(), so it does not depend on how parity built the module. No confound.

### Interference and A/B needs
No companions. Each of the two sub-changes affects disjoint kernels with their own profile rows (align vs the pre-expert RMSNorm), and each logs an in-prepare A/B per T class (CUDA graph, cold L2). The two halves are already separable from a single run without a control. Reverting stack-tonight-five's align port (ab-revert) would only reproduce what this hypothesis does at T > 1024. An ab-tune of the gate threshold is not worth a check while prefill has 8 checks left. Make the result unambiguous by changing the prepare gate, not by adding a companion: gate the align on end-to-end _experts_marlin time and print both the align-only and the end-to-end numbers.

### Open questions
- Does sgl_kernel's moe_align_block_size at Marlin's small block sizes (bsm 16-64) put the -1 bucket last? If so, a one-thread fix-up of n_padded to the local total, at about 2 us, would restore local-only blocks for Marlin and silu and remove the weakest link entirely.
- Is the ~25 us SGLang 32-expert align at T=3012 measured or extrapolated? The cited ~17+10 us figure was for the 256-expert call. The prepare log line will answer this, so record it in the attempt file.
- At T=753 _align_parallel is 16.7 us, and SGLang's two-kernel align is probably ~8-12 us there, close to the 0.9x gate. Expect turn_1 align to stay on _align_parallel, which makes turn_1's expected gain pre-norm only (~3 us).


## Implementation notes

Base: worktree base e6cf5ce8 differs from master 771e9743 only by a claim commit; `git diff e6cf5ce8 master -- muse_prefill_layer/submission.py` is empty, so no merge needed. Only `muse_prefill_layer/submission.py` was edited (+323/-3).

What changed:
- `_experts_marlin(..., align_mode=None)`: align chosen by a static per-T-class dict `state["align_mode"]` (classes 753 / 3012 / 16384 by shape only, `_gate_class`, `_align_mode`). Modes: `par` (master's `_align_parallel`), `sgl32` (`moe_align_block_size(local_ids, bsm, 32)`), `sgl32fix` (the same plus `_align_fix_npad`, a one-program Triton kernel that sets n_padded to BSM x the number of local blocks among the first n_padded/BSM blocks), `n256` (master's old fallback when `align_par_ok` is False).
- `_align_gate_setup` (prepare, after `_silu_sorted_setup`): probes that SGLang's -1 bucket is last on synthetic routings (random, skewed, all-remote at T=753 and 3012) and that the fixed n_padded gives the same per-expert route sets as `_align_parallel`; enables `sgl32fix` only if so. For each class it checks the candidates' `_experts_marlin` output against the baseline mode's (random and skewed routing, rel L2 <= 1e-3, bitwise equality printed), then times the END-TO-END `_experts_marlin` (align + w13 + silu + w2 + combine) in a CUDA graph with a 288 MB cold-L2 flush, 31 interleaved reps in rotating order. A candidate is switched on only if gain > max(half-IQR of both, 1 us) and ratio <= 0.99; the largest gain wins. Align-only times (par / sgl32 / sgl32fix) are printed as well. `PATH CHOSEN align T=...` per class. Any exception keeps the baseline.
- Pre-norm: persistent Triton `_pre_expert_rmsnorm_kernel` (grid 4 x SMs); stores only the bf16 checkpoint weight `state["pre_w"]` and adds 1 in fp32 inside the kernel (no widened fp32 copy). `_pre_norm_setup` times nw 4/8 against the module's call per class (cold-L2 graph), requires err vs an fp32 reference <= 1.25 x master's + 1e-4 and rel L2 vs master <= 5e-3, gate ratio <= 0.98, else the module call. `PATH CHOSEN pre-norm T=...`.
- kernel_fn hot path has no prints, `.cpu()` or `.item()`: only dict lookups on shapes.

Review items and what was done:
- Gate on end-to-end `_experts_marlin`, print both align-only and end-to-end times, ON only beyond noise: done (above).
- Open question, -1 bucket last: probed at runtime; if true, one-program n_padded fix-up (`sgl32fix`) restores local-only blocks for Marlin and the sorted silu; both variants are timed end to end.
- Correctness compares local-derived rows only: the compared tensor is the combine output, which skips remote slots; remote cache3 rows are never read.
- Pre-norm weight not stored widened: bf16 w, +1 in-kernel in fp32.
- Gate keyed by T class only, static per class: yes.
- Spec deviation (review wins): the spec's 15-case Marlin-pair check and fixed 0.9x align-only gate are replaced by the end-to-end gate; the spec's use of the module's `_get_effective_weight()` fp32 copy is replaced by bf16 w plus in-kernel +1, so the spacing-vs-master check of the old patch is dropped in favour of err vs an fp32 reference.
- Open question on the 25 us SGLang figure: the prepare log prints the measured numbers; record them in the attempt file.

## Rules check

- [x] No CUDA extension, Python/Triton only; sm90a bullets n/a (no wgmma/load_inline)
- [x] No child process/subprocess/multiprocessing (diff checked)
- [x] No host syncs, .cpu(), prints or allocation on input values in kernel_fn path; syncs/prints only in prepare()
- [x] CUDA-graph capturable: new kernels use shape-derived grids, no data-dependent control flow
- [x] No device pointers, tensors or buffers in module globals; everything lives in `state` (pre_w, align_mode, norm_grid)
- [x] Weights: pre_w is a lossless bf16 copy of the BF16 weight; no precision change
- [x] Nothing cached from inputs; synthetic data only in prepare()
- [x] Any exception or failed check lands on master's path (align_base / module norm), never raises; loud PATH CHOSEN lines
- [x] First line marker `# okc: problem=muse_prefill_layer` kept
- [x] triton.autotune not used
- Note (not a rules item): not run on this node (no GPU/torch); compile risk untested

okc_lint: `OK with warnings` (only host-sync warnings in prepare() and two dtype warnings on the uint8 flush buffer in prepare()). py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus, by reading the code; no GPU). **Verdict: dispatch-with-fixes.** Rules OK.

Fix:
- `_align_fix_npad_kernel` now has `do_not_specialize=["n_blocks"]`. n_blocks = cdiv(T*8 + 33*(bsm-1), bsm) changes with T, and Triton specializes integer arguments on %16 and ==1. Without the fix, a T whose n_blocks falls in a different specialization from the prepare probes (127 and 409) would JIT-compile on its first kernel_fn call. This only matters if sgl32fix is chosen. After the edit, lint is OK with warnings (prepare-only syncs and the uint8 flush buffer), py_compile is OK, and __pycache__ is removed.

Checked and fine:
- Master equivalence. `align_base` is "par" when `align_par_ok` holds, else "n256", the same mapping as master. T > 4096 has no align_mode entry, so it uses base. `_check_experts_e2e` at T=8192 therefore still runs master's align. Note: the classes are ranges that cover every T. The tests at T=300/600/753 run the 753 decision and T=1506/4096 run the 3012 decision, so no T is "outside" a class. Both new paths are correct at any T (bsm is 64 for every T >= 200).
- Static choice. Both choices are dict lookups on `_gate_class(x.shape[0])`, decided in prepare on seeded synthetic data. kernel_fn gains no sync, print, .cpu/.item, try/except or value-dependent branch.
- Pre-norm weight. Only `pre_w` is stored: a bf16 [4096] clone. The +1 is added in fp32 inside the kernel. The fp32 `w1` exists only inside prepare.
- Align check. It compares the combine output against base: rel L2 <= 1e-3 and finite. That output reads only local slots, and the remote y rows are unwritten in both modes. Marlin's per-row output does not depend on how routes are grouped into blocks (use_atomic_add=False), so the outputs should be bit-exact and the gate can pass.
- Graph capture in prepare. An eager warm-up and a side-stream warm-up run before capture, so every Triton kernel compiles outside capture. A capture or replay exception lands in the per-class `except`, and state is written only after success. The graph and its private pool are dropped on unwind, and `empty_cache` runs per class. Only a sticky CUDA fault could poison the context, and the eager warm-up would hit that first.
- Fallbacks. A probe error disables only sgl32fix. A per-class error keeps base, and an outer error resets every class. Pre-norm errors, per class or outer, fall back to the module call. Every path prints a `PATH CHOSEN` line.
- Per-state. All new buffers (`pre_w`) and decisions live in `state`. Module globals hold only constants and kernels.
- Prepare budget. The added work is about 3 small Triton compiles, ~20 graph captures and a few hundred replays at the ms scale: an estimated 15-40 s on top of master. The SSDQ attempt's 170 s budget guard suggests master is well under that.
- Expectation. sgl_kernel's moe_align_block_size likely shifts ids by +1, which would put the -1 bucket first, not last. Then the probe disables sgl32fix, and only plain sgl32 is timed. It walks the remote -1 blocks, so the extra Marlin and silu work may eat the align saving. The end-to-end gate handles that by staying on par.

Residual risk: the align half may stay OFF at both classes (sgl32fix disabled by the probe, plain sgl32 not beating par end to end), which would leave only the pre-norm row gain. Compile and runtime are untested on a GPU.

## Result

- job `9cfd4f58d24044b2` (check), status done, passed True, tests 5/5
- speedup 1.0694x, 95% CI [1.0533, 1.0854], score 362274.43392283 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/turn-align-sglang32-prenorm-triton-restack.result.json`, `muse_prefill_layer/hypotheses/done/neutral/turn-align-sglang32-prenorm-triton-restack.patch`; branch `hyp/muse_prefill_layer/turn-align-sglang32-prenorm-triton-restack`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1101.29 | 1046.14 | 1.0052 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1529.55 | 1406.91 | 1.0036 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3318.71 | 3362.75 | 1.0009 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4849.27 | 4920.09 | 0.9993 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14705.32 | 14917.48 | 0.9953 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14264.64 | 14039.6 | 0.9906 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17746.69 | 18558.1 | 1.0014 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21733.75 | 21667.63 | 0.989 | flat |

no shape moved outside noise. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: turn-align-sglang32-prenorm-triton-restack`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

