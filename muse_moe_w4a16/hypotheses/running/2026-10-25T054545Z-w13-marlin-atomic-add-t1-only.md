---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-t1-only
created: 2026-10-25T054545Z
base: master
area: w13 marlin
change: >
  Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f,
  1.3558x check, on master 9080a71) with its gate narrowed to num_tokens == 1. In _forward, the
  w13 moe_wna16_marlin_gemm call gets common with use_atomic_add=True only when num_tokens == 1
  (decode_1 and the bs=1 test); every other T uses master's w13 kwargs byte for byte; w2 keeps
  master's atomic add at T<=32. Use a new constant W13_ATOMIC_MAX_TOKENS = 1 (do not reuse
  W2_ATOMIC_MAX_TOKENS: LESSONS, give each path its own bound). Keep the earlier run's prepare
  self-check of the w13 atomic mode at T=1 against master (rel-L2 vs fp32 dequant) and its
  fallback to master's kwargs. The Marlin kernel name is unchanged, so the bs=1 Marlin row is the
  attribution. Hard rules: CUDA-graph capturable; kernel_fn does no host sync
  (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except
  and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in
  registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes.
  The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may
  hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "the earlier run's bs=1 row: 44.02 vs 44.88 us adjusted, raw p10/p50 44.9/46.0 vs 46.1/47-48 in both master runs; profile Marlin pair 34.2 vs 42.1 us"
  - shapes: bs=8 *
    direction: flat
    why: "back to master: w13 atomic at bs=8 had raw p50 117 vs 109-110 us in master runs, so T=2..32 is dropped"
  - shapes: bs=16 *
    direction: flat
    why: master path
  - shapes: bs=32 *
    direction: flat
    why: master path (earlier run +1.2%)
risk: low
depends_on: w13-marlin-atomic-add-on-master-small-m
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T054817Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T054817Z
base_sha: ea4c938779678da09fceb3d72b468aef6f3272d9
base_blob: 60dfe35d5417d9c0f89067ff313efdc2c41d03a2
branch: hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only
worktree: .worktrees/muse_moe_w4a16/w13-marlin-atomic-add-t1-only
mode: check
restack_of: w13-marlin-atomic-add-on-master-small-m
stale_base: null
implemented_at: 2026-10-25T054919Z
impl_commit: f29a0b9e34299e122d6c05ad93492885c2c9484d
submission_sha256: 99f131fd0b4c055483fe2da22f8581d675ec5db7d54b59f186dde3c730c07785
norm_sha256: b42c84d36edc8fc938435770ac7fbaefe197a66865d46a41316df5d174a407c1
submission_blob: 34e5b52476fd39c5296e0de9a53a3c5797cd52e1
lint_warnings: 19
attempt_file: 2026-10-25T054919Z-check-w13-marlin-atomic-add-t1-only.md
prepared_at: 2026-10-25T054919Z
---

# w13-marlin-atomic-add-t1-only

## Change

Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f, 1.3558x check, on master 9080a71) with its gate narrowed to num_tokens == 1. In _forward, the w13 moe_wna16_marlin_gemm call gets common with use_atomic_add=True only when num_tokens == 1 (decode_1 and the bs=1 test); every other T uses master's w13 kwargs byte for byte; w2 keeps master's atomic add at T<=32. Use a new constant W13_ATOMIC_MAX_TOKENS = 1 (do not reuse W2_ATOMIC_MAX_TOKENS: LESSONS, give each path its own bound). Keep the earlier run's prepare self-check of the w13 atomic mode at T=1 against master (rel-L2 vs fp32 dequant) and its fallback to master's kwargs. The Marlin kernel name is unchanged, so the bs=1 Marlin row is the attribution. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=1 *` faster: the earlier run's bs=1 row: 44.02 vs 44.88 us adjusted, raw p10/p50 44.9/46.0 vs 46.1/47-48 in both master runs; profile Marlin pair 34.2 vs 42.1 us
- `bs=8 *` flat: back to master: w13 atomic at bs=8 had raw p50 117 vs 109-110 us in master runs, so T=2..32 is dropped
- `bs=16 *` flat: master path
- `bs=32 *` flat: master path (earlier run +1.2%)

## Risk

Identical code to the 1.3558 run would dedupe, so this is the only way to restack it; the expected bs=1 gain (~-1 to -2 us, z~1-2) is near resolution, and the bs=8 recovery is the second half of the bet. If it reads flat, close the w13 atomic knob.

## Adversarial review

Reviewed 2026-10-25T054817Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as one run with no companions. The change is a shape-gated kwarg flip on the w13 Marlin call at T=1, so the bs=1 row is the only readout. The expected gain is about 0.9 us of 44 us (z about 1.5 in the prior run), roughly 0.3% on the headline, so treat it as the run that closes the w13-atomic knob rather than as a likely promotion.

### Justification
Mechanism: at T=1 with block_size_m=8, w13 has about 4 m-blocks x 64 n-tiles. The prior run's slice estimate gave about 2 split-K slices per CTA. Atomic add replaces the fp32 global-reduce pass between those slices, which plausibly saves a microsecond or two on a launch-bound kernel (Marlin pair 42.1 us against a 11.4 us floor). At T>=8 the estimate falls to 1 slice, so atomic buys nothing there, which fits narrowing the gate to T=1. The prior evidence (job 171684977ec1470f, w13 atomic at T<=32 on master 9080a71) is bs=1 44.02 vs 44.88 us with err_ns 573. That is about 1.5 sigma, inside the +/-1.5% the judge calls noise. Weakest link: the profiled Marlin pair read 34.2 vs 42.1 us (-7.9 us), but end-to-end moved only 0.86 us. Either the profile comes from a different job and harness (BEST's 42.1 is from 7a0d900a, not a paired run) or most of the profiled saving is hidden in launch/graph overhead. Do not cite the 8 us as the expected effect. The bs=8 'recovery' half of the bet cannot be measured: the prior bs=8 row (107.8 vs 106.0) was already flat within noise, so going back to master at T=2..32 can only read flat. The headline CI (+/-1.5%) cannot resolve a ~0.3% headline change. Read only the bs=1 row and its Marlin top_kernels entry against the paired master runs. Expected directions match TASK's equal load weights (60 each, geomean over loads).

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (landed, same knob on the second Marlin call, same T=1 path): w13's output feeds silu_and_mul and then w2-atomic. The self-check gate is measured against w2-atomic master (T=1 rel-L2 0.00788 vs tol 1e-2). If w2 were not atomic, the w13 timing would not change (separate launch), but the correctness margin would. Timing confounding is negligible, numeric coupling only.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: these fix block_size_m=8 and the sorted_ids layout at T=1, which set m_blocks and therefore the split-K slice count that atomic add acts on. They are shape-fixed and unchanged by this hypothesis, so they set the operating point but do not confound the A/B.
- align-parallel-per-expert-verify, post-expert-norm-triton-small-m, sglang-marlin-path-parity: these share the launch sequence at T=1 (align, then post-norm combine) but touch neither the w13 kernel nor its memory traffic. No confounding.

### Interference and A/B needs
No companion. An ab-revert of w2-marlin-atomic-add-small-m would change the bs=1 baseline by about the same 1-us magnitude as the effect, adding noise rather than resolving it. A control has nothing to isolate, because the change is a single kwarg flip. The real ambiguity is statistical: a 2% effect at a 1.3% per-shape error. A second identical run is deduped, and a companion would not help. With moe at 5 checks left and 5 queued, the check is better spent elsewhere. If bs=1 reads flat or slower, close the w13 atomic knob as the hypothesis says. If it reads faster by >=1 us again, that is two independent same-direction readings and enough to submit-test.

### Open questions
- Was the Marlin pair 34.2 us (hyp) vs 42.1 us (master) measured under the same profiler and graph mode? If master's paired profile also shows about 34 us, the profile-level 'gain' is a harness artifact and the prior is weaker than stated.
- Does Marlin's atomic path zero C internally? The prior self-check passed 'atomic' (no memset) with poisoned output buffers at T=1/4/8, which suggests yes. The w2 atomic write into cache3 aliases cache1, so the builder should keep the poison test to catch a regression there.


## Implementation notes

Base: claim 3-way applied the earlier w13 atomic patch (staged) on ea4c938; not stale. Narrowed it in muse_moe_w4a16/submission.py (worktree, not committed): W13_ATOMIC_MAX_TOKENS 32 -> 1 (own constant, W2 bound untouched); W13_CHECK_TOKENS (1,4,8,16,32) -> (1,), so prepare self-checks only T=1 against current master (w2 mode as chosen), atomic then atomic_zero then master fallback; the slice-estimate print now covers T=1 only. _forward gate unchanged in form (`num_tokens <= W13_ATOMIC_MAX_TOKENS`), so every T>=2 uses master's w13 kwargs. Diff vs HEAD: +142/-3.

Review points applied:
- W13_ATOMIC_TOL = 1e-2 kept; `PATH CHOSEN` print kept (per band, plus the all-master VOID line).
- Old print-only _marlin_exec_config_probe dropped entirely (function, constants, call site, and the now-unused `inspect`/`time` imports).
- Poisoned-output test kept at T=1 (cache13.fill_ with 3e4 before the w13 call; guards the w2-atomic alias into cache3/cache1).
- kernel_fn has no try/except/print; it only looks up a prepare-time mode by shape band. try/except exist only in prepare().
- Open questions (profile 34.2 vs 42.1 us harness equivalence) cannot be answered by the build; read only the bs=1 row and its Marlin top_kernels entry. Do not cite the 8 us.
- CUDA/sm90a brief bullets n/a (pure Python change). No process spawn. Two-state rule: modes dict lives in state per prepare; no module globals hold tensors.

## Rules check

- [x] first-line marker `# okc: problem=muse_moe_w4a16` unchanged (diff touches only the named areas)
- [x] no triton.autotune added
- [x] no host sync / .item / allocation on input values in kernel_fn (checked diff: only a shape lookup)
- [x] no try/except or print in kernel_fn; no per-call mutable state
- [x] CUDA-graph capturable: w13 change is a kwarg flip (plus cache1.zero_() only in atomic_zero mode, a graph-safe memset)
- [x] nothing cached from inputs; state["w13_modes"] derived from synthetic data in prepare
- [x] no child process/subprocess; NVFP4 weights unchanged; BF16 activations unchanged
- [x] no module/static global holds device pointers or buffers
- [x] self-check never raises (wrapped); falls back to master
- [x] n/a: sm90a/mbarrier/TMA items, no CUDA in this change

okc_lint: "OK with warnings" (only host-sync warnings in prepare(), outside kernel_fn's call graph). py_compile OK; __pycache__ removed.
