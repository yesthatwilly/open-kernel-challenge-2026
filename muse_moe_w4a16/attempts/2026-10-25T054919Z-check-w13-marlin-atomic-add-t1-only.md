# muse_moe_w4a16 · 2026-10-25T054919Z · check · w13-marlin-atomic-add-t1-only

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T054545Z-w13-marlin-atomic-add-t1-only.md` (branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only`).

- **Job:** `1048c6edf3e54892` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `f29a0b9`

## Goal
Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f, 1.3558x check, on master 9080a71) with its gate narrowed to num_tokens == 1. In _forward, the w13 moe_wna16_marlin_gemm call gets common with use_atomic_add=True only when num_tokens == 1 (decode_1 and the bs=1 test); every other T uses master's w13 kwargs byte for byte; w2 keeps master's atomic add at T<=32. Use a new constant W13_ATOMIC_MAX_TOKENS = 1 (do not reuse W2_ATOMIC_MAX_TOKENS: LESSONS, give each path its own bound). Keep the earlier run's prepare self-check of the w13 atomic mode at T=1 against master (rel-L2 vs fp32 dequant) and its fallback to master's kwargs. The Marlin kernel name is unchanged, so the bs=1 Marlin row is the attribution. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## What changed, and where
Base: claim 3-way applied the earlier w13 atomic patch (staged) on ea4c938; not stale. Narrowed it in muse_moe_w4a16/submission.py (worktree, not committed): W13_ATOMIC_MAX_TOKENS 32 -> 1 (own constant, W2 bound untouched); W13_CHECK_TOKENS (1,4,8,16,32) -> (1,), so prepare self-checks only T=1 against current master (w2 mode as chosen), atomic then atomic_zero then master fallback; the slice-estimate print now covers T=1 only. _forward gate unchanged in form (`num_tokens <= W13_ATOMIC_MAX_TOKENS`), so every T>=2 uses master's w13 kwargs. Diff vs HEAD: +142/-3.

Review points applied:
- W13_ATOMIC_TOL = 1e-2 kept; `PATH CHOSEN` print kept (per band, plus the all-master VOID line).
- Old print-only _marlin_exec_config_probe dropped entirely (function, constants, call site, and the now-unused `inspect`/`time` imports).
- Poisoned-output test kept at T=1 (cache13.fill_ with 3e4 before the w13 call; guards the w2-atomic alias into cache3/cache1).
- kernel_fn has no try/except/print; it only looks up a prepare-time mode by shape band. try/except exist only in prepare().
- Open questions (profile 34.2 vs 42.1 us harness equivalence) cannot be answered by the build; read only the bs=1 row and its Marlin top_kernels entry. Do not cite the 8 us.
- CUDA/sm90a brief bullets n/a (pure Python change). No process spawn. Two-state rule: modes dict lives in state per prepare; no module globals hold tensors.

## Rules check (each must be true before the run)
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

## Risks / what could fail
low
Identical code to the 1.3558 run would dedupe, so this is the only way to restack it; the expected bs=1 gain (~-1 to -2 us, z~1-2) is near resolution, and the bs=8 recovery is the second half of the bet. If it reads flat, close the w13 atomic knob.

## Result (after the run)
- job `1048c6edf3e54892` (check), status done, passed True, tests 5/5
- speedup 1.3461x, 95% CI [1.3259, 1.3663], score 9990.472608713128 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-t1-only.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-atomic-add-t1-only.patch`; branch `hyp/muse_moe_w4a16/w13-marlin-atomic-add-t1-only`
