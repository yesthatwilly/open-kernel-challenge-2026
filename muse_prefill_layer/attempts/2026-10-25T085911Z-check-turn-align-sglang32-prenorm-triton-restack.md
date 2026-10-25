# muse_prefill_layer · 2026-10-25T085911Z · check · turn-align-sglang32-prenorm-triton-restack

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T054848Z-turn-align-sglang32-prenorm-triton-restack.md` (branch `hyp/muse_prefill_layer/turn-align-sglang32-prenorm-triton-restack`).

- **Job:** `9cfd4f58d24044b2` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `fb2bd510`

## Goal
Built on master (a81d19c). Two row-level wins already measured in the archive, applied together, shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: master's _align_parallel_kernel scales with T because every program streams all n route ids: 16.7 us at T=753, 68.3 us at T=3012 (stack-tonight-five profile) and 394 us at T=16384 (align-local-parallel-first-turn profile), while SGLang's moe_align_block_size over the 32 local experts (master's first_turn call) costs count_and_sort 87 + align 51 = 139 us at T=16384, and the pre-stack master's 256-expert version cost ~17+~10 us at T=3012. In _experts_marlin replace _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm) with moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS) when T > 1024 (and at T <= 1024 only if prepare's timing says so), keeping local-routes-only blocks (ids are 0-31 / -1). Its sorted_ids/block_experts/n_padded layout is the one _silu_mul_sorted and Marlin already accept (the silu setup self-checks against moe_align_block_size output); extend _align_par_setup's 15-case check to compare the Marlin pair's output on the new align vs master's on the same routing (bit-exact expected) and fall back to _align_parallel on any mismatch. Prepare timing in CUDA graphs with a cold-L2 flush at T=753 and T=3012 picks per T class, ON only at <= 0.9x of _align_parallel. (2) PRE-EXPERT NORM: take moe-pre-post-norm-triton-multitoken's Triton _pre_expert_rmsnorm_kernel (its patch in done/neutral) for the pre-expert norm only (its post-expert combine lost and stays master's), with its existing per-T prepare A/B (measured 13.0 -> 9.7 us at T=753, 24.8 -> 20.1 us at T=3012, ~77 -> 69 us at T=16384; vs fp32 ref error identical to master's). No CUDA extension; graph-safe; nothing cached from inputs; no prints in kernel_fn.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
low
Expected headline +0.3-0.5%, inside the CI: read it on the _align_* and pre-norm kernel rows, not the headline. Cheap insurance: the rewrites' Marlin fallback at turn_4 inherits the faster align.

## Result (after the run)
- job `9cfd4f58d24044b2` (check), status done, passed True, tests 5/5
- speedup 1.0694x, 95% CI [1.0533, 1.0854], score 362274.43392283 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/turn-align-sglang32-prenorm-triton-restack.result.json`, `muse_prefill_layer/hypotheses/done/neutral/turn-align-sglang32-prenorm-triton-restack.patch`; branch `hyp/muse_prefill_layer/turn-align-sglang32-prenorm-triton-restack`
