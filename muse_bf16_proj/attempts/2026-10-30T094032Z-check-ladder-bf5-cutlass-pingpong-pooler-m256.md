# muse_bf16_proj · 2026-10-30T094032Z · check · ladder-bf5-cutlass-pingpong-pooler-m256
Hypothesis: `muse_bf16_proj/hypotheses/queue/2026-10-30T084229Z-ladder-bf5-cutlass-pingpong-pooler-m256.md` (branch

- **Job:** `894a19e27e4c41fc` (`okc status <job>`)
`ladder/bf5`, worktree `.worktrees/ladder-bf5`, from master 94958afa).
- **File:** `muse_bf16_proj/submission.py` at git `551257ed` (358,451 B, sha256 d714c3b3cab614a0...)

## Goal
Campaign 2026-10-30-public-ladder rung BF5 (one mechanism; user-requested): does a CUTLASS sm90 TMA warp-specialized
pingpong persistent GEMM (swap-AB 128x64x64) move the pooler verify_32 row (m=256, 30.0 us) by >= 1/3 of the
pre-registered -2.8 us? The in-prepare race on the H200 decides the cell.

## What changed, and where (`git diff 94958afa 551257ed`)
- Near the top (after the imports): section (14)'s build. The CUTLASS source composes CollectiveBuilder mainloop
  (`KernelTmaWarpSpecializedPingpong`) + epilogue (`TmaWarpSpecialized`) + `PersistentScheduler`, BF16 in/out, FP32
  accumulate, from the image's FlashInfer-bundled CUTLASS headers. It is compiled by `load_inline` in a background
  thread started at import (explicit `-gencode sm_90a`; no env changes).
- New section (14) after (13): `_bf5_route_ok` (cfg/shape/state only, exactly m=256), `_bf5_forward` (torch.empty +
  one CUTLASS launch), `_bf5_prepare` (join build <= 200 s into the state build; workspace query; `_cell_check`; 3
  read-flush `_cell_gate` rounds with BF1's rule; profiler check for exactly one CUTLASS kernel), `_bf5_summary`.
  Flag `bf5_pooler256_ok` joins NEW_CELL_FLAGS.
- kernel_fn: `_bf5_route_ok` is checked first in the non-router dense block (before khalf/NN). `_ab_component` maps
  pooler m=256 to the bf5 flag when it is on. `_prepare_state` calls `_bf5_prepare` for both states; prepare() prints
  `[bf5] SUMMARY` last.

## Node validation (H100 gpunode06 GPU 5; takemyjob hopper/ladder-bf5 418ee82 probes/bf5)
Run `runs/20261030-022445-gpunode06` (committed bytes); replicates: queue job `50-h100-ladder-bf5-val-replicate`.
- Build 62 s in the background; import 224.1 s (master 219.9 s); prepare 13.8 s.
- check_implementation: 26 shapes x 3 inputs, all pass, error word 0. Outputs at m=256 are bit-identical to the NN
  cuBLAS route.
- A/B bf5/cur at pooler m=256: 0.948 (seed 3000), 0.946 (seed 9001). Identity: cur2 0.989/0.990, bf5B 0.938/0.935.
  The race routed (pooled 0.957). m=200 is untouched (1.000).

## Rules check (against the diff)
- [x] Lossless: BF16 weights and activations used as they are (no repack, no cast), FP32 accumulate, one BF16
  rounding; outputs bit-identical to today's route.
- [x] Libraries: CUTLASS headers from the image (rules l.88 allows building on CUTLASS / FlashInfer); `load_inline`
  only; no network or installs.
- [x] CUDA-graph capturable: the routed call is torch.empty + one CUTLASS launch on the current stream (workspace 0;
  initialize() is host-side param setup). Capture + replay were validated on the H100. Prepare-only syncs.
- [x] Nothing derived from `data` is cached; arguments are rebuilt per call. `_BF5` holds build state; `_BF5_SUM`
  holds prepare text.
- [x] All work is inside the timed call; no output caching or replay. No timing patches or harness edits.
- [x] The background thread only compiles (load_inline) at import and is joined in prepare(), never during a timed call.
- [x] First line `# okc: problem=muse_bf16_proj` kept; py_compile OK; size 358,451 B < 1,000,000; prepare 13.8 s.
- [x] No third-party file copied; design cited as github.com/alygg77/okc-muse-spark-kernels@6cae49a
  kernels/muse_bf16_proj.py; no Nathen Fernandes (MIT) code.
- [ ] okc_lint: run before dispatch (main thread).

## Risks
low-medium.
- The judge image has a different CPU count: the background nvcc competes with the main-thread builds, so import could
  grow by more than the H100's +4-6 s. The 200 s join deadline keeps prepare bounded; a late build leaves the cell off.
- One row only (-1.6 us H100-scaled): the board effect (~+0.3-0.5%) is inside one check's noise margin. Read the row.
- If flashinfer's bundled CUTLASS is missing on the judge, the build fails and the cell stays off (no loss).

## Result (after the run)
- job `<job>` ...

## Result
- job `894a19e27e4c41fc` (check), passed, **1.320x [1.300, 1.340]** vs board 1.310: judge race m256 cur/bf5 0.905-0.909 (public 0.907; H100 0.945-0.96), ROUTE; v32 pooler 27.9 us (was 30.0); build 55.8 s background.
