# muse_bf16_proj · 2026-10-25T204107Z · submit · unpooler-k-halves-batched-gemm-m65-m128

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T153141Z-unpooler-k-halves-batched-gemm-m65-m128.md` (branch `hyp/muse_bf16_proj/unpooler-k-halves-batched-gemm-m65-m128`).

- **Job:** `e82feba3dd8d4efb` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `03252b00`

## Goal
Built on current master (blob 353e3b19). For layer == 'unpooler' at full width (cfg n == 8192) and 65 <= cfg m <= 128 only (scored row verify_16 m=128). The m=130 and m=1000 tests stay on master. WHY: master runs torch.mm on the NN weight here, nvjet 128x64 NNT, 20.4 us kernel / 18.95 row against a 14.64 floor (about 3.5 TB/s). That grid is 128 CTAs of 64 weight columns x full k=4096, so each CTA re-reads the whole 128 x 4096 x token tile (1 MB) from L2. x traffic is then about 128 MB against 67 MB of weight. This is the 'x K-slice re-read about 2x the weight bytes' that REWRITE_READ_2026-10-25 named as the m=128 mainloop limiter. CHANGE: xh = x.view(m, 2, 2048).permute(1, 0, 2) (strided, no copy); wh = state['unpooler_t'].view(2, 2048, 8192) (the existing NN repack, contiguous halves); p = torch.bmm(xh, wh) -> [2, m, 8192]; out = torch.add(p[0], p[1]). With n=8192 and k=2048 per batch, cuBLAS can use 128x128 tiles (64 N-tiles x 2 batches = 128 CTAs), which halves the x bytes per weight byte. A TN-view candidate on state['unpooler'] goes into the same selector. Precision as the pooler halves (exact single-nonzero rows, two bf16 roundings). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Every other layer and m stays byte-identical, including the ouns slice kernel and the swap-AB paths.

## What changed, and where
Only `muse_bf16_proj/submission.py` in the worktree changed (+197/-4). Ported `_khalf_forward`, `_khalf_kernel_names`, relaxed `_khalf_names_ok` and `_khalf_select` from the sibling pooler patch, retargeted at the unpooler (n 8192, k 4096, halves of 2048; candidate N = `unpooler_t.view(2, 2048, 8192)`, candidate T = TN view of `state['unpooler']`, strides (2048, 1, 4096)).
- Own flag `unpooler_khalf_ok` (+ per-m dict `state['unpooler_khalf']`) added to NEW_CELL_FLAGS; `_ab_flags` also carries the dict; state B inherits both; `_ab_component` gained a `state` arg and checks the flag BEFORE the nn_ok range, so a miss falls back to master's NN `torch.mm` and never turns nn_ok off. nn_p256_ok condition untouched.
- Selector runs once on state A (step 10 in `_prepare_state`, after `_lt_prepare`): master exact path vs N vs T, synthetic x, 256 MB flush, 41 interleaved reps, empty-graph baseline, >= 1.02x to switch, reject > 6e-3 rel L2. Prints medians+IQR, bmm-only medians, kernel names, xh.stride(), PATH CHOSEN; FINAL FLAGS line also prints unpooler_khalf. Any exception keeps master. No exact kernel count; only copy/contiguous kernels or splitKreduce disqualify; empty events print "name check skipped".
- Scope: exact m == 128 only (UNPOOLER_KHALF_MS = (128,)), per the task, narrower than the md's 65..128.
- Deviation: kernel_fn also checks cheap host-side shape preconditions (x.dim()==2, shape[0]==cfg m, shape[1]==4096, is_contiguous) so the `view` can never raise; no print/try/state mutation there. Profiler failure inside the selector returns [] (check skipped) instead of aborting.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path is view/permute/bmm/add only, no sync or value-dependent control flow (checked in diff).
- [x] Nothing derived from `data` is cached across calls: the selector uses synthetic x in prepare(); state holds only weight views/choice.
- [x] Weights: no new repack; reuses the existing lossless unpooler_t and views of the original weight, BF16 stays BF16.
- [x] Activations BF16: bmm outputs bf16, add in bf16 (two roundings, as the pooler halves; verified at prepare vs 6e-3 rel L2).
- [x] Attention sees every key: n/a: projection problem, no attention.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (diff adds only torch calls and torch.profiler).
- [x] All work inside the timed call; no output caching or replay.
- [x] Marker `# okc: problem=muse_bf16_proj` kept; no triton.autotune added; no host syncs in kernel_fn.

Lint (`okc_lint.py`): `OK with warnings` (0 errors; warnings are the existing and new prepare()-only synchronize/torch.empty uint8 flush buffers, outside kernel_fn). `py_compile` clean, __pycache__ removed.

## Risks / what could fail
medium
Weakest of the four primaries (best case about 1.4 us on one row, about +0.3% headline). Its value is diagnostic: it tests the x-re-read limiter that bf16-verify16-cutlass-m128-onetile-splitk-rewrite is built to remove. If cuBLAS keeps 128x64 tiles for the halves, the add makes it lose, and the selector should keep master.

## Result (after the run)
- job `e82feba3dd8d4efb` (submit), status done, passed True, tests 9/9
- speedup 1.2452x, 95% CI [1.2265, 1.2639], score 3879.5068978094864 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/unpooler-k-halves-batched-gemm-m65-m128.result.json`, `muse_bf16_proj/hypotheses/done/neutral/unpooler-k-halves-batched-gemm-m65-m128.patch`; branch `hyp/muse_bf16_proj/unpooler-k-halves-batched-gemm-m65-m128`
