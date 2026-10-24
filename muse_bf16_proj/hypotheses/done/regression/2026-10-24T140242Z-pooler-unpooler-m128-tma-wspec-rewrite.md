---
problem: muse_bf16_proj
slug: pooler-unpooler-m128-tma-wspec-rewrite
created: 2026-10-24T140242Z
base: master
area: pooler GEMM + unpooler GEMM (m=128)
change: >
  Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler}
  at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m masked; the m=130 unpooler test
  stays cuBLAS). It starts from the regression pooler-unpooler-m128-wgmma-splitk-semaphore (branch
  hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore, _bf16_tc_splitk_kernel) and
  fixes the two costs that run's own prepare() diagnostics measured (d34fd76f9c444728.log: wgmma
  emitted, 140-154 regs, no spills; warm-L2 mainloop 26.0 us for both layers = 2.6 TB/s, reduce
  tail 6.5 us pooler / 4.3 us unpooler). Mainloop: (1) the weight tile arrives by TMA: a
  TensorDescriptor built once in prepare() from the weight alone
  (triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128]); this image's Triton
  is 3.8.0 with the tensor_descriptor API, per pooler-splitk2-tma-weight-stream's log) and w =
  w_desc.load([n0, k]) [BLOCK_N=128, BLOCK_K=128], fed as tl.dot(x_tile, tl.trans(w), acc); x
  [128, 128] stays a masked tl.load (2 MB, L2-resident). (2) BLOCK_K 128 instead of 64 (256 B
  contiguous per weight row per stage instead of 128 B), num_stages 3 (64 KB per stage, 192 KB),
  num_warps 8 (two warpgroups, wgmma m64n128 each). (3) Warp specialization: the K loop is `for k
  in tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=True)` so a producer warp
  issues TMA while the consumer warpgroups run wgmma; if Triton 3.8 rejects warp_specialize on
  sm90 at compile time in prepare(), fall back to the same loop without it and print which variant
  compiled (a constant fixed in prepare, never input-derived). Grid as before: pooler (n 4096, k
  8192) (32, SPLIT_K=4) = 128 CTAs, K-slice 2048; unpooler (8192, 4096) (64, SPLIT_K=2) = 128
  CTAs. Reduce tail: master's ticket semaphore (masked fp32 partial store into prepare()-allocated
  state[layer+'_m128_ws'] [SPLIT_K, 128, n], debug_barrier, one gpu-scope acq_rel ticket per
  N-tile, debug_barrier, atomic_xchg reset) with one change: the CTA that draws the last ticket
  does NOT store or reload its own partial; it keeps acc in registers and adds the other SPLIT_K-1
  partials (loaded .cg in index order, skipping its own pid_k), then stores bf16 once. That
  removes one 64 KB store and one 64 KB reload from the critical path (pooler reloads 3 partials
  instead of 4, unpooler 1 instead of 2). The sum order then depends on which CTA wins; still
  fp32, well inside the 1% bound. prepare(): build descriptors, allocate workspaces and zeroed
  int32 counters [n // 128], warm m in {65, 96, 128} per layer on synthetic x, re-zero counters,
  and print (try/except, diagnostics only) per layer: variant compiled, n_regs, n_spills, PTX
  counts of wgmma.mma_async and cp.async.bulk.tensor, and warm-L2 event times of the full kernel
  vs a constexpr REDUCE=False variant (as the parent did), so the log attributes any loss to
  mainloop or tail. kernel_fn gates on layer, cfg m in 65..128, cfg n/k and x bf16 contiguous;
  everything else keeps master.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "incumbent 26.0 us (2.6 TB/s) vs floor 14.6 us (67 MB; math 8.7 us at 989). Parent Triton: mainloop 26.0 warm + 6.5 tail = 29.4 us kernel. Milestones: M1 correct with printed mainloop <= 21 us warm-L2 (TMA + BLOCK_K 128 + warp specialization, ~3.3 TB/s); M2 tail <= 4 us (own partial kept in registers); M3 kernel <= 24 us, beating nvjet"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "incumbent 20.3 us vs floor 14.6. Parent: mainloop 26.0 warm + 4.3 tail. Milestones: M1 mainloop <= 18 us warm; M2 tail <= 2.5 us; M3 kernel <= 19.5 us"
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched (cuBLAS)
  - shapes: "* m=32"
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet at m=128 for pooler (nvjet_sm90_tst_64x64_64x13_2x1, 26.0 us) and unpooler (nvjet_sm90_tst_128x64_64x8_1x2, 20.3 us)
review: proceed
status: done
review_at: 2026-10-24T164610Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T164753Z
base_sha: 1221521e184643a43f27ff79cef68b5fed5ad2b1
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite
worktree: .worktrees/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite
mode: submit
implemented_at: 2026-10-24T165200Z
impl_commit: fd7d7ee356c58d5a2dd113f6d6ea476d57cf8953
submission_sha256: 850a87881ace734d04302dcd6cac627825dfdc1367b3deb1ab8f685b4fa65c04
norm_sha256: 28b34e0951842c66b92e645f661656706f61b25c92f12f0a3036f0038895c389
submission_blob: 87f758fdc12055bf479245175d17662450df69eb
lint_warnings: 9
attempt_file: 2026-10-24T181732Z-submit-pooler-unpooler-m128-tma-wspec-rewrite.md
prepared_at: 2026-10-24T181732Z
job: 61da9b91dab44bf4
dispatched_at: 2026-10-24T181804Z
outcome: regression
verdict_detail: regression
speedup: 1.163939020820857
speedup_ci: [1.1464799355085442, 1.1813981061331698]
harvested_at: 2026-10-24T184716Z
best_at_harvest: 1.1958517819004566
promotion: null
---

# pooler-unpooler-m128-tma-wspec-rewrite

## Change

Built on master (658d63b). New kernel _bf16_m128_tma_ws_kernel for layer in {pooler, unpooler} at 65 <= cfg m <= 128 only (scored: verify_16 m=128; rows >= m masked; the m=130 unpooler test stays cuBLAS). It starts from the regression pooler-unpooler-m128-wgmma-splitk-semaphore (branch hyp/muse_bf16_proj/pooler-unpooler-m128-wgmma-splitk-semaphore, _bf16_tc_splitk_kernel) and fixes the two costs that run's own prepare() diagnostics measured (d34fd76f9c444728.log: wgmma emitted, 140-154 regs, no spills; warm-L2 mainloop 26.0 us for both layers = 2.6 TB/s, reduce tail 6.5 us pooler / 4.3 us unpooler). Mainloop: (1) the weight tile arrives by TMA: a TensorDescriptor built once in prepare() from the weight alone (triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128]); this image's Triton is 3.8.0 with the tensor_descriptor API, per pooler-splitk2-tma-weight-stream's log) and w = w_desc.load([n0, k]) [BLOCK_N=128, BLOCK_K=128], fed as tl.dot(x_tile, tl.trans(w), acc); x [128, 128] stays a masked tl.load (2 MB, L2-resident). (2) BLOCK_K 128 instead of 64 (256 B contiguous per weight row per stage instead of 128 B), num_stages 3 (64 KB per stage, 192 KB), num_warps 8 (two warpgroups, wgmma m64n128 each). (3) Warp specialization: the K loop is `for k in tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=True)` so a producer warp issues TMA while the consumer warpgroups run wgmma; if Triton 3.8 rejects warp_specialize on sm90 at compile time in prepare(), fall back to the same loop without it and print which variant compiled (a constant fixed in prepare, never input-derived). Grid as before: pooler (n 4096, k 8192) (32, SPLIT_K=4) = 128 CTAs, K-slice 2048; unpooler (8192, 4096) (64, SPLIT_K=2) = 128 CTAs. Reduce tail: master's ticket semaphore (masked fp32 partial store into prepare()-allocated state[layer+'_m128_ws'] [SPLIT_K, 128, n], debug_barrier, one gpu-scope acq_rel ticket per N-tile, debug_barrier, atomic_xchg reset) with one change: the CTA that draws the last ticket does NOT store or reload its own partial; it keeps acc in registers and adds the other SPLIT_K-1 partials (loaded .cg in index order, skipping its own pid_k), then stores bf16 once. That removes one 64 KB store and one 64 KB reload from the critical path (pooler reloads 3 partials instead of 4, unpooler 1 instead of 2). The sum order then depends on which CTA wins; still fp32, well inside the 1% bound. prepare(): build descriptors, allocate workspaces and zeroed int32 counters [n // 128], warm m in {65, 96, 128} per layer on synthetic x, re-zero counters, and print (try/except, diagnostics only) per layer: variant compiled, n_regs, n_spills, PTX counts of wgmma.mma_async and cp.async.bulk.tensor, and warm-L2 event times of the full kernel vs a constexpr REDUCE=False variant (as the parent did), so the log attributes any loss to mainloop or tail. kernel_fn gates on layer, cfg m in 65..128, cfg n/k and x bf16 contiguous; everything else keeps master.

## Expected

- `layer=pooler load=verify_16 m=128` faster: incumbent 26.0 us (2.6 TB/s) vs floor 14.6 us (67 MB; math 8.7 us at 989). Parent Triton: mainloop 26.0 warm + 6.5 tail = 29.4 us kernel. Milestones: M1 correct with printed mainloop <= 21 us warm-L2 (TMA + BLOCK_K 128 + warp specialization, ~3.3 TB/s); M2 tail <= 4 us (own partial kept in registers); M3 kernel <= 24 us, beating nvjet
- `layer=unpooler load=verify_16 m=128` faster: incumbent 20.3 us vs floor 14.6. Parent: mainloop 26.0 warm + 4.3 tail. Milestones: M1 mainloop <= 18 us warm; M2 tail <= 2.5 us; M3 kernel <= 19.5 us
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched
- `* m=256` flat: untouched (cuBLAS)
- `* m=32` flat: untouched

## Risk

Big-lever rewrite: the pooler m=128 row is the largest single gap on the board (11.4 us over floor), unpooler 5.7 us. The parent's diagnostics show the loss is the mainloop (26 us warm-L2, 2.6 TB/s with cp.async 128-byte row segments) plus a 4-6.5 us reduce tail, not a missing wgmma. Three mainloop changes are bundled on purpose (rewrite stage, judged on the replaced kernels' time against Milestones, not the headline); the printed variant and mainloop/tail times attribute them. Risks: Triton 3.8's warp_specialize may be Blackwell-only (fallback compiles the plain pipelined loop and says so); LESSONS says TMA added little over deep cp.async at m<=16, but there the stream was BLOCK_K 256 and BW-bound, here the shape is co-bound (128 FLOP per weight byte vs ~206 ridge) and the warps must be freed for wgmma. Overlaps bf16-cublaslt-algo-pick-in-prepare-m128-m256 on these two rows; per-shape rows decide which a later combo keeps. Stage 2 (if M1 met): stream-K over 132 SMs or BLOCK_N 256 with one warpgroup per 128 rows; also extend to qkv_gate/o at m=128 (the held qkv-gate-o-m128-wgmma-splitk-semaphore).

## Milestones

Rewrite stage 1 of `cuBLAS nvjet at m=128 for pooler (nvjet_sm90_tst_64x64_64x13_2x1, 26.0 us) and unpooler (nvjet_sm90_tst_128x64_64x8_1x2, 20.3 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T164610Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite aimed at the largest gap on the board: pooler m=128 is 25.7 us against a 14.6 floor, and unpooler is 20.4 against 14.6. Master has not made it moot, because every landed gate is at m<=32 or on the router. Two things must change before it goes out as a scored submit. First, the reduce tail as written is self-contradictory, and the literal reading is a race. Second, none of the 9 tests reaches m 65..128, so a prepare() self-check with a cuBLAS fallback is needed.

### Justification
The mechanism is plausible. The parent's own diagnostics (d34fd76f9c444728.log) show wgmma emitted, 140-154 regs, no spills, and a warm no-reduce mainloop of 26.0 us on both layers, or 2.6 TB/s. Its judged kernels ran 28.2 us (pooler) and 25.6 us (unpooler), against today's nvjet at 25.7 and 20.4 us (FINDINGS verify_16 rows). Both layers move the same 67 MB and have the same per-CTA work (128x2048 weight slice, 512 KB), and their mainloop times are identical. That points to a shared per-CTA streaming limit, which wider 256 B TMA rows and producer/consumer overlap target directly. The rewrite is not just the m<=16 TMA result again (LESSONS: TMA added little there), because m=128 runs at ~128 FLOP per weight byte and the loads have to leave the MMA warps' issue slots. The expected directions match TASK.md: verify_16 pooler/unpooler should be faster and everything else flat, since the 65..128 gate is disjoint from master's m<=16, unpooler n==1024 and 17..32 swap-AB gates, and m=130/256/512/1000 stay cuBLAS. Each row has weight 62 inside one load of five shapes, so a ~6 us pooler gain is worth roughly +1.5-2% on the geomean. The weakest link is the milestone yardstick. M1/M2 are stated in prepare() event times, but the parent's warm event 'full' figure (32.55 us) was above its cold-L2 judged kernel time (28.2 us), so those numbers carry ~3-4 us of event/launch overhead. Judge M1-M3 on profile top_kernels, and use the prepare prints only as a mainloop-vs-tail ratio. On design, the floor at BLOCK_M=BLOCK_N=128 is not just DRAM. Every CTA also re-reads its x K-slice from L2 (pooler: 512 KB x per 512 KB of weight), so L2-to-SM traffic is ~2x the weight bytes. That may cap this tiling above ~1.5x of the floor, and it argues for the stage-2 BLOCK_N 256 or multicast idea. It does not block stage 1, whose target is <=24 / <=19.5 us.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: only as the incumbent. The new path replaces its _bf16_gemm_dispatch_impl call at pooler/unpooler m=128 and nowhere else, so this is not a confounder, just the comparison baseline.
- bf16-shape-gated-triton-winners-combo, bf16-m32-swapab-wgmma-triton: disjoint gates (m<=16 / unpooler n==1024 / 17..32), their own workspaces and counters, and cold L2 per replay. No shared launch, memory traffic or numerical path at m=128, so no interaction. The only coupling is code: the spec says 'Built on master (658d63b)', but master is 8b9ad43, so it must be built on 8b9ad43 and must not disturb the swap-AB gate.
- router-triton-hilo-fused-control-splitk-all-m, router-hilo-at-m8: different layer, untouched. No confound.

### Interference and A/B needs
No companions. No landed hypothesis touches pooler or unpooler at m=128, so ab-revert or ab-tune would measure nothing new. This is a rewrite stage judged on the replaced kernels' own time, and the prepare() prints (variant compiled, regs, PTX counts of wgmma and cp.async.bulk.tensor, mainloop vs REDUCE=False) already attribute the result to TMA/warp specialization or to the tail. Splitting the three bundled mainloop changes into controls would spend checks the next stage needs. The in-flight bf16-cublaslt-algo-pick-in-prepare-m128-m256 overlaps on these two rows by a different mechanism; per-shape rows decide which one a combo keeps.

### Open questions
- Does Triton 3.8's tl.range(warp_specialize=True) accept a loop that mixes a TMA descriptor load (weight) with a masked pointer tl.load (x) on sm90? If it only compiles the plain variant, the run measures TMA plus BLOCK_K 128 alone, and the printed variant must be read before judging M1. Consider a host TensorDescriptor for x built per call in kernel_fn (no caching; OOB rows zero-filled, which removes the mask). Check that it captures correctly in the CUDA graph.
- Is the 26 us mainloop limited by DRAM issue or by L2-to-SM traffic from re-reading x (~2x the weight bytes at BLOCK_N=128)? If M1 stalls near ~20 us even with warp specialization compiled, stage 2 should take BLOCK_N 256 (one x read per 256 weight rows) before stream-K.
- Masked rows at 65..127 are never scored and never tested. Is it worth warming and self-checking them, or should the gate be m == 128 only, to shrink the untested surface on a scored run?


## Implementation notes

Built on worktree master 1221521 (has bf16-m32-swapab-wgmma-triton; the swap-AB gate and every other path are unchanged). The only file touched is `muse_bf16_proj/submission.py`, with +244 lines and nothing deleted:

- New section (6) after swap-AB: `_bf16_m128_tma_ws_kernel`. The tile is BLOCK_M = N = K = 128 with num_warps 8. The weight arrives by TMA as `w_desc.load([n0, k])`. The descriptor is `triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [128, 128])` and is built once in prepare() from the weight alone. x is a masked `tl.load` (rows >= M are zero) and the dot is `tl.dot(x_tile, tl.trans(w_tile), acc)`. The K loop is `tl.range(k_start, k_start + K_SLICE, BLOCK_K, warp_specialize=WARP_SPEC)`, where WARP_SPEC is a constexpr fixed in prepare(). Grid is (N // 128, SPLIT_K): pooler (32, 4), unpooler (64, 2), K-slice 2048.
- Reduce, per the REQUIRED note. EVERY CTA, the winner included, stores its masked fp32 partial into `state[layer+"_m128_ws"]` [SPLIT_K, 128, n]. Then come debug_barrier, one gpu-scope acq_rel ticket per N-tile, and debug_barrier. The last-ticket CTA starts from its in-register `acc` and adds all SPLIT_K slices in index order. Its own slice is masked off with `m_mask & (pid_k != s)`, so it costs no memory traffic and is not reloaded. The CTA then stores bf16 once and resets with atomic_xchg. The spec text said the winner does "NOT store" its own partial. As the review required, I did not implement that: only the reload is skipped. The constexpr REDUCE=False variant is used only by the diagnostics.
- `_m128_prepare` runs per layer. It allocates the descriptor, workspace and int32 counters [n // 128] under its own keys (`<layer>_m128_desc/_ws/_cnt`). It then tries the variants (warp_specialize, num_stages) in the order (T,3), (T,2), (F,3), (F,2). `OutOfResources` steps down to 2 stages. Any other compile or run exception, such as warp_specialize being rejected on sm90, moves to the next variant. The first variant that compiles AND passes `_m128_selfcheck` is fixed in `state["m128_variants"][layer]`. If none passes, the value is None and that layer keeps cuBLAS. After that, prepare() warms m in {65, 96, 128}, prints the diagnostics and re-zeroes the counters. The selected variant is printed on the `[m128_ws <layer>] selected variant:` line.
- The self-check runs on synthetic x at m in {65, 96, 128}, and each m runs twice. The workspace is NaN-poisoned first, so a masked row leaking into out shows up as a failure. Row 0 is a one-hot 2.0 and must be within 1e-3 rel L2. Every row must be within 5e-3 rel L2 against an fp64 reference, the output must be finite, and the counters must read 0 after each launch.
- The diagnostics (try/except, prints only) report n_regs, n_spills, PTX counts of `wgmma.mma_async` and `cp.async.bulk.tensor`, and warm-L2 event times at m=128 for the full kernel vs REDUCE=False (tail = the difference). Per the review, read these as a mainloop-vs-tail ratio and judge M1-M3 on the profile top_kernels.
- kernel_fn gate: after the swap-AB check and before cuBLAS. It requires `state["m128_variants"][layer]` not None, layer in {pooler, unpooler}, 65 <= cfg m <= 128, cfg n/k equal to the layer's, and x a contiguous bf16 [m, k]. m = 130, 256, 512 and 1000 keep cuBLAS.
- Not adopted: the review's optional host TensorDescriptor for x. The spec says x stays a masked tl.load. If the printed variant shows warp_specialize=False, this run measures TMA plus BLOCK_K 128 only, and the x-descriptor is the natural stage-1b follow-up. On the gate (open question 3), I kept 65..128 as specified. The self-check covers 65, 96 and 128.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite diff`:

- [x] CUDA-graph capturable. kernel_fn's new path is one Triton launch plus `torch.empty`. There is no `.item()`, `.cpu()`, sync or print. Branches depend only on cfg, shapes, dtype, contiguity and a prepare()-fixed variant. All `.item()` and synchronize calls are in `_m128_selfcheck`, `_m128_diagnostics` and `_m128_prepare`, which run in prepare() only.
- [x] Nothing derived from `data` is cached. The workspace and counters are allocated in prepare(). Partials are rewritten for rows < m on every call, and the reducer reads only those rows. The descriptor is built from the weight only. No `triton.autotune` is used.
- [x] Weights: the bf16 weights are used as-is (contiguous) through a TMA descriptor. There is no repack and no precision change. The router path is untouched.
- [x] Activations stay bf16 (x as given); accumulation is fp32 and the output bf16, as in cuBLAS. KV-cache / query FP8: n/a, since this problem has no attention.
- [x] Attention sees every key: n/a, since this problem has no attention. No weight-based skipping: every weight tile is read.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The only new import is `triton.tools.tensor_descriptor`.
- [x] All work happens inside the timed call; no output caching or replay. The self-check and warm-ups use synthetic x only, and their outputs are dropped.
- [x] First line `# okc: problem=muse_bf16_proj` is kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
warn  host-sync    L794: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L799: if row_err.max().item() > 5e-3:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L800: return f"m={m} rep={rep}: max row rel L2 {row_err.max().item():.3e}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L801: if row_err[0].item() > 1e-3:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L802: return f"m={m} rep={rep}: one-hot row rel L2 {row_err[0].item():.3e}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L803: if int(cnt.abs().sum().item()) != 0:  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L831: end.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L864: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L886: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```

`python3 -I -m py_compile` passed; the __pycache__ was removed.

## Result

- job `61da9b91dab44bf4` (submit), status done, passed True, tests 9/9
- speedup 1.1639x, 95% CI [1.1465, 1.1814], score 4150.389250283167 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-tma-wspec-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-tma-wspec-rewrite.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-m128-tma-wspec-rewrite`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.91 | 13.43 | 0.9684 | lost |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 6.23 | 1.004 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 19.0 | 19.1 | 1.0161 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.94 | 6.23 | 0.987 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.73 | 6.14 | 0.9713 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.39 | 13.75 | 0.9644 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.66 | 6.75 | 1.004 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.85 | 19.91 | 1.0027 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.61 | 18.35 | 1.0069 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.03 | 5.92 | 1.0232 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.69 | 13.97 | 0.9595 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.67 | 6.38 | 1.0166 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 20.0 | 20.05 | 1.0031 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.48 | 18.43 | 1.0149 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.55 | 6.69 | 0.9867 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.18 | 13.65 | 0.945 | lost |
| layer=o load=verify_1 m=8 | 62 | 6.32 | 6.58 | 0.9795 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.14 | 19.18 | 1.0028 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.13 | 6.18 | 1.0034 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.91 | 5.6 | 1.0408 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.81 | 17.09 | 0.9985 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.09 | 7.42 | 0.98 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.55 | 31.69 | 0.8256 | lost |
| layer=unpooler load=verify_16 m=128 | 62 | 19.79 | 25.68 | 0.7762 | lost |
| layer=router load=verify_16 m=128 | 62 | 7.35 | 7.74 | 0.9605 | lost |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.62 | 19.39 | 0.9899 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.01 | 9.02 | 0.9866 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.53 | 30.62 | 0.9884 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.5 | 24.18 | 0.9933 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.3 | 9.54 | 0.9791 | flat |

gained on router verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; pooler verify_16; unpooler verify_16; router verify_16. Expected hit: 4/7. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `cuBLAS nvjet at m=128 for pooler (nvjet_sm90_tst_64x64_64x13_2x1, 26.0 us) and unpooler (nvjet_sm90_tst_128x64_64x8_1x2, 20.3 us)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_bf16_proj "<what to fix or tune>" --kind rewrite --depends-on pooler-unpooler-m128-tma-wspec-rewrite --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.

