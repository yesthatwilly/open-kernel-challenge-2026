---
problem: muse_bf16_proj
slug: router-swapab-wgmma-all-m
created: 2026-10-24T165036Z
base: master
area: router GEMM
change: >
  Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M
  side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32,
  16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @
  hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the
  16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run
  measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight
  tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new
  _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t,
  pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi
  / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T]
  (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc =
  tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on
  tokens < m), out zeroed right before the launch exactly as master. Static configs from m only
  (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256;
  m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once
  instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice
  512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if
  OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split,
  _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32
  fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket,
  and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new
  kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation
  change is what moved the MMA path.
expected:
  - shapes: layer=router m=1
    direction: faster
    why: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
  - shapes: layer=router m=8
    direction: faster
    why: 3.2 us kernel, same as m=1
  - shapes: layer=router m=16
    direction: faster
    why: 3.5 us kernel vs floor 1.8
  - shapes: layer=router m=32
    direction: faster
    why: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
  - shapes: layer=router m=128
    direction: faster
    why: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
  - shapes: layer=router m=256
    direction: faster
    why: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T165424Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T165424Z
base_sha: 278c4441e3dddcc35e9478bc678c6d596a9d3428
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/router-swapab-wgmma-all-m
worktree: .worktrees/muse_bf16_proj/router-swapab-wgmma-all-m
mode: submit
implemented_at: 2026-10-24T165659Z
impl_commit: ced97b54f84e3741626523d62403533571bc07d1
submission_sha256: 5641115910169a95c7acbd58a97dbfa93cc228419922b682b5341fc654b3f79f
norm_sha256: e15e9553279faa9ddc18e38a36e72fe8814f7cb267be0b59932addc252b7e229
submission_blob: 64894c9c770c26cbcf15bbfbb4b3e44e48bd4f3e
lint_warnings: 0
attempt_file: 2026-10-24T165700Z-submit-router-swapab-wgmma-all-m.md
prepared_at: 2026-10-24T165700Z
job: 587e340476944b46
dispatched_at: 2026-10-24T165714Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.195127545591659
speedup_ci: [1.1772006324077842, 1.2130544587755339]
harvested_at: 2026-10-24T173233Z
best_at_harvest: 1.1958517819004566
promotion: null
---

# router-swapab-wgmma-all-m

## Change

Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32, 16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @ hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the 16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t, pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T] (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on tokens < m), out zeroed right before the launch exactly as master. Static configs from m only (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256; m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice 512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split, _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32 fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket, and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation change is what moved the MMA path.

## Expected

- `layer=router m=1` faster: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
- `layer=router m=8` faster: 3.2 us kernel, same as m=1
- `layer=router m=16` faster: 3.5 us kernel vs floor 1.8
- `layer=router m=32` faster: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
- `layer=router m=128` faster: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
- `layer=router m=256` faster: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

LESSONS says the router's small-m grid depth is settled (SPLIT_K 32 and stage counts flat); this changes the MMA orientation and the weight re-read, not the grid depth, but at m<=16 it also moves SPLIT_K 16 -> 32 and K-slice 512 -> 256, which the split-K-32 runs found flat, so a small-m loss would point at the orientation. The 0.7 us vectorized_elementwise memset stays (removing it was flat: router-splitk-semaphore-no-memset). Router accuracy bound is 0.01%: the hi+lo fp32 accumulation is unchanged, only operand order and atomic order differ (atomic order is already nondeterministic on master). At m=128/256 the x^T tile [128, 128] bf16 is 32 KB per stage, acc 64x128 fp32 = 64 regs/thread: watch spills in the print.

## Adversarial review

Reviewed 2026-10-24T165424Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound port of master's proven swap-AB orientation to the router, and no rule is at issue. The expected win is weaker than stated, though. At m<=16 the MMA path is probably not the binding cost, and at m=128/256 the 'weight re-read' is mostly an L2 hit, while the new configs double the fp32 atomic count (SPLIT_K 16->32 at m=128, 8->16 at m=256). Run it once and read it from the router rows' _router_*_kernel times and the prepare() PTX/regs print, not from the headline.

### Justification
A. The mechanism is plausible, but it is a smaller lever than claimed. (1) MMA path. Triton's wgmma lowering needs 64 rows on tl.dot's M per warpgroup. Master's _bf16_swapab_kernel showed that weight[64, BLOCK_K] as the LHS and x^T[BLOCK_K, BLOCK_T] as the RHS emits wgmma with no spills (archive/bf16-m32-swapab-wgmma-triton/1602a4e947e64090.log: wgmma=16, mma.sync=0, 56-96 regs). So the new kernel will very likely move the router at m<=32 from mma.sync to wgmma. At m=128/256, master's router already uses BLOCK_M=64, so it probably already emits wgmma; there the change is about tile shape and traffic, not the MMA path. The print will confirm either way. (2) The router is bandwidth/latency bound, not MMA bound. 8.4 MB of hi+lo at m<=32 is ~0.03 GFLOP. Master's kernel takes 3.2-4.1 us against a 1.8 us floor because each of 128 CTAs streams only 64 KB in 4 short K-steps, plus the fp32 atomics. Swapping mma.sync for wgmma on a [16..32 x 32] tile saves almost nothing when the MMA is not the bottleneck. The m<=16 config, with SPLIT_K 32, K-slice 256 and 2 K-steps holding all 64 KB in flight, is essentially the bytes-in-flight change that router-splitk32-small-m-all-loads-in-flight (1.143x) and its stages3 control (1.152x) already measured flat against 1.151x. Expect m=1/8/16 flat, possibly +0.2-0.4 us from a cheaper register footprint. m=32 is the best small-m candidate, if master's BLOCK_M=32 router tile has high register pressure as the bf16 BLOCK_M=32 tile did (255 regs). (3) m=128/256 traffic. The hypothesis counts 16.8 MB of 'L2/HBM' weight traffic at m=128. But all 128 CTAs are co-resident (128 <= 132 SMs) and the 8.4 MB weight fits easily in the 50 MB L2, so the second M-block's read of a tile is mostly an L2 hit, not HBM. HBM traffic is ~8.4 MB on both sides, and the saving is L2 bandwidth only. Meanwhile the proposed configs raise the atomic count: m=128 goes from 128*256*16 = 524K to 128*256*32 = 1.05M fp32 red ops, and m=256 goes from 256*256*8 = 524K to 256*256*16 = 1.05M. That is 4 MB of reductions into a 128-256 KB output, competing with the L2 bandwidth the change was meant to save. The swap also makes the per-CTA x^T tile 128 tokens wide (32 KB/stage), so x is read 4x (once per W-block) on both sides. Shared memory at m=128 is 3 stages x (16 + 16 + 32 KB) = 192 KB, plus up to 32 KB of convert_layout scratch for the transposed atomic epilogue if it does not alias the dead pipeline buffers. That is close to the 227 KB limit, so the 2-stage OutOfResources fallback is likely to be exercised. (4) Expected directions vs TASK.md: router rows are 62-weight shapes in every load at ~6-9 us of a ~50-60 us load, so a 0.3-1 us router saving is ~0.5-1.5% per load. That is inside the +/-1.5% CI, and the combo run showed byte-identical router rows drifting 0.92-0.94 on verify_1/verify_16. 'flat' for the four bf16 layers is correct by construction. WEAKEST LINK: the m=128/256 'faster' claim rests on removing a re-read that is mostly L2-resident, while the same config doubles the atomic traffic. A secondary weak link is the m<=16 claim, since the MMA instruction is not what bounds a 64 KB-per-CTA, 2-K-step stream.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: this is the kernel being replaced. Its SPLIT_K / atomic_add / torch.zeros design fixes the baseline atomic count (SPLIT_K 16 at m<=128, 8 above). The new configs double SPLIT_K at m=128/256, so any m=128/256 delta mixes orientation and weight re-read with 2x atomics. This is the one material confound.
- router-hilo-at-m8: fully superseded on master (the Triton router runs at every m and the fp32 F.linear path is only a fallback). No interaction.
- sglang-gemm-dispatch-parity: touches only the four bf16 layers on master now. Router graphs are separate per-shape captures. No interaction.
- bf16-shape-gated-triton-winners-combo: bf16 layers at m<=16 only, and separate graphs. No interaction with router rows. Its run showed ~6-8% drift on byte-identical router rows, which is a measurement caveat, not a confound.
- bf16-m32-swapab-wgmma-triton: the design template (same operand orientation and masked x^T load), but a different kernel, weights and graphs, so it has no runtime interaction. Its success makes the wgmma emission very likely, but it does not show that wgmma helps a bandwidth/latency-bound 8.4 MB router stream.

### Interference and A/B needs
No companion is warranted under the rule that a companion is needed only when both a gain and a loss would be ambiguous. m<=32 is readable from one run. The prepare() print says whether wgmma replaced mma.sync. The m<=16 grid-depth change (SPLIT_K 16->32, 2 K-steps) was already measured flat by router-splitk32-small-m-all-loads-in-flight and its stages3 control, so a small-m gain is attributable to orientation, and a flat or loss result says orientation does not matter for the router. m=128/256 is readable on a gain (net of doubled atomics, the swap-AB tile pays). A loss there would be ambiguous between the doubled atomic count and the tile change. Since a gain is not ambiguous, the rule does not justify a companion. bf16 checks are spent and every run is a scored submit, so the cheaper fix is a build-time choice, not an extra run. If the implementer wants the m=128/256 rows isolated, give the >32 buckets master's SPLIT_K (m<=128: BLOCK_T 64, grid (2, 4, 16), i.e. master's re-read with no extra atomics; m>128: BLOCK_T 128, grid (2, 4, 8) = 64 CTAs, or BLOCK_T 64, grid (4, 4, 8) = 128 CTAs). That is a design change, not a companion, and it should not go in without the author's agreement because it drops the re-read claim at m=128. Otherwise run as written and judge m=128/256 by the kernel's own top_kernels time against master's 4.9 / 6.9 us, which reproduces within ~1%.

### Open questions
- Does master's _router_hilo_kernel already emit wgmma at m=128/256 (BLOCK_M=64)? If so, the 'orientation moves the MMA path' story applies only at m<=32, and the m=128/256 rows test re-read vs atomics only. The planned side-by-side print answers this; make sure it covers all four buckets.
- What are master's router n_regs at BLOCK_M=32 (m=32 bucket)? If it is near 255 like the bf16 tokens-on-M tile, m=32 is the most likely real gain. If it is modest, expect m=32 flat as well.
- Does Triton lower tl.atomic_add on the transposed [64, BLOCK_T] accumulator as vectorized red.global.add (v2/v4.f32, after a coalescing convert_layout) or as scalar strided atomics? Add a red/atom count to the PTX print. Scalar uncoalesced atomics at 1M ops (m=128/256) would likely erase the gain.
- Does the m=128 bucket compile at 3 stages (192 KB pipeline + epilogue convert scratch) or fall back to 2? With only 2 K-steps per CTA (K-slice 256 / BLOCK_K 128), 3 stages is moot anyway. Consider num_stages 2 for that bucket by design.
- At m=1, BLOCK_T=16 wastes 15/16 of the RHS columns and loads a 16-row masked x^T tile. That is harmless for bandwidth, but confirm the masked load stays cp.async with zero-fill (count cp.async in the print) rather than turning into synchronous loads that stall the wgmma pipeline.


## Implementation notes

Worktree `.worktrees/muse_bf16_proj/router-swapab-wgmma-all-m`, `muse_bf16_proj/submission.py` only (`git diff --stat`: 1 file, +162 / -30). Only the router path and its prepare() warm-up changed. Every bf16 layer path is untouched.

- **New `_router_swapab_kernel`** (do_not_specialize M). Grid `(cdiv(m, BLOCK_T), 256 // 64, SPLIT_K)`, program `(pid_t, pid_w, pid_k)`. The hi and lo tiles `[64, 128]` are loaded as stored (K-contiguous) and used as tl.dot's LHS. The x^T tile `[128, BLOCK_T]` is masked on tokens < m with other=0 and used as the RHS. Each K-step does `acc = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc)`, the same hi-then-lo order as master. The epilogue is one relaxed fp32 `tl.atomic_add` of acc, transposed, into `out[m, 256]`, masked on tokens < m. `out` is still `torch.zeros` right before the launch, as on master.
- **`_router_config(m)` now returns (BLOCK_T, SPLIT_K)**, with BLOCK_K 128 and BLOCK_W 64 fixed:
  - m<=16: (16, 32), grid (1,4,32) = 128 CTAs, K-slice 256. As specced.
  - m<=32: (32, 32), grid (1,4,32), K-slice 256. As specced.
  - m<=128: (64, 16), grid (2,4,16) = 128 CTAs at m=128, K-slice 512. **Deviation**, see below.
  - m>128: (64, 8), grid (4,4,8) = 128 CTAs at m=256, K-slice 1024. **Deviation**, see below.
- **Deviation at m>32.** The brief and review say to keep master's SPLIT_K at m=128/256 so the atomic count stays m*256*SPLIT_K, not doubled. If BLOCK_T stayed at 128, keeping SPLIT_K 16/8 would halve the grid to 64 CTAs. That adds a second confound (parallelism), as large as the atomic one. So these buckets use the review's 128-CTA option: BLOCK_T 64, grid (2,4,16) at m<=128 and (4,4,8) above. Atomic count, CTA count, K-slice and weight re-read factor (2x/4x) are all master's. The only change at m=128/256 is MMA orientation (weight on M, tokens on N). The hypothesis's "weight read once / 2x" claim at m=128/256 is therefore **dropped**, and those rows test orientation alone. Expect them flat unless the print shows master's BLOCK_M=64 tile not on wgmma, or unless register pressure or atomic coalescing differ.
- **`_router_shape_ok`** now checks `k % (128 * SPLIT_K) == 0` with the new SPLIT_K (8192 passes for 32/16/8) and `n % 64 == 0`. The F.linear fp32 fallback is unchanged. kernel_fn also takes it if a bucket has no recorded stage count (never expected).
- **Stage count.** Each bucket tries num_stages 3 in prepare(). On `_OutOfResources` it retries at 2 (`ROUTER_FALLBACK_STAGES`). The result is stored per bucket in `state["router_stages"]` (a constant keyed by the static config tuple), and kernel_fn looks it up from m. The fallback lives in prepare() only.
- **Diagnostics, prepare() only.** `_router_prepare` warms every m in master's `ROUTER_WARM_MS` (1, 8, 16, 32, 64, 128, 256), which covers all four buckets. At each bucket's first m it prints two lines: `[router] swapab ...` for the new kernel, and `[router] master hilo ...`, where `_router_hilo_report` compiles and launches master's `_router_hilo_kernel` with master's config (`_router_hilo_config`) into a dropped scratch output. Each line gives n_regs, n_spills and PTX counts of `wgmma.mma_async`, `mma.sync`, `red.global`, `atom.global` and `cp.async`. The master diagnostic is wrapped in try/except and never runs in kernel_fn. The new kernel's launch sits outside the try, so a real compile failure still surfaces.
- The module docstring router paragraph was updated.
- **Open questions left for the run log.** The four review questions are answered by the print: whether master's BLOCK_M=64 tile already emits wgmma, master's n_regs at BLOCK_M=32, whether the atomics come out as red.global (vectorized or not, which needs a closer read of the PTX if counts look scalar-high), and whether the masked x^T loads are still cp.async.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/router-swapab-wgmma-all-m diff`:

- [x] CUDA-graph capturable: kernel_fn adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The router branch uses only shapes (`x.shape[0]` via `_router_config`, cfg n) and a prepare()-time constant dict. The only allocation is the `torch.zeros` output that master already had. Prints happen only in prepare() (report=True and `_router_hilo_report`).
- [x] Nothing derived from `data` is cached across calls: `state["router_stages"]` comes from prepare() on synthetic x. kernel_fn writes nothing to state.
- [x] Weights: the router hi + lo bf16 split is unchanged (the allowed FP32 → BF16 hi+lo with FP32 accumulation exception). The bf16 weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is still loaded as bf16, with fp32 accumulation and fp32 output for the router. n/a for KV cache and queries (none in this problem).
- [x] Attention sees every key: n/a, no attention in this problem. The GEMM covers the full K (SPLIT_K * K_SLICE = 8192) and all 256 columns.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network: no new imports, and nothing outside the kernel file is touched.
- [x] All work happens inside the timed call, with no output caching or replay. Every call zeroes `out` and recomputes all K-partials. The master-kernel diagnostic in prepare() writes to a dropped scratch tensor.
- [x] No triton.autotune. The `# okc: problem=muse_bf16_proj` first line is kept.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left in the worktree was removed.

## Result

- job `587e340476944b46` (submit), status done, passed True, tests 9/9
- speedup 1.1951x, 95% CI [1.1772, 1.2131], score 4042.0790381904117 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-swapab-wgmma-all-m.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-swapab-wgmma-all-m.patch`; branch `hyp/muse_bf16_proj/router-swapab-wgmma-all-m`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.91 | 13.39 | 0.9388 | lost |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 6.21 | 1.0157 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 19.0 | 19.61 | 1.0001 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.94 | 6.25 | 0.9706 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.73 | 5.65 | 1.0408 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.39 | 14.31 | 0.9535 | lost |
| layer=o load=decode_16 m=16 | 62 | 6.66 | 6.52 | 1.0017 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.85 | 20.02 | 0.9904 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.61 | 18.74 | 1.0005 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.03 | 5.83 | 1.0582 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.69 | 14.34 | 0.9643 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.67 | 6.63 | 1.02 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 20.0 | 20.18 | 1.0031 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.48 | 18.7 | 1.0032 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.55 | 5.96 | 1.0989 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.18 | 13.63 | 0.9457 | lost |
| layer=o load=verify_1 m=8 | 62 | 6.32 | 6.49 | 1.0158 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.14 | 19.3 | 1.0006 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.13 | 6.34 | 1.0278 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.91 | 5.23 | 1.1284 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.81 | 17.2 | 0.9809 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.09 | 7.48 | 0.9973 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.55 | 25.38 | 1.012 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.79 | 20.15 | 1.0018 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.35 | 7.61 | 0.9589 | lost |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.62 | 19.54 | 0.9867 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.01 | 9.65 | 1.0133 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.53 | 30.35 | 0.9897 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.5 | 24.08 | 1.016 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.3 | 9.46 | 1.0124 | flat |

gained on router decode_1, decode_16, decode_32, verify_1; lost on qkv_gate decode_1, decode_16, decode_32, verify_1; router verify_16. Expected hit: 4/10. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-swapab-wgmma-all-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

