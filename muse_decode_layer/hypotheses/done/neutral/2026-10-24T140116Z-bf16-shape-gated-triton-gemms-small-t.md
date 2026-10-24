---
problem: muse_decode_layer
slug: bf16-shape-gated-triton-gemms-small-t
created: 2026-10-24T140116Z
base: master
area: pooler GEMM
change: >
  Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code
  is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel, _pooler_forward,
  _qkv_splitk_sem_kernel/_qkv_forward, _unpool_forward, _o_skinny_gemm and their constants
  POOLER_* (BLOCK_K 256, SPLIT_K 2), QKV_* (SPLIT_K 4), UNPOOL_* (SPLIT_K 8), O_* (full K, 128
  programs)) to muse_decode_layer's own dense GEMMs, which have exactly bf16_proj's shapes:
  qkv_gate [T,8192]x[2304,8192]^T, o [T,1024]x[8192,1024]^T, pooler [T,8192]x[4096,8192]^T,
  unpooler slice [T,4096]x[1024,4096]^T (rows 0-1023, row stride 4096). Copy the kernels and
  launch helpers verbatim; the area is all four dense-GEMM call sites in kernel_fn (labelled
  pooler GEMM, the largest of them). Routing, by tokens and weight shape only (no data-dependent
  branch), mirroring the combo's gates: qkv = _qkv_forward at 1 <= T <= 16, else _gemm; o =
  _o_skinny_gemm at T <= 16, else _gemm; pooler (main stream, still overlapping the side-stream
  router) = _pooler_forward at T <= 16, else _gemm; unpooler slice = _unpool_forward on
  dense['unpooler'][:UNPOOL_SLICE] at T <= 8 (followed by the existing F.pad all-gather), else
  unchanged. T >= 32 (decode_32, verify_16, verify_32) keeps _bf16_gemm_dispatch_impl everywhere
  as the control. Each path keeps its own fp32 split-K workspace and int32 ticket counter (pooler
  ws [2,16,4096] cnt [64]; qkv ws [4,16,2304] cnt [36]; unpooler ws [8,16,1024] cnt [16]),
  allocated in prepare(); prepare() warms every routed specialization on synthetic bf16 inputs (T
  1..16 for qkv/o/pooler, 1..8 for the unpooler slice), synchronises and zero_()s every cnt, as
  the combo's prepare() does. The router (side stream, hi/lo torch.mm) and everything else are
  untouched.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "T=1, the combo measured per GEMM: pooler 21.2->19.2, qkv 14.4->13.2, unpooler slice 7.1->5.8, o 6.7->6.6 us: ~4.5 us off 173/182 us, +2-3%"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: pooler 21.3->19.6, unpooler 7.0->6.5, qkv 14.0->13.5, o flat: ~2.5-3 us off 238/248 us, ~+1%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "T=16: pooler 21.3->19.5, o 6.8->6.4, qkv flat: ~2-2.5 us off 260/407 us, ~+0.5-1%, near drift"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: cuBLAS kept"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: control
  - shapes: "*load=verify_32 *"
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T164553Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T164553Z
base_sha: 1fd1f0ad0ecb64aecc30d851ca5c1c9149f46de0
base_blob: cc466fce207709c98ab0e78c2435c1c459890fdb
branch: hyp/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t
worktree: .worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t
mode: submit
implemented_at: 2026-10-24T164845Z
impl_commit: 060c5bd147ca9c61a42f45940fed168db05affef
submission_sha256: 57cf77d42680ebadde8963a382e902a9da9aef62cbd1e43c49316083d709d71e
norm_sha256: 6b554775127b23873d9ab8f08bd2d1e80afac710c523d40cc752e631c5daf2b2
submission_blob: 4ba81ce06744ae630e16d82a46eeb6bebf7de2ff
lint_warnings: 3
attempt_file: 2026-10-24T164845Z-submit-bf16-shape-gated-triton-gemms-small-t.md
prepared_at: 2026-10-24T164845Z
job: 6076ae1f250143aa
dispatched_at: 2026-10-24T164859Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.1688022177648374
speedup_ci: [1.1512701844983648, 1.1863342510313097]
harvested_at: 2026-10-24T173233Z
best_at_harvest: 1.1664628806987323
promotion: null
---

# bf16-shape-gated-triton-gemms-small-t

## Change

Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel, _pooler_forward, _qkv_splitk_sem_kernel/_qkv_forward, _unpool_forward, _o_skinny_gemm and their constants POOLER_* (BLOCK_K 256, SPLIT_K 2), QKV_* (SPLIT_K 4), UNPOOL_* (SPLIT_K 8), O_* (full K, 128 programs)) to muse_decode_layer's own dense GEMMs, which have exactly bf16_proj's shapes: qkv_gate [T,8192]x[2304,8192]^T, o [T,1024]x[8192,1024]^T, pooler [T,8192]x[4096,8192]^T, unpooler slice [T,4096]x[1024,4096]^T (rows 0-1023, row stride 4096). Copy the kernels and launch helpers verbatim; the area is all four dense-GEMM call sites in kernel_fn (labelled pooler GEMM, the largest of them). Routing, by tokens and weight shape only (no data-dependent branch), mirroring the combo's gates: qkv = _qkv_forward at 1 <= T <= 16, else _gemm; o = _o_skinny_gemm at T <= 16, else _gemm; pooler (main stream, still overlapping the side-stream router) = _pooler_forward at T <= 16, else _gemm; unpooler slice = _unpool_forward on dense['unpooler'][:UNPOOL_SLICE] at T <= 8 (followed by the existing F.pad all-gather), else unchanged. T >= 32 (decode_32, verify_16, verify_32) keeps _bf16_gemm_dispatch_impl everywhere as the control. Each path keeps its own fp32 split-K workspace and int32 ticket counter (pooler ws [2,16,4096] cnt [64]; qkv ws [4,16,2304] cnt [36]; unpooler ws [8,16,1024] cnt [16]), allocated in prepare(); prepare() warms every routed specialization on synthetic bf16 inputs (T 1..16 for qkv/o/pooler, 1..8 for the unpooler slice), synchronises and zero_()s every cnt, as the combo's prepare() does. The router (side stream, hi/lo torch.mm) and everything else are untouched.

## Expected

- `*load=decode_1 *` faster: T=1, the combo measured per GEMM: pooler 21.2->19.2, qkv 14.4->13.2, unpooler slice 7.1->5.8, o 6.7->6.6 us: ~4.5 us off 173/182 us, +2-3%
- `*load=verify_1 *` faster: T=8: pooler 21.3->19.6, unpooler 7.0->6.5, qkv 14.0->13.5, o flat: ~2.5-3 us off 238/248 us, ~+1%
- `*load=decode_16 *` faster: T=16: pooler 21.3->19.5, o 6.8->6.4, qkv flat: ~2-2.5 us off 260/407 us, ~+0.5-1%, near drift
- `*load=decode_32 *` flat: control: cuBLAS kept
- `*load=verify_16 *` flat: control
- `*load=verify_32 *` flat: control

## Risk

Expected ~+1% geomean: modest but the four kernels are measured winners at these exact shapes, and each replaces an nvjet+splitKreduce pair (two graph nodes) with one launch. Risks: (1) in this layer the pooler overlaps the side-stream router; a 128-CTA Triton pooler may contend with the router's GEMMs differently than nvjet does; read the pooler's own top_kernels time against the bf16 run. (2) The judge uses a cold L2 before each replay here too, as in bf16_proj, so per-kernel times should carry over. (3) The ticket-semaphore reduce is shared-pattern code that passed in bf16_proj; counters are per path, never shared. If the headline is flat, read the per-GEMM rows (nvjet 128x8/64x8 splitK and splitKreduce rows should disappear at T<=16).

## Adversarial review

Reviewed 2026-10-24T164553Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean port of four Triton GEMMs, each measured faster at exactly these shapes in muse_bf16_proj. The expected headline is only about +0.7-1.0% geomean, inside one run's CI, so the result has to be read from the per-GEMM top_kernels rows and the per-shape rows. The two real risks are that a verbatim copy of bf16_proj's cfg-keyed guards KeyErrors or silently falls back to cuBLAS, and that the 128-CTA, 160 KB-smem Triton pooler starves the side-stream router.

### Justification
The mechanism is plausible. bf16_proj measured each kernel faster than nvjet+splitKreduce on the same weight shapes (pooler [4096,8192], qkv_gate [2304,8192], o [8192,1024], unpooler slice [1024 rows of 4096-stride]), also under a cold L2. Each one also removes a graph node, and LESSONS puts that at ~2-4 us per nvjet+splitKreduce pair against ~0.3 us per Triton launch. FINDINGS supports the size of the target. On master the nvjet 128x8 splitK row is 31.9-32.8 us at decode_1 and verify_1 (20.7-21.5% of the sliding and full shapes). At decode_16 the nvjet 64x16 and 128x16 splitK rows add up to ~46 us. The pooler alone runs at ~1.4x its ~14 us floor (21 us), so ~2 us per GEMM is realistic. The expected directions match the gates: T=1, 8 and 16 are routed, and T=32/128/256 stay on cuBLAS. The magnitudes are a little understated in relative terms, because they divide by SGLang's 173-182 us rather than master's 146-157 us. In absolute terms: decode_1 about -4.5 us (+2.9-3.1%), verify_1 -2.5-3 us (+1.2-1.4%), decode_16 sliding -2 us (+0.9%) and full -2 us (+0.6%). Over six equally weighted loads that is about +0.8% on the geomean, below the ~1.5% one run can resolve. Expect a 'neutral' headline even if every GEMM wins. WEAKEST LINK: the pooler's 21.2->19.2 us win was measured with the pooler alone on the GPU. Here it runs on the main stream concurrently with the side-stream router: fp32 F.linear gemv at T=1, two hi/lo torch.mm plus FusedMuseSparkRouting at T=8/16. The Triton pooler is one wave of 128 CTAs at 4 stages x (64x256 + 16x256) x 2 B = 160 KB of smem each, so one CTA per SM fills 128 of 132 SMs for its whole duration. That can serialise the router behind the pooler, where today it fits under the ~22-25 us nvjet pooler (DECISIONS 092231Z). If that happens, the pooler's own row drops while the shape time rises by up to the router branch (~9-18 us). qkv, o and the unpooler run on the main stream with nothing beside them, so their bf16_proj numbers should carry over almost exactly (LESSONS: Triton times reproduce within ~1%).

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: it sets what runs beside the pooler. At T=1 that is the fp32 F.linear gemv; at T>=2 it is two bf16 hi/lo torch.mm on the side stream. How well the 128-CTA Triton pooler overlaps the router depends on that branch's SM and smem demand. A pooler-row gain with a flat or worse shape time would be an interaction effect, not the kernel. With the starter's fp32 router at T=8 the branch would be longer and contention worse. This is the only landed item that can make the measured effect differ from the change's own.
- post-expert-norm-triton-small-t: it is not a timing confounder, but it now produces the unpooler slice's input at T<32 (contiguous bf16 [T,4096], so the stride(1)==1 guard passes). It also removed ~9 launches, so the unpooler's graph-node saving is now a larger share of a shorter shape. If it were absent, x would come from fused_post_expert_norm_weighted_sum, whose output strides should be confirmed. The gain direction would not change.
- fa3-hdim128-head-pairing-batched: only the decode_16 full row overlaps (batch>1 full layers). The o GEMM's input is the FusedRMSNormGate output after the un-pad, a fresh contiguous tensor, so the o path sees the same input as unpaired. No interaction is expected.
- marlin-block-m-cap-16: it affects only T=128/256 (verify_16/32), which are this hypothesis's cuBLAS controls. It is disjoint and does not confound.

### Interference and A/B needs
No companion is warranted. A gain is not ambiguous: the replaced kernels have their own top_kernels rows (the nvjet 128x8 / 64x16 / 128x16 splitK and splitKreduce rows should vanish at T<=16, replaced by _bf16_splitk_kernel, _qkv_splitk_sem_kernel and _pooler_skinny_kernel rows), so each GEMM's contribution can be read per shape. A loss is mostly readable too. If the Triton rows are faster than the nvjet rows they replace but a shape's time rose, the pooler/router overlap is to blame, because nothing else in the change runs concurrently. The only companion that would sharpen this is a control keeping the pooler on _gemm while routing qkv/o/unpooler to Triton. It is worth minting only as a follow-up if the run shows that signature, not before. Checks are spent and this goes out as a scored submit, so spending a second scored attempt in advance on a ~1% effect is poor value. ab-revert of the router change is not meaningful: reverting it changes the T=8 router to the fp32 SIMT GEMM and moves the critical path, so it answers a different question.

### Open questions
- Will the 128-CTA, 1-CTA-per-SM Triton pooler leave the side-stream router to run after it, rather than beside it, at T=8/16? Compare decode_16 and verify_1 shape time minus kernel sum against master's. If it does, a pooler kernel at 3 stages (two CTAs per SM, ~120 KB) or with fewer CTAs is the follow-up.
- The expected geomean gain (~+0.8%) is inside one run's CI. Is a scored attempt the right vehicle, or should bf16_proj's m=32 swap-AB wins (pooler +9.3%, o +4.1%, qkv +3.6% at m=32) be ported into this same run, so that decode_32 (T=32, both layers) also moves and the headline clears noise? That would drop decode_32 as a control. It is the caller's call: as written the hypothesis is clean but probably reads neutral.
- Confirm that attn_input (a judge input), the o_norm_gate output and ffn_input from attn_boundary have stride(1)==1 and a 16-divisible stride(0) at every tested T. If any is a non-contiguous view, the asserting gate catches it in the tests rather than in a silent fallback.


## Implementation notes

Worktree `.worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t`, `muse_decode_layer/submission.py` only (uncommitted; +377/-4).

- **Kernels copied verbatim** from `muse_bf16_proj/submission.py` (master): `_bf16_splitk_kernel`, `_qkv_splitk_sem_kernel` and `_pooler_skinny_kernel`. Their constants are unchanged: POOLER_* (BLOCK_N 64, BLOCK_K 256, SPLIT_K 2, 4 warps, 4 stages), QKV_* (BLOCK_N 64, BLOCK_K 128, SPLIT_K 4, 4 stages), UNPOOL_* (BLOCK_N 64, BLOCK_K 128, SPLIT_K 8, 3 stages) and O_* (BLOCK_N 64, BLOCK_K 256, full K, 128 programs, 4 stages). The launch helpers `_pooler_forward`, `_qkv_forward`, `_unpool_forward` and `_o_skinny_gemm` are the combo's, with one change: BLOCK_M is the literal 16, because only m <= 16 is ever routed. bf16_proj's `16 if m <= 16 else 32` therefore always picked 16. The m = 32 swap-AB code (`_bf16_swapab_kernel`, `_swapab_*`) and the router kernel were **not** ported, as the review requires.
- **Guards rewritten (review REQUIRED).** decode_layer's Config has no m/n/k keys, so the guards (`_bf16_x_ok`, `_bf16_w_ok`, then `_pooler_shape_ok(x, w)`, `_qkv_shape_ok(x, w)`, `_unpool_shape_ok(x, w)` and `_o_shape_ok(x, w)`) read only tensor shapes, dtype and inner stride. x must be bf16 2-D with stride(1) == 1 and 1 <= T <= 16 at the right K. w must be the exact [N, K] row-major weight; for the unpooler it is the leading >= 1024 rows of a [*, 4096] row-major weight.
- **Routing in kernel_fn** uses token count only: `small_t = 1 <= tokens <= SMALL_T_TRITON_GEMM (16)`.
  - qkv_gate: `_qkv_forward`
  - o: `_o_skinny_gemm`
  - pooler: `_pooler_forward` on the main stream. It is still launched after `side_stream.wait_stream(main_stream)`, and the side-stream router block is unchanged.
  - unpooler slice: `_unpool_forward(expert_output, dense["unpooler"][:UNPOOL_SLICE])` inside the existing `tokens <= SLICED_UNPOOL_MAX_TOKENS` branch, followed by the unchanged `F.pad` all-gather.
  - T >= 17 (decode_32, verify_16, verify_32 and the T = 24/33/40 tests) never enters these paths and keeps `_gemm`.
- **Deviation: a logged fallback instead of a silent one or an assert.** The review's open question asked whether attn_input, the o_norm_gate output and ffn_input are contiguous at every tested T. I can't run anything here, so I took a middle path that favours correctness on a scored submit. If a routed T's tensor fails its precondition, `_fallback()` prints one line per (layer, shape, stride, dtype) and the call then takes `_gemm`. It keys only on shapes and strides, never on values, and the print is a Python string, not a tensor, so it is safe under capture. **Read the run log for `[small-t-gemm]` lines.** Any such line means that GEMM silently stayed on cuBLAS, so its expected gain is absent.
- **prepare()** now calls `_prepare_small_t_gemms(state)` before the final synchronize. It allocates `state["small_t"]`, which holds a separate ws/cnt pair per path: pooler ws [2,16,4096] / cnt [64], qkv ws [4,16,2304] / cnt [36], unpool ws [8,16,1024] / cnt [16]. o has no workspace. It warms qkv, pooler and o at T 1..16 and the unpooler slice at T 1..8 on synthetic randn bf16 inputs against the real weights, then calls `torch.cuda.synchronize()`, `zero_()`s every cnt, and synchronises again. M is `do_not_specialize` and BLOCK_M is fixed, so in practice this is one compile per kernel variant.
- The module docstring got one added bullet under "Differs from SGLang".

## Rules check

Checked against `git -C .worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t diff`:

- [x] CUDA-graph capturable. kernel_fn has no `.item()`, `.cpu()`, synchronize or tensor print. The only additions are `torch.empty` outputs, the same kind of allocation `_gemm` and the rest of kernel_fn already make. Branches read `tokens` (config), tensor shapes, strides and dtype; none reads a value. The ticket counters self-reset in-kernel, so graph replays start from 0. The synchronize calls are in prepare() only.
- [x] Nothing derived from `data` is cached across calls. The ws/cnt buffers are allocated in prepare(), every partial a reducer reads is written in the same launch (masked rows), and the outputs are fresh. `_FALLBACK_SEEN` stores only (layer, shape, stride, dtype) tuples for a one-time log line, never values or tensors.
- [x] Weights. The bf16 weights are used as stored (`dense[...]`, contiguous, unchanged); the unpooler slice is a view read in place. The NVFP4 experts and the router hi/lo split are untouched.
- [x] Activations stay BF16 where SGLang keeps BF16. Each kernel is bf16 x bf16 with fp32 accumulation and stores bf16, like cuBLAS. The split-K partials are fp32 and are summed in a fixed order. The KV cache and FP8 query path are untouched.
- [x] n/a: attention is untouched. FA3 still sees every key, with the same window and context.
- [x] No timing patches, harness frames or I/O, module edits, reference import, subprocesses or network. The only new import-level code is `@triton.jit` functions and constants.
- [x] All work happens inside the timed call. prepare() runs only warm-ups on synthetic inputs and discards their outputs. There is no output caching or replay.
- [x] The first-line marker `# okc: problem=muse_decode_layer` is kept, and no `triton.autotune` is used.
- [x] Only the named area changed: the four dense-GEMM call sites and their support code. The router side stream, routing, experts and norms are unchanged in the diff.

okc_lint output:

```
warn  host-sync    L580: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L584: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L627: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t/muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Result

- job `6076ae1f250143aa` (submit), status done, passed True, tests 13/13
- speedup 1.1688x, 95% CI [1.1513, 1.1863], score 19353.402702518822 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/bf16-shape-gated-triton-gemms-small-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/bf16-shape-gated-triton-gemms-small-t.patch`; branch `hyp/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 145.83 | 142.23 | 1.0413 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 156.95 | 152.72 | 1.0418 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 225.79 | 245.8 | 0.9555 | lost |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 364.88 | 384.01 | 0.9719 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 251.94 | 265.63 | 0.9946 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 535.78 | 531.03 | 1.0795 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 200.79 | 215.88 | 0.9644 | lost |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 218.06 | 218.28 | 0.9669 | lost |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 462.37 | 455.09 | 1.0176 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 603.87 | 604.43 | 1.0392 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 639.36 | 647.92 | 1.0097 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 932.92 | 918.99 | 1.0052 | flat |

gained on sliding decode_1; full decode_1, decode_32, verify_16; lost on sliding decode_16, verify_1; full verify_1. Expected hit: 2/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-shape-gated-triton-gemms-small-t`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

