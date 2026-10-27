# muse_prefill_layer · 2026-10-27T141758Z · check · o-proj-triton-gemm-turn-gated

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-27T133236Z-o-proj-triton-gemm-turn-gated.md` (branch `hyp/muse_prefill_layer/o-proj-triton-gemm-turn-gated`).

- **Job:** `af3b0d0d84834ee9` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `d1039266`

## Goal
Built on master (blob 42ded471). Only the call attn_output = _gemm(o, dense['o']) in kernel_fn changes, and only at 200 < T <= 4096. MECHANISM: the o projection is [T, 1024] x [8192, 1024]^T (K = 1024). At turn_1 (T = 753) cuBLAS runs it as nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN in 39.7 us (sliding) / 40.1 us (full) against a 12.8 us math floor (12.6 GFLOP) and ~6.4 us of bytes: 3.1x, the largest gap to the floor among the dense GEMMs (the others run at ~80-85% of peak). With K = 1024 each 112x128 tile has only 16 k-steps, so 448 tiles in 3.4 waves are dominated by per-tile prologue/epilogue and the partial last wave. CHANGE: a Triton bf16 GEMM _o_proj_kernel (C = A @ W^T, tl.dot with fp32 accumulation, one bf16 rounding at the store, K = 1024 constexpr, unmasked K loop, masked M edge, grouped tile order GROUP_M = 8 for L2 reuse of W, plain pointer loads with num_stages pipelining: no TMA descriptors, no allocator hook) with three candidate configs (BM, BN, BK, stages, warps) = (128, 128, 64, 4, 8), (128, 256, 64, 3, 8), (64, 256, 64, 4, 4). GATE in prepare() per class, low (T = 753) and high (T = 3012), after master's setup and before the rega-lo / fsilu phases (start deadline 150 s, skipped otherwise): correctness per config on random A (and A from a real o_norm_gate output) against an fp64 reference: finite, rel L2 <= 1.05x cuBLAS's own rel L2, worst row <= 2x cuBLAS's; then one-GEMM CUDA graphs (cuBLAS _gemm and each passing config) plus an empty graph, 2 rounds x 21 interleaved alternations, cold L2 (256 MB read flush before each replay, as the judge), empty-graph baseline subtracted; ON with the fastest config only when it is <= 0.97x cuBLAS in BOTH rounds (the interleaved selector with margin is the form that carried over in muse_bf16_proj; a single prepare timing did not). One torch.profiler call with the path ON asserts the launched kernel name is _o_proj_kernel. state['o_proj_cfg'][cls] = config tuple or None; kernel_fn branches on tokens (a shape) only. T > 4096 and T <= 200 keep _gemm. Print per class cuBLAS us, each config's us and rel L2, and '[o-proj] PATH CHOSEN'.

## What changed, and where
Worktree `.worktrees/muse_prefill_layer/o-proj-triton-gemm-turn-gated`, only `muse_prefill_layer/submission.py` (+300 / -2, uncommitted). Follows the adversarial review's binding details.

- **kernel_fn**: the one call `attn_output = _gemm(o, dense["o"])` becomes `_o_projection(o, dense["o"], state, tokens)`. Shape-only dispatch: at 200 < T <= 4096 it takes `state["o_proj_cfg"][class]` (class low T <= 1024 = REGA_LOW_MAX, high above, the rega split); None, T <= 200, T > 4096, or any dtype/shape/contiguity other than bf16 [T, 1024] contiguous falls back to master's `_gemm`. Output is a fresh [T, 8192] bf16 contiguous tensor, the layout attn_boundary already consumes.
- **`_o_proj_kernel`** (new Triton kernel, `do_not_specialize=["M"]` like the file's other Triton kernels; N = 8192, K = 1024 and the tile sizes constexpr; no other int args, no strides passed): C = A @ W^T, `tl.dot` with an fp32 accumulator, one bf16 rounding at the store; unmasked K loop (16 steps of BK 64, plain `range` so num_stages pipelines it); plain pointer loads, no TMA, no allocator hook; M edge rows wrap (`% M`) on load and are masked on store; grouped tile order GROUP_M = 8.
- **Candidates** (BM, BN, BK, stages, warps): (128, 128, 64, 4, 8), (128, 256, 64, 3, 8), (256, 128, 64, 3, 8). 64x256 dropped per review; 256x128 added (3 x 48 KB = 147 KB smem).
- **`_o_proj_phase`**, called LAST in prepare(): after `_rega_fsilu_phase` and after the align2 / fa3c / rega-lo / fsilu summary prints, so its lines are the last of the log. Start deadline 245 s (whole phase skipped), step deadline 270 s (no compile / class gate after it). Never raises; any miss leaves cuBLAS.
  1. Compile: one launch per config at T = 753 (one compile each, since M is not specialized); a config that fails to compile/launch is dropped.
  2. Per class (low T = 753, high T = 3012), `_o_proj_class_gate`: correctness on a random A and on a real `o_norm_gate` output (gate as a strided view of a qkv-shaped tensor, as kernel_fn passes it), against an fp64 reference on every 4th row plus the last 8 rows (the masked edge): finite, rel L2 <= 1.05x cuBLAS's own and worst row <= 2x cuBLAS's (using `_ssdq_metrics`, the judge's row rule); against cuBLAS's full output within the file's E2E_CHECK_TOL (2e-2) and rows within E2E_ROW_FACTOR x that; plus 256 sentinel rows past M in the output buffer must stay untouched. Then one-GEMM CUDA graphs (cuBLAS `_gemm`, each passing config, an empty graph) timed by the file's `_rega_fsilu_bench` in cold mode (256 MB read flush before each replay, as the judge), 2 rounds x 21 interleaved rotated/flipped alternations, empty-graph median subtracted per round. The fastest config (mean of the round medians) goes ON only when cand / cuBLAS <= 0.97 in BOTH rounds.
  3. With a class ON, one torch.profiler call of `_o_projection` at the gate's T: exactly one launched kernel whose name contains `_o_proj_kernel`, else the class goes OFF. A profiler that is unavailable or returns no events keeps the route and says so in the notes line (as muse_bf16_proj's `_cell_kernels` convention does).
  4. Graph-capture hazard: for each ON class it launches `_o_projection` at T = 201, 300, 600, 753, 1024, 1025, 1506, 3012, 4096 (every routed T the judge uses, plus the class edges) before any capture, each checked against cuBLAS (E2E tolerance; a miss turns the class OFF). With M do_not_specialize these reuse the one compile per config.
- **Which config each test T inherits**: T = 300, 600, 753 -> the low class's pick (gated at 753); T = 1506, 4096 -> the high class's pick (gated at 3012); T = 200 (batch 3 x q_len 200 is T = 600, low) - only a T <= 200 total would stay cuBLAS, and none of the tests is. The phase prints this per test T in a `[o-proj] tests inherit:` line.
- **Log lines** (compact, <= 900 chars each, all after fsilu's): per class `[o-proj] low T=753: cuBLAS a/b us rel .. row ..; 128x128 a/b us (ratio r0/r1) rel .. row ..; ...` (or `FAIL (reason)`), a notes line (compile/profiler), `[o-proj] phase Xs (compile, low, high); prepare total`, `[o-proj] tests inherit: T=300:low->... ; confounders: fsilu {state['rega_fsilu_ok']}, rega_low_variant v`, and last `[o-proj] PATH CHOSEN: low (T in (200, 1024]) triton 128x256 (...) [why]; high (T in (1024, 4096]) cuBLAS [why]`.
- Top docstring: one note on the o projection line. The prepare() comment that said fsilu's lines were the last now says only o-proj's follow them.
- **Deviations**: (a) "the file's existing GEMM tolerance": the file has no dense-GEMM tolerance; the closest is E2E_CHECK_TOL (2e-2, two bf16 pipelines compared to each other) with E2E_ROW_FACTOR, used for the vs-cuBLAS check. The tight check is the fp64 one (1.05x / 2x cuBLAS's own error). (b) The fp64 reference is computed on sampled rows (every 4th + the last 8), as the review allows ("on a sample"). (c) Fastest = lowest mean of the two round medians; that config alone must pass 0.97 in every round (the literal reading; no fallback to a second-fastest config).

Uncertainties (cannot run here: no GPU / torch / Triton on this node): Triton compile time for the three configs (expected ~5-15 s; prepare was ~216 s, so the phase should start well before 245 s); whether the plain-pointer 128x256 / 256x128 configs pipeline well at 3 stages with 8 warps (an accumulator of 128 fp32 registers per thread); whether the profiler event name for a Triton launch contains `_o_proj_kernel` (it should: Triton names the cubin function after the Python function).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/o-proj-triton-gemm-turn-gated diff` (ATTEMPT_TEMPLATE.md items):

- [x] CUDA-graph capturable: `_o_projection` / `_o_proj_gemm` only allocate the output with `torch.empty` and launch one Triton kernel; no `.item()` / `.cpu()` / `synchronize()` / tensor prints in kernel_fn's call graph (all new syncs and the profiler are in prepare-only `_o_proj_phase` / `_o_proj_class_gate` / `_o_proj_kernels`). Branches are on `tokens`, dtype, shape and contiguity only. M is do_not_specialize and every routed T is launched in prepare before any capture, so no JIT compile can happen during capture.
- [x] Nothing derived from `data` is cached across calls: state gains only `o_proj_cfg` = {class: config tuple or None}, decided in prepare from synthetic inputs.
- [x] Weights: unchanged; the kernel reads master's `state["dense"]["o"]` (bf16 [8192, 1024]) as is. No repack, no precision change.
- [x] Activations BF16 where SGLang keeps BF16: A and C stay bf16 with fp32 accumulation and one bf16 rounding (as cuBLAS's bf16 GEMM); KV cache and queries untouched.
- [x] Attention sees every key: n/a to this change (attention code untouched).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: new code imports only `torch.profiler` (inside a prepare-only helper, for the kernel-name assertion); no triton.autotune.
- [x] All work happens inside the timed call; no output caching or replay: the o GEMM runs in every kernel_fn call; prepare's gate graphs are its own and deleted.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`): `OK with warnings`, exit 0. 115 warnings vs master's 108: the 7 new ones are `torch.cuda.synchronize()` lines in the prepare-only o-proj phase (L6465, L6468, L6498, L6522, L6576, L6621, L6629), all flagged "outside kernel_fn's call graph; fine in prepare()". `python3 -I -m py_compile` passed; __pycache__ removed.

```
warn  host-sync    L471: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L513: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L547: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L603: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L609: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L619: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L624: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L628: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L634: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L639: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L645: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L660: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L814: sorted_ids = sorted_ids.cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L815: block_experts = block_experts.cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L816: used = min(int(n_padded.cpu()[0]) // bsm, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L819: blocks = (block_experts[:used] == e).nonzero().flatten().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L823: routes += seg[seg < n].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L838: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L861: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L911: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1104: idx = (a != b).nonzero().flatten()[0].item()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1105: return f"{name}[{idx}] master {a.view(-1)[idx].item()} vs new {b.view(-1)[idx].item()}"  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1123: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1162: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1182: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1194: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1320: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1325: err = ((g - r).abs() / (r.abs() + 1e-3)).max().item()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2067: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2074: view = tensor[:int(rows.item())]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2145: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2149: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2189: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2196: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2197: word = int(states[i]["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2198: marks = states[i]["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2213: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2233: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2307: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2308: word = int(state["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2309: marks = state["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2335: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2337: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2338: word = int(state["ssdq_err"][0].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2339: marks = state["ssdq_err"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2369: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2370: prof = state["ssdq_prof"].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3469: q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L3470: q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L3630: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3633: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3671: word, m13, m2 = [int(v) for v in err.tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3726: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3727: be = block_experts[:used].cpu().long()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3752: pads.append(torch.empty(2 * 1024 * 1024, dtype=torch.uint8, device=device))  (allowed only where SGLang does the same)
warn  host-sync    L3757: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3789: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3794: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3813: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3852: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3910: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3912: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3941: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3947: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4057: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4067: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4087: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4089: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4153: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L4299: return (torch.empty(shape, dtype=torch.uint8, device=device), torch.empty(shape, dtype=torch.uint8, device=device))  (allowed only where SGLang does the same)
warn  host-sync    L4313: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4332: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4351: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4405: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4448: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4477: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4489: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4604: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4623: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4662: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4665: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4686: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4825: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4832: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5415: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5416: be = block_experts[:used].cpu().long()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5419: routes = torch.bincount(lids[lids >= 0].long().cpu(), minlength=LOCAL_EXPERTS)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5469: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5474: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5543: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5562: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5580: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5634: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5707: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5900: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5901: f, a, r, hgl = fast.cpu().tolist(), acc.cpu().tolist(), ref.cpu().tolist(), hg.cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5902: fs, as_ = torch.signbit(fast).cpu().tolist(), torch.signbit(acc).cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5993: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6008: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6011: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6039: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6048: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6054: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6111: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6195: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6229: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6347: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6465: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6468: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6498: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6522: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6576: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6621: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L6629: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L6694: k_buf, v_buf = (torch.empty((batch, width, 2, 64), dtype=torch.uint8, device=q8.device)  (allowed only where SGLang does the same)
OK with warnings: /home/user/git/musespark/.worktrees/muse_prefill_layer/o-proj-triton-gemm-turn-gated/muse_prefill_layer/submission.py
```

## Risks / what could fail
low
Likely just under the 1.2% judge margin alone (turn_1 load -2 to -3%); a later stage can fuse o_norm_gate into the A loads (head_dim 64 = one BK chunk, so the per-head RMSNorm and sigmoid gate fit the prologue) to remove that kernel and the o round trip. Risks: Triton at small M may not beat nvjet (LESSONS: a 128x128x64 split-K Triton kernel lost at m=128 in bf16_proj; this is no-split-K at M = 753 with a short K, where cuBLAS is visibly off its peak); Triton JIT for three configs costs ~10-15 s of prepare (prepare ~216 s today). Read the scored turn_1 profile: the nvjet 112x128 row should be gone and an _o_proj_kernel row in its place; a PATH CHOSEN line alone proves only the selector (LESSONS 03:41Z).

## Result (after the run)
- job `af3b0d0d84834ee9` (check), status done, passed True, tests 5/5
- speedup 1.4863x, 95% CI [1.4640, 1.5086], score 260652.82145331614 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.488912488847924, "ci": [1.466578801515205, 1.5112461761806426], "job": "ec3bd01859574bf1"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-rega-w13-fast-silu-epilogue.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/o-proj-triton-gemm-turn-gated.result.json`, `muse_prefill_layer/hypotheses/done/neutral/o-proj-triton-gemm-turn-gated.patch`; branch `hyp/muse_prefill_layer/o-proj-triton-gemm-turn-gated`
