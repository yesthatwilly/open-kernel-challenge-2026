# muse_bf16_proj · 2026-10-27T104308Z · check · qkv-swapab16-five-stage-ring-m8-m16

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-26T221149Z-qkv-swapab16-five-stage-ring-m8-m16.md` (branch `hyp/muse_bf16_proj/qkv-swapab16-five-stage-ring-m8-m16`).

- **Job:** `c65ffcb6d9ab4f95` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `998413f4`

## Goal
Built on master blob 2bcfe7d4. qkv_gate at m=2..16 (verify_1 m=8 12.86 us, decode_16 m=16 12.99 us in 3834ceec) runs _bf16_swapab_kernel through _swapab16_forward with SWAPAB16_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): 144 CTAs, 48 regs, ~80 KB smem, 2 CTAs/SM, so only ~3 x 16 KB of weight is in flight per CTA, while the pooler's swapab16 path keeps ~3 x 32 KB in flight and streams ~3.4 TB/s against qkv's ~2.9 TB/s (row basis). Give qkv a 5-stage ring at the same split, tile and occupancy. In _swapab16_prepare, compile a num_stages=5 qkv_gate variant on synthetic x, print its n_regs and compiled metadata.shared, and keep it only if shared <= 113 KB (2 CTAs/SM still fit); then warm every m in 1..16 at 5 stages so the timed call never compiles. New flag qkv16_s5_ok (False by default, appended to NEW_CELL_FLAGS): kernel_fn passes num_stages=5 instead of state['swapab16_stages']['qkv_gate'] when the flag is on and key16 == 'qkv_gate' (a cfg / flag-only choice; everything else in the call is identical). Gate with master's _cell_gate on that flag at m=8 and at m=16 (cand = 5 stages, master = 4 stages, same workspace and tickets), using a judge-like read flush because the 37.7 MB weight fits in L2: add flush_mode='read' to _cell_gate (flush.view(torch.int32).sum() in place of flush.zero_(); default unchanged for every other caller). Two rounds per m; route only if all four master/cand ratios are >= 1.03. State B inherits the flag (and re-warms 5 stages); add AB_CASES (qkv_gate, 2304, 8192, 8) and (qkv_gate, 2304, 8192, 16), mapped to qkv16_s5_ok in _ab_component. One PATH CHOSEN line with the four ratios and the printed smem. m=17..32 (24 KB per stage, where a fifth stage would break 2 CTAs/SM), m=1 and every other layer stay untouched.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/qkv-swapab16-five-stage-ring-m8-m16`, `muse_bf16_proj/submission.py` only (uncommitted, base 8f8cca6a / blob d51c745a). Diff: +175 / -3.

- **Section (6) constants** (after `SWAPAB16_UNPOOL_WARM_MS`): `QKV16_S5_FLAG = "qkv16_s5_ok"`, `QKV16_S5_STAGES = 5`, `QKV16_S5_MIN_M = 2`, `QKV16_S5_MAX_SHARED = 113 * 1024`, `QKV16_S5_CANARY_MS = (8, 16)`.
- **`_swapab16_forward`**: new optional `compiled_out=None` list argument that receives the launch's compiled kernel (prepare-only, for `metadata.shared`); default behavior unchanged.
- **`_qkv16_s5_warm(state)`** (new, section 6), called from `_swapab16_prepare` right after `state["swapab16_stages"]` is set, so it runs in **both** state A and state B, before any graph capture (the first capture is `_ouns_gate`). It compiles the 5-stage qkv_gate variant on synthetic x at m=16 (`report=True` prints regs and PTX counts), then prints `compiled.metadata.shared` (falling back to `compiled.shared`) and `n_regs`. It refuses if `shared` is unavailable or larger than 113 KB, with the "12 of 144 CTAs into a second wave" reason. Otherwise it warms every m in 1..16 at 5 stages and runs the canary: a seeded synthetic x at m=8 and m=16 through `_swapab16_forward` at the incumbent stage count and at 5 stages, compared with `torch.equal`. Any difference, refusal or exception leaves `state["qkv16_s5_ready"] = False`, with the reason in `qkv16_s5_info`. The function never raises and re-zeros the qkv tickets at the end.
- **Section (12) `_qkv16_s5_prepare(state, t_start)`** (new, after khalf's `NEW_CELL_FLAGS` append; state A only, run last after khalf in `_prepare_state`). It needs `qkv16_s5_ready`. It then runs a kernel_fn-level canary: flag on vs flag off must be `torch.equal` at m=8 and m=16 on synthetic x. For each m in (8, 16) it runs 2 rounds of master's `_cell_gate(..., "qkv_gate", m, 2304, 8192, QKV16_S5_FLAG, state, flush_mode=mode, route_txt=bar)`, with mode in `(B12_GATE_FLUSH, "write")`. That is the existing `flush_mode`; `_cell_gate` itself was not changed. It routes only if **all four** read-flush master/cand ratios are >= 1.03 (`QKV16_S5_MIN_RATIO`). The write-flush ratio is printed beside each one but is not gated on. There is also a 200 s start deadline, the same pattern as khalf's 170 s. One `_decision("PATH CHOSEN S5 qkv_gate m=2..16: ...")` line carries the smem and all four rounds (read ratio and write ratio with master / cand us). `_decision` lines are re-printed at the end of prepare, so the line appears in the end-of-prepare DECISIONS re-print.
- `NEW_CELL_FLAGS += (QKV16_S5_FLAG,)`, so the flag is in the A/B flag hand-off, the A/B failure switch-off and the FINAL FLAGS line.
- **`_prepare_state`**: `state[QKV16_S5_FLAG] = False` is set next to the KHALF init. State A calls `_qkv16_s5_prepare` after `_khalf_prepare`. State B inherits `flags[QKV16_S5_FLAG] and state["qkv16_s5_ready"]`: B re-compiles, warms and canaries the variant in its own `_swapab16_prepare`. If its own check fails, B prints why and keeps 4 stages.
- **`AB_CASES`**: + `("qkv_gate", 2304, 8192, 8)`, `("qkv_gate", 2304, 8192, 16)`. **`_ab_component`**: qkv_gate at 2 <= m <= 16 maps to `qkv16_s5_ok` when the flag is on in state A (khalf pattern). When the flag is off, the cell reports "master path", so a miss can switch off only the 5-stage ring.
- **`kernel_fn`**: on the swapab16 branch, `stages16 = state["swapab16_stages"][key16]`, replaced by `QKV16_S5_STAGES` only when `key16 == "qkv_gate" and cfg["m"] >= 2 and state[QKV16_S5_FLAG]`. Grid, SPLIT_K, BLOCK_K, workspace, tickets and the reduce order are unchanged. The choice depends only on cfg and the flag.

Deviations from the `change` paragraph, all from the adversarial review:
- `flush_mode='read'` was **not** added to `_cell_gate`, because master already has it (`CELL_FLUSH_MODES`, `B12_GATE_FLUSH = "read"`). The gate calls it with `flush_mode=B12_GATE_FLUSH`.
- The 5-stage selection is restricted to cfg m >= 2, which leaves m=1 structurally untouched even if b12 and bulk are both off. The variant is still warmed at m=1..16 as required.
- The canary was added in two places: inside `_qkv16_s5_warm` (every state, direct launches) and at the kernel_fn level (state A, before the gate).
- Expected numbers were re-anchored to master d51c745a's rows in check a6424247: verify_1 m=8 is 12.71 us and decode_16 m=16 is 13.12 us (confirmed in the archived profile.txt). The `change` paragraph's 3834ceec figures are left as history.
- No torch.profiler kernel-name assertion. Both sides launch `_bf16_swapab_kernel`, so the name cannot tell 4 stages from 5. Evidence that routing happens comes from the kernel_fn canary plus the gate's measured cand vs master graphs.

Uncertainties:
- I could not run this here (no GPU, torch or Triton). The `compiled.metadata.shared` attribute path is assumed from Triton 3.x. If it is missing, the flag stays off and the log says so.
- Expected smem is ~100 KB (5 x (16 KB w + 4 KB x) plus barriers). That is under the limit, but this is unverified until the run prints it.
- The added gate cost is 8 `_cell_gate` calls (2 m x 2 rounds x read/write) and 4 more A/B captures. That should take well under a few seconds next to the ~5.7 s prepare in a6424247.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/qkv-swapab16-five-stage-ring-m8-m16 diff`:

- [x] CUDA-graph capturable: kernel_fn gained one branch on `key16`, `cfg["m"]` and a prepare-fixed state flag, with no `.item()`, `.cpu()`, synchronize or tensor prints. Every new synchronize and `.item()` is in `_qkv16_s5_warm` and `_qkv16_s5_prepare` (prepare only; the lint marks them "outside kernel_fn's call graph"). There is no allocation or control flow on input values, and the 5-stage variant is compiled and warmed for m 1..16 before any capture in both states.
- [x] Nothing derived from `data` is cached across calls: state holds only the flag, the ready bool, the smem int and an info string, all computed from synthetic x in prepare.
- [x] Weights: there is no repack or new copy. The same bf16 `state["qkv_gate"]` is read in place, and BF16 stays BF16.
- [x] Activations BF16 where SGLang keeps BF16: the same kernel with bf16 x and output and fp32 accumulation / fp32 split-K partials; only `num_stages` changes.
- [x] n/a: no attention in this problem (dense projections only), so there is no KV cache, query precision or key skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds none. The gate reuses master's `_cell_gate` and torch.cuda events in prepare only.
- [x] All work happens inside the timed call, with no output caching or replay: kernel_fn launches the full GEMM each call. The flag only picks the pipeline depth.
- [x] First line is still `# okc: problem=muse_bf16_proj`. There is no `triton.autotune`. `py_compile` passes and the `__pycache__` was removed.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`), exit 0:

```
warn  host-sync    L1092: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1094: nbad = int((o_base.view(torch.int16) != o_s5.view(torch.int16)).sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1526: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1572: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1581: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1587: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1591: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1620: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1641: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1686: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2086: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2707: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2710: code = int(err.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2739: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2743: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2768: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2782: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2807: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2816: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2819: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2885: code = int(state["bulk_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2916: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2929: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2954: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2965: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2970: algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2976: key = tuple(int(v) for v in algos[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2981: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3028: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3510: lf = frag(lo_enc).to(torch.uint8).reshape(ctas, nch, ns, mt, 256)  (allowed only where SGLang does the same)
warn  dtype        L3512: cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).to(torch.uint8).reshape(ctas, nch, ns, mt, 128)  (allowed only where SGLang does the same)
warn  dtype        L3517: blocks = torch.zeros((ctas, B12_ESC_BYTES), dtype=torch.uint8)  (allowed only where SGLang does the same)
warn  host-sync    L3518: per_cta = torch.bincount((idx[:, 0] // r).cpu(), minlength=ctas) if n_esc else torch.zeros(ctas, dtype=torch.int64)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3523: rr, kk = idx[:, 0].cpu(), idx[:, 1].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3524: vals = bits[idx[:, 0], idx[:, 1]].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3527: for c in torch.nonzero(per_cta).flatten().tolist():  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3568: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3680: code = int(state["b12_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3735: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3874: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3878: nbad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3911: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3963: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4012: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4036: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4249: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4253: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4266: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4306: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_bf16_proj/qkv-swapab16-five-stage-ring-m8-m16/muse_bf16_proj/submission.py
```

## Risks / what could fail
low
Low: one Triton launch parameter behind a flag and a two-m, two-round gate; no new module. If the m <= 16 qkv stream is DRAM-bound rather than bound by bytes in flight, the gate keeps 4 stages and the check answers that. The Do-not-try entries on qkv swap-AB concern deeper split-K or two CTAs per SM; this changes neither (split-K 4 and 2 CTAs/SM as today). Sizing: up to -0.8 us on verify_1 (0.34 %/us) and decode_16 (0.27 %/us) is +0.2-0.5%, below the 1.24% one-run margin alone. Decode-portable: the same knob applies to decode's qkv at T=8/16.

## Result (after the run)
- job `c65ffcb6d9ab4f95` (check), status done, passed True, tests 9/9
- speedup 1.2507x, 95% CI [1.2319, 1.2695], score 3862.5029568699656 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.273478298751914, "ci": [1.2543761242706353, 1.2925804732331925], "job": "bc147c5b87bc465d"}; base result: SGLang (no base result)
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/qkv-swapab16-five-stage-ring-m8-m16.result.json`, `muse_bf16_proj/hypotheses/done/regression/qkv-swapab16-five-stage-ring-m8-m16.patch`; branch `hyp/muse_bf16_proj/qkv-swapab16-five-stage-ring-m8-m16`
