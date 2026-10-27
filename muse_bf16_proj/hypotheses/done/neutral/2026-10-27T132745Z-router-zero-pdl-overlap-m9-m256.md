---
problem: muse_bf16_proj
slug: router-zero-pdl-overlap-m9-m256
created: 2026-10-27T132745Z
base: master
area: router GEMM (m=9..256, zero-fill node overlapped with the router kernel by PDL)
change: >
  Built on master (8f233e88 lineage). At m > 8 the router is two graph nodes: torch.zeros((m,
  256), fp32) (profile: at::native vectorized_elementwise fill, 0.7 us at m=16/32, 0.8 at m=128,
  0.9 at m=256 in a6424247) followed by the Triton router kernel (_router_swapab_kernel at 9 <= m
  <= 32 via _router_forward, _router_hilo_kernel at m > 32 via _router_hilo_forward) whose fp32
  K-partials are tl.atomic_add'ed into the zeroed output. The router kernel's whole K-loop
  (hi/lo/x loads + both tl.dot) does not depend on the zeroed output; only the final atomic_add
  does. CHANGE: (1) new Triton kernel _router_zero_pdl_kernel(out_ptr, numel, BLOCK: tl.constexpr)
  whose FIRST statement is tl.extra.cuda.gdc_launch_dependents(), then a masked tl.store of zeros
  over its BLOCK (grid cdiv(numel, 4096), BLOCK 4096, 4 warps); out becomes torch.empty((m, 256),
  fp32). (2) Both router kernels get a constexpr PDL: tl.constexpr; when PDL is True,
  tl.extra.cuda.gdc_wait() is placed after the K-loop and immediately before tl.atomic_add
  (nothing else moves); the router launch passes launch_pdl=True. PDL=False compiles today's
  kernel byte-identically. So the router CTAs stream their weights and do all MMAs while the zero
  kernel runs, and gdc_wait (primary grid complete + its stores visible) orders the zeros before
  every atomic. Flags router_pdl_ok_<m> for m in (16, 32, 128, 256) in state, read by
  _router_forward/_router_hilo_forward through kernel_fn (cfg/shape/flag-only branching; m=9..15
  maps to the 16 flag, 17..31 to 32, 33..128 to 128, >128 to 256). The m <= 8 cluster path
  (router_kernel<8,8>, no memset) is untouched. Prepare, state A, new section after
  _router_prepare: capability probe (hasattr(tl.extra.cuda, 'gdc_wait') and
  'gdc_launch_dependents'; the launch accepts launch_pdl; compiled PTX of both kernels contains
  griddepcontrol.wait / griddepcontrol.launch_dependents) else print why and keep master; per m:
  _cell_check-style correctness vs an fp64 reference (rel L2 <= AB_ROUTER_REL_L2 5e-5, the
  power-of-two probe row) AND a race check: capture kernel_fn once, replay 32 times with refilled
  x, every replay vs master within 5e-5 (a zero landing after an atomic doubles or drops a
  partial: errors of order 1, never 1e-5); then two read-flush _cell_gate rounds
  (flush_mode='read', the khalf rule: master/cand >= 1.03 in both rounds and cand IQR <= 1.5x
  max(master IQR, 0.5 us)); then _cell_kernels asserting exactly [_router_zero_pdl_kernel, the
  router kernel] and no vectorized_elementwise. State B inherits the flags; AB_CASES gains router
  m=16/32/128/256 mapped to their flags. One PATH CHOSEN R-PDL line per m with both rounds' net
  us, re-printed at the end of prepare (16 KB tail).
expected:
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: "the 0.7 us zero-fill node runs under the router K-loop instead of before it: row 5.48 -> ~5.0 us (-6..-9%, above the 1.7-3.6% drift)"
  - shapes: layer=router load=decode_32 m=32
    direction: faster
    why: "same: 5.95 -> ~5.4 us"
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "0.8 us fill hidden under the hilo K-loop: 7.95 -> ~7.3 us"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "0.9 us fill hidden: 9.51 -> ~8.8 us"
  - shapes: layer=router load=*_1 m=*
    direction: flat
    why: m <= 8 keeps the cluster kernel (no memset)
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "router-splitk-semaphore-no-memset: that deleted the memset with a ticket-semaphore last-arriver reduce, a serialized epilogue that DISTILL 3.1 measured x1.07-1.10 slower than the memset under graph replay; this keeps the memset and the relaxed atomics exactly and only lets the router kernel's weight stream and MMAs run while the zero kernel executes (PDL), with no new reduce or serialization"
review: proceed
status: done
review_at: 2026-10-27T141019Z
review_companions: null
stage: harvested
claimed_at: 2026-10-27T141019Z
base_sha: 93f6f7fa2f41100d0f75cf209e2ff5e6c65f97e7
base_blob: d51c745a0e43d20ebf2cbb3d582f3f8ead72f039
branch: hyp/muse_bf16_proj/router-zero-pdl-overlap-m9-m256
worktree: .worktrees/muse_bf16_proj/router-zero-pdl-overlap-m9-m256
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-27T142100Z
impl_commit: 5cacba603e5067efe55ae157d43fbebef3fb1895
submission_sha256: 6c4e693c2acfbe8e089941aae9bd3e09844d48094750096c98e060409a3a8a14
norm_sha256: 16d343d98cd37482b1ab9e64a3ea0283525eeb3f80eb3da5a1efc315a56c1275
submission_blob: af2b30819ebb4235c69ef00ac9c6b294a5eaf5ca
lint_warnings: 54
attempt_file: 2026-10-27T142101Z-check-router-zero-pdl-overlap-m9-m256.md
prepared_at: 2026-10-27T142101Z
job: a20f80486c5b4bc1
dispatched_at: 2026-10-27T142114Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.271476551021886
speedup_ci: [1.2524044027565577, 1.2905486992872142]
harvested_at: 2026-10-27T142518Z
best_at_harvest: 1.273478298751914
promotion: null
---

# router-zero-pdl-overlap-m9-m256

## Change

Built on master (8f233e88 lineage). At m > 8 the router is two graph nodes: torch.zeros((m, 256), fp32) (profile: at::native vectorized_elementwise fill, 0.7 us at m=16/32, 0.8 at m=128, 0.9 at m=256 in a6424247) followed by the Triton router kernel (_router_swapab_kernel at 9 <= m <= 32 via _router_forward, _router_hilo_kernel at m > 32 via _router_hilo_forward) whose fp32 K-partials are tl.atomic_add'ed into the zeroed output. The router kernel's whole K-loop (hi/lo/x loads + both tl.dot) does not depend on the zeroed output; only the final atomic_add does. CHANGE: (1) new Triton kernel _router_zero_pdl_kernel(out_ptr, numel, BLOCK: tl.constexpr) whose FIRST statement is tl.extra.cuda.gdc_launch_dependents(), then a masked tl.store of zeros over its BLOCK (grid cdiv(numel, 4096), BLOCK 4096, 4 warps); out becomes torch.empty((m, 256), fp32). (2) Both router kernels get a constexpr PDL: tl.constexpr; when PDL is True, tl.extra.cuda.gdc_wait() is placed after the K-loop and immediately before tl.atomic_add (nothing else moves); the router launch passes launch_pdl=True. PDL=False compiles today's kernel byte-identically. So the router CTAs stream their weights and do all MMAs while the zero kernel runs, and gdc_wait (primary grid complete + its stores visible) orders the zeros before every atomic. Flags router_pdl_ok_<m> for m in (16, 32, 128, 256) in state, read by _router_forward/_router_hilo_forward through kernel_fn (cfg/shape/flag-only branching; m=9..15 maps to the 16 flag, 17..31 to 32, 33..128 to 128, >128 to 256). The m <= 8 cluster path (router_kernel<8,8>, no memset) is untouched. Prepare, state A, new section after _router_prepare: capability probe (hasattr(tl.extra.cuda, 'gdc_wait') and 'gdc_launch_dependents'; the launch accepts launch_pdl; compiled PTX of both kernels contains griddepcontrol.wait / griddepcontrol.launch_dependents) else print why and keep master; per m: _cell_check-style correctness vs an fp64 reference (rel L2 <= AB_ROUTER_REL_L2 5e-5, the power-of-two probe row) AND a race check: capture kernel_fn once, replay 32 times with refilled x, every replay vs master within 5e-5 (a zero landing after an atomic doubles or drops a partial: errors of order 1, never 1e-5); then two read-flush _cell_gate rounds (flush_mode='read', the khalf rule: master/cand >= 1.03 in both rounds and cand IQR <= 1.5x max(master IQR, 0.5 us)); then _cell_kernels asserting exactly [_router_zero_pdl_kernel, the router kernel] and no vectorized_elementwise. State B inherits the flags; AB_CASES gains router m=16/32/128/256 mapped to their flags. One PATH CHOSEN R-PDL line per m with both rounds' net us, re-printed at the end of prepare (16 KB tail).

## Expected

- `layer=router load=decode_16 m=16` faster: the 0.7 us zero-fill node runs under the router K-loop instead of before it: row 5.48 -> ~5.0 us (-6..-9%, above the 1.7-3.6% drift)
- `layer=router load=decode_32 m=32` faster: same: 5.95 -> ~5.4 us
- `layer=router load=verify_16 m=128` faster: 0.8 us fill hidden under the hilo K-loop: 7.95 -> ~7.3 us
- `layer=router load=verify_32 m=256` faster: 0.9 us fill hidden: 9.51 -> ~8.8 us
- `layer=router load=*_1 m=*` flat: m <= 8 keeps the cluster kernel (no memset)
- `layer=qkv_gate *` flat: untouched

## Risk

Headline ~+0.3-0.4% (four router rows x ~0.5 us; 0.18-0.23 %/us on the verify loads, ~0.6% of each decode_16/32 load): sub-margin alone, so judge it on the four router rows and the R-PDL gate lines, then stack. Risk 1 (VOID): the judge's Triton lacks tl.extra.cuda.gdc_wait / launch_pdl (added in Triton 3.4-era; torch 2.14.1+cu130 should ship newer). Before dispatch, grep a recent judge log or the sglang env for the Triton version; the probe prints it and keeps master otherwise. Risk 2: a programmatic edge must survive CUDA-graph capture (CUDA 12.3+ captures PDL edges; the judge is cu130); the race check catches a lost edge as large errors. Risk 3: gdc_wait must sit after the loop: if Triton hoists or sinks it, the PTX order check (griddepcontrol.wait after the last wgmma, before the red/atom) is the guard. Decode-portable: decode's T=128 'hilo router' cell (W6-B) is the same torch.zeros + _router_hilo_kernel pair. Do not combine with router-hilo-m128-m256-readflush-config-sweep in one check (two routable novelties on the same rows, DISTILL 3.2.7).

## Adversarial review

Reviewed 2026-10-27T141019Z by an adversarial subagent (model opus).

**Verdict: proceed.** Plausible and a different lever from every earlier router attempt: it overlaps the graph's memset node with the router kernel (earlier attempts tuned the kernel or replaced the memset with a ticket epilogue). Deleting the memset at m<=8 saved 0.40-0.54 us, and PDL paid between our own kernels in decode. The in-prepare interleaved read-flush gate gives master vs candidate on identical surroundings; no companion. Tighten two guards: the PTX check must count exactly one griddepcontrol.wait, and the race check needs a slow-primary stand-in.

### Justification
Router rows at m=16/32/128/256 (5.48/5.95/7.95/9.51 us vs floors 1.8-2.7 us) are a fill node (0.7-0.9 us) plus the Triton split-K kernel with atomics. The m<=8 memset deletion (~0.4-0.5 us) is the realistic ceiling; m=128/256 predictions are ~0.2 us optimistic; a realistic 5-6% still clears drift and the 1.03 gate. Headline ~+0.3%, unreadable: judge on the four router rows and the R-PDL gate lines, not profile top_kernels (PDL-inflated rows). Prior router-splitk-semaphore-no-memset serialized the reduce; this keeps the relaxed atomics and the memset. Weakest link: whether Triton 3.8's launch_pdl under graph capture produces a real programmatic edge, and whether Triton auto-inserts a griddepcontrol.wait at entry; either gives a misleading flat result. Assert exactly one griddepcontrol.wait per router kernel (after the last mma, before the red/atom) and exactly one launch_dependents in the zero kernel at entry.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32: sets the K-loop available to hide the fill; a precondition shared by both arms.
- router-triton-hilo-fused-control-splitk-all-m: the split-K atomic + zeros design; a precondition.
- restack-three-cell-wins (router_kernel<8,8>, no memset at m<=8): m<=8 rows stay flat.
- Queued router-hilo-m128-m256-readflush-config-sweep: same rows; keep separate checks; whichever lands second re-gates.

### Interference and A/B needs
No companion. Add a third gate arm (Triton zero kernel + router PDL=False, no launch_pdl) and print its net us beside cand/master, to separate the fill change from PDL.

### Open questions
- Does Triton 3.8 insert griddepcontrol.wait at entry with launch_pdl=True? Count waits and check position.
- Race check: add a stand-in primary that spins ~10 us before storing zeros; a debug variant without the wait must fail.
- Does graph capture keep the programmatic edge on the judge (cu130)? Print the edge type if cheap.
- At m=256: 16 zero CTAs + 128 router CTAs at ~144 KB smem; confirm no placement delay.
- Check both rounds' IQRs before trusting a marginal 1.03 pass.


## Implementation notes

Implemented 2026-10-27T141953Z in the worktree's `muse_bf16_proj/submission.py` only (uncommitted; base 93f6f7fa). Diff: 1 file, +482 / -39.

- **Kernels.** `_router_hilo_kernel` and `_router_swapab_kernel` get `PDL: tl.constexpr`. With `PDL` true there is one `tl.extra.cuda.gdc_wait()` after the K-loop, right before `tl.atomic_add`; nothing else moves. Every launch passes `PDL`, and `PDL=False` with no `launch_pdl` is master's kernel. New `_router_zero_pdl_kernel(out_ptr, numel, BLOCK)`: `gdc_launch_dependents()` first, then a masked zero store, grid cdiv(numel, 4096), BLOCK 4096, 4 warps. New prepare-only `_router_zero_slow_kernel`: the same trigger, about 20000 cycles of `%clock64` spin (roughly 10 us), then the same stores. kernel_fn never launches it.
- **Launch / forward.** New helpers `_router_swapab_launch` / `_router_hilo_launch` take `pdl` and `launch_pdl`. The routed forwards always pass them equal, `zmode == 1`. `_router_out` gives `torch.zeros` (zmode 0, master) or `torch.empty` + the zero kernel (zmode 1 or 2). `_router_forward` / `_router_hilo_forward` take `zmode=0` by default, so `_router_prepare`'s calls are unchanged. kernel_fn computes `zmode = _router_zmode(m, state)` from the class flag, which depends only on shape and state. Flags: `router_pdl_ok_{16,32,128,256}`. m <= 8 maps to no flag, so the cluster path and master's swap-AB fallback stay untouched. m 9..16 maps to the 16 flag, 17..32 to 32, 33..128 to 128, and > 128 to 256. Mode 2 (`ROUTER_ZMODE_NOPDL`: zero kernel, PDL off, no launch_pdl) is set only on `_cell_gate`'s third-arm copy of state.
- **Probe + warm (`_router_pdl_probe`, state A, right after `_router_prepare`, before any capture).**
  - Prints the Triton version. Feature-detects `gdc_wait` / `gdc_launch_dependents`, and prints whether `CUDAOptions` has `launch_pdl` (information only; the launch itself is the real test).
  - Per class m (16 / 32 / 128 / 256), one announced, synced launch each of the zero kernel, the PDL router (launch_pdl=True) and master's variant.
  - PTX checks (`_router_pdl_ptx_why`):
    - zero kernel: exactly one `launch_dependents`, before the first `st.global`, and no wait.
    - each PDL router variant: EXACTLY ONE `griddepcontrol.wait`, after the last `wgmma.mma_async` / `mma.sync` and before the first line-start `red.` / `atom.`. An auto-inserted entry wait makes the count 2 and fails the check.
    - master variant: no griddepcontrol at all.
  - Any miss or exception sets `state["router_pdl_cap"]` to the reason, prints it, and keeps master for every class.
  - Then `_router_pdl_warm` runs m in (9, 16, 32, 64, 128, 256) through `_router_forward` / `_router_hilo_forward` (kernel_fn's exact kwargs) in zmodes 0, 1 and 2. ROUTER_WARM_MS m <= 8 never take a flag. State B, when it inherits any flag on, warms zmodes 0 and 1 before the A/B captures.
- **Selector (`_router_pdl_prepare`, state A, new step (8b) right after the router cluster prepare).** It cannot sit literally right after `_router_prepare`: kernel_fn needs `swapab_stages` / `swapab16_stages` / `router_cluster_ok` first. Per class:
  - (a) `_cell_check` through kernel_fn at m (9, 16) / (32) / (64, 128) / (256), with new optional args `rel_bound=AB_ROUTER_REL_L2` (5e-5) and `repeat_rel=5e-5`. The bitwise repeat check becomes a tolerance compare (relaxed fp32 atomics), and the power-of-two probe row stays <= 1e-4. Defaults leave every earlier caller unchanged.
  - (b) `_router_pdl_race`: graphs on one x buffer for master (flag off), the candidate (kernel_fn, flag on), the slow-primary stand-in (spin kernel, then the PDL router with its wait, launched the same way) and a no-wait control (stand-in, then master's router body with launch_pdl=True). 32 rounds of refilled x. Candidate and stand-in must stay within 5e-5 of master on every replay. The no-wait control's wrong-replay count is printed only, never routed. When it is 0, the line says the race was not observable (no early launch?).
  - (c) Two `_cell_gate` rounds, flush_mode `B12_GATE_FLUSH` ("read"), rule master/cand >= 1.03 in both rounds and cand IQR <= 1.5x max(master IQR, 0.5 us). New optional `_cell_gate(extra=...)` adds the third arm "nopdl" (zero kernel + PDL off), interleaved in rotation and printed with its net us beside cand/master. Routing still reads cand vs master only. `extra=None` keeps the two-arm loop exactly.
  - (d) `_cell_kernels` must see exactly one `_router_zero_pdl_kernel` and one `_router_swapab_kernel` / `_router_hilo_kernel`, and no `vectorized_elementwise` / fill (order-insensitive).
  - One compact `PATH CHOSEN R-PDL router m=<b>` `_decision` line per class: both rounds' ratio, net master/cand/nopdl us, the three IQRs, the race summary and why not. These lines are re-printed in the end-of-prepare DECISIONS block. If the probe fails, a single line covers all classes.
- **A/B.** `NEW_CELL_FLAGS += ROUTER_PDL_FLAGS`, so the flags join A/B inheritance, the failure-path reset and FINAL FLAGS. `AB_CASES` gains router m = 16 / 32 / 256 (128 was already there). `_ab_component` maps a router m > 8 cell to its class flag when that flag is on, so a miss switches off only that class.
- **Deviations / notes.**
  - Section placement is (8b), not literally right after `_router_prepare`, for the reason given above. Only the probe and warm-up sit there.
  - Correctness also covers m = 9 (16 class) and m = 64 (128 class); the gates run at 16 / 32 / 128 / 256 only.
  - The no-wait negative control reuses master's kernel body with launch_pdl=True instead of adding a no-wait variant to the routed kernels, so nothing debug-only is in the routed code.
  - Edge type under capture is not printed (no cheap Python API). The race check (real edge, wait honoured) and the third arm (overlap vs fill change) stand in for it.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/router-zero-pdl-overlap-m9-m256 diff`:

- [x] CUDA-graph capturable: kernel_fn's new code is `_router_zmode` (a dict lookup on shape and state flags), `torch.empty` and two Triton launches (one with launch_pdl). No `.item()`, `.cpu()`, synchronize or tensor prints; every sync and `.item()` the diff adds is in prepare-only helpers (lint: all host-sync warnings are outside kernel_fn's call graph). No control flow on input values.
- [x] Nothing derived from `data` is cached across calls: the output is allocated per call; state holds only bool flags and the prepare-time capability string.
- [x] Weights: untouched. The router still uses master's bf16 hi + lo split with fp32 accumulation, and no new weight copies exist.
- [x] Activations BF16 where SGLang keeps BF16: x stays bf16, and the router output stays fp32 as before.
- [x] n/a: no attention in this problem (dense projections only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The only new import is stdlib `re` (PTX text check). `triton.backends.nvidia.compiler.CUDAOptions` is read, never modified.
- [x] All work happens inside the timed call: the zero fill and the router GEMM run every call. No output caching or replay; the race and gate graphs are prepare-only and discarded.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`): **OK with warnings**. 0 errors. The 54 warnings are all `host-sync ... (outside kernel_fn's call graph; fine in prepare())` or the existing uint8 flush-buffer `dtype` notes. New ones point at the R-PDL prepare helpers' synchronize calls (L554-L750). `py_compile` passed, and no `__pycache__` is left in the worktree.

## Result

- job `a20f80486c5b4bc1` (check), status done, passed True, tests 9/9
- speedup 1.2715x, 95% CI [1.2524, 1.2905], score 3799.362242361045 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.273478298751914, "ci": [1.2543761242706353, 1.2925804732331925], "job": "bc147c5b87bc465d"}; base result: SGLang (no base result)
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-zero-pdl-overlap-m9-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-zero-pdl-overlap-m9-m256.patch`; branch `hyp/muse_bf16_proj/router-zero-pdl-overlap-m9-m256`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 13.92 | 10.64 | 1.3086 | gained |
| layer=o load=decode_1 m=1 | 62 | 6.56 | 5.65 | 1.1613 | gained |
| layer=pooler load=decode_1 m=1 | 62 | 21.82 | 14.94 | 1.4609 | gained |
| layer=unpooler load=decode_1 m=1 | 62 | 7.04 | 5.1 | 1.3803 | gained |
| layer=router load=decode_1 m=1 | 62 | 10.5 | 4.97 | 2.1112 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.08 | 13.16 | 1.0699 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.86 | 6.19 | 1.1096 | gained |
| layer=pooler load=decode_16 m=16 | 62 | 21.47 | 19.11 | 1.1233 | gained |
| layer=unpooler load=decode_16 m=16 | 62 | 18.35 | 18.36 | 0.9997 | flat |
| layer=router load=decode_16 m=16 | 62 | 14.78 | 5.27 | 2.8049 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.34 | 14.48 | 0.9901 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.94 | 6.71 | 1.0347 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 21.9 | 19.83 | 1.1048 | gained |
| layer=unpooler load=decode_32 m=32 | 62 | 18.8 | 18.51 | 1.0157 | flat |
| layer=router load=decode_32 m=32 | 62 | 15.49 | 5.96 | 2.598 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.92 | 12.75 | 1.0916 | gained |
| layer=o load=verify_1 m=8 | 62 | 6.94 | 6.43 | 1.0796 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 21.31 | 19.0 | 1.1218 | gained |
| layer=unpooler load=verify_1 m=8 | 62 | 6.85 | 5.34 | 1.2834 | gained |
| layer=router load=verify_1 m=8 | 62 | 26.05 | 5.06 | 5.1526 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.54 | 15.57 | 1.0626 | gained |
| layer=o load=verify_16 m=128 | 62 | 7.49 | 7.5 | 0.9988 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.22 | 23.11 | 1.0912 | gained |
| layer=unpooler load=verify_16 m=128 | 62 | 19.98 | 19.21 | 1.0401 | gained |
| layer=router load=verify_16 m=128 | 62 | 16.37 | 7.06 | 2.32 | gained |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.52 | 19.3 | 1.0114 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.04 | 8.98 | 1.0072 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.86 | 29.77 | 1.0367 | gained |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.26 | 1.0063 | flat |
| layer=router load=verify_32 m=256 | 62 | 17.5 | 9.06 | 1.9326 | gained |

gained on qkv_gate decode_1, decode_16, verify_1, verify_16; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; unpooler decode_1, verify_1, verify_16; router decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-zero-pdl-overlap-m9-m256`) that adds a further change in the same area is the only way to learn more.

