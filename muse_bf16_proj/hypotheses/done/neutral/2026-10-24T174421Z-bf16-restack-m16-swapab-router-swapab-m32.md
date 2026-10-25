---
problem: muse_bf16_proj
slug: bf16-restack-m16-swapab-router-swapab-m32
created: 2026-10-24T174421Z
base: master
area: restack (m<=16 swap-AB GEMMs + router swap-AB)
change: >
  Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a
  3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16
  kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3.
  Its patch is in done/neutral/bf16-m16-swapab-wgmma-extend.patch on branch
  hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend. (2) router-swapab-wgmma-all-m, but gated to
  m<=32 only. Above that, keep master's BLOCK_M=64 hilo router kernel, which already emits wgmma
  (the m=128/256 rows read -2%/flat). Patch on branch
  hyp/muse_bf16_proj/router-swapab-wgmma-all-m. The harvest analysis found the merge-file of the
  two clean. Do NOT include bf16-cublaslt-algo-pick (it lost at m=256). No other change.
expected:
  - shapes: layer=qkv_gate load=* m=1
    direction: faster
    why: m16 swap-AB, qkv 18% faster kernel
  - shapes: layer=o load=* m=8
    direction: faster
    why: m16 swap-AB
  - shapes: layer=router load=* m=32
    direction: faster
    why: router swap-AB +3-5% at m<=32
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T174842Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T174842Z
base_sha: 4a04b3757546365d2227c1fc1762bc6c5072e92b
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32
worktree: .worktrees/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32
mode: submit
implemented_at: 2026-10-24T175227Z
impl_commit: 8f97b729cc67149da5ac481e415f4402b9f1c88e
submission_sha256: cda00e0118587bcdc1b3eed567f1677202b348f7cf16d9a96d931e0527e443cc
norm_sha256: 5cf9fc70e2475e57ee93028771a3d36050cbb41a5e6635e39317bd202723a0e0
submission_blob: 970b415b746fa03cb92e419d270be92588273178
lint_warnings: 0
attempt_file: 2026-10-24T175228Z-submit-bf16-restack-m16-swapab-router-swapab-m32.md
prepared_at: 2026-10-24T175228Z
job: 1e90292c7b954d96
dispatched_at: 2026-10-24T175236Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2215006063849365
speedup_ci: [1.2031780972891624, 1.2398231154807104]
harvested_at: 2026-10-24T184717Z
best_at_harvest: 1.1958517819004566
promotion: done
promotion_job: 60b20fd0bf924267
promotion_at: 2026-10-25T000254Z
board: "{\"rank\": 320, \"team\": \"yesthatwilly\", \"speedup\": 1.2215, \"first\": false, \"ahead\": 319, \"to_first\": 1.318}"
---

# bf16-restack-m16-swapab-router-swapab-m32

## Change

Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16 kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3. Its patch is in done/neutral/bf16-m16-swapab-wgmma-extend.patch on branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend. (2) router-swapab-wgmma-all-m, but gated to m<=32 only. Above that, keep master's BLOCK_M=64 hilo router kernel, which already emits wgmma (the m=128/256 rows read -2%/flat). Patch on branch hyp/muse_bf16_proj/router-swapab-wgmma-all-m. The harvest analysis found the merge-file of the two clean. Do NOT include bf16-cublaslt-algo-pick (it lost at m=256). No other change.

## Expected

- `layer=qkv_gate load=* m=1` faster: m16 swap-AB, qkv 18% faster kernel
- `layer=o load=* m=8` faster: m16 swap-AB
- `layer=router load=* m=32` faster: router swap-AB +3-5% at m<=32
- `layer=* load=verify_32 m=256` flat: untouched

## Risk

Restack of measured pieces; expected headline about 1.215-1.222 vs 1.196 (harvest analysis). Run as a scored experiment; if it gains, it is the board entry.

## Adversarial review

Reviewed 2026-10-24T174842Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. The two pieces cover separate shapes and both were measured on the same master blob (a8bb42a). I re-ran git merge-file and it is clean and parses. The m<=32 router gate matches the PTX evidence. The expected headline is credible at about 1.21-1.22, not the top of 1.215-1.222. Nearly all of the gain is the m16 piece. The router piece adds about 0.2-0.4%, which is below noise.

### Justification
Mechanism: m16 swap-AB puts 64 weight rows on the wgmma M side at m<=16. Its prepare log shows wgmma replacing mma.sync for all four keys, with fewer registers and no spills. Kernel profile vs the master average (submit eeb99 + check 1602): qkv_gate 11.55/12.25/11.8 -> 9.4/10.1/9.9 us at m=1/16/8, o 4.7/4.9/5.15 -> 4.1/4.4/4.2, unpooler slice 4.45/4.75 -> 3.9/4.3. The cold-L2 shape medians move much less than these kernel times: qkv_gate d1 13.0 -> 12.46, v1 13.18 -> 12.67, d16 13.5 -> 13.28 us. Only part of the profiled 2 us qkv saving survives the timed method, likely because 37.7 MB is partly L2-resident when profiled. Router piece: the log shows master hilo at m<=32 is mma.sync (16-64) and the swap-AB is wgmma (16). Kernel time is flat at m=1 and m=8 (4.0/3.95 -> 4.0/3.9 us) and improves only at m=16 (4.25 -> 4.0) and m=32 (4.75 -> 4.3). Gate: at m>=64, master hilo BLOCK_M=64 and swap-AB compile to identical counts (regs 127/128, wgmma 16, red 72, cp.async 83). The m=128/256 router rows (kernel 5.0->5.2, 6.9->7.2) are noise or slight loss on the same code, so m<=32 is the right gate. Keeping hilo above 32 costs nothing and removes a variable. Headline, three estimates, all 30 shapes with load geomean: (a) splicing per-shape speedups (router m<=32 from 587e34, non-router m<=16 from 89fabe, master average elsewhere) gives 1.2229. (b) Router rows spliced onto the m16 run gives 1.2209. (c) Profile kernel deltas applied to master-average medians gives +3.1% (master 1.190 -> 1.227 in that frame, from m16 +2.9%, router +0.2%). (a) and (b) are biased upward: both component runs were harvested as winners, and rows are spliced from each. Untouched shapes swing +/-3-6% between runs. Examples: router d16 is 2.334 in the m16 run vs 2.44-2.51 on master though m16 never touches the router; o v1 is 1.065 in the router run vs 1.017/1.048 on master though the router run never touches o. The router run's largest row, v1 m=8 router 4.99 vs 4.43-4.48, has no kernel-time support (3.95 -> 3.9 us), so it is noise that inflates the router piece's share of the harvest estimate. Realistic outcome: about +1.2-2.2% over 1.196, so roughly 1.21-1.22, landing near or just above BEST's CI upper edge of 1.2138. Expected directions: qkv_gate m=1 faster and o m=8 faster are solid (o v1 kernel 5.15 -> 4.2). router m=32 faster is right in sign, but 0.45 us on about 6.3 us is at the shape's noise level, so it may not read; router decode_16 is the other honest one. 'verify_32 m=256 flat' is correct, and the restack also leaves verify_16 m=128 alone (hilo is kept there). Weakest link: the router half. Its headline value (+0.2-0.4%) is below the 1.5% CI and its per-shape signal is near noise. In the merge it is only a liability if the gate is implemented wrong (see open_questions). It also changes two things at once, orientation (mma.sync->wgmma) and SPLIT_K 16->32, which is acceptable for a restack.

### Confounders on the current kernel
- No landed hypotheses are listed (landed: []). Both component patches were built on the same master blob a8bb42a (checked: e82b94f, 278c444 and 8b9ad43 all resolve to it), so each measured delta is against exactly this base.
- Shared code between the pieces: none on the timed path. The m16 piece routes first in kernel_fn, and _swapab16_key('router') returns '' so the router falls through untouched. The router piece changes only the router branch and _router_config/_router_forward. They meet only in the docstring, prepare() (different insertion points) and two near-duplicate diagnostic helpers (_ptx_counts vs _print_ptx_counts, different names, prepare-only). git merge-file of bb41afb and 64894c9 over a8bb42a exits 0 and the result parses.
- Shared shapes: none. The m16 piece covers pooler/qkv_gate/o at m<=16 and the unpooler n=1024 slice. The router piece covers the router at m<=32. Each shape is its own CUDA graph with a cold L2, so the only coupling is the per-load sum and geomean plus run-to-run noise.
- Indirect: master's m17..32 swap-AB (bf16-m32-swapab-wgmma-triton) uses the same _bf16_swapab_kernel. The m16 piece compiles separate constexpr variants (BLOCK_T=WS_M=16) with its own workspaces and tickets, so neither can perturb the other's buffers or counters.

### Interference and A/B needs
No companion is warranted. Each sub-change was already measured alone on this exact base: m16 alone is 89fabef70bd745f7 at 1.2115, router all-m alone is 587e340476944b46 at 1.1951. The okc dedupe returns cached results for identical code, so an m16-only or router-only control would add nothing. The pieces share no timed code and no shapes, so a gain reads as the sum of known parts. A flat or loss result also reads without a control: compare the touched rows to the two component archives. If m<=16 non-router rows hold their gains and only noise moves, it is noise. If router m>=128 rows regress by several us, the gate fell into the F.linear fallback (see open_questions). If m16 rows regress, the merge broke something. An ab-tune of the router gate (m<=16 vs m<=32) would be a test below noise and is not worth a run.

### Open questions
- Gate implementation trap: in the router patch, kernel_fn requires `_router_config(m) in state['router_stages']` and otherwise falls through to `F.linear(x.float(), router)`. The patch's _router_config returns (BLOCK_T, SPLIT_K) for every m. If the restack makes _router_config return None (or anything not in router_stages) for m>32 without a separate hilo path, m=128/256 run the fp32 F.linear path, which is correct but much slower. That would be a silent regression the 9 tests would not catch. The implementer should have kernel_fn send m<=32 to _router_forward (swapab) and m>32 to master's original hilo forward with _router_hilo_config and ROUTER_NUM_STAGES. They should warm hilo at m=64/128/256 in prepare() and keep a prepare-time assert that every ROUTER_WARM_MS bucket resolves to a kernel path, not the fallback.
- _router_shape_ok must take SPLIT_K from the config of the path actually used: 32 for swapab at m<=32, 16/8 for hilo above. Both divide 8192/128, so this is only a correctness-of-code note.
- Gate must be inclusive (m<=32) so decode_32 router (the expected row) takes swap-AB. Router tests at m=8 and m=9 (outliers) both exercise the swap-AB path; no test exercises the router at m in 33..127.
- Consider dropping the prepare-only _router_hilo_report diagnostic and one of the duplicate PTX-count helpers. They are harmless, but prepare already went from 2.45 s to about 3.9 s in each component run.
- Set the board expectation at about 1.21-1.22. Expect a submit that lands inside BEST's CI more often than clearly above it. If it reads between 1.205 and 1.214, it is still the best point estimate on master's lineage.


## Implementation notes

- Base: worktree submission.py at master blob a8bb42a. `git merge-file` of bb41afb (bf16-m16-swapab-wgmma-extend) and 64894c9 (router-swapab-wgmma-all-m) over a8bb42a exited 0 (clean); that result is the starting point. Outside the router region the file equals bb41afb: the diff vs bb41afb touches only the module docstring's router paragraph, the ROUTER_* constants and _router_* functions, the one `_router_prepare(state)` call in prepare() and the router tail of kernel_fn.
- Gate (inclusive, `ROUTER_SWAPAB_MAX_M = 32`): new `_router_path(x, hi, n, stages)` (shapes only) returns "swapab" for m <= 32 (bucket present in state["router_stages"]), "hilo" for m > 32, "" otherwise. kernel_fn sends "swapab" to the swap-AB `_router_forward`, "hilo" to `_router_hilo_forward`, and only "" to the F.linear fp32 fallback.
- `_router_hilo_forward` is master's `_router_forward` verbatim, with only its name and its config call renamed (`_router_config` -> `_router_hilo_config`). `_router_hilo_config` is master's `_router_config` logic verbatim (docstring changed only). `_router_hilo_kernel`, ROUTER_NUM_STAGES = 3 and ROUTER_NUM_WARPS = 4 are unchanged. Verified with an AST source diff against a8bb42a: forward and kernel identical modulo rename, config identical except the docstring. So for m > 32 the launched path (config, grid, kernel, stages, warps, zeroed output) is master's.
- `_router_config` (swap-AB) trimmed to its two m <= 32 buckets: (16, 32) at m <= 16, (32, 32) at m <= 32. The patch's m > 32 rows were dropped since they are never used.
- `_router_shape_ok` takes SPLIT_K (and the N tile) from the path actually used: swap-AB SPLIT_K 32 / BLOCK_W 64 at m <= 32; `_router_hilo_config` SPLIT_K 16/8 / BLOCK_N at m > 32 (exactly master's check).
- `_router_prepare`: m in ROUTER_WARM_MS <= 32 (1, 8, 16, 32) choose stages per swap-AB bucket and warm as in the patch; m = 64/128/256 warm master's hilo forward. After warm-up, for the problem's [256, 8192] router, it asserts every ROUTER_WARM_MS m resolves to "swapab" or "hilo" (never the fallback) and prints `[router] m=<m>: <path>`.
- Dropped `_router_hilo_report` (prepare-only diagnostic) and the duplicate `_ptx_counts` helper; the router report line now uses the m16 piece's `_print_ptx_counts(tag, compiled)` (loses red/atom counts in that log line; diagnostic only).
- Deviation from spec: none in substance. Diff stat vs master: 1 file, +364 / -23.

## Rules check

- [x] CUDA-graph capturable: router routing in kernel_fn branches on x.shape / cfg["n"] / the prepare-time stage dict only; no .item()/.cpu()/synchronize()/prints in kernel_fn (prints and the assert are prepare-only). The m16 piece is unchanged from its measured patch (routes on cfg and shapes).
- [x] Nothing derived from `data` is cached across calls: state only gains swapab16 workspaces/tickets/stages and router_stages, all built in prepare() from weights and synthetic x.
- [x] Weights: router as bf16 hi + lo with fp32 accumulation (as master); bf16 weights read in place as bf16.
- [x] Activations BF16 where SGLang keeps BF16 (bf16 x in, bf16 out for the four layers, fp32 router out). KV cache / queries: n/a, no attention in this problem.
- [x] Attention sees every key: n/a, no attention in this problem (dense projections only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds only Triton launches, torch allocations, prepare-time prints and an assert.
- [x] All work happens inside the timed call; no output caching or replay (each call zeroes or allocates its output and launches).
- [x] First line still `# okc: problem=muse_bf16_proj`; no triton.autotune.

okc_lint output:
```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32/muse_bf16_proj/submission.py
```
`python3 -I -m py_compile` passed; __pycache__ removed from the worktree.

## Result

- job `1e90292c7b954d96` (submit), status done, passed True, tests 9/9
- speedup 1.2215x, 95% CI [1.2032, 1.2398], score 3954.807696982551 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.patch`; branch `hyp/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.91 | 12.18 | 1.0275 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.32 | 5.89 | 1.0675 | gained |
| layer=pooler load=decode_1 m=1 | 62 | 19.0 | 18.84 | 1.0206 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.94 | 6.15 | 1.0356 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.73 | 5.38 | 1.0812 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.39 | 13.15 | 1.0286 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.66 | 6.23 | 1.0783 | gained |
| layer=pooler load=decode_16 m=16 | 62 | 19.85 | 19.47 | 1.017 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.61 | 18.51 | 1.0102 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.03 | 5.47 | 1.1108 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.69 | 14.19 | 0.99 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.67 | 6.34 | 1.035 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 20.0 | 19.73 | 1.0047 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.48 | 18.08 | 1.0202 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.55 | 6.2 | 1.0805 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.18 | 12.5 | 1.019 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.32 | 6.13 | 1.045 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 19.14 | 18.89 | 1.0124 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.13 | 6.0 | 1.0434 | gained |
| layer=router load=verify_1 m=8 | 62 | 5.91 | 5.47 | 1.0848 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.81 | 17.05 | 0.9934 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.09 | 6.98 | 0.9993 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.55 | 25.0 | 1.0216 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.79 | 19.96 | 1.0003 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.35 | 7.28 | 1.0091 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.62 | 19.69 | 0.9823 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.01 | 9.26 | 1.0046 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.53 | 31.19 | 0.9822 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.5 | 24.42 | 1.0003 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.3 | 9.6 | 0.9848 | flat |

gained on o decode_1, decode_16, decode_32, verify_1; unpooler decode_1, verify_1; router decode_1, decode_16, decode_32, verify_1. Expected hit: 3/4. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-restack-m16-swapab-router-swapab-m32`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

