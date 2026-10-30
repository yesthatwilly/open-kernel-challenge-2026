# muse_moe_w4a16 · 2026-10-30T005158Z · submit · engine-a-xl-arms-t256-postnorm-stack

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-30T002759Z-engine-a-xl-arms-t256-postnorm-stack.md` (branch `hyp/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `eb21652b`

## Goal
Promote check-proven gain engine-a-xl-arms-t256-postnorm-stack (2.0667x [2.0357,2.0977], check job 73156455c9f242f6) to the board.

Stack two shelved moe riders on master (2.0352, f2b510f5): (1) the Engine A XL arms from takemyjob hopper/ea-enc2-1029 04306fa (patch_enc2.py: E4XL / E5XL = E4/E5 + ldmatrix.x4 B preload + per-record loop trim; bit-exact; research/h100-ea-enc2-1029.md: consume-only 1.014-1.055 over the routed arm, largest at T=256 E5XL 1.039-1.055) added as extra arms in master's per-class [ea-enc] race (same rules: exact, <= 0.98 vs master every round; at T=256 also print the isolated cold-half ratio); (2) at T=256 only, the hybrid's post-norm weighted sum replaced by ea_combine with PDL (takemyjob hopper/glue-1029 f3d0fea; research/h100-glue-1029.md: x1.016 on the T=256 row, rel L2 4.19e-3 vs master 4.59e-3, not bitwise), raced at T=256 through kernel_fn with a tolerance gate (fp32 reference, _ea_canary's rule) and <= 0.99 every round, else master's post-norm. Keep the build/join deadlines (EAE_JOIN_S 220) and size < 1,000,000 B (moe is 948,330 B).

## What changed, and where
**Mechanism:** Engine A XL arms (E4XL / E5XL = E4 / E5 + ldmatrix.x4 B preload + trimmed stream-K record loop, bitwise E) raced as extra arms in the [ea-enc] per-class race on the v4x s-twin K1 + K2 (T=8/16/32/128 routes and the T=256 hybrid's cold half).

**Scope cut (main thread, recorded in DECISIONS):** only the Engine A XL arms are built. The T=256 post-norm -> ea_combine swap (part 2 of `change`) is DROPPED: no kernel_fn or combine change, and no {E, EXL} x {post-norm, ea_combine} cells or canary. Expected rows are the XL rows only (verify_1, verify_16, verify_32, decode_16, decode_32).

What changed (worktree `muse_moe_w4a16/submission.py`, [ea-enc] block only, +231/-22):
- `_EAE_LT` (new CUDA text, namespace eak): `lds128` / `lds64` / `imad_step` and `consume_lt<DQ>`. consume_lt is re-derived from master's `consume_e<DQ, 1, 1>` (ext_a4b / mul_bf16x2_h / scale_pair_imad_rt for E4, the lop3 A5b decode for E5, X1 four-ldmatrix.x4 B preload), not from the probe's scale_enc / frag_a4b helpers. Same loads in the same order on caller-carried shared addresses.
- `_EAE_LT_LOOP`: the probe's trimmed consumer loop (eaenc2 L), ported onto eak4x's expert_body_x (anchored on the eak4x-only `pend0` / acc-init block). It runs under `if constexpr (DQ >= 3 && !DEFER)`, after master's `b >= e` empty-CTA early return, so that return is kept. It has the same flush points (super-group end or e - 1), the same ticket / finish / epi code, and consume2 for passes of more than 8 routes.
- `_eae_src4xl()`: a new module '4xl' = `_eae_src4x()` + `_EAE_LT` + the loop + a host `TORCH_CHECK(pdl & EAE_XL_BIT)` in both launchers. `_eae_start` builds it in its own thread (`eaenc-4xl`, started after 4x and 6). A slow or failed XL build is just not joined by EAE_JOIN_S (still 220), and E4 / E5 are unaffected.
- Selector: `EAE_XL_BIT = 16384` (bit 14). It is clear of EA_PDL_BITS (1, cw << 4 = bits 4..7, 256, 512) and of E5's 4096; the ea4x kernels read only bits 0 and 4..7. `EAE_ARMS` maps arm digit -> (bank copy, XL): 3 E4, 4 E5, 5 E4XL, 6 E5XL. XL reuses the E4 / E5 recoded banks (`w13e3` / `w13e4`), so there is no new weight copy. `_eae_pair`, `_eae_xcore` and `_EAE_CORES` take digits 5 / 6. The bank-drop `used` set maps through EAE_ARMS, so an XL pick keeps its bank alive. `_eae_off` also pops `extexl`.
- `_eae_info` returns every row the module reports. `ok{}` now has 8 v4x rows (4x + 4xl: K1/K2 x E4/E5 each). XL rows are admitted only with lmem == 0 and occ >= 2; E rows keep master's gate.
- Race (`_eae_run`): for 'x' and 'cold' kinds the arms are E4, E5, E4XL and E5XL (E5 / E5XL only when the scales allow A5b), at every class including T=256 (cold half). Exactness uses master's `_eae_hy` (NaN-poisoned h / y on the same draws, including the v4cxs cost-weighted splits) plus kernel_fn out before and after the timed replays plus err / tickets. Bar: `<= 0.98` vs master every round. Tie rule: the best passing E arm wins unless the best passing XL arm's per-round XL/E median is `< EAE_XL_TIE = 0.995`. XL is picked outright only when no E arm passes.
- Lifetimes: XL adds no new per-call buffers. Every captured graph and the cold-iso `h3/y3/l3` stay in `keep` through `_rt_time`, as on master.
- Prints: build seconds per module (`4x / 4xl / 6`), join wait and phase start; `regs/lmem/occ` per module, with the 4xl row labelled `K1e4xl K1e5xl K2e4xl K2e5xl`; per T, exact y/n plus ratio median[min, max] over rounds for each arm, `cold iso` for all four arms at T=256, and `XL/E` (per-round E4XL/E4 and E5XL/E5); PATH CHOSEN per T.
- Deviation: no runtime print of the submission's size. It would need a stat of `__file__`, which I left out to avoid tripping the harness-I/O rule. Size: **959,292 B** (< 1,000,000).
- Review fixes applied: the port is non-mechanical (consume_lt from consume_e); a new non-colliding pdl bit; 8 info / ok rows; XL only (no X), in its own module / thread; EAE_JOIN_S 220; lmem 0 / occ 2 gate; E4XL and E5XL both raced at 256 (no E5XL pre-select); the tie rule defaults to E; the 4x / 4xl / 6 build print. The review's post-norm questions are moot under the scope cut.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn's path changes only which ext / bank / pdl bits `_eae_pair` / `_eae_xcore` pass (dict lookups on the route string set in prepare). The diff adds no sync, `.item()` or input-value branch on the timed path. The new syncs / `.sum()` reads are in the prepare-only race, as on master.
- [x] Nothing derived from `data` is cached across calls: the XL arm is chosen in prepare from synthetic draws; no input-derived state.
- [x] Weights: XL reads master's existing lossless E4 / E5 recoded copies (bit permutation of the NVFP4 codes, A5b scale fields). No new copy, no precision change.
- [x] Activations BF16 where SGLang keeps BF16: same mma bf16 x bf16 -> fp32, same epilogue and same h / y bf16 as E (bitwise, gated exact).
- [x] n/a: attention (moe problem); every local route is computed.
- [x] No timing patches, harness frames / I/O, module edits, reference import, subprocesses or network: the extra module goes through master's `_rt_build` path like 4x / 6. No process or os calls were added.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
OK with warnings: .worktrees/muse_moe_w4a16/engine-a-xl-arms-t256-postnorm-stack/muse_moe_w4a16/submission.py
(warnings: master's pre-existing prepare-only host-sync / uint8-dtype notes; none new on the kernel_fn graph)
```
py_compile OK; __pycache__ removed.

### Node validation (gpunode06 GPU 1, nodejudge `xlmoe-val`, cold builds, budgets enforced, 2026-10-30 00:41Z)
Bytes sha256 091c05b8670a... (959,292 B). Prepare took 131 s of 300; all 5 TASK tests passed (rel L2 4.1e-3..5.6e-3); 6 shapes x 6 reps ran clean. SGLang was NOT timed: the config derived from hhwh-moe has `baseline: null`, so the run has no headline.
```
[ea-enc] build 4x 37s, 4xl 37s, 6 37s (bg from 0s), join wait 0.0s, phase at 129s
[ea-enc] regs/lmem/occ 4x K1e4 82/0/2 ...; 4xl K1e4xl 94/0/2 K1e5xl 92/0/2 K2e4xl 94/0/2 K2e5xl 92/0/2; ... E5 on
[ea-enc] T=8 v4xs: E4 y 0.989[0.984,1.004] E5 y 0.996[..] E4XL y 0.985[0.976,0.999] E5XL y 0.994[..] | XL/E E4XL 0.995 E5XL 0.996
[ea-enc] T=32 v4cxs: E4 y 0.992 E5 y 0.991 E4XL y 0.988[0.965,1.004] E5XL y 0.994 | XL/E 0.997 / 1.000
[ea-enc] T=128 v4cxs: E4 y 0.997 E5 y 0.991 E4XL y 0.991 E5XL y 0.988[0.969,1.004] | XL/E 0.997 / 1.001
[ea-enc] T=256 v4xs: E4 y 0.995 E5 y 0.993 E4XL y 0.991[0.980,0.993] E5XL y 0.991 | cold iso E4 0.993 E5 0.987 E4XL 0.988 E5XL 0.984[0.976,0.991] | XL/E 0.996 / 0.996
[ea-enc] PATH CHOSEN all T master (no arm <= 0.98 every round; T=16 route v4xe, T=1 v1) (race 2.4s, prepare at 131s)
```
Every XL arm is exact at every class. XL/E is 0.995-0.997 at T8/32/128/256 for E4XL and at 256 for E5XL. On the node no arm, E or XL, clears the 0.98-every-round bar, so the node picks master everywhere, the same as master's own E arms on the node. Results: scratchpad `build-moe-xl/hp/prototypes/hopper-probe/runs/20261029-174132-gpunode06/out/`.

## Risks / what could fail
medium

## Result (after the run)
(pending)
