# muse_decode_layer · 2026-10-30T012159Z · submit · engine-a-xl-arms-t1-t256

Hypothesis: `muse_decode_layer/hypotheses/done/gain/2026-10-30T000244Z-engine-a-xl-arms-t1-t256.md` (branch `hyp/muse_decode_layer/engine-a-xl-arms-t1-t256`).

- **Job:** `beaed2ad8eb541bf` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `38f04af8`

## Goal
Promote check-proven gain engine-a-xl-arms-t1-t256 (1.7468x [1.7206,1.7730], check job 9263781bc2a040e9) to the board.

Add the H100-validated XL arms (takemyjob hopper/ea-enc2-1029 04306fa, research/h100-ea-enc2-1029.md: ldmatrix.x4 B preload + trimmed per-record loop with address math hoisted, on top of E4/E5; E4XL and E5XL, bit-exact) to decode's landed Engine A E race (decode master = engine-a-encodings-wh5ds-decode-stack, scored e4233998 1.7051). Race them per T class alongside E4/E5 with the same rules (isolated-pair pick <= 0.98 every round, exact, then the layer veto <= 1.00 serial and fork). Targets: T=256 cold half (E/m iso .977-.983 just above the 0.98 bar in 824b60fa; node E5XL consume 1.039-1.055 over E5) and T=1 (layer E4/E4 .997-1.007 just above the 1.00 veto; XL trims K1/K2 loops); also every routed class (T=8/16/32/128) may move to XL. Keep everything else unchanged. Size: decode is 993,828 B; the XL delta must fit under 1,000,000 (trim Python only, CUDA strings of other modules byte-identical). Prints: the [ea-enc] block with XL arms per class.

## What changed, and where
**Mechanism:** Engine A XL arms (E4XL/E5XL: consume_e's B preload on loop-carried shared addresses + trimmed stream-K record loop, moe's landed '4xl') raced per class in decode's ea4x E race (T=8/16/32 + T=128/256 cold half); T=1 (ea6) unchanged.

- Ported moe's landed XL module (muse_moe_w4a16 master 59f2a35c; check 73156455 2.0667) into decode's `[ea-enc]` race. Decode's generated ea4x source is byte-identical to moe's, so the XL consumer loop `_EAE_LT_LOOP` is moe's code with comments stripped (same tokens), inserted before the eak4x consumer loop (found by index after `namespace eak4x {`, not a 260 B anchor).
- Size (review fix): `consume_lt` is not a copy. `_eae_src4xl()` derives it from decode's `_EAE_CONSUME` by 7 anchored replacements + 2 load rewrites (address-form entry: recl/recs/xb/m16/m78 from the caller; `ld.shared` via `lds128`/`lds64`, asm volatile as in moe). With XB=1, NT=1 the `if constexpr` branches reduce to moe's consume_lt exactly (checked by diffing the generated 4xl source against moe's `_eae_src4xl()`: only comments, the template header, the discarded constexpr branches, the call-site cast and the guard text differ). No X arms (dq 11/12) ported.
- Own module/thread `'4xl'` in `_eae_start` (same join deadline EAE_JOIN_S); launches guarded by `TORCH_CHECK(pdl & 16384)`; arms '5'/'6' = (bank dq 3/4, XL) via `EAE_ARMS`; `_eae_pair` / `_eae_xcore` / `_EAE_CORES` / `_eae_off` / `_eae_lab` / `used` extended. XL instances admitted only with lmem 0 and occupancy >= 2 (`ok[('4xl', K, dq)]`).
- Race rules unchanged: exact (h, y, output, NaN-filled buffers, tickets/err 0) and iso <= 0.98 every round, then layer veto <= 1.00 (serial + fork for the cold half). Ties keep the E arm: an XL arm replaces the best passing E arm only at per-round XL/E median < 0.995 (moe's EAE_XL_TIE). New: if the XL pick fails the veto or canary, the E arm is tried (the route master would have taken), so XL can only displace, never lose, an E route.
- T=1: no XL arm (ea6 v6pe2; review weakest link 1 applied); PATH CHOSEN prints `T=1 ... (no XL arm: ea6)`.
- Buffer lifetimes: decode's `_eae_class` builds iso graphs over `_eae_fn` closures whose inputs live in `d` through `_rt_time` and allocates inside capture (graph-private pool); no tensors allocated outside capture are referenced by a graph (the moe UAF pattern is absent).
- Size: 993,828 -> **999,792 B** (< 1,000,000). Python-only trims: removed dead, never-referenced names (`_ea4x_passes`, `EA4X_START_S/GATE_S/COPY_S/PROBE_S/MARGIN/TIGHT/FLOOR/SEED/CLASSES/CANARY_T`, `HC_WG_H`, `CAMPAIGN_HCF`, `CAMPAIGN_ENGINE_D`; each occurred once in the file). Every long string constant on master is present unchanged; `_eae_src4x()` and `_eae_src6()` outputs are byte-identical to master's (checked by extraction).
- Prints: `[ea-enc] build 4x/4xl/6 s`, `regs/lmem/occ ... 4xl K1e4 r/l/o ...`, per T `iso <arm> y/n med[min,max]` for E4/E5/E4XL/E5XL, `E4XL/E4 med[min,max]` per passing XL, `layer <arm> s/f`, PATH CHOSEN with race s and prepare-at.
- **Node validation** (gpunode06 GPU 1, nodejudge `xldec-val`, cold builds, SGLang in process, 13 TASK tests + 3 shapes x 4 reps; run takemyjob hopper-probe runs/20261029-180231-gpunode06, snapshot sha 0303080a28bc): **passed=True**, no fault; prepare 114.6 s (est. run 361 s / 900 s). `[ea-enc] build 4x 34s, 4xl 34s, 6 33s, join wait 0.0s, phase at 113s`; XL regs/lmem/occ K1e4 94/0/2, K1e5 92/0/2, K2e4 94/0/2, K2e5 92/0/2 (admitted; E 82-83/0/2). Iso (all arms exact y): T=8 E4 .986[.972,.992] E5 .998 E4XL .990[.980,1.001] E5XL .986[.964,.991]; T=16 E4 .988 E5 .988 E4XL .990 E5XL .986[.980,.991]; T=128 E4 .993 E5 .992 E4XL .993[.977,.998] E5XL .993; T=256 E4 .999 E5 .996[.979,1.000] E4XL .997 E5XL .992[.977,.996]; T=32 route not a v4x s twin on this node. PATH CHOSEN master everywhere on the H100 (no arm <= 0.98 every round; the landed E stack also routed no E on the node), so the layer veto and the XL tie rule did not execute here; they run only on the judge.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/engine-a-xl-arms-t1-t256 diff` (162+/41-, submission.py only).

- [x] CUDA-graph capturable: the XL arms are the same ea4x_w13/ea4x_w2 launches (one module handle, pdl bit 16384) inside the existing graphed core; no `.item()`/`.cpu()`/sync/prints/allocation added to kernel_fn's path; all new host work (race, tie rule, veto fallback) is in prepare().
- [x] Nothing derived from `data` is cached across calls (routes are chosen in prepare on synthetic draws).
- [x] Weights: XL reads the same lossless A4b/A5b recoded banks (`w13e3/e4`, `w2e3/e4`) as the landed E arms; no new repack.
- [x] Activations BF16, KV FP8, queries unchanged; XL arithmetic is consume_e's (bitwise; node exact y on all arms).
- [x] n/a: attention untouched (expert block only).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (load_inline as the existing modules).
- [x] All work inside the timed call; no output caching or replay.
- [x] Size 999,792 B < 1,000,000; other modules' CUDA strings byte-identical to master.

```
okc_lint: OK with warnings (pre-existing host-sync-in-prepare / uint8 dtype warnings only; 0 errors)
py_compile: OK (__pycache__ removed)
```

## Risks / what could fail
medium

## Result (after the run)
(pending)
