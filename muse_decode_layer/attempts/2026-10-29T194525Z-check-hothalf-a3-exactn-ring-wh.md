# muse_decode_layer · 2026-10-29T194525Z · check · hothalf-a3-exactn-ring-wh

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-29T190020Z-hothalf-a3-exactn-ring-wh.md` (branch `hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh`).

- **Job:** `44f303647100481b` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `2957d173`

## Goal
Port the H100-validated 'wh' stack into decode's wgmma hot half (wg_kernel<true>/<false> and its dq_stage, muse_decode_layer/submission.py ~:15330; moe twin muse_moe_w4a16/submission.py:1128). Three parts, all bit-exact vs master: (1) A3 dequant = moe Engine A's dq2 form (paired-scale prmt+IMAD with mul_bf16x2_h<H> broadcast, plus ext_prop nibble extraction on a prepare-time word-permuted copy of the hot experts' code bank), exactly as landed in prefill (muse_prefill_layer/submission.py, the rega dq module of experts-rega-dq2-dequant-cost-diag, master 1b14174f) and as in takemyjob hopper/hh2-1029 e3bcfcd probes/hh2; (2) exact-N tiles ('nx'): each hot expert's N tile sized to its real route count rounded to the wgmma N granule instead of the current padded buckets (576 columns computed for 386 routes at T=256); (3) for tiles with N >= 48, a 4-deep register ring with wgmma_wait<2> ('w4b' arm limited to N>=48; neutral/slightly worse at N32, so N32 keeps master's ring). Keep the fork (W6-M), routing, cold half (ea4x), PDL and everything else unchanged. Route the new hot half by an in-prepare interleaved race per T class (T=128, T=256) vs master's hot half: on only if bit-exact (torch.equal) and forked pair ratio <= 0.98 every round; print [hh-wh] RATIO/ROUTE lines in a <= 2 KB summary at the end of prepare. Prepare-time and size budget: decode is near the 1,000,000 B cap (check current size; trim diag text if needed) and the prepare deadline; build the new module in a background thread from the top of prepare if it is a separate load_inline module.

## What changed, and where
**Mechanism:** the T=128/256 wgmma hot half (w13 + w2) runs as the new module's `wg_kernel<G, 7>` = A3 dequant on a prepare-time word-permuted copy of the whole wg bank, plus exact-N tiles (N = 16 nbk) and an R4 register ring with wait<2> for N >= 48, routed per T class on the isolated-pair gate (verify_16 / verify_32 rows).

Edited only `muse_decode_layer/submission.py` (+581/-5; 955,413 -> 984,021 B, under the 990 KB comment-strip line). Nothing compiled or ran on this workstation; built and prepare-validated on gpunode06 (below).

What changed
- **New module `muse_decode_wg_wh`**, derived by `_wh_src()` from master's `_WG_SRC_HEAD` / `_WG_SRC_BODY` with 22 asserted (old, new) edits, each old text occurring exactly once (else OFF). The kernel is templated `wg_kernel<G, V>`, with V bits 1 = A3, 2 = exact N, 4 = ring. Four arms x {w13, w2} = 8 kernels:
  - D0 = `<G,0>`: master's dq_stage, buckets and loop, compiled in this TU (the control).
  - a3 = `<G,1>` and a3nx = `<G,3>`: prepare-only attribution arms, never routed.
  - wh = `<G,7>`: the routed arm.
  - The profile tells master's `wg_kernel<true>` / `<false>` apart from `wg_kernel<true, 7>` / `<false, 7>`.
  - The kernel text equals hh2's `hh_wh.cuh` (takemyjob e3bcfcd), diffed. The helpers (hh_rt16 / hh_scale_pair / hh_mul_h / hh_ext_prop / hh_dq_step) and the ring are verbatim. The only deltas: master's dq_stage is kept, and the A3 form is named `dq_stage_a3`; exact N is a device-side `switch (m.nbk)` in `if constexpr (V & 2)`; the bucket arms index the same 8-map set at `(1 << bucket) - 1`, i.e. boxes of 16 / 32 / 64 / 128 rows and master's expect_tx.
  - Tensor maps are still encoded per call: 8 maps with box rows 16 (b + 1). `Wg<48/80/96/112>` comes from `_wg_wgmma_src(nbs)`, master's generator with a parameter whose default is unchanged.
- **Build:** `_wh_start` runs at the very top of `prepare()`, which is now a thin wrapper: `_wh_start`, then `_prepare_main` (master's former body), then a `finally` that joins. The build is a non-daemon thread at nice +10. TORCH_CUDA_ARCH_LIST is set to 9.0a by the main thread and restored after the final join (prefill's idiom).
  - It builds with master's WG_CUDA_CFLAGS plus `-Xptxas -Werror` first. On failure it greps the compiler text for C75xx. With no C75xx and less than 100 s elapsed it rebuilds without -Werror; any C75xx means OFF. The [hh-wh] BUILD line reports this, so a clean -Werror build proves no C75xx. No cuobjdump.
  - `wh_init` returns cudaFuncGetAttributes per kernel. ATTRS prints regs/spill for all 8 kernels. Any kernel that is not launchable, has occupancy < 1, is not an sm_90a image (bin 90), lacks the register pool, or has local bytes > 0 turns the whole phase OFF.
- **Dispatch:** `_hc_experts` calls `_wh_pair(state, state["whk"][class], ...)`: None runs master's two `_wg_gemm` launches unchanged, 7 runs the wh module. The branch is on the T class only (a shape), never on route counts. Master's gather is kept, as is its w2 PDL setting (`state["wg_pdl"]`, False on master, so neither arm sets the PDL attribute on w2; the griddepcontrol.wait in both kernels is a no-op then). The new module has its own error word `whk_err`.
- **Phase `_wh_phase`** runs after master's hot/cold, cells and anatomy, and just before `_sfk_gate`, so the [w6m-fork] gate sees the final route. Each step is behind a 240 s guard, and an exception sends both classes to master.
  1. Scale scan: every scale byte of the whole wg bank (w13 + w2) must lie in 0x08..0x7E, else OFF.
  2. `_wh_perm`: a copy of the WHOLE bank, every 32-bit code word bit-permuted (the hh2 `perm_word` / prefill LUT layout, checked here against both over 200k random words, with 0 ext_prop mismatches). Each chunk is checked lossless by the inverse; scale bytes are unchanged. It is derived from master's retiled bank (weights only), never from a routing.
  3. Exactness per class: 2 router draws (`_ead_cases`, the gates' inputs) + 4 synthetic routings (every local expert hot over the 4, with route counts giving nbk 1..8 at T=256, local_start 96). Master's eager pair is compared bitwise (torch.equal) to every arm, on both the w13 output `wg_hs[:n_padded]` and the junk-filled w2 output. Error words are checked as [0, magic].
  4. Timing: the ISOLATED pair (w13 -> w2, gather outside the graph) as master's dispatched launch, D0, wh, a3 and a3nx. Graphs are timed with `_rt_time` (read flush, interleaved / rotated, empty graph subtracted), 2 inputs x 2 rounds.
  5. Route ON iff D0 and wh are bit-exact, D0/m is in 0.97..1.03 in every round (else INVALID, master kept), wh/m <= 0.97 AND wh/D0 <= 0.97 in every round, and the replay check passes.
  6. Replay check: wh graphs (gather + w13 + w2) on two scratch sets (state's, and fresh xs/hs/err), refilled 8 times (router draws and all-N synthetic routings alternate), replayed in alternating order: 16 replays per class, each bitwise vs master's eager pair, plus the error words.
  7. FORK: for every exact class, `_ead_layer_gate` (as `_sfk_gate`: whole layer, real ea4x cold half, interleaved, read-flushed, 2 inputs x 2 rounds) over {master hot, wh hot} x {serial, fork}, where fork is `side_serial` False as in `_sfk_gate`. Prints [hh-wh] FORK lines with per-round wh/m under the fork and serially, plus master's fork/serial.
  8. Lines: the ROUTE line lists per-round wh/m, wh/D0, D0/m, a3/D0 and a3nx/D0, the pair us and exactness counts. `_EAD_INPUTS` is trimmed back afterwards (as `_sfk_gate`). If nothing routes, the permuted bank is freed.
- **Log:** the [hh-wh] summary (BUILD, ATTRS, SCALES, ROUTE x2, FORK x2, PATH) is printed last, after the `[w6m-fork] end` line, each line <= 300 chars. During the phase only ~10 short synced step markers print (`[hh-wh] > exact T=256`, ...).
- **Ring hazard** (review 6, unchanged from hh2):
  - The k16 step j writes ring slot j % 4 only after step j - 1's wait<2>, which retired every group <= j - 3. A slot is therefore rewritten 4 commits after the group that read it. An `asm volatile` clobber pins the next slot behind each wait.
  - The previous smem stage is released after this stage's second wait<2> (every group <= its last k16 retired). Items end on wait<0> and release their last stage.
  - The bounded mbar_wait / error-word path is master's.

Deviations / choices
- The pair is timed as master dispatches it: w2 without the PDL attribute, because master's `wg_pdl` is False. The review's "w13 -> PDL w2" wording describes the probe's pair. Routing on the launch that actually runs is the review's intent ("vs master's dispatched launch").
- "Fork" in the 2x2 is `_sfk_gate`'s `side_serial` toggle (the W6-M side-stream fork). CAMPAIGN_HCF is False on master, so the hot and cold halves share the main stream and overlap only through ea4x/wg PDL.
- The judge's prepare log is a ~16,000-char tail, so the summary is not echoed twice.
- The decode-only scope holds: no moe edits.
- Prior (review clerical fix): the frontmatter `prior` now names decode CLAIMS:98 (A3 lead, the parent) and CLAIMS:59 (wg2 re-split, refuted), not S1a / CLAIMS:72.

## Rules check (each must be true before the run)
Each item was checked against `git -C .worktrees/muse_decode_layer/hothalf-a3-exactn-ring-wh diff`.

- [x] CUDA-graph capturable: the kernel_fn path adds only `_wh_pair`, which makes two `wh_gemm` launches or master's two `_wg_gemm` launches. It chooses by a dict lookup on the T class (a shape). No `.item()` / `.cpu()` / sync / tolist, no allocation and no branch on route counts: exact N is selected in-kernel from the chunk's nbk. Every sync / tolist in the diff is in `_wh_replay` / `_wh_phase` (prepare only).
- [x] Nothing derived from `data` is cached across calls. `state["whk"]` holds per-class route ints decided in prepare on synthetic / router-draw inputs. Tensor maps are encoded per call from the scratch passed in that call.
- [x] Weights: the permuted bank is a lossless prepare-time copy of master's retiled wg bank: a fixed bit permutation of each code word, inverse-checked per chunk, with scale bytes unchanged and checked. NVFP4 stays NVFP4 (the same 4-bit codes and E4M3 scales). Dequant is bit-exact to master (torch.equal gate on every routed class).
- [x] Activations BF16 as master (same gather, same bf16 hs / out). No KV / attention change.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No subprocess / multiprocessing / os.fork / spawn in the diff. The build is torch `load_inline` (its ninja / nvcc, as master's modules) inside a `threading.Thread`. No timing patches, harness frames, network or reference import.
- [x] All work happens inside the timed call; no output caching or replay. The prepare-only graphs are freed.
- [x] Bounded waits: the new kernels keep master's bounded `mbar_wait` (error word, never a hang). sm_90a-only PTX stays under `__CUDA_ARCH_FEAT_SM90_ALL`, with error word 2 in the else-branch. TORCH_CUDA_ARCH_LIST=9.0a is set around the build and restored after the join.
- [x] Any miss (build, -Werror C75xx, attributes / spill, scale scan, permutation, exactness, fidelity, ratio, replay, deadline, exception) leaves master's hot half. Loud `[hh-wh] ROUTE` / `PATH CHOSEN` lines name the reason.
- [x] Decode-only: `git diff --stat` shows only `muse_decode_layer/submission.py`; no moe edit.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer`):
```
OK with warnings: .worktrees/muse_decode_layer/hothalf-a3-exactn-ring-wh/muse_decode_layer/submission.py
(warnings: host-sync / dtype lines, all "outside kernel_fn's call graph; fine in prepare()" or pre-existing master
 dtype lines; the new ones are L16345-16498, inside _wh_replay / _wh_phase)
```
py_compile ok; __pycache__ removed.

## Risks / what could fail
medium
Prior (campaign, not a churn slug): S1a / Engine S hot kernel was refuted at x0.89-0.95 of wg (CLAIMS:72); this keeps master's wg_kernel and its fork and changes only its dequant instruction form (A3, judge-proven on prefill), its N tiling and its ring depth for N>=48, all bit-exact. Kill/read: verify_32 sliding above ~416 us (the A3-only level) means the stack parts did not transfer. The fork arm on the node used a calibrated cold-half stand-in, not ea4x; only the judge measures the real fork. Registers: all variants 168 regs, 0 spill, no C75xx on cu130.

## Result (after the run)
- job `44f303647100481b` (check), status done, passed True, tests 13/13
- speedup 1.6794x, 95% CI [1.6542, 1.7046], score 13469.02530616763 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.6903955680487197, "ci": [1.665039634527989, 1.7157515015694504], "job": "7cc034f917274439"}; base result: muse_decode_layer/hypotheses/BEST.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/hothalf-a3-exactn-ring-wh.result.json`, `muse_decode_layer/hypotheses/done/neutral/hothalf-a3-exactn-ring-wh.patch`; branch `hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh`
