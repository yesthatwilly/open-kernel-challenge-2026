# muse_decode_layer · 2026-10-29T212727Z · check · hothalf-wh5ds-moe-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-29T202838Z-hothalf-wh5ds-moe-port.md` (branch `hyp/muse_decode_layer/hothalf-wh5ds-moe-port`).

- **Job:** `35cc220cd79e4668` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `620b3802`

## Goal
Port moe's landed wh5ds hot half back to decode: take the module from muse_moe_w4a16/submission.py on master (landed hothalf-wh-a5b-port, scored a5318a1f: arms D0=0, a3=1, wh=7, wh5ds=14 = A3 + A5b code re-encoding / 7-bit packed scales + per-stage descriptor + split commit on the wh exact-N/ring kernel; isolated-pair route gate <= 0.97 vs master and D0 every round, wh5ds also <= 0.98 vs wh; production no-loss gate) and decode's own routed wh module (branch hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh, check 44f30364: wh routed T=128 and T=256). Replace decode's wh-only arm set with D0/a3/wh/wh5ds, route per T class (128 and 256) by the isolated pair, plus a no-loss gate on decode's layer timing at each class (wh/m layer ratio <= 1.00 every round, as decode's 44f30364 FORK lines measured). Keep decode's fork setting (CAMPAIGN_HCF False), routing, cold half and PDL unchanged. Prints as in 44f30364 ([hh-wh] BUILD/ATTRS/SCALES/ROUTE/FORK/PATH CHOSEN, <= 1.5 KB at the end of prepare). Size: decode is 955,413 B on master; 44f30364's wh module brought it to 984,021 B; the extra wh5ds arm must still fit under 1,000,000 B (trim a3nx and dead diag text).

## What changed, and where
**Mechanism:** moe's landed wh/wh5ds wgmma hot half (wg_kernel<G,7>/<G,14>: A3 or A5b dequant + exact-N + A ring + per-stage descriptor + split commit) routed per T class at decode's T=128/256 hybrid hot half (_hc_experts), with the pick made on the isolated pair and the layer gate used only as a veto.

- Built 2026-10-29T2126Z in `.worktrees/muse_decode_layer/hothalf-wh5ds-moe-port/muse_decode_layer/submission.py`. Size **988,094 B** (master 955,413; +32,681). Diff: 1 file, +685 / -5.
- **Kernel text:** `_WH_CPP`, `_WH_DQ`, `_WH_RING`, `_WH_NX_SWITCH`, `_WH_EDITS` and `_WH_MAPS` are copied verbatim from muse_moe_w4a16/submission.py on master. I applied `_wh_src` to decode's `_WG_SRC_HEAD/_BODY` offline via AST: every asserted edit target matched exactly once, and the generated CUDA TU is **byte-identical** to moe's (48,422 B). That is the module that compiled and ran on the judge (54b70249/a5318a1f), with the same register and spill attributes. I made one change from moe: the ext name is `muse_decode_wg_wh5_<sha10>`. Arms: D0=0, a3=1, wh=7, wh5ds=14. a3nx is dropped.
- **Plumbing** comes from decode's 44f30364 branch:
  - `prepare()` → `_wh_start` (a non-daemon nvcc thread; TORCH_CUDA_ARCH_LIST=9.0a, restored in `finally`), then `_prepare_main`, then `_wh_join(restore=True)`. The build thread is joined before `prepare` returns.
  - `_wh_phase` runs before `_sfk_gate`.
  - `_wg_wgmma_src(nbs=WG_BUCKETS)` gains a parameter.
  - `_hc_experts`' wg hot half calls `_wh_pair(state, whk[_hc_class(T)], ...)`. This is the only dispatch site. `_wh_pair` with v=None runs master's two `_wg_gemm` calls, unchanged.
- **Route (per class, review fix 4):**
  - Exactness: torch.equal on 2 router draws plus 4 all-N synthetic routings, output and hs, every arm. The err word must be clean.
  - Timing: an isolated-pair graph race of m/D0/a3/wh/wh5ds over 2 inputs × rounds. D0 must be exact and D0/m must be within 0.97..1.03.
  - Candidates are wh5ds and wh. Each must be ≤ 0.97 vs m and vs D0 in every round; wh5ds must also be ≤ 0.98 vs wh in every round. Priority is wh5ds, then wh. a3 is a candidate only when neither passes.
  - The first candidate that passes replay 16/16 is picked.
- **Gate (veto only):** `_wh_gate` → `_ead_layer_gate` on {m, picked} × {serial, fork} (`side_serial`). Every round must be ≤ 1.00 in both modes. If wh5ds is vetoed, wh is tried (replayed first if it had not been) and gated. Otherwise the class stays on master. The gate never chooses between arms.
- **Unchanged:** master routing, the T=16 ea4x race, cold half, PDL, CAMPAIGN_HCF, and all of kernel_fn except the hot-half call.
- **Kept:**
  - the scale scan (0x08..0x7E)
  - the per-chunk inverse checks for the A3 permute and A5b ENC4/pack copies
  - `_wh_selftest`
  - the attrs check (regs×384 ≥ 168-reg pool, lmem 0, sm_90 image)
  - the 240 s phase guard (checked before every step)
  - `_EAD_INPUTS` trimming
- Bank copies that end up unrouted are freed: whk_x5 unless a routed v has bit 8, whk_bank unless one has bit 1.
- **Prints:** at the very end of prepare (after the SFK end line), ≤ 300 chars each. They are:
  - BUILD, ATTRS (4 arms × w13/w2), SCALES
  - ROUTE T=128/256 (wh/m, wh/D0, wh5ds/wh per round; a3/m max; D0/m range; us; exactness; replay)
  - GATE T=… (picked arm layer/m, s and f, per round; PASS/VETO; fallback)
  - PATH CHOSEN with the `wg_kernel<G,V>` profile symbol
  - the last `[ea4x-dl] route (board …)` line from EA4X_LINES, re-printed

  Estimated total is ~1.4 KB.
- **Deviations:**
  - (a) The per-step `> step` marker prints are cut, as MUST 1 asks (this overrides BUILD_BRIEF's announce-each-launch rule). `step()` still syncs before each step and records its name, and every OFF/raise line reports `at <step>`, so a fault still names its step.
  - (b) The 2x2 FORK diagnostic is replaced by the veto GATE line (master vs picked arm, both modes).
  - (c) T=128 still routes when its isolated pair qualifies. The review left this open; the route is cheap, bit-exact and vetoed.
- **Not validated:** no GPU, torch or nvcc on this node. The optional H100 node validation was skipped because the coordinator's 20-minute limit was too short for a full decode prepare. The new Python is unexercised. The main risks are in `_wh_phase`'s new route/gate control flow, especially the gate's graph capture of kernel_fn with `whk[cls]` set to 14 under fork mode.

## Rules check (each must be true before the run)
Each item below was checked against `git -C .worktrees/muse_decode_layer/hothalf-wh5ds-moe-port diff`.

- [x] CUDA-graph capturable: kernel_fn's only change is `_hc_experts` calling `_wh_pair`. That function chooses the arm from a dict keyed by `_hc_class(num_tokens)`, which is shape-only. It makes no `.item()`/`.cpu()`/sync calls and allocates nothing new. All syncs, `.tolist()` and `int()` calls are in prepare-only `_wh_*` helpers; lint marks them "outside kernel_fn's call graph".
- [x] Nothing derived from `data` is cached across calls. `state["whk"]`, `whk_bank`, `whk_x5`, `whk_ext` and `whk_err` are set in prepare from the weights only. The `_EAD_INPUTS` additions are trimmed back.
- [x] Weights are repacked losslessly in prepare only. The A3 copy is a code-bit permutation; the A5b copy is an ENC4 code permutation plus 7-bit (b-8) scale packing that is valid only when every scale byte is in 0x08..0x7E (scanned). Each chunk is checked by its inverse, and the dequant equivalence is self-tested. NVFP4/int4 stays 4-bit and BF16 stays BF16.
- [x] Activations: BF16 in and out, unchanged. The outputs are torch.equal to master's on 6 routings plus 16 replays, or the class stays on master. KV and queries are untouched.
- [x] n/a: attention is untouched (MoE hot half only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds no subprocess, os.fork, multiprocessing or spawn: the build is a `threading.Thread` around `load_inline`, as on master's moe.
- [x] All work happens inside the timed call; there is no output caching or replay. kernel_fn computes the hot half fresh on every call.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a is set for prepare and restored in `finally`. WG_CUDA_CFLAGS (`-lineinfo`, WG_GENCODE) are reused. The attrs check requires the sm_90 image, 168 regs and 0 lmem, and any miss falls to master. Waits use master's bounded `mbar_wait` (err/dead).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
OK with warnings: .worktrees/muse_decode_layer/hothalf-wh5ds-moe-port/muse_decode_layer/submission.py
(209 warnings: master's 202 + 7 new `host-sync ... outside kernel_fn's call graph; fine in prepare()` in _wh_phase/_wh_replay; no errors)
```
py_compile: OK; __pycache__ removed.

## Risks / what could fail
medium
Decode's wh check 44f30364 routed and moved the verify_32 sliding hot pair x0.80, but the headline read flat (1.6794 vs 1.6904) on a noisy draw; moe's wh5ds then landed (54b70249 1.9827, scored a5318a1f 1.9748). Read this on the [hh-wh] lines and the wg_kernel<G,14> profile rows first, headline last.

## Result (after the run)
- job `35cc220cd79e4668` (check), status done, passed True, tests 13/13
- speedup 1.6968x, 95% CI [1.6714, 1.7223], score 13330.900973619004 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.6903955680487197, "ci": [1.665039634527989, 1.7157515015694504], "job": "7cc034f917274439"}; base result: muse_decode_layer/hypotheses/BEST.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/hothalf-wh5ds-moe-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/hothalf-wh5ds-moe-port.patch`; branch `hyp/muse_decode_layer/hothalf-wh5ds-moe-port`
