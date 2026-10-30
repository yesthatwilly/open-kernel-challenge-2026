# muse_decode_layer · 2026-10-30T092018Z · submit · ladder-dc-at7e-port

- **Job:** `dfb318ee06b842f7` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `a82ea6e0` (ladder/dcat7e on E3 ed478481), 996,666 B


## Proposed CLAIMS line (decode)
- [lead] DC-AT7e: attention's fvg full-verify kernel (fvg_kernel<1,PV8>, E4M3 P / FP32 denominator / FP8 KV) ported to decode v16f/v32f, raced in prepare on whole-layer graphs: node A/B v16f -32.6 us (x1.071), v32f -74.9 us (x1.110), identity <= 0.5%; attention vs fp64 0.92x the FA3 incumbent's error | rows: load=verify_16 layer=full, load=verify_32 layer=full | evidence: ladder/dcat7e a82ea6e0; runs/20261030-015814-gpunode06 (val 13/13, prepare 126.6 s), runs/20261030-020750-gpunode06 (A/B) | board: - | blob: - | reopen: judge check | 2026-10-30 ladder

## Next
- If the check lands: decode has 3.3 KB left. The BF1 port needs a further prune, and that is the user's call (see
  e3-1030.md "Further room").
- If the judge shows no gain at v16f/v32f: R10 holds on the judge, and fvg should next be raced on the judge's own
  layer timing.

---
# Attempt draft: check ladder/dcat7e a82ea6e0 (DC-AT7e)

- **Goal.** Measure on the judge attention's landed fvg full-verify kernel (AT7/AT7c/AT7e) inside decode's layer at
  v16f / v32f.
  - Prediction: v16f -16..-32 us, v32f -30..-59 us, every other row 0.
  - Node A/B: -32.6 / -74.9 us.
- **Base.** E3 ed478481 (check add080da 1.686x). The bytes are 996,666 B (cap 1,000,000).
- **Parts touched.**
  - New module `muse_decode_fvg_v1`: attention ee566de0's `_FVG_CUDA` / `_FVG_CPP` with comments stripped (token
    stream identical), built in a background thread at import.
  - `_attn_core`: an early return to `_fvg_attn` when `state["fvg"][batch]` is set and the config / shape gate passes.
  - `_prepare_main`: a `_fvg_setup` phase after `_dc0f_gate`, whose lines go into FINAL PATHS (`[dc-fvg]`).
  - Nothing else changed.
- **Rules checklist** (rules/technical-rules.llms.txt vs the diff):
  - [x] single file; native CUDA via load_inline only
  - [x] CUDA-graph capturable: the cooperative launch is captured in prepare (probe), with no host sync in the routed
        path, and branches use config / shapes only
  - [x] weights untouched; DC0b FP32 router at T <= 8 unchanged; FP32 w2 split-K reduce pin unchanged
  - [x] activations BF16 where SGLang keeps them; FP8 KV cache stays FP8; queries rounded to E4M3 as FA3 does;
        E4M3 P for PV as FA3 does; FP32 softmax denominator from the unrounded P
  - [x] attends to every key of the full context (stream-K over all tiles, causal over the 8 verify tokens); no
        top-k / sparse / skipping
  - [x] no output caching or replay; buffers are workspace only (counters reset by the kernel and checked in
        prepare); no harness tampering
  - [x] prepare 126.6 s cold on the node (< 300 s); the fvg phase has a deadline at 250 s
- **Risks.**
  - R10: the judge's layer may not show the node's gain. The race decides per row with margin 1.01 in every round, so
    the worst case is the incumbent.
  - The cooperative launch on the judge's GPU: if capture fails, the code falls back to a plain launch (coop 0) with
    the same grid. Its occupancy of 2 CTAs/SM is asserted.
  - Prepare grows by about 7 s and the import build by about 35 s in the background; est. judge run 348 / 900 s.
  - The fvg race is the last prepare phase. If an earlier phase runs late (> 250 s), fvg is skipped and the incumbent
    is kept.
- **Read-out.**
  - Look for `[dc-fvg] PATH CHOSEN v16f/v32f` in FINAL PATHS, and the kernel `fvg_kernel<1, true>` on the v16f /
    v32f profile rows (in place of `flash::...` + FlashAttnFwdCombine).
  - Hit = >= 1/3 of the predicted delta per row; other rows level with add080da.


## Why a scored run
Check 82c6bac9 1.699x beats the board 1.690, same bytes a82ea6e0 (E3 prune + DC-AT7e port on DC0b). Standing rule: scored submit + promotion when a check beats the board.

## Result (after the run)
- job `dfb318ee06b842f7` (submit), passed, **1.699x [1.673, 1.724]** vs board 1.690 -> landed on master (bytes a82ea6e0).
