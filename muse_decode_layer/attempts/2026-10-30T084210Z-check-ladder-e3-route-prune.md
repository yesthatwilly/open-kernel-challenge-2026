# muse_decode_layer · 2026-10-30T084210Z · check · ladder-e3-route-prune


- **Job:** `add080da32034f4c` (`okc status <job>`)
- Goal: confirm a pure size trim of decode is neutral on the judge, so its freed 42,171 B can carry the landed
  component ports (BF1, AT7/AT7c/AT7e). Prediction: every route choice unchanged; headline inside DC0b's own spread
  (DC0b check 50d63183 / scored submit). Expected removed-arm effect on the scored rows: 0 us.
- Bytes: ladder/e3 ed478481, muse_decode_layer/submission.py 957,829 B (cap 1,000,000; DC0b 5e4601ed was 995,734 B).
- Parts touched (4 commits on DC0b 5e4601ed):
  - 21998c23: base reset to DC0b (DC1 t1xl twins dropped, a judge null: check 50d63183).
  - d4ba19ed E3a: removed the split-KV A+B module `muse_full_decode_splitkv_v1` (CUDA 18,369 B + glue). It shows up
    on 0 of the 106 d16f/d32f judge rows across all 53 jobs since sk landed (10-27), and sk won every one. sk's B>1
    selector incumbent changes from split-KV A+B to unpaired FA3 + norm-gate (the B=1 incumbent, now for any B,
    tol 2e-2). If sk loses at B>1, the fallback is the master paired FA3 path.
  - aa9b247a E3b+c: removed the Triton `_align_parallel` (T >= 33 Marlin / Engine A dispatch when hot/cold is off).
    It appears in 0 of the 45 hot/cold-era jobs; its last judge appearance was 10-27. Also removed the uncalled
    `_w2_atomic_self_check` and the `w2_fill` plumbing. `state["w2_mode"] = "master"` (FP32 split-K reduce) is kept
    unchanged.
  - ed478481 E3d: pure diagnostics (swap-AB PTX-count report, w6p kill/expected and t128 Q3 static lines).
- Static proof: 874 of the 891 top-level statements are AST-identical to DC0b, and the other 17 are listed in
  research/e3-1030.md. Every Triton kernel kept is byte-identical, and so are 25 of the 27 long source strings (the
  two gone are `_SPLITKV_CPP` and `_SPLITKV_CUDA`). No compiled module's source or name changed except the one
  removed. okc_lint OK.
- Rules checklist (rules/technical-rules.llms.txt, checked against the diff):
  - [x] single file, load_inline only; nothing new compiled
  - [x] CUDA-graph capturable: kernel_fn lost two branches and gained no host sync (the removed `.item()`/sync
        calls were all in prepare)
  - [x] weights' precision unchanged; router: DC0b FP32 router at T <= 8 untouched (`_dc0*`, `ROUTER_FP32_MAX_TOKENS`)
  - [x] activations BF16 / FP8 KV / FP8 queries unchanged; the FP32 w2 split-K reduce pin kept (`w2_mode` master)
  - [x] attention attends every key (sk / c320 / skw / FA3 unchanged; the removed split-KV was a full-context kernel)
  - [x] no caching or replay of outputs, no harness tampering
  - [x] prepare < 300 s (node cold val below)
- Risks:
  - A judge where sk fails its checks at B>1 would now fall back to paired FA3, not split-KV A+B (slower). That has
    never happened on the judge (106/106 rows).
  - "Nearby shapes" (final review): at 33 <= T < 128, when hot/cold is off, the MoE dispatch now uses SGLang's
    moe_align_block_size instead of align_par. That costs a few us at non-scored T. It is not a scored row.
  - The sk B>1 selector now times sk against FA3 instead of split-KV. sk's margin over FA3 is larger, so the
    decision can only stay "sk".
- Read-out: route lines must match DC0b's judge log ([skfull] FULL-LINE d16f/d32f, c320t, b12p, ea6/ea4x/ea-enc,
  hh-wh, w6b, dc0b-fork). Headline within DC0b's CI is a pass.

## Result
- job `add080da32034f4c` (check), passed 13/13, **1.686x [1.661, 1.711]**: level with DC0b (checks 1.684/1.682, board 1.690). No regression from the prune; E3 is the base for decode ports (lands with the first port that scores).
