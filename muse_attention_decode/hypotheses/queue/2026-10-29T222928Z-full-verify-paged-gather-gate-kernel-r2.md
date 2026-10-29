---
problem: muse_attention_decode
slug: full-verify-paged-gather-gate-kernel-r2
created: 2026-10-29T222928Z
base: master
mint_blob: "2eb03f76ceefa9bf81e28616ced3b33be3a70443"
best_at_mint: 1.634
area: full-layer verify attention (FA3 call at verify_16 / verify_32)
change: >
  Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one
  build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the judge's PyTorch headers
  require C++20 (#error 'C++20 or later compatible compiler is required to use PyTorch.' in
  main.o), so the background build failed and master was kept (check 06c05172). Set -std=c++20 in
  _FVG_FLAGS, matching master's other extensions (c320, sliding window use '-std=c++20'), and also
  pass extra_cflags=['-O3', '-std=c++20'] to the fvg load_inline so the host main.cpp builds with
  C++20. Nothing else changes: same route gates, race (<= 0.95 per row), cooperative launch,
  layouts and [fvg] prints.
expected:
  - shapes: layer=full load=verify_16*
    direction: faster
    why: node e2e 0.83-0.84; H200 est 0.87-0.95
  - shapes: layer=full load=verify_32*
    direction: faster
    why: node e2e 0.82; H200 est 0.86-0.92
  - shapes: layer=sliding*
    direction: same
    why: not routed
risk: high
depends_on: full-verify-paged-gather-gate-kernel
priority: 1
kind: primary
controls_for: null
prior: "full-verify-paged-gather-gate-kernel: its kernel never ran (C++17 build flag failed against PyTorch's C++20 headers); this is the same change with -std=c++20"
target: check
value_gain: 3.0
value_p: 0.35
value_ev: 1.05
value_cost: 150k tokens, 1 judge run
evidence: >
  H100 fa3gate (research/h100-fa3gate-1029.md): e2e 272 vs 331 us at v32 (0.82), 147-151 vs
  176-181 us at v16 (0.83-0.84); judge board 4e386647 full verify_16 162.7 us, verify_32 302.1 us;
  judge 06c051723643462f fvg build failed on the C++17 flag, master kept
user_requested: true
review: required
status: queued
---

# full-verify-paged-gather-gate-kernel-r2

## Change

Same change as full-verify-paged-gather-gate-kernel (restacked from its patch), with the one build fix its judge run needed: _FVG_FLAGS used -std=c++17, and the judge's PyTorch headers require C++20 (#error 'C++20 or later compatible compiler is required to use PyTorch.' in main.o), so the background build failed and master was kept (check 06c05172). Set -std=c++20 in _FVG_FLAGS, matching master's other extensions (c320, sliding window use '-std=c++20'), and also pass extra_cflags=['-O3', '-std=c++20'] to the fvg load_inline so the host main.cpp builds with C++20. Nothing else changes: same route gates, race (<= 0.95 per row), cooperative launch, layouts and [fvg] prints.

## Expected

- `layer=full load=verify_16*` faster: node e2e 0.83-0.84; H200 est 0.87-0.95
- `layer=full load=verify_32*` faster: node e2e 0.82; H200 est 0.86-0.92
- `layer=sliding*` same: not routed

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 fa3gate (research/h100-fa3gate-1029.md): e2e 272 vs 331 us at v32 (0.82), 147-151 vs 176-181 us at v16 (0.83-0.84); judge board 4e386647 full verify_16 162.7 us, verify_32 302.1 us; judge 06c051723643462f fvg build failed on the C++17 flag, master kept
- **Gain if it works:** 3.0% headline; **p** 0.35; **EV** 1.05% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (EV 1.05% < 1.5%), minted because the user described it.
- **Cost:** 150k tokens, 1 judge run.

## Risk

06c05172 never ran the kernel: the extension failed to compile (C++17 flag vs PyTorch's C++20 requirement); its 1.6154 is master's level. The review of the parent (proceed; H200 re-sized to ~+2-3.5%) applies unchanged.
