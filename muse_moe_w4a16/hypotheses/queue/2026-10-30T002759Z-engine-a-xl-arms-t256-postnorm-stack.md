---
problem: muse_moe_w4a16
slug: engine-a-xl-arms-t256-postnorm-stack
created: 2026-10-30T002759Z
base: master
mint_blob: "d80df5ecc71315466253dae86b28e4f6dfc41642"
best_at_mint: 2.0352
area: "moe expert block: Engine A XL dequant arms (all classes) + T=256 ea_combine replacing the post-norm weighted sum"
change: >
  Stack two shelved moe riders on master (2.0352, f2b510f5): (1) the Engine A XL arms from
  takemyjob hopper/ea-enc2-1029 04306fa (patch_enc2.py: E4XL / E5XL = E4/E5 + ldmatrix.x4 B
  preload + per-record loop trim; bit-exact; research/h100-ea-enc2-1029.md: consume-only
  1.014-1.055 over the routed arm, largest at T=256 E5XL 1.039-1.055) added as extra arms in
  master's per-class [ea-enc] race (same rules: exact, <= 0.98 vs master every round; at T=256
  also print the isolated cold-half ratio); (2) at T=256 only, the hybrid's post-norm weighted sum
  replaced by ea_combine with PDL (takemyjob hopper/glue-1029 f3d0fea; research/h100-glue-1029.md:
  x1.016 on the T=256 row, rel L2 4.19e-3 vs master 4.59e-3, not bitwise), raced at T=256 through
  kernel_fn with a tolerance gate (fp32 reference, _ea_canary's rule) and <= 0.99 every round,
  else master's post-norm. Keep the build/join deadlines (EAE_JOIN_S 220) and size < 1,000,000 B
  (moe is 948,330 B).
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "T=256: E5XL consume 1.039-1.055 over E5/E4 + ea_combine x1.016"
  - shapes: load=decode_16*
    direction: faster
    why: E4XL consume 1.02-1.047 over E4 at T8-32
  - shapes: load=decode_32*
    direction: faster
    why: E4XL
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "engine-a-a4b-a5b-encodings-race: landed E4/E5; this adds the XL loop trim and the T=256 combine swap that the ea-enc2 and glue lanes measured"
target: check
value_gain: 1.5
value_p: 0.5
value_ev: 0.75
value_cost: 400k tokens, 1 judge run
evidence: >
  H100 ea-enc2 (runs 20261029-160207-gpunode06): E4XL consume-only 1.014-1.047x at T8-32, E5XL
  1.039-1.055x at T=256 over the routed arm, SASS 617 -> 566 issued/record; H100 glue
  (research/h100-glue-1029.md): T=256 post-norm 10.0 us of a 321.4 us row, ea_combine+PDL x1.016
  (2 replicates x 50 pairs); judge 702f1004: consume gains transferred at ~60-70%
user_requested: true
review: required
status: queued
---

# engine-a-xl-arms-t256-postnorm-stack

## Change

Stack two shelved moe riders on master (2.0352, f2b510f5): (1) the Engine A XL arms from takemyjob hopper/ea-enc2-1029 04306fa (patch_enc2.py: E4XL / E5XL = E4/E5 + ldmatrix.x4 B preload + per-record loop trim; bit-exact; research/h100-ea-enc2-1029.md: consume-only 1.014-1.055 over the routed arm, largest at T=256 E5XL 1.039-1.055) added as extra arms in master's per-class [ea-enc] race (same rules: exact, <= 0.98 vs master every round; at T=256 also print the isolated cold-half ratio); (2) at T=256 only, the hybrid's post-norm weighted sum replaced by ea_combine with PDL (takemyjob hopper/glue-1029 f3d0fea; research/h100-glue-1029.md: x1.016 on the T=256 row, rel L2 4.19e-3 vs master 4.59e-3, not bitwise), raced at T=256 through kernel_fn with a tolerance gate (fp32 reference, _ea_canary's rule) and <= 0.99 every round, else master's post-norm. Keep the build/join deadlines (EAE_JOIN_S 220) and size < 1,000,000 B (moe is 948,330 B).

## Expected

- `load=verify_32*` faster: T=256: E5XL consume 1.039-1.055 over E5/E4 + ea_combine x1.016
- `load=decode_16*` faster: E4XL consume 1.02-1.047 over E4 at T8-32
- `load=decode_32*` faster: E4XL

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 ea-enc2 (runs 20261029-160207-gpunode06): E4XL consume-only 1.014-1.047x at T8-32, E5XL 1.039-1.055x at T=256 over the routed arm, SASS 617 -> 566 issued/record; H100 glue (research/h100-glue-1029.md): T=256 post-norm 10.0 us of a 321.4 us row, ea_combine+PDL x1.016 (2 replicates x 50 pairs); judge 702f1004: consume gains transferred at ~60-70%
- **Gain if it works:** 1.5% headline; **p** 0.5; **EV** 0.75% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.5% < 2.5%; EV 0.75% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge run.

## Risk

Each part ~+0.5-1%; stacked ~+1.5%. The E encodings' node consume-only gains transferred on the judge (702f1004: 4-10% consume -> 3-5% per class), so XL's 2-5% should show ~1.5-3% per class.
