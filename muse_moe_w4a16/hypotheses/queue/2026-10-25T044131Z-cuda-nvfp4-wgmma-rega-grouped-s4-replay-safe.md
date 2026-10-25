---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
created: 2026-10-25T044131Z
base: master
area: "wgmma NVFP4 grouped w13+silu/w2 at 128<=T<=512: per-state tensor maps, graph-replay safety"
change: >
  Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition
  2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py): the 8 TMA maps live in ONE
  process-global `Maps g_maps[2]` inside the shared load_inline module (:972). wg_set_maps encodes
  them over the CALLING state's wg_xs/wg_hs (:1026-1045; called from prepare at :2122-2124), and
  every launch passes g_maps[which] by value (:1087), which the graph then freezes. The judge
  prepares several copies (setup record `copies: 2`) and rotates them across repetitions. The last
  prepare() re-points g_maps at its own scratch, so on every other copy: the gather writes that
  copy's state["wg_xs"] (:1364) and w13 writes its state["wg_hs"] (:1368), but the w13 TMA reads
  the OTHER copy's xs and the w2 TMA reads the OTHER copy's hs. Those hold rows from whatever the
  other copy last ran. Reps 0/1 pass (the other copy's rows still match those inputs); rep 2 is
  the first replay where the other copy has run different inputs. It is not a rules violation:
  n_padded, chunks and buckets are read on the device (:836-889), and kernel_fn branches on shape
  only (:2364). The s3 self-check could not see this. It used one state, captured per case and
  replayed the SAME inputs 3 times (:1897-1900). FIX. Base: master 9080a71 (w2 atomic-add landed).
  T<128 and T>512 run master's Marlin unchanged. Port the s3 kernels with variant A
  (wgmma_wait<0>) ONLY in every bucket. Drop variant B, the vmask argument and the A/B canary
  timing. (1) Delete the g_maps/g_maps_ok globals. Add wg_make_maps(xs, hs), which returns a CPU
  uint8 tensor [2, sizeof(Maps)] stored in state["wg_maps"]. wg_gemm takes that tensor plus the
  B-source scratch tensor, memcpys row `which` into the __grid_constant__ Maps argument, and
  TORCH_CHECKs that the scratch data_ptr equals the one recorded when the maps were encoded (keep
  a per-blob int64 ptr pair in state). Each copy's graph then freezes maps over its OWN scratch.
  (2) Keep wg_xs/wg_hs/wg_err per state (no module-level scratch, no static device pointers
  anywhere). (3) Audit the module for any other static/global that holds a device pointer, and
  print the result. MANDATORY self-check before PATH CHOSEN=WGMMA. Any failure prints one `OFF:
  <reason>` line and master's Marlin runs. (a) Refill replay: for M in (128, 256, 257), allocate
  static x / topk_ids / topk_weights, capture kernel_fn ONCE, then for >=3 different routings
  (random, skewed 4-hot, all-one-expert, mixed local+remote) copy_ the new values into the SAME
  buffers, replay, and compare each output to eager Marlin on the same values (rel-L2 <= 1e-2, row
  bound 5x, no NaN/Inf). (b) Cross-copy: build a second state B with its own scratch and maps
  (call prepare() again, or the wgmma setup alone). Capture GA on state A and GB on state B, then
  run GA, GB, GA, GB, GA with a fresh routing refilled before each replay, and compare each to
  Marlin. (c) Print one line per replay: routing tag, rel-L2, and the map scratch ptr vs that
  state's buffer ptr. Keep the s3 cold timing lines (M=128/256 wgmma vs Marlin) unchanged. The
  first measurement is an okc check, never a submit.
expected:
  - shapes: "*load=verify 16*"
    direction: faster
    why: bs=128 passes every recheck; prepare cold wgmma 669 vs Marlin 921 us at M=128
  - shapes: "*load=verify 32*"
    direction: faster
    why: bs=256 cold wgmma 838 vs Marlin 1251 us at M=256
  - shapes: "*load=decode*,*load=verify 1*"
    direction: neutral
    why: T<128 runs master 9080a71 Marlin unchanged
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: Marlin w13+silu+w2 at 128<=T<=512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe

## Change

Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition 2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py): the 8 TMA maps live in ONE process-global `Maps g_maps[2]` inside the shared load_inline module (:972). wg_set_maps encodes them over the CALLING state's wg_xs/wg_hs (:1026-1045; called from prepare at :2122-2124), and every launch passes g_maps[which] by value (:1087), which the graph then freezes. The judge prepares several copies (setup record `copies: 2`) and rotates them across repetitions. The last prepare() re-points g_maps at its own scratch, so on every other copy: the gather writes that copy's state["wg_xs"] (:1364) and w13 writes its state["wg_hs"] (:1368), but the w13 TMA reads the OTHER copy's xs and the w2 TMA reads the OTHER copy's hs. Those hold rows from whatever the other copy last ran. Reps 0/1 pass (the other copy's rows still match those inputs); rep 2 is the first replay where the other copy has run different inputs. It is not a rules violation: n_padded, chunks and buckets are read on the device (:836-889), and kernel_fn branches on shape only (:2364). The s3 self-check could not see this. It used one state, captured per case and replayed the SAME inputs 3 times (:1897-1900).

FIX. Base: master 9080a71 (w2 atomic-add landed). T<128 and T>512 run master's Marlin unchanged. Port the s3 kernels with variant A (wgmma_wait<0>) ONLY in every bucket. Drop variant B, the vmask argument and the A/B canary timing. (1) Delete the g_maps/g_maps_ok globals. Add wg_make_maps(xs, hs), which returns a CPU uint8 tensor [2, sizeof(Maps)] stored in state["wg_maps"]. wg_gemm takes that tensor plus the B-source scratch tensor, memcpys row `which` into the __grid_constant__ Maps argument, and TORCH_CHECKs that the scratch data_ptr equals the one recorded when the maps were encoded (keep a per-blob int64 ptr pair in state). Each copy's graph then freezes maps over its OWN scratch. (2) Keep wg_xs/wg_hs/wg_err per state (no module-level scratch, no static device pointers anywhere). (3) Audit the module for any other static/global that holds a device pointer, and print the result.

MANDATORY self-check before PATH CHOSEN=WGMMA. Any failure prints one `OFF: <reason>` line and master's Marlin runs. (a) Refill replay: for M in (128, 256, 257), allocate static x / topk_ids / topk_weights, capture kernel_fn ONCE, then for >=3 different routings (random, skewed 4-hot, all-one-expert, mixed local+remote) copy_ the new values into the SAME buffers, replay, and compare each output to eager Marlin on the same values (rel-L2 <= 1e-2, row bound 5x, no NaN/Inf). (b) Cross-copy: build a second state B with its own scratch and maps (call prepare() again, or the wgmma setup alone). Capture GA on state A and GB on state B, then run GA, GB, GA, GB, GA with a fresh routing refilled before each replay, and compare each to Marlin. (c) Print one line per replay: routing tag, rel-L2, and the map scratch ptr vs that state's buffer ptr. Keep the s3 cold timing lines (M=128/256 wgmma vs Marlin) unchanged. The first measurement is an okc check, never a submit.

## Expected

- `*load=verify 16*` faster: bs=128 passes every recheck; prepare cold wgmma 669 vs Marlin 921 us at M=128
- `*load=verify 32*` faster: bs=256 cold wgmma 838 vs Marlin 1251 us at M=256
- `*load=decode*,*load=verify 1*` neutral: T<128 runs master 9080a71 Marlin unchanged

## Risk

Stage 4 of the cuda-nvfp4-wgmma-rega-grouped rewrite. It fixes s3 (3bdec0e58d684c17): every bucket canary, the self-check and the 5 judge tests passed, then a benchmark recheck failed on the first cross-copy replay. The held moe cuda-nvfp4-wgmma-rega-midm-bucket-cap32 and -w2-only-midm hypotheses and the prefill experts-turn1-cuda-wgmma-rega-port-bucket32 port all reuse the same _wg_* module, including the global g_maps. They must rebuild on this stage's commit (per-state maps plus the refill and cross-copy self-check), not on s2 or s3.

## Milestones

Rewrite stage 4 of `Marlin w13+silu+w2 at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
