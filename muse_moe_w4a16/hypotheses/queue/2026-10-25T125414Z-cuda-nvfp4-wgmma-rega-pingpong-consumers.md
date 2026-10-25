---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-pingpong-consumers
created: 2026-10-25T125414Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Consumer-side schedule only, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate's
  commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, job
  7be125e9e1ea42e3), rebased onto master. MECHANISM: in s5's consume_item both consumer warpgroups
  (WG1 rows 0-63, WG2 rows 64-127 of the tile) run in lockstep on the same stage: mbar_wait(full)
  -> dq_stage (pure ALU, ~24 instr per k16 step per thread) -> 8 register-A wgmmas -> commit ->
  wgmma_wait<0> -> arrive(empty). So the SM's ALUs idle while the tensor core runs the 8-deep
  dependent wgmma chain plus the wait, and the tensor core idles during both dequants. The
  b128-n32-deep-ring run (job 6759e36bbd4648fc) showed weight bytes in flight are not the limit:
  w13 ran at 1.83-2.13 TB/s at 37, 49 and 64.5 KB per SM. That is ~1150 clk per 9216-byte stage
  per SM, against ~500 clk of dequant issue. s3's variant B (double-buffered a[] inside one
  warpgroup, wait<1>) was no faster, plausibly because ptxas serializes register-A wgmmas whose
  source registers are rewritten. CHANGE: a two-consumer ping-pong with s5's variant A kept inside
  each warpgroup, so no register-A fragment is ever rewritten with a group in flight. Add two
  named barriers, PP_A = 2 and PP_B = 3, each with count 256 (bar 1 stays the epilogue barrier).
  Per k-stage, WG1 does: [bar.sync PP_A, except on its very first k-stage of the launch] dq_stage;
  fence; 8 wgmma; commit; bar.arrive PP_B; wgmma_wait<0>; arrive(empty). WG2 does: bar.sync PP_B;
  dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A; wgmma_wait<0>; arrive(empty). So WG2
  dequantizes stage s while WG1's wgmmas for s run, and WG1 dequantizes s+1 while WG2's run. After
  the item loop, WG1 does one final bar.sync PP_A so every WG2 arrive is matched (equal counts:
  both warpgroups walk the same items and the same KS per item). The epilogue's named_bar(1, 256)
  is unchanged and also re-aligns the pair at each item boundary. Count invariants for review: WG1
  syncs on PP_A = total k-stages - 1 + 1 final = WG2 arrives on PP_A; WG2 syncs on PP_B = WG1
  arrives on PP_B. The `dead` path (mbar_wait timeout) must still execute every bar op, then skip
  the math, so a timeout can never turn into a named-barrier hang. Nothing else moves: weight
  bank, producer, ring (STAGES 4), buckets, gather, epilogues, per-call tensor maps, the s5
  selector and the kernel_fn gate. BUILD: keep s5's wg_kernel as variant `s5` and add the
  ping-pong as a template flag PP (so two instantiations per kernel). Bump the load_inline module
  name; TORCH_CUDA_ARCH_LIST=9.0a; -std=c++20 -O3 -lineinfo; print regs/spill per instantiation
  and the ptxas wgmma serialization note if visible. PREPARE: s5's per-bucket canary
  (N16/N32/N64/N128) and three-state copy check (A, deepcopy B, tensor clone C; refilled graph
  replays; launch marker) on the PP kernel, at M=128, 255, 256, 257 and all four routings
  including all-one-expert. Compare against the fp32-dequant reference; the bound is <=1.5x
  Marlin's own error. PP's outputs should be bitwise identical to s5's (same math, same order), so
  print max |PP - s5| and expect 0. SELECTOR: s5's graph-replay selector (cold L2, 41 interleaved
  reps, empty-graph baseline, margin 1.02x on the judge-like zipf a=0.3 routing) with variant
  candidates {s5, PP} vs Marlin per bucket. Also print the deep-ring run's per-kernel diagnostic
  line (touched experts, max routes per expert, w13 and w2 us and TB/s) for both variants at T=128
  on zipf a=0.3 AND the 17-touched judge-like routing, and at T=256. Assert that the chosen
  variant is the printed winner. Hard rules: CUDA-graph capturable; kernel_fn does no host sync,
  allocation or branching on input values, no prints, no try/except, no per-call mutable state;
  nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers); activations BF16;
  attend every route; no child processes. No module or static global may hold device pointers, TMA
  maps or buffers, because the judge clones TWO state copies (pointer-equality gates fail).
expected:
  - shapes: bs=256 *
    direction: faster
    why: "B256 is always wgmma. s5's judged rows were w13+silu 239-300 us and w2 119-148 us, ~1.85-2.0 TB/s, against a 108.9 us shape floor. If dequant/MMA serialization is the limiter, ping-pong hides the 8-wgmma chain plus wait under the other warpgroup's dequant. Target w13 <=190, w2 <=95, shape row <=340 us (from 405)"
  - shapes: bs=128 *
    direction: faster
    why: "B128 flips from Marlin (306 us judged pair) only if the selector's zipf a=0.3 ratio rises from 0.80x to >=1.02x, which needs ~1.28x on the wgmma kernels. The 17-touched judge-like routing was already 1.015x. Read the printed per-kernel TB/s lines even if the gate stays MARLIN: they attribute the consumer limit"
  - shapes: bs=32 *
    direction: flat
    why: "control: T<128 runs master's Marlin path"
  - shapes: bs=1 *
    direction: flat
    why: control
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-pingpong-consumers

## Change

Consumer-side schedule only, built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, job 7be125e9e1ea42e3), rebased onto master. MECHANISM: in s5's consume_item both consumer warpgroups (WG1 rows 0-63, WG2 rows 64-127 of the tile) run in lockstep on the same stage: mbar_wait(full) -> dq_stage (pure ALU, ~24 instr per k16 step per thread) -> 8 register-A wgmmas -> commit -> wgmma_wait<0> -> arrive(empty). So the SM's ALUs idle while the tensor core runs the 8-deep dependent wgmma chain plus the wait, and the tensor core idles during both dequants. The b128-n32-deep-ring run (job 6759e36bbd4648fc) showed weight bytes in flight are not the limit: w13 ran at 1.83-2.13 TB/s at 37, 49 and 64.5 KB per SM. That is ~1150 clk per 9216-byte stage per SM, against ~500 clk of dequant issue. s3's variant B (double-buffered a[] inside one warpgroup, wait<1>) was no faster, plausibly because ptxas serializes register-A wgmmas whose source registers are rewritten. CHANGE: a two-consumer ping-pong with s5's variant A kept inside each warpgroup, so no register-A fragment is ever rewritten with a group in flight. Add two named barriers, PP_A = 2 and PP_B = 3, each with count 256 (bar 1 stays the epilogue barrier). Per k-stage, WG1 does: [bar.sync PP_A, except on its very first k-stage of the launch] dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B; wgmma_wait<0>; arrive(empty). WG2 does: bar.sync PP_B; dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A; wgmma_wait<0>; arrive(empty). So WG2 dequantizes stage s while WG1's wgmmas for s run, and WG1 dequantizes s+1 while WG2's run. After the item loop, WG1 does one final bar.sync PP_A so every WG2 arrive is matched (equal counts: both warpgroups walk the same items and the same KS per item). The epilogue's named_bar(1, 256) is unchanged and also re-aligns the pair at each item boundary. Count invariants for review: WG1 syncs on PP_A = total k-stages - 1 + 1 final = WG2 arrives on PP_A; WG2 syncs on PP_B = WG1 arrives on PP_B. The `dead` path (mbar_wait timeout) must still execute every bar op, then skip the math, so a timeout can never turn into a named-barrier hang. Nothing else moves: weight bank, producer, ring (STAGES 4), buckets, gather, epilogues, per-call tensor maps, the s5 selector and the kernel_fn gate. BUILD: keep s5's wg_kernel as variant `s5` and add the ping-pong as a template flag PP (so two instantiations per kernel). Bump the load_inline module name; TORCH_CUDA_ARCH_LIST=9.0a; -std=c++20 -O3 -lineinfo; print regs/spill per instantiation and the ptxas wgmma serialization note if visible. PREPARE: s5's per-bucket canary (N16/N32/N64/N128) and three-state copy check (A, deepcopy B, tensor clone C; refilled graph replays; launch marker) on the PP kernel, at M=128, 255, 256, 257 and all four routings including all-one-expert. Compare against the fp32-dequant reference; the bound is <=1.5x Marlin's own error. PP's outputs should be bitwise identical to s5's (same math, same order), so print max |PP - s5| and expect 0. SELECTOR: s5's graph-replay selector (cold L2, 41 interleaved reps, empty-graph baseline, margin 1.02x on the judge-like zipf a=0.3 routing) with variant candidates {s5, PP} vs Marlin per bucket. Also print the deep-ring run's per-kernel diagnostic line (touched experts, max routes per expert, w13 and w2 us and TB/s) for both variants at T=128 on zipf a=0.3 AND the 17-touched judge-like routing, and at T=256. Assert that the chosen variant is the printed winner. Hard rules: CUDA-graph capturable; kernel_fn does no host sync, allocation or branching on input values, no prints, no try/except, no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers); activations BF16; attend every route; no child processes. No module or static global may hold device pointers, TMA maps or buffers, because the judge clones TWO state copies (pointer-equality gates fail).

## Expected

- `bs=256 *` faster: B256 is always wgmma. s5's judged rows were w13+silu 239-300 us and w2 119-148 us, ~1.85-2.0 TB/s, against a 108.9 us shape floor. If dequant/MMA serialization is the limiter, ping-pong hides the 8-wgmma chain plus wait under the other warpgroup's dequant. Target w13 <=190, w2 <=95, shape row <=340 us (from 405)
- `bs=128 *` faster: B128 flips from Marlin (306 us judged pair) only if the selector's zipf a=0.3 ratio rises from 0.80x to >=1.02x, which needs ~1.28x on the wgmma kernels. The 17-touched judge-like routing was already 1.015x. Read the printed per-kernel TB/s lines even if the gate stays MARLIN: they attribute the consumer limit
- `bs=32 *` flat: control: T<128 runs master's Marlin path
- `bs=1 *` flat: control

## Risk

Medium risk. Named barriers have no timeout, so a mismatched arrive/sync count hangs the launch. The count invariants are in the change text: equal items and KS per warpgroup, one final sync, and every bar op still executed on the dead path. A reviewer should hand-simulate items with KS=1 and with KS of 16 and 32 (K/128 for w13 = 32, w2 = 16). Bitwise equality with s5 (max |PP - s5| = 0) is the cheapest correctness proof, since only the issue order changes. If PP is flat on both kernels, ALU issue (not serialization) is the limit, and the sibling cuda-nvfp4-wgmma-rega-producer-scale-expand is the one to read. The two are independent and each is attributable on its own w13/w2 rows; stack them only after both are read. Do not confuse this with s3 variant B (double-buffered a[] inside one warpgroup), which was flat. This keeps variant A inside each warpgroup and only offsets the two. Judge on the bs=256 wg_kernel rows and on the prepare TB/s lines, not the headline (bs=128/256 move ~1.5% between runs).

## Milestones

Rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
