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
review: proceed
status: running
review_at: 2026-10-25T044714Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T050458Z
base_sha: 8261c0480931bf1fdcb44c0d04d1dbf675b0c06a
base_blob: bf65a3afdf6ebe0d9b73a817ffd2be1240f91870
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
mode: null
restack_of: null
stale_base: true
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

## Adversarial review

Reviewed 2026-10-25T044714Z by an adversarial subagent (model opus).

**Verdict: proceed.** Diagnosis confirmed from the s3 branch (hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix; its worktree no longer exists). The process-global Maps g_maps[2] / g_maps_ok / g_map_rows[2] (s3 submission.py:972-974) are re-encoded by every prepare() over that state's wg_xs/wg_hs (:1026-1047, called at :2122-2124) and passed by value at launch (:1087), so a graph captured on one judge copy TMA-reads the other copy's scratch. No other module-global holds a device pointer in the s3 module, so the planned per-state map blob is enough. Proceed; it is a rewrite stage with a sound fix and the right self-check.

### Justification
A: The mechanism holds. The gather (:1364) and the w13 output (:1368) use state['wg_xs'] and state['wg_hs'], but both TMA reads go through g_maps[which], which encodes the LAST prepared state's buffers. The last-prepared copy is self-consistent. The other copy reads stale rows, which matches 'passes tests and reps 0/1, fails recheck rep 2 at rel-L2 1.21'. The full audit of the s3 C++ found nothing else that holds a pointer. The other C++ statics are constexpr or static_assert (Smem<G>::BYTES and the like); get_encode re-queries every time (no static cache); cudaFuncSetAttribute is per-function and idempotent. On the Python side, WG_BUILD_INFO (:1135) is diagnostic only. wg_err (:2135), wg_banks (:2130), wg_vmask (:2146) and wg_xs/wg_hs are already per state. The load_inline module is shared across prepare() calls in one process (same name, so the same .so), and that is exactly why any C++ global is cross-copy state. The g_maps_ok flip in wg_set_maps (:1030) is a second hazard: if a later prepare's encode fails, every copy's next launch dies on TORCH_CHECK, not only that copy's. The per-state blob removes this too. Weakest link: the expected speedups are the s3 prepare-time cold timings (669 vs 921 us at M=128, 838 vs 1251 us at M=256). No judge benchmark or profile of the wgmma path exists yet, because s3 failed before timing. Against floors of 101.3 / 108.9 us (TASK.md), both paths are far off-floor, so milestone 2 (at or below the incumbent) is plausible but unproven under judge timing, and milestone 3 is not expected this stage.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, the base): changes only the T<=32 w2 Marlin call; the wgmma path covers 128<=T<=512, so it is disjoint. Neutral control shapes stay neutral.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: the wgmma path calls its own moe_align_block_size at block 16 (s3 :1356) instead of master's align, so the align change is part of the replaced subsystem and is not separable from the kernel time. Read the profile per kernel (align + gather + 2x wg_kernel + norm vs align + 2x Marlin + act_and_mul + norm), not the headline.
- marlin-block-m-two-smaller-on-postnorm: tunes Marlin's block_size_m at T>=128, i.e. the incumbent being replaced. The comparison is against this tuned Marlin, which is correct and is not a confounder.

### Interference and A/B needs
No companions. This is a rewrite stage whose purpose is to be correct under the judge's multi-copy rotation. The control is built in: T<128 and T>512 run master Marlin, and the prepare-time cold A/B against Marlin at M=128/256 is kept. An ab-revert of a landed hypothesis would not change whether the fix works.

### Open questions
- Does the judge capture each copy's graph after ALL prepares, or right after its own prepare? The per-state fix is correct either way, but it decides whether s3's failure could ever have shown on rep 0. The self-check should cover both orders: capture GA before and after building B.
- Do the judge's copies hold distinct weight tensors (different data_ptr / values)? That does not matter for moe (wg_banks is per state), but it decides whether the bf16 deepgemm g_tb global (below) is latent or live.

