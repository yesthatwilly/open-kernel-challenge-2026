---
problem: muse_prefill_layer
slug: attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload
created: 2026-10-25T152235Z
base: master
area: "FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 2: warp-specialized producer, 128-key tiles, intra-WG overlap + ping-pong, gated exp offload)"
change: >
  Rewrite stage 2 of the hand-written sm_90a FP8 full-prefill attention. Port onto current master
  (commit 1932a9c4; its attention code is unchanged from stage 1's base) the attention block of
  branch hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc (impl 227ec481;
  patch in done/neutral/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc.patch). Keep its
  infrastructure verbatim: _build_attn (TORCH_CUDA_ARCH_LIST=9.0a around load_inline,
  __CUDA_ARCH_FEAT_SM90_ALL guard, error word), _attn_image_check, _attn_canary (sync and print
  per launch), _attn_self_check, _attn_gate (cold-L2 CUDA-graph timing per full class against FA3
  with its combine), _full_attn_route and attn_combine_kernel. Replace only attn_fwd_kernel's
  body. Stage 1 was correct but 1.53-1.67x slower than FA3 on every full class because of its
  structure: 64-key tiles, every wgmma followed by wait_group 0, no warp specialization, 36 KB
  smem at 2 CTAs/SM, so the tensor work and the softmax never overlap. FA3 itself sits at ~600
  TFLOP/s on every full shape: 911.7 / 4542.4 / 8352.6 us on first_turn ctx 0 / 32768 / 65536,
  1643.1 us turn_4, 417.5 us turn_1. Those are 1.64-1.67x the MUFU exponential floor of 555 / 2770
  / 4990 / 1000 / 250 us (score elements T x keys x 16 heads at 16 ex2/clk/SM, 132 SMs, ~1.83
  GHz). With head_dim 64 there are only 256 MMA FLOP per exponential, so the bound is the softmax
  pipe, not the tensor cores. STAGE 2 KERNEL. (a) CTA = 3 warpgroups, 384 threads, 1 CTA/SM,
  persistent over stage 1's static longest-first item list (item = 16 tokens x the 8 GQA heads of
  one kv head = 128 query rows; split-KV only where stage 1 split). WG0 is the producer:
  setmaxnreg.dec to 40. WG1 and WG2 are consumers of 64 rows each: setmaxnreg.inc to 232. (b) Key
  tile 128: per stage 8 KB K + 8 KB V + 8 KB Vt, 3 stages. The producer prefetches the next tile's
  page_table indices into smem, gathers the 64-byte K and V rows of the item's kv head with plain
  cp.async.cg 16 B (no cache-hint forms, LESSONS 143), and signals a full mbarrier per stage with
  cp.async.mbarrier.arrive.noinc. It then transposes V into Vt (K-major for the FP8 PV operand,
  using stage 1's ldmatrix + prmt mapping), fences the async proxy, and arrives on a vt_full
  barrier. Consumers arrive on an empty barrier when the PV that read a stage retires. (c)
  Consumers: S(j+1) = Q K(j+1)^T by wgmma m64n128k32 e4m3 SS (Q staged in smem once per item),
  committed before the softmax of S(j), with wait_group 1 instead of 0 (FA3's intra-warpgroup
  overlap). Softmax of S(j): row max by quad shuffles, scale folded into one FFMA, exponentials,
  row sum, P packed to E4M3 with FA3's Max_offset 8 (LESSONS 144), and stage 1's
  accumulator-to-A-fragment permute. Then O += P(j) Vt(j) by m64n64k32 RS. Inter-warpgroup
  ping-pong on named barriers (bar.sync id, 256), so WG1's GEMMs overlap WG2's softmax. (d)
  Exponential offload as a template int EXP_SW in {0, 2, 3}: that many of every 8 S columns
  compute 2^x on the FMA pipe. Clamp x to [-127, 0] so masked -inf columns give ~0, never NaN.
  Then j = floor(x), f = x - j, a degree-3 minimax polynomial for 2^f on [0, 1) (rel err ~1e-4,
  far below E4M3's rounding of P), and an integer add of j << 23 into the exponent bits. The other
  columns use ex2.approx. Compile all three instances. Log ptxas registers and local-memory spill
  per instance (cudaFuncGetAttributes numRegs / localSizeBytes). Any instance with a local spill >
  0 is excluded from the gate (LESSONS 149). (e) Gate: per full class, time FA3 (with its combine)
  against each surviving instance, cold L2, median of >= 25 CUDA-graph replays. Route the fastest
  instance only if it is <= 0.95 x FA3, else FA3 exactly as master. Log '[cuda full attn s2] GATE
  <class>: FA3 X us, sw0 Y, sw2 Z, sw3 W -> <choice>' and PATH CHOSEN. Sliding, page-64 and
  kv-scaled cases always run FA3. The self-check (stage 1's cases plus one coherent-key
  long-context case, calibrated against an fp32 reference and FA3's own error rather than a fixed
  1e-2) runs for every instance before the gate. The first measurement must be an okc check, never
  a scored submit.
expected:
  - shapes: "*ctx=65536*layer=full*load=first_turn*"
    direction: faster
    why: "Replaced: FA3 8352.6 us of 18349.9 (45%), exp floor 4990 us. M1: <= FA3. M2: <= 0.8x FA3 (~6680 us, -1670 us, -9% of the shape). Stretch 0.7x (~1.17x floor). Stage 1 was 1.53-1.67x FA3."
  - shapes: "*ctx=32768*layer=full*load=first_turn*"
    direction: faster
    why: "Replaced: FA3 4542.4 us of 14825.7 (30%), floor 2770. At M2 -900 us, -6% of the shape."
  - shapes: "*ctx=0*layer=full*load=first_turn*"
    direction: faster
    why: "Replaced: FA3 911.7 us of 11599.6, floor 555. At M2 -180 us, -1.6% of the shape."
  - shapes: "*layer=full*load=turn_4*"
    direction: faster
    why: "Replaced: FA3 1643.1 us of 3713.8 (42%), floor 1000. At M2 -330 us, -9% of the shape. Full-layer FA3 is 16.6% of the turn_4 load, so M2 is ~3.3% of that load."
  - shapes: "*layer=full*load=turn_1*"
    direction: faster
    why: "Replaced: FA3 417.5 + combine 6.9 us of 1137.5, floor 250. Thin grid (64 items on 132 SMs, so it needs stage 1's split). At M2 -85 us, -7% of the shape."
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 is kept on sliding layers (25.3 / 61.8 / 306.4 us).
risk: high
depends_on: attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1"
review: required
status: queued
---

# attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload

## Change

Rewrite stage 2 of the hand-written sm_90a FP8 full-prefill attention. Port onto current master (commit 1932a9c4; its attention code is unchanged from stage 1's base) the attention block of branch hyp/muse_prefill_layer/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc (impl 227ec481; patch in done/neutral/attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc.patch). Keep its infrastructure verbatim: _build_attn (TORCH_CUDA_ARCH_LIST=9.0a around load_inline, __CUDA_ARCH_FEAT_SM90_ALL guard, error word), _attn_image_check, _attn_canary (sync and print per launch), _attn_self_check, _attn_gate (cold-L2 CUDA-graph timing per full class against FA3 with its combine), _full_attn_route and attn_combine_kernel. Replace only attn_fwd_kernel's body. Stage 1 was correct but 1.53-1.67x slower than FA3 on every full class because of its structure: 64-key tiles, every wgmma followed by wait_group 0, no warp specialization, 36 KB smem at 2 CTAs/SM, so the tensor work and the softmax never overlap. FA3 itself sits at ~600 TFLOP/s on every full shape: 911.7 / 4542.4 / 8352.6 us on first_turn ctx 0 / 32768 / 65536, 1643.1 us turn_4, 417.5 us turn_1. Those are 1.64-1.67x the MUFU exponential floor of 555 / 2770 / 4990 / 1000 / 250 us (score elements T x keys x 16 heads at 16 ex2/clk/SM, 132 SMs, ~1.83 GHz). With head_dim 64 there are only 256 MMA FLOP per exponential, so the bound is the softmax pipe, not the tensor cores. STAGE 2 KERNEL. (a) CTA = 3 warpgroups, 384 threads, 1 CTA/SM, persistent over stage 1's static longest-first item list (item = 16 tokens x the 8 GQA heads of one kv head = 128 query rows; split-KV only where stage 1 split). WG0 is the producer: setmaxnreg.dec to 40. WG1 and WG2 are consumers of 64 rows each: setmaxnreg.inc to 232. (b) Key tile 128: per stage 8 KB K + 8 KB V + 8 KB Vt, 3 stages. The producer prefetches the next tile's page_table indices into smem, gathers the 64-byte K and V rows of the item's kv head with plain cp.async.cg 16 B (no cache-hint forms, LESSONS 143), and signals a full mbarrier per stage with cp.async.mbarrier.arrive.noinc. It then transposes V into Vt (K-major for the FP8 PV operand, using stage 1's ldmatrix + prmt mapping), fences the async proxy, and arrives on a vt_full barrier. Consumers arrive on an empty barrier when the PV that read a stage retires. (c) Consumers: S(j+1) = Q K(j+1)^T by wgmma m64n128k32 e4m3 SS (Q staged in smem once per item), committed before the softmax of S(j), with wait_group 1 instead of 0 (FA3's intra-warpgroup overlap). Softmax of S(j): row max by quad shuffles, scale folded into one FFMA, exponentials, row sum, P packed to E4M3 with FA3's Max_offset 8 (LESSONS 144), and stage 1's accumulator-to-A-fragment permute. Then O += P(j) Vt(j) by m64n64k32 RS. Inter-warpgroup ping-pong on named barriers (bar.sync id, 256), so WG1's GEMMs overlap WG2's softmax. (d) Exponential offload as a template int EXP_SW in {0, 2, 3}: that many of every 8 S columns compute 2^x on the FMA pipe. Clamp x to [-127, 0] so masked -inf columns give ~0, never NaN. Then j = floor(x), f = x - j, a degree-3 minimax polynomial for 2^f on [0, 1) (rel err ~1e-4, far below E4M3's rounding of P), and an integer add of j << 23 into the exponent bits. The other columns use ex2.approx. Compile all three instances. Log ptxas registers and local-memory spill per instance (cudaFuncGetAttributes numRegs / localSizeBytes). Any instance with a local spill > 0 is excluded from the gate (LESSONS 149). (e) Gate: per full class, time FA3 (with its combine) against each surviving instance, cold L2, median of >= 25 CUDA-graph replays. Route the fastest instance only if it is <= 0.95 x FA3, else FA3 exactly as master. Log '[cuda full attn s2] GATE <class>: FA3 X us, sw0 Y, sw2 Z, sw3 W -> <choice>' and PATH CHOSEN. Sliding, page-64 and kv-scaled cases always run FA3. The self-check (stage 1's cases plus one coherent-key long-context case, calibrated against an fp32 reference and FA3's own error rather than a fixed 1e-2) runs for every instance before the gate. The first measurement must be an okc check, never a scored submit.

## Expected

- `*ctx=65536*layer=full*load=first_turn*` faster: Replaced: FA3 8352.6 us of 18349.9 (45%), exp floor 4990 us. M1: <= FA3. M2: <= 0.8x FA3 (~6680 us, -1670 us, -9% of the shape). Stretch 0.7x (~1.17x floor). Stage 1 was 1.53-1.67x FA3.
- `*ctx=32768*layer=full*load=first_turn*` faster: Replaced: FA3 4542.4 us of 14825.7 (30%), floor 2770. At M2 -900 us, -6% of the shape.
- `*ctx=0*layer=full*load=first_turn*` faster: Replaced: FA3 911.7 us of 11599.6, floor 555. At M2 -180 us, -1.6% of the shape.
- `*layer=full*load=turn_4*` faster: Replaced: FA3 1643.1 us of 3713.8 (42%), floor 1000. At M2 -330 us, -9% of the shape. Full-layer FA3 is 16.6% of the turn_4 load, so M2 is ~3.3% of that load.
- `*layer=full*load=turn_1*` faster: Replaced: FA3 417.5 + combine 6.9 us of 1137.5, floor 250. Thin grid (64 items on 132 SMs, so it needs stage 1's split). At M2 -85 us, -7% of the shape.
- `*layer=sliding*` flat: FA3 is kept on sliding layers (25.3 / 61.8 / 306.4 us).

## Risk

High risk: FA3-class warp-specialized engineering written without a local GPU. Stage 1's core is the only proven piece (correct, better than FA3 against fp64). The gate makes every outcome flat-or-better by construction and the GATE lines make it readable per class and per exp-offload fraction, so a run where everything gates to FA3 is a measurement, not a loss. Headline stake at M2: full-layer FA3 is 12.7% / 16.6% / 9.8% of the turn_1 / turn_4 / first_turn loads, so 0.8x FA3 is worth ~2.5-3.3% per load, ~+2.5% on the headline (more than every other attention lever combined). The main sub-risk is that the polynomial costs FP32 issue slots (Hopper has no FFMA2): ~6-7 instructions per offloaded element, which is why the fraction is gated and EXP_SW 0 is a candidate too. setmaxnreg needs the 3-WG launch shape exactly (384 threads, __launch_bounds__(384, 1)); a mismatch is an illegal-instruction fault that poisons the context, so build with -lineinfo and run only as an okc check. Runs after tomorrow's 00:00Z reset; independent of fa3-full-contiguous-kv-triton-gather-gated: if that lands, its gate should compare against the contiguous FA3 (the incumbent) rather than the paged one.

## Milestones

Rewrite stage 2 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
