---
problem: muse_bf16_proj
slug: bf16-lossless-exp4-warpspec-persistent-stage3
created: 2026-10-25T003715Z
base: master
area: lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)
change: >
  Build on the stage-2 branch of bf16-lossless-exp4-occupancy-dequant-once-stage2 (codec 12.17
  bits/elem, ratio 0.761, bit-exact; load_inline + per-route in-process self-check + fallback, its
  scope cut to pooler m 1/8/16/32, unpooler-full m 16/32 and qkv_gate m 1/8/16/32, o and the
  unpooler slice on master). Base stays bf16 master (the 1.2215 restack is not landed). First read
  the stage-2 run: its [lossless] occupancy print (CTAs/SM, warps/SM per NT) and its pooler m=1
  kernel time against the kill gate (<= 19 us). Stage 3 attacks what stage 1 and 2 leave: one warp
  per SMSP doing load waits, dequant and mma.sync serially. Change _lossless_swapab_kernel<NT> to
  a warp-specialized persistent kernel: (1) PRODUCER/CONSUMER: per CTA 1 producer warp that only
  issues plain cp.async.cg 16-byte copies of the compressed stream (no createpolicy or cache-hint
  variants, per LESSONS 2026-10-25) into a ring of S stages, signalling consumers through named
  barriers (bar.arrive / bar.sync with distinct IDs for full and empty slots), and 2 or 3 consumer
  groups of 4 warps that dequant and mma.sync; consumers never execute cp.async.wait_group. Target
  >= 12 resident warps/SM and print cudaOccupancyMaxActiveBlocksPerMultiprocessor plus the
  achieved warps/SM per NT. (2) DECODED-FRAGMENT PIPELINE: each consumer warp decodes k16 A
  fragments for step s+1 into a second register set while step s mma.sync chains issue
  (double-buffered registers, dequant exactly once per fragment, reused across all NT/8 token
  fragments); keep regs <= 128 at NT=32 (print regs and spills from cudaFuncGetAttributes, no
  cuobjdump). (3) PERSISTENT EVEN SPLIT: grid = SM count x CTAs/SM from the occupancy query
  (computed once in prepare, a constant); split each layer compressed groups across that grid as
  contiguous group ranges with equal BYTE counts (escape groups make sizes uneven, so build a
  prepare-time prefix table of group byte offsets), with a ticket-reduce of fp32 partials per
  output tile (as master swap-AB does: fixed summation order, self-resetting counter). (4) Keep
  the stage-2 cold-L2 prepare timing prints (flush a > L2 buffer before each event) per route next
  to the incumbent Triton time on the same synthetic x. Build: stays on mma.sync, but set
  TORCH_CUDA_ARCH_LIST=9.0a around load_inline anyway and add -lineinfo; in the self-check,
  synchronize and print after each kernel so any fault is attributed (an illegal instruction
  poisons the context; no child processes). Routing, codec, prepare-time compression and outputs
  unchanged; self-check stays bit-exact against stage 2. Run as a check only. If stage 2 missed
  the kill gate by more than 1.5x (pooler m=1 kernel > 28 us), mark this superseded instead of
  running it.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: "M2: incumbent Triton 19.0 us kernel / 18.8 row; raw floor 14.0, compressed floor 10.7; stage 1 ran 40.6 us at 4 warps/SM. Needs >= 2.7 TB/s compressed to tie, M3 15.5 us (~3.3 TB/s compressed)"
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: "M2: incumbent 19.3 us; floor 14.0/10.7; M3 <= 15.5"
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: "M2: incumbent 19.5; floor 14.1/10.7; dequant-once keeps NT growth flat"
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: "M2: incumbent swap-AB 20.5; floor 14.1/10.8"
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: "M2: incumbent nvjet 18.5 row; floor 14.1/10.7"
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: "M2: incumbent 18.1 row; floor 14.1/10.8"
  - shapes: layer=qkv_gate load=decode_* m=*
    direction: faster
    why: "M2 on cold-L2 rows: incumbent rows 12.2-14.2, floor 7.9-8.0 raw / 6.0 compressed; judge rows not profile (L2-resident weight)"
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: "M2: incumbent row 12.5, floor 7.9/6.0"
  - shapes: layer=o *
    direction: flat
    why: on master since stage 2
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: layer=* load=verify_16 m=128
    direction: flat
    why: m>=128 untouched
  - shapes: layer=* load=verify_32 m=256
    direction: flat
    why: m>=128 untouched
risk: high
depends_on: bf16-lossless-exp4-occupancy-dequant-once-stage2
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: master Triton weight streams at m<=32 for pooler (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m 17..32), unpooler-full (nvjet m=16, swap-AB m=32) and qkv_gate (_qkv_splitk_sem_kernel m<=16, swap-AB m 17..32)
review: hold
status: hold
review_at: 2026-10-25T010658Z
review_companions: null
hold_reason: "adversarial review: Stage 2's pooler m=1 kernel took 30.8 us in the judge profile, past this hypothesis's own supersede threshold (> 28 us) and 1.62x short of the 19.0 us kill gate"
---

# bf16-lossless-exp4-warpspec-persistent-stage3

## Change

Build on the stage-2 branch of bf16-lossless-exp4-occupancy-dequant-once-stage2 (codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-route in-process self-check + fallback, its scope cut to pooler m 1/8/16/32, unpooler-full m 16/32 and qkv_gate m 1/8/16/32, o and the unpooler slice on master). Base stays bf16 master (the 1.2215 restack is not landed). First read the stage-2 run: its [lossless] occupancy print (CTAs/SM, warps/SM per NT) and its pooler m=1 kernel time against the kill gate (<= 19 us). Stage 3 attacks what stage 1 and 2 leave: one warp per SMSP doing load waits, dequant and mma.sync serially. Change _lossless_swapab_kernel<NT> to a warp-specialized persistent kernel: (1) PRODUCER/CONSUMER: per CTA 1 producer warp that only issues plain cp.async.cg 16-byte copies of the compressed stream (no createpolicy or cache-hint variants, per LESSONS 2026-10-25) into a ring of S stages, signalling consumers through named barriers (bar.arrive / bar.sync with distinct IDs for full and empty slots), and 2 or 3 consumer groups of 4 warps that dequant and mma.sync; consumers never execute cp.async.wait_group. Target >= 12 resident warps/SM and print cudaOccupancyMaxActiveBlocksPerMultiprocessor plus the achieved warps/SM per NT. (2) DECODED-FRAGMENT PIPELINE: each consumer warp decodes k16 A fragments for step s+1 into a second register set while step s mma.sync chains issue (double-buffered registers, dequant exactly once per fragment, reused across all NT/8 token fragments); keep regs <= 128 at NT=32 (print regs and spills from cudaFuncGetAttributes, no cuobjdump). (3) PERSISTENT EVEN SPLIT: grid = SM count x CTAs/SM from the occupancy query (computed once in prepare, a constant); split each layer compressed groups across that grid as contiguous group ranges with equal BYTE counts (escape groups make sizes uneven, so build a prepare-time prefix table of group byte offsets), with a ticket-reduce of fp32 partials per output tile (as master swap-AB does: fixed summation order, self-resetting counter). (4) Keep the stage-2 cold-L2 prepare timing prints (flush a > L2 buffer before each event) per route next to the incumbent Triton time on the same synthetic x. Build: stays on mma.sync, but set TORCH_CUDA_ARCH_LIST=9.0a around load_inline anyway and add -lineinfo; in the self-check, synchronize and print after each kernel so any fault is attributed (an illegal instruction poisons the context; no child processes). Routing, codec, prepare-time compression and outputs unchanged; self-check stays bit-exact against stage 2. Run as a check only. If stage 2 missed the kill gate by more than 1.5x (pooler m=1 kernel > 28 us), mark this superseded instead of running it.

## Expected

- `layer=pooler load=decode_1 m=1` faster: M2: incumbent Triton 19.0 us kernel / 18.8 row; raw floor 14.0, compressed floor 10.7; stage 1 ran 40.6 us at 4 warps/SM. Needs >= 2.7 TB/s compressed to tie, M3 15.5 us (~3.3 TB/s compressed)
- `layer=pooler load=verify_1 m=8` faster: M2: incumbent 19.3 us; floor 14.0/10.7; M3 <= 15.5
- `layer=pooler load=decode_16 m=16` faster: M2: incumbent 19.5; floor 14.1/10.7; dequant-once keeps NT growth flat
- `layer=pooler load=decode_32 m=32` faster: M2: incumbent swap-AB 20.5; floor 14.1/10.8
- `layer=unpooler load=decode_16 m=16` faster: M2: incumbent nvjet 18.5 row; floor 14.1/10.7
- `layer=unpooler load=decode_32 m=32` faster: M2: incumbent 18.1 row; floor 14.1/10.8
- `layer=qkv_gate load=decode_* m=*` faster: M2 on cold-L2 rows: incumbent rows 12.2-14.2, floor 7.9-8.0 raw / 6.0 compressed; judge rows not profile (L2-resident weight)
- `layer=qkv_gate load=verify_1 m=8` faster: M2: incumbent row 12.5, floor 7.9/6.0
- `layer=o *` flat: on master since stage 2
- `layer=router *` flat: untouched
- `layer=* load=verify_16 m=128` flat: m>=128 untouched
- `layer=* load=verify_32 m=256` flat: m>=128 untouched

## Risk

Biggest risk: stage 2 may already show the codec cannot stream with mma.sync (kill gate); then this stage is moot. Second: named-barrier producer/consumer code can deadlock if a barrier thread count is wrong; the self-check prints before and after each launch. Milestones: M1 all 9 tests pass, every route PATH CHOSEN lossless; M2 pooler m=1 kernel <= 19.0 us and every routed row at or below its incumbent; M3 pooler <= 15.5 us, qkv rows <= 11.5 us. Headline if M3: decode loads lose ~3.5 us pooler + ~3 us unpooler + ~2 us qkv, about +6-8% on 4 of 6 loads.

## Milestones

Rewrite stage 3 of `master Triton weight streams at m<=32 for pooler (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m 17..32), unpooler-full (nvjet m=16, swap-AB m=32) and qkv_gate (_qkv_splitk_sem_kernel m<=16, swap-AB m 17..32)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T010658Z by an adversarial subagent (model opus).

**Verdict: hold.** Stage 2's pooler m=1 kernel took 30.8 us in the judge profile, past this hypothesis's own supersede threshold (> 28 us) and 1.62x short of the 19.0 us kill gate. Stage 2's log also shows the kernel is decode/issue-bound inside the SM, not memory-latency-bound, which is what warp-specialized cp.async producers and a persistent split target. Mark superseded; do not spend a check on it as written.

### Justification
Stage 2 (job f1024212dd2f49f3, 0.887x [0.874, 0.900], 9/9 pass, bit-exact): pooler m=1 30.8 us (incumbent 19.0), m=8/16/32 31.0/34.7/39.6; unpooler m=16/32 32.7/34.3 (incumbent ~18.5/18.1); qkv_gate m=1/8/16/32 18.9/19.2/22.2/25.7 (incumbent 12.2-14.2). Every routed row 1.5-2.1x slower. Stage 1->2 bought 1.32x for 2x warps; stage 3 needs another 1.62x to tie. Warm vs cold L2 barely differ (pooler 33.1 vs 40.3, unpooler 31.3 vs 38.6), and qkv_gate's L2-resident compressed stream runs at ~1.5 TB/s compressed vs the incumbent's ~3 TB/s raw from L2: decode ALU/issue and the mma.sync chain are the limit. Only part 2 (double-buffered decoded A fragments) targets it, and it competes for registers. NT growth +29-36% suggests dequant-once did not take effect or per-token work dominates. Compression ratio 0.761 caps the byte saving at ~24%; qkv_gate is L2-resident, so compression can only lose there.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton / bf16-shape-gated-triton-winners-combo: the incumbents; stage 2 misses by 1.5-2.1x, loss belongs to the lossless kernel.
- Base drift: BEST is the unlanded 1.2215 restack, whose incumbents are faster still on these rows.

### Interference and A/B needs
No companion. The deciding question is physics: can the exp4 decoder plus mma.sync reach >= 3.3 TB/s compressed? If revived, the next stage should first measure decode-only throughput from L2-resident data (no mma, no output) in prepare; if decode alone cannot beat ~2.7 TB/s compressed, the line is dead.

### Open questions
- Why does stage-2 kernel time grow 29-36% from NT=8 to NT=32 if dequant happens once per fragment?
- Decoder ALU ops per element?
- Drop qkv_gate from scope in any revival?
- Is this rewrite slot better spent elsewhere given ~+6-8% on 4 of 6 loads vs the 1.2215 restack?

