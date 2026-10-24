---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-w13-silu-w2
created: 2026-10-24T165851Z
base: master
area: whole-layer fusion
change: >
  bs=1 only (num_tokens == 1: decode_1 and the bs=1 test); every other M runs master unchanged.
  Replace the whole bs=1 MoE (_align_small, Marlin w13, silu_and_mul, Marlin w2,
  _post_norm_combine_kernel: 5 launches, 51.2 us of profiled GPU time) with ONE persistent CUDA
  kernel compiled in prepare() with torch.utils.cpp_extension.load_inline (one TU, -O3
  -arch=sm_90a; pattern from muse_attention_decode _build_splitkv). At bs=1 the intermediate is at
  most 8 routes x I=4096 bf16 = 64 KB, so it never needs a full round trip through a second
  launch. ROUTING IN-KERNEL: every CTA reads the 8 topk_ids and topk_weights of the token, maps id
  - local_start into [0, local_count) (bank start/count are kernel args from prepare), and
  compacts the local routes into R <= 8 (slot, expert, weight) entries in smem, in slot order
  (deterministic, identical in every CTA). Remote slots are skipped; R == 0 makes the last CTA
  write a zero row. WEIGHT LAYOUT (prepare, lossless): the same NVFP4 repack as
  cuda-nvfp4-swapab-mma-persistent-small-m (mma.sync m16n8k16 A-fragment nibble order, one 16 B
  cp.async per lane per 16-row x 64-k sub-tile; e4m3 block scales permuted to 8 B per lane group;
  w13 gate/up rows interleaved per tile; gs folded as gs * 2^7; round-trip assert). If that
  hypothesis has been implemented, reuse its device functions verbatim; otherwise implement them
  here as specified there (dequant: v = (q & 0x80008000) | ((q & 0x70007000) >> 6), q <<= 4, times
  a bf16x2 scale from cvt.rn.f16x2.e4m3x2 x 2^119, exact). PHASES, ONE GRID: grid = 132 CTAs (1
  per SM, 256 threads = 8 warps, launch_bounds(256, 1), ~100 KB dynamic smem so residency is
  guaranteed). Global item order: all phase-1 items first, then all phase-2 items; CTA c takes
  items c, c+132, ... as one flat (item, k-stage) cp.async sequence (STAGES = 4, BK = 256, 3
  stages = ~27-54 KB of weight in flight per SM), so the ring never drains between items or
  phases. Phase 1 (w13 + silu): item = (route r, w13 tile j) with 64 weight rows (32 gate + 32 up,
  matching columns), full K = 4096 (16 stages), 128 tiles per route -> 128R items (256 at R = 2:
  ~2 per CTA, equal bytes, 144 KB each). The token x (8 KB bf16) is loaded once into smem per CTA.
  mma.sync with the weight as A and the token in column 0 of the n8 B fragment (the other 7
  columns zero; MMA throughput is not the limit, it gives fp32 accumulation for free). Epilogue:
  acc * gs_eff13 -> silu(gate) * up in fp32 -> bf16 h[r, 32j .. 32j+32) into a device scratch
  (prepare-allocated [8, I] bf16), then st.release.gpu flag[r][j / 8] += 1 (one flag per
  256-column chunk of h, 8 tiles each). Phase 2 (w2): item = (route r, w2 tile n of 128 H rows,
  k-split s of 4 over I, 1024 columns each) -> 128R items, 72 KB each. The CTA issues its weight
  cp.async stages for this item BEFORE waiting (weights do not depend on h, so the wait overlaps
  the stream), then spins with ld.acquire.gpu on the 4 flags covering its 1024 columns until each
  equals epoch_target, then loads that h slice (2 KB) into smem and runs the MMA loop. Partial
  results: fp32 [4, 8, H] scratch; per (r, n) ticket: store partial, __threadfence, atomicAdd; the
  last of the 4 splits sums them in split order (ld.global.cg), multiplies by gs_eff2, writes
  bf16-rounded y[r, 128n .. 128n+128) to scratch (bf16 rounding matches master, which keeps y in
  bf16), resets the ticket, and bumps a per-token done counter. Phase 3 (post-norm combine): the
  CTA whose done counter increment completes R x 32 tiles computes out[h] = sum over r of g_r *
  rmsnorm(y_r * (1 + post_norm_w)) in fp32 (eps 1e-8, the _post_norm_combine_kernel formula),
  writes bf16 out [1, H], and resets the done counter. EPOCHS, NO MEMSETS: a device int epoch
  (prepare-zeroed) is read by every CTA at entry; flags count up and phase 2 waits for flag == 8 *
  (epoch + 1) with flags never reset (int32 wrap is ~2^28 calls away); the CTA that finishes phase
  3 (or the last CTA by an exit ticket when R == 0) increments epoch after a __threadfence.
  Deadlock freedom: all 132 CTAs are co-resident (1 per SM, grid == SM count, asserted in prepare
  from the device properties and the cudaOccupancyMaxActiveBlocksPerMultiprocessor result >= 1)
  and every CTA finishes all its phase-1 items before its first phase-2 wait, so every awaited
  tile is owned by a CTA that is not waiting. kernel_fn launches it on the current stream with no
  host branch on data (CUDA-graph safe); out is torch.empty. CORRECTNESS SELF-CHECK (prepare,
  untimed, fixed torch.Generator, real bank weights): bs=1 cases R = 0, 1, 2, 3, 8 local routes
  (incl. all-remote and all-local), local_start 0, run the fused kernel 50 times in a row (to
  exercise epoch and ticket reuse) and compare out against master path on the same inputs; rel L2
  <= 1e-2 per case, print max rel-L2, regs, localSizeBytes, smem and PATH CHOSEN loudly
  ([cuda-moe-bs1] SELF-CHECK PASSED/FAILED). Also run it once under torch.cuda.graph capture +
  replay in prepare to prove capture works. Assert tickets and done counter are 0 afterwards.
  FALLBACK: a load_inline exception, a failed check, a failed residency assert or a graph-capture
  error sets the flag off and bs=1 runs master byte for byte.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "decode_1 today: 48.4 us scored, 51.2 us profiled over 5 launches (Marlin pair 44.6, post-norm 3.4, silu 1.6, align 1.6) for ~2 local experts (~55 MB, floor 11.4 us). One persistent kernel at ~3.5 TB/s with phase-2 weight prefetch overlapping phase 1 and no inter-launch gaps targets <= 20 us for the whole MoE (M2 <= 46 us, M3 <= 18 us): decode_1 alone ~2.4x, headline ~+16%"
  - shapes: bs=8 *
    direction: flat
    why: "control: num_tokens > 1 runs master byte for byte"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens > 1 runs master byte for byte"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens > 1 runs master byte for byte"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32
review: proceed
status: running
review_at: 2026-10-24T170224Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T170224Z
base_sha: b9590f4d98a9e3d4afd62cdaab9dd4d48694655f
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-w13-silu-w2
worktree: .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-w13-silu-w2
mode: null
---

# cuda-fused-moe-bs1-persistent-w13-silu-w2

## Change

bs=1 only (num_tokens == 1: decode_1 and the bs=1 test); every other M runs master unchanged. Replace the whole bs=1 MoE (_align_small, Marlin w13, silu_and_mul, Marlin w2, _post_norm_combine_kernel: 5 launches, 51.2 us of profiled GPU time) with ONE persistent CUDA kernel compiled in prepare() with torch.utils.cpp_extension.load_inline (one TU, -O3 -arch=sm_90a; pattern from muse_attention_decode _build_splitkv). At bs=1 the intermediate is at most 8 routes x I=4096 bf16 = 64 KB, so it never needs a full round trip through a second launch.
ROUTING IN-KERNEL: every CTA reads the 8 topk_ids and topk_weights of the token, maps id - local_start into [0, local_count) (bank start/count are kernel args from prepare), and compacts the local routes into R <= 8 (slot, expert, weight) entries in smem, in slot order (deterministic, identical in every CTA). Remote slots are skipped; R == 0 makes the last CTA write a zero row.
WEIGHT LAYOUT (prepare, lossless): the same NVFP4 repack as cuda-nvfp4-swapab-mma-persistent-small-m (mma.sync m16n8k16 A-fragment nibble order, one 16 B cp.async per lane per 16-row x 64-k sub-tile; e4m3 block scales permuted to 8 B per lane group; w13 gate/up rows interleaved per tile; gs folded as gs * 2^7; round-trip assert). If that hypothesis has been implemented, reuse its device functions verbatim; otherwise implement them here as specified there (dequant: v = (q & 0x80008000) | ((q & 0x70007000) >> 6), q <<= 4, times a bf16x2 scale from cvt.rn.f16x2.e4m3x2 x 2^119, exact).
PHASES, ONE GRID: grid = 132 CTAs (1 per SM, 256 threads = 8 warps, launch_bounds(256, 1), ~100 KB dynamic smem so residency is guaranteed). Global item order: all phase-1 items first, then all phase-2 items; CTA c takes items c, c+132, ... as one flat (item, k-stage) cp.async sequence (STAGES = 4, BK = 256, 3 stages = ~27-54 KB of weight in flight per SM), so the ring never drains between items or phases.
Phase 1 (w13 + silu): item = (route r, w13 tile j) with 64 weight rows (32 gate + 32 up, matching columns), full K = 4096 (16 stages), 128 tiles per route -> 128R items (256 at R = 2: ~2 per CTA, equal bytes, 144 KB each). The token x (8 KB bf16) is loaded once into smem per CTA. mma.sync with the weight as A and the token in column 0 of the n8 B fragment (the other 7 columns zero; MMA throughput is not the limit, it gives fp32 accumulation for free). Epilogue: acc * gs_eff13 -> silu(gate) * up in fp32 -> bf16 h[r, 32j .. 32j+32) into a device scratch (prepare-allocated [8, I] bf16), then st.release.gpu flag[r][j / 8] += 1 (one flag per 256-column chunk of h, 8 tiles each).
Phase 2 (w2): item = (route r, w2 tile n of 128 H rows, k-split s of 4 over I, 1024 columns each) -> 128R items, 72 KB each. The CTA issues its weight cp.async stages for this item BEFORE waiting (weights do not depend on h, so the wait overlaps the stream), then spins with ld.acquire.gpu on the 4 flags covering its 1024 columns until each equals epoch_target, then loads that h slice (2 KB) into smem and runs the MMA loop. Partial results: fp32 [4, 8, H] scratch; per (r, n) ticket: store partial, __threadfence, atomicAdd; the last of the 4 splits sums them in split order (ld.global.cg), multiplies by gs_eff2, writes bf16-rounded y[r, 128n .. 128n+128) to scratch (bf16 rounding matches master, which keeps y in bf16), resets the ticket, and bumps a per-token done counter.
Phase 3 (post-norm combine): the CTA whose done counter increment completes R x 32 tiles computes out[h] = sum over r of g_r * rmsnorm(y_r * (1 + post_norm_w)) in fp32 (eps 1e-8, the _post_norm_combine_kernel formula), writes bf16 out [1, H], and resets the done counter.
EPOCHS, NO MEMSETS: a device int epoch (prepare-zeroed) is read by every CTA at entry; flags count up and phase 2 waits for flag == 8 * (epoch + 1) with flags never reset (int32 wrap is ~2^28 calls away); the CTA that finishes phase 3 (or the last CTA by an exit ticket when R == 0) increments epoch after a __threadfence. Deadlock freedom: all 132 CTAs are co-resident (1 per SM, grid == SM count, asserted in prepare from the device properties and the cudaOccupancyMaxActiveBlocksPerMultiprocessor result >= 1) and every CTA finishes all its phase-1 items before its first phase-2 wait, so every awaited tile is owned by a CTA that is not waiting. kernel_fn launches it on the current stream with no host branch on data (CUDA-graph safe); out is torch.empty.
CORRECTNESS SELF-CHECK (prepare, untimed, fixed torch.Generator, real bank weights): bs=1 cases R = 0, 1, 2, 3, 8 local routes (incl. all-remote and all-local), local_start 0, run the fused kernel 50 times in a row (to exercise epoch and ticket reuse) and compare out against master path on the same inputs; rel L2 <= 1e-2 per case, print max rel-L2, regs, localSizeBytes, smem and PATH CHOSEN loudly ([cuda-moe-bs1] SELF-CHECK PASSED/FAILED). Also run it once under torch.cuda.graph capture + replay in prepare to prove capture works. Assert tickets and done counter are 0 afterwards.
FALLBACK: a load_inline exception, a failed check, a failed residency assert or a graph-capture error sets the flag off and bs=1 runs master byte for byte.

## Expected

- `bs=1 *` faster: decode_1 today: 48.4 us scored, 51.2 us profiled over 5 launches (Marlin pair 44.6, post-norm 3.4, silu 1.6, align 1.6) for ~2 local experts (~55 MB, floor 11.4 us). One persistent kernel at ~3.5 TB/s with phase-2 weight prefetch overlapping phase 1 and no inter-launch gaps targets <= 20 us for the whole MoE (M2 <= 46 us, M3 <= 18 us): decode_1 alone ~2.4x, headline ~+16%
- `bs=8 *` flat: control: num_tokens > 1 runs master byte for byte
- `bs=16 *` flat: control: num_tokens > 1 runs master byte for byte
- `bs=32 *` flat: control: num_tokens > 1 runs master byte for byte
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

Independent of cuda-nvfp4-swapab-mma-persistent-small-m but overlapping at bs=1: if both land, this one owns bs=1 (it also removes the align, silu and post-norm launches and the phase boundary). It shares that file's layout and dequant, so implementing after it is cheaper; if run first, it is the smaller proof of the CUDA NVFP4 mainloop (one shape, one kernel). Also overlaps the running w2-l2-prefetch-decode1 (decode_1 only): judge this against master, not stacked. Risks: spin-wait correctness (release/acquire flag protocol, co-residency: if the judge or an L2-flush kernel overlaps the replay on the same stream it cannot, but assert grid == SMs and 1 block/SM), epoch bookkeeping across graph replays (the 50-run self-check and the capture test cover it), R up to 8 at bs=1 test routings (items scale to 1024, still balanced), and the fixed 1/8 MMA column use (harmless: the kernel is weight-stream bound). Milestones on the replaced kernels (the fused kernel row vs align+Marlin pair+act_and_mul+post-norm = 51.2 us profiled): M1 the 5 tests pass with PATH CHOSEN = fused in the log; M2 fused kernel <= 46 us; M3 <= 18 us (~3 TB/s). Stage 2 if M2 holds but M3 fails: deeper ring (6 stages), 2 m16 tiles per warp, or split phase-1 K so items shrink when R is 1.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T170224Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound rewrite-stage-1 design for decode_1 (weight-stream bound, 5 launches -> 1, all other shapes byte-identical controls), but the epoch/flag protocol as written hangs whenever R grows between calls (including inside the prepare self-check, which would be a charged timeout, not a fallback); fix that and deepen phase-1 bytes in flight at build time, then run as a check after the reset.

### Justification
Mechanism is plausible: at bs=1 with ~2 local routes the MoE moves ~55 MB (w13 18 MB + w2 9 MB per expert incl. e4m3 scales), floor 11.4 us, and master spends 51.2 us profiled (Marlin pair 44.6 us = ~1.2 TB/s, plus align/silu/post-norm and inter-launch gaps). A persistent weight-streaming kernel at 3-3.5 TB/s lands at ~16-20 us, and the headline arithmetic checks out (decode_1 load ratio 77.8/48.4=1.61 -> ~3.5-3.9 is a ~2.2-2.4x factor on one of six equally weighted loads, i.e. +14-16% geomean; even a 27 us result gives ~+10%). Expected directions match TASK.md: only bs=1 changes, bs=8/16/32/128/256 run master byte for byte and are clean in-run controls. The dequant bit math is correct and exact: (q & 0x80008000) | ((q & 0x70007000) >> 6) places e2m1 sign at bit 15, the 2 exponent bits at bf16 exponent bits 8:7 and the mantissa bit at bit 6, giving e2m1 * 2^-126 (0.5 is a bf16 subnormal, which mul/fma.rn.bf16x2 keep); cvt.rn.f16x2.e4m3x2 is exact, sf*2^119 < 2^128 has <=4 significant bits so the bf16 cast is exact, the product e2m1*sf*2^-7 has <=6 significant bits (range 2^-17..21) and is exact in bf16, and gs*2^7 is an exact power-of-two fold. Bf16 h and bf16-rounded y match master's numerical path; fp32 split-K partials match Marlin's fp32 reduce. Co-residency (132 CTAs, launch_bounds(256,1), ~100 KB smem, grid == SM count asserted) plus 'all my phase-1 items before my first phase-2 wait' is a valid deadlock argument on a dedicated stream. WEAKEST LINK (performance): phase 1 carries 2/3 of the bytes but its stage is only 64 rows x BK 256 = 8 KB codes + 1 KB scales, so 3 stages ahead = ~27 KB per SM (~3.6 MB grid-wide), below the Little-law need of ~4.8 MB at 4.8 TB/s x ~1 us cold-DRAM latency; LESSONS shows 64 KB in flight per CTA reached only 3.3 TB/s and 128 KB 3.73 TB/s, so phase 1 likely runs ~2-2.5 TB/s and the <=18 us M3 target slips to ~22-26 us. Fix at build: make phase-1 stages BK=512 (18 KB, same slot size as phase 2) or 6-8 phase-1 stages in the same ring bytes. Secondary: phase 1 has 64 rows per item but 8 warps of m16 cover 128 rows, so the spec leaves 4 warps idle or needs an unspecified intra-CTA K split + smem reduce; per-SM dequant ALU (~2+ ops/weight over ~860K weights/SM ~ 6-9 us of issue) must overlap the stream, which needs all 8 warps busy. Also, phase-2 items for route 1 depend on the second phase-1 item of CTAs 0-123, so phase 2 is effectively behind a grid barrier with only ~54 KB/SM of prefetch overlap; budget the tail.

### Confounders on the current kernel
- post-expert-norm-triton-small-m: its _post_norm_combine_kernel (3.4 us at decode_1) is one of the 5 replaced launches and its fp32 formula (eps 1e-8, g * rmsnorm(y*(1+w))) is what phase 3 re-implements; no confound because bs=1 is replaced wholesale and other shapes keep it, but the self-check must compare against this master path, not SGLang's fused_post_expert_norm_weighted_sum.
- triton-map-align-on-block-m-restack (and the triton-map-align-small-m-port it carries): _align_small (1.6 us at decode_1) is replaced at bs=1 by in-kernel routing; the gain attributed to the fused kernel includes removing this launch, which is intended and readable from the profile (one kernel row vs five). Its block_size_m >=128 part is shape-disjoint.
- marlin-block-m-two-smaller-on-postnorm: bs>=128 only, disjoint from bs=1; no interaction.
- sglang-marlin-path-parity: the base pipeline; the fused path's correctness target. No confound.

### Interference and A/B needs
No companion warranted. The change is gated on num_tokens == 1 and replaces every bs=1 kernel, so there is no landed bs=1 sub-path left whose revert or retune could change the result; bs=8/16/32/128/256 are byte-identical in-run controls that expose run drift. As a rewrite it is judged on the fused kernel's own profiled time against the 51.2 us it replaces and the 11.4 us floor, which one run reads directly from top_kernels plus the [cuda-moe-bs1] PATH CHOSEN line in the log (read the log tail first: a FAILED/fallback line means the run measured master). Judge against master, not stacked with the running w2-l2-prefetch-decode1 (same shape). The next stage, not a control, is the follow-up for any shortfall.

### Open questions
- BUILD BLOCKER: the epoch protocol hangs when R varies. Flags flag[r][c] are bumped only for routes r < R of the current call, but phase 2 waits for flag == 8*(epoch+1) with epoch counting all calls; after any call with R=1, flag[1][*] lags the epoch, so the next call with R>=2 spins forever. The self-check's case sequence R=0,1,2,3,8 triggers it in prepare. Fix: drop epochs (launches on one stream are already ordered) and have the last CTA by an all-CTA exit ticket reset flags[0..R), (r,n) tickets, the done counter and the exit ticket to 0 at the end of every launch; or bump all 8 routes' flags every call. Also run the self-check cases interleaved (e.g. R=2,1,8,0,3 repeated) and graph-replay with varying R, not 50 calls of a single R.
- Release/acquire detail: h is written by many threads, so do __syncthreads(), then thread 0 issues fence/red.release.gpu.add on the flag (an st.release is not an increment). The waiter: thread 0 ld.acquire.gpu spin, then __syncthreads, then load the h slice with cp.async.cg / ld.global.cg (L2), never .ca/__ldg, since L1 is not coherent within the launch. Same for split-K partials and y (winner and phase-3 CTA read with .cg after an acquire on the ticket/done counter).
- Phase-1 tile shape vs 8 warps: 64 weight rows is 4 m16 tiles; specify an intra-CTA K split across the two warp groups with an smem reduce, or use 128-row phase-1 tiles (64 gate + 64 up, 64 tiles per route: 128 items at R=2, still ~1 per CTA) which also doubles bytes per stage.
- Prefer a cooperative launch (cudaLaunchCooperativeKernel, capturable in graphs on CUDA 12) over the occupancy assert alone, so co-residency is guaranteed by the driver; keep the device-properties assert (132 SMs on H200) as well.
- The frontmatter says replaces 'Marlin w13+w2 at bs<=32' but the change is bs=1 only; judge milestones on decode_1 (and the bs=1 test) only: M2 fused row <= 46 us, M3 <= 18 us. Expect M2 easily and M3 only with the deeper phase-1 ring.
- Add one fp32-dequant reference for a single expert (local e2m1/e4m3 decode, not the reference module) as the sibling cuda-nvfp4-swapab-mma-persistent-small-m does, so a nibble-order error cannot hide behind a master-vs-fused comparison that happens to be close.
- Implementation order with cuda-nvfp4-swapab-mma-persistent-small-m: whichever runs first, keep the repack/dequant device functions shared verbatim so a mainloop bug is found once.

