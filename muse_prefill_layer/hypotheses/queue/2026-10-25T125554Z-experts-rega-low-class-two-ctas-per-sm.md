---
problem: muse_prefill_layer
slug: experts-rega-low-class-two-ctas-per-sm
created: 2026-10-25T125554Z
base: master
area: "w13 + w2 expert GEMMs at 200 < T <= 1024 (turn_1, rega low class): two resident CTAs per SM with a 64- or 32-route slab, to hide consumer dequant latency"
change: >
  Rewrite stage 3 of the rega line, built on master a1a5497d
  (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX
  = 1024, i.e. turn_1 T=753 and the T=300/753 tests). 'high' (turn_4) and 'ft' (first_turn) keep
  master's instantiations byte for byte. WHY: turn_1 is the weakest load (1.348x, against 1.428x
  for turn_4 and 1.444x for first_turn), and rega is 48% of it. In job 07a069bc69ba42a8 the low
  class ran rega_kernel<true,false> (w13+silu) at 282.4/257.8 us and rega_kernel<false,false> (w2)
  at 146.0/125.2 us. At T=753 every one of the 32 local experts is touched, so the kernels stream
  ~604 MB (w13) and ~302 MB (w2) of NVFP4 codes+scales, about 2.1 TB/s against 4.8 (floor ~126 +
  63 us). muse_moe_w4a16 has already ruled out bytes in flight
  (cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: STAGES 7/10 flat, 1.83 -> 1.87 TB/s) and
  per-warpgroup dequant/MMA overlap (s3 variant B, double-buffered A, no faster). LESSONS line 183
  says 'look at the consumer/dequant'. What remains is latency hiding. One 384-thread CTA per SM
  gives each scheduler only 2 consumer warps, and their dequant -> wgmma -> wait<0> chain is
  serial. The CHANGE doubles resident consumer warps, with two CTAs per SM. (1) Template the
  kernel on the slab width NMAX and pipeline depth STG, as the moe branch did (git show
  hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring:muse_moe_w4a16/submission.py, its
  Geo<NMAX,STG>, chunk cut CH = NMAX/16 blocks, consumer buckets above NMAX not instantiated). Add
  an occupancy parameter OCC (1 or 2) that sets __launch_bounds__(THREADS, OCC). Master's
  instantiations are <G, BIG, NMAX=128, STG=4, OCC=1> and must stay identical. (2) Add two
  low-class candidates for each G: V64 = <G, false, 64, 3, 2> and V32 = <G, false, 32, 4, 2>.
  Per-CTA smem: V64 dynamic 1024 + 3 x (9216 + 2 x 64 x 128) + 64 x PITCH (w2 95232 B), V32
  dynamic 1024 + 4 x 17408 + 32 x PITCH (w2 79360 B), each plus ~14.3 KB static. Two CTAs plus the
  1 KB per-CTA reserve fit in 228 KB; static_assert it. The epilogue staging is NMAX rows, not
  128. (3) Registers at OCC=2. Compile so that numRegs is exactly 80 (__launch_bounds__(384, 2)),
  producer setmaxnreg.dec 24 and consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 = 29696 <= 384
  x 80). Before any launch, prepare() reads cudaFuncGetAttributes and
  cudaOccupancyMaxActiveBlocksPerMultiprocessor for each candidate. It requires numRegs == 80,
  localSizeBytes == 0 and 2 CTAs/SM; otherwise that candidate is OFF. setmaxnreg.inc hangs if the
  pool is short (see the comment near master line 3610), so never launch an unchecked candidate.
  Print regs, local bytes, smem and CTAs/SM per instantiation. (4) Launch with grid = 2 x SM count
  for the OCC=2 candidates. Store this in the state as an int next to rega_grid, never as a module
  global. Keep the kernel persistent, with items strided by gridDim.x as now. (5) Dispatch:
  _experts_rega passes a variant int read from state['rega_low_variant'] (0 = master, 1 = V64, 2 =
  V32), only when T <= REGA_LOW_MAX. rega_gemm selects the instantiation by (which, variant,
  block-table class) on the host from shapes and that int. The judge prepares once and copies the
  state by cloning tensors, so nothing may depend on data_ptr equality; maps are still encoded per
  call. (6) Self-check each candidate with master's existing low-class machinery: eager check
  against the fp32 reference at T=300 and 753 (random, skewed, one-expert), error word 0, launch
  markers, finite output, then the A/B/C copy/replay check with refilled inputs. Any miss turns
  only that candidate OFF. (7) Gate at T=753 against master's low class (rega NMAX128), using
  cold-L2 graph medians, >= 9 interleaved alternations and the empty-graph baseline subtracted.
  Take the WORSE ratio of two routings: uniform random, and a popularity-skewed routing with about
  1.8x the random local-route count (the judge's soft first_turn routing had ~29k local routes
  against ~16k random at T=16384). ON only at <= 0.95. Of passing candidates, pick the lower
  median. Print per candidate the w13 and w2 kernel-graph us and achieved TB/s (touched-expert
  bytes / time). Print '[rega] PATH CHOSEN low: master|V64|V32' among the final prepare() lines.
  kernel_fn rules: no host sync, no prints, no try/except, no per-call mutable state, dispatch on
  T and prepare-set ints only. NVFP4 weights stay NVFP4 (register dequant only), BF16 activations,
  fp32 accumulation. Build exactly as master (TORCH_CUDA_ARCH_LIST=9.0a, -gencode
  arch=compute_90a,code=sm_90a, wgmma/setmaxnreg under __CUDA_ARCH_FEAT_SM90_ALL). Use a new
  extension name (muse_prefill_rega_v3), keeping master's v2 -> v1 build fallback chain.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low-class rega w13+w2 428/383 us at ~2.1 TB/s (floor ~189 us on 906 MB) -> ~290-330 us if two resident CTAs lift it to ~3 TB/s: sliding turn_1 777 -> ~660-700 us (-10 to -15%), full 1136 -> ~1030-1060"
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: high class keeps master NMAX128/STG4/OCC1 byte-identical
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: ft class keeps master instantiation byte-identical
risk: high
depends_on: experts-turn-rega-s4-first-turn-n128
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024
review: proceed
status: queued
review_at: 2026-10-25T130000Z
review_companions: null
---

# experts-rega-low-class-two-ctas-per-sm

## Change

Rewrite stage 3 of the rega line, built on master a1a5497d (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX = 1024, i.e. turn_1 T=753 and the T=300/753 tests). 'high' (turn_4) and 'ft' (first_turn) keep master's instantiations byte for byte. WHY: turn_1 is the weakest load (1.348x, against 1.428x for turn_4 and 1.444x for first_turn), and rega is 48% of it. In job 07a069bc69ba42a8 the low class ran rega_kernel<true,false> (w13+silu) at 282.4/257.8 us and rega_kernel<false,false> (w2) at 146.0/125.2 us. At T=753 every one of the 32 local experts is touched, so the kernels stream ~604 MB (w13) and ~302 MB (w2) of NVFP4 codes+scales, about 2.1 TB/s against 4.8 (floor ~126 + 63 us). muse_moe_w4a16 has already ruled out bytes in flight (cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: STAGES 7/10 flat, 1.83 -> 1.87 TB/s) and per-warpgroup dequant/MMA overlap (s3 variant B, double-buffered A, no faster). LESSONS line 183 says 'look at the consumer/dequant'. What remains is latency hiding. One 384-thread CTA per SM gives each scheduler only 2 consumer warps, and their dequant -> wgmma -> wait<0> chain is serial. The CHANGE doubles resident consumer warps, with two CTAs per SM. (1) Template the kernel on the slab width NMAX and pipeline depth STG, as the moe branch did (git show hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-b128-n32-deep-ring:muse_moe_w4a16/submission.py, its Geo<NMAX,STG>, chunk cut CH = NMAX/16 blocks, consumer buckets above NMAX not instantiated). Add an occupancy parameter OCC (1 or 2) that sets __launch_bounds__(THREADS, OCC). Master's instantiations are <G, BIG, NMAX=128, STG=4, OCC=1> and must stay identical. (2) Add two low-class candidates for each G: V64 = <G, false, 64, 3, 2> and V32 = <G, false, 32, 4, 2>. Per-CTA smem: V64 dynamic 1024 + 3 x (9216 + 2 x 64 x 128) + 64 x PITCH (w2 95232 B), V32 dynamic 1024 + 4 x 17408 + 32 x PITCH (w2 79360 B), each plus ~14.3 KB static. Two CTAs plus the 1 KB per-CTA reserve fit in 228 KB; static_assert it. The epilogue staging is NMAX rows, not 128. (3) Registers at OCC=2. Compile so that numRegs is exactly 80 (__launch_bounds__(384, 2)), producer setmaxnreg.dec 24 and consumer setmaxnreg.inc 104 (128 x 24 + 256 x 104 = 29696 <= 384 x 80). Before any launch, prepare() reads cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor for each candidate. It requires numRegs == 80, localSizeBytes == 0 and 2 CTAs/SM; otherwise that candidate is OFF. setmaxnreg.inc hangs if the pool is short (see the comment near master line 3610), so never launch an unchecked candidate. Print regs, local bytes, smem and CTAs/SM per instantiation. (4) Launch with grid = 2 x SM count for the OCC=2 candidates. Store this in the state as an int next to rega_grid, never as a module global. Keep the kernel persistent, with items strided by gridDim.x as now. (5) Dispatch: _experts_rega passes a variant int read from state['rega_low_variant'] (0 = master, 1 = V64, 2 = V32), only when T <= REGA_LOW_MAX. rega_gemm selects the instantiation by (which, variant, block-table class) on the host from shapes and that int. The judge prepares once and copies the state by cloning tensors, so nothing may depend on data_ptr equality; maps are still encoded per call. (6) Self-check each candidate with master's existing low-class machinery: eager check against the fp32 reference at T=300 and 753 (random, skewed, one-expert), error word 0, launch markers, finite output, then the A/B/C copy/replay check with refilled inputs. Any miss turns only that candidate OFF. (7) Gate at T=753 against master's low class (rega NMAX128), using cold-L2 graph medians, >= 9 interleaved alternations and the empty-graph baseline subtracted. Take the WORSE ratio of two routings: uniform random, and a popularity-skewed routing with about 1.8x the random local-route count (the judge's soft first_turn routing had ~29k local routes against ~16k random at T=16384). ON only at <= 0.95. Of passing candidates, pick the lower median. Print per candidate the w13 and w2 kernel-graph us and achieved TB/s (touched-expert bytes / time). Print '[rega] PATH CHOSEN low: master|V64|V32' among the final prepare() lines. kernel_fn rules: no host sync, no prints, no try/except, no per-call mutable state, dispatch on T and prepare-set ints only. NVFP4 weights stay NVFP4 (register dequant only), BF16 activations, fp32 accumulation. Build exactly as master (TORCH_CUDA_ARCH_LIST=9.0a, -gencode arch=compute_90a,code=sm_90a, wgmma/setmaxnreg under __CUDA_ARCH_FEAT_SM90_ALL). Use a new extension name (muse_prefill_rega_v3), keeping master's v2 -> v1 build fallback chain.

## Expected

- `*load=turn_1*` faster: low-class rega w13+w2 428/383 us at ~2.1 TB/s (floor ~189 us on 906 MB) -> ~290-330 us if two resident CTAs lift it to ~3 TB/s: sliding turn_1 777 -> ~660-700 us (-10 to -15%), full 1136 -> ~1030-1060
- `*load=turn_4*` unchanged: high class keeps master NMAX128/STG4/OCC1 byte-identical
- `*load=first_turn*` unchanged: ft class keeps master instantiation byte-identical

## Risk

Risk: high, but gated. A null, where two CTAs/SM give no lift, is itself a finding: it would mean the limiter is in the per-item path (decode_item and table scans, epilogue, the single producer thread's TMA issue), not latency. It is printed as per-candidate TB/s. Ways it can fail: the register cap of 80 spills (then that candidate is OFF, printed); NMAX=32 splits experts with more than 32 routes into extra chunks that re-read the weight tile (from L2 when adjacent; V64 avoids most of this); build time for 4 extra instantiations (master prepare is about 115-130 s; budget 200 s for rega, and ft has its own 150 s start deadline, so print phase timings and skip the new candidates first if the budget runs short, never ft). Attribution: the low-class rows get new template names (rega_kernel<G,false,64,3,2> or <G,false,32,4,2>) on turn_1 only. turn_4 and first_turn rows must stay within ~2-3% drift. Read the turn_1 kernel rows, not the headline; LESSONS warns that turn_1 drift is large (up to 7-10% on untouched kernels). Independent of align-two-phase-all-t-rega (different area and kernels). Both build on master.

## Milestones

Rewrite stage 3 of `rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T130000Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a rewrite stage with no companion runs. Two CTAs per SM at 80 registers is physically reachable for the low class, because at T=753 the consumer never needs master's 232-register NB=128 budget. V32 should fit 104 consumer registers with room to spare and V64 is tight but plausible. Fix the producer/consumer register split (use 32/104, not 24/104), and expect V32 to fail the skewed-routing gate because of chunk splitting.

### Justification
MECHANISM. Profile job 07a069bc: the low class runs rega_kernel<true,false> at 282.4/257.8 us and <false,false> at 146.0/125.2 us, about 2.1 TB/s on ~906 MB against a ~189 us byte floor, so there is about 2x headroom. Bytes in flight are ruled out twice: the moe deep ring went from STAGES 4 to 7/10 with flat 1.83->1.87 TB/s, and today's LESSONS line agrees. That leaves issue and latency in the consumer. Back-of-envelope per 9216 B stage per SM: the 8 consumer warps issue about 180 dequant instructions each, about 1440 warp-instructions, or about 360 clk over 4 schedulers. The wgmma m64n32k16 x8 per warpgroup costs about 128 clk of tensor time, about 256 clk for both WGs. The byte budget at 4.8 TB/s is about 440 clk per stage, and the measured time is about 1000 clk. The kernel therefore sits at roughly 2-3x its issue floor, with a serial dequant -> commit -> wait<0> -> arrive chain per WG and only 2 consumer warps per scheduler. That is the latency-bound picture TLP is meant to fix. The ceiling after doubling warps is the dequant ALU issue (~360 clk/stage, ~4.5 TB/s at best), so ~3 TB/s is a reasonable target. Variant B (double-buffered A) being no faster is the strongest counter-evidence, because it is the ILP version of the same latency hiding. But ptxas commonly serializes register-A wgmma when fragment registers are rewritten near in-flight groups, so B's null does not transfer to TLP. REGISTERS (the asked-for hard look). master's 232 is sized for the NB=128 bucket: acc 64 + a0[8][4] 32 + ~30-40 addressing/item/loop. At T=753 a soft routing gives about 24-45 local routes per expert, so master already runs mostly the NB=32/64 buckets and never needs 232. Consumer live set at peak, inside the 8-mma issue: V32 (max bucket 32) has acc 16 + a0 32 + about 25-35 (Item m 6, smem/ep/full0/empty0/n/kb/s/slab/KS/cw/lane/ctid, gs/sorted_ids/out pointers, err/dead) = about 75-85 <= 104, so it should be spill-free. V64 (max bucket 64) has acc 32 + a0 32 + the same = about 90-100 at peak, plus transient dq_stage temporaries (uint4 q, uint2 sc, hs[4], fs). That is tight against 104 and could spill a few words. ptxas allocates one budget for the whole switch region, so the NB=64 case decides it. The localSizeBytes==0 gate catches this and turns V64 OFF without hanging; my estimate is about 60-70% that V64 compiles clean. The producer side is the real defect in the spec. Master gives it 40 registers. 24 is spill-prone for decode_item (two integer divisions, five table loads), the 64-bit wsrc arithmetic, the expect_tx/bulk_g2s/tma_2d sequence and the mbar_wait timeout state. Use dec 32 / inc 104 instead: 32x128 + 104x256 = 30720 = exactly 384x80, which is legal because dec frees 6144 and inc takes 6144. The fallback is 40/96 (29696) only if V64 at 104 still spills, which is unlikely since 96 is worse for V64. SMEM checks out: V64 w2 95232 + ~14.3 KB static = ~109.6 KB, x2 + 2 KB reserve = ~221 KB <= 228 KB. V32 has ample room. Remember that the slab TMA boxes are already per bucket (16<<bucket rows), so NMAX only bounds which buckets exist and the stage/epilogue reservation. Shrink SLAB_HALF to NMAX rows per stage, with each half still 1024-aligned. EXPECTED: turn_1 faster, turn_4/first_turn unchanged is consistent with a low-class-only gate. Saving ~100-140 us of 428/383 us is ~13-18% of sliding turn_1, a ~3% headline move at most, which is inside LESSONS' 7-10% turn_1 drift. Judge it on the rega_kernel<G,false,64|32,...> rows, not the headline. WEAKEST LINK: chunk splitting under the narrower slab. Each extra chunk re-dequantizes the full weight tile, and dequant ALU is exactly the resource being saturated. At the judge's soft routing (~1.8x random local routes per FINDINGS, i.e. ~40-45 per expert at T=753), V32 splits most experts. Dequant work grows ~1.5-2x, so V32 will likely lose the skewed half of the worse-of-two gate. V64 splits only the popular experts. The whole result therefore rests on V64 compiling without spills. Second-weakest: ptxas may report numRegs < 80 if it does not need 80. The pool would then be short of 30720 and setmaxnreg.inc would hang. The spec's numRegs==80 requirement makes that a clean OFF rather than a hang, but it could switch both candidates OFF for a benign reason.

### Confounders on the current kernel
- experts-turn-rega-s4-first-turn-n128 (parent, landed): the template refactor (NMAX/STG/OCC parameters, setmaxnreg immediates via template constants, SLAB/EPI sized by NMAX) touches the kernel source shared by the 'high' and 'ft' instantiations. If <G,BIG,128,4,1> does not produce identical SASS, turn_4/first_turn drift becomes attributable to the refactor, not noise. The build is a new extension (v3), so print numRegs/localSizeBytes/smem for the master instantiations and assert they equal v2's, which is the cheapest stand-in for byte identity.
- experts-turn-rega-s4-port-t4096 (landed): it is the low-class incumbent the gate compares against. No interaction beyond being the baseline. Its single-CTA tail at 132 CTAs is replaced by a 264-CTA stride, so any change in tail/wave quantization is part of the change, not a confounder.
- stack-tonight-five-prefill (landed): its _align_parallel produces the block table the kernel cuts into chunks (CH = NMAX/16). Narrower chunks depend on the align keeping each expert's blocks contiguous (error word 4). This holds today and needs no A/B. The queued align-two-phase-all-t-rega is independent but must keep that invariant if both land.

### Interference and A/B needs
No companion runs. This is a rewrite stage with an in-prepare interleaved gate against master's own low class, per-candidate printed TB/s, and distinct template names in the profile, so the single check already attributes the result. A gain shows as rega_kernel<G,false,64|32,...> rows on turn_1 below 282/146 us. A null is readable from the printed per-candidate TB/s (latency hiding did not help, so the limiter is per-item/producer/ALU issue). The one ambiguity a companion could resolve is whether a gain comes from 2 CTAs/SM or from the narrower slab and shallower ring alone. That costs nothing to settle inside prepare: also build a print-only, never-selectable diagnostic V64 at OCC=1 (<G,false,64,3,1>, master's 40/232 split) and print its gate median next to V64-OCC2. Because bytes in flight are already ruled out, the OCC1-vs-OCC2 gap isolates TLP. With 2 prefill checks left today, spending one on a companion would be waste.

### Open questions
- Does ptxas emit numRegs exactly 80 for __launch_bounds__(384,2) with setmaxnreg present, or could it choose less? If less, the 32/104 split no longer fits and the candidate goes OFF. Print it, and consider an alternate immediate set keyed off the reported value (e.g. 24/96 for 72) as a second compiled variant only if this happens.
- Does V64's consumer region fit 104 registers at the NB=64 bucket without local memory? If V64 spills, a V64 with the NB=64 case dropped is just V32. A better fallback is dequantizing one 64-k record at a time (a0[4][4], 16 registers, two commit/wait pairs per stage), which frees 16 registers per consumer at the cost of one more wait per stage.
- Actual local-route distribution at the judged turn_1 shape: mean and max routes per local expert at T=753 soft routing. This decides how often V32/V64 split chunks. Print the max chunks per expert in the gate line for both routings.
- Does 264-CTA striding change the tail? w13/w2 item counts at T=753 divided by 132 vs 264 CTAs. Print items per CTA.
- Keep the per-candidate w13 and w2 kernel-graph us separate: the w2 kernel (K = intermediate, fewer items) may respond differently from w13 and could justify a mixed pick (e.g. V64 for w13, master for w2).

