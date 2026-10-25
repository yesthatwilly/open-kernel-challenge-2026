---
problem: muse_moe_w4a16
slug: cuda-fused-split-i-moe-midt-cluster-reduce
created: 2026-10-25T153417Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 1 of a fused split-I MoE for mid T: 9 <= num_tokens <= 32 only (decode_16 bs=16, decode_32
  bs=32). num_tokens <= 8 and >= 33 (incl. the bs=33 test) run master byte for byte. Built on
  current master (40e2e860), new code only, own load_inline module. If
  cuda-fused-split-i-moe-tiny-t has already landed on master when this is claimed, build on it and
  reuse its fsi_kernel, combine kernel and gate machinery, adding only what is new here; otherwise
  implement the shared parts as that file specifies (the two are shape-disjoint and merge
  cleanly). WHY: decode_32 is the weakest load (1.03x: 201.9-204.6 us vs SGLang 208-211) and
  decode_16 is 1.28x; both are ~95% Marlin (decode_32 pair 175-199 us, decode_16 115-166 us)
  against floors of 70.8 and 52.5 us. Marlin there runs block_size_m 8 (T=16) / 16 (T=32) and
  processes each m-block of an expert as its own problem, so a hot expert with more routes than
  one block streams its weights more than once, and every phase boundary drains the grid.
  MECHANISM (as the tiny-T sibling): split I into chunks of KC columns; a unit (expert e, chunk j)
  computes h[:, j] over the full K = 4096 for ALL routes of e (at T <= 32 an expert has at most T
  <= 32 routes, since a token's 8 ids are distinct), applies silu * up, keeps h in smem as bf16,
  then streams W2[:, j] and adds its partial y. Every weight byte is read exactly once per call
  whatever the routing, with no cross-unit dependency. WHAT IS NEW vs the tiny-T file: (1) tokens
  on N use master's N32 instantiation (wgmma m64n32k16, proven by the bucket canary), with
  master's _align_small called at block_size_m 32 so each active expert is exactly one 32-row
  block; (2) KC = 128 or 256 (2 or 4 adjacent q13 tiles per unit; 16-32 chunks x ~9-12 active
  experts = 144-384 units) chosen by the prepare selector; (3) the fp32 partial-y reduction goes
  through a thread-block cluster: launch fsi with cudaLaunchKernelEx and a cluster dimension CL =
  4 (grid rounded down to a multiple of 4, <= SM count, co-residency checked with
  cudaOccupancyMaxActiveClusters before the first launch, else the path stays off). The four CTAs
  of a cluster take the four I-chunks j0..j0+3 of the SAME expert (unit numbering: u = ((e * nJ/4)
  + q) * 4 + rank, so cluster c walks u-groups c, c + nclusters, ...), each computes its w2
  partial for one 128-row H tile at a time into its own smem (128 x 32 fp32 = 16 KB),
  cluster.sync(), then rank r sums the four partials of H rows [32r, 32r+32) of that tile through
  distributed shared memory (ld.shared::cluster via mapa) in a fixed rank order and issues
  red.global.add.v4.f32 for the real routes only; cluster.sync() before the smem buffer is reused.
  That cuts global fp32 reds 4x (at T=32 with ~60 local routes: 60 x 4096 x 16 chunks / 4 / 4 =
  ~250 K red.v4 per call at KC=256) and keeps them off the critical path of the weight stream. The
  ring keeps streaming the next stage's weights across the cluster.sync because the producer
  warpgroup does not take part in it (only the two consumer warpgroups and a cluster-scope
  mbarrier). (4) COMBINE: the tiny-T file's Triton combine (fp32 y32 rows, master's formula,
  zero-after-read), here also at T=32 where master uses SGLang's
  fused_post_expert_norm_weighted_sum. y32 is a per-state fp32 [32 * 8, 4096] scratch zeroed in
  prepare. PREPARE GATE: T classes {9..16} and {17..32}. Correctness at T = 9, 16, 17, 32 and the
  bs=33 boundary untouched, routings skewed, random-local, one-expert (one expert with T routes)
  and all-remote, against an fp32-dequant reference, bound max(2x master's error, 1e-2) aggregate
  and 5x per row; the s5 three-state copy check with refilled graph replays proving y32 returns to
  zero; then per class and per KC full kernel_fn graphs, cold L2, 41 interleaved A/B replays vs
  master, enable only at >= 1.02x p50 on the skewed routing, assert the choice equals the printed
  winner, PATH CHOSEN line per class with both timings, the fsi kernel's event time and TB/s, and
  the cluster co-residency numbers. BUILD (as master's wg module): own load_inline module name,
  TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo,
  wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies
  with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation
  numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No
  module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and
  every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the
  judge prepares two states and copies them by cloning tensors). First launch in-process after a
  small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no
  child process. The first measurement is an okc check, never a scored submit. STAGES PLANNED: s1
  this file; s2 tune from the log (KC, CL = 2/4/8, ring depth, a third consumer warpgroup); s3
  merge with the tiny-T line into one T <= 32 path. Fallback at every stage: master's Marlin path
  per class.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "decode_32: master shape 201.9-204.6 us (1.03x, the weakest load), Marlin pair 175-199 us + act 2.5 + post-norm 3.6-4.5; floor 70.8 us. M1: PATH CHOSEN = FSI for 17..32. M2: fsi + combine <= 0.9x of Marlin pair + act + post-norm (<= ~170 us). M3: fsi <= 115 us (~3 TB/s on ~340 MB), shape <= 130 us."
  - shapes: bs=16 *
    direction: faster
    why: "decode_16: master shape 144.8-146.8 us, Marlin pair 115-166 us; floor 52.5 us. M2: fsi + combine <= ~110 us. M3: fsi <= 85 us."
  - shapes: bs=1 *
    direction: flat
    why: "control: T <= 8 is master (or the tiny-T sibling if it landed first)"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13 + act_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2 + silu_and_mul) and the post-norm tail at 9 <= num_tokens <= 32
review: required
status: queued
---

# cuda-fused-split-i-moe-midt-cluster-reduce

## Change

Stage 1 of a fused split-I MoE for mid T: 9 <= num_tokens <= 32 only (decode_16 bs=16, decode_32 bs=32). num_tokens <= 8 and >= 33 (incl. the bs=33 test) run master byte for byte. Built on current master (40e2e860), new code only, own load_inline module. If cuda-fused-split-i-moe-tiny-t has already landed on master when this is claimed, build on it and reuse its fsi_kernel, combine kernel and gate machinery, adding only what is new here; otherwise implement the shared parts as that file specifies (the two are shape-disjoint and merge cleanly). WHY: decode_32 is the weakest load (1.03x: 201.9-204.6 us vs SGLang 208-211) and decode_16 is 1.28x; both are ~95% Marlin (decode_32 pair 175-199 us, decode_16 115-166 us) against floors of 70.8 and 52.5 us. Marlin there runs block_size_m 8 (T=16) / 16 (T=32) and processes each m-block of an expert as its own problem, so a hot expert with more routes than one block streams its weights more than once, and every phase boundary drains the grid. MECHANISM (as the tiny-T sibling): split I into chunks of KC columns; a unit (expert e, chunk j) computes h[:, j] over the full K = 4096 for ALL routes of e (at T <= 32 an expert has at most T <= 32 routes, since a token's 8 ids are distinct), applies silu * up, keeps h in smem as bf16, then streams W2[:, j] and adds its partial y. Every weight byte is read exactly once per call whatever the routing, with no cross-unit dependency. WHAT IS NEW vs the tiny-T file: (1) tokens on N use master's N32 instantiation (wgmma m64n32k16, proven by the bucket canary), with master's _align_small called at block_size_m 32 so each active expert is exactly one 32-row block; (2) KC = 128 or 256 (2 or 4 adjacent q13 tiles per unit; 16-32 chunks x ~9-12 active experts = 144-384 units) chosen by the prepare selector; (3) the fp32 partial-y reduction goes through a thread-block cluster: launch fsi with cudaLaunchKernelEx and a cluster dimension CL = 4 (grid rounded down to a multiple of 4, <= SM count, co-residency checked with cudaOccupancyMaxActiveClusters before the first launch, else the path stays off). The four CTAs of a cluster take the four I-chunks j0..j0+3 of the SAME expert (unit numbering: u = ((e * nJ/4) + q) * 4 + rank, so cluster c walks u-groups c, c + nclusters, ...), each computes its w2 partial for one 128-row H tile at a time into its own smem (128 x 32 fp32 = 16 KB), cluster.sync(), then rank r sums the four partials of H rows [32r, 32r+32) of that tile through distributed shared memory (ld.shared::cluster via mapa) in a fixed rank order and issues red.global.add.v4.f32 for the real routes only; cluster.sync() before the smem buffer is reused. That cuts global fp32 reds 4x (at T=32 with ~60 local routes: 60 x 4096 x 16 chunks / 4 / 4 = ~250 K red.v4 per call at KC=256) and keeps them off the critical path of the weight stream. The ring keeps streaming the next stage's weights across the cluster.sync because the producer warpgroup does not take part in it (only the two consumer warpgroups and a cluster-scope mbarrier). (4) COMBINE: the tiny-T file's Triton combine (fp32 y32 rows, master's formula, zero-after-read), here also at T=32 where master uses SGLang's fused_post_expert_norm_weighted_sum. y32 is a per-state fp32 [32 * 8, 4096] scratch zeroed in prepare. PREPARE GATE: T classes {9..16} and {17..32}. Correctness at T = 9, 16, 17, 32 and the bs=33 boundary untouched, routings skewed, random-local, one-expert (one expert with T routes) and all-remote, against an fp32-dequant reference, bound max(2x master's error, 1e-2) aggregate and 5x per row; the s5 three-state copy check with refilled graph replays proving y32 returns to zero; then per class and per KC full kernel_fn graphs, cold L2, 41 interleaved A/B replays vs master, enable only at >= 1.02x p50 on the skewed routing, assert the choice equals the printed winner, PATH CHOSEN line per class with both timings, the fsi kernel's event time and TB/s, and the cluster co-residency numbers. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit. STAGES PLANNED: s1 this file; s2 tune from the log (KC, CL = 2/4/8, ring depth, a third consumer warpgroup); s3 merge with the tiny-T line into one T <= 32 path. Fallback at every stage: master's Marlin path per class.

## Expected

- `bs=32 *` faster: decode_32: master shape 201.9-204.6 us (1.03x, the weakest load), Marlin pair 175-199 us + act 2.5 + post-norm 3.6-4.5; floor 70.8 us. M1: PATH CHOSEN = FSI for 17..32. M2: fsi + combine <= 0.9x of Marlin pair + act + post-norm (<= ~170 us). M3: fsi <= 115 us (~3 TB/s on ~340 MB), shape <= 130 us.
- `bs=16 *` faster: decode_16: master shape 144.8-146.8 us, Marlin pair 115-166 us; floor 52.5 us. M2: fsi + combine <= ~110 us. M3: fsi <= 85 us.
- `bs=1 *` flat: control: T <= 8 is master (or the tiny-T sibling if it landed first)
- `bs=8 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

Distinct from the held cuda-nvfp4-wgmma-rega-midm-bucket-cap32 / -w2-only-midm (those keep two launches of master's wg_kernel with the w13 -> w2 boundary and hs round trip; the rega tiny path already showed that structure losing 1.2-1.5x at T=4-7) and from the held cuda-fused-moe-midm-single-launch-routed (flag-gated w13 -> w2 dataflow on the never-run swap-AB mma.sync mainloop). Risks: thread-block clusters have not run on this judge yet in this repo (the co-residency query and a canary launch gate it; a refusal just leaves the class on Marlin); cluster.sync per 128-row H tile could stall consumers (s2 can move the reduction to a once-per-unit pass over a 4096 x 32 fp32 buffer if smem allows, or drop to CL = 2). The dequant-bound consumer (~2 TB/s in the B256 kernel) is shared with the other rega lines; M2 does not need it fixed, M3 does.

## Milestones

Rewrite stage 1 of `Marlin w13 + act_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2 + silu_and_mul) and the post-norm tail at 9 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
