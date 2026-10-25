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
review: proceed
status: running
review_at: 2026-10-25T212522Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T212550Z
base_sha: ce2461b1ea24995a3c81d11487611f052d8ecc35
base_blob: 35b8a851e4ef5fa8fd66b12d09caac88b7ea7615
branch: hyp/muse_moe_w4a16/cuda-fused-split-i-moe-midt-cluster-reduce
worktree: .worktrees/muse_moe_w4a16/cuda-fused-split-i-moe-midt-cluster-reduce
mode: check
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-1 rewrite that targets the two weakest small-M loads, it breaks no rule, and the in-run gate plus untouched control shapes make one check readable without companions. The headline milestones are optimistic, though. It inherits the wgmma rega consumer (about 2 TB/s), and its KC=256 unit counts quantize badly on 132 SMs. A cluster.sync on every 128-row H tile may also cost more than the work between syncs. Expect decode_16 to go live and decode_32 to be a coin flip.

### Justification
MECHANISM. The structure is plausible: each expert's weights are read once whatever the routing, there is no w13->w2 boundary, and hs never round-trips. FINDINGS supports the target: decode_32 Marlin runs at 176.8 us on ~340 MB (about 1.9 TB/s) against a 70.8 us floor, and decode_16 at 156.4 us on ~252 MB (about 1.6 TB/s) against a 52.5 us floor. The unit counts also check out: about 28.3 MB of NVFP4 bytes per expert gives ~12 active experts at T=32 and ~9 at T=16, matching the stated 144-384 units.

WEAKEST LINK: the consumer's TB/s. The M3 milestones (fsi <= 115 us at T=32, ~3 TB/s; <= 85 us at T=16) assume a consumer this repo has never measured. LESSONS line 73, b128-n32-deep-ring and the tiny-m-bucket16 result (N16 rega tied Marlin at T=1, 46.0 vs 45.5 us) put the rega consumer at ~1.6-2.1 TB/s. Dequant cost is per weight element, so N32 does not help it. At 1.8-2.0 TB/s:
- decode_32: fsi is ~170-189 us against Marlin's 176.8. M2 (<= 170 us) is a coin flip and the gate may leave T=17..32 on MASTER.
- decode_16: fsi is ~126-140 us against 156.4. That beats the incumbent but misses M2's ~110 us.
DIAGNOSIS action 3 explicitly sequences this file after the consumer fix (pingpong-consumers / producer-scale-expand). Also note LESSONS line 182: the prefill pingpong and scale-expand variants never went live, so that fix may never arrive. That is a scheduling concern, not grounds for hold; a bandwidth result at N32 is useful either way.

SECOND WEAKNESS: wave quantization. Per-unit cost is set by weight bytes, so units are uniform, and the waves fall as follows:
- KC=256: 144 units (T=16) and 192 units (T=32) on 132 SMs are 1.09 and 1.45 waves, i.e. ~55% and ~73% tail efficiency.
- KC=128: 288 and 384 units are 2.18 and 2.91 waves, i.e. ~73% and ~97%.
KC=256 at T=16 is close to self-defeating. The selector must see both KC values, and the log should print units, waves and clusters actually co-resident per class.

THIRD WEAKNESS: cluster.sync granularity. Each unit walks 32 H tiles with two cluster.syncs per tile, so 64 syncs per unit. The W2 work between syncs is only 128 x KC/2 bytes: 8 KB at KC=128, which is ~0.2 us of an SM's DRAM share, and 16 KB at KC=256. Cluster-barrier latency (hundreds of ns) is about the same as that work. The four ranks are also locked together, so any skew between them stalls all four. The Risk section names this, but stage 1 has no in-run way to separate sync cost from consumer cost.

PRIOR. The 'prior' is effectively given in the Risk section: midm-bucket-cap32, w2-only-midm and fused-moe-midm-single-launch-routed, all held. The difference is real. Those keep the w13->w2 boundary and the hs round trip, or use the swap-AB mma.sync mainloop, which ran 1.7-2x slower. Nothing on the Do-not-try list covers a split-I fused design, and the Marlin knobs at small M are confirmed exhausted. This is not a re-run.

EXPECTED. The directions are consistent with TASK weights (each load counts equally; decode_16 and decode_32 are 2 of the 6). Expect the bs=1/8/128/256 rows flat within row noise of +-3-4% (LESSONS 182).

STALE BASE. The change says 'current master (40e2e860)'. Master is now bdbbc97b (combo-toprow), and the incumbent at T=32 is gated by C1a (_align_parallel) and C1b (post-norm). Build on bdbbc97b and keep those flags' paths intact for T outside 9..32.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (landed, T<=32): it defines the incumbent Marlin pair that the FSI gate competes against at T=9..32. A tuned-differently w2 (fp32 reduce) would make the incumbent slower and inflate the FSI ratio. This is part of the baseline, not an interaction, so no revert run is needed.
- post-expert-norm-triton-small-m (landed, T<32): FSI replaces this combine at T=9..16 with its own fp32-y32 Triton combine, and at T=32 it replaces SGLang's fused_post_expert_norm_weighted_sum. The combine is about 3.5 us, so roughly 2% of a decode_16 or decode_32 row difference could be the combine swap rather than the fused GEMM. Print combine time separately in the gate.
- triton-map-align-on-block-m-restack / align-two-phase-port-from17 (landed) plus master's C1a d32_align: FSI calls _align_small at block_size_m 32, while the incumbent at T=32 runs align2 or _align_parallel. The align kernels are 2.7-4.5 us, so a difference of 1-2 us is below resolution but can tilt a near-1.0 gate at decode_32.
- combo-toprow (master bdbbc97b, not in the landed list): its C1b hoisted post-norm and C1a flags at T==32 decide which incumbent FSI is gated against. If C1b is ON on the judge, the post-norm comparison above changes.
- Queued restack-occ2-wg2-b256-two-stream-halves-on-master: it adds two-stream Marlin halves at 8<=T<=32, exactly this file's band. If it lands first, the incumbent becomes 2-3% faster at decode_16 and decode_32. FSI's gate must then compare against that path, and its gain shrinks by the same amount. Re-base and gate against whichever is on master when claimed.
- marlin-block-m-two-smaller-on-postnorm, wg-pdl-gather-w13-w2-chain, align-parallel-per-expert-verify, sglang-marlin-path-parity: T>=128 or the base only. No interaction with T=9..32, and they keep the bs=128/256 control rows honest.

### Interference and A/B needs
No companion check is warranted. This is a rewrite stage judged on its own kernel rows, and the design already carries an in-run A/B. The prepare gate times FSI against the exact incumbent path per T class under cold-L2 graph replay and prints PATH CHOSEN with both timings. bs=1/8/128/256 run master byte for byte as controls. The open questions (consumer TB/s, cluster-sync cost, wave tail, combine swap) are cheaper and cleaner to settle inside the same prepare than with extra okc checks.

Add them as prepare-time diagnostic variants of the same module, each timed and printed per class, which s2 can then pick from:
- (a) CL=1 with plain red.global.add.v4.f32, no cluster. This isolates the cluster-reduce cost and is the stated fallback anyway.
- (b) KC=128 and KC=256, with units, waves and co-resident clusters printed.
- (c) the fsi kernel alone, with combine time printed separately.
- (d) kernel-only TB/s for each.

Align the gate with today's lesson. Require >= 1.03x over the incumbent on the WORSE of a judge-like (zipf / _wg_judge_ids) routing and a random-local routing, in two interleaved 41-rep rounds with an IQR bound. The file's single-round >= 1.02x p50 on skewed routing only is the selector that mis-routed bf16 cells today. Without these, a stage-1 loss would be ambiguous: you could not tell consumer, sync, tail and combine apart. With them, the single check is fully interpretable.

### Open questions
- Has the consumer fix (pingpong-consumers / producer-scale-expand, or wg2-pingpong-consumers-occ2-b256) harvested by claim time? DIAGNOSIS action 3/4 sequences this file after it. If not, should stage 1 still run as a bandwidth probe of N32 rega? The reviewer's view is yes, but M3 should be expected to miss.
- Has cuda-fused-split-i-moe-tiny-t landed, or will this stage implement the shared fsi_kernel, combine and gate from scratch? A from-scratch build of both the shared and the new parts on a GPU-less node is a larger compile-risk surface.
- Will restack-occ2-wg2-b256-two-stream-halves-on-master land before this is claimed? It changes the T=8..32 incumbent this gate competes against.
- How many CL=4 clusters does cudaOccupancyMaxActiveClusters actually report on the judge's H100 (GPC SM counts are not all multiples of 4)? Is the effective grid 128 or 132 CTAs, and how does that change the KC wave arithmetic?
- Would double-buffering the 16 KB partial tile (two buffers, one cluster.sync per tile instead of two), or a once-per-unit DSMEM reduction, fit the smem budget alongside the ring and h? If so, consider it for stage 1, since sync cost is the likeliest stage-1 failure after consumer TB/s.

