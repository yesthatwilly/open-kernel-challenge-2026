---
problem: muse_moe_w4a16
slug: cuda-fused-split-i-moe-tiny-t
created: 2026-10-25T153417Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 1 of a fused split-I MoE for tiny T: 1 <= num_tokens <= 8 only (decode_1 bs=1, verify_1
  bs=8, the bs=1 and bs=4 tests). num_tokens >= 9 runs master byte for byte (Marlin below 128, the
  s5 bucket gate above). Built on current master (40e2e860, the s5 bucket gate), new code only, in
  its own load_inline module. WHY: at T<=8 master is two Marlin launches with act_and_mul between
  them, at ~1.5-1.9 TB/s of unique weight bytes (decode_1 profile: Marlin pair 35.6 us for ~2
  local experts = ~57 MB; verify_1: 86-105 us for ~6 experts = ~179 MB). The floors are 11.4 and
  37.3 us. Every earlier small-T design kept the w13 -> w2 phase boundary or synchronised across
  it: the bs=1 persistent fused kernel used per-chunk flags and spins and went bimodal, then sat
  at 62 us; the swap-AB mma.sync kernel was 1.7-2x slower; the rega tiny path (split-K wgmma) only
  reached parity at T=1 (46.0 vs 45.5 us) and lost 1.2-1.5x at T=4-7. This design removes the
  boundary instead of synchronising it. MECHANISM: y = W2 . (silu(W1g x) * (W1u x)) is linear in
  the I-columns of h, so split I into chunks of KC columns. A work unit u = (active local expert
  e, I-chunk j) computes its own h[:, j] from the FULL K = 4096 of x (gate and up rows for those
  KC columns), applies silu * up in registers, rounds h to bf16 (as Marlin's cache2 is bf16),
  keeps it in shared memory, and immediately streams W2[:, j] (all 4096 H rows, KC k-columns) to
  add its partial y into an fp32 scratch with red.global.add. No unit waits for another unit: no
  flags, no spins, no grid barrier, no h round trip through global memory. Units = active experts
  x (4096 / KC), so with KC = 64 two experts give 128 units and six give 384: the grid fills at
  bs=1, which Marlin's per-expert tiling does not. KERNEL fsi_kernel: persistent, grid = SM count
  (state['wg_grid']), one CTA per SM, master's warp-specialised layout (1 producer warpgroup with
  setmaxnreg.dec, 2 consumer warpgroups), weights as the register-A operand of wgmma m64n16k16
  (the N16 instantiation master's bucket canary already proved; tokens of the expert on N, <= 8
  real rows padded to 16). Unit list on device: run master's _align_small at block_size_m 16
  (unchanged function, called with BSM 16; at T <= 8 every active expert is exactly one 16-row
  block) and master's wg_gather into the state's wg_xs, then each CTA derives units u = block b x
  chunk j from n_padded / block_experts in its prologue (master's s_be table code) and walks u =
  blockIdx.x, += gridDim.x. WEIGHTS: reuse master's lossless wg bank (state['wg_banks'], built in
  prepare whatever the bucket flags say: q13 gated tiles of 128 rows = 64 gate + 64 up h-columns,
  q2 tiles of 128 H rows, both as 4608-byte records per 64-k block). With KC = 64 a unit's w13
  part is q13 tile j of expert e (64 records, K = 4096) and its w2 part is the j-th 64-k record of
  each of the 32 q2 row tiles (32 records). One flat stage sequence per unit through the same
  mbarrier ring (STAGES 4, two records per stage): 32 w13 stages, then 16 w2 stages, so the ring
  never drains between the halves and the w2 bytes are already in flight when the w13 epilogue
  runs. B operands: w13 stages TMA the expert's 16 gathered x rows (k-slab of 128) from wg_xs
  exactly as master's N16 path; the w2 B operand is the h chunk the w13 epilogue wrote to shared
  memory in the same 128B-swizzled K-major layout master's B slab uses (16 rows x 64 k bf16 = 2
  KB). EPILOGUES: w13 = master's gated silu epilogue (fp32 acc * gs13_eff, silu(gate) * up, bf16)
  but into that smem tile instead of wg_hs; then a named barrier over the two consumer warpgroups.
  w2 = per 128-row H tile, acc * gs2_eff, transposed through smem (master's stmatrix epilogue
  path) so each thread holds 4 consecutive H values of one route, then red.global.add.v4.f32 into
  y32[route, h] for the REAL rows only (route = sorted_ids[slot]; padded slots skipped). y32 is a
  per-state fp32 scratch [8 * 8, 4096] zeroed in prepare. COMBINE: a new Triton kernel, master's
  _post_norm_combine_kernel formula (out[t] = sum over local k of g * rmsnorm(y * (1 +
  post_norm_w)), eps 1e-8, fp32) reading y32 at route rows t*8+k instead of bf16 y, and writing
  zeros back into every y32 row it read, so the scratch is zero again for the next call or replay
  (zero-after-read; remote routes are never written). Launches per call: _align_small, wg_gather,
  fsi_kernel, combine (master: 5). KC: compile KC = 64 and KC = 128 (a unit = two adjacent q13
  tiles and two w2 records per row tile) as two instantiations and let the prepare selector pick
  per T class; KC = 128 halves the fp32 reds and the unit count. PREPARE GATE (same machinery as
  master's _wg_select): T classes {1} and {2..8}. Correctness first: for T in 1, 2, 4, 7, 8 and
  routings skewed, random-local, one-expert and all-remote, compare against an fp32-dequant
  reference; bound max(2x master's own error on the same case, 1e-2) aggregate and 5x per row;
  also the s5 three-state copy check (prepared state, copy.deepcopy, tensor-cloning walk) with
  refilled graph replays, which must also prove y32 returns to zero after each replay. Then per
  class and per KC: full kernel_fn graphs, cold L2 (>= 256 MB write before every replay), 41
  interleaved A/B replays against master's path, enable a class only if its p50 beats master by >=
  1.02x on the judge-like skewed routing, and assert the enabled choice equals the printed winner.
  Print one PATH CHOSEN line per class with both timings, the chosen KC, the fsi kernel's own
  event time and its TB/s of weight bytes. kernel_fn branches on num_tokens and the prepare flags
  only. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and
  only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under
  __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy /
  L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes /
  smem (a spill line is a bug: fix before spending the check). No module-global device pointer,
  tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded
  per call from the tensor passed in that call (master's s5 rule; the judge prepares two states
  and copies them by cloning tensors). First launch in-process after a small synthetic canary,
  then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first
  measurement is an okc check, never a scored submit. STAGES PLANNED: s1 this file (correct,
  gated, measured per class); s2 = tuning from the s1 log (KC, ring depth, a third consumer
  warpgroup since N16 accumulators are only 8 fp32 regs per thread, the N8 instantiation
  m64n8k16); s3 = extend to 9 <= T <= 32 (see the sibling
  cuda-fused-split-i-moe-midt-cluster-reduce) or merge with it. Fallback at every stage: the class
  stays on master's Marlin path.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "decode_1: master shape 44.8-45.8 us, Marlin pair 35.6-36.7 us + act_and_mul 1.6 + post-norm 2.9 (profile); floor 11.4 us. M1: PATH CHOSEN = FSI for T=1 with the self-check passing. M2: fsi_kernel + combine <= 0.9x of Marlin pair + act + post-norm (<= 36 us). M3: fsi_kernel <= 22 us (2.6 TB/s on ~57 MB), shape <= 32 us. 128 units at KC=64 fill 132 SMs where Marlin's per-expert tiling does not."
  - shapes: bs=8 *
    direction: faster
    why: "verify_1: master shape 104.5-109.5 us, Marlin pair 86-105 us; floor 37.3 us. M2: fsi + combine <= 85 us. M3: fsi <= 60 us (3 TB/s on ~179 MB). Risk: ~16-20 local routes x 4096 x 64 chunks of fp32 red (~1.3 M red.v4 at KC=64, half at KC=128) may cost several us; the selector picks KC per class."
  - shapes: bs=16 *
    direction: flat
    why: "control: T >= 9 runs master byte for byte"
  - shapes: bs=32 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control (Marlin, B128 gate off)
  - shapes: bs=256 *
    direction: flat
    why: control (s5 wgmma B256)
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13 + act_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2 + silu_and_mul) and the post-norm combine at 1 <= num_tokens <= 8
review: required
status: queued
---

# cuda-fused-split-i-moe-tiny-t

## Change

Stage 1 of a fused split-I MoE for tiny T: 1 <= num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1 and bs=4 tests). num_tokens >= 9 runs master byte for byte (Marlin below 128, the s5 bucket gate above). Built on current master (40e2e860, the s5 bucket gate), new code only, in its own load_inline module. WHY: at T<=8 master is two Marlin launches with act_and_mul between them, at ~1.5-1.9 TB/s of unique weight bytes (decode_1 profile: Marlin pair 35.6 us for ~2 local experts = ~57 MB; verify_1: 86-105 us for ~6 experts = ~179 MB). The floors are 11.4 and 37.3 us. Every earlier small-T design kept the w13 -> w2 phase boundary or synchronised across it: the bs=1 persistent fused kernel used per-chunk flags and spins and went bimodal, then sat at 62 us; the swap-AB mma.sync kernel was 1.7-2x slower; the rega tiny path (split-K wgmma) only reached parity at T=1 (46.0 vs 45.5 us) and lost 1.2-1.5x at T=4-7. This design removes the boundary instead of synchronising it. MECHANISM: y = W2 . (silu(W1g x) * (W1u x)) is linear in the I-columns of h, so split I into chunks of KC columns. A work unit u = (active local expert e, I-chunk j) computes its own h[:, j] from the FULL K = 4096 of x (gate and up rows for those KC columns), applies silu * up in registers, rounds h to bf16 (as Marlin's cache2 is bf16), keeps it in shared memory, and immediately streams W2[:, j] (all 4096 H rows, KC k-columns) to add its partial y into an fp32 scratch with red.global.add. No unit waits for another unit: no flags, no spins, no grid barrier, no h round trip through global memory. Units = active experts x (4096 / KC), so with KC = 64 two experts give 128 units and six give 384: the grid fills at bs=1, which Marlin's per-expert tiling does not. KERNEL fsi_kernel: persistent, grid = SM count (state['wg_grid']), one CTA per SM, master's warp-specialised layout (1 producer warpgroup with setmaxnreg.dec, 2 consumer warpgroups), weights as the register-A operand of wgmma m64n16k16 (the N16 instantiation master's bucket canary already proved; tokens of the expert on N, <= 8 real rows padded to 16). Unit list on device: run master's _align_small at block_size_m 16 (unchanged function, called with BSM 16; at T <= 8 every active expert is exactly one 16-row block) and master's wg_gather into the state's wg_xs, then each CTA derives units u = block b x chunk j from n_padded / block_experts in its prologue (master's s_be table code) and walks u = blockIdx.x, += gridDim.x. WEIGHTS: reuse master's lossless wg bank (state['wg_banks'], built in prepare whatever the bucket flags say: q13 gated tiles of 128 rows = 64 gate + 64 up h-columns, q2 tiles of 128 H rows, both as 4608-byte records per 64-k block). With KC = 64 a unit's w13 part is q13 tile j of expert e (64 records, K = 4096) and its w2 part is the j-th 64-k record of each of the 32 q2 row tiles (32 records). One flat stage sequence per unit through the same mbarrier ring (STAGES 4, two records per stage): 32 w13 stages, then 16 w2 stages, so the ring never drains between the halves and the w2 bytes are already in flight when the w13 epilogue runs. B operands: w13 stages TMA the expert's 16 gathered x rows (k-slab of 128) from wg_xs exactly as master's N16 path; the w2 B operand is the h chunk the w13 epilogue wrote to shared memory in the same 128B-swizzled K-major layout master's B slab uses (16 rows x 64 k bf16 = 2 KB). EPILOGUES: w13 = master's gated silu epilogue (fp32 acc * gs13_eff, silu(gate) * up, bf16) but into that smem tile instead of wg_hs; then a named barrier over the two consumer warpgroups. w2 = per 128-row H tile, acc * gs2_eff, transposed through smem (master's stmatrix epilogue path) so each thread holds 4 consecutive H values of one route, then red.global.add.v4.f32 into y32[route, h] for the REAL rows only (route = sorted_ids[slot]; padded slots skipped). y32 is a per-state fp32 scratch [8 * 8, 4096] zeroed in prepare. COMBINE: a new Triton kernel, master's _post_norm_combine_kernel formula (out[t] = sum over local k of g * rmsnorm(y * (1 + post_norm_w)), eps 1e-8, fp32) reading y32 at route rows t*8+k instead of bf16 y, and writing zeros back into every y32 row it read, so the scratch is zero again for the next call or replay (zero-after-read; remote routes are never written). Launches per call: _align_small, wg_gather, fsi_kernel, combine (master: 5). KC: compile KC = 64 and KC = 128 (a unit = two adjacent q13 tiles and two w2 records per row tile) as two instantiations and let the prepare selector pick per T class; KC = 128 halves the fp32 reds and the unit count. PREPARE GATE (same machinery as master's _wg_select): T classes {1} and {2..8}. Correctness first: for T in 1, 2, 4, 7, 8 and routings skewed, random-local, one-expert and all-remote, compare against an fp32-dequant reference; bound max(2x master's own error on the same case, 1e-2) aggregate and 5x per row; also the s5 three-state copy check (prepared state, copy.deepcopy, tensor-cloning walk) with refilled graph replays, which must also prove y32 returns to zero after each replay. Then per class and per KC: full kernel_fn graphs, cold L2 (>= 256 MB write before every replay), 41 interleaved A/B replays against master's path, enable a class only if its p50 beats master by >= 1.02x on the judge-like skewed routing, and assert the enabled choice equals the printed winner. Print one PATH CHOSEN line per class with both timings, the chosen KC, the fsi kernel's own event time and its TB/s of weight bytes. kernel_fn branches on num_tokens and the prepare flags only. BUILD (as master's wg module): own load_inline module name, TORCH_CUDA_ARCH_LIST=9.0a and only -gencode arch=compute_90a,code=sm_90a, -O3 -lineinfo, wgmma/setmaxnreg bodies under __CUDA_ARCH_FEAT_SM90_ALL (else error word 2), plain bulk copies with no createpolicy / L2::cache_hint operand, print compile seconds and per-instantiation numRegs / localSizeBytes / smem (a spill line is a bug: fix before spending the check). No module-global device pointer, tensor map or tensor: every scratch tensor lives in the state and every tensor map is encoded per call from the tensor passed in that call (master's s5 rule; the judge prepares two states and copies them by cloning tensors). First launch in-process after a small synthetic canary, then torch.cuda.synchronize() and a print, so a fault is attributed; no child process. The first measurement is an okc check, never a scored submit. STAGES PLANNED: s1 this file (correct, gated, measured per class); s2 = tuning from the s1 log (KC, ring depth, a third consumer warpgroup since N16 accumulators are only 8 fp32 regs per thread, the N8 instantiation m64n8k16); s3 = extend to 9 <= T <= 32 (see the sibling cuda-fused-split-i-moe-midt-cluster-reduce) or merge with it. Fallback at every stage: the class stays on master's Marlin path.

## Expected

- `bs=1 *` faster: decode_1: master shape 44.8-45.8 us, Marlin pair 35.6-36.7 us + act_and_mul 1.6 + post-norm 2.9 (profile); floor 11.4 us. M1: PATH CHOSEN = FSI for T=1 with the self-check passing. M2: fsi_kernel + combine <= 0.9x of Marlin pair + act + post-norm (<= 36 us). M3: fsi_kernel <= 22 us (2.6 TB/s on ~57 MB), shape <= 32 us. 128 units at KC=64 fill 132 SMs where Marlin's per-expert tiling does not.
- `bs=8 *` faster: verify_1: master shape 104.5-109.5 us, Marlin pair 86-105 us; floor 37.3 us. M2: fsi + combine <= 85 us. M3: fsi <= 60 us (3 TB/s on ~179 MB). Risk: ~16-20 local routes x 4096 x 64 chunks of fp32 red (~1.3 M red.v4 at KC=64, half at KC=128) may cost several us; the selector picks KC per class.
- `bs=16 *` flat: control: T >= 9 runs master byte for byte
- `bs=32 *` flat: control
- `bs=128 *` flat: control (Marlin, B128 gate off)
- `bs=256 *` flat: control (s5 wgmma B256)

## Risk

Not a duplicate of the held cuda-fused-moe-midm-single-launch-routed or of the bs=1 persistent line (done/neutral cuda-fused-moe-bs1-persistent-*): both kept a w13 -> w2 dependency across CTAs (flags, spins, h scratch) and the mma.sync/cp.async mainloop that ran at ~1.5 TB/s; this one has no cross-unit dependency at all and reuses master's TMA ring + register-A wgmma, which runs and is correct on the judge. Main risks: (1) fp32 red throughput at verify_1 (mitigated by KC=128 and the gate); (2) per-unit re-reading of the expert's x rows (16 x 8 KB per unit, L2 resident); (3) the consumer dequant limit the B128/B256 lines are fighting (~2 TB/s): M3 needs it at ~2.6-3 TB/s, but M2 only needs parity-plus because the phase boundary, act_and_mul, the cache1/cache2 round trips and one launch disappear. Numerics: y is summed over 64 fp32 partials in a nondeterministic order (fine for a 1.5% bound; the self-check prints the run-to-run spread). Gate is shape + prepare flag only.

## Milestones

Rewrite stage 1 of `Marlin w13 + act_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2 + silu_and_mul) and the post-norm combine at 1 <= num_tokens <= 8`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
