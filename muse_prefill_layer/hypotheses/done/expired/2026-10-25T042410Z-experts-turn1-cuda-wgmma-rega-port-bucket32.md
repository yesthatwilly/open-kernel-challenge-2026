---
problem: muse_prefill_layer
slug: experts-turn1-cuda-wgmma-rega-port-bucket32
created: 2026-10-25T042410Z
base: master
area: "w13 + w2 expert GEMMs at T <= 1024 (turn_1): port of muse_moe_w4a16's register-A swap-AB wgmma NVFP4 grouped GEMM, 16/32-row buckets only"
change: >
  Built on master (a81d19c, stack-tonight-five-prefill). At turn_1 the two Marlin calls are
  650-770 us per layer (64-68% of the sliding and 47% of the full turn_1 shape) against a 189 us
  byte floor (~905 MB of NVFP4 weights + scales for the 32 local experts), i.e. Marlin streams at
  only ~1.3 TB/s. Cause: Marlin puts tokens on the mma.sync M side with moe_block_size 64 while an
  expert gets only ~24 local routes at T=753, so 2.7x of the MMA work is padding; smaller
  block_size_m re-reads weights and regressed (experts-marlin-block-m-local-rows 0.984). A swap-AB
  kernel puts the weights on the M side and the ~24 tokens on the wgmma N side, reading each
  weight once. Port that kernel from
  muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix.patch (the
  _wg_* module: w13 with fused silu*up epilogue and w2, NVFP4 dequantized in registers into the
  wgmma A fragment, tokens as B via TMA, sm_90a, in-process build). Use ONLY its 16- and 32-row N
  buckets, the two its per-bucket diagnostic proved correct (w13 rel 6.3e-3 / 6.5e-3, w2 3.4e-3 /
  3.5e-3); the 64/128 buckets were non-finite and are not instantiated. Adapt the dimensions to
  this problem (expert input K=4096, w13 N=8192, w2 N=4096 K=4096, 32 local experts; assert the
  template constants match), and the routing contract to master: in _experts_marlin, when
  x.shape[0] <= 1024 and state['wg_turn_ok'], call master's _align_parallel(local_ids, 0,
  LOCAL_EXPERTS, 32) (local routes only) so every 32-row block is one N=32 tile of one expert (an
  expert with more than 32 routes gets consecutive tiles, scheduled adjacently so its weight tile
  is an L2 hit), then the ported w13+silu kernel writes cache2 and the ported w2 kernel writes y
  in master's [T, 8, 4096] slot layout; fused_post_expert_norm_weighted_sum is unchanged. The
  kernel's weight repack is a lossless permutation of the stored E2M1 codes and e4m3 scales done
  in prepare(). T > 1024 (turn_4, first_turn) and every test with T > 1024 run master's code
  unchanged. Build rules (LESSONS 2026-10-25): torch load_inline with
  os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before the build (restored after) and only
  -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3 -lineinfo in extra_cuda_cflags; all
  wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device
  error word and returns; print the nvcc flags, compile seconds, and regs / local bytes / smem /
  binaryVersion per kernel. In-process only (never a child process: the rules reject submissions
  that start processes). Prepare canary: one tiny launch per kernel, each followed by
  torch.cuda.synchronize() and a flushed print, then a finiteness check that names the first
  non-finite tensor; self-checks must force every tile/bucket shape (an all-one-expert routing and
  a skewed one, not only random routings) and compare against an fp32 dequant reference with a
  bound calibrated on the incumbent's own error (not a fixed 1e-2 against another approximate
  kernel). Hard gate: interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of
  the new kernels vs the incumbent at the scored T; ON only at <= 0.92x; print both times and PATH
  CHOSEN and assert the chosen one is the faster. Any failure prints one OFF: <reason> line and
  master runs unchanged. The first measurement must be an okc check, never a scored submit. Gate
  shape: T=753 soft routing, new pair (w13+silu, w2) vs master's Marlin pair + _silu_mul_sorted.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 650-770 us -> <= 450 us (Milestone M2; the moe s2 prepare warm-eager figure was 515.9 vs Marlin 911.3 us at M=128); turn_1 sliding ~1100 -> ~800 us, full ~1530 -> ~1230 us; floor 189 us for the replaced kernels
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: T=3012 > 1024 keeps Marlin
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps the Triton engine
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and the silu-mul between them at T <= 1024 (turn_1, T=753)
review: proceed
status: done
review_at: 2026-10-25T043311Z
review_companions: null
hold_reason: "Reviewer: proceed but wait for moe s3 wide-bucket-fix (3bdec0e58d684c17) harvest, which decides the register-A hazard variant; then build per the review builder_instruction (168-reg cap, wait<0>/hazard-free order)."
outcome: expired
verdict_detail: "retired: already replaced by rega s4 (experts-turn-rega-s4-port-t4096)"
expired_at: 2026-10-25T211931Z
---

# experts-turn1-cuda-wgmma-rega-port-bucket32

## Change

Built on master (a81d19c, stack-tonight-five-prefill). At turn_1 the two Marlin calls are 650-770 us per layer (64-68% of the sliding and 47% of the full turn_1 shape) against a 189 us byte floor (~905 MB of NVFP4 weights + scales for the 32 local experts), i.e. Marlin streams at only ~1.3 TB/s. Cause: Marlin puts tokens on the mma.sync M side with moe_block_size 64 while an expert gets only ~24 local routes at T=753, so 2.7x of the MMA work is padding; smaller block_size_m re-reads weights and regressed (experts-marlin-block-m-local-rows 0.984). A swap-AB kernel puts the weights on the M side and the ~24 tokens on the wgmma N side, reading each weight once. Port that kernel from muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix.patch (the _wg_* module: w13 with fused silu*up epilogue and w2, NVFP4 dequantized in registers into the wgmma A fragment, tokens as B via TMA, sm_90a, in-process build). Use ONLY its 16- and 32-row N buckets, the two its per-bucket diagnostic proved correct (w13 rel 6.3e-3 / 6.5e-3, w2 3.4e-3 / 3.5e-3); the 64/128 buckets were non-finite and are not instantiated. Adapt the dimensions to this problem (expert input K=4096, w13 N=8192, w2 N=4096 K=4096, 32 local experts; assert the template constants match), and the routing contract to master: in _experts_marlin, when x.shape[0] <= 1024 and state['wg_turn_ok'], call master's _align_parallel(local_ids, 0, LOCAL_EXPERTS, 32) (local routes only) so every 32-row block is one N=32 tile of one expert (an expert with more than 32 routes gets consecutive tiles, scheduled adjacently so its weight tile is an L2 hit), then the ported w13+silu kernel writes cache2 and the ported w2 kernel writes y in master's [T, 8, 4096] slot layout; fused_post_expert_norm_weighted_sum is unchanged. The kernel's weight repack is a lossless permutation of the stored E2M1 codes and e4m3 scales done in prepare(). T > 1024 (turn_4, first_turn) and every test with T > 1024 run master's code unchanged. Build rules (LESSONS 2026-10-25): torch load_inline with os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before the build (restored after) and only -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3 -lineinfo in extra_cuda_cflags; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error word and returns; print the nvcc flags, compile seconds, and regs / local bytes / smem / binaryVersion per kernel. In-process only (never a child process: the rules reject submissions that start processes). Prepare canary: one tiny launch per kernel, each followed by torch.cuda.synchronize() and a flushed print, then a finiteness check that names the first non-finite tensor; self-checks must force every tile/bucket shape (an all-one-expert routing and a skewed one, not only random routings) and compare against an fp32 dequant reference with a bound calibrated on the incumbent's own error (not a fixed 1e-2 against another approximate kernel). Hard gate: interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of the new kernels vs the incumbent at the scored T; ON only at <= 0.92x; print both times and PATH CHOSEN and assert the chosen one is the faster. Any failure prints one OFF: <reason> line and master runs unchanged. The first measurement must be an okc check, never a scored submit. Gate shape: T=753 soft routing, new pair (w13+silu, w2) vs master's Marlin pair + _silu_mul_sorted.

## Expected

- `*load=turn_1*` faster: Marlin pair 650-770 us -> <= 450 us (Milestone M2; the moe s2 prepare warm-eager figure was 515.9 vs Marlin 911.3 us at M=128); turn_1 sliding ~1100 -> ~800 us, full ~1530 -> ~1230 us; floor 189 us for the replaced kernels
- `*load=turn_4*` unchanged: T=3012 > 1024 keeps Marlin
- `*load=first_turn*` unchanged: T=16384 keeps the Triton engine

## Risk

Milestones (judged on the replaced kernels' own rows at turn_1, not the headline): M0 builds sm_90a, 0 B local, canary and every bucket finite; M1 prepare A/B <= 1.0x Marlin; M2 <= 0.92x so the gate turns it ON and the judge's turn_1 expert rows sum below 500 us. Kill: any bucket non-finite after the port, or A/B ratio > 1.0 at T=753 -> the line is dead at turn_1. Turn_1 rows drift -7 to -10% on untouched code (LESSONS 03:41Z), so read the kernel rows and PATH CHOSEN, never the shape z alone. Source kernel is unproven at its own judge (moe stage 3 still running); this run uses only its verified buckets.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and the silu-mul between them at T <= 1024 (turn_1, T=753)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T043311Z by an adversarial subagent (model opus).

**Verdict: proceed.** Highest-EV prefill move on the board (turn_1 load is 1.01x and Marlin is 64-68%/47% of its shapes; a 750->450 us expert pair is ~1.33x on the turn_1 load, ~+10% headline). Sound swap-AB design, but SEQUENCE IT AFTER moe s3 (job 3bdec0e58d684c17) is harvested and port from s3's commit with its hazard-free A-fragment variant, because the N16/N32 buckets may be passing by luck.

### Justification
Mechanism holds: at T=753 ~753 local routes / 32 experts = ~24 rows per expert, Marlin pads to moe_block_size 64 on the mma M side (2.7x MMA waste) and streams 905 MB at ~1.3 TB/s vs the 189 us floor; swap-AB puts the 24 tokens on wgmma N and reads each weight once. The moe and prefill expert dims are identical (K=4096, w13 N=8192, w2 N=4096 K=4096, 32 local experts, same NVFP4 layout), so the port is dimensional, not a redesign. Profile supports it: Marlin 768.6/735.9 us at turn_1 (master check c4d14d62), 649.7/709.5 in submit 502f2eab. Weakest link: the only speed evidence is the moe s2 warm-eager figure (515.9 vs 911.3 us at M=128), and that M=128 timing ran with 64/128-bucket chunks that were producing NaN, so N16/N32 have NO clean timing yet; moe s3 is the first run that prints cold-L2 per-bucket N16/N32 times vs Marlin. Second weak link: the moe s3 review raises a register-A WAR hazard (A fragments rewritten while an RS wgmma is in flight); if s3's variant A (wait<0>) or B (double-buffered a[]) is what fixes N64/N128, then N16/N32 correctness in s2 was luck and the port must carry the fix. Register cap: the real cap is 168 under __launch_bounds__(384,1) (setmaxnreg.inc 232 does not lift ptxas's compile budget); at N16/N32 the accumulators are 8/16 regs so the cap does not bind, and an N16/N32-only instantiation (no 64/128 code in the same function) should drop regs and the w2 8 B spill.

### Confounders on the current kernel
- stack-tonight-five-prefill: supplies the incumbent this gate is measured against (_align_parallel local-routes align + _silu_mul_sorted + Marlin pair); the gate must time new pair vs exactly that, which the hypothesis does. Not a confound of the result, but the new path must consume _align_parallel's Marlin-format (sorted_ids, block_experts, n_padded) at bsm=32, not moe_align_block_size's, and its self-check must run on that output.
- experts-hybrid-marlin-turn-triton-first-turn: defines MARLIN_MAX_TOKENS=4096; the new T<=1024 branch sits inside _experts_marlin under that gate. Tests with T=300/753 (and the T=200 outliers test) take the new path, so correctness is exercised by the judge's tests too.
- sglang-pipeline-parity, experts-triton-w4a16-first-turn-blockt256: no interaction (first_turn engine untouched).

### Interference and A/B needs
No companion. It is a kind:rewrite judged on its own Marlin-replacement rows at turn_1, and its hard prepare gate prints both times and PATH CHOSEN, so a loss means 'gated off' (master re-measured) and a gain is attributable from the kernel rows. turn_1 shape rows drift -7 to -10% on untouched code, so the builder/harvester must read the per-kernel rows and PATH CHOSEN, never the shape z.

### Open questions
- Does moe s3 (3bdec0e58d684c17) show the A-fragment hazard is real (variant A/B fixes N64/N128)? If yes, port s3's variant for N16/N32; if s3 fails both variants, still apply variant A (wgmma wait<0> before overwriting A regs) - correct per the PTX ISA, small speed cost.
- s3's cold-L2 N16/N32 pair times vs Marlin: if w13+w2 at N32 does not beat Marlin there, the turn_1 port's M1 is unlikely; demote before spending the check.
- Grid order: are the chunks of one expert adjacent for the SAME weight N-tile (N-tile major, chunk minor)? Otherwise an expert with >32 routes re-reads its 28 MB of weights from HBM.
- Does the soft distribution at T=753 give some experts 100+ local routes (4+ chunks)? Print the chunk histogram in prepare from a soft synthetic routing.

