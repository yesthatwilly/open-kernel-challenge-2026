---
problem: muse_prefill_layer
slug: experts-turn-rega-s4-port-t4096
created: 2026-10-25T054847Z
base: master
area: "w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets"
change: >
  Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at
  turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_4 (64%/43%), against
  byte/math floors of 189 us and 306 us, so the two turn loads sit at 1.01x SGLang while
  first_turn is 1.19x. Every Triton attempt lost to Marlin below first-turn sizes (LESSONS
  03:41Z). The kernel that beats Marlin at these sizes now exists and is replay-safe:
  muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch (job
  4e7f8be895a145ba). It has the SAME expert dimensions as this problem (H=4096 expert input,
  I=4096, w13 N=8192, w2 N=4096, K=4096), 168 regs, 0 spill, variant A (wgmma_wait<0> before the
  in-register dequant) proven finite in all four N buckets (16/32/64/128), and its prepare timing
  on all-local routings (the prefill regime) was, cold L2: 1024 routes wgmma 580.8 us vs
  Marlin+silu 921.6 us (0.63x), 2048 routes 699.7 vs 1278.0 us (0.55x). Port its _wg_* module
  (wg_gather into the state's xs scratch, wg_gemm w13 with the fused silu*up epilogue into the
  state's hs scratch, wg_gemm w2 into master's slot-ordered y [T, 8, 4096]) into _experts_marlin:
  when 200 < T <= 4096 and state['rega_ok'][T class] (T classes: <=1024 and 1025-4096, gated
  separately), align with moe_align_block_size(local_ids, WG_BSM=16, LOCAL_EXPERTS) exactly as s4
  does (local ids are already 0-31 / -1 here, so drop s4's expert_map gather and its
  per-(start,count) bank dict: there is one bank, w13[0..31]), run the gather and the two GEMMs,
  then master's fused_post_expert_norm_weighted_sum unchanged (slot_expert_ids=local_ids skips
  remote slots). Bucket choice per expert is s4's (by that expert's route count). The weight
  repack is s4's lossless permutation of the stored E2M1 codes and e4m3 scales, done once in
  prepare() from weights['w13'/'w2'] (not from the nibble-reordered Triton bank). Size the xs/hs
  scratch per state for T=4096 x 8 local routes worst case (2 x 268 MB per state). Expect up to
  ~3012 routes at turn_4, so N=128 tiles dominate there; at turn_1 (~753 routes, ~24 per expert)
  the 16/32 buckets dominate. Gate shapes: T=753 and T=3012, new (align + gather + w13/silu + w2)
  vs master's (_align_parallel + Marlin w13 + _silu_mul_sorted + Marlin w2). T > 4096 (first_turn)
  and every test outside 200 < T <= 4096 run master unchanged. RULES (all stages): kernel_fn stays
  CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input
  values, no data-dependent host control flow, no prints, no try/except and no per-call mutable
  Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from
  inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are
  dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that
  family stays held), BF16 activations, fp32 accumulation; every route of every token is computed.
  No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two
  states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device
  pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are
  either encoded per launch on the host from the tensors being passed (as __grid_constant__
  params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in
  the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only
  Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and
  _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A
  with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert,
  mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant
  reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss,
  non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path
  runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline
  (restore after); extra_cuda_cflags exactly
  ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg
  asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in
  the C++ bindings never call a templated Tensor::data_ptr<T>(); use
  reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol
  at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes /
  smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag;
  LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved
  alternations against master's exact incumbent block, empty-graph baseline subtracted, at the
  scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win,
  not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST
  lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a
  scored submit.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 736-769 us -> ~450-500 us (s4 measured 0.63x Marlin at 1024 all-local routes, cold L2); turn_1 sliding ~1101 -> ~830 us, full ~1530 -> ~1260 us; floor 189 us
  - shapes: "*load=turn_4*"
    direction: faster
    why: Marlin pair 2115-2148 us -> ~1100-1300 us (s4 0.55x at 2048 routes; N=128 bucket); turn_4 sliding ~3319 -> ~2400 us, full ~4849 -> ~3950 us; floor 306 us
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: T=16384 keeps the Triton NVFP4 engine
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and _silu_mul_sorted at 200 < T <= 4096
review: proceed
status: running
review_at: 2026-10-25T111120Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T111120Z
base_sha: c6fe58c3ab272abce5f98a2a19f884600fd2f6a8
base_blob: f2d91dd181aeb44de20d2eed62ab7e564f2b181f
branch: hyp/muse_prefill_layer/experts-turn-rega-s4-port-t4096
worktree: .worktrees/muse_prefill_layer/experts-turn-rega-s4-port-t4096
mode: null
restack_of: null
stale_base: null
---

# experts-turn-rega-s4-port-t4096

## Change

Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_4 (64%/43%), against byte/math floors of 189 us and 306 us, so the two turn loads sit at 1.01x SGLang while first_turn is 1.19x. Every Triton attempt lost to Marlin below first-turn sizes (LESSONS 03:41Z). The kernel that beats Marlin at these sizes now exists and is replay-safe: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.patch (job 4e7f8be895a145ba). It has the SAME expert dimensions as this problem (H=4096 expert input, I=4096, w13 N=8192, w2 N=4096, K=4096), 168 regs, 0 spill, variant A (wgmma_wait<0> before the in-register dequant) proven finite in all four N buckets (16/32/64/128), and its prepare timing on all-local routings (the prefill regime) was, cold L2: 1024 routes wgmma 580.8 us vs Marlin+silu 921.6 us (0.63x), 2048 routes 699.7 vs 1278.0 us (0.55x). Port its _wg_* module (wg_gather into the state's xs scratch, wg_gemm w13 with the fused silu*up epilogue into the state's hs scratch, wg_gemm w2 into master's slot-ordered y [T, 8, 4096]) into _experts_marlin: when 200 < T <= 4096 and state['rega_ok'][T class] (T classes: <=1024 and 1025-4096, gated separately), align with moe_align_block_size(local_ids, WG_BSM=16, LOCAL_EXPERTS) exactly as s4 does (local ids are already 0-31 / -1 here, so drop s4's expert_map gather and its per-(start,count) bank dict: there is one bank, w13[0..31]), run the gather and the two GEMMs, then master's fused_post_expert_norm_weighted_sum unchanged (slot_expert_ids=local_ids skips remote slots). Bucket choice per expert is s4's (by that expert's route count). The weight repack is s4's lossless permutation of the stored E2M1 codes and e4m3 scales, done once in prepare() from weights['w13'/'w2'] (not from the nibble-reordered Triton bank). Size the xs/hs scratch per state for T=4096 x 8 local routes worst case (2 x 268 MB per state). Expect up to ~3012 routes at turn_4, so N=128 tiles dominate there; at turn_1 (~753 routes, ~24 per expert) the 16/32 buckets dominate. Gate shapes: T=753 and T=3012, new (align + gather + w13/silu + w2) vs master's (_align_parallel + Marlin w13 + _silu_mul_sorted + Marlin w2). T > 4096 (first_turn) and every test outside 200 < T <= 4096 run master unchanged. RULES (all stages): kernel_fn stays CUDA-graph capturable with no host syncs (.item/.cpu/synchronize), no allocation sized by input values, no data-dependent host control flow, no prints, no try/except and no per-call mutable Python state; dispatch only on shapes/config and on flags prepare() set; nothing cached from inputs. Precision per TASK.md: NVFP4 expert weights stay NVFP4 in global memory and are dequantized inside the kernel only (no per-call or stored BF16 copy of any expert weight; that family stays held), BF16 activations, fp32 accumulation; every route of every token is computed. No child processes (in-process torch load_inline only). TWO JUDGE STATES: the judge prepares two states (setup copies: 2) and rotates them, so no module, static or C++ global may hold device pointers, TMA maps or buffers; scratch buffers belong to the state dict, and TMA descriptors are either encoded per launch on the host from the tensors being passed (as __grid_constant__ params) or stored per state. Do NOT port host-side 'live' checks that can silently fall back in the timed call (muse_moe_w4a16 s4 printed PATH CHOSEN = WGMMA, yet its judged profile shows only Marlin at bs=128/256, because kernel_fn also required key in state['wg_banks'] and _wg_state_live(state)). prepare() self-check: capture one graph per state, then replay A/B/A/B/A with inputs refilled in place between replays (routings: random, skewed 4-hot, all-one-expert, mixed local/remote; force every tile/bucket shape) and compare each output to an fp32 dequant reference with a bound calibrated on master's own error (rel-L2 and worst row); any miss, non-finite value or nonzero device error word prints one 'OFF: <reason>' line and master's path runs unchanged. BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' just before load_inline (restore after); extra_cuda_cflags exactly ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets the error word and returns; in the C++ bindings never call a templated Tensor::data_ptr<T>(); use reinterpret_cast<T*>(t.data_ptr()) (the ssdq build died at import on the undefined symbol at::TensorBase::data_ptr<long long>); print nvcc flags, compile seconds and regs / local bytes / smem / CTAs per SM / binaryVersion per kernel (a spill or 1 CTA/SM at >200 regs is a red flag; LESSONS 2026-10-25). GATE: CUDA-graph replays with a >=256 MB cold-L2 flush, >=7 interleaved alternations against master's exact incumbent block, empty-graph baseline subtracted, at the scored T with a soft routing; ON only if the new median is <= 0.95x the incumbent (it must win, not tie; LESSONS 04:36Z); print both medians and one PATH CHOSEN line per T class as the LAST lines of prepare() (the archived log keeps the tail). First measurement is an okc check, never a scored submit.

## Expected

- `*load=turn_1*` faster: Marlin pair 736-769 us -> ~450-500 us (s4 measured 0.63x Marlin at 1024 all-local routes, cold L2); turn_1 sliding ~1101 -> ~830 us, full ~1530 -> ~1260 us; floor 189 us
- `*load=turn_4*` faster: Marlin pair 2115-2148 us -> ~1100-1300 us (s4 0.55x at 2048 routes; N=128 bucket); turn_4 sliding ~3319 -> ~2400 us, full ~4849 -> ~3950 us; floor 306 us
- `*load=first_turn*` unchanged: T=16384 keeps the Triton NVFP4 engine

## Risk

Supersedes the held experts-turn1-cuda-wgmma-rega-port-bucket32 (its hold waited on the moe s3 harvest; s3 failed on the two-state bug and s4 fixed it with variant A in every bucket). Check the profile for the wg kernel names before reading any turn row: the moe s4 judged run never ran its kernel. Turn_1 rows drift -7% to -10% on unchanged code, so read the Marlin vs wg kernel rows.

## Milestones

Rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and _silu_mul_sorted at 200 < T <= 4096`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T111120Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a rewrite stage, but re-base the spec on today's master (ca1d41b): at turn_1 the incumbent is still Marlin (693/733 us), and that is where the unique value is. At turn_4 the incumbent is now ssdq (1878/1865 us, not Marlin 2148/2115), so the T in (1024,4096] class needs a three-way gate (rega vs ssdq vs Marlin) and an explicit dispatch order. The port must come from the s5 copy-safe code, not s4.

### Justification
A. The mechanism is plausible: same expert dimensions (K=4096, w13 N=8192, w2 N=4096, 32 local experts), and swap-AB register-A wgmma dequantizes each weight tile once per <=128-route bucket. That is the right shape for ~24 routes/expert at turn_1 (bucket 32) and ~94 at turn_4 (bucket 128). The floors check out: ~905 MB of NVFP4 weights + scales is about 189 us at H200 bandwidth, and 2*3012*4096*12288 FLOP is about 306 us. Both incumbents are far off: Marlin at turn_1 is 3.7x its floor, ssdq at turn_4 is 6.1x its floor and, by master's own counters, producer-bound (consumers wait 84%). Shape weights: turn_1 sliding 46 + full 16, turn_4 sliding 46 + full 16, against first_turn 138+48. The turn rows are about a third of the score, so a real expert win there moves the headline. WEAKEST LINK: the speed evidence. The 0.63x/0.55x figures are muse_moe_w4a16 prepare-time timings on synthetic all-local routings. The only JUDGED wgmma rows that exist (muse_moe s5 copy-safe-forced, job d4ad4f31d2764050) show wg_kernel w13+w2 = 238+111 = 349 us vs Marlin 306 us at bs=128 (a LOSS, 1.14x) and 271+126 = 397 vs 483 us at bs=256 (0.82x). That is about 2.3 TB/s of weight stream at small M, not 0.55x Marlin. Extrapolated to turn_1 (~753 routes), 0.8x Marlin (~560 us vs 693) is a more honest expectation than ~450-500 us. Against ssdq at turn_4 (already 0.87x Marlin), rega must reach <=0.83x Marlin just to clear a 0.95 gate vs ssdq; the prepare-time number says yes, the judged number says it is a coin flip. Stale facts in the change paragraph: base 'a81d19c', 'turn loads at 1.01x SGLang' (turn_4 is now 1.12x/1.07x), and a turn_4 gate 'vs master's Marlin pair'. The incumbent and the floor gap at turn_4 must be restated against ssdq.

### Confounders on the current kernel
- experts-turn4-ssdq-s1b-dataptr-symbol-fix (landed today, master ca1d41b): it owns T in (1024,4096] when state['ssdq_ok']. _experts checks ssdq first, so a naive port into _experts_marlin never runs at turn_4 (ssdq intercepts). Conversely, if the port puts rega ahead of ssdq and gates only against Marlin, a turn_4 row can regress vs the ssdq incumbent while 'passing' its gate. The turn_4 row is interpretable only if the gate compares rega against what master actually runs at T=3012 (ssdq when ON, else Marlin) and the PATH CHOSEN line names all three. Its scratch (xs/hs per state) also coexists with ssdq's: about 1 GB more device memory across two states. Harmless on H200, but check that both fit before the timing gate runs.
- experts-hybrid-marlin-turn-triton-first-turn (landed): it set MARLIN_MAX_TOKENS=4096, which bounds this change at T<=4096. first_turn must stay byte-identical, so any first_turn row movement is noise or a scratch/allocator side effect, not the change.
- stack-tonight-five-prefill (landed): _align_parallel and _silu_mul_sorted on the Marlin path are part of the turn_1 incumbent block. rega replaces align (moe_align_block_size with BSM=16) + silu (fused epilogue) + both GEMMs, so the turn_1 gate must time the whole incumbent block (align+Marlin+silu+Marlin) against the whole new block, as the spec says. Read the top-kernel rows, not just the headline: turn_1 drifts 7-10% on unchanged code.

### Interference and A/B needs
No companion. This is a rewrite stage, and the readout is the replaced kernels' own us in profile top_kernels at T=753 and T=3012, compared against the incumbent each T class actually runs. The ssdq interaction is a dispatch/gate design issue, not something an A/B run fixes: an ab-revert of ssdq would only re-measure the stale Marlin comparison. Required spec changes instead. (1) Build on master ca1d41b. (2) _experts order for T<=4096: rega if state['rega_ok'][class], else ssdq if T>1024 and ssdq_ok, else Marlin. (3) The T in (1024,4096] gate at T=3012 times all three in the same interleaved cold-L2 loop (>=7 alternations). rega ON only if <=0.95x min(ssdq, Marlin), and all three medians are printed in the last lines regardless of outcome, so the turn_4 comparison is read even when rega loses its gate. (4) The T<=1024 gate at T=753 is rega vs master's Marlin block, unchanged. (5) Port from muse_moe_w4a16 s5 (cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced.patch: per-call TMA encode as __grid_constant__, no wg_mptr_py/_wg_state_live, deepcopy + clone states in the replay check, and a torch.profiler LAUNCHED=wgmma assertion in prepare). Do not port s4, whose judged run never launched the kernel. Overlap with the queued experts-turn4-ssdq-s2-tma-raw-weight-ring: both target T=3012 and they do not stack. Running both today is a fair race, because each is gated against the incumbent at prepare time. If s2-tma lands first, this hypothesis's three-way gate automatically compares against the improved ssdq, as long as the gate times whatever _experts would dispatch rather than a hard-coded ssdq_kernel. experts-turn-rega-s4-w2-only-t4096 is now incoherent at turn_4, because ssdq replaces w13 too. Restrict it to T<=1024 or retire it if this lands. The held experts-turn1-cuda-wgmma-rega-port-bucket32 is superseded by this one.

### Open questions
- At turn_4, does rega beat the CURRENT ssdq (1870 us) or only Marlin? The judged muse_moe rows (0.82x Marlin at bs=256, 1.14x at bs=128) suggest the margin over ssdq may be thin. The three-way prepare print answers this even if rega is gated off.
- Which s-stage is the port source? s5-copy-safe-forced is the only rega build with judged kernel rows. muse_moe's in-flight s5-copy-safe-bucket-gate may change the bucket policy. Pin the source patch path in the build.
- Does the turn_1 gain survive judge-like soft routing? At ~24 routes/expert with skew, some experts hit bucket 64/128 and many hit bucket 16. s5's loss at bs=128 (~same per-expert counts as turn_1's low end) is the warning sign.
- If both this and experts-turn4-ssdq-s2-tma-raw-weight-ring pass, which lands first? The second must be re-gated against the first's master, not merged blind.

