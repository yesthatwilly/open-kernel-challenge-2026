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
status: done
review_at: 2026-10-25T111120Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T111120Z
base_sha: c6fe58c3ab272abce5f98a2a19f884600fd2f6a8
base_blob: f2d91dd181aeb44de20d2eed62ab7e564f2b181f
branch: hyp/muse_prefill_layer/experts-turn-rega-s4-port-t4096
worktree: .worktrees/muse_prefill_layer/experts-turn-rega-s4-port-t4096
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T112615Z
impl_commit: cbec93e99c3d87bf325820062d8daf93a41b3969
submission_sha256: ea5b5ea0bbea782ee143ec92ec2a43f74a9a277b43bb22bce908e0e8bc029ba5
norm_sha256: bde5993f4966b997dc84622f3b16b884aa893547377dfce8b01f04879e8ed5c1
submission_blob: 712edc5ea5be2dde8f558a016d0877b285340ee4
lint_warnings: 51
attempt_file: 2026-10-25T112615Z-check-experts-turn-rega-s4-port-t4096.md
prepared_at: 2026-10-25T112615Z
job: 2910b2fb4b744865
dispatched_at: 2026-10-25T112626Z
outcome: gain
verdict_detail: gain
speedup: 1.3274587526558561
speedup_ci: [1.3075468713660183, 1.347370633945694]
harvested_at: 2026-10-25T115711Z
best_at_harvest: 1.1075501334948785
promotion: merged
promotion_commit: 23c1d5fa90e63b93b142d0f2461c4647a1e9d68c
promotion_attempt: 2026-10-25T115714Z-submit-experts-turn-rega-s4-port-t4096.md
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

## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-turn-rega-s4-port-t4096`, only `muse_prefill_layer/submission.py` edited (+1331/-4). Base: the worktree HEAD c6fe58c3 has a byte-identical `submission.py` to current master 537653db (`git diff c6fe58c3 master -- muse_prefill_layer/submission.py` is empty; it carries ssdq s1b), so no 3-way merge was needed; everything outside the change is master's.

What was added (all new code sits between `_experts_ssdq`'s block and `kernel_fn`; master's functions are untouched except `_experts` and the last lines of `prepare`):
- `_REGA_CPP/_REGA_HEAD/_REGA_BODY`: the s5 copy-safe kernel (git show hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced:muse_moe_w4a16/submission.py, lines 504-1125), renamed `rega_*`, one `load_inline` extension `muse_prefill_rega_v1`, `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored, `extra_cuda_cflags = ['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a']`, all wgmma/setmaxnreg behind `__CUDA_ARCH_FEAT_SM90_ALL` with an `#else` that sets error word 2; no templated `data_ptr<T>()` (reinterpret_cast, and a pre-build audit `_rega_source_audit` reuses the ssdq regex); regs/local bytes/smem/CTAs per SM/binaryVersion/flags/compile seconds are printed.
- `_experts_rega(x, topk_weights, local_ids, state)`: master's `_align_parallel` at block 16 -> `rega_gather` -> `rega_gemm` w13 (fused silu*up) into hs -> `rega_gemm` w2 scattered to master's slot-ordered y [T*8, 4096] -> master's `fused_post_expert_norm_weighted_sum` (slot_expert_ids=local_ids). xs/hs/y are allocated per call (graph pool, as master's ssdq path does), so no persistent scratch exists to be aliased by a state copy.
- `_rega_bank`: s5's lossless retiling of w13/w2 codes and e4m3 scales (round trip asserted equal), done once in prepare() from `weights[...]`; global scales fp32 x 128 (exact). NVFP4 bytes only: no BF16 copy of any weight.
- `_rega_setup` (called at the end of `prepare` after `_ssdq_setup`): guards (align_par_ok, 200 s budget, sm_90, weight shapes, smem), source audit, build, attribute / occupancy / register-pool check, global/static audit, retile, canary with before/after prints, 7-case eager self-check, copy / replay check, two gates, final PATH CHOSEN. Any failure prints one `OFF:` line and both flags stay False.
- `_experts`: for T <= MARLIN_MAX_TOKENS: rega if `T > REGA_MIN_TOKENS (200)` and `state['rega_ok']['low' if T <= 1024 else 'high']`; else ssdq if `T > SSDQ_MIN_TOKENS` and `ssdq_ok`; else Marlin. Reads a shape and prepare-time flags only. T > 4096 untouched.

Changes to the s5 kernel for T up to 4096 (s5 served T <= 512 only):
- Block table 384 -> 2304 blocks (T=4096 needs 2078), chunk table 512 (a chunk is <= 8 blocks, so <= ~320 are reachable). The one-thread-per-block prologue was replaced by rounds of 384 blocks with a running base, run starts found with `atomicCAS` into a per-expert table, and per-expert `gfirst/gcount` instead of per-chunk group arrays (no O(run) walk-back). Overflow of either table or a second run of one expert sets error word 3 / 4 (never a silent truncation). Static smem ~13.8 KB + 203,776 B dynamic < 227 KB; a Python model of the new chunking (`scratchpad/rega/model.py`) matched the host mirror of the original chunk rule and covered every (chunk, tile) item exactly once on 60 random layouts up to 2078 blocks. The mock-header g++ -fsyntax-only pass over the whole CUDA source is clean (no nvcc / GPU on this node: nothing was compiled for sm_90a or run).
- Launch marker: w13 writes err[1], w2 writes err[2] (err is int32[3] per state) so BOTH kernels are proven launched.
- silu*up epilogue: fp32 from the fp32 accumulators and rounded to bf16 once (s5 rounded gate and up to bf16 first, SGLang-parity; dropped per the rules, so rega is slightly MORE exact than Marlin there). The self-check reference keeps h in fp32 and rounds a once.
- The only file-scope state in s5 (the `cuTensorMapEncodeTiled` function pointer) is gone: every `rega_gemm` call resolves the entry point and encodes its 4 maps from the bsrc passed in that call into a stack `Maps` passed as `__grid_constant__`.

Adversarial-review items, each applied:
1. Base on master: done (same submission.py as 537653db).
2. Dispatch order rega > ssdq > Marlin: done in `_experts`, classes 'low' (T<=1024) and 'high' gated separately.
3. Three-way gate at T=3012: `_rega_gate` times in ONE interleaved loop (9 alternations, order rotated and flipped, >=256 MB cold-L2 flush before each replay, empty-graph baseline subtracted) rega, `dispatch` (= `_experts` with rega off, i.e. whatever master would actually run: ssdq when ON, else Marlin), `ssdq` itself when ON, and `marlin`. rega ON only if <= 0.95x the fastest of the alternatives. All medians are printed in the last lines of prepare whatever the outcome. If a correctness stage fails first there is nothing valid to time and the OFF line says why.
4. T<=1024 gate at T=753: rega vs `_experts` with rega off (= master's whole Marlin block: align + Marlin + silu + Marlin), plus `_experts_marlin` as a cross-check column.
5. Port source: s5 copy-safe code (not s4); per-call TMA encode as `__grid_constant__`; no `wg_mptr_py`, no `_wg_state_live`, no key/pointer check in the timed call; the copy check (`_rega_replay_check`) builds state B with `copy.deepcopy` (ext module through the memo) and state C with a tensor-cloning walk after a pad allocation (over the pieces `_experts_rega` reads, not the whole state, to avoid cloning every bank), captures one graph per state per T in (753, 3012, 4096), replays A,B,C,A,B per routing in (random, skewed, one) with inputs refilled in place, compares with the fp32 reference (bound max(2x Marlin's own error, 8e-3), worst row 5x); LAUNCHED asserted from both launch markers + error word 0 in each state's own error tensor after every replay (error-word route, no torch.profiler).
Open questions: bucket policy is s5-copy-safe-forced's (pinned in the header comment; the in-flight bucket-gate variant is not ported). The gate routing is uniform random top-8 (T/32 ~ 24 and ~94 routes per expert), not a skewed judge-like routing: a judged soft routing with some hot experts uses the 64/128 buckets more, so read the judged kernel rows. Memory: no persistent scratch; per call at T=3012 about 200 MB xs + 200 MB hs + 200 MB y from the graph pool, bank 0.9 GB per state.

Deviations / risks: (a) scratch is per call rather than per state (spec said per state; per call cannot be aliased by a state copy and is how master's ssdq works; it costs nothing at replay). (b) Only the post-sort align at bsm=16 was not covered by master's `_align_par_setup` (it self-checks the bsm Marlin picks); the rega self-check is end-to-end against the fp32 reference, which catches an align fault. (c) Biggest risk: the build (never compiled here; a first `okc check`, not a scored submit) and the T up to 4096 table logic (s5 only ran T <= 512 on the judge); the canary / self-check prints name the kernel on a fault, and a non-launching or wrong kernel keeps master.

## Rules check

- [x] CUDA-graph capturable: `_experts_rega` has no `.item()/.cpu()/synchronize()/prints` (the tracer is only passed by prepare and is a callable object, not reachable by name from kernel_fn); no allocation sized by input values (rows come from `block_experts.shape`, a shape), no control flow on values; the host-side map encode is stream-free and runs at capture.
- [x] Nothing derived from `data` is cached across calls: the only new per-state tensors are the retiled bank and the error/marker tensor; xs/hs/y are per call; no module/static/file-scope variable in C++ or Python (audited at prepare by `_rega_audit_cpp` / `_rega_audit_globals`; the C++ audit was run locally on the generated source: 0 findings).
- [x] Weights: repacked losslessly in `prepare()` only (E2M1 codes + e4m3 scales permuted, round trips asserted equal, fp32 global scale x 128 exact); NVFP4 stays NVFP4, dequantized inside the kernel only; no stored BF16 dequant of any expert weight.
- [x] Activations BF16 where SGLang keeps BF16 (xs, hs, y are bf16, fp32 accumulation); silu*up is fp32 and rounded to bf16 once; KV cache FP8 and attention untouched (n/a: not changed).
- [x] Attention sees every key: n/a, attention code untouched; every route of every token is computed by the expert kernels.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (`load_inline` in process only; okc_lint clean of process/network/harness findings).
- [x] All work happens inside the timed call; no output caching or replay; no pointer-equality gate or live-state check in `kernel_fn` (dispatch is on T and `state['rega_ok'][class]` only).
- [x] sm_90a only: `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` (restored), `-gencode=arch=compute_90a,code=sm_90a`, wgmma/setmaxnreg under `__CUDA_ARCH_FEAT_SM90_ALL` with an error-word-2 `#else`, arch flags printed.
- [x] Two judge states: per-state data is cloned with the state; the copy check proves graphs on a deepcopy and a tensor-clone state replay correctly with refilled inputs and each leave their own launch markers.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_prefill_layer/submission.py --problem muse_prefill_layer`): `OK with warnings`, 0 errors; 48 host-sync warnings, all outside kernel_fn's call graph (prepare-only `synchronize`/`.tolist()`, most of them master's own), 3 dtype warnings (uint8 for the lossless NVFP4 byte bank and one pad buffer). `python3 -I scratchpad/undef.py`: `[]`. `py_compile`: ok, `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 (opus) against master and the s5 copy-safe-forced source. **Verdict: dispatch-with-fixes.**

- The diff against s5 is limited to what the notes describe: the new prologue, silu*up in fp32 rounded once, per-call map encode with the file-scope `g_encode_pfn` removed, err[1]/err[2] markers and `reinterpret_cast` accessors. The rest of the kernel body is s5's.
- Prologue: `_align_parallel` at bsm 16 gives n_blocks = cdiv(8T + 480, 16), which is 2078 at T=4096. That is within MAXBLK 2304 and within the `block_experts.numel()` TORCH_CHECK, and sorted_ids.numel() equals the xs rows. Blocks come sorted by expert with a -1 tail. A Python emulation of the rounds, CAS, gfirst/gcount and decode_item ran on align outputs for T from 201 to 4096 with uniform, all-local, skewed and one-expert routings. Every (chunk, tile) was covered exactly once and every local block was in exactly one chunk, with no error word. The maxima were 2066 blocks and 272 chunks (MAXC 512). The ballot, s_wsum and sync ordering is race-free.
- Scratch xs/hs/y is sized from block_experts.shape and T only, from the graph pool. Dispatch is rega, then ssdq, then Marlin, decided by T and the prepare flags only. The gates are 'low' at T=753 against dispatch (Marlin) and 'high' at T=3012 against dispatch, ssdq and Marlin in one interleaved loop. kernel_fn has no try/except, prints or mutable state. Python and C++ have no file-scope device state (C++ audit re-run: 0 findings). The replay check uses separate per-state err tensors that are zeroed before each replay, so it is meaningful. Any failure in setup leaves rega_ok False and master's ssdq/Marlin path unchanged.

Fix:
1. `rega_kernel`: added a shared `s_bad` flag that is set together with error word 4 (an expert with two block runs). When it is set, `total = 0`. Without this, decode_item's group arithmetic could index outside s_cblk and then produce a wild weight or output address, and the resulting illegal address would poison the CUDA context and kill master's fallback. With the sorted align this cannot happen, so the change is defensive only. undef.py `[]`, py_compile ok, okc_lint OK with warnings (prepare-only syncs), __pycache__ removed.

Residual risk: the code has never been compiled (build errors or regs < 168 turn the path OFF cleanly). A device fault in the first canary launch would still poison the context. The prepare budget is tight only if the build takes more than ~150 s.

## Result

- job `2910b2fb4b744865` (check), status done, passed True, tests 5/5
- speedup 1.3275x, 95% CI [1.3075, 1.3474], score 291838.7477011382 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.1075501334948785, "ci": [1.0909368814924554, 1.1241633854973017], "job": "810b5f7decc84206"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-turn4-ssdq-s1b-dataptr-symbol-fix.result.json
- **outcome: gain** (gain)
- files: `muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-port-t4096.result.json`, `muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-port-t4096.patch`; branch `hyp/muse_prefill_layer/experts-turn-rega-s4-port-t4096`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1081.49 | 730.17 | 1.4116 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1460.63 | 1105.64 | 1.2731 | gained |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3012.25 | 2167.2 | 1.3305 | gained |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4686.3 | 3748.07 | 1.2089 | gained |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14970.94 | 14585.46 | 1.0034 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14118.46 | 14010.59 | 1.0055 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18379.93 | 17741.73 | 0.9943 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22181.89 | 21874.7 | 0.9855 | flat |

gained on sliding turn_1, turn_4; full turn_1, turn_4. Expected hit: 2/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) and _silu_mul_sorted at 200 < T <= 4096`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-turn-rega-s4-port-t4096 --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- candidate for promotion; stack further changes with `depends_on` once it lands.

