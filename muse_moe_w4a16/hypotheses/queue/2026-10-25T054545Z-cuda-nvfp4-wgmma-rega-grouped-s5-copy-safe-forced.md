---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced
created: 2026-10-25T054545Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 5 of the wgmma register-A grouped NVFP4 rewrite, built on the s4 commit
  (cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe, job 4e7f8be895a145ba) rebased onto master 9080a71
  (w2 atomic add at T<=32 stays). FINDING that motivates it: s4 logged PATH CHOSEN = WGMMA for
  128<=T<=512 and passed its own two-state replay check, yet the judge's profile at bs=128 and
  bs=256 shows only `Marlin<...>` (2 calls, 306.2 / 483.2 us) plus act_and_mul: the wgmma kernels
  never ran in the judged call. The likely cause is s4's kernel_fn gate `_wg_state_live(state)`,
  which compares a stored Python tuple state['wg_mptr_py'] with the live
  state['wg_xs'/'wg_hs'].data_ptr(): any state the judge makes by copying the prepared dict (new
  tensors, copied tuple and copied CPU map blob) fails it and silently runs Marlin. s4's
  self-check built state B with a second setup, never by copying, so it could not see this.
  CHANGE: (1) delete state['wg_maps'], state['wg_mptr'], state['wg_mptr_py'] and _wg_state_live;
  wg_gemm (C++ host side) encodes the two cuTensorMapEncodeTiled descriptors it needs on every
  call from the pointer of the bsrc tensor it is already passed (state['wg_xs'] for w13,
  state['wg_hs'] for w2) into a local CUtensorMap passed by value as a __grid_constant__ kernel
  parameter. This is host-only work at capture time (no device sync, no allocation, nothing
  persisted), so a graph captured on any state, original or copied, points at that state's own
  scratch. (2) kernel_fn gate becomes shape + prepare flag only: state.get('wgmma_ok') and
  128<=T<=512 and expert_map is not None. (3) prepare self-check adds state B =
  copy.deepcopy(state) and state C = {k: v.clone() if tensor else v} (build each inside prepare's
  own try/except; skip a copy method that raises and print which), capture one graph per state,
  then alternate replays A,B,C,A,B with refilled inputs at T=128/256/257 over random, skewed-4hot,
  all-one-expert and mixed-local-remote routings; every replay must match an fp32-dequant
  reference within master's own error x1.5 (print both). (4) NO timing gate in this stage: enable
  whenever correctness passes, because this stage exists to get the first judged per-shape wgmma
  kernel rows (five stages so far, zero judged wgmma rows). (5) Prove the launched kernel: in
  prepare, run one torch.profiler pass over a graph replay of each state at T=128 and assert the
  wgmma kernel names appear and `Marlin` does not; print 'LAUNCHED = wgmma' or turn the path off
  with the reason. (6) Print the prepare cold-L2 graph-replay timings at a judge-like routing (see
  notes) beside Marlin's, compactly and late in the log (the log is a ~15 KB tail). Keep variant A
  (wgmma_wait<0>), 168 regs, 0 spill as in s4; T<128 and T>512 run master byte for byte. Hard
  rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no
  allocation or branching on input values, no prints, no try/except and no per-call mutable state;
  nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16
  where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states
  (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA
  maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "first judged wgmma rows: prepare cold-L2 pair 580.8 vs Marlin 921.6 us (0.63x, all-local routing); incumbent judged Marlin pair 306 us, floor 101.3 us; M2 = wgmma pair row <= 0.9x the Marlin pair row in the same profile"
  - shapes: bs=256 *
    direction: faster
    why: prepare cold-L2 699.7 vs 1278.0 us (0.55x); incumbent judged Marlin pair 451-483 us, floor 108.9 us
  - shapes: bs=1 *
    direction: flat
    why: "control: T<128 runs master unchanged"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512
review: proceed
status: queued
review_at: 2026-10-25T054930Z
review_companions: null
---

# cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced

## Change

Stage 5 of the wgmma register-A grouped NVFP4 rewrite, built on the s4 commit (cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe, job 4e7f8be895a145ba) rebased onto master 9080a71 (w2 atomic add at T<=32 stays). FINDING that motivates it: s4 logged PATH CHOSEN = WGMMA for 128<=T<=512 and passed its own two-state replay check, yet the judge's profile at bs=128 and bs=256 shows only `Marlin<...>` (2 calls, 306.2 / 483.2 us) plus act_and_mul: the wgmma kernels never ran in the judged call. The likely cause is s4's kernel_fn gate `_wg_state_live(state)`, which compares a stored Python tuple state['wg_mptr_py'] with the live state['wg_xs'/'wg_hs'].data_ptr(): any state the judge makes by copying the prepared dict (new tensors, copied tuple and copied CPU map blob) fails it and silently runs Marlin. s4's self-check built state B with a second setup, never by copying, so it could not see this. CHANGE: (1) delete state['wg_maps'], state['wg_mptr'], state['wg_mptr_py'] and _wg_state_live; wg_gemm (C++ host side) encodes the two cuTensorMapEncodeTiled descriptors it needs on every call from the pointer of the bsrc tensor it is already passed (state['wg_xs'] for w13, state['wg_hs'] for w2) into a local CUtensorMap passed by value as a __grid_constant__ kernel parameter. This is host-only work at capture time (no device sync, no allocation, nothing persisted), so a graph captured on any state, original or copied, points at that state's own scratch. (2) kernel_fn gate becomes shape + prepare flag only: state.get('wgmma_ok') and 128<=T<=512 and expert_map is not None. (3) prepare self-check adds state B = copy.deepcopy(state) and state C = {k: v.clone() if tensor else v} (build each inside prepare's own try/except; skip a copy method that raises and print which), capture one graph per state, then alternate replays A,B,C,A,B with refilled inputs at T=128/256/257 over random, skewed-4hot, all-one-expert and mixed-local-remote routings; every replay must match an fp32-dequant reference within master's own error x1.5 (print both). (4) NO timing gate in this stage: enable whenever correctness passes, because this stage exists to get the first judged per-shape wgmma kernel rows (five stages so far, zero judged wgmma rows). (5) Prove the launched kernel: in prepare, run one torch.profiler pass over a graph replay of each state at T=128 and assert the wgmma kernel names appear and `Marlin` does not; print 'LAUNCHED = wgmma' or turn the path off with the reason. (6) Print the prepare cold-L2 graph-replay timings at a judge-like routing (see notes) beside Marlin's, compactly and late in the log (the log is a ~15 KB tail). Keep variant A (wgmma_wait<0>), 168 regs, 0 spill as in s4; T<128 and T>512 run master byte for byte. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: first judged wgmma rows: prepare cold-L2 pair 580.8 vs Marlin 921.6 us (0.63x, all-local routing); incumbent judged Marlin pair 306 us, floor 101.3 us; M2 = wgmma pair row <= 0.9x the Marlin pair row in the same profile
- `bs=256 *` faster: prepare cold-L2 699.7 vs 1278.0 us (0.55x); incumbent judged Marlin pair 451-483 us, floor 108.9 us
- `bs=1 *` flat: control: T<128 runs master unchanged
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Judged on the profile rows at bs=128/256 (wgmma gather + w13+silu + w2 vs Marlin pair + act_and_mul), not the headline; a regression on the verify rows is an acceptable outcome of a check because it is the measurement. Judge-like routing for the prepare print: s4 timed all-local routing (1024 local routes at M=128) where Marlin took 921 us, but the judge's Marlin pair at bs=128 is ~306 us, so s4's 0.63x forecast is from the wrong regime. Use a skewed Zipf-like popularity over 256 experts that makes this GPU's 32 the busiest, and calibrate it so the prepare Marlin pair lands near 0.8x the judged rows (LESSONS: prepare undercounts 20-25%): ~245 us at T=128, ~370 us at T=256; print local route counts. Run as okc check only. Risk: if the per-call descriptor encode is not what makes the judge's copy run wgmma, the profile still shows Marlin; the LAUNCHED assertion covers prepare states only, so read the judged profile kernel names first.

## Milestones

Rewrite stage 5 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054930Z by an adversarial subagent (model opus).

**Verdict: proceed.** The diagnosis holds up against the s4 archive. The judge prepared once (setup record copies=2 with prepare_s 39.34 s, the same as the log's single 'prepare so far 39.3s'), so its second state is a copy and not a second prepare(). Any tensor-cloning copy leaves the stored wg_mptr_py tuple pointing at the old scratch, which makes _wg_state_live false, and the judged bs=128/256 profiles show only Marlin. Encoding the TMA maps per call from the state's own bsrc pointer is host-only work done at capture time, is legal and is graph-safe. Proceed with the fixes below: the LAUNCHED check, a recursive state C, and the M2 comparator.

### Justification
A: Evidence for the copy claim comes from the s4 archive (muse_moe_w4a16/hypotheses/archive/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe/). (1) The okc.json setup record is {copies: 2, baseline_copies: 4, prepare_pad: 2223104, prepare_s: 39.344}. The log's last prepare line reads 'setup 35.7s (prepare so far 39.3s)', so prepare_s is ONE prepare and the second state did not come from prepare(). TASK.md, AGENTS.md and rules/ never define 'copies'. This is an inference from the record, not a documented fact. (2) s4 kernel_fn (branch hyp/.../s4-replay-safe, submission.py:2680-2681) gates on _wg_state_live (:1363-1367). That gate compares the Python tuple state['wg_mptr_py'] with the live wg_xs/wg_hs data_ptr(). A copy that clones tensors keeps the tuple, so the gate fails silently and runs Marlin. That fits profiles 4 and 5 (Marlin x2 at 306.2 / 483.2 us, act_and_mul present, no wg_kernel) and a headline of 1.343894 against master's 1.343895. If the judge had re-run prepare, the pointers would match and wgmma would have run. (3) The judge's copy cannot be copy.deepcopy: state['wg_ext'] is a Python module, deepcopy raises on a module, and the s4 setup passed. The copy is therefore a structure walk that clones tensors and shares everything else, likely with a pad allocation between copies (prepare_pad). The s3 failure fits this model too. Its global g_maps described state A's scratch, and copy B's TMA read A's stale rows. Per-call encode: cuTensorMapEncodeTiled is a pure host driver call. It touches no device, does not sync and does not allocate. The descriptor depends only on bsrc.data_ptr() and its shape, which are state scratch rather than input data, so nothing is cached from inputs and the path does not branch on values. Passed by value as a __grid_constant__ parameter, it is frozen into the graph node at capture. Each copy's graph therefore carries its own copy's scratch, which is exactly the property needed. Host cost is about 1 us per encode, paid only at capture or eager time and never on replay. Weakest link: performance in the judge's routing regime. The s4 forecast (580.8 vs 921.6 us cold at M=128) used all-local routing (1024 local routes). The judge's Marlin pair is 306 us at bs=128, which implies roughly a third as many routes. Nothing yet shows how the wgmma grouped kernel scales when routes are sparse and the per-expert N buckets are small. At 580 us for 1024 routes it sits at about 5.7x the 101.3 us floor, so it is not weight-bound, and its time may not fall in proportion to the route count. Milestone 2 is genuinely uncertain, and that is what this stage measures. A second weak link is item (5), the torch.profiler LAUNCHED assertion; see rules_flags.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, the base): T<=32 only; it does not overlap the 128<=T<=512 wgmma path. Control shapes bs=1/8/16/32 should stay flat.
- marlin-block-m-two-smaller-on-postnorm: tunes the incumbent Marlin block_size_m at T>=128, which is the comparator being replaced. It does not confound the wgmma rows, but the forced path removes the Marlin rows from bs=128/256 in this run, so the incumbent figure must come from another job (s4 4e7f8be895a145ba: 306.2/483.2 us, or master 7a0d900a1dd34666).
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: the wgmma path runs its own align at block 16, so align time belongs to the replaced subsystem. Compare align + gather + 2x wg_kernel + norm against align + 2x Marlin + act_and_mul + norm per kernel, not on the headline.
- post-expert-norm-triton-small-m / sglang-marlin-path-parity: shared post-norm/combine; at T>=128 both paths call SGLang's fused_post_expert_norm_weighted_sum, so it cancels.

### Interference and A/B needs
No companion. This is a rewrite stage judged on its own kernel rows, and forcing the path means the judged profile names the launched kernels directly. The one thing a companion could separate (did the copy fix work, or did something else block wgmma) is read from the bs=128/256 kernel names in this run's own profile. The sibling s5-copy-safe-bucket-gate is identical code plus a selector, and its 'wgmma rows appear' depends on exactly the same fix. Moe has 5 checks left and 5 queued, so run forced first and hold the bucket-gate and s6 until this job's profile shows wg_kernel rows. If it shows Marlin again, the copy hypothesis is wrong, and the bucket-gate would burn a check on the same failure.

### Open questions
- Milestone M2 says 'wgmma pair row <= 0.9x the Marlin pair row in the same profile', but forcing wgmma at 128<=T<=512 leaves no Marlin row at bs=128/256 in this profile. Restate it against s4 job 4e7f8be895a145ba (306.2 / 483.2 us) and master 7a0d900a1dd34666, allowing for cross-run noise.
- What does the judge's copy actually do? Is it a recursive clone with a pad between copies, and does it clone CPU tensors (wg_maps blob, wg_mptr)? The per-call encode works under every plausible variant, but a log line in kernel_fn is impossible. The judged profile kernel names are the only confirmation.
- Does the wgmma grouped kernel's time scale down with route count at judge-like skew (around 300-400 local routes at bs=128)? The calibrated prepare print in item (6) should give the answer before the profile does.

