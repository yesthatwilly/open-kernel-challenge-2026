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
review: required
status: queued
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
