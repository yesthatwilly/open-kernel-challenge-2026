---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
created: 2026-10-25T054545Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4
  base rebased onto master 9080a71, same per-call cuTensorMapEncodeTiled from the bsrc pointer
  inside wg_gemm, same removal of wg_maps/wg_mptr/wg_mptr_py and _wg_state_live, same deepcopy +
  tensor-clone three-state refill-replay self-check, same profiler LAUNCHED assertion), plus a
  per-T-bucket selector so the path can only ship where it wins. Buckets: B128 = 128<=T<=255
  (verify_16, skewed-128 test) and B256 = 256<=T<=512 (verify_32, bs=257 test). For each bucket,
  prepare times master's full Marlin path against the full wgmma path (gather + w13+silu + w2 +
  the shared align/combine) at T=128 and T=256 with the LESSONS-173 selector: both captured as
  CUDA graphs, a >=256 MB L2 flush before every replay, 41 interleaved A/B reps, an empty-graph
  baseline subtracted, under the judge-like skewed routing calibrated so prepare's Marlin pair is
  ~0.8x the judged rows (~245 us at T=128, ~370 us at T=256; print local route counts) plus one
  random routing. A bucket is enabled only if wgmma's median wins by >= 1.02x on BOTH routings;
  print both medians and the ratio, and assert the enabled side is the faster printed one.
  kernel_fn branches on T against two prepare-time booleans state['wg_on_128'] and
  state['wg_on_256'] (shape + flag only). Hard rules: CUDA-graph capturable; kernel_fn does no
  host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no
  try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4
  (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child
  processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static
  global may hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=128 *
    direction: faster
    why: only if B128 selector enables wgmma (>=1.02x on judge-like routing); otherwise flat (master)
  - shapes: bs=256 *
    direction: faster
    why: only if B256 selector enables wgmma; s4 prepare all-local ratio 0.55x is the most favourable bucket
  - shapes: bs=1 *
    direction: flat
    why: "control: T<128 is master"
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 5
replaces: Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512
review: proceed
status: done
review_at: 2026-10-25T054944Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T110823Z
base_sha: 928151c3ea4e903fc4a05ae7ac59672291b71d0d
base_blob: a13d5fea2e79f85df573670cbd55a8a10fbc3d2c
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T111451Z
impl_commit: fc959d9c0db31531d6ed750a69bb33b541435805
submission_sha256: df4015221c841885cfd0bb09c73db61dacbcf319ab400e161e5626ba8b5aca69
norm_sha256: cfff66e4073b3a4672d1dd38d33c134d33a4a1717a3300d97304097ceac0af62
submission_blob: 1774e42c833541afe7204a9d9cd3d6bb265be5b1
lint_warnings: 66
attempt_file: 2026-10-25T111451Z-check-cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.md
prepared_at: 2026-10-25T111451Z
job: 7be125e9e1ea42e3
dispatched_at: 2026-10-25T111459Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3743691121494652
speedup_ci: [1.3537535754672232, 1.394984648831707]
harvested_at: 2026-10-25T115711Z
best_at_harvest: 1.3438953504561806
promotion: submitted
hand_landed: true
promotion_reason: master moved since base; merge by hand
promotion_job: 4e4a0e26c85944f2
promotion_at: 2026-10-25T145823Z
---

# cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate

## Change

The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4 base rebased onto master 9080a71, same per-call cuTensorMapEncodeTiled from the bsrc pointer inside wg_gemm, same removal of wg_maps/wg_mptr/wg_mptr_py and _wg_state_live, same deepcopy + tensor-clone three-state refill-replay self-check, same profiler LAUNCHED assertion), plus a per-T-bucket selector so the path can only ship where it wins. Buckets: B128 = 128<=T<=255 (verify_16, skewed-128 test) and B256 = 256<=T<=512 (verify_32, bs=257 test). For each bucket, prepare times master's full Marlin path against the full wgmma path (gather + w13+silu + w2 + the shared align/combine) at T=128 and T=256 with the LESSONS-173 selector: both captured as CUDA graphs, a >=256 MB L2 flush before every replay, 41 interleaved A/B reps, an empty-graph baseline subtracted, under the judge-like skewed routing calibrated so prepare's Marlin pair is ~0.8x the judged rows (~245 us at T=128, ~370 us at T=256; print local route counts) plus one random routing. A bucket is enabled only if wgmma's median wins by >= 1.02x on BOTH routings; print both medians and the ratio, and assert the enabled side is the faster printed one. kernel_fn branches on T against two prepare-time booleans state['wg_on_128'] and state['wg_on_256'] (shape + flag only). Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=128 *` faster: only if B128 selector enables wgmma (>=1.02x on judge-like routing); otherwise flat (master)
- `bs=256 *` faster: only if B256 selector enables wgmma; s4 prepare all-local ratio 0.55x is the most favourable bucket
- `bs=1 *` flat: control: T<128 is master
- `bs=32 *` flat: control

## Risk

Dispatch in parallel with cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced (same round): the forced twin gives the per-shape kernel rows whatever the selector decides, this one is the land candidate. If both buckets come out enabled, the two runs are a replication of the same judged path, which is still useful given +-20% Marlin profile noise. Risk: the prepare routing still mis-forecasts the judge (LESSONS bf16 cuBLASLt pick); the 1.02x margin and two routings limit that.

## Milestones

Rewrite stage 5 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054944Z by an adversarial subagent (model opus).

**Verdict: proceed.** The copy claim checks out against the judge's own setup record and the s4 result, and encoding the TMA maps on the host per call is legal and graph-safe. Proceed as the land candidate, but dispatch it after the forced twin is harvested, not alongside it. Its only extra information over the forced twin is the selector's decision, and with 5 checks left that decision is worth a check only once the forced run shows wgmma rows in the judged profile.

### Justification
COPIES CLAIM, verified from evidence because TASK.md, AGENTS.md and rules/ never mention copies (TASK.md only says the judge runs 'steps of its own'). The source is the judge's setup record in okc.json, e.g. s4 archive/.../4e7f8be895a145ba.okc.json.gz: {event: setup, copies: 2, baseline_copies: 4, prepare_pad, prepare_s: 39.34}. Three facts follow. (1) prepare_s 39.3 s matches the log's single-prepare total ('prepare so far 39.3s'), so the second copy is most likely made by copying the prepared state, not by a second prepare(). (2) The copy count differs per side (SGLang 4, submission 2) and sits beside prepare_pad, so it looks sized from the state's memory footprint, i.e. tensors are duplicated. (3) s4 logged PATH CHOSEN = WGMMA and passed its own two-prepare replay check, yet the judged run matched master to 1e-6 (1.343894 vs 1.343895), and the profile shows only Marlin at bs=128/256 (306.2 / 483.2 us, 2 calls). That fits the stored-pointer guard _wg_state_live failing on cloned tensors. It also fits a guard that is wrong even on the original, but s5 deletes the guard, so the fix covers both. Caveat: s4's state held wg_ext (a module object), and plain copy.deepcopy cannot pickle modules. So the judge's copier is not plain deepcopy. It is more likely a container tree-map that clones tensors and passes other objects through. The self-check's state C, not state B, is the judge-like one. PER-CALL ENCODE: legal and graph-safe. cuTensorMapEncodeTiled is a pure host encoder with no stream work, no sync and no allocation. Under torch.cuda.graph it runs once at capture, and the CUtensorMap passed by value as a __grid_constant__ parameter is frozen into the kernel node. Replays never re-run host code, so the encode costs nothing in timed replays and binds each graph to the scratch of the state it was captured on. The maps derive from state scratch, not from data, so nothing derived from inputs is cached. It is ordinary launch marshalling inside kernel_fn, like Marlin's own host argument setup. MECHANISM/SHAPES: wgmma has never had a judged row; s3 died on recheck and s4 never launched. The prepare forecasts (0.63x/0.55x) come from an all-local regime where Marlin takes 921/1278 us, against judged Marlin pairs of 306/483 us, so the expected wins are unproven. The incumbent pairs are about 3x and 4.4x the floors (101.3/108.9 us), so there is room. Bucket mapping matches TASK.md: verify_16 = bs128 and verify_32 = bs256 are the only timed shapes in 128..512, and the bs=257 and skewed-128 tests exercise the gate untimed. WEAKEST LINK: the selector decides from a calibrated synthetic routing in prepare. Prepare has already mis-forecast the judge (prepare undercounts 20-25%, and the earlier wrong-regime forecasts), while the 1.02x margin is much smaller than the +-20% Marlin profile noise. If the true per-bucket gain is under about 1.2x, the gate can come out either way. A disabled bucket then gives a master-identical row and no wgmma data, which is why the forced twin must carry the measurement.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71, the base): changes only the T<=32 w2 call; disjoint from 128<=T<=512, so bs=1/8/16/32 are clean controls.
- marlin-block-m-two-smaller-on-postnorm: tunes Marlin's block_size_m at T>=128, i.e. the incumbent the selector races. Racing the tuned incumbent is correct, not a confounder, but the selector must call master's own _moe_marlin path with master's block_size_m (16 at T=128, 32 at T=256), not a re-derived config.
- triton-map-align-on-block-m-restack / align-parallel-per-expert-verify: the wgmma path runs its own align at block 16, while master runs the parallel per-expert align at T>128. The spec calls align/combine 'shared', but it is not, so each side of the selector must be timed as its full kernel_fn graph (align included), or the A/B misattributes about 5-10 us per side. Same for reading the judged profile: sum align+gather+2x wg+norm vs align+2x Marlin+act_and_mul+norm.
- post-expert-norm-triton-small-m: T<32 only; disjoint.

### Interference and A/B needs
No companions. The forced twin is this run's control: identical code, selector off, so it gives the judged wgmma rows whatever the gate decides. An ab-revert of any landed hypothesis would not change whether the copy fix makes wgmma launch, and none of them shares the 128..512 launch sequence except the incumbent being replaced. The real issue is sequencing under a budget of 5 checks with 5 queued. If both twins run in parallel and the forced run shows no wgmma rows (copy fix ineffective), both checks are spent on one fact. If the forced run shows wgmma slower in both buckets, this run should reproduce master and teaches nothing beyond the selector's agreement. Recommend dispatching this run only after the forced run's profile shows wgmma kernel names at bs=128/256, and only if at least one bucket is at or below the Marlin pair. If the forced run wins clearly in both buckets, this run is still the safer ship candidate, so run it then.

### Open questions
- Does the judge copy by tensor tree-map (clone tensors, share everything else) or by some other route? The s4 module-in-state plus pass suggests a tree-map. Print in prepare the id() of state['wg_ext'] and the data_ptr of wg_xs, and in the judge log nothing more (no prints in kernel_fn), so a future failure can be triaged from the self-check lines alone.
- Does the B128 gate's T=128 calibration represent the judged verify_16 routing well enough? Print the Marlin pair at the calibrated routing beside the judged 306.2 us (bs128) and 483.2 us (bs256) so the forecast error is readable at harvest.
- If the forced twin shows wgmma rows at or below Marlin in only one bucket, does this gate enable exactly that bucket? That agreement is the only new fact this run can add, so print the per-bucket decision late in the log tail.
- Prepare budget: s4 was 39 s. Three-state captures, a profiler pass and 2 buckets x 2 routings x 41 cold-flushed graph A/B reps should add well under 60 s, but print per-phase timings.


## Implementation notes

Base: the worktree was on s4 (928151c, stale). master's muse_moe_w4a16/submission.py is unchanged since 9080a71 (git diff 9080a71 master empty), which is the base of s5-forced, so no 3-way merge was needed. submission.py = the s5-forced submission verbatim (from branch hyp/.../s5-copy-safe-forced, same job-proven code: per-call cuTensorMapEncodeTiled, no wg_maps/_wg_state_live, three-state copy check, device launch marker, LAUNCHED line), plus only these changes (221 changed lines vs forced):
- Removed the forced "no timing gate" and print-only `_wg_time_vs_marlin`. Added `_wg_calibrate` (alpha pick so Marlin pair ~0.8x judged: 245 us T=128, 370 us T=256; same code as before), `_wg_graph_of`, `_wg_ab_graphs` (3 untimed warm-up replays, 41 interleaved reps with alternating order, 256 MB flush.zero_() before EVERY replay, events per replay, medians) and `_wg_select`.
- `_wg_select` runs after the copy check and only if the path is otherwise ok. Per bucket (T=128 -> B128, T=256 -> B256) it captures TWO full-path graphs on static inputs: `_moe_marlin` (master's align + block_size_m + w13 + silu + w2 + norm, w2_mode master) and `_moe_wgmma` (align + gather + w13+silu + w2 + norm), plus an empty-ish baseline graph (dummy add_); baseline median is subtracted from both. Routings: the calibrated Zipf routing and one random routing; local route counts, Marlin bsm, both medians, ratio and the calibrated Marlin pair vs judged 306.2/483.2 us are printed per routing and per bucket. Enabled only if Marlin/wgmma >= 1.02 on BOTH routings; also required: wg_err word 0, launch marker == WG_MAGIC, finite output and end-to-end rel-L2 vs Marlin <= 2e-2 on the same graphs. Asserts that an enabled bucket has wgmma faster on every printed routing. Any exception -> both buckets Marlin.
- state['wg_on_128'], state['wg_on_256'] (False by default, set at the end of `_wg_setup` from the selector). kernel_fn branches on shape and these flags only: 128<=T<=255 -> wg_on_128, 256<=T<=512 -> wg_on_256, else Marlin. `wgmma_ok` remains as the "path is correct" capability flag but kernel_fn no longer reads it. The copy check sets both flags True on its three states so it exercises wgmma at M=128, 256 and 257.
- Logging: `PATH CHOSEN B128 / B256 = WGMMA|MARLIN` lines and a combined PATH CHOSEN line are printed last; the diagnostic summary contains the selector lines; LAUNCHED = wgmma line retained from the copy check. Per-phase times (copy check, selector, setup) are printed. Review open question: id(state['wg_ext']) and wg_xs data_ptr are printed in prepare.
- Review items applied: (1) selector times each side as its full graph, align included, and calls master's own `_moe_marlin` so master's block_size_m is used (review "Confounders"); (2) Marlin pair printed beside judged 306.2/483.2 us; (3) per-bucket decision printed late in the log; (4) per-phase prepare timings printed. Not applied: the review's advice to dispatch only after the forced twin is harvested is a sequencing decision for the caller. Spec deviation: none besides using 256 MB flush and 3 warm-up replays.
- Expected from s5-forced judge data: B128 Marlin (wgmma 353 vs Marlin 310 us) and B256 wgmma (402 vs 457 us).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only a Python branch on shape + two prepare-time booleans; no sync/.item/allocation on values (checked in diff: selector code is prepare-only, called from `_wg_setup`).
- [x] Nothing derived from `data` cached across calls: flags come from synthetic prepare-time routings only.
- [x] Weights: unchanged from s5-forced (lossless retile in prepare, NVFP4 stays NVFP4, dequant in registers).
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention: n/a (MoE problem); every route attended (no skipping), unchanged.
- [x] No timing patches, harness/IO/secrets, module edits, reference import, subprocesses or network: no subprocess/fork/spawn added; selector only uses torch events/graphs.
- [x] All work inside the timed call; no output caching or replay. No module/static globals hold device pointers (selector tensors are locals; `_wg_audit_globals` unchanged and still run; flags are in state).
- [x] Two-state judge copies: flags are plain bools in state and survive a tree-map clone; copy check (A, deepcopy B, clone C) unchanged.

okc_lint: `OK with warnings` (0 errors; only host-sync/dtype warnings outside kernel_fn's call graph, prepare-only). undef.py: `submission.py []` (no undefined names). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading the code against the s5-forced branch and its judged log (archive d4ad4f31d2764050).

**Verdict: dispatch-with-fixes.** kernel_fn branches only on T and the two prepare-time bools. It has no try/except, prints, syncs or per-call state. No module global holds pointers, maps or buffers. The flags the copy check turns on are overwritten unconditionally after `_wg_setup`'s try/except, from `ok and sel`. A selector exception or a skip leaves both buckets on Marlin. The selector times the full `_moe_marlin` path (master's bsm, align included) against the full `_moe_wgmma` path as graphs on the same static buffers, interleaved, cold, with the baseline subtracted. Prepare took 44.6 s in s5-forced, so the budget has plenty of headroom.

Fixes (worktree submission.py; undef, py_compile and okc_lint OK, 0 errors):
- **The selector now gates on the judge-like routing only. The random routing is still printed and still checked for health (err word, marker, finite output, rel-L2), but it no longer gates.** Reason: the both-routings gate could essentially never pass B256. At equal T the uniform random routing has about half the local routes of the judge-like one (T=256: ~256 vs 464). In the s5-forced prepare, wgmma's time barely moves with route count (456 us at 241 routes, 466 us at 464) while Marlin's grows with it (376 to 495 us). So the random routing forecasts Marlin ~0.8x of wgmma at T=256, below the 1.02x margin, although the judge measured wgmma faster at bs=256 (402 vs 457 us). The run would have reproduced master, which is a category-1 gate that always fails. This departs from the spec's "BOTH routings"; revert it if the double gate was meant to be strict anyway.
- The tautological assert was removed. The faster-side assert now checks the gating routing.
- The LAUNCHED line now says it describes the copy check, where both flags are forced on. Each per-bucket PATH CHOSEN line now ends with "(selector)", "(selector skipped or raised)" or "(wgmma path off)", so a reader can tell which path ran in each bucket.

Residual risk: the prepare forecast says B128 = Marlin (wgmma 1.21x slower) and B256 = wgmma by ~1.06x eager. B256 is close to the 1.02x margin, so graph-timing noise could still disable it and give a master-identical run.

## Result

- job `7be125e9e1ea42e3` (check), status done, passed True, tests 5/5
- speedup 1.3744x, 95% CI [1.3538, 1.3950], score 9785.216999650855 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3438953504561806, "ci": [1.3237369201993379, 1.364053780713023], "job": "7a0d900a1dd34666"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 46.89 | 45.78 | 0.9988 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 145.58 | 144.81 | 1.0026 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 197.23 | 204.6 | 0.9997 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 113.34 | 104.53 | 1.0152 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 349.79 | 314.26 | 1.0602 | gained |
| bs=256 load=verify_32 sharding=ep8 | 60 | 428.84 | 404.85 | 1.0617 | gained |

gained on verify_16 verify_16; verify_32 verify_32. Expected hit: 4/4. Noise ±1.5%.

## Follow-ups

- rewrite stage 5 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_moe_w4a16 "<what to fix or tune>" --kind rewrite --depends-on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate --stage 6`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

