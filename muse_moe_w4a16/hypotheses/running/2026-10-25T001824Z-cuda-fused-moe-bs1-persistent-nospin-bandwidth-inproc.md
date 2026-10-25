---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
created: 2026-10-25T001824Z
base: master
area: whole-layer fusion
change: >
  Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the
  bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only,
  record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2
  start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per
  phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with
  a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back);
  (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave
  (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu
  reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1
  chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer
  written between replays (cold L2, as the judge does) and asserts the replay distribution is
  unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child
  process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on
  the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() ==
  (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny
  synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a
  print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with
  -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be
  an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work"
  - shapes: bs=8 *
    direction: flat
    why: "control: master"
  - shapes: bs=16 *
    direction: flat
    why: "control: master"
  - shapes: bs=32 *
    direction: flat
    why: "control: master"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-fused-moe-bs1-persistent-w13-silu-w2
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1
review: proceed
status: running
review_at: 2026-10-25T002132Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T002132Z
base_sha: 0efab7bdfa79cfea6a430d51c5b7c038a7bd2af3
base_blob: 62b00bb77cbc9e2fce8ed39dd25d87fb78503a5d
branch: hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
worktree: .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
mode: null
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003453Z
impl_commit: ea3464f424a209eb0f46541178cd3dd11062426f
submission_sha256: 8483d78ed40c60aa5c92c13671a1ce2654db503d1d85dd90b05105278a285344
norm_sha256: 89e64b5b2950de0e165973e23425229de43651c16ee304064b31962d847fe3a5
submission_blob: c80e3a831390ffb38cbcb7560231b59a64a2c6a0
lint_warnings: 33
---

# cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Change

Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only, record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2 start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back); (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer written between replays (cold L2, as the judge does) and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

supersedes cuda-fused-moe-bs1-persistent-nospin-bandwidth, held for a child-process self-check; Part A only (stage 2), Part B split to stage 3 per review

## Milestones

Rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002132Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a diagnostic rewrite stage. The re-mint fixes the blocking child-process issue and splits off Part B as asked. However, two of its three slow-mode fixes have little leverage on the code stage 1 actually shipped, so the per-CTA timestamp log is the real deliverable here, and the bounded-spin and unimodality-assert handling must not turn a slow mode into wrong outputs or a silent fallback.

### Justification
The target is sound. In stage 1 (job 1105a02aa3fa4e56), moe_bs1_kernel profiled at 37.7 us against 51.2 us for the incumbent's replaced kernels; FINDINGS now shows decode_1 at 48.8 us, 87.9% of it Marlin. The scored per-pair speedup was 1.568 against a speedup of medians of 1.289, with err 1.29 us. That fits a two-mode distribution on our side, so removing the slow mode should put decode_1 near 1.8-1.9x. That is about +3% on the headline, not the +16% claimed for stage 1. The floor arithmetic (~55 MB, 11.4 us) is right. The M2 target of ~42 us scored is plausible if the slow mode goes away. WEAKEST LINK: the proposed fixes assume a cause that was never measured, and the shipped stage-1 code weakens them further. (a) __nanosleep(32) cannot account for a ~24 us second mode, because the sleep granularity is tens of ns. Replacing it with a tight poll will change almost nothing. (b) The spec's 'second wave' premise comes from the stage-1 spec (64-row tiles, 128R items), but the patch shipped ROWS=128 tiles (64 gate + 64 up), so T1=64 and there are 64R phase-1 items. At R=2 that is 128 items on 132 CTAs, which is already one wave: 128 CTAs get one item and 4 get none. Item (3) is a no-op at decode_1 unless R>2. (c) Each phase-2 item waits on a 1024-column chunk fed by 16 phase-1 tiles held by 16 different CTAs, and items are ordered (r, s, n). Phase 2 is therefore close to a global barrier on phase 1, and reordering phase-2 items by finish time has little room to help. The most likely remaining causes are a late-starting CTA (an SM still occupied at launch, or a clock or power state) or L2/HBM contention patterns. Only the per-CTA entry globaltimer and %smid, recorded under judge-like cold-L2 replays, can tell these apart. So this stage's value is the diagnostic log plus M1. The milestone should be read as 'the cause is identified', not 'M2 reached'. Implementers must start from the archived patch (ROWS=128, STAGES=8, SPLIT2=4, SPIN_MAX=1<<22, sticky C_ERR), not from the stage-1 spec text.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (gain): _post_norm_combine_kernel is part of the bs=1 incumbent the fused kernel replaces, but it is bypassed at num_tokens == 1, so it only defines the incumbent's time and causes no runtime interaction.
- triton-map-align-on-block-m-restack (gain): supplies _align_small at bs=1, which is likewise bypassed by the fused path. It defines the incumbent and does not confound.
- align-parallel-per-expert-verify (master, neutral): only num_tokens > 32, so it is disjoint from bs=1. The rebase leaves the bs=1 base identical to stage 1's.
- marlin-block-m-two-smaller-on-postnorm (gain): only num_tokens >= 128, so it is disjoint.
- sglang-marlin-path-parity (gain): supplies the Marlin bank that the fallback path and the self-check's master reference use. It does not confound the fused kernel's timing.
- Not landed but running: stack-onewarp-hoisted-dispatchcombine(-from17) changes the bs<=8 align and the bs<32 combine. If it lands first, the bs=1 incumbent drops by ~1-2 us, so read this stage against master at dispatch.

### Interference and A/B needs
No companion is needed. Every landed hypothesis that touches bs=1 is bypassed when num_tokens == 1, so neither an ab-revert nor an ab-tune would change the reading. The internal attribution problem (whether the spin change or the reordering killed the slow mode) is better answered by the in-kernel per-phase timestamps and the spin-iteration counts the self-check already logs than by a control run. If the slow mode persists, the next stage follows from the log.

### Open questions
- What is R at the judge's decode_1 seed? At R=2 the shipped kernel already runs phase 1 in one wave (64R = 128 items <= 132 CTAs), so step (3) changes nothing. Should the one-wave sizing target R=3..8 instead, where it does matter?
- Does the self-check under cold-L2 graph replays reproduce the 41/59 bimodality at all? If not, the per-CTA log cannot explain the judge's slow mode, and the next stage would need the diagnostic to run in the timed path under a flag (which costs a check).
- Is the slow mode a late CTA entry (an occupied SM or the launch tail of the judge's L2-flush/read kernel) rather than a phase-2 wait? If entry skew is ~24 us, neither the spin nor the ordering fix helps, and the remedy is a grid < SM count (e.g. 128 CTAs) or a work-stealing atomic item counter instead of the static c, c+G assignment.
- Would a dynamic item queue (atomicAdd ticket over items) remove the dependency on a straggler CTA entirely? It is a better candidate for stage 3 than reordering.


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc`, file `muse_moe_w4a16/submission.py` only, uncommitted. Built on stage 1's shipped code (ROWS=128, STAGES=8, SPLIT2=4), as the review requires.

- **Rebase onto current master.** The worktree branched from stage 1's implementation (0efab7b), which predates master's `align-parallel-per-expert-verify` (1a5da3d). I ported that commit's submission.py diff with `git apply --3way`, which applied cleanly and was then unstaged. `git diff master -- submission.py` is now additions only, apart from the old `kernel_fn` def/docstring lines, so every M != 1 runs current master byte for byte.
- **(1) Per-CTA diagnostic log.** `moe_bs1_kernel` is now `template <bool DBG>`. The timed path launches `<false>`: `_fused_forward` passes an empty int64 `dbg_off` tensor, and the host picks the instantiation on `numel` (a shape branch, fixed at capture). Each `if (DBG ...)` is compiled out of the timed kernel. `<true>` writes 12 u64 per CTA: entry `%globaltimer`, `%smid`, setup done, the end of the last phase-1 item, the first phase-2 wait's start and end, total spin iterations, total wait `clock64` cycles, phase-1 and phase-2 item counts, exit time, and whether the CTA ran the combine.
- **(2) Tight poll.** `__nanosleep(32)` is removed from the phase-2 `ld.acquire` spin. The bound is SPIN_MAX = 1<<21 tight polls, about 1 s.
  - **Applies the caller/review fix:** the sticky flag no longer poisons later calls. A bound hit sets a new per-launch `C_ABORT`, which ends the remaining waits of that launch only and is reset by the exit ticket. `C_ERR` is now a sticky counter that only reports; the kernel never reads it. Stage 1's `ld_acquire(ctrl + C_ERR)` early-out, which made every later launch break its waits after one bound hit, is gone.
  - Nothing on the timed path reads a flag or switches paths. The prepare-time decision is made once, and loudly.
- **(3) One-wave phase 1 / phase-2 ordering: NO CODE CHANGE (the review wins).**
  - At decode_1's likely R=2 the shipped kernel has 64R = 128 phase-1 items on 132 CTAs, which is already one wave.
  - Static round-robin already hands the first phase-2 items, starting at item n1, to CTAs n1 % G and later. Those are exactly the CTAs with the fewest phase-1 items, which finish first.
  - The (r, s, n) phase-2 order is already monotone in the readiness of the chunk it waits on, because the phase-1 tile index rises with (r, s).
  - Contiguous blocks of ceil(64R/G) tiles per CTA would make every chunk depend on the last wave, which is worse.
  - The dynamic-ticket item queue the review suggests is left for stage 3.
- **Self-check (in-process, no child process):**
  - **Arch gate.** The build sets `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` and restores it after. `-arch=sm_90a` was dropped, so the only gencode is torch's, and the flags from `_get_cuda_arch_flags()` are printed. The gate requires all of the following, and anything else gives `PATH CHOSEN = master`:
    - capability == (9, 0);
    - gencode tokens limited to {compute_90a, sm_90a};
    - `cudaFuncGetAttributes` binaryVersion and ptxVersion == 90 for both instantiations;
    - a new `arch_probe_kernel` in the same module reporting `__CUDA_ARCH_FEAT_SM90_ALL` and `__CUDA_ARCH__` 900.
  - **Deviation:** if the private `_get_cuda_arch_flags` helper is missing, the probe kernel is the evidence (the line logs "unknown").
  - **Build flags:** `-O3 -lineinfo`; the extension is renamed `..._v2`.
  - **Canaries.** The first launches are an R=1 synthetic canary of `<false>`, then of `<true>`. Each is printed before, synchronized, printed after, and compared with master.
  - **Stage-1 checks kept.** The 50 eager launches, the 2 fp32-dequant checks and the 10 graph replays are unchanged, with a synchronize after every launch. Round 0 prints around each launch; later rounds print one line per round, to protect the 16 kB log tail.
  - **Failure handling.** A correctness failure or a bound hit gives a loud fallback in prepare.
- **Diagnostic (`_fused_diag`, REPORT ONLY, after PASSED).** Each replay is preceded by a `fill_` of a buffer of 2 x L2 + 16 MB (cold L2).
  - **Main timing.** 100 interleaved replays of the timed fused graph and of master's bs=1 graph (control) at R=2. Each prints p10/p50/p90/min/max, p90/p10 and a mode split (the widest step > 8% between consecutive sorted samples).
  - **UNIMODAL line.** `UNIMODAL yes/NO` against p90/p10 < 1.3. **It never changes the path** (applies the review/caller fix).
  - **Other R.** 50 replays each at R = 1, 3 and 8.
  - **Logged run.** 100 replays of the DBG graph with the per-CTA log copied out after each replay. Replays are split by their own mode. For each mode it prints medians and maxima of: entry skew, the late-entry SMs, setup, the last phase-1 end, the first-wait maximum and median, spins, kernel span, and replay time minus span (time outside the kernel). For each mode's median replay it prints per-CTA min/med/max of entry, setup, phase-1 end, wait start and end, exit, spins, wait cycles and item counts, plus which SM ran the combine. It also prints a globaltimer resolution bound.
  - **Exceptions.** An exception inside the diagnostic is printed and the path is kept.
  - **Final decision.** A spin-bound hit during the diagnostic (a stall of about 1 s, a hang, never a slow mode) still gives a loud prepare-time `PATH CHOSEN = master`. The last line is `PATH CHOSEN = fused for num_tokens == 1 ...`.
- **Reading the log.** Large entry skew or a large "time - span" in the slow mode means a late CTA or launch, not the kernel's dataflow; the stage-3 fix would then be grid < SMs or a work-stealing ticket. A large phase-2 first-wait with a late phase-1 end means a phase-1 straggler. If the self-check replays are unimodal, the judge's slow mode was not reproduced, which is the review's open question 2.
- **Unverifiable here.** There is no nvcc or GPU on this node, so CUDA compile and run risks remain:
  - the function-pointer ternary launch of the template;
  - `reinterpret_cast<const void*>` of the kernel templates;
  - torch accepting "9.0a" in `TORCH_CUDA_ARCH_LIST`;
  - the ptxVersion that `cudaFuncGetAttributes` reports for an image with no PTX embedded (if it is not 90, the arch gate turns the path OFF and the run measures master).

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc diff`:

- [x] CUDA-graph capturable: `kernel_fn` and `_fused_forward` have no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The only branches are on shape, config, and the prepare-time `state["fused"]`. The DBG/non-DBG choice is on the `numel` of a prepare-time tensor. All syncs and `.tolist()` calls are in `prepare()`/`_fused_setup`/`_fused_diag`, and lint confirms they are outside kernel_fn's call graph.
- [x] Nothing derived from `data` is cached across calls. The scratch buffers and counters are prepare-time, and the kernel resets every counter but the report-only C_ERR at exit. C_ERR is never read on the timed path.
- [x] Weights: unchanged from stage 1. The repack is a lossless NVFP4 fragment shuffle in `prepare()` with a round-trip assert, and global scales are multiplied by 2^7 exactly.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (h and y are bf16, accumulation fp32). n/a for KV cache and queries: this is MoE.
- [x] n/a: no attention in this problem.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import or network. **No subprocess, multiprocessing, os.fork or spawn** (grep is empty). The `os` import only sets and restores `TORCH_CUDA_ARCH_LIST` around `load_inline` in prepare.
- [x] All work happens inside the timed call, with no output caching or replay. The diagnostic graphs and replays run only in `prepare()` on synthetic inputs, and their results are only printed.
- [x] The first measurement must be an `okc check`, never a scored submit (per spec and LESSONS 2026-10-25).

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`): 33 × `warn host-sync ... (outside kernel_fn's call graph; fine in prepare())`. These are the synchronize, `.tolist()` and randperm calls in `_align_self_check`, `_fused_case`, `_fp32_moe`, `_arch_gate`, `_fused_setup`, `_fused_diag` and `prepare`. Final line: `OK with warnings: submission.py`. `py_compile` passed, and `__pycache__` was removed.

## Pre-flight audit

Audited 2026-10-25 (opus). **Verdict: dispatch-with-fixes.**

- Confirmed: `git diff master` is additions only (plus the kernel_fn rename/docstring and `return {` to `state = {`), and `_master_kernel_fn` is master's body byte for byte, so bs != 1 runs master. kernel_fn passes the empty `dbg_off`, so the timed path always launches `<false>`, and every DBG write is a compile-time `if (DBG ...)`. Phase-1 items precede phase-2 items in each CTA's static sequence, grid = SM count, and occupancy 1 is asserted, so every CTA becomes resident once the previous same-stream kernel drains: no deadlock. A bound hit (after about 1 s) ends that launch's waits only, gives wrong output for that one call (the judge checks every timed call, so it is detectable, not silent), and does not reach the next call (C_ABORT and the counters are reset by the exit ticket). The diagnostic replays are about 450 cold-L2 replays of about 0.1 ms each, under 1 s. kernel_fn reads none of their results. There is no child process, nothing cached from `data`, the NVFP4 repack is lossless and round-trip asserted, h and y are bf16, and the path is graph-capturable.
- Fix 1, gate that may always fail: the arch gate no longer requires `ptxVersion == 90`, which is now logged only. For a PTX-less sm_90a cubin that value is not documented well enough to gate on, and a wrong guess would silently measure master. The image is still proven by capability (9, 0), compute_90a/sm_90a-only gencode tokens, binaryVersion 90, and the same-module probe (`__CUDA_ARCH_FEAT_SM90_ALL` and `__CUDA_ARCH__` 900).
- Fix 2, build robustness: the `want_dbg ? &k<true> : &k<false>` function-pointer triple-chevron launch is now two explicit `moe_bs1_kernel<true|false><<<...>>>` launches through a shared argument macro. `moe_bs1_info` now takes `auto*` pointers to the specializations before `reinterpret_cast<const void*>`, not a template-id cast. Behaviour is unchanged.
- Lint: OK with warnings (host syncs outside kernel_fn only). py_compile OK. __pycache__ removed.

Residual risk: never compiled here (nvcc, torch accepting `9.0a`), and a sticky device fault in the in-process canary cannot fall back, so the first measurement must be `okc check`.
