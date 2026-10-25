---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-tiny-m-bucket16
created: 2026-10-25T042150Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Tiny-M branch of the wgmma register-A grouped rewrite: bs=1 (decode_1) and the bs=4 test, which
  master runs at 44.9 us (Marlin pair ~42 us) vs a floor of 11.4 us - the largest ratio of any
  shape (3.9x). Not the dead bs1 fused persistent kernel: no grid-wide flags, no spins, no phase
  dependency - two ordinary launches (wg_kernel<w13+silu> then wg_kernel<w2>) using only the
  proven N16 bucket (1-7 tokens pad to 16 rows; MMA waste is irrelevant, the shape is
  weight-bandwidth bound). Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current
  master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched
  shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as
  stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo,
  wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: own bound constant WG_TINY_MAX_TOKENS = 7; every item is N16 (cap 16
  rows per item on device); align is master's _align_small; combine stays master's
  _post_norm_combine_kernel. For bandwidth at 1-2 experts, the w13 grid has 64 N-tiles x experts
  items, which under-fills 132 CTAs at one expert: add a K-split of 2 for w13 and 4 for w2 when
  the item count < 132 (fp32 partials in a T-sized workspace reduced in the consumer epilogue by
  the last-arriving CTA via a per-item ticket that the last CTA resets to 0, so replays start
  clean; no spin waits). Hard gate in prepare(): after the correctness self-check passes, time the
  new path against master's path on the same synthetic routings (skewed and random, cold L2 via a
  >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T
  bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice
  equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T
  bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert,
  all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant
  reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not
  against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is
  unmeasured. First measurement is an okc check, never a submit. T buckets: 1 and 4. Because the
  judge's timed loop has shown bimodal bs=1 behaviour unseen in prepare (LESSONS 2026-10-25), also
  report the judge's runs: read the decode_1 median vs raw median in the result.
expected:
  - shapes: bs=1 *
    direction: faster
    why: master 44.9 us (Marlin pair ~42.1 + combine 3.2 + act 1.6 + align 1.6), floor 11.4; M2 wgmma pair < 42 us; M3 <= 25 us
  - shapes: bs=8 *
    direction: flat
    why: "control: above the 7-token gate"
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 1 <= num_tokens <= 7
review: proceed
status: running
review_at: 2026-10-25T042648Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T042648Z
base_sha: 570f5e4ac5465931183addaf7a472e01668ced15
base_blob: 1ae7e9cac6dc078a0e71c7a6b0b77ec48b0ed01f
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-tiny-m-bucket16
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-tiny-m-bucket16
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T044530Z
impl_commit: 81f149af81434325318683110c27e78d72eff58c
submission_sha256: e3cc16bfd7dd263822fedff42f57c2b5bbcbf5690a5f1b89f1939d9d71e37473
norm_sha256: 2b1a0c6181cc5112ef61e582be067f8bfaeabd5a8f3646a7c40d7d278c216a98
submission_blob: 12ff8a5ce35b01937a68af59254faf69382f692a
lint_warnings: 69
attempt_file: 2026-10-25T044530Z-check-cuda-nvfp4-wgmma-rega-tiny-m-bucket16.md
prepared_at: 2026-10-25T044530Z
job: 3df5c69750884600
dispatched_at: 2026-10-25T044538Z
---

# cuda-nvfp4-wgmma-rega-tiny-m-bucket16

## Change

Tiny-M branch of the wgmma register-A grouped rewrite: bs=1 (decode_1) and the bs=4 test, which master runs at 44.9 us (Marlin pair ~42 us) vs a floor of 11.4 us - the largest ratio of any shape (3.9x). Not the dead bs1 fused persistent kernel: no grid-wide flags, no spins, no phase dependency - two ordinary launches (wg_kernel<w13+silu> then wg_kernel<w2>) using only the proven N16 bucket (1-7 tokens pad to 16 rows; MMA waste is irrelevant, the shape is weight-bandwidth bound). Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: own bound constant WG_TINY_MAX_TOKENS = 7; every item is N16 (cap 16 rows per item on device); align is master's _align_small; combine stays master's _post_norm_combine_kernel. For bandwidth at 1-2 experts, the w13 grid has 64 N-tiles x experts items, which under-fills 132 CTAs at one expert: add a K-split of 2 for w13 and 4 for w2 when the item count < 132 (fp32 partials in a T-sized workspace reduced in the consumer epilogue by the last-arriving CTA via a per-item ticket that the last CTA resets to 0, so replays start clean; no spin waits). Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets: 1 and 4. Because the judge's timed loop has shown bimodal bs=1 behaviour unseen in prepare (LESSONS 2026-10-25), also report the judge's runs: read the decode_1 median vs raw median in the result.

## Expected

- `bs=1 *` faster: master 44.9 us (Marlin pair ~42.1 + combine 3.2 + act 1.6 + align 1.6), floor 11.4; M2 wgmma pair < 42 us; M3 <= 25 us
- `bs=8 *` flat: control: above the 7-token gate
- `bs=16 *` flat: control
- `bs=32 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

Kill: prepare gate keeps Marlin at T=1, or the judged decode_1 is not below 44.9 us by > 1.5%, or decode_1 shows a slow mode in more than 10/100 replays. The ticket reduction is the only new device logic; if it is judged risky, drop it (K-split 1) and accept an under-filled grid for stage 3.

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 1 <= num_tokens <= 7`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042648Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-3 rewrite aimed at the largest floor gap (decode_1: 44.9 us vs an 11.4 us floor). It reuses only the N16 instantiation, which stage 2 proved correct, and it adds no grid-wide spin. The weakest link is bandwidth: the stage-2 pipeline keeps only about 37 KB of weight bytes in flight per SM (4 stages x 9216 B), and every earlier small-M kernel in this lineage reached at most ~1.5 TB/s. Proceed with no companions, but make the K-split decision on device, call _align_small at BSM=16, and prefer an N16-only instantiation with deeper stages.

### Justification
Mechanism. decode_1 is purely weight-bandwidth bound. With I=H=4096 an expert is ~28 MB of codes plus e4m3 scales. The prepare DIAG of the fused bs1 kernel puts R=2 local experts at bs=1, which is ~56 MB, matching the 11.4 us floor at 4.8 TB/s. At 1-7 tokens, swap-AB register-A wgmma with tokens padded to N16 wastes MMA work, and that does not matter here, so the design fits the shape. The item math is right: w13 has inter/64 = 64 tiles per expert and w2 has hidden/128 = 32. At R=2 that is 128 and 64 items on a 132-CTA persistent grid, so a K-split of 2 and 4 gives 256 items each. Removing Marlin at T<=7 also removes the remaining w13 Marlin fp32-reduce cross-CTA lock. The w2 atomic-add landing cut the +23 us slow mode from 44/100 to 10/100 replays, which strongly suggests the slow mode comes from inter-CTA serial reduction. That makes the ticket-based last-CTA reduction (no waits) the right choice. The M2 target (pair < 42.1 us) only needs ~1.35 TB/s, which is reachable. The M3 target (<= 25 us) needs ~2.3 TB/s. Expected directions agree with TASK.md: decode_1 faster, and every shape at bs>=8 runs master byte for byte (bs=4 is a test only). WEAKEST LINK: achieved bandwidth at T=1 has never been measured for this mainloop. Stage 2 timed only M=128/256, with warm L2 and eager launches. The stage-2 STAGE_BYTES reserves 2*128*128 B of token slab per stage, sized for N128, yet N16 uses only 4 KB of it, so 4 stages hold just ~37 KB of weight bytes in flight per SM. Little's law at ~0.8-1 us loaded latency needs ~30-36 KB/SM for 4.8 TB/s, so this design has no margin. The fused bs1 kernel had 8 stages and 158 KB of smem and still reached only ~1.46 TB/s. The swap-AB mma and Triton split-K GEMV small-M kernels lost to Marlin by 1.7-2x. Second weakest point: 'only the proven N16 bucket' was proven correct only at M>=128 with random routings. One-expert or all-local routing at T=1/4 and the new K-split epilogue are new code paths. Third: the change says 'two ordinary launches', but the stage-2 path is align + wg_gather + w13 + w2 + combine, which is 5 launches like master. Budget the gather (a few us cold) against M3.

### Confounders on the current kernel
- w2-marlin-atomic-add-small-m (9080a71): this landing set the bar (44.9 us, Marlin pair 42.1) and changed the shape of the slow mode (10/100 replays). The rewrite replaces both Marlin calls at T<=7, so a decode_1 gain combines bandwidth with removal of the residual w13-lock slow mode. That split is readable from the judge's raw p10/p50 vs the adjusted median and needs no extra run. The prepare gate must time master WITH atomic add (master's w2_mode as chosen by its own self-check), not stage 2's Marlin-with-fp32-reduce timing harness, or the 3% gate compares against a slower straw man.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port (_align_small): at bs=1, master's _align_small runs at block_size_m=8 (_block_size_m(1,32)=8). The wg kernel assumes 16-row blocks (n_padded/16, block_experts per 16 slots, chunk = 8 blocks), so _align_small must be called with BSM=WG_BSM=16, which is a new Triton specialization that must be warmed in prepare. Reusing kernel_fn's block_size_m silently mis-chunks. The stage-2 path used SGLang's moe_align_block_size(…,16,count) plus an at::native expert_map gather instead. A loss could come from the align choice (it is 1.6 us of the budget) rather than the GEMMs, so the profile row must be read per kernel.
- post-expert-norm-triton-small-m (_post_norm_combine_kernel): this is compatible. The kernel skips slots with local id -1 and never loads them, and stage 2's w2 writes route-indexed rows with 0 remote writes. The ids passed must be _align_small's local_ids (with -1 for remote), not the global ids. There is no timing confound because the 3.1 us combine is identical on both sides.
- sglang-marlin-path-parity / marlin-block-m-two-smaller / align-parallel-per-expert-verify: these have no interaction. They are the T>=8 fallback and control paths, which stay master byte for byte.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on the replaced kernels' own time (profile rows for wg_gather + wg_kernel<w13+silu> + wg_kernel<w2> vs Marlin pair + act_and_mul at bs=1). The prepare gate prints per-bucket interleaved cold-L2 timings of both paths, and bs>=8 serves as an in-run master control. An ab-revert of w2-marlin-atomic-add would only change the Marlin side, which the rewrite bypasses at T<=7. A gain or a loss is therefore already attributable from one run plus its PATH CHOSEN lines. One cheap in-run control is worth adding as instrumentation, not as a separate run: in the prepare DIAG, also time the N16 path with K-split forced to 1 (same binary, a device flag). If the split path wins, that separates under-filled-grid gains from mainloop bandwidth. If the ticket reduction is buggy, it also tells stage 4 whether K-split 1 alone beats Marlin.

### Open questions
- Will the N16 path use its own instantiation (template TINY / N16-only kernel)? The stage-2 kernel switches over all four buckets in one function, so it runs at 168 regs, and w2 already spills 8 B. Adding K-split epilogue code there raises spill risk and perturbs the N64/N128 code that s3 (job 3bdec0e58d684c17) is fixing in parallel. An N16-only kernel can also shrink the per-stage token slab from 32 KB to 4 KB and raise STAGES from 4 to ~12-16, i.e. 110-150 KB of weight bytes in flight per SM vs 37 KB. That is the single biggest bandwidth lever toward M3.
- How are the K-split counts chosen for 3+ local experts? 'Split only when items < 132' gives 192 items at R=3 (1.45 waves, about a third of the second wave idle). Pick the split that minimises ceil(items*s/132)/s on device instead.
- Can wg_gather be dropped at T<=7, by TMA-loading token rows straight from x through a per-item row table or by having _align_small write xs? At bs=1 it is a separate cold launch of a few us against a 25 us M3 target.
- Does the self-check FORCE one-expert and all-8-local routings at T=1 and T=4, in the N16 + K-split path and in graph replay, with fp32-dequant reference error printed beside master's? Stage 2 proved N16 only under random routing at M>=128.
- The prepare gate has been non-predictive of the judge at bs=1 twice (it showed unimodal while the judge was bimodal). The kill criteria should read the judge's raw decode_1 distribution (p10/p50 and slow-mode count) rather than trust PATH CHOSEN timings. The change already asks for this; make it a hard part of the harvest.
- Merge order with siblings wgmma-rega-midm-bucket-cap32 / w2-only-midm (8<=T<=32) and s3 (>=128). All three edit the same load_inline module. Whichever lands second must rebase onto the renamed module, and the T ranges must stay disjoint (7 | 8).


## Implementation notes

Rebase: the worktree was at s2 (570f5e4); master 9080a71 has NONE of stage 1/2 (no CUDA block, no `_moe_marlin`, no wgmma hooks), so `git merge-file master base s2` conflicted over the whole inserted CUDA block. Resolved by taking s2's side (docstring + the full CUDA/python block); master's `prepare`/`_forward`/`kernel_fn` hunks merged cleanly: `diff master.py new` has zero master-only lines, so master's w2 atomic-add path (`W2_ATOMIC_MAX_TOKENS`, `_w2_atomic_self_check`, `_forward(.., w2_mode)`) is intact. Stage 1/2's dispatch hooks did not come across (they were never on master); I did not re-add them: stage 2's 128..512 path is DORMANT (`WG_S2_ENABLED = False`, `_wg_setup` not called), so bs >= 8 is master byte for byte and prepare does not pay a second build. Its code stays in the file for the s3 merge.

Added (all in submission.py): a separate module `muse_moe_w4a16_wgmma_tiny_v1` (own `load_inline`, sm_90a only via TORCH_CUDA_ARCH_LIST=9.0a around the build, `-std=c++20 -O3 -lineinfo`, wgmma under `__CUDA_ARCH_FEAT_SM90_ALL` with error word 2 otherwise, printed arch flags) with N16-only `wgt_kernel<GATED>` + `wgt_gather_kernel`. It reuses only stage 2's PTX helper head, the N16 wgmma wrapper and the lossless `_wg_bank` retiling; the shared `wg_kernel` is untouched. Review fixes applied (they override the spec): bs<=7 only; 16 stages (13312 B, ~213 KB smem, 4 KB token slab); `wgmma_wait<0>` before the next dequant overwrites A plus register fences on A (register cap 168 under `__launch_bounds__(384,1)`; numRegs/spill printed); K split chosen on the GPU (`pick_split`, S in 1,2,4,8, integer cost, host mirror for logging), partials in an fp32 workspace, last-arriving CTA per (block, tile) group sums in K-slice order via a ticket (no spins); `wgt_gather` zeroes all tickets every call and the last CTA resets its own, and the host zeroes tickets on any error word; `_align_small` called with BSM=16 and warmed for T=1..7; combine = master's `_post_norm_combine_kernel`. LAUNCH COUNT: 5 (align, gather, w13+silu, w2, combine), not 2. Gate: per bucket (T=1; T=2-7 timed at 4 and 7) against master `_forward` with the chosen w2 mode (atomic add on), CUDA-graph replays, >=320 MB cold-L2 flush, interleaved A/B, 80 reps, p10/p50/p90 printed, skewed + local-random routings, enabled only if p50 <= 0.97 x master on every routing and the self-check passed; one PATH CHOSEN line per bucket, asserted equal to the enabled flag; ablation timing of K split forced 1 vs auto at T=1. Self-check: T in 1,2,4,7 x 6 routings (one-expert, skewed, all-local, random-local, uniform random, all-remote) x split auto/1/2/4/8 against an fp32 dequant reference with master's error printed beside it (bound max(2x master, 1e-2) aggregate / 4e-2 worst token), launch/done prints around each launch, graph capture of auto and split 4 with 3 replays, PLUS (coordinator finding from s3 job 3bdec0e58d684c17) one graph per T replayed with 6 different routings through refilled static buffers, each judged against the fp32 reference. Every routing-dependent quantity (n_padded, used blocks, split, tickets) is computed on the device per call; host-frozen values depend on T only.

Deviations: stage 2's dispatch is not carried (see Rebase); `-std=c++20` added to the flags (stage 2 omitted it). Nothing was compiled or run here (no GPU/nvcc): biggest risk is the first nvcc/ptxas build and TMA/descriptor behaviour of the new 16-stage N16 kernel, then the ~300 s prepare budget (build + retile + self-check + gate; phases are skipped past `WGT_SETUP_BUDGET_S`=230 s and the path then stays master). Expected cost per stage is bandwidth-near-compute (wait<0> serialises dequant and wgmma within a warpgroup); a double-buffered A is the follow-up if M3 is missed.

## Rules check

- [x] CUDA-graph capturable: kernel_fn / `_moe_tiny` have no sync/.item()/.cpu(); every `.item()`/`synchronize` is in prepare-only helpers (lint warns only there).
- [x] Nothing derived from `data` is cached across calls (scratch buffers are rewritten per call, tickets zeroed per call).
- [x] Weights: lossless NVFP4 retiling in prepare() only (`_wg_bank`, round-trip asserted); no precision change.
- [x] Activations BF16 where SGLang keeps BF16 (h and y stored bf16 as Marlin does; fp32 accumulate).
- [x] n/a: attention/KV items (MoE problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, child process, subprocess or network (load_inline only, in-process).
- [x] All work inside the timed call; no output caching or replay.
- [x] Shape-only host branching (T<=7 and prepare-time flags); bs>=8 is master unchanged.

okc_lint: "OK with warnings" (only host-sync warnings in prepare-only code); py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent, by reading the code (no GPU or nvcc on this node). **Verdict: dispatch-with-fixes.**

Fixes (all in the worktree's `muse_moe_w4a16/submission.py`):
1. **Restored `import os` and `import time`.** The rebase took master's import block, which has neither. prepare() now opens with `t_prepare = time.perf_counter()`, and `_wgt_build` and `_wgt_setup` use `os.environ` and `time`. As written, prepare() raised NameError on its first line and failed the whole run, at every shape. okc_lint did not catch it. An AST undefined-name scan now reports only `_moe_marlin`, which appears only in dormant stage-2 code that is never called.
2. **Per-state tensor maps, no module globals.** This is the same root cause the coordinator found in s3 (job 3bdec0e58d684c17). The tiny module kept its maps in C++ globals (`TMaps g_tmaps[2]`, `g_tmaps_ok`, `g_tmap_rows`), and `wgt_set_maps` re-pointed them at the calling state's xs/hs. The judge calls prepare() twice and rotates the two states, so the second prepare would have re-pointed the first state's captured graphs at the other state's scratch. Now:
   - `wgt_set_maps(xs, hs, maps)` encodes the maps into a per-state CPU uint8 tensor: 2 x TMaps, then the two row counts.
   - `wgt_gemm(which, maps, ...)` copies the maps into a local TMaps and passes it by value. The C++ checks the size, and the gemm refuses maps that were never encoded.
   - A new `_wgt_scratch()` builds one state's xs, hs, ws, tickets, err and maps. `state["wgt"]` holds them.
   - The tiny module now has no mutable globals or statics, and no Python module global holds a tensor.
3. **Two-state replay self-check.** `_wgt_replay_check` now builds a second state's scratch with its own maps, as a second prepare() would. For each T it captures one graph per state, A first then B. It then replays A, B, A, B, ... (12 replays), refilling a fresh routing in place before each replay. Each replay is judged against the fp32 dequant reference, with master's error printed beside it. Under the old globals, state A's graph would fail this check.

Verified, no change needed:
- **Routing-dependent values are computed on the device per call:** n_padded, used blocks, the K split (`pick_split`, which the producer and consumer compute identically) and the ticket targets. The host freezes only T-dependent sizes and the SM-count grid.
- **Tickets are race-free.** The gather of call N+1 zeroes the tickets only after the w2 of call N has finished; stream order guarantees it. w13 and w2 use disjoint ticket rows.
- **The split reduction is deterministic.** The last CTA sums the slices in K-slice order, so the graph output equals the eager output bitwise.
- **No stale scratch is read.** The gather rewrites every xs row of the used blocks, sentinel rows included, and w13 rewrites every hs row of the used blocks.
- **The A registers are not overwritten early.** Each stage runs `wgmma_wait<0>`, then `fence_acc` and `fence_a`, then the empty arrive, before the next dequant.
- **Shared memory fits.** Dynamic smem is 1024 + 16 x 13312 + 16 x 272 = 218368 B, plus about 264 B static, which is within 227 KB. The opt-in attribute is set in `wgt_init` before the first launch. The stage, slab and box offsets are all 1024-aligned, and expect_tx = 9216 + 2 x 2048 B matches the copies.
- **Master paths are unchanged.** Master's submission.py has not changed since 9080a71, and the diff against it removes no lines. bs >= 8 goes straight to master's `_forward(.., w2_mode)`, so the w2 atomic path is intact. Stage 2 is neither built nor called, and its `g_maps` globals sit only in the uncompiled `_WG_SRC_BODY`.
- **A skipped or failed phase leaves master.** A budget skip (230 s, or 200 s before the self-check), a raise, or a self-check or gate miss all set `wgt_ok={}` and print `PATH CHOSEN = MASTER`.
- **kernel_fn stays clean.** kernel_fn and `_moe_tiny` contain no sync, print, try/except or per-call host state.

Residual risk: the first nvcc/ptxas build of the new 16-stage N16 kernel and its TMA/mbarrier behaviour are unproven on hardware. A device fault would be sticky and fail the run rather than fall back. T=3, 5 and 6 are enabled without their own self-check.
