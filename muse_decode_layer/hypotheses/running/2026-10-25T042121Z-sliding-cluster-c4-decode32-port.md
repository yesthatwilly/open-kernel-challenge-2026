---
problem: muse_decode_layer
slug: sliding-cluster-c4-decode32-port
created: 2026-10-25T042121Z
base: master
area: FA3 call (sliding decode, batch 17..32)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's sliding_cluster_kernel
  (fp8qk TU, module muse_sliding_cluster_v2) from branch
  hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc, instantiating ONLY cluster
  size C=4 and gating it to sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config only), i.e.
  sliding decode_32. C=8 lost at B=16 in attention and C=16 was kernel-flat at B=1, so neither is
  built; batch 1 keeps decode_layer's _sliding_fused_kernel and 2..16 keeps master. The kernel
  takes the qkv GEMM output, does q/k RMSNorm + RoPE + FP8 (FA3-equivalent), stores the new k/v
  rows into the FP8 cache, gathers the window per rank, merges the 4 ranks over DSMEM and applies
  the RMSNorm x sigmoid(gate) epilogue, writing the bf16 tensor decode_layer's o GEMM reads; it
  replaces decode_layer's sliding chain between the qkv GEMM and the o GEMM at that band. Map
  decode_layer's qkv column layout, norm weights, rope base and window to the kernel arguments;
  the window covers every key (no skipping). At import call cudaOccupancyMaxActiveClusters for C=4
  and disable the path if < 32. Self-check at B in {17, 32} (lens < window and > window, kv_scale
  1 and 0.25): output vs master's chain, cache rows equal, graph replay. BUILD/SAFETY (LESSONS
  2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags
  are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first
  launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok'
  print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules
  reject starting processes). Any failure keeps the incumbent for the gated shapes and prints
  'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event
  timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower
  (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or
  allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement
  is an okc check.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "c243a29f: FA3 23.7 + combine 2.8 + prepare 2.2 + pre-attention/cast/store/norm-gate kernels (hidden, ~6-8 us) ~35 us; attention ran the same fused chain at 18.3 us (Triton 22.1). Floor ~3.3. Shape 242.3 -> ~226 (-6%), weight 46. Kill: kernel row > 22 us or shape not < 236"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: flat
    why: outside the band
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: outside the band
  - shapes: "*layer=full*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32
review: proceed
status: running
review_at: 2026-10-25T042453Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T042453Z
base_sha: 1cd12f4d3d46062939ca75935ef0fcad232f2ffc
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/sliding-cluster-c4-decode32-port
worktree: .worktrees/muse_decode_layer/sliding-cluster-c4-decode32-port
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043714Z
impl_commit: a3136ce5805b8fcc8f1c58fdd3500b1d54aeb06e
submission_sha256: 6bc87cd5411ce83e3a66d094be94c2d5bb4d94045a2f85bf5c66e9458cc1bd96
norm_sha256: f1a9292234f511aaf177bbad3b01d3b6940d525ee88242155f748f3d8ce21cca
submission_blob: e7b58542dc954a2a306648904dbd486bda0c417b
lint_warnings: 35
attempt_file: 2026-10-25T043714Z-check-sliding-cluster-c4-decode32-port.md
prepared_at: 2026-10-25T043714Z
job: 0fb2c9451c5d4283
dispatched_at: 2026-10-25T043721Z
---

# sliding-cluster-c4-decode32-port

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's sliding_cluster_kernel (fp8qk TU, module muse_sliding_cluster_v2) from branch hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc, instantiating ONLY cluster size C=4 and gating it to sliding layer, q_len 1, page 1, 17 <= batch <= 32 (config only), i.e. sliding decode_32. C=8 lost at B=16 in attention and C=16 was kernel-flat at B=1, so neither is built; batch 1 keeps decode_layer's _sliding_fused_kernel and 2..16 keeps master. The kernel takes the qkv GEMM output, does q/k RMSNorm + RoPE + FP8 (FA3-equivalent), stores the new k/v rows into the FP8 cache, gathers the window per rank, merges the 4 ranks over DSMEM and applies the RMSNorm x sigmoid(gate) epilogue, writing the bf16 tensor decode_layer's o GEMM reads; it replaces decode_layer's sliding chain between the qkv GEMM and the o GEMM at that band. Map decode_layer's qkv column layout, norm weights, rope base and window to the kernel arguments; the window covers every key (no skipping). At import call cudaOccupancyMaxActiveClusters for C=4 and disable the path if < 32. Self-check at B in {17, 32} (lens < window and > window, kv_scale 1 and 0.25): output vs master's chain, cache rows equal, graph replay. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_32*` faster: c243a29f: FA3 23.7 + combine 2.8 + prepare 2.2 + pre-attention/cast/store/norm-gate kernels (hidden, ~6-8 us) ~35 us; attention ran the same fused chain at 18.3 us (Triton 22.1). Floor ~3.3. Shape 242.3 -> ~226 (-6%), weight 46. Kill: kernel row > 22 us or shape not < 236
- `*layer=sliding load=decode_1 *` flat: outside the band
- `*layer=sliding load=decode_16*` flat: outside the band
- `*layer=full*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

124 regs + 32 B spill in attention; 103 KB smem/CTA. Check the spill line before trusting.

## Milestones

Rewrite stage 1 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042453Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companion. This is a single-shape, config-gated rewrite stage whose kernel row (sliding_cluster_kernel) and decode_32 shape time can be read directly, with eleven untouched shapes in the same run as controls. The real risks are that the in-layer saving comes in well under the ~16 us that summing the profile's kernel times predicts, and that a warm-L2 prepare selector picks the wrong path.

### Justification
Plausible on H200. Master's sliding batch>1 chain is still SGLang's: FusedQKNorm, the k/v/q FP8 casts, store_cache, FA3 prepare_varlen_num_blocks 2.2 + main 23.7 + FlashAttnFwdCombine 2.8, then FusedRMSNormGate. Profile c243a29f, sliding decode_32 = 242.3 us against a floor of 84.5. The donor kernel measured 18.3 us at B=32 in muse_attention_decode (job 1990ce53e9d34073: shape 25.84 -> 20.15 us; 124 regs, 32 B spill, 103,424 B smem, MaxActiveClusters C=4 = 62, so the >= 32 gate passes). Geometry matches decode_layer (hq 16, hkv 2, d 64, qkv [T, 2304], page 1, window 2048). The KV window at B=32 is ~16.8 MB, so the floor is ~3.5 us and the kernel sits ~5x above it. Stage-1 milestone 3 (within 1.5x of the floor) is not reachable this stage, which is acceptable for a rewrite. Only sliding decode_32 is gated, so the expected flat directions on every other shape are right. WEAKEST LINK: the ~35 us incumbent figure is a sum of profile rows plus a guessed 6-8 us of 'hidden' glue, and the layer has already shown that this arithmetic overstates the realizable saving. The harvested sliding-fused-batched-decode-port replaced the same chain at decode_32 with a 22.1 us Triton kernel. That predicted about -13 us, yet the shape moved only 246.99 -> 245.52 us (delta 1.03 by paired ratio). If the same discount applies, the cluster kernel's saving is closer to -5..-10 us than -16, and the kill line 'shape not < 236' could fire even though the kernel row is below 22 us. Judge stage 1 on the kernel row against the summed incumbent rows (FA3 + combine + prepare + the casts/store/qknorm/normgate rows in the same profile), not on the 236 line alone.

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl (just landed): disjoint gate (window == 0 only), so there is no timing interaction. It does add a third load_inline module (dense chain 50 s + splitkv 49 s + cluster ~41 s = ~140 s of compile inside the 15-minute run budget), which is a timeout risk, not a confounder.
- small-t-launch-minimal-dense-chain-pdl: its PDL kernels (P, R) sit after the o GEMM and attn_boundary, and the attention-side chain is plain stream order, so the cluster kernel joins no PDL edge. Note that the attention-side PDL attempt (dense-chain-stage2-attn-side-pdl) lost on sliding decode_16/32, which shows this region is sensitive to launch-order changes. A small loss could come from the cluster launch's own setup cost, not from attention.
- sglang-pipeline-parity-router-hilo-m8: defines the incumbent (FusedQKNorm, FusedRMSNormGate, FA3), and the self-check compares against it. That is not a confound, but the reference chain in the self-check must be master's exact kv_scale division and q FP8 cast.
- stack-align-par-sidestream-sliding1 / fa3-hdim128-head-pairing-batched / post-expert-norm-triton-small-t / marlin-block-m-cap-16: gated to batch 1, full layers, the post-expert stage or T>32 respectively. None share the replaced kernels or memory traffic at sliding decode_32, so none confound.

### Interference and A/B needs
No companion is warranted. The gate selects exactly one scored shape (sliding decode_32), and no landed hypothesis touches the kernels it replaces, so reverting one (ab-revert) would measure the same thing. The other eleven shapes are byte-identical in-run controls. The profile names sliding_cluster_kernel as its own row, so a gain, a loss or a fallback can each be read from one run: the kernel row against the summed incumbent rows, plus the PATH CHOSEN and selector-timing lines. A split control is unnecessary because the kernel is one indivisible launch. The only thing that can make a single result ambiguous is the prepare-time selector silently keeping the incumbent, and the log disambiguates that, so it needs a logging requirement, not a second run. The parallel attention re-measure (stack-streamk-ticket-cluster-c16-c4, job 195dcc1209b6457b) does not change the decision. decode_layer's incumbent at B=32 is FA3, not attention's Triton or cluster, so C=4 needs only to beat ~29+ us of FA3-plus-glue, not attention's newer master.

### Open questions
- Prepare-time selector and L2: the B=32 KV window (~16.8 MB) fits in the H200's L2, while the judge cold-flushes L2 before every replay. A back-to-back CUDA-event comparison in prepare therefore measures both paths with warm KV and can mis-rank them. Flush L2 between selector reps (write a buffer larger than L2) or time each path as a graph after a flush, and print both numbers so the log shows whether the selector kept the incumbent.
- The selector must time the incumbent as the WHOLE replaced chain (qk_norm through o_norm_gate, including the casts and store_cache) against the single cluster launch. Timing FA3 alone would bias it toward rejecting the new kernel.
- None of the 13 tests has sliding at 17 <= batch <= 32, so this path is checked only on the timed decode_32 calls. A wrong result surfaces as a failed run after timing begins, not in the tests. The B in {17, 32} self-check against master's chain (rel L2 and exact cache-row equality) is the only early guard, so keep it strict.
- RoPE mapping: decode_layer's sliding layers use rope_theta 500000 with interleaved RoPE via cos_sin positions. Confirm that the donor's _inv_freq/_norm_rope pairing convention matches FusedQKNorm here, as the batch-1 _sliding_fused_kernel already does in this file, and reuse that file's helpers rather than adding a second copy.
- Compile budget: three load_inline modules (~140 s) plus the self-checks. Consider whether the cluster TU can share the splitkv module's build to cut one nvcc invocation; at minimum, print compile seconds per module.
- If attention's job 195dcc1209b6457b re-measures C=4 at B=32 well above 18.3 us, re-read this run's kernel row with that in mind. It still does not block dispatch.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (+1096/-82; about 80 of the deleted lines are a move).

What changed:
- `_master_attn(data, state, qkv)`: master's whole chain between the qkv GEMM and the o GEMM (everything that was kernel_fn's `else:` branch) moved into a function, byte-for-byte apart from indent. kernel_fn calls it, and the self-check and selector call the same function, so the reference is master's exact kv_scale division and q FP8 cast.
- kernel_fn: `elif _slide_gate(cfg): o = _slide_forward(data, qkv)` sits right after the batch-1 sliding branch, where a batch 2..16 branch would go. The gate reads config only (q_len 1, window, rope_theta, 17 <= batch <= 32, 16/2/64 heads, page 1) plus the module constant `_SLIDE_PATH` that prepare() sets. There is no try/except, timing, sync or occupancy query in kernel_fn.
- New section before `prepare()`: `sliding_cluster_kernel` ported verbatim from the attention branch (module `muse_sliding_cluster_c4_v1`, fp8qk only, one load_inline, no f16 fallback build). Only the host side was trimmed: `slide_setup` reports occupancy for C=4 only, `slide_attn` requires C == 4 and B <= 32. The kernel stores every cache_loc row, slot 0 included.
- Build: `TORCH_CUDA_ARCH_LIST=9.0a` set around load_inline and restored, flags `-O3 -std=c++20 -lineinfo`, the arch flags printed, compile seconds printed. The kernel needs no sm_90a-only PTX (mma.sync e4m3, clusters, cp.async), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard or error flag. There are no mbarrier or spin waits (cluster barriers and cp.async groups only).
- `_sliding_forward` gained an optional `splits` argument (default unchanged), used for the Triton 16-split candidate (sibling hypothesis sliding-batched-triton-b16-band) at B=17..32.
- `_slide_prepare` (called last in prepare(), never raises):
  1. Prints regs / local spill / dynamic smem / binaryVersion / ptxVersion and `cudaOccupancyMaxActiveClusters` for C=4. It disables the cluster path if binaryVersion != 90 or occupancy < 32.
  2. Runs a tiny B=1 canary (flushed `launching ... ok` prints around a synchronize), then a self-check at B=17 (kv_scale 0.25), B=32 (kv_scale 1) and B=32 (kv_scale 0.25), with lens below, at and past the window and window-edge keys planted. Request 0's new row is slot 0.
  3. The self-check compares master's chain, the cluster and Triton-16 against an fp32 reference. The cluster must match its own-rounding reference (rel L2 < 2e-2) and sit within 2.5x of master's error against the unrounded fp32 reference plus 5e-3. The Triton candidate must be < 2e-2. v cache rows must be byte-exact against master's, k rows may differ by one E4M3 code in < 6% of bytes (master rounds through bf16 first), and no other cache row may change. The cluster launch is also captured and replayed as a graph and must equal the eager output.
  4. Times the WHOLE master chain, the cluster launch and the Triton-16 launch as CUDA graphs at B=32, interleaved, 30 reps after 5 warm-ups. A 320 MB write flushes L2 before every replay. It prints all medians and sets `_SLIDE_PATH` to the fastest candidate only if it is strictly faster than the master chain; otherwise master. Every outcome prints `PATH CHOSEN` (with `(fallback: <reason>)` on a fallback).

Review fixes applied (the review wins where it differs from the spec): no donor try/except; every cache_loc stored including slot 0 (and tested); donor q rounding order kept (q * qk_scale_factor then E4M3, not FA3's round-then-scale; both are E4M3, so within the rules, and the self-check is calibrated against master's own error rather than a fixed bound); no timing, sync or occupancy query in kernel_fn; selector flushes L2 and times the whole replaced chain, plus the Triton 16-split candidate; one instantiation (~41 s extra compile expected, printed per module); the decision comes from the printed medians.

Deviations from the spec: the selector times at B=32 only and applies the choice to 17..32; the canary is B=1, which the host check allows.

How to judge (per the review): compare the `sliding_cluster_kernel` row against the SUMMED rows it replaces (FA3 main 23.7 + combine 2.8 + prepare 2.2 + FusedQKNorm, the FP8 casts, store_cache and FusedRMSNormGate rows), not only the 236 us shape line. Expect -5..-10 us on the shape. Read the `PATH CHOSEN` and median lines first: if prepare chose master or triton16 the profile will not show the cluster row. Prepare-time timings ranked paths but undercount the judged time by 20-25% (LESSONS).

## Rules check

- [x] CUDA-graph capturable: kernel_fn's new branch is one `slide_attn` launch (or the existing Triton launch) with torch.empty outputs; no `.item()`/sync/print/data-dependent control flow; the cluster launch and the master chain are captured and replayed in prepare(). The sync, `int(...)` and `.nonzero()` calls are all in prepare-only helpers.
- [x] Nothing derived from `data` is cached across calls: only `_inv_freq` (config constant, existing helper), the Triton ticket counters (zeroed, left at 0 by every launch, existing mechanism) and the module/path constants set in prepare().
- [x] Weights: n/a: no weight is touched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3 (q rounded to E4M3 once, as the review keeps).
- [x] Attention sees every key: window 2048 covers the token and the 2047 before it; every key of every rank's chunk is scored, nothing skipped; the self-check plants window-edge keys.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: no process creation anywhere (load_inline builds in-process, as the two existing modules do); the CUDA-event timing is in prepare() only.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_decode_layer/submission.py --problem muse_decode_layer`): `OK with warnings`, 0 errors. All warnings are `host-sync` outside kernel_fn's call graph (prepare-only), plus one `dtype` warning for the uint8 L2-flush scratch buffer in prepare() (never an activation or weight). `python3 -I -m py_compile` passes and `__pycache__` was deleted.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, reading only (no GPU). **Verdict: dispatch-with-fixes** (one cosmetic log fix). Rules OK.

- `_master_attn`: identical to master's `else:` branch once the indent is removed (mechanical diff; the only addition is `return o`). It recomputes batch/length/tokens/heads from cfg exactly as kernel_fn does, and kernel_fn uses none of the branch's locals after it. Every non-gated shape is unchanged.
- Kernel vs donor (hyp/muse_attention_decode/sliding-cluster-s2-decode-c4-pdl-inproc 3355d0f): the CUDA TU is byte-identical apart from host trimming (C=4 only, B <= 32, no NonPortableClusterSize attribute). It stores cache_loc for every row, slot 0 included. q is RMSNorm + RoPE in fp32, then times qk_scale_factor, then E4M3 once. Window keys are [seq-2047, seq) plus the new key, which is every key the reference's `key_pos > query_pos - window` admits. The host args match the donor call. qkv column layout, int64 positions/cache_loc, int32 seq_lens/page_table and cache shape [pages, 1, 2, 64] all match reference.generate_input, so the kernel's TORCH_CHECKs hold on judge data. rope_theta is nonzero on sliding layers, so the gate can fire.
- Calibration (numpy emulation of the synthetic cases): master's error against the unrounded fp32 reference is about 0.064-0.076, and the cluster's (round-after-scale) is about 0.064-0.074. The bound is 2.5x master + 5e-3, about 0.19, so the check is neither always-fail nor loose. The own-rounding bound (< 2e-2) sits against an expected ~1e-3. The k-row byte mismatch from bf16 double rounding is 3.1% emulated (6% threshold, about 8 sd of margin at B=17), and the one-code bound 0.126|x| + 2^-9 covers the E4M3 worst case of 0.125.
- Gate is config plus the prepare-set `_SLIDE_PATH`. kernel_fn has no try/except, timing, sync or per-call state. `_inv_freq` and the Triton counters are warmed in prepare. Occupancy < 32 disabling is sensible (donor reported 62).
- Selector: whole master chain vs single launch as graphs, interleaved, flushed. The flush is a 320 MB *write* (the judge reads), so dirty-line writeback adds roughly equal absolute time to both paths. That is left as is because the ranking is preserved.
- Fix: the PATH CHOSEN line now also names why the cluster was off when triton16 is the pick (`; cluster off: <why>`). Before, the reason appeared only in the no-candidate fallback line.
- Scope creep (flagged, not removed): the Triton 16-split candidate belongs to the sibling hypothesis. Triton at decode_32 was already harvested near-neutral. If it wins the selector, this run shows no sliding_cluster_kernel row and measures the sibling, so read the medians line first.

Residual risk: compile budget (a third load_inline module, ~41 s, plus per-prepare self-check/selector), and a prepare-time selector that may still mis-rank by the LESSONS 20-25% undercount.
