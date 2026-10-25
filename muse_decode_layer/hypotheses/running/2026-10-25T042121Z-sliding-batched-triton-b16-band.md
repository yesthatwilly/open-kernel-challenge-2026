---
problem: muse_decode_layer
slug: sliding-batched-triton-b16-band
created: 2026-10-25T042121Z
base: master
area: FA3 call (sliding decode, batch 2..16)
change: >
  Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV
  kernel A/B on full decode 1 < batch <= 64). Re-apply decode's harvested
  sliding-fused-batched-decode-port (done/neutral patch: the Triton _sliding_fused_kernel batched
  path with 16 splits, ported from attention's landed sliding-deep-splits-on-cuda-splitkv), but
  narrow its config gate from 'sliding, q_len 1, batch > 1' to sliding, q_len 1, page 1, 2 <=
  batch <= 16 (sliding decode_16 and the batch-8/9 tests). batch 17+ (sliding decode_32) stays on
  master FA3 so that band is left for sliding-cluster-c4-decode32-port and the two can later stack
  without overlap. No other change; apply the patch hunks onto master and resolve any conflict in
  kernel_fn by keeping both master's full-decode split-KV route and this sliding route.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "the port moved sliding decode_16 217.1 -> 209.1 us (-3.7%, gained) on its base; master row ~205-210 -> ~200. Weight 46. Kill: not faster than master by z>=2"
  - shapes: "*layer=sliding load=decode_32*"
    direction: flat
    why: outside the narrowed gate
  - shapes: "*load=decode_1 *"
    direction: flat
    why: outside the gate
  - shapes: "*layer=full*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042452Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T042453Z
base_sha: 94f52a0a74f65619042e18250fd7cb4a206cf66a
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/sliding-batched-triton-b16-band
worktree: .worktrees/muse_decode_layer/sliding-batched-triton-b16-band
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T042712Z
impl_commit: 1cabea713e533afb5702682a3a7bd2a095c140e6
submission_sha256: 4d7058499be102b9ede8f6d4e75520364f48dc530b8e28ca3eb28a5803c7de98
norm_sha256: 88f354bf180a612d29e0fcb2a25a382a5ab8a2c231a3e3b9bad730e078755f81
submission_blob: 97d379721e56bf4fe6307b85b74f821940f73e67
lint_warnings: 30
attempt_file: 2026-10-25T042712Z-check-sliding-batched-triton-b16-band.md
prepared_at: 2026-10-25T042712Z
job: 02bf417257e14a3d
dispatched_at: 2026-10-25T042729Z
---

# sliding-batched-triton-b16-band

## Change

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Re-apply decode's harvested sliding-fused-batched-decode-port (done/neutral patch: the Triton _sliding_fused_kernel batched path with 16 splits, ported from attention's landed sliding-deep-splits-on-cuda-splitkv), but narrow its config gate from 'sliding, q_len 1, batch > 1' to sliding, q_len 1, page 1, 2 <= batch <= 16 (sliding decode_16 and the batch-8/9 tests). batch 17+ (sliding decode_32) stays on master FA3 so that band is left for sliding-cluster-c4-decode32-port and the two can later stack without overlap. No other change; apply the patch hunks onto master and resolve any conflict in kernel_fn by keeping both master's full-decode split-KV route and this sliding route.

## Expected

- `*layer=sliding load=decode_16*` faster: the port moved sliding decode_16 217.1 -> 209.1 us (-3.7%, gained) on its base; master row ~205-210 -> ~200. Weight 46. Kill: not faster than master by z>=2
- `*layer=sliding load=decode_32*` flat: outside the narrowed gate
- `*load=decode_1 *` flat: outside the gate
- `*layer=full*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

Its earlier run's verify_32 'loss' was on untouched paths (drift).

## Adversarial review

Reviewed 2026-10-25T042452Z by an adversarial subagent (model opus).

**Verdict: proceed.** Config-only, shape-disjoint re-run of a kernel already measured three times on the dense-chain base (sliding decode_16 +3.8% isolated, +5.5% and +5.2% in the two stacks), now limited to 2 <= batch <= 16; sliding decode_16 is the only shape that changes and the other 11 shapes are in-run controls, so no companion is needed. Two build points must be fixed in the change text: keep S=16 (master's _num_splits would give 4 at B=16) and do not paste the patch's helper hunk, which would duplicate master's helpers.

### Justification
Plausible and already shown. At sliding decode_16, master still runs SGLang's chain between the qkv GEMM and the o GEMM: FusedQKNorm+RoPE, k/v FP8 casts, store_cache, q cast, FA3 prepare + main (~13.6 us) + FlashAttnFwdCombine (~3.7 us), then FusedRMSNormGate. None of the landed hypotheses touched that chain. sliding-decode1 only gates batch 1, splitkv only gates window == 0, and the dense chain starts after the o GEMM. The same _sliding_fused_kernel (master's copy is byte-identical to the patch's helpers; I diffed them) moved sliding decode_16 217.1 -> 209.1 us (paired ratio 1.038) alone in b01fd2df. On the same dense-chain base it moved it 217.1 -> 195.4 (1.055) in 2193fc0d and 217.1 -> 190.5 (1.052) in b94e7973. That is three readings of 3.8-5.5%, all above the 1.5% per-shape noise. Expected directions match the gate. Batch 1 is master's _sliding_forward, full layers are splitkv/paired FA3, verify is q_len 8, and decode_32 is batch 32 > 16, so every one of those is a byte-identical control. Floors: sliding decode_16 is at 205.0 us against a 76.7 us floor; FA3 is not in its top three (Marlin 97.0, nvjet 26.3, P kernel 25.1), and the ~25-28 us attention chain is the part being cut. Headline impact is small. Sliding decode_16 at -8..-11 us x weight 46 saves ~370-500 us of a ~14.7 ms decode_16 load (~2.5-3.4%), about +0.4-0.6% on the 6-load geomean, well inside the ±1.5% CI. Judge it per shape: the kill rule is z>=2 on decode_16, and this cannot become a promotion on the headline by itself. Weakest link: the number of splits. The measured gain used S=16 hard-coded (512 CTAs at B=16, ~2 MB fp32 workspace). Master's _sliding_forward calls _num_splits(), which returns 4 at B=16. An implementer who widens master's batch-1 gate and reuses _sliding_forward will silently run S=4, a configuration that has never been measured inside the layer, and the result then attributes to the split count, not to the port. Second weakest: the parenthetical '(sliding decode_16 and the batch-8/9 tests)' is wrong. The batch-8 test (ctx 2047) uses the default page 64 (reference.py PAGE = 64), so the page==1 gate sends it to FA3. Only the batch-9 test (page 1, mixed spread) exercises the new path before timing. That is still adequate, because the batch-9 test passed in all three earlier runs.

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl (just landed, b047510): no mechanistic confound. Its gate is window == 0 and the new gate is window != 0, so sliding decode_16 never reaches it. Two mechanical caveats only: (1) kernel_fn now nests the splitkv/paired branch inside master's else, so the patch's kernel_fn hunk (base 90040c7) will not apply and must be hand-merged; (2) the splitkv module builds at import and self-checks in prepare(), which adds compile time against the 15-minute budget but is already paid on master. It also explains why full decode_16/32 sit at ~330/490 us in the base now; do not mistake that for a movement of this run.
- stack-align-par-sidestream-sliding1 (8fa8fbf): shares code with this change, not shapes. It brought in the helpers (_counters, _inv_freq, _sliding_fused_kernel) and master's batch-1 _sliding_forward. _counters is keyed by (batch, hkv), so the B=16 and B=1 ticket buffers are separate. Its T>32 align/side-stream work never runs at T=16. The patch's first hunk would re-define all of these helpers; Python would keep the later, identical copy, so behaviour is unchanged, but the file is messier and a later stack could break. Reuse master's copies instead.
- small-t-launch-minimal-dense-chain-pdl (f1f2f35): T<=32 PDL chain that starts after attn_boundary/o GEMM. The fused Triton kernel issues no griddepcontrol, but neither does master's FA3/FusedRMSNormGate before the cuBLAS o GEMM, so the PDL edge is unchanged. Both stacks that showed +5.2-5.5% were built on this base, so the interaction has already been measured and is not a confound.
- fa3-hdim128-head-pairing-batched, marlin-block-m-cap-16, post-expert-norm-triton-small-t, sglang-pipeline-parity-router-hilo-m8: disjoint (full-layer FA3 or downstream MoE/router), and the o GEMM operand is still [T,1024] bf16 in the same layout. No confound.

### Interference and A/B needs
No companion is warranted. The gate reads config only (window != 0, q_len 1, page 1, 2 <= batch <= 16). Only sliding decode_16 changes kernels, and the run itself carries 11 untouched shapes as noise controls, so a gain or a loss there belongs to the fused path. No landed hypothesis shares the replaced kernels, so an ab-revert would measure the same thing. A splits control (S=4 via _num_splits vs S=16) would isolate a sub-choice, but only if the implementer keeps S=16 as specified. Pin it in the build instead of spending a check on it. Relation to sliding-cluster-c4-decode32-port: the bands are disjoint and both attribute. This hypothesis covers sliding/q_len 1/page 1/batch 2..16 (scored: decode_16); the cluster port covers 17..32 (scored: decode_32); batch 1 stays on master's _sliding_forward. Each run's target shape is a byte-identical control in the other run, so the two results do not mix. One caution for the pair: decode_32 was not left on FA3 because Triton had nothing to offer there. This same Triton path gained +3.1%, +4.1% and +4.8% (paired) at sliding decode_32 in b01fd2df, 2193fc0d and b94e7973. The cluster port's own incumbent is FA3, so its decode_32 result attributes cleanly as cluster vs FA3. But its expected 242 -> ~226 (-6%) only beats the Triton alternative by ~1-3% per shape, which is under 2 sigma. The decision about who owns decode_32 therefore needs either the cluster port's prepare-time selector to also time the Triton S=16 path at B=32 (preferred: free, no check spent), or, if the cluster port lands, a later band-widen check of this path to 2..32 run against the cluster path. That is a follow-up for the cluster hypothesis, not a companion here. To keep both ports stackable without a textual fight, structure kernel_fn's sliding dispatch as an if/elif chain on config: batch==1 -> _sliding_forward; 2<=batch<=16 -> this path; (cluster later adds 17<=batch<=32); else master. Both hypotheses then touch adjacent elif lines.

### Open questions
- Change text should say explicitly: splits = 16 at q_len 1 (as the patch's _sliding_forward_batched), NOT _num_splits(batch, hkv) (= 4 at B=16), and reuse master's helpers rather than applying the patch's first hunk. Otherwise the run measures an untested S=4 configuration.
- Correct the gate description: with page == 1, the batch-8 test (page 64) runs FA3 and only the batch-9 test exercises this path. Either accept that or drop the page==1 term (the kernel handles PAGE as a constexpr and passed the page-64 batch-8 test in all three earlier runs). Dropping it would add test coverage and does not touch the cluster port's band, which is gated by batch.
- Read the fused kernel's own time on sliding decode_16 from the profile (_sliding_fused_kernel row) against master's FA3 main + combine + glue, so the per-kernel saving is recorded, not just the shape delta.
- stack-align-par-sliding-batched (b94e7973) failed on full verify_1 (rc 112, 13/13 tests) on a shape neither of its parents gates, and master (which carries align-par) has since passed. Treat it as unrelated unless the full verify_1 failure reappears here.
- For the cluster port: make its prepare-time selector time the Triton S=16 path at B=32 alongside FA3 and the cluster kernel, so the decode_32 band owner is chosen on measured kernel time.


## Implementation notes

Edited only `muse_decode_layer/submission.py` in the worktree (+67/-2). Hand-merged the patch's kernel branch onto master; the patch's helper hunk was NOT applied (master's `_inv_freq`, `_counters`, `_sliding_fused_kernel` are used as they are).
- New `_sliding_b16_gate(cfg)`: window != 0, q_len 1, page_size 1, 2 <= batch <= 16, hq 16 / hkv 2 / d 64. Config only. New `_sliding_forward_b16` (same launch as master's `_sliding_forward`, `splits = 16` hard-coded) and `_sliding_b16_or_none` (try/except, per-config on/off dict, prints `PATH CHOSEN = FUSED TRITON S=16` once, or `PATH CHOSEN = MASTER (FA3)` after a failure; never raises).
- `kernel_fn`: `o = None`; `if batch == 1 sliding: _sliding_forward` / `elif _sliding_b16_gate(cfg): o = _sliding_b16_or_none(...)` / `if o is None:` master's existing body unchanged (indentation kept; a batch 17..32 elif can sit directly under the b16 elif). Master's splitkv/paired full-decode route is untouched.
- Review fixes applied (the review wins over the spec text): S=16 hard-coded rather than `_num_splits` (4 at B=16); no duplicated helpers; gate has the page==1 term, so only the batch-9 test exercises the path (the page-64 batch-8 test stays on FA3), as the review accepted. Batch 1 stays on master's `_sliding_forward`.
- Not done (profile read is post-run): record the `_sliding_fused_kernel` row vs FA3 main + combine + glue from the profile.

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/synchronize in the new code; the branch is config-only; workspace allocations are by shape, not values.
- [x] Nothing derived from `data` is cached across calls: only a config-keyed on/off flag and master's zero-reset ticket counters (hold no data).
- [x] Weights untouched (n/a: no weight change).
- [x] Activations BF16, KV cache FP8, query bf16 in the fused kernel (finer than E4M3), same kernel as master's batch-1 path.
- [x] Attention sees every key of the window (kernel unchanged; all splits cover [lo, seq-1] plus the new keys).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess or child process.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune; Triton compile errors fall back to master, never raise.

okc_lint: `OK with warnings` (only pre-existing host-sync warns in prepare(), outside kernel_fn's call graph). py_compile OK, __pycache__ removed.
