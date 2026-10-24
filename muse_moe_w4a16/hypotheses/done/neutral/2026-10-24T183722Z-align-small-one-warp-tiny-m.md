---
problem: muse_moe_w4a16
slug: align-small-one-warp-tiny-m
created: 2026-10-24T183722Z
base: master
area: MoE align/dispatch
change: >
  num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep
  master's launches exactly. In _align_small (host wrapper), replace the fixed num_warps=4 of the
  _align_small_kernel launch with num_warps = 1 if n = num_tokens*8 <= 64 else 4. Nothing in the
  kernel changes. The prepare() warm-up loop already calls _align_small for every tokens 1..32, so
  it compiles the 1-warp variants for 1..8 automatically; _align_self_check runs through the same
  wrapper and so checks the 1-warp variants too. Mechanism: at n <= 64 every tensor in the kernel
  is tiny (ids [<=64], one-hot [<=64, 32], CAP_BLOCK <= 512, MB_BLOCK <= 64), so 4 warps leave
  most threads idle while every tl.sum / tl.cumsum across the 4 warps goes through shared memory
  and a bar.sync; one warp does all of them with shuffles only, so the kernel's serial chain of ~8
  reductions/scans gets shorter. One-warp register use: the [64, 32] one-hot and its cumsum are 64
  int32 per thread per temporary, under the 255 limit (print n_regs/n_spills of the 1-warp bs=8
  variant in prepare via the compiled kernel's .n_regs/.n_spills).
expected:
  - shapes: bs=1 *
    direction: faster
    why: _align_small_kernel 1.4 us -> ~1.0-1.2 us, ~0.5% of 49 us
  - shapes: bs=8 *
    direction: faster
    why: 2.7 us -> ~1.8-2.2 us, ~0.5-0.8% of 108 us
  - shapes: bs=16 *
    direction: flat
    why: untouched
  - shapes: bs=32 *
    direction: flat
    why: untouched
  - shapes: bs=128 *
    direction: flat
    why: untouched
  - shapes: bs=256 *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T183956Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T183957Z
base_sha: 6ef6502619e893cbf1037aeccc0405617e73620d
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/align-small-one-warp-tiny-m
worktree: .worktrees/muse_moe_w4a16/align-small-one-warp-tiny-m
mode: submit
implemented_at: 2026-10-24T184146Z
impl_commit: 1333b371d035b1380ecf095a92cea3b903f993c6
submission_sha256: aa88ec9dd62eba95ff78c88aabb0300e47c36e8bb79f29a120e5c2f104f8466c
norm_sha256: 179ed1c50ddffe533effbcc8aabafdda031ecefaa1fbfa203a06017978de4cfd
submission_blob: 92f9e59575322a03267aab74308a495aeeaf19f1
lint_warnings: 8
attempt_file: 2026-10-24T184146Z-submit-align-small-one-warp-tiny-m.md
prepared_at: 2026-10-24T184146Z
job: 38532ba10e394e3c
dispatched_at: 2026-10-24T184157Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.328709212434543
speedup_ci: [1.308778574248025, 1.348639850621061]
harvested_at: 2026-10-24T184720Z
best_at_harvest: 1.3249153782392282
promotion: null
---

# align-small-one-warp-tiny-m

## Change

num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep master's launches exactly. In _align_small (host wrapper), replace the fixed num_warps=4 of the _align_small_kernel launch with num_warps = 1 if n = num_tokens*8 <= 64 else 4. Nothing in the kernel changes. The prepare() warm-up loop already calls _align_small for every tokens 1..32, so it compiles the 1-warp variants for 1..8 automatically; _align_self_check runs through the same wrapper and so checks the 1-warp variants too. Mechanism: at n <= 64 every tensor in the kernel is tiny (ids [<=64], one-hot [<=64, 32], CAP_BLOCK <= 512, MB_BLOCK <= 64), so 4 warps leave most threads idle while every tl.sum / tl.cumsum across the 4 warps goes through shared memory and a bar.sync; one warp does all of them with shuffles only, so the kernel's serial chain of ~8 reductions/scans gets shorter. One-warp register use: the [64, 32] one-hot and its cumsum are 64 int32 per thread per temporary, under the 255 limit (print n_regs/n_spills of the 1-warp bs=8 variant in prepare via the compiled kernel's .n_regs/.n_spills).

## Expected

- `bs=1 *` faster: _align_small_kernel 1.4 us -> ~1.0-1.2 us, ~0.5% of 49 us
- `bs=8 *` faster: 2.7 us -> ~1.8-2.2 us, ~0.5-0.8% of 108 us
- `bs=16 *` flat: untouched
- `bs=32 *` flat: untouched
- `bs=128 *` flat: untouched
- `bs=256 *` flat: untouched

## Risk

Low-gain, low-risk launch-latency tune; the headline cannot resolve it, the _align_small_kernel rows at bs=1/8 can (Triton rows reproduce within ~1%). If 1 warp spills at bs=8 (n_spills > 0 in the log) the expected direction there flips to slower; that is the main risk. Overlap: the running bs1 fused rewrite replaces this kernel at bs=1 and the queued midm rewrite at 2..32; the running swap-AB keeps _align_small but at a larger BSM (8*NT), which does not change the variant's size class at bs<=8 (BSM 8).

## Adversarial review

Reviewed 2026-10-24T183956Z by an adversarial subagent (model opus).

**Verdict: proceed.** A host-side num_warps 4->1 tune of _align_small_kernel at num_tokens<=8 is plausible, rules-clean, shape-disjoint from every other landed change, and readable from the per-kernel profile rows (1.4 us at bs=1, 2.7 us at bs=8 in job 3cabc739), though the headline (<1% at two of six loads) cannot resolve it.

### Justification
Profile 3cabc739f1af493e confirms the hypothesis's numbers: _align_small_kernel is 1.4 us (3.2%) at bs=1 and 2.7 us (2.2%) at bs=8, directly on the critical path before the w13 Marlin launch. The kernel is one program with a serial chain of tl.sum/tl.cumsum over tiny tensors (bs=1: BLOCK_N=8, CAP_BLOCK=256, MB_BLOCK=32, BSM=8; bs=8: BLOCK_N=64, CAP_BLOCK=512, MB_BLOCK=64, BSM=8). At 4 warps, any reduction or scan whose axis Triton's blocked layout splits across warps needs a shared-memory round trip and a bar.sync; at 1 warp everything is in-thread or shuffles. So the mechanism is plausible. Expected size: saving 0.2-0.9 us is 0.3-0.8% of the bs=1 and bs=8 shape times, which moves the geomean by about 0.1-0.25%. That is far inside the run's ±1.5% CI, so only the _align_small_kernel profile rows can judge it. Weakest link: the profile shape does not clearly point to sync cost. The kernel goes 1.4 us (bs=1, n=8) -> 2.7 us (bs=8, n=64) -> 2.8 us (bs=16, n=128). The jump from bs=1 to bs=8 goes with the [BLOCK_N, 32] one-hot growing from 256 to 2048 elements. That suggests per-element in-thread work (one-hot, two cumsums along axis 0, the onehot*seg_start and rank products), not only cross-warp sync. At 1 warp that work is 4x more serial per thread (64 elements per thread per [64,32] temporary, against 16 at 4 warps), so bs=8 could come out flat or slower even with no spills. bs=1 ([8,32] = 8 per thread at 1 warp) is the likelier winner. The expected directions are consistent with TASK.md: bs=16 and up are untouched because n=128 > 64 keeps num_warps=4, and the weights are equal across loads.

### Confounders on the current kernel
- triton-map-align-on-block-m-restack: introduced _align_small_kernel with the fixed num_warps=4 that this change tunes. It is the change's subject, not a confounder. Its block_size_m pick (two smaller only at >=128 tokens) leaves BSM=8 at bs<=8, so the tuned variants are the same ones it measured.
- align-parallel-per-expert-verify: only num_tokens > 32 (_align_parallel_kernel). Shape-disjoint, no interaction.
- post-expert-norm-triton-small-m: a different kernel at the tail of the same small-m launch sequence. Its time does not depend on the align kernel's warp count, so no interaction beyond graph launch order.
- marlin-block-m-two-smaller-on-postnorm: only num_tokens >= 128. Disjoint.
- sglang-marlin-path-parity: provides the Marlin consumers of sorted_ids/block_experts. Their contents are bit-identical under the change (same kernel, same outputs), so Marlin time cannot move because of it.

### Interference and A/B needs
No companion is warranted. The change is one scalar on one launch at shapes no other landed hypothesis touches, and the outputs are bitwise identical, so downstream kernels cannot change. Attribution is direct: compare the _align_small_kernel row at bs=1 and bs=8 against 1.4/2.7 us in 3cabc739f1af493e (and the same rows in 45eacf4ca2e849d2), with the bs=16 row (2.8 us, untouched variant) as the in-run control for profiler drift. A gain is unambiguous. A loss at bs=8 with n_spills=0 would mean in-thread serial work outweighs the saved syncs. That would naturally lead to a follow-up (num_warps=2 at n<=64, or 1 warp only at n<=8, i.e. bs=1 only), not a companion now, because the parent's single result already says which direction 1 warp goes at each size.

### Open questions
- _align_self_check covers _align_small only at tokens 1, 8, 16, 32. The 1-warp variants at tokens 2-7 (including the judge's bs=4 test, n=32) are compiled in warm-up but not checked against SGLang's align in prepare. The kernel body is unchanged, so the risk is low, but adding tokens=4 to the self-check shapes would cover the bs=4 test before the judge does.
- If bs=1 improves and bs=8 does not, is the sync cost or the per-thread one-hot/cumsum work the dominant term at n=64? The n_regs/n_spills print plus the bs=8 row will tell. Consider num_warps=2 at 32 < n <= 64 as the follow-up.
- Overlap with in-flight rewrites: cuda-fused-moe-bs1-persistent-w13-silu-w2 (bs=1) and cuda-fused-moe-midm-single-launch-routed (2..32) would retire this kernel at these shapes if they land. The gain here is worth landing only while _align_small remains on the bs<=8 path.


## Implementation notes

In `_align_small` (host wrapper, submission.py ~L277) the `_align_small_kernel` launch now passes `num_warps=1 if n <= 64 else 4` (n = num_tokens*8, so 1 warp at tokens <= 8). Kernel body unchanged. Triton's tl.sum/tl.cumsum are layout-generic, so the 1-warp variant is correct with no kernel change (nothing is hard-sized for 4 warps). The prepare() warm-up loop calls `_align_small` for tokens 1..32, so the num_warps=1 variants for 1..8 compile there, and `_align_self_check` (tokens 1, 8, 16, 32) runs through the same wrapper. Deviations: did not add the n_regs/n_spills print (no GPU to verify; the profile row at bs=8 decides) and did not add tokens=4 to the self-check shapes (reviewer's optional suggestion, kept diff minimal).

## Rules check

- [x] CUDA-graph capturable: diff is one host-side int expression on a static shape; no syncs, no value-dependent control flow.
- [x] Nothing derived from `data` is cached across calls: n comes from the shape only.
- [x] Weights: n/a: no weight handling touched.
- [x] Activations dtype: n/a: only a launch parameter changed; outputs bitwise identical.
- [x] Attention: n/a: MoE align kernel only.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff contains none.
- [x] All work inside the timed call; no output caching or replay.
- [x] No triton.autotune; first-line marker `# okc: problem=muse_moe_w4a16` unchanged.

Lint: `OK with warnings: muse_moe_w4a16/submission.py` (only prepare()-side host-sync warnings, pre-existing and outside kernel_fn).

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading only (no GPU on this node). **Verdict: dispatch.** No fixes.

- The diff is one line: `num_warps=1 if n <= 64 else 4` on the `_align_small_kernel` launch. The kernel body is unchanged and contains nothing tied to the warp count; Triton reductions and scans give the same result under any layout. n = numel is shape-only, so there is no host sync and no value-dependent control flow, and the change is graph-capturable.
- Gates and fallback: no new gate. prepare() warm-up compiles the 1-warp variants for tokens 1..8 with kernel_fn's own block_size_m (the two-smaller pick applies only at >=128 tokens), so no variant compiles at capture or timing. As on master, `_align_self_check` only prints on a mismatch. It checks the 1-warp variants at tokens 1 and 8.
- Scope: matches the change paragraph. bs>=16 (n>=128) still launch exactly as on master. The n_regs/n_spills print and adding tokens=4 to the self-check were left out, as the implementation notes say. The bs=8 profile row still decides the result.
- Lint: OK with warnings (only pre-existing prepare()-side host syncs).

Residual risk: 1 warp at bs=8 ([64,32] temporaries, 64 int32 per thread each) may spill or serialize, making bs=8 flat or slower. bs=4 (n=32) is not self-checked. Neither affects correctness.

## Result

- job `38532ba10e394e3c` (submit), status done, passed True, tests 5/5
- speedup 1.3287x, 95% CI [1.3088, 1.3486], score 10121.477200687748 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-small-one-warp-tiny-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-small-one-warp-tiny-m.patch`; branch `hyp/muse_moe_w4a16/align-small-one-warp-tiny-m`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.82 | 49.07 | 0.9981 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 148.37 | 147.99 | 1.0009 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.88 | 200.17 | 1.0004 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.74 | 106.8 | 0.9938 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 318.26 | 323.2 | 1.0214 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 454.28 | 457.74 | 1.0027 | flat |

no shape moved outside noise. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-small-one-warp-tiny-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

