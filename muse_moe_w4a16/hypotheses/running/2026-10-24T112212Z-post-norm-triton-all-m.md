---
problem: muse_moe_w4a16
slug: post-norm-triton-all-m
created: 2026-10-24T112212Z
base: master
area: post-expert norm+combine
change: >
  On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the
  end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too) combines with master's
  _post_norm_combine_kernel instead of SGLang's fused_post_expert_norm_weighted_sum. The kernel
  itself is unchanged: grid (num_tokens,), num_warps=8, BLOCK_H = next_power_of_2(H) = 4096, one
  program per token looping over its 8 slots with tl.static_range, skipping remote slots (local id
  -1) without loading them, fp32 math, (1 + w) applied in-kernel from the bf16 post_norm_w, bf16
  store into a fresh [M, H] output. y.contiguous() stays (a no-op on Marlin's contiguous output).
  state['norm_w'] and the fused_post_expert_norm_weighted_sum import become unused; leave them or
  delete them, nothing else changes (block_size_m picks, align, both Marlin calls, silu_and_mul as
  in the dependency).
expected:
  - shapes: bs=128 *
    direction: faster
    why: SGLang _post_expert_norm_weighted_sum_kernel is 6.1 us at bs=128 while _post_norm_combine_kernel is 3.1-3.2 us at bs=8/16 (64-128 slots) and is L2-resident on w2 output; expect ~2 us saved, ~0.6% of the ~345 us restacked shape, readable in top_kernels rather than the headline
  - shapes: bs=256 *
    direction: faster
    why: "SGLang kernel 5.2 us; ours ~3.5-4.5 us for 256 programs: ~1 us, ~0.3% of ~470 us"
  - shapes: bs=32 *
    direction: flat
    why: "SGLang kernel 3.7 us vs ours ~3.3 us: under drift, but bs=32 is the weakest shape and every us there counts"
  - shapes: bs=1 *
    direction: flat
    why: "control: same kernel as master"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
risk: low
depends_on: marlin-block-m-two-smaller-on-postnorm
priority: 4
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T124248Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T124248Z
base_sha: a98ef454e7901f7ff47efb4c4c0e77e7261da827
base_blob: c7bf9ffedc48b757f56765004e5d65f6ad5ba841
branch: hyp/muse_moe_w4a16/post-norm-triton-all-m
worktree: .worktrees/muse_moe_w4a16/post-norm-triton-all-m
mode: null
---

# post-norm-triton-all-m

## Change

On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too) combines with master's _post_norm_combine_kernel instead of SGLang's fused_post_expert_norm_weighted_sum. The kernel itself is unchanged: grid (num_tokens,), num_warps=8, BLOCK_H = next_power_of_2(H) = 4096, one program per token looping over its 8 slots with tl.static_range, skipping remote slots (local id -1) without loading them, fp32 math, (1 + w) applied in-kernel from the bf16 post_norm_w, bf16 store into a fresh [M, H] output. y.contiguous() stays (a no-op on Marlin's contiguous output). state['norm_w'] and the fused_post_expert_norm_weighted_sum import become unused; leave them or delete them, nothing else changes (block_size_m picks, align, both Marlin calls, silu_and_mul as in the dependency).

## Expected

- `bs=128 *` faster: SGLang _post_expert_norm_weighted_sum_kernel is 6.1 us at bs=128 while _post_norm_combine_kernel is 3.1-3.2 us at bs=8/16 (64-128 slots) and is L2-resident on w2 output; expect ~2 us saved, ~0.6% of the ~345 us restacked shape, readable in top_kernels rather than the headline
- `bs=256 *` faster: SGLang kernel 5.2 us; ours ~3.5-4.5 us for 256 programs: ~1 us, ~0.3% of ~470 us
- `bs=32 *` flat: SGLang kernel 3.7 us vs ours ~3.3 us: under drift, but bs=32 is the weakest shape and every us there counts
- `bs=1 *` flat: control: same kernel as master
- `bs=8 *` flat: control
- `bs=16 *` flat: control

## Risk

Expected effect is small (~0.5-2 us per shape, well under the ~1.5% geomean resolution in LESSONS): it is a cleanup-and-measure run whose verdict should come from the replaced kernel's own time (SGLang _post_expert_norm_weighted_sum_kernel 3.7/6.1/5.2 us at bs=32/128/256 vs _post_norm_combine_kernel on the same shapes), not the headline. Run it after the restack so the bs>=128 shapes are smaller and the combine is a larger share. Risk: our kernel's per-token serial 8-row loop could lose at bs=256 if 256 programs x 8 dependent 8 KB loads underfill bandwidth; if _post_norm_combine_kernel exceeds SGLang's time at bs>=128 the right follow-up is the threshold at 128, not a rewrite. Correctness: same kernel already passes the 5 tests at bs<32; bs=257 just adds programs.

## Adversarial review

Reviewed 2026-10-24T124248Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a low-risk cleanup that master has not made moot. Master (cefc282ebd) still calls SGLang's fused_post_expert_norm_weighted_sum at bs>=32, at 4.2/6.4/5.5 us for bs=32/128/256. Swapping in the already-validated _post_norm_combine_kernel can be read from top_kernels without companions, even though the headline effect (<1%) is below drift.

### Justification
The mechanism is plausible. Master's profile (c00ef7295c154867) shows _post_norm_combine_kernel almost flat in M: 3.5 us at bs=1, 3.3 at bs=8 and 3.5 at bs=16 (2.8/3.1/3.2 in bacd1f502f4e46fa). That means it is latency-bound per program (8 dependent slot loads behind a scalar branch), not bandwidth-bound. At bs=128 the grid is 128 programs on 132 SMs, roughly one wave, so the expected time is close to the bs=1 single-program latency, about 3-4 us, against SGLang's 6.4 us. At bs=256 two 8-warp programs per SM should co-reside (about 32 fp32 regs/thread for acc+v at BLOCK_H=4096/256 threads), so expect about 3.5-4.5 us against 5.5 us. At bs=32 (4.2 us vs about 3.3 us) the saving is under 1 us. The data is small and L2-resident: at most 16 MB of y at bs=256 if every slot were local, realistically a few MB, just written by w2 Marlin into a 50 MB L2. The floors (101/109 us at verify) are dominated by Marlin weight streaming, so the combine is about 1-2% of each shape and the predicted 0.3-0.6% per shape is consistent. The expected directions fit TASK.md's equal shape weights. bs=1/8/16 run byte-identical code and are an honest in-run control. Weakest link: the bs=256 estimate rests on two assumptions, that latency stays per program and does not grow with grid size, and that one token's 8 serial 8 KB row loads do not underfill bandwidth. The bs=128 SGLang time (6.4 us) exceeding its bs=256 time (5.5 us) shows its cost is not simple in M either, so bs=256 is the shape where ours could tie or lose. The second weak point is resolvability: per LESSONS, a <1.5% effect does not show on the geomean. The verdict must come from the replaced kernel's own top_kernels time (Triton reproduces within about 1%) and from shape time minus kernel sum, not from the headline or the per-shape paired ratio alone. A side benefit the hypothesis does not count: dropping SGLang's Python wrapper at bs>=32 may remove allocation or other graph nodes, which shows up as a lower (shape time - kernel sum) gap.

### Confounders on the current kernel
- post-expert-norm-triton-small-m (192f225d8f): it supplies the exact kernel being extended, so it is the base, not a confounder. Its launch config (grid (M,), num_warps=8, one program per token, serial static_range over 8 slots) was chosen for M<32 and never tuned for M=128-257. A flat or loss at bs>=128 would indict that config (per-token serialism), not the idea of replacing SGLang's kernel. The hypothesis's own fallback (threshold at 128) handles that. A split-slot or split-H grid is a later ab-tune, not needed to read this run.
- marlin-block-m-two-smaller-on-postnorm (cefc282ebd): it changes block_size_m at bs>=128 (48->16, 64->32), which changes Marlin's internal tiling and runtime but not y's layout ([M*8, H] contiguous, remote rows uninitialised) or its L2 residency. The combine reads y after w2 completes, so no overlap interaction is possible. Its only effect is to make the combine a larger share of bs=128/256 (about 340/467 us instead of about 476/622 us), which slightly helps resolvability. Not a confounder.
- sglang-marlin-path-parity (a646cb4151): it provides y and local_ids (expert_map-mapped, -1 for remote) that both combine kernels consume identically. Not a confounder.

### Interference and A/B needs
No companion is warranted. The change replaces one kernel with another whose own time appears in the same profile row slot, so both outcomes are readable from one run. A gain shows as _post_norm_combine_kernel at bs>=32 below SGLang's 4.2/6.4/5.5 us in c00ef7295c154867. A loss shows as it above them, and the fix is the threshold at 128 the hypothesis already names. Reverting a landed hypothesis has no meaning here: reverting post-expert-norm-triton-small-m removes the kernel under test, and reverting the block-m change would only shrink the combine's share. A threshold sub-change control (e.g. keep SGLang at bs=256 only) would split a sub-1% effect that cannot be resolved per shape anyway. If bs=128 and bs=256 diverge in sign, the kernel times already say which threshold to use. bs=1/8/16 are unchanged code and serve as the drift control.

### Open questions
- top_kernels in result.json lists only 4 rows. At bs=256, index_elementwise (3.1 us) is 4th, so a ~3 us _post_norm_combine_kernel may drop out of the JSON. Read its bs=32/128/256 time from the full profiles/submission.<job>.txt and record it in the attempt file next to SGLang's 4.2/6.4/5.5 us.
- Record (shape time - sum of kernel times) at bs>=32 before and after. If it shrinks by more than the kernel delta, SGLang's wrapper was adding graph nodes beyond its Triton kernel.
- If _post_norm_combine_kernel is slower than SGLang's at bs=256 but faster at bs=128/32, the follow-up is SMALL_M_POST_NORM = 257 (or a bs<256 gate), not a redesign. If it ties everywhere, keep it only for code simplicity, since it would land as neutral.
- Correctness at the new shapes: the bs=33, bs=257 and skewed bs=128 tests now exercise this kernel. M=257 gives a non-multiple-of-anything grid, which is fine for one program per token. Confirm 5/5 pass and note the max row error versus SGLang's bf16 (1+w) path (ours is fp32 (1+w), expected equal or better).

