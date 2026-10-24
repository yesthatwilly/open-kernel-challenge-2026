---
problem: muse_prefill_layer
slug: silu-mul-triton-local-rows-marlin-turn
created: 2026-10-24T165335Z
base: master
area: silu-mul (Marlin path, T <= 4096)
change: >
  Built on master (635cfee). In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS =
  4096: turn_1 T=753, turn_4 T=3012 and every test), replace silu_and_mul(cache1.view(-1,
  gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) (SGLang's act_and_mul_kernel)
  with a new Triton kernel _silu_mul_sorted_kernel that walks the moe_align_block_size output
  instead of every slot: grid = (cdiv(EM, BLOCK_R), size_n // BLOCK_N) with EM =
  sorted_ids.shape[0] (a static shape), BLOCK_R = 16 rows, BLOCK_N = 512 (size_n = 4096, so 8
  column programs per row block), num_warps = 4. Each program loads n_padded from the device and
  returns if its first sorted position is >= n_padded; loads its BLOCK_R sorted ids, masks ids >=
  num_slots (T * 8, the align padding), and for each valid slot s reads cache1[s, c0:c0+BLOCK_N]
  (gate) and cache1[s, size_n + c0 : size_n + c0 + BLOCK_N] (up), computes silu(gate) * up in fp32
  exactly as SGLang (gate * sigmoid(gate) * up, rounded once to bf16) and stores cache2[s,
  c0:c0+BLOCK_N]. Remote slots (local id -1) are never in sorted_ids' valid range
  (moe_align_block_size drops them, as Marlin's is_ep path relies on), so, exactly as today, their
  cache2 rows are left unwritten and nothing downstream reads them (Marlin w2 walks the same
  sorted_ids; fused_post_expert_norm_weighted_sum skips slot_expert_ids == -1). Before relying on
  that, assert in prepare() on a synthetic align (random local ids with -1s) that every slot with
  local id >= 0 appears exactly once in sorted_ids[:n_padded] and no slot with -1 does; if that
  does not hold (e.g. -1 routes are bucketed), mask by local_ids[s] >= 0 inside the kernel
  instead. prepare(): compile the kernel for both Marlin block sizes the turn shapes use and check
  it bit-for-bit against silu_and_mul on the local rows of a synthetic [T*8, 8192] cache1 at T=753
  and T=3012. _experts_triton (T > 4096, first_turn) keeps SGLang's silu_and_mul unchanged so this
  does not confound experts-triton-w13-fused-silu-epilogue. Mechanism: act_and_mul_kernel costs
  14.2 us at turn_1 and 52.9 us at turn_4 (master fb1e52d2119d4175) against a local-rows floor of
  ~3.9 us and ~15.4 us (T local rows x (8192 read + 4096 write) bf16 at 4.8 TB/s), a consistent
  ~3.5x gap (also 297 vs 84 us at first_turn): it launches over all T*8 slots and reaches only
  ~1.3 TB/s. Reading only sorted local rows with wide 2-D tiles should reach ~3+ TB/s.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "act_and_mul_kernel 14.2-14.7 us -> ~5-6 us (local rows only, ~3 TB/s): ~8-9 us of ~1080-1420 us (~0.7%); judge on the silu row, not the headline"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "act_and_mul_kernel 52.2-52.9 us -> ~20 us: ~30 us of 3335-4938 us (~0.8%)"
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 runs _experts_triton, untouched (byte-identical)
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T165654Z
review_companions: null
hold_reason: null
---

# silu-mul-triton-local-rows-marlin-turn

## Change

Built on master (635cfee). In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, turn_4 T=3012 and every test), replace silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) (SGLang's act_and_mul_kernel) with a new Triton kernel _silu_mul_sorted_kernel that walks the moe_align_block_size output instead of every slot: grid = (cdiv(EM, BLOCK_R), size_n // BLOCK_N) with EM = sorted_ids.shape[0] (a static shape), BLOCK_R = 16 rows, BLOCK_N = 512 (size_n = 4096, so 8 column programs per row block), num_warps = 4. Each program loads n_padded from the device and returns if its first sorted position is >= n_padded; loads its BLOCK_R sorted ids, masks ids >= num_slots (T * 8, the align padding), and for each valid slot s reads cache1[s, c0:c0+BLOCK_N] (gate) and cache1[s, size_n + c0 : size_n + c0 + BLOCK_N] (up), computes silu(gate) * up in fp32 exactly as SGLang (gate * sigmoid(gate) * up, rounded once to bf16) and stores cache2[s, c0:c0+BLOCK_N]. Remote slots (local id -1) are never in sorted_ids' valid range (moe_align_block_size drops them, as Marlin's is_ep path relies on), so, exactly as today, their cache2 rows are left unwritten and nothing downstream reads them (Marlin w2 walks the same sorted_ids; fused_post_expert_norm_weighted_sum skips slot_expert_ids == -1). Before relying on that, assert in prepare() on a synthetic align (random local ids with -1s) that every slot with local id >= 0 appears exactly once in sorted_ids[:n_padded] and no slot with -1 does; if that does not hold (e.g. -1 routes are bucketed), mask by local_ids[s] >= 0 inside the kernel instead. prepare(): compile the kernel for both Marlin block sizes the turn shapes use and check it bit-for-bit against silu_and_mul on the local rows of a synthetic [T*8, 8192] cache1 at T=753 and T=3012. _experts_triton (T > 4096, first_turn) keeps SGLang's silu_and_mul unchanged so this does not confound experts-triton-w13-fused-silu-epilogue. Mechanism: act_and_mul_kernel costs 14.2 us at turn_1 and 52.9 us at turn_4 (master fb1e52d2119d4175) against a local-rows floor of ~3.9 us and ~15.4 us (T local rows x (8192 read + 4096 write) bf16 at 4.8 TB/s), a consistent ~3.5x gap (also 297 vs 84 us at first_turn): it launches over all T*8 slots and reaches only ~1.3 TB/s. Reading only sorted local rows with wide 2-D tiles should reach ~3+ TB/s.

## Expected

- `*load=turn_1*` faster: act_and_mul_kernel 14.2-14.7 us -> ~5-6 us (local rows only, ~3 TB/s): ~8-9 us of ~1080-1420 us (~0.7%); judge on the silu row, not the headline
- `*load=turn_4*` faster: act_and_mul_kernel 52.2-52.9 us -> ~20 us: ~30 us of 3335-4938 us (~0.8%)
- `*load=first_turn*` flat: T=16384 runs _experts_triton, untouched (byte-identical)

## Risk

Small and inside the headline CI (~0.7-0.8% on two of three loads); its value is a clean per-kernel row (the act_and_mul row should vanish, replaced by _silu_mul_sorted_kernel at <= ~40% of its time) and a stackable piece for a later turn-shape combo. Correctness risk is the sorted_ids contract for -1 routes under N_EXPERTS=256 with local ids 0-31: the prepare() assert settles it, with the local_ids mask as the fallback. Grid is a static bound (EM from the shape), with early exit past n_padded read on device: no host sync, graph-safe. bf16 in/out, fp32 math, the same as SGLang's kernel. Does not touch the Triton expert GEMM or the Marlin calls.

## Adversarial review

Reviewed 2026-10-24T165654Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, low-risk, single-kernel change whose act_and_mul row (14.2/14.7/52.9/52.2 us in fb1e52d2119d4175) can be read directly from the profile. The weakest link is its central premise, which is contradicted by master's own code: -1 routes ARE bucketed into sorted_ids[:n_padded] (in blocks whose block_experts entry is -1), so the 'fallback' mask is the real path, and the mechanism is wider tiles plus sorted-order L2 locality, not skipping remote slots that SGLang's kernel already skips.

### Justification
Profile check: act_and_mul_kernel is 14.2 us (turn_1 sliding), 14.7 (turn_1 full), 52.9 (turn_4 sliding), 52.2 (turn_4 full), 1.0-1.5% of each shape. It scales ~3.7x for 4x the tokens, so the cost is per-row work and not launch overhead. The local-row floor (~T rows x 24.6 KB at 4.8 TB/s = ~3.9 / ~15.4 us) is computed correctly, so SGLang's kernel runs at ~1.3 TB/s and a 2-D-tiled Triton elementwise kernel reaching 2.5-3.5 TB/s is plausible on H200. cache1 was just written by Marlin w13 (~12 MB of local rows at turn_1), so part of the read may come from L2, which makes ~5-6 us reachable. Expected directions match TASK.md: turn_1/turn_4 faster by ~0.7-0.8% of shape time, first_turn flat (T=16384 > MARLIN_MAX_TOKENS goes to _experts_triton, untouched). The hypothesis says honestly that this is inside the +-1.5% headline CI and must be judged on the kernel row. WEAKEST LINK: the claim that remote slots never appear in sorted_ids' valid range is contradicted by submission.py itself. _moe_nvfp4_gemm_kernel has an 'off_expert == -1' branch, and ZERO_NONLOCAL_W2's comment (line 181) says 'the m-blocks of non-local routes (expert -1)'. Marlin's is_ep path skips those blocks; it does not rely on them being absent. So the prepare() assert as written will fail and the local_ids-mask fallback is what gets built. Make that the primary design, not a fallback. Better still, exit early per row tile when block_experts[pos // block_size_m] == -1. In practice block_size_m is always 64 at every prefill T (the candidate loop never breaks before 64 for T >= 200: T*8/32/64 >= 0.78 at T=200, and it breaks at 64 or ends there). BLOCK_R=16 divides 64, so one block_experts load per program skips whole remote tiles without a per-row gather of local_ids. The consequence for the mechanism: SGLang's act_and_mul (expert_ids=local_ids, expert_step=1) already skips -1 rows, so 'it launches over all T*8 slots' is not what separates the two kernels. The new grid also covers ~T*8 + padding positions, most of them remote. The gain has to come from tile shape and occupancy (16 rows x 512 cols per CTA instead of SGLang's one row per block) and from sorted (expert-major) order matching Marlin's write order for L2 hits. Both are plausible but unmeasured, so the ~5-6 us target may come in nearer 8-10 us. Minor points: 'compile for both Marlin block sizes the turn shapes use' is a single size (64); a bit-for-bit check against CUDA expf-based silu may differ by 1 bf16 ulp under Triton's exp/sigmoid lowering, so allow <= 1 ulp (or compute exactly 1/(1+exp(-x)) in fp32 the way SGLang does) rather than hard-failing prepare().

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn: this sets the T<=4096 gate that makes _experts_marlin the only path the change touches. The change is confined to that branch, so the gate scopes the result rather than confounding it. With the gate reverted (all-Triton), the change would not run at the turn shapes at all.
- experts-triton-w4a16-first-turn-blockt256 / swapab stages: _experts_triton (T>4096) keeps SGLang's silu_and_mul byte-identical, so the first_turn shapes are an in-run null control. No interaction.
- sglang-pipeline-parity (routing remap to local_ids with -1): this defines the -1 contract the new kernel depends on. It does not confound the timing, but it is why the remote-slot handling must be a mask or block-level -1 exit and not an assumption.

### Interference and A/B needs
No companion is warranted. The result can be read directly: the act_and_mul_kernel row disappears and _silu_mul_sorted_kernel replaces it in the same profile, at the same shapes, against the same Marlin producer and consumer. The first_turn shapes are an untouched in-run control for drift. A gain or loss on the kernel row is attributable without an A/B. The headline can't show it at all (~0.7% vs +-1.5% CI), and no companion would change that. One caveat for whoever dispatches: prefill checks are spent and this would go out as a scored submit whose headline cannot move outside noise. Its value is the per-kernel row and a stackable turn-shape piece, so consider batching it with another turn-shape change, or running it as a check after the reset.

### Open questions
- Does SGLang's moe_align_block_size with N_EXPERTS=256 and -1 ids place the -1 routes in contiguous trailing blocks (so a tile-level exit removes them almost for free), or scatter them? This decides whether the walk costs ~T*8 positions or ~T.
- How much of cache1's local rows is still L2-resident after Marlin w13 at turn_4 (~49 MB of local rows, about L2 size)? This sets whether turn_4 can reach ~20 us or stays nearer 30 us.
- Is the prepare() reference check bit-exact achievable given CUDA expf vs Triton exp lowering, or should it accept 1 bf16 ulp?

