---
problem: muse_prefill_layer
slug: qk-norm-fp8-cache-store-fusion
created: 2026-10-23T230250Z
base: master
area: pre-attention norm+rope + cache store
change: >
  One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8),
  v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q
  and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM
  output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with
  eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on
  sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin
  table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1)
  exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching
  parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k
  rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1
  (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns
  1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of
  v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3
  call stays the parity path. Before submitting, compare against the parity kernels on random T=64
  inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP
  FP8 differences from fp32 summation order in the RMS.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters"
  - shapes: "*load=turn_4*"
    direction: flat
    why: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
  - shapes: "*load=first_turn*"
    direction: flat
    why: ~0.1% of 17-26 ms
risk: medium
depends_on: sglang-pipeline-parity
priority: 4
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T071251Z
review_companions: null
---

# qk-norm-fp8-cache-store-fusion

## Change

One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1) exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1 (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns 1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3 call stays the parity path. Before submitting, compare against the parity kernels on random T=64 inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP FP8 differences from fp32 summation order in the RMS.

## Expected

- `*load=turn_1*` faster: four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters
- `*load=turn_4*` flat: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
- `*load=first_turn*` flat: ~0.1% of 17-26 ms

## Risk

Medium risk on exactness, low on performance. The cache-entry bound is 8% relative and the new entries are FP8-rounded in both paths, so a one-ULP FP8 flip on a few lanes is harmless; a RoPE pairing or cos/sin layout mistake would fail the first sliding test. Torch's .to(float8_e4m3fn) and Triton's .to(tl.float8e4nv) both round to nearest even without saturation; q and k are RMS-normalized so no lane can overflow, v comes from the same bf16 GEMM output in both paths. Rules: queries FP8 as FA3 does, cache FP8, nothing cached, no host sync. The gain is small by construction; a flat result on turn_1 is the honest outcome and attributes to launch overhead not mattering at M=753.

## Adversarial review

Reviewed 2026-10-24T071251Z by an adversarial subagent (model opus).

**Verdict: proceed.** Correct and within the rules, but the expected gain is about 0.3% on the geomean against a CI of at least +/-1.5%, so a neutral headline is what the run will most likely show. Run it once with no companions, and judge it from the profile's kernel table (the five replaced kernels against the one fused kernel), not from the speedup alone.

### Justification
Mechanism: on master the attention front runs five launches between the qkv GEMM and FA3 in the scored (kv_scale == 1) path: FusedQKNorm, k.to(FP8), v.to(FP8) on a strided slice of qkv, store_cache and q.view(T,16,64).to(FP8). Fusing them into one Triton kernel is plausible, and it is the same shape of fusion that gave attention_decode 1.127x. Size of the prize: at T=753 these kernels move only about 1.7 MB of q/k plus 0.2 MB of v in and about 1 MB of FP8 out. That is roughly 1 us of DRAM time, so nearly all the saving is launch and inter-kernel gaps inside the graph, about 1.5-3 us per kernel removed, or 5-10 us per shape. The hypothesis says 10-15 us, which is on the optimistic side. FINDINGS gives only the top three kernels per shape for master (Marlin MoE 45-66%, the nvjet GEMMs, FA3 on full layers). The only full per-kernel profile on disk (8a2df1fde4f54a7e) is the starter calibration and does not apply. So no measured number backs the 10-15 us. Direction and shapes: the expected directions agree with TASK.md. turn_1 (46x sliding at ~1080 us, 16x full at ~1435 us) gains about 0.5-1% on its load, turn_4 about 0.2%, first_turn about 0.05%. Each load counts equally in the geomean, so the problem-level gain is about 0.25-0.4%, well inside the +/-1.5% interval. Even turn_1's per-shape gain sits at or under the per-shape noise. The sister decode_layer cache-store-fused-fp8-cast was neutral on every shape, which fits launch savings not showing at this scale. Weakest link: the payoff is close to the resolution floor of the measurement. Exact numerical parity is the other weak spot: the hypothesis assumes FusedQKNorm applies norm and then RoPE in fp32 with a single bf16 rounding at the end. If FusedQKNorm rounds to bf16 between norm and rope, the fused kernel will differ by one ULP in places. That is harmless under the 8% cache bound and the attention bound, but it would break the pre-submit 'bitwise equal' check the change itself specifies, so read the FusedQKNorm source before trusting that check.

### Confounders on the current kernel
- sglang-pipeline-parity (the only landed hypothesis): this change replaces part of its attention front, so the interaction is the change itself, not a confounder. Nothing else landed touches the pre-FA3 launch sequence. moe-align-side-stream is neutral and not on master, and its side-stream fork comes after attention, so it cannot overlap or compete with the new kernel.

### Interference and A/B needs
No companion is warranted. The problem is resolution, not attribution. An ab-revert has nothing to revert: the only landed item is the base whose kernels are being replaced. An ab-tune of the parity path would not change these five launches. A control that isolates one sub-change, such as fusing only the casts and store_cache while keeping FusedQKNorm, splits an effect of about 0.3% that is already below the interval into two smaller ones. Both arms would come back neutral, which spends quota without making the result easier to read. The result can be read without companions as follows. In the okc profile for the run, compare the new kernel's time and the attention-front span against the five parity kernels' summed time on the turn_1 shapes. If the fused kernel is shorter by a few us, the mechanism is confirmed even with a flat headline. If the fused kernel is slower than FusedQKNorm alone (per-row programs with poor occupancy at T=753), that explains a regression. Grab the parity kernel times from master's profile, or from the parent job's .txt if it has more than the top three, so the comparison is like for like.

### Open questions
- Does FusedQKNorm round to bf16 once after norm+rope, or once after norm and again after rope? This decides whether the pre-submit check can be bitwise.
- What are master's actual per-kernel times for FusedQKNorm, the three casts and store_cache at T=753? FINDINGS lists only the top three kernels, so the 10-15 us estimate has no measurement behind it.
- fa3-contiguous-kv-gather (in flight) changes how the FA3 call reads the cache. If it lands first, this change needs a rebase so the fused store feeds whatever layout that branch expects, and the two should not be credited together without a rerun.

