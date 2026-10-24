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
review: required
status: queued
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
