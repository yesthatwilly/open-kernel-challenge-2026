# muse_prefill_layer · 2026-10-24T233142Z · check · silu-mul-triton-local-rows-marlin-turn

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T165335Z-silu-mul-triton-local-rows-marlin-turn.md` (branch `hyp/muse_prefill_layer/silu-mul-triton-local-rows-marlin-turn`).

- **Job:** `b2259479e57e4196` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `d4d4c07`

## Goal
Built on master (635cfee). In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, turn_4 T=3012 and every test), replace silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1) (SGLang's act_and_mul_kernel) with a new Triton kernel _silu_mul_sorted_kernel that walks the moe_align_block_size output instead of every slot: grid = (cdiv(EM, BLOCK_R), size_n // BLOCK_N) with EM = sorted_ids.shape[0] (a static shape), BLOCK_R = 16 rows, BLOCK_N = 512 (size_n = 4096, so 8 column programs per row block), num_warps = 4. Each program loads n_padded from the device and returns if its first sorted position is >= n_padded; loads its BLOCK_R sorted ids, masks ids >= num_slots (T * 8, the align padding), and for each valid slot s reads cache1[s, c0:c0+BLOCK_N] (gate) and cache1[s, size_n + c0 : size_n + c0 + BLOCK_N] (up), computes silu(gate) * up in fp32 exactly as SGLang (gate * sigmoid(gate) * up, rounded once to bf16) and stores cache2[s, c0:c0+BLOCK_N]. Remote slots (local id -1) are never in sorted_ids' valid range (moe_align_block_size drops them, as Marlin's is_ep path relies on), so, exactly as today, their cache2 rows are left unwritten and nothing downstream reads them (Marlin w2 walks the same sorted_ids; fused_post_expert_norm_weighted_sum skips slot_expert_ids == -1). Before relying on that, assert in prepare() on a synthetic align (random local ids with -1s) that every slot with local id >= 0 appears exactly once in sorted_ids[:n_padded] and no slot with -1 does; if that does not hold (e.g. -1 routes are bucketed), mask by local_ids[s] >= 0 inside the kernel instead. prepare(): compile the kernel for both Marlin block sizes the turn shapes use and check it bit-for-bit against silu_and_mul on the local rows of a synthetic [T*8, 8192] cache1 at T=753 and T=3012. _experts_triton (T > 4096, first_turn) keeps SGLang's silu_and_mul unchanged so this does not confound experts-triton-w13-fused-silu-epilogue. Mechanism: act_and_mul_kernel costs 14.2 us at turn_1 and 52.9 us at turn_4 (master fb1e52d2119d4175) against a local-rows floor of ~3.9 us and ~15.4 us (T local rows x (8192 read + 4096 write) bf16 at 4.8 TB/s), a consistent ~3.5x gap (also 297 vs 84 us at first_turn): it launches over all T*8 slots and reaches only ~1.3 TB/s. Reading only sorted local rows with wide 2-D tiles should reach ~3+ TB/s.

## What changed, and where
Built in the main thread 2026-10-24 ~23:31Z (user-approved; reviewed "proceed" earlier, unheld to spend an expiring check). `_silu_mul_sorted_kernel`: grid (cdiv(EM, 16), size_n / 512), 4 warps; loads n_padded on device and skips blocks past it; per row the sorted slot id, masked to slots < T*8 AND local_ids[slot] >= 0. Deviation: rather than asserting in prepare() that SGLang's align never buckets remote routes, the kernel always masks on local_ids, which is correct either way. Math in fp32 as SGLang (g / (1 + exp(-g)) * u, one bf16 rounding); prepare() compares against silu_and_mul on the local rows of synthetic caches at T=753 and 3012 with a 2^-7 relative bound (exp may differ in the last fp32 bits, so not bit-exact) and falls back to silu_and_mul on any failure (`[silu-sorted] ON|OFF ... PATH CHOSEN`). Runs before `_check_experts_e2e`, so the end-to-end check covers it. `_experts_triton` (first_turn) unchanged.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: one Triton launch with a static grid; the n_padded early-out is in-kernel; no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16 (silu-mul output bf16 as SGLang's); KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (self-check in prepare()).
- [x] okc_lint: no errors.

## Risks / what could fail
low
Small and inside the headline CI (~0.7-0.8% on two of three loads); its value is a clean per-kernel row (the act_and_mul row should vanish, replaced by _silu_mul_sorted_kernel at <= ~40% of its time) and a stackable piece for a later turn-shape combo. Correctness risk is the sorted_ids contract for -1 routes under N_EXPERTS=256 with local ids 0-31: the prepare() assert settles it, with the local_ids mask as the fallback. Grid is a static bound (EM from the shape), with early exit past n_padded read on device: no host sync, graph-safe. bf16 in/out, fp32 math, the same as SGLang's kernel. Does not touch the Triton expert GEMM or the Marlin calls.

## Result (after the run)
- job `b2259479e57e4196` (check), status done, passed True, tests 5/5
- speedup 1.0512x, 95% CI [1.0354, 1.0669], score 368549.1830727498 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/silu-mul-triton-local-rows-marlin-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/silu-mul-triton-local-rows-marlin-turn.patch`; branch `hyp/muse_prefill_layer/silu-mul-triton-local-rows-marlin-turn`
