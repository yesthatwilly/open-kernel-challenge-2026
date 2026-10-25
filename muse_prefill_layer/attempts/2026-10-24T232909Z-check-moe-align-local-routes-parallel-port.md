# muse_prefill_layer · 2026-10-24T232909Z · check · moe-align-local-routes-parallel-port

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T184852Z-moe-align-local-routes-parallel-port.md` (branch `hyp/muse_prefill_layer/moe-align-local-routes-parallel-port`).

- **Job:** `7e91d9ee93144387` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `67f7473`

## Goal
Built on master (635cfee). Port muse_moe_w4a16's measured align (master there: _align_parallel_kernel and its host wrapper _align_parallel in muse_moe_w4a16/submission.py, hypothesis align-parallel-per-expert-verify, +3-4% per shape at bs 128/256 there) to prefill at every T, replacing two things in muse_prefill_layer/submission.py: the _local_expert_ids remap launch in kernel_fn and the moe_align_block_size(local_ids, block, N_EXPERTS) call in both _experts_marlin (block_size_m from master's loop, 64 on every prefill shape and test) and _experts_triton (cfg['BLOCK_T'], 256 at first_turn). Copy the kernel and wrapper verbatim except one change: prefill's local_start is an input tensor (data['local_start'], int32 [1], on device), so pass the tensor and read it in the kernel with `local_start = tl.load(start_ptr)` instead of moe's host int argument (no .item(), nothing cached); local_count = LOCAL_EXPERTS (32) as a host int. Restructure the call sites: kernel_fn passes topk_ids and data['local_start'] to _experts; each _experts_* path picks its block size as today and calls _align_parallel(topk_ids, local_start, 32, block), which returns local_ids [T, 8] int32 (0..31, -1 remote: same values as today's _local_expert_ids) and (sorted_ids, block_experts, n_padded) over the 32 local experts only, capacity T*8 + 32*(block-1), sentinel T*8 in padding slots, -1 in block_experts only past the used blocks. local_ids keeps feeding silu_and_mul(expert_ids=...) and fused_post_expert_norm_weighted_sum(slot_expert_ids=...) exactly as now. MECHANISM: SGLang's align buckets the remote routes (7/8 of all routes) into m-blocks of their own (muse_moe_w4a16 self-check logs: 'all-remote: n_pad=0 r_pad=64'). At turn_1 that is ~83 of ~120 Marlin m-blocks, at turn_4 ~330 of ~395, at first_turn ~448 of ~530 Triton token tiles, where w2 zero-stores ~940 MB. With local routes only, Marlin's block list is all real work, the Triton w2 call writes nothing for remote slots, and the count_and_sort + moe_align kernels (5.2/17.4/87 us + 50 us at first_turn) plus the remap launch become one launch. Remote rows of the w2 output are then never written; SGLang's fused_post_expert_norm_weighted_sum skips slot_expert_ids == -1 (muse_moe_w4a16 master relies on exactly this at bs >= 32). prepare() checks, each printing one line, with fallback to master's path for the run on any failure ('[align-local] ON' / 'OFF: <reason>'): (1) per-expert route sets and block counts equal moe_align_block_size's (ignoring its -1 blocks) on synthetic routings at T = 200, 753, 3012, 16384 with block 64 and 256 and local_start 0 and 224; (2) NaN-poison: fused_post_expert_norm_weighted_sum on a y whose remote slots are NaN equals the call with those slots zero; (3) _check_experts_e2e, adapted to take global topk_ids plus a local_start tensor, still passes. Marlin keeps is_ep=True; nothing else changes.

## What changed, and where
Built in the main thread 2026-10-24 ~23:30Z with user-approved self-review (no new agents before the reset). DEVIATION, scoped down for time and risk:
- Marlin path only (T <= MARLIN_MAX_TOKENS: turn_1, turn_4, every test). `_experts_triton` (first_turn, 131K routes) keeps master's align: the kernel's per-program chunk loop is serial in n, unmeasured at that size.
- The `_local_expert_ids` remap launch is kept; `_align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm)` takes its output (ids already 0-31 / -1), so local_start stays on device with no kernel change. The remap fusion is left for a follow-up.
- Kernel, wrapper and `_align_par_ref_view` copied unchanged from the decode port audited tonight (moe-align-parallel-large-t-port, 1.2433 on the judge); new `_align_par_setup` in prepare(): warms every BLOCK_SIZE_M_CANDIDATES variant, compares per-expert route sets and block counts with moe_align_block_size(local_ids, bsm, N_EXPERTS) at T = 9/200/753/3012/4096 x random/skewed/all-remote, sets state["align_par_ok"], prints `[align-local] ON|OFF ... PATH CHOSEN`. It runs before `_check_experts_e2e`, which then checks the Marlin path end to end against the Triton path with the chosen align.
- Remote rows of the Marlin outputs are no longer written; silu_and_mul(expert_ids=local_ids) and fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids) skip -1 slots, as muse_moe_w4a16 master relies on.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: allocations are shape-only, no host sync or value-dependent control flow in kernel_fn (the gate is a prepare()-set bool).
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (self-check in prepare()).
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Port of a measured muse_moe_w4a16 win; touches only the dispatch, not either expert engine, so it composes with experts-cuda-wgmma-mixed-input-grouped-gemm (its stage 1 keeps moe_align_block_size at 256 and still zero-stores -1 items; this removes them at the source). Read the Marlin rows at turn_1/turn_4 to settle whether remote m-blocks cost Marlin anything. Risks: _align_parallel streams all T*8 ids twice in each of 33 programs; at first_turn (131072 ids) that may take ~50-80 us, not the ~5 us it took at bs=256 (judge on its own row against count_and_sort + moe_align); the sorted order within an expert differs from SGLang's (ascending route index), harmless for Marlin and the Triton kernel.

## Result (after the run)
- job `7e91d9ee93144387` (check), status done, passed True, tests 5/5
- speedup 1.0658x, 95% CI [1.0498, 1.0817], score 363501.1905121105 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-align-local-routes-parallel-port.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-align-local-routes-parallel-port.patch`; branch `hyp/muse_prefill_layer/moe-align-local-routes-parallel-port`
