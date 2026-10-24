# muse_moe_w4a16 · 2026-10-24T165402Z · check · w2-l2-prefetch-decode1

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T164836Z-w2-l2-prefetch-decode1.md` (branch `hyp/muse_moe_w4a16/w2-l2-prefetch-decode1`).

- **Job:** `ac6897f284934dad` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `f47176b`

## Goal
At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the w2 weights of this token's local experts before the w13 Marlin call, so the w2 moe_wna16_marlin_gemm reads them from L2 instead of DRAM. TASK.md: every timed replay starts with a cold L2 (the judge streams a buffer larger than L2 first), and at bs=1 the two Marlin calls take 44.6 us for roughly one local expert (~28 MB of w13+w2 codes and scales), i.e. ~0.6 TB/s: latency-bound, not bandwidth-bound. Add a Triton kernel _l2_prefetch_w2_kernel(topk_ids_ptr, w2_ptr, w2s_ptr, sink_ptr, local_start, local_count, w2_expert_words, w2s_expert_words, TOP_K_C: tl.constexpr, BLOCK: tl.constexpr), grid (P,) with P = 2 * SM count (264), num_warps 4, BLOCK = 2048 int32 words. Each program loads the 8 topk_ids, maps them to local ids (id - local_start in [0, local_count), else skip), skips a slot whose local id equals an earlier slot's (dedupe, static_range over the 8 slots), and for each remaining local expert e loops over its own strided share of that expert's w2 slab (int32 view of bank["w2"][e], w2_expert_words = bank["w2"][0].numel() * element_size / 4) and of its scale slab (int32 view of bank["w2_s"][e]) with tl.load(..., eviction_policy="evict_last"), XOR-accumulating the words into one int32, and finally stores that int32 to sink_ptr[program_id] (sink = a torch.empty(P, int32) made once in prepare and never read; it only stops the loads being eliminated). The int32 views and per-expert word counts are built once in prepare() from the bank tensors (both are stacked [32, ...] contiguous banks; assert it). kernel_fn at num_tokens == 1 launches it on the current stream right after _align_small and before the w13 Marlin call; nothing else changes (same block_size_m, align, Marlin flags, silu_and_mul, _post_norm_combine_kernel). Only w2 is prefetched (~9.4 MB per local expert, at most ~19 MB for the common 1-2 local experts, well under H200's 50 MB L2, and it must survive w13's ~19 MB stream through L2, which evict_last favours). prepare() compiles the kernel once on dummy ids. A token with no local expert prefetches nothing.

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/w2-l2-prefetch-decode1`, `muse_moe_w4a16/submission.py` only (82 insertions, 0 deletions).

- New `@triton.jit(do_not_specialize=["local_start", "local_count"]) _l2_prefetch_w2_kernel(topk_ids_ptr, w2_ptr, w2s_ptr, sink_ptr, local_start, local_count, w2_expert_words, w2s_expert_words, TOP_K_C, BLOCK)` as specified: static_range over the 8 slots, local id = id - local_start in [0, local_count), dedupe against every earlier slot (nested static_range), then for each distinct local expert a grid-strided loop (`range(pid, cdiv(words, BLOCK), num_programs)`) over BLOCK=2048-word chunks of its w2 code slab and then its w2 scale slab, `tl.load(..., eviction_policy="evict_last")`, masked on the tail, int64 offsets, XOR into an int32 [BLOCK] vector; `tl.xor_sum` of it stored to `sink[pid]`.
- `_l2_prefetch_state(bank, device)` (prepare only): asserts `bank["w2"]` and `bank["w2_s"]` are contiguous with leading dim == count (32) and per-expert bytes divisible by 4; builds the flat int32 views, per-expert word counts, P = 2 * SM count programs, and the `sink` (torch.empty(P, int32), never read). Stored as `bank["l2_prefetch"]`.
- `_l2_prefetch_w2(topk_ids, pf, local_start, local_count)`: the launch, grid (P,), num_warps 4.
- prepare(): after `_align_self_check`, builds the state and launches once on dummy ids `torch.full((1, 8), start)` to compile; the existing final `torch.cuda.synchronize()` follows.
- kernel_fn: `if num_tokens == 1 and expert_map is not None:` launches the prefetch with `data["topk_ids"]` and the bank's prepare-time start/count. Branch is on shape only.
- Deviation from the `change` paragraph (required by the adversarial review and the brief): the launch sits right after `silu_and_mul` and immediately before the w2 `moe_wna16_marlin_gemm`, not after `_align_small`/before w13. This rules out w13's stream evicting the lines, so a flat result reads as "w2 Marlin at M=1 is not B-load-latency bound". No other launch, flag, block_size_m or combine path changed; every M != 1 runs master byte for byte.
- Module docstring gained one paragraph describing the change; two module constants (`L2_PREFETCH_W2_TOKENS = 1`, `L2_PREFETCH_BLOCK = 2048`).
- Uncertainties (no GPU here, nothing was run): Triton must accept a runtime-start/step `range(pid, n, nprog)` inside a runtime `if` inside a static_range (standard scf.for/scf.if; expected fine); `tl.xor_sum` on int32 is assumed available in the judge's Triton; w2_s after the Marlin scale processing is assumed to have >= 4-byte-aligned per-expert slabs (asserted in prepare, so a violation fails loudly in prepare, not silently).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/w2-l2-prefetch-decode1 diff`.

- [x] CUDA-graph capturable: the new kernel_fn code is one Triton launch with a fixed grid from prepare(); no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no allocation (sink is a prepare()-time buffer); the branch is on `num_tokens` (shape) and `expert_map` (prepare state), never on input values. Expert selection happens on the GPU from `topk_ids`.
- [x] Nothing derived from `data` is cached across calls: the sink receives an XOR of weight words only and is never read; nothing from `data` is stored in `state`.
- [x] Weights: no repack or precision change; the prefetch reads int32 *views* of the existing Marlin banks (same bytes) and writes nothing to them.
- [x] Activations BF16 where SGLang keeps BF16: no activation path touched. KV cache / queries: n/a (no attention in this problem).
- [x] Attention sees every key: n/a: no attention in muse_moe_w4a16; no expert or weight skipping either (Marlin calls unchanged).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none (lint clean of these rules).
- [x] All work happens inside the timed call; no output caching or replay: the prefetch runs inside kernel_fn after the judge's L2 flush and pays the full DRAM read each call; the prepare() launch is compile-only on dummy ids and its L2 contents are flushed by the judge before every timed replay.
- [x] First line still `# okc: problem=muse_moe_w4a16`; no triton.autotune added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):

```
warn  host-sync    L279: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L281: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L283: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L302: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L307: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L308: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L392: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L399: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

All warnings are pre-existing prepare()-time syncs (none in kernel_fn's call graph). `python3 -I -m py_compile` OK; __pycache__ removed.

## Risks / what could fail
medium
Attribution: judge the bs=1 paired ratio and the Marlin row at decode_1 (should fall by more than the new _l2_prefetch_w2_kernel row adds). Not caching across calls: the prefetch runs inside the timed call, reads only prepare()-time weights at expert ids taken from this call's topk_ids, and its sink is never read. Risks: (1) Marlin at bs=1 may be bound by its fp32 stripe reduce/lock handshakes rather than B-load latency, in which case L2 hits buy little and the ~3 us prefetch is a pure loss (decode_1 -5%); (2) w13's stream may evict the prefetched lines despite evict_last (H200 L2 is two ~25 MB partitions). If it wins, follow-ups: also prefetch w13 when exactly one local expert is present, and fold the prefetch programs into the _align_small launch to drop the extra graph node. If flat, an ab-tune with the prefetch placed after w13 (immediately before w2) separates eviction from latency.

## Result (after the run)
- job `ac6897f284934dad` (check), status done, passed True, tests 5/5
- speedup 1.2615x, 95% CI [1.2426, 1.2805], score 10660.458923530832 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3075759186466844, "ci": [1.287962279866984, 1.3271895574263846], "job": "3cabc739f1af493e"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/w2-l2-prefetch-decode1.result.json`, `muse_moe_w4a16/hypotheses/done/regression/w2-l2-prefetch-decode1.patch`; branch `hyp/muse_moe_w4a16/w2-l2-prefetch-decode1`
