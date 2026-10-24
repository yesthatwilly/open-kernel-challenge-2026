Implement one REWRITE hypothesis (stage 1). Follow /home/user/git/musespark/.claude/skills/churnsubs/IMPLEMENT_PROMPT.md exactly (read it first).
- problem: muse_moe_w4a16
- hypothesis md: /home/user/git/musespark/muse_moe_w4a16/hypotheses/running/2026-10-24T081339Z-small-m-grouped-gemv-splitk-rewrite.md (read the spec, Milestones and Adversarial review)
- worktree: /home/user/git/musespark/.worktrees/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite (based on master = parity port)
Build the spec's persistent split-K grouped W4A16 GEMV kernels for w13 and w2 at num_tokens <= 16 (Marlin unchanged above). Also read the regressed muse_moe_w4a16/hypotheses/done/regression/triton-w13-grouped-gemv-small-m.* and its archive profile for what failed.
Review requirements (mandatory):
- Pipelining across items: Triton only pipelines the innermost loop. Flatten (work item x k-step) into ONE linear tl.range loop so loads prefetch across item boundaries, OR give w2 SPLIT_K so each item's K loop is >= 4 steps; record which.
- Ticket ordering: partial stores -> tl.debug_barrier() -> one gpu-scope acq_rel scalar atomic -> tl.debug_barrier() -> winner reloads partials with cache_modifier='.cg'. Separate w13 and w2 counter arrays; self-resetting; prepare()'s self-test must leave them at zero.
- Remote routes: replicate Marlin's is_ep checks (skip blocks with block_experts == -1, mask sorted_ids >= M*8) in the work-list decode.
- Row semantics: w13 gathers x[slot // 8] (top_k=8), w2 gathers cache2[slot] (top_k=1); cache3 aliases cache1 as in master; confirm gate rows [0, I) / up rows [I, 2I) match what silu_and_mul expects.
- Live-block count read on device from n_padded (no .item()); fixed constexpr configs; no autotune; keep the bf16 dot (no fp16 activations).
- prepare() self-test from weights-only synthetic data with a fixed torch.Generator vs nvfp4.dequantize: print its rel-L2 LOUDLY in both outcomes and the path chosen; a silent fallback would waste the run.
- NVFP4 bytes stay as stored (a lossless byte permutation copy is allowed); never a BF16 weight copy.
Record in Implementation notes: the grid/items per shape, the pipelining choice, the dequant method, and the stage-1 readout (new kernels' own us per shape vs Marlin 41.1/107.6/150.9 us combined).
