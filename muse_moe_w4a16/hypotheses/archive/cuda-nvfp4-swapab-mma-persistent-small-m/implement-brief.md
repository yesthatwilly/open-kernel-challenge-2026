# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_moe_w4a16
- hypothesis md (main tree): /home/user/git/musespark/muse_moe_w4a16/hypotheses/running/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md
- worktree: /home/user/git/musespark/.worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m (master 1.312x)
- Rewrite stage 1 in CUDA via load_inline; reuse muse_attention_decode master's _build_splitkv / self-check / fallback structure. REQUIRED from the review: (1) XOR-swizzle (or pad) the activation slab in smem, rows are 512 B apart and conflict 8-way otherwise; (2) interleave gate and up rows within each warp so the w13 silu epilogue needs no cross-warp exchange; (3) prepare() warm-compiles _align_small and runs its self-check at the new block sizes 16 and 32; (4) index by the route id from sorted_ids, not the sorted position; (5) treat a register spill as non-fatal (print it); (6) separate ticket buffers for the w13 and w2 launches; (7) print PATH CHOSEN (cuda|marlin) and per-shape self-check results. Runs as a CHECK after the quota reset (risk high). Self-review the fragment indexing twice.
