# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_moe_w4a16
- hypothesis md (main tree): /home/user/git/musespark/muse_moe_w4a16/hypotheses/running/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md
- worktree: /home/user/git/musespark/.worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-w13-silu-w2 (master 1.312x)
- Rewrite stage 1, a single persistent CUDA kernel via load_inline (compile at import or in prepare, never during capture). The proven in-repo pattern is muse_attention_decode master's _build_splitkv / self-check / fallback; reuse its structure.
- REQUIRED from the review: (1) the phase-2 wait (flag == 8*(epoch+1)) hangs once R grows between calls because flags are bumped only for r < R: reset flags, tickets and counters with an all-CTA exit ticket each call, and cap EVERY spin loop with a bounded iteration count that sets an error flag instead of hanging; (2) phase 1 keeps only ~27 KB/SM in flight: raise BK to 512 or add ring stages; (3) assert co-residency (1 CTA/SM, grid == SM count) via the occupancy API in prepare; (4) the self-check (R=0,1,2,3,8, 50 repeats, a graph capture) must fall back to master on any failure or error flag. Runs as a CHECK after the quota reset (risk high). Self-review synchronization and indexing twice.
