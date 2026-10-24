# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_decode_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_decode_layer/hypotheses/running/2026-10-24T140057Z-post-expert-norm-triton-small-t.md
- worktree: /home/user/git/musespark/.worktrees/muse_decode_layer/post-expert-norm-triton-small-t (master)
- Port _post_norm_combine_kernel verbatim from hyp/muse_moe_w4a16/post-expert-norm-triton-small-m (git show it), adapted to decode_layer's variable names; post_norm_w stays the original bf16 tensor, (1+w) in fp32 in-kernel; tokens with no local slot get an explicit zero row. Follow the review notes.
