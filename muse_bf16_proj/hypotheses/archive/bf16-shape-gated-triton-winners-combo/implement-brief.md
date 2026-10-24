# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_bf16_proj
- hypothesis md (main tree): /home/user/git/musespark/muse_bf16_proj/hypotheses/running/2026-10-24T112147Z-bf16-shape-gated-triton-winners-combo.md
- worktree: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-shape-gated-triton-winners-combo (master)
- Follow the review notes in the md.
- Choose the pooler m<=16 kernel per the review: compare the m<=16 pooler rows (per-shape ratio vs SGLang and the kernel's own time) in muse_bf16_proj/hypotheses/done/neutral/{pooler-splitk2-tma-weight-stream,pooler-triton-splitk2-blockn64-semaphore-control-blockk256,pooler-triton-splitk2-blockn64-semaphore}.result.json and use the fastest; prefer the non-TMA one if within ~1%. Copy each component kernel verbatim from its branch hyp/muse_bf16_proj/<slug>.
