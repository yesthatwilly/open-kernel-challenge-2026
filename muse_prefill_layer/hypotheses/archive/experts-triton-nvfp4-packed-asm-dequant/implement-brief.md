# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_prefill_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_prefill_layer/hypotheses/running/2026-10-24T140058Z-experts-triton-nvfp4-packed-asm-dequant.md
- worktree: /home/user/git/musespark/.worktrees/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant (master 1.060x: Marlin at T<=4096, Triton NVFP4 above)
- This is a SCORED submit: correctness first; no judge test exceeds 4096 tokens, so the prepare() checks are the only gate for the Triton path.
- REQUIRED from the review: make the packed flag a constexpr kernel argument; in prepare() try the packed variant (compile + _check_expert_gemms) in try/except and on failure log it and fall back to the current dequant; print ttgir convert_layout/local_alloc counts for both packed and unpacked variants.
