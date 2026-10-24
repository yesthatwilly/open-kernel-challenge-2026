# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_prefill_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_prefill_layer/hypotheses/running/2026-10-24T092506Z-experts-triton-w4a16-swapab-reglhs.md
- worktree: /home/user/git/musespark/.worktrees/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs (based on the stage-1 rewrite commit 6a70bfa; build on its _moe_nvfp4_gemm_kernel)
- This is a rewrite stage: a bigger change than usual is expected, confined to the expert GEMM kernel and its prepare() layout.
- Reviewer build notes to follow: load the e4m3 scales once per K step as a compact [128, 8] tile and broadcast, not per K-half; build the nibble-reordered weights as NEW state tensors (do not overwrite weights["w13"] / ["w2"]); keep the untimed prepare() assert that the kernel dequant equals nvfp4.dequantize for one expert slice, adapted to the new layout. NVFP4 stays NVFP4 (a lossless code permutation only), dequant in registers only.
