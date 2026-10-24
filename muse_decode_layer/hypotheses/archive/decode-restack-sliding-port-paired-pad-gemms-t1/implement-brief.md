# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_decode_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_decode_layer/hypotheses/running/2026-10-24T174421Z-decode-restack-sliding-port-paired-pad-gemms-t1.md
- worktree: /home/user/git/musespark/.worktrees/muse_decode_layer/decode-restack-sliding-port-paired-pad-gemms-t1 (master blob cc466fce)
- This is a RESTACK of three measured patches on the same base blob; no new ideas. Branches: hyp/muse_decode_layer/sliding-fused-triton-decode1-port, hyp/muse_decode_layer/paired-heads-pad-unpad-triton, hyp/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t. Do 3-way `git merge-file` merges onto master's submission.py and resolve the conflicts by hand.
- Change one thing only: gate the Triton GEMMs to `tokens == 1`, a pure shape predicate. T>1 must take master's GEMM path byte for byte. You may trim the GEMM warmup to m=1.
- REQUIRED from the review (merge risk): the port re-indents the whole attention block under `else:`. The paired edits (`_pair_pad_q_kernel` replacing the torch.zeros + slice copies, and the `if paired: ... else: o_norm_gate` epilogue) must land inside that else branch. `o` must stay defined on all three routes: sliding batch-1 fused, paired, and plain. Self-review the merged kernel_fn line by line against each branch's own kernel_fn. Keep each piece's own self-checks.
- Update the md's expected list to add "full decode_1 faster (T=1 GEMMs)" and "full verify_16 faster (paired)".
