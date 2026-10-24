# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_bf16_proj
- hypothesis md (main tree): /home/user/git/musespark/muse_bf16_proj/hypotheses/running/2026-10-24T174421Z-bf16-restack-m16-swapab-router-swapab-m32.md
- worktree: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32 (master 1.196x, blob a8bb42a)
- This is a RESTACK of two measured patches; no new ideas. Branches hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend and hyp/muse_bf16_proj/router-swapab-wgmma-all-m both build on blob a8bb42a. Do a 3-way `git merge-file` of their submission.py over master's (the reviewer confirmed it is clean and parses), then gate the router swap-AB to m<=32.
- REQUIRED from the review (gate trap): kernel_fn must send router m<=32 to the swap-AB `_router_forward` and m>32 to MASTER's original hilo forward (with `_router_hilo_config` and ROUTER_NUM_STAGES), never to the F.linear fp32 fallback. Warm hilo at m=64/128/256 in prepare(). Add a prepare-time assert that every ROUTER_WARM_MS bucket resolves to a kernel path, not the fallback. `_router_shape_ok` must take SPLIT_K from the path actually used (32 for swap-AB, 16/8 for hilo). The gate is inclusive (m<=32). Drop `_router_hilo_report` and one of the duplicate PTX-count helpers if trivial.
- Verify the result by diff: for m>32 the router code path must be byte-for-byte master's.
