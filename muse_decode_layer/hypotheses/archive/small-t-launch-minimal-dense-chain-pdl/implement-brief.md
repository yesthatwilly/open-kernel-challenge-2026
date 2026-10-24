# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_decode_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_decode_layer/hypotheses/running/2026-10-24T172138Z-small-t-launch-minimal-dense-chain-pdl.md
- worktree: /home/user/git/musespark/.worktrees/muse_decode_layer/small-t-launch-minimal-dense-chain-pdl (based on master)
- This is a kind: rewrite, stage 1, a hand-written GPU kernel built blind (nothing can be compiled here). It will spend one scarce check, so correctness of syntax, PTX forms and indexing matters more than polish: re-read every asm block against the PTX ISA, keep a self-check plus a fallback to master, and print loud log lines (build result, PATH CHOSEN).
- Follow the review notes in the md and DECISIONS.md. REQUIRED from the review: Read nothing derived from inputs or the previous kernel before griddepcontrol.wait; prefetch prepare-built weights only. Fix kernel P's grid (288 CTAs x 128 KB = 3 waves): use a persistent/stream-K grid of SM-count CTAs, or a 64-96 KB ring so every CTA is resident. The self-check must confirm a programmatic edge formed under graph capture (or measure overlap), retry with pdl=0 on failure, and time P with PDL on and off via CUDA events (print only). Match master's contracts exactly: local_ids -1 for remote slots, contiguous fp32 topk_weights, the Marlin block-size cap; at T=1 call attn_boundary with router_dtype=None. Add self-check cases with no local slot and a near-tie routing.
- The working CUDA load_inline precedent is /home/user/git/musespark/muse_attention_decode/submission.py on master (compiled in 41 s on the judge).
