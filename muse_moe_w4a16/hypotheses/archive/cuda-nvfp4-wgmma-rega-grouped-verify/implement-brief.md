# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_moe_w4a16
- hypothesis md (main tree): /home/user/git/musespark/muse_moe_w4a16/hypotheses/running/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md
- worktree: /home/user/git/musespark/.worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-verify (based on master)
- This is a kind: rewrite, stage 1, a hand-written GPU kernel built blind (nothing can be compiled here). It will spend one scarce check, so correctness of syntax, PTX forms and indexing matters more than polish: re-read every asm block against the PTX ISA, keep a self-check plus a fallback to master, and print loud log lines (build result, PATH CHOSEN).
- Follow the review notes in the md and DECISIONS.md. REQUIRED from the review: BLOCKING: the smem table as specified exceeds 227 KB. Use N=128 at 4-5 stages of BK=128 (or BK=64 at 8-10) and N=64 at 4 stages of BK=256, and assert per-bucket smem <= 227 KB in Python before compiling. Round gate and up to bf16 before silu. Use evict_normal when an expert has more than one chunk. Form chunks with a warp-ballot/prefix scan. Stage the w2 scatter through smem (stmatrix .trans) for coalesced row writes. cuTensorMapEncodeTiled: link -lcuda in load_inline, or fall back to cp.async.bulk 1-D if that cannot be made to work. Log items per shape and PATH CHOSEN. You may reuse device functions from the worktree /home/user/git/musespark/.worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m (same weight bank and dequant), but this file builds on master.
- The working CUDA load_inline precedent is /home/user/git/musespark/muse_attention_decode/submission.py on master (compiled in 41 s on the judge).
