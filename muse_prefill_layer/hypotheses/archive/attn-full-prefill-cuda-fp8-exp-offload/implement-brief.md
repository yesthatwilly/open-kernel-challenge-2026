# Implementing one churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). No GPU, torch or Triton on this node: you cannot run anything.

1. Read <problem>/AGENTS.md, <problem>/TASK.md, the hypothesis md (main-tree path given to you), including its `## Adversarial review` section, and the worktree's <problem>/submission.py.
2. Implement the md's `change` paragraph in the worktree's <problem>/submission.py ONLY, modestly, touching only the named area. Follow the adversarial review's rules notes and open-question guidance where they are implementation details (not new ideas). Keep the first-line marker `# okc: problem=<problem>`. No triton.autotune, no host syncs in kernel_fn, nothing derived from inputs cached in state.
3. Append to the hypothesis md (main-tree path) `## Implementation notes` (what changed, where, any deviation from the spec and why) and `## Rules check`: every checklist item from /home/user/git/musespark/ATTEMPT_TEMPLATE.md, each ticked `[x]` only after checking the actual `git -C <worktree> diff` (at least 7 boxes; an item that does not apply is ticked "n/a: reason").
4. Run `python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py <worktree>/<problem>/submission.py --problem <problem>`, paste its output under Rules check, fix any errors. `python3 -I -m py_compile` the file and delete any __pycache__ it leaves in the worktree.
5. Do NOT commit, stage, or run okc. Report: summary of the change, diff stat, lint result, uncertainties.

## This pick

- problem: muse_prefill_layer
- hypothesis md (main tree): /home/user/git/musespark/muse_prefill_layer/hypotheses/running/2026-10-24T172146Z-attn-full-prefill-cuda-fp8-exp-offload.md
- worktree: /home/user/git/musespark/.worktrees/muse_prefill_layer/attn-full-prefill-cuda-fp8-exp-offload (based on master)
- This is a kind: rewrite, stage 1, a hand-written GPU kernel built blind (nothing can be compiled here). It will spend one scarce check, so correctness of syntax, PTX forms and indexing matters more than polish: re-read every asm block against the PTX ISA, keep a self-check plus a fallback to master, and print loud log lines (build result, PATH CHOSEN).
- Follow the review notes in the md and DECISIONS.md. REQUIRED from the review: Software exp in fp32 FFMA only; drop the HFMA2 option. Clamp the input to [-127,0] before range reduction, round with the FADD magic constant (not FRND/F2I), and unit-test -inf, -200 and tiny positive inputs. Add a split-KV from config for turn_1 full (e.g. 11 splits) with a small combine, keeping every key. Include the judge's full-layer test (batch 3, ctx 5000, L 200, outlier Q/K) in the self-check. Make the offload fraction a constexpr, time fraction 0 against the tuned fraction with CUDA events in prepare() (print only), and log [cuda full attn] ON/OFF.
- The working CUDA load_inline precedent is /home/user/git/musespark/muse_attention_decode/submission.py on master (compiled in 41 s on the judge).
