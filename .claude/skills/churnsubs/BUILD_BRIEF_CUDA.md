# Build brief: CUDA rewrite stages, 2026-10-25

You are implementing ONE churnsubs hypothesis. Follow /home/user/git/musespark/.claude/skills/churnsubs/IMPLEMENT_PROMPT.md exactly. Your message gives the problem, the hypothesis md (main tree) and the worktree; edit only `<worktree>/<problem>/submission.py`.

Read, in order: the hypothesis md including its `## Adversarial review` (apply every required fix it names; where the review corrects the spec, the review wins, and say so in Implementation notes), `<problem>/hypotheses/REWRITE_READ_2026-10-25.md` (the per-shape read of the previous stage), `/home/user/git/musespark/hypotheses/LESSONS.md` (all 2026-10-25 entries), `<problem>/AGENTS.md`, `<problem>/TASK.md`.

Hard requirements, every build:
- NO child process, subprocess, multiprocessing, os.fork or spawn anywhere: the rules statically reject submissions that start processes.
- If any load_inline CUDA uses wgmma, setmaxnreg or other sm_90a-only PTX: set `os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"` around load_inline and restore the previous value after; wrap that PTX in `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)` with an `#else` that sets an error flag the self-check reads; print the arch flags used.
- Self-check in prepare(), in-process: announce each launch with a flushed print BEFORE it, `torch.cuda.synchronize()` and a flushed print AFTER it, so a fault names its kernel. Use plain `cp.async.cg` instead of createpolicy/L2::cache_hint variants unless the review says otherwise. Add `-lineinfo`.
- Bound every mbarrier/spin wait (fail fast with an error flag, never hang to the 15 min timeout).
- Any build error, self-check miss or unsupported shape lands on master's (or the parent's) path and never raises; loud `PATH CHOSEN` lines per shape class.
- No host syncs, allocation on input values or data-dependent control flow in kernel_fn; CUDA-graph capturable; nothing cached from inputs; precision rules as in TASK.md.

When done: write `## Implementation notes` (what changed, deviations, which review fixes you applied) and `## Rules check` in the hypothesis md, run okc_lint and py_compile, delete __pycache__, do NOT commit or run okc. Reply under 120 words: change summary, diff stat, lint result, the biggest compile/run risk.
