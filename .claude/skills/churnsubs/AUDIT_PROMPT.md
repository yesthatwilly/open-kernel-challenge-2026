# Pre-flight audit of one implemented churnsubs hypothesis

Repo: /home/user/git/musespark (shell is nushell; use `bash -c` for bash syntax). There is no GPU,
torch or Triton on this node, so you cannot run the kernel. Audit by reading, and by emulating in
plain Python/numpy where that settles a question.

Inputs: the problem `{PROBLEM}`, the hypothesis md `{HYP_MD}` (main tree: spec, adversarial review,
Implementation notes, Rules check), and the worktree `{WORKTREE}`, which has the uncommitted
implementation in `{PROBLEM}/submission.py`. The base is master's `{PROBLEM}/submission.py`. The
implementer was {IMPLEMENTER}. Read `{PROBLEM}/AGENTS.md`, `{PROBLEM}/TASK.md` and
`rules/technical-rules.llms.txt` before judging.

Find what would waste the run or break a rule. In order of what earlier audits actually caught:
1. **Gates that always fail.** A self-check, tolerance, alignment, byte-match or capacity gate that
   can never pass on the judge's shapes and data, so the new path silently never runs.
2. **Fallback cascades.** A failure in the new path that raises, hangs or poisons state instead of
   falling back cleanly to the incumbent, turning a neutral run into a failed one. A build error,
   a self-check miss and an unsupported shape must each land on the incumbent and print a loud
   `PATH CHOSEN` line.
3. **Correctness.** Wrong indexing, masking, strides, dtype, accumulation or epilogue for any shape
   in TASK.md, including tails and non-multiples. Check the actual shapes, not a typical one.
4. **Rules.** Host syncs, allocation, or control flow on input values inside the timed call
   (kernel_fn's call graph); CUDA-graph capturability; nothing derived from `data` cached across
   calls; weight precision (NVFP4 stays NVFP4, BF16 stays BF16, lossless repacks in prepare()
   only); BF16 activations; KV FP8; queries no coarser than FP8 E4M3; every key attended; no
   harness access, processes or network.
5. **Fidelity to the spec.** Does the diff implement the `change` paragraph, touching only the
   named area? Flag scope creep.
6. **Performance traps** that make the result unreadable: a path that is never taken, a perf-killing
   sync, autotune, recompiling per call.

Three checks every audit makes (from the 2026-10-25 audit comparison):
- **Deadlines.** Find the line where the import/prepare clock starts (before or after `import torch`)
  and take master's real import and prepare times from a recent judge log in
  `{PROBLEM}/hypotheses/archive/`. Judge each phase deadline against those numbers, not a guess.
- **Variance vetoes.** Every IQR or variance veto in a gate needs an absolute floor of at least about
  0.5 us on the incumbent's spread. A timer-resolution IQR otherwise vetoes a real win.
- **Evidence.** For new CUDA indexing (fragments, swizzles, tile/page maps), emulate it in numpy and
  save the script in the scratchpad. Any emulation, diff or mutation test you cite must leave its
  script or output behind, and the audit section names the path.

You MAY fix problems in category 1-4 directly in `{WORKTREE}/{PROBLEM}/submission.py`, minimally.
Do not redesign the change. Do not commit, stage or run okc. After editing, run
`python3 -I /home/user/git/musespark/.claude/skills/churnsubs/bin/okc_lint.py {WORKTREE}/{PROBLEM}/submission.py --problem {PROBLEM}`
and `python3 -I -m py_compile` on it, and delete any __pycache__ that leaves in the worktree.
Then append to the hypothesis md a `## Pre-flight audit` section: the verdict, each fix with its
reason, and the residual risk in one line.

Reply with ONLY a JSON object:
{"verdict": "dispatch" | "dispatch-with-fixes" | "hold", "fixes": ["..."], "rules_ok": true|false,
 "residual_risk": "one line", "hold_reason": "only if hold"}
`hold` only for a rules problem you cannot fix minimally, or a build that cannot work as written.
Do not write `{PROBLEM}/hypotheses/CLAIMS.md`: the main thread records one claim per harvested run with
`churn.py claim-add`. When a hold or a fix shows the mechanism itself cannot pay on these shapes (not
just this build), say so in `residual_risk` or `hold_reason` in one sentence that can become that claim.
