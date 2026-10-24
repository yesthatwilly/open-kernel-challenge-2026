Implement one kernel hypothesis. Follow /home/user/git/musespark/.claude/skills/churnsubs/IMPLEMENT_PROMPT.md exactly (read it first).
- problem: muse_prefill_layer
- hypothesis md: /home/user/git/musespark/muse_prefill_layer/hypotheses/running/2026-10-24T075409Z-experts-marlin-block-m-local-rows.md
- worktree: /home/user/git/musespark/.worktrees/muse_prefill_layer/experts-marlin-block-m-local-rows (based on master = SGLang parity port)
One-expression change: in the block_size_m selection loop, replace the T*8/LOCAL_EXPERTS rows-per-expert estimate with T*8/N_EXPERTS (the real local rows under EP8), so T=753 picks 32 and T>=1506 keeps 64. The same block_size_m must feed moe_align_block_size and both Marlin calls. Branch on token count only. Record the before/after picks per scored T in Implementation notes.
