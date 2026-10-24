Implement one kernel hypothesis. Follow /home/user/git/musespark/.claude/skills/churnsubs/IMPLEMENT_PROMPT.md exactly (read it first).
- problem: muse_prefill_layer
- hypothesis md: /home/user/git/musespark/muse_prefill_layer/hypotheses/running/2026-10-24T075422Z-experts-marlin-block-m-48-turn1.md
- worktree: /home/user/git/musespark/.worktrees/muse_prefill_layer/experts-marlin-block-m-48-turn1 (based on master)
One-knob change: block_size_m = 48 for 512 < num_tokens <= 1024 (T=753 turn_1, and the T=600/753 tests); every other T keeps master's pick. The same block size feeds moe_align_block_size and both Marlin calls; branch on token count only. This is the A/B sibling of experts-marlin-block-m-local-rows (block 32 at T=753), so do not touch anything else. Record before/after picks per scored and test T in Implementation notes.
