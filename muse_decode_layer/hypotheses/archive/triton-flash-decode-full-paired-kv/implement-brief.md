Implement one REWRITE hypothesis (stage 1). Follow /home/user/git/musespark/.claude/skills/churnsubs/IMPLEMENT_PROMPT.md exactly (read it first).
- problem: muse_decode_layer
- hypothesis md: /home/user/git/musespark/muse_decode_layer/hypotheses/running/2026-10-24T075400Z-triton-flash-decode-full-paired-kv.md (read the spec and its Adversarial review)
- worktree: /home/user/git/musespark/.worktrees/muse_decode_layer/triton-flash-decode-full-paired-kv (based on master)
Replace the FA3 call on full-attention decode layers (window 0, q_len 1) only with the spec's Triton split-KV kernel pair (main + combine), reading each token's two KV heads as one 128-byte FP8 row through the page table, block-diagonal query tile. Everything else (sliding layers, verify q_len=8, the rest of the layer) stays byte-identical.
Review requirements (mandatory):
- Corrected incumbent for the readout: FA3 is 374.4 us of the 601.7 us decode_32 full shape, 188.8 us of decode_16 full, and 18.4 + 6.8 (combine) = 25.2 us at decode_1 full (master profile 0730135cdcac43b7). Use these numbers, not 491.6/623, in Implementation notes, and add a stage-1 pass bar there: replaced-kernel time <= 0.8x FA3 (<=300 us decode_32, <=150 us decode_16), with achieved KV bandwidth reported against the 137 us floor.
- Pipelining of indirect loads: structure the main loop so the next tile's page_table indices are loaded before the current tile's math (explicit prefetch), not relying on Triton's pipeliner for a doubly-indirect load.
- k_scale/v_scale are 0-d device tensors: pass pointers, load in-kernel, never .item().
- Fully masked splits (spread=mixed tests: lengths 16..80000): kernel 1 writes m=-inf, l=0 without NaN; kernel 2 uses m_max from finite partials only. No NaN/Inf ever.
- Mask both the page_table load and the K/V gather by t < cache_seqlens[b] and by t//P < page_table.shape[1]; support page_size 1 and 64 (constexpr) or branch to FA3 for 64 and say so.
- Every key t < cache_seqlens[b] is visited; grid and split length from shapes only; scratch sized from B and S only. No autotune, no host syncs; warm every variant in prepare().
